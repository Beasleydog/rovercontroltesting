#!/usr/bin/env python3
"""Simple TSS rover socket helpers.

This talks to TSS over UDP. DUST should already be connected to TSS.
"""

from __future__ import annotations

import csv
import json
import math
import socket
import struct
import threading
import time
from datetime import datetime
from pathlib import Path

import socketio


SERVER_HOST = "172.24.119.191"
SERVER_PORT = 14141
SOCKET_TIMEOUT = 1.0
REMOTE_SERVER_ENABLED = False
REMOTE_SERVER_URL = "http://127.0.0.1:5001"

LIDAR_LOG_PATH = "lidar_log.csv"
LIDAR_LOG_HZ = 0.0
LIDAR_LOG_ALL = False
LIDAR_MAX_VALID_CM = 1000.0

GET_ROVER_JSON = 0
GET_EVA_JSON = 1
GET_LTV_JSON = 2
GET_LTV_ERRORS_JSON = 3

CMD_CABIN_HEATING = 1103
CMD_CABIN_COOLING = 1104
CMD_LIGHTS = 1106
CMD_BRAKES = 1107
CMD_THROTTLE = 1109
CMD_STEERING = 1110
STEERING_COMMAND_SIGN = -1.0
CMD_PING = 2050
CMD_DEBUG_PING = 2051

REMOTE_COMMAND_EVENT_BY_ID = {
    CMD_CABIN_HEATING: "rover-heating",
    CMD_CABIN_COOLING: "rover-cooling",
    CMD_LIGHTS: "rover-headlights",
    CMD_BRAKES: "rover-brakes",
    CMD_THROTTLE: "rover-throttle",
    CMD_STEERING: "rover-steering",
    CMD_PING: "rover-ping",
    CMD_DEBUG_PING: "rover-debug-ping",
}


def configure_remote_server(enabled: bool, url: str | None = None) -> None:
    global REMOTE_SERVER_ENABLED, REMOTE_SERVER_URL
    REMOTE_SERVER_ENABLED = bool(enabled)
    if url:
        REMOTE_SERVER_URL = str(url)


class RemoteRoverClient:
    def __init__(self, url: str) -> None:
        self.url = str(url)
        self.sio = socketio.Client(reconnection=True)
        self._state_lock = threading.Lock()
        self._connected_event = threading.Event()
        self._rover_event = threading.Event()
        self._ltv_event = threading.Event()
        self._eva_event = threading.Event()
        self._ltv_errors_event = threading.Event()
        self._latest_rover: dict | None = None
        self._latest_ltv: dict | None = None
        self._latest_eva: dict | None = None
        self._latest_ltv_errors: dict | None = None

        @self.sio.event
        def connect() -> None:
            self._connected_event.set()

        @self.sio.event
        def disconnect() -> None:
            self._connected_event.clear()

        @self.sio.on("rover-telemetry")
        def on_rover_telemetry(data) -> None:
            payload = dict(data) if isinstance(data, dict) else {}
            with self._state_lock:
                self._latest_rover = payload
            self._rover_event.set()

        @self.sio.on("ltv-telemetry")
        def on_ltv_telemetry(data) -> None:
            payload = dict(data) if isinstance(data, dict) else {}
            with self._state_lock:
                self._latest_ltv = payload
            self._ltv_event.set()

        @self.sio.on("eva-telemetry")
        def on_eva_telemetry(data) -> None:
            payload = dict(data) if isinstance(data, dict) else {}
            with self._state_lock:
                self._latest_eva = payload
            self._eva_event.set()

        @self.sio.on("ltv-errors-telemetry")
        def on_ltv_errors_telemetry(data) -> None:
            payload = dict(data) if isinstance(data, dict) else {}
            with self._state_lock:
                self._latest_ltv_errors = payload
            self._ltv_errors_event.set()

    def connect(self, timeout_seconds: float = 5.0) -> None:
        self.sio.connect(self.url, wait=True, wait_timeout=timeout_seconds)
        if not self._connected_event.wait(timeout_seconds):
            raise RuntimeError(f"Timed out connecting to remote rover server at {self.url}")

    def close(self) -> None:
        if self.sio.connected:
            self.sio.disconnect()

    def emit(self, event_name: str, payload=None) -> bool:
        self.sio.emit(event_name, payload)
        return True

    def _payload_event_for_get(self, command: int) -> tuple[str, threading.Event, str]:
        mapping = {
            GET_ROVER_JSON: ("_latest_rover", self._rover_event, "rover"),
            GET_LTV_JSON: ("_latest_ltv", self._ltv_event, "ltv"),
            GET_EVA_JSON: ("_latest_eva", self._eva_event, "eva"),
            GET_LTV_ERRORS_JSON: ("_latest_ltv_errors", self._ltv_errors_event, "ltv-errors"),
        }
        if command not in mapping:
            raise RuntimeError(f"Remote server does not support GET command {command}")
        return mapping[command]

    def get_json_for_command(self, command: int, timeout_seconds: float = SOCKET_TIMEOUT) -> dict:
        attr_name, ready_event, label = self._payload_event_for_get(command)
        if not ready_event.wait(timeout_seconds):
            raise RuntimeError(f"Timed out waiting for remote {label} telemetry from {self.url}")
        with self._state_lock:
            payload = getattr(self, attr_name)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Remote {label} telemetry is unavailable")
            return dict(payload)


def open_rover_socket() -> socket.socket:
    if REMOTE_SERVER_ENABLED:
        client = RemoteRoverClient(REMOTE_SERVER_URL)
        client.connect(timeout_seconds=max(5.0, SOCKET_TIMEOUT))
        return client
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(SOCKET_TIMEOUT)
    return sock


def close_rover_socket(sock: socket.socket) -> None:
    if isinstance(sock, RemoteRoverClient):
        sock.close()
        return
    sock.close()


def server_address() -> tuple[str, int]:
    return (SERVER_HOST, SERVER_PORT)


def unix_timestamp() -> int:
    return int(time.time())


def clamp_throttle(value: float) -> float:
    return max(-100.0, min(100.0, float(value)))


def clamp_steering(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def send_packet(sock: socket.socket, packet: bytes, response_size: int = 8192) -> bytes:
    sock.sendto(packet, server_address())
    response, _ = sock.recvfrom(response_size)
    return response


def send_float_command(sock: socket.socket, command: int, value: float) -> bool:
    if isinstance(sock, RemoteRoverClient):
        event_name = REMOTE_COMMAND_EVENT_BY_ID.get(int(command))
        if event_name is None:
            raise RuntimeError(f"Remote server does not support command {command}")
        payload = float(value)
        if int(command) == CMD_BRAKES:
            return sock.emit(event_name, bool(payload >= 0.5))
        if int(command) in (CMD_PING, CMD_DEBUG_PING) and abs(payload - 1.0) <= 1e-6:
            return sock.emit(event_name, None)
        return sock.emit(event_name, payload)
    packet = struct.pack(">IIf", unix_timestamp(), command, float(value))
    response = send_packet(sock, packet, response_size=64)
    if len(response) < 4:
        raise RuntimeError(f"Incomplete ACK for command {command}: {response!r}")
    return any(response[:4])


def send_get_command(sock: socket.socket, command: int) -> bytes:
    if isinstance(sock, RemoteRoverClient):
        payload = sock.get_json_for_command(int(command), timeout_seconds=SOCKET_TIMEOUT)
        return json.dumps(payload).encode("utf-8")
    packet = struct.pack(">II", unix_timestamp(), command)
    return send_packet(sock, packet)


def extract_json_bytes(response: bytes) -> bytes:
    payload = response[8:] if len(response) > 8 else response
    start = payload.find(b"{")
    if start == -1:
        raise RuntimeError(f"Could not locate JSON payload in response: {response[:32]!r}")
    return payload[start:].split(b"\x00", 1)[0]


def fetch_rover_json(sock: socket.socket) -> dict:
    response = send_get_command(sock, GET_ROVER_JSON)
    return json.loads(extract_json_bytes(response).decode("utf-8"))


def fetch_ltv_json(sock: socket.socket) -> dict:
    response = send_get_command(sock, GET_LTV_JSON)
    return json.loads(extract_json_bytes(response).decode("utf-8"))


def sanitize_lidar_value(raw_value: object) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return -1.0

    if math.isnan(value) or math.isinf(value):
        return -1.0
    if value == -1.0:
        return -1.0
    if value < 0.0:
        return -1.0
    if value > LIDAR_MAX_VALID_CM:
        return -1.0
    return value


def sanitize_lidar_scan(raw_lidar: object) -> list[float] | None:
    if not isinstance(raw_lidar, list):
        return None
    return [sanitize_lidar_value(value) for value in raw_lidar]


def fetch_rover_telemetry(sock: socket.socket) -> dict:
    rover = fetch_rover_json(sock)
    telemetry = rover.get("pr_telemetry")
    if not isinstance(telemetry, dict):
        raise RuntimeError("ROVER.json did not contain pr_telemetry")
    sanitized_lidar = sanitize_lidar_scan(telemetry.get("lidar"))
    if sanitized_lidar is not None:
        telemetry["lidar"] = sanitized_lidar
    return telemetry


def is_dust_connected(sock: socket.socket) -> bool:
    return bool(fetch_rover_telemetry(sock).get("dust_connected", False))


def wait_for_dust(sock: socket.socket, timeout_seconds: float = 15.0, poll_seconds: float = 0.5) -> bool:
    end_time = time.monotonic() + timeout_seconds
    while time.monotonic() < end_time:
        if is_dust_connected(sock):
            return True
        time.sleep(max(0.0, poll_seconds))
    return is_dust_connected(sock)


def set_throttle(sock: socket.socket, value: float) -> bool:
    return send_float_command(sock, CMD_THROTTLE, clamp_throttle(value))


def set_steering(sock: socket.socket, value: float) -> bool:
    # DUST steering command sign is opposite the math/control convention used locally.
    return send_float_command(sock, CMD_STEERING, clamp_steering(value) * STEERING_COMMAND_SIGN)


def set_brakes(sock: socket.socket, enabled: bool) -> bool:
    return send_float_command(sock, CMD_BRAKES, 1.0 if enabled else 0.0)


def set_lights(sock: socket.socket, enabled: bool) -> bool:
    return send_float_command(sock, CMD_LIGHTS, 1.0 if enabled else 0.0)


def set_cabin_heating(sock: socket.socket, enabled: bool) -> bool:
    return send_float_command(sock, CMD_CABIN_HEATING, 1.0 if enabled else 0.0)


def set_cabin_cooling(sock: socket.socket, enabled: bool) -> bool:
    return send_float_command(sock, CMD_CABIN_COOLING, 1.0 if enabled else 0.0)


def read_lidar(sock: socket.socket) -> list[float]:
    lidar = fetch_rover_telemetry(sock).get("lidar")
    if not isinstance(lidar, list):
        raise RuntimeError("ROVER.json did not contain a lidar array")
    return [float(value) for value in lidar]


def log_lidar_fast(sock: socket.socket, output_path: str, poll_hz: float = 0.0, log_all: bool = False) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    delay = 0.0 if poll_hz <= 0 else 1.0 / poll_hz
    last_scan: list[float] | None = None

    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if csv_file.tell() == 0:
            writer.writerow(["unix_time", "iso_time", "lidar_json"])

        while True:
            scan = read_lidar(sock)
            if log_all or scan != last_scan:
                now = time.time()
                iso_time = datetime.now().isoformat(timespec="milliseconds")
                writer.writerow([f"{now:.3f}", iso_time, json.dumps(scan, separators=(",", ":"))])
                csv_file.flush()
                print(f"{iso_time} lidar={scan}")
                last_scan = scan

            if delay > 0:
                time.sleep(delay)


def run_example() -> None:
    sock = open_rover_socket()
    try:
        print(f"Logging lidar to {LIDAR_LOG_PATH}")
        log_lidar_fast(sock, LIDAR_LOG_PATH, poll_hz=LIDAR_LOG_HZ, log_all=LIDAR_LOG_ALL)
    finally:
        close_rover_socket(sock)


if __name__ == "__main__":
    run_example()
