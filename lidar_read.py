from __future__ import annotations

import csv
import ctypes
import json
import time
from pathlib import Path

from rover_control import (
    close_rover_socket,
    fetch_ltv_json,
    fetch_rover_json,
    open_rover_socket,
    send_float_command,
    set_brakes,
    set_lights,
    set_steering,
    set_throttle,
    wait_for_dust,
)


CLEAN_LOG_ROOT = Path("cleanlog")
POLL_HZ = 20.0
MAX_THROTTLE = 100.0
MAX_STEERING = 1.0
ENABLE_PING_LOCATION_LOGGING = True
PING_LOG_INTERVAL_SEC = 1.0
CMD_LTV_PING = 2050
CMD_LTV_PING_UNLIMITED = 2051
USE_UNLIMITED_PING = False
PING_SETTLE_SEC = 1.0
# (sensor_local_x_cm, sensor_local_y_cm, sensor_yaw_deg, sensor_pitch_deg)
LIDAR_SENSOR_LAYOUT = [
    (250.0, 245.0, 30.0, 0.0),
    (325.0, 75.0, 20.0, -20.0),
    (325.0, 0.0, 0.0, 0.0),
    (325.0, -75.0, -20.0, -20.0),
    (250.0, -245.0, -30.0, 0.0),
    (325.0, 75.0, 0.0, -25.0),
    (325.0, -75.0, 0.0, -25.0),
    (40.0, 235.0, 90.0, -20.0),
    (40.0, -235.0, -90.0, -20.0),
    (-215.0, 270.0, 140.0, 0.0),
    (-320.0, 80.0, 180.0, 0.0),
    (-320.0, -50.0, 180.0, 0.0),
    (-215.0, -215.0, -140.0, 0.0),
    (325.0, 75.0, 20.0, -10.0),
    (325.0, -75.0, -20.0, -10.0),
    (250.0, 245.0, 15.0, 0.0),
    (250.0, -245.0, -15.0, 0.0),
]
LIDAR_SENSOR_LABELS = [
    "front_left_wheel_hub_yaw_p30",
    "front_left_frame_yaw_p20_pitch_n20",
    "front_center_frame_forward",
    "front_right_frame_yaw_n20_pitch_n20",
    "front_right_wheel_hub_yaw_n30",
    "front_left_frame_pitch_n25",
    "front_right_frame_pitch_n25",
    "left_mid_frame_left_pitch_n20",
    "right_mid_frame_right_pitch_n20",
    "rear_left_wheel_hub_back_yaw_p40",
    "rear_left_frame_backward",
    "rear_right_frame_backward",
    "rear_right_wheel_hub_back_yaw_n40",
    "front_left_frame_yaw_p20_pitch_n10",
    "front_right_frame_yaw_n20_pitch_n10",
    "front_left_wheel_hub_yaw_p15",
    "front_right_wheel_hub_yaw_n15",
]
LIDAR_SENSOR_COUNT = len(LIDAR_SENSOR_LAYOUT)
RAW_TSS_TELEMETRY_FIELDS = [
    "rover_pos_x",
    "rover_pos_y",
    "rover_pos_z",
    "heading",
    "pitch",
    "roll",
]
LIDAR_GROUPS = {
    "front": [0, 1, 2, 3, 4, 5, 6, 13, 14, 15, 16],
    "left": [7],
    "right": [8],
    "back": [9, 10, 11, 12],
}

VK_W = 0x57
VK_A = 0x41
VK_S = 0x53
VK_D = 0x44
VK_Q = 0x51
VK_ESC = 0x1B
STD_OUTPUT_HANDLE = -11


class COORD(ctypes.Structure):
    _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]


class SMALL_RECT(ctypes.Structure):
    _fields_ = [
        ("Left", ctypes.c_short),
        ("Top", ctypes.c_short),
        ("Right", ctypes.c_short),
        ("Bottom", ctypes.c_short),
    ]


class CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure):
    _fields_ = [
        ("dwSize", COORD),
        ("dwCursorPosition", COORD),
        ("wAttributes", ctypes.c_ushort),
        ("srWindow", SMALL_RECT),
        ("dwMaximumWindowSize", COORD),
    ]


def key_down(vk_code: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)


def rewrite_console_block(lines: list[str], line_count: int) -> None:
    handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
    info = CONSOLE_SCREEN_BUFFER_INFO()
    if not ctypes.windll.kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)):
        for line in lines:
            print(line)
        return

    width = max(1, int(info.dwSize.X))
    current_y = int(info.dwCursorPosition.Y)
    target_y = max(0, current_y - max(0, line_count))
    written = ctypes.c_ulong(0)

    for offset, line in enumerate(lines):
        row_y = target_y + offset
        row_start = COORD(0, row_y)
        ctypes.windll.kernel32.FillConsoleOutputCharacterW(
            handle,
            ctypes.c_wchar(" "),
            width,
            row_start,
            ctypes.byref(written),
        )
        ctypes.windll.kernel32.FillConsoleOutputAttribute(
            handle,
            info.wAttributes,
            width,
            row_start,
            ctypes.byref(written),
        )
        ctypes.windll.kernel32.SetConsoleCursorPosition(handle, row_start)
        text = line[: max(0, width - 1)]
        ctypes.windll.kernel32.WriteConsoleW(
            handle,
            ctypes.c_wchar_p(text),
            len(text),
            ctypes.byref(written),
            None,
        )

    ctypes.windll.kernel32.SetConsoleCursorPosition(handle, COORD(0, target_y + len(lines)))


def desired_drive_state() -> tuple[float, float, bool]:
    forward = key_down(VK_W)
    reverse = key_down(VK_S)
    left = key_down(VK_A)
    right = key_down(VK_D)
    exit_requested = key_down(VK_Q) or key_down(VK_ESC)

    throttle = 0.0
    if forward and not reverse:
        throttle = MAX_THROTTLE
    elif reverse and not forward:
        throttle = -MAX_THROTTLE

    steering = 0.0
    if left and not right:
        steering = -MAX_STEERING
    elif right and not left:
        steering = MAX_STEERING

    return (throttle, steering, exit_requested)


def lidar_values_cm(raw_telemetry: dict) -> list[float]:
    lidar = raw_telemetry.get("lidar")
    values = [-1.0] * LIDAR_SENSOR_COUNT
    if not isinstance(lidar, list):
        return values
    for sensor_idx in range(LIDAR_SENSOR_COUNT):
        raw_value = lidar[sensor_idx] if sensor_idx < len(lidar) else -1.0
        try:
            values[sensor_idx] = float(raw_value)
        except (TypeError, ValueError):
            values[sensor_idx] = -1.0
    return values


def format_group_line(group_name: str, sensor_ids: list[int], values_cm: list[float]) -> str:
    parts = [f"{sensor_idx:02d}={values_cm[sensor_idx]:7.1f}" for sensor_idx in sensor_ids]
    return f"{group_name:<5} " + " ".join(parts)


def console_lines_for_values(values_cm: list[float]) -> list[str]:
    return [
        format_group_line("front", LIDAR_GROUPS["front"], values_cm),
        format_group_line("left", LIDAR_GROUPS["left"], values_cm),
        format_group_line("right", LIDAR_GROUPS["right"], values_cm),
        format_group_line("back", LIDAR_GROUPS["back"], values_cm),
    ]


def make_log_path() -> Path:
    CLEAN_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return CLEAN_LOG_ROOT / f"lidar_read_{timestamp}.csv"


def cleanlog_header() -> list[str]:
    if ENABLE_PING_LOCATION_LOGGING:
        return [
            "iso_time_utc",
            "elapsed_s",
            "step_idx",
            "rover_pos_x",
            "rover_pos_y",
            "rover_pos_z",
            "ping_strength",
        ]
    lidar_columns: list[str] = []
    for sensor_idx, (_, _, sensor_yaw_deg, sensor_pitch_deg) in enumerate(LIDAR_SENSOR_LAYOUT):
        sensor_label = (
            LIDAR_SENSOR_LABELS[sensor_idx]
            if sensor_idx < len(LIDAR_SENSOR_LABELS)
            else f"sensor_{sensor_idx:02d}"
        )
        lidar_columns.append(
            f"lidar_{sensor_idx:02d}_{sensor_label}_yaw_{sensor_yaw_deg:g}_pitch_{sensor_pitch_deg:g}_cm"
        )
    return ["iso_time_utc", "elapsed_s", "step_idx", *RAW_TSS_TELEMETRY_FIELDS, *lidar_columns]


def fmt_raw(value) -> str:
    return json.dumps(value, separators=(",", ":"))


def read_ltv_signal_strength(sock) -> float:
    ltv_json = fetch_ltv_json(sock)
    signal = ltv_json.get("signal", {})
    return float(signal.get("strength", 0.0))


def request_ping_and_read_strength(sock) -> float:
    ping_command = CMD_LTV_PING_UNLIMITED if USE_UNLIMITED_PING else CMD_LTV_PING
    send_float_command(sock, ping_command, 1.0)
    time.sleep(max(0.0, float(PING_SETTLE_SEC)))
    return read_ltv_signal_strength(sock)


def fetch_rover_json_resilient(sock, attempts: int = 5, retry_delay_sec: float = 0.05) -> dict:
    last_error: Exception | None = None
    for attempt_idx in range(max(1, int(attempts))):
        try:
            rover_json = fetch_rover_json(sock)
        except (json.JSONDecodeError, UnicodeDecodeError, RuntimeError) as exc:
            last_error = exc
            if attempt_idx + 1 >= max(1, int(attempts)):
                break
            time.sleep(max(0.0, float(retry_delay_sec)))
            continue
        if not isinstance(rover_json, dict):
            last_error = RuntimeError("ROVER.json response was not a JSON object")
            time.sleep(max(0.0, float(retry_delay_sec)))
            continue
        return rover_json
    if last_error is None:
        raise RuntimeError("Failed to fetch rover JSON for an unknown reason.")
    raise RuntimeError(f"Failed to fetch rover JSON after {max(1, int(attempts))} attempts: {last_error}")


def wait_for_dust_resilient(sock, timeout_seconds: float = 20.0, poll_seconds: float = 0.5) -> bool:
    end_time = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < end_time:
        try:
            rover_json = fetch_rover_json_resilient(sock)
            raw_telemetry = rover_json.get("pr_telemetry")
            if isinstance(raw_telemetry, dict) and bool(raw_telemetry.get("dust_connected", False)):
                return True
        except RuntimeError:
            pass
        time.sleep(max(0.0, float(poll_seconds)))
    try:
        rover_json = fetch_rover_json_resilient(sock)
        raw_telemetry = rover_json.get("pr_telemetry")
        return isinstance(raw_telemetry, dict) and bool(raw_telemetry.get("dust_connected", False))
    except RuntimeError:
        return False


def run_test() -> None:
    log_path = make_log_path()
    sock = open_rover_socket()
    ping_sock = open_rover_socket() if ENABLE_PING_LOCATION_LOGGING else None
    delay_sec = 0.0 if POLL_HZ <= 0.0 else (1.0 / POLL_HZ)
    start_time = time.monotonic()
    step_idx = 0
    last_throttle: float | None = None
    last_steering: float | None = None
    last_brakes: bool | None = None
    status_line_count = 0
    last_console_lines: list[str] | None = None
    last_ping_log_monotonic: float | None = None
    last_ping_strength: float | None = None

    try:
        print("WASD drive, Q/Esc quit.")
        print("Grouped lidar readout logging to", log_path)
        print("front = front-facing cluster, left/right = side sensors, back = rear cluster")
        if ENABLE_PING_LOCATION_LOGGING:
            print("Ping/location logging enabled at 1 Hz; CSV will contain only rover x/y/z and ping.")

        if not wait_for_dust_resilient(sock, timeout_seconds=20.0, poll_seconds=0.5):
            raise RuntimeError("DUST is not connected to TSS.")
        set_lights(sock, True)

        with log_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(cleanlog_header())

            while True:
                throttle_cmd, steering_cmd, exit_requested = desired_drive_state()
                brakes_cmd = throttle_cmd == 0.0

                if throttle_cmd != last_throttle:
                    set_throttle(sock, throttle_cmd)
                    last_throttle = throttle_cmd
                if steering_cmd != last_steering:
                    set_steering(sock, steering_cmd)
                    last_steering = steering_cmd
                if brakes_cmd != last_brakes:
                    set_brakes(sock, brakes_cmd)
                    last_brakes = brakes_cmd

                rover_json = fetch_rover_json_resilient(sock)
                raw_telemetry = rover_json.get("pr_telemetry")
                if not isinstance(raw_telemetry, dict):
                    raise RuntimeError("ROVER.json did not contain pr_telemetry")
                values_cm = lidar_values_cm(raw_telemetry)
                elapsed_s = time.monotonic() - start_time
                now_monotonic = time.monotonic()
                ping_due = (
                    ENABLE_PING_LOCATION_LOGGING
                    and (
                        last_ping_log_monotonic is None
                        or (now_monotonic - last_ping_log_monotonic) >= PING_LOG_INTERVAL_SEC
                    )
                )
                ping_strength: float | None = None
                if ping_due:
                    if ping_sock is None:
                        raise RuntimeError("Ping logging is enabled but no ping socket is available.")
                    ping_strength = request_ping_and_read_strength(ping_sock)
                    last_ping_strength = ping_strength
                    last_ping_log_monotonic = now_monotonic
                row = [
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    f"{elapsed_s:.3f}",
                    str(step_idx),
                ]
                if ENABLE_PING_LOCATION_LOGGING:
                    row.extend(
                        [
                            fmt_raw(raw_telemetry.get("rover_pos_x")),
                            fmt_raw(raw_telemetry.get("rover_pos_y")),
                            fmt_raw(raw_telemetry.get("rover_pos_z")),
                            "" if ping_strength is None else f"{ping_strength:.3f}",
                        ]
                    )
                else:
                    row.extend(fmt_raw(raw_telemetry.get(field)) for field in RAW_TSS_TELEMETRY_FIELDS)
                    row.extend(fmt_raw(value) for value in values_cm)
                writer.writerow(row)
                fh.flush()

                console_lines = console_lines_for_values(values_cm)
                if ENABLE_PING_LOCATION_LOGGING:
                    ping_display = "n/a" if last_ping_strength is None else f"{last_ping_strength:.3f}"
                    console_lines.append(f"ping_strength={ping_display}")
                if console_lines != last_console_lines:
                    rewrite_console_block(console_lines, line_count=status_line_count)
                    status_line_count = len(console_lines)
                    last_console_lines = console_lines

                step_idx += 1
                if exit_requested:
                    print()
                    break
                if delay_sec > 0.0:
                    time.sleep(delay_sec)
    finally:
        try:
            set_throttle(sock, 0.0)
            set_steering(sock, 0.0)
            set_brakes(sock, True)
        finally:
            close_rover_socket(sock)
            if ping_sock is not None:
                close_rover_socket(ping_sock)
        print("Rover commands reset and socket closed.")


if __name__ == "__main__":
    run_test()
