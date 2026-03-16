from __future__ import annotations

import csv
import ctypes
import time
from datetime import datetime
from pathlib import Path

from rover_control import (
    close_rover_socket,
    fetch_rover_telemetry,
    open_rover_socket,
    set_brakes,
    set_steering,
    set_throttle,
    wait_for_dust,
)


RUNS_DIR = Path("runs")
POLL_HZ = 20.0
MAX_THROTTLE = 100.0
MAX_STEERING = 1.0
FRONT_LIDAR_SENSOR_INDEX = 2
FRONT_LIDAR_SENSOR_LABEL = "front_center_frame_forward"
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
        print(line[: max(0, width - 1)], end="")

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


def front_lidar_distance_cm(telemetry: dict) -> float:
    lidar = telemetry.get("lidar")
    if not isinstance(lidar, list) or FRONT_LIDAR_SENSOR_INDEX >= len(lidar):
        return -1.0
    try:
        return float(lidar[FRONT_LIDAR_SENSOR_INDEX])
    except (TypeError, ValueError):
        return -1.0


def active_lidar_hits_text(telemetry: dict) -> str:
    lidar = telemetry.get("lidar")
    if not isinstance(lidar, list):
        return "active_hits=none"

    active_hits: list[str] = []
    for sensor_idx, raw_value in enumerate(lidar):
        try:
            distance_cm = float(raw_value)
        except (TypeError, ValueError):
            continue
        if distance_cm <= 0.0:
            continue
        sensor_label = (
            LIDAR_SENSOR_LABELS[sensor_idx]
            if 0 <= sensor_idx < len(LIDAR_SENSOR_LABELS)
            else f"sensor_{sensor_idx}"
        )
        active_hits.append(f"{sensor_idx}:{sensor_label}={distance_cm:.1f}")

    if not active_hits:
        return "active_hits=none"
    return "active_hits=" + ", ".join(active_hits)


def make_log_path() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return RUNS_DIR / f"front_lidar_test_{timestamp}.csv"


def run_test() -> None:
    log_path = make_log_path()
    sock = open_rover_socket()
    delay_sec = 0.0 if POLL_HZ <= 0.0 else (1.0 / POLL_HZ)
    start_time = time.monotonic()
    step_idx = 0
    last_throttle: float | None = None
    last_steering: float | None = None
    last_brakes: bool | None = None
    status_line_count = 0

    try:
        print("Controls: hold W/A/S/D to drive, press Q or Esc to quit.")
        print(f"Logging front lidar sensor {FRONT_LIDAR_SENSOR_INDEX} ({FRONT_LIDAR_SENSOR_LABEL}) to {log_path}")

        if not wait_for_dust(sock, timeout_seconds=20.0, poll_seconds=0.5):
            raise RuntimeError("DUST is not connected to TSS.")

        with log_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "elapsed_s",
                    "step_idx",
                    "iso_time",
                    "throttle_cmd",
                    "steering_cmd",
                    "brakes_cmd",
                    "front_lidar_sensor_idx",
                    "front_lidar_sensor_label",
                    "front_lidar_distance_cm",
                    "telemetry_throttle",
                    "telemetry_steering",
                    "telemetry_speed",
                ]
            )

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

                telemetry = fetch_rover_telemetry(sock)
                front_lidar_cm = front_lidar_distance_cm(telemetry)
                elapsed_s = time.monotonic() - start_time
                iso_time = datetime.now().isoformat(timespec="milliseconds")
                telemetry_throttle = float(telemetry.get("throttle", 0.0))
                telemetry_steering = float(telemetry.get("steering", 0.0))
                telemetry_speed = float(telemetry.get("speed", 0.0))
                active_hits_text = active_lidar_hits_text(telemetry)

                writer.writerow(
                    [
                        f"{elapsed_s:.3f}",
                        step_idx,
                        iso_time,
                        f"{throttle_cmd:.3f}",
                        f"{steering_cmd:.3f}",
                        int(brakes_cmd),
                        FRONT_LIDAR_SENSOR_INDEX,
                        FRONT_LIDAR_SENSOR_LABEL,
                        f"{front_lidar_cm:.3f}",
                        f"{telemetry_throttle:.3f}",
                        f"{telemetry_steering:.3f}",
                        f"{telemetry_speed:.3f}",
                    ]
                )
                fh.flush()

                rewrite_console_block(
                    [
                        (
                            f"step={step_idx:05d} throttle={throttle_cmd:7.1f} steering={steering_cmd:4.1f} "
                            f"brakes={int(brakes_cmd)} front_lidar_cm={front_lidar_cm:8.1f}"
                        ),
                        active_hits_text,
                    ],
                    line_count=status_line_count,
                )
                status_line_count = 2

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
        print("Rover commands reset and socket closed.")


if __name__ == "__main__":
    run_test()
