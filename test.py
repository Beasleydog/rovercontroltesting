from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rover_control import (
    close_rover_socket,
    fetch_rover_json,
    fetch_rover_telemetry,
    open_rover_socket,
    set_brakes,
    set_steering,
    set_throttle,
    wait_for_dust,
)


RUNS_DIR = Path("runs")
SAMPLE_PERIOD_SECONDS = 0.5
REVERSE_LEFT_SECONDS = 5.0
BRAKE_SECONDS = 1.0
FORWARD_SECONDS = 20.0
USE_RAW_LIDAR_VALUES = False

REVERSE_THROTTLE = -100.0
FORWARD_THROTTLE = 100.0
LEFT_STEERING = -1.0

LIDAR_NO_HIT_VALUE = -1.0
LIDAR_PLOT_MAX_CM = 1050.0
LIDAR_SENSOR_COUNT = 17
# Sensor 2 in DUST docs: front center of vehicle frame, vehicle forward.
FRONT_CENTER_FORWARD_SENSOR_IDX = 2
FRONT_CENTER_FORWARD_SENSOR_LABEL = "sensor_02_front_center_forward"
POSE_UNITS_TO_CM = 100.0


PHASES = [
    {
        "name": "reverse_left",
        "duration_s": REVERSE_LEFT_SECONDS,
        "throttle": REVERSE_THROTTLE,
        "steering": LEFT_STEERING,
        "brakes": False,
    },
    {
        "name": "brake",
        "duration_s": BRAKE_SECONDS,
        "throttle": 0.0,
        "steering": 0.0,
        "brakes": True,
    },
    {
        "name": "forward_full",
        "duration_s": FORWARD_SECONDS,
        "throttle": FORWARD_THROTTLE,
        "steering": 0.0,
        "brakes": False,
    },
]


def parse_lidar(telemetry: dict) -> np.ndarray:
    raw = telemetry.get("lidar", [])
    lidar = np.asarray(raw, dtype=np.float32).reshape(-1)
    if lidar.size < LIDAR_SENSOR_COUNT:
        padded = np.full((LIDAR_SENSOR_COUNT,), LIDAR_NO_HIT_VALUE, dtype=np.float32)
        padded[: lidar.size] = lidar
        return padded
    if lidar.size > LIDAR_SENSOR_COUNT:
        return lidar[:LIDAR_SENSOR_COUNT]
    return lidar


def fetch_test_telemetry(sock, use_raw_lidar_values: bool) -> dict:
    if not use_raw_lidar_values:
        return fetch_rover_telemetry(sock)

    rover = fetch_rover_json(sock)
    telemetry = rover.get("pr_telemetry")
    if not isinstance(telemetry, dict):
        raise RuntimeError("ROVER.json did not contain pr_telemetry")
    return telemetry


def wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def dust_heading_to_math_deg(raw_heading_deg: float) -> float:
    return wrap_angle_deg(90.0 - raw_heading_deg)


def run_phase(
    sock,
    phase_name: str,
    duration_s: float,
    throttle: float,
    steering: float,
    brakes: bool,
    records: list[dict[str, object]],
    run_start_monotonic: float,
    next_sample_time: float,
) -> float:
    set_brakes(sock, brakes)
    set_steering(sock, steering)
    set_throttle(sock, throttle)

    phase_start = time.monotonic()
    phase_end = phase_start + duration_s
    print(
        f"[phase] {phase_name:>18s}  duration={duration_s:5.2f}s  "
        f"throttle={throttle:6.2f} steering={steering:5.2f} brakes={int(brakes)}"
    )

    while True:
        now = time.monotonic()
        if now >= phase_end:
            break

        if now < next_sample_time:
            sleep_s = min(0.002, max(0.0, next_sample_time - now))
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            continue

        telemetry = fetch_test_telemetry(sock, use_raw_lidar_values=USE_RAW_LIDAR_VALUES)
        lidar = parse_lidar(telemetry)
        records.append(
            {
                "elapsed_s": now - run_start_monotonic,
                "phase": phase_name,
                "speed": float(telemetry.get("speed", 0.0)),
                "heading_deg": dust_heading_to_math_deg(float(telemetry.get("heading", 0.0))),
                "rover_x_cm": float(telemetry.get("rover_pos_x", 0.0)) * POSE_UNITS_TO_CM,
                "rover_y_cm": float(telemetry.get("rover_pos_y", 0.0)) * POSE_UNITS_TO_CM,
                "rover_z_cm": float(telemetry.get("rover_pos_z", 0.0)) * POSE_UNITS_TO_CM,
                "lidar": lidar.copy(),
            }
        )

        next_sample_time += SAMPLE_PERIOD_SECONDS
        if next_sample_time < now:
            next_sample_time = now + SAMPLE_PERIOD_SECONDS

    return next_sample_time


def save_lidar_csv(records: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "elapsed_s",
        "phase",
        "speed",
        "heading_deg",
        "rover_x_cm",
        "rover_y_cm",
        "rover_z_cm",
    ] + [f"lidar_{i:02d}_cm" for i in range(LIDAR_SENSOR_COUNT)]

    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for row in records:
            lidar = np.asarray(row["lidar"], dtype=np.float32).reshape(-1)
            writer.writerow(
                [
                    f"{float(row['elapsed_s']):.6f}",
                    str(row["phase"]),
                    f"{float(row['speed']):.6f}",
                    f"{float(row['heading_deg']):.6f}",
                    f"{float(row['rover_x_cm']):.6f}",
                    f"{float(row['rover_y_cm']):.6f}",
                    f"{float(row['rover_z_cm']):.6f}",
                    *[f"{float(v):.6f}" for v in lidar],
                ]
            )


def save_lidar_plot(records: list[dict[str, object]], out_png: Path) -> None:
    if not records:
        raise RuntimeError("No lidar samples were recorded; cannot build plot.")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    t = np.asarray([float(r["elapsed_s"]) for r in records], dtype=np.float32)
    d = np.stack([np.asarray(r["lidar"], dtype=np.float32) for r in records], axis=0)
    if not (0 <= FRONT_CENTER_FORWARD_SENSOR_IDX < d.shape[1]):
        raise RuntimeError(
            f"Configured FRONT_CENTER_FORWARD_SENSOR_IDX={FRONT_CENTER_FORWARD_SENSOR_IDX} "
            f"is outside recorded lidar columns ({d.shape[1]})."
        )
    d_sensor = d[:, FRONT_CENTER_FORWARD_SENSOR_IDX].astype(np.float32, copy=False)
    d_sensor[d_sensor < 0.0] = np.nan

    fig, ax = plt.subplots(figsize=(20, 11), dpi=220)
    ax.plot(
        t,
        d_sensor,
        linewidth=1.1,
        alpha=0.95,
        color="#ff7f0e",
        label=FRONT_CENTER_FORWARD_SENSOR_LABEL,
    )

    phase_elapsed = 0.0
    for phase in PHASES[:-1]:
        phase_elapsed += float(phase["duration_s"])
        ax.axvline(phase_elapsed, linestyle="--", linewidth=1.0, color="black", alpha=0.6)

    ax.set_title("LIDAR Distance Over Time (Sensor 2: Front Center Forward)")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Distance (cm)")
    ax.set_ylim(0.0, LIDAR_PLOT_MAX_CM)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def stop_rover(sock) -> None:
    try:
        set_steering(sock, 0.0)
    except Exception:
        pass
    try:
        set_throttle(sock, 0.0)
    except Exception:
        pass
    try:
        set_brakes(sock, True)
    except Exception:
        pass


def main() -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = RUNS_DIR / f"lidar_sequence_{timestamp}.csv"
    plot_path = RUNS_DIR / f"lidar_sequence_{timestamp}.png"

    sock = open_rover_socket()
    records: list[dict[str, object]] = []

    try:
        if not wait_for_dust(sock, timeout_seconds=20.0, poll_seconds=0.5):
            raise RuntimeError("DUST is not connected to TSS.")

        run_start = time.monotonic()
        next_sample_time = run_start

        for phase in PHASES:
            next_sample_time = run_phase(
                sock=sock,
                phase_name=str(phase["name"]),
                duration_s=float(phase["duration_s"]),
                throttle=float(phase["throttle"]),
                steering=float(phase["steering"]),
                brakes=bool(phase["brakes"]),
                records=records,
                run_start_monotonic=run_start,
                next_sample_time=next_sample_time,
            )

    finally:
        stop_rover(sock)
        close_rover_socket(sock)

    save_lidar_csv(records, csv_path)
    save_lidar_plot(records, plot_path)
    print(f"Use raw lidar values: {USE_RAW_LIDAR_VALUES}")
    print(f"Saved lidar samples: {csv_path}")
    print(f"Saved lidar plot:    {plot_path}")


if __name__ == "__main__":
    main()
