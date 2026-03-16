from __future__ import annotations

import csv
import math
import time
from pathlib import Path

from main import POSE_UNITS_TO_CM, dust_heading_to_math_deg
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
SAMPLE_PERIOD_SECONDS = 0.2
MIN_MOTION_SAMPLE_CM = 5.0
DEFAULT_ANALYSIS_IGNORE_SECONDS = 4.0

CALIBRATION_PHASES = [
    {"name": "settle_start", "duration_s": 3.0, "throttle": 0.0, "steering": 0.0, "brakes": True, "analysis_ignore_s": 3.0},
    {"name": "forward_straight", "duration_s": 14.0, "throttle": 40.0, "steering": 0.0, "brakes": False, "analysis_ignore_s": 4.0},
    {"name": "brake_1", "duration_s": 4.0, "throttle": 0.0, "steering": 0.0, "brakes": True, "analysis_ignore_s": 4.0},
    {"name": "reverse_straight", "duration_s": 14.0, "throttle": -40.0, "steering": 0.0, "brakes": False, "analysis_ignore_s": 4.0},
    {"name": "brake_2", "duration_s": 4.0, "throttle": 0.0, "steering": 0.0, "brakes": True, "analysis_ignore_s": 4.0},
    {"name": "forward_left", "duration_s": 14.0, "throttle": 25.0, "steering": 1.0, "brakes": False, "analysis_ignore_s": 4.0},
    {"name": "brake_3", "duration_s": 4.0, "throttle": 0.0, "steering": 0.0, "brakes": True, "analysis_ignore_s": 4.0},
    {"name": "forward_right", "duration_s": 14.0, "throttle": 25.0, "steering": -1.0, "brakes": False, "analysis_ignore_s": 4.0},
    {"name": "brake_4", "duration_s": 4.0, "throttle": 0.0, "steering": 0.0, "brakes": True, "analysis_ignore_s": 4.0},
]


def wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


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


def candidate_heading_transforms() -> dict[str, callable]:
    return {
        "raw": lambda h: h,
        "raw+90": lambda h: h + 90.0,
        "raw-90": lambda h: h - 90.0,
        "raw+180": lambda h: h + 180.0,
        "90-raw": lambda h: 90.0 - h,
        "-90-raw": lambda h: -90.0 - h,
        "-raw": lambda h: -h,
        "180-raw": lambda h: 180.0 - h,
    }


def append_sample(
    records: list[dict[str, float | str | bool]],
    phase_name: str,
    cmd_throttle: float,
    cmd_steering: float,
    cmd_brakes: bool,
    run_start_monotonic: float,
    telemetry: dict,
) -> None:
    raw_x = float(telemetry.get("rover_pos_x", 0.0))
    raw_y = float(telemetry.get("rover_pos_y", 0.0))
    raw_z = float(telemetry.get("rover_pos_z", 0.0))
    raw_heading = float(telemetry.get("heading", 0.0))
    records.append(
        {
            "elapsed_s": time.monotonic() - run_start_monotonic,
            "phase": phase_name,
            "phase_elapsed_s": 0.0,
            "cmd_throttle": cmd_throttle,
            "cmd_steering": cmd_steering,
            "cmd_brakes": cmd_brakes,
            "raw_rover_x": raw_x,
            "raw_rover_y": raw_y,
            "raw_rover_z": raw_z,
            "raw_heading_deg": raw_heading,
            "rover_x_cm": raw_x * POSE_UNITS_TO_CM,
            "rover_y_cm": raw_y * POSE_UNITS_TO_CM,
            "rover_z_cm": raw_z * POSE_UNITS_TO_CM,
            "rover_heading_math_deg": dust_heading_to_math_deg(raw_heading),
            "telemetry_speed": float(telemetry.get("speed", 0.0)),
            "telemetry_distance_traveled": float(telemetry.get("distance_traveled", 0.0)),
        }
    )


def run_phase(
    sock,
    phase: dict[str, float | bool | str],
    records: list[dict[str, float | str | bool]],
    run_start_monotonic: float,
    next_sample_time: float,
) -> float:
    phase_name = str(phase["name"])
    duration_s = float(phase["duration_s"])
    throttle = float(phase["throttle"])
    steering = float(phase["steering"])
    brakes = bool(phase["brakes"])
    set_brakes(sock, brakes)
    set_steering(sock, steering)
    set_throttle(sock, throttle)
    print(
        f"[phase] {phase_name:>16s} duration={duration_s:4.1f}s "
        f"throttle={throttle:6.1f} steering={steering:5.2f} brakes={int(brakes)}"
    )

    phase_end = time.monotonic() + duration_s
    phase_start = time.monotonic()
    while True:
        now = time.monotonic()
        if now >= phase_end:
            break
        if now < next_sample_time:
            time.sleep(min(0.005, max(0.0, next_sample_time - now)))
            continue
        telemetry = fetch_rover_telemetry(sock)
        append_sample(
            records=records,
            phase_name=phase_name,
            cmd_throttle=throttle,
            cmd_steering=steering,
            cmd_brakes=brakes,
            run_start_monotonic=run_start_monotonic,
            telemetry=telemetry,
        )
        records[-1]["phase_elapsed_s"] = now - phase_start
        next_sample_time += SAMPLE_PERIOD_SECONDS
        if next_sample_time < now:
            next_sample_time = now + SAMPLE_PERIOD_SECONDS
    return next_sample_time


def save_csv(records: list[dict[str, float | str | bool]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "elapsed_s",
        "phase",
        "phase_elapsed_s",
        "cmd_throttle",
        "cmd_steering",
        "cmd_brakes",
        "raw_rover_x",
        "raw_rover_y",
        "raw_rover_z",
        "raw_heading_deg",
        "rover_x_cm",
        "rover_y_cm",
        "rover_z_cm",
        "rover_heading_math_deg",
        "telemetry_speed",
        "telemetry_distance_traveled",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def analyze_records(records: list[dict[str, float | str | bool]]) -> list[str]:
    transforms = candidate_heading_transforms()
    lines: list[str] = []
    lines.append(f"sample_count={len(records)}")
    lines.append(f"pose_units_to_cm={POSE_UNITS_TO_CM}")
    lines.append("current_heading_formula=90-raw")
    phase_ignore = {
        str(phase["name"]): float(phase.get("analysis_ignore_s", DEFAULT_ANALYSIS_IGNORE_SECONDS))
        for phase in CALIBRATION_PHASES
    }

    moving_samples: list[dict[str, float | str | bool]] = []
    for prev, cur in zip(records, records[1:]):
        dx = float(cur["rover_x_cm"]) - float(prev["rover_x_cm"])
        dy = float(cur["rover_y_cm"]) - float(prev["rover_y_cm"])
        pose_delta_cm = math.hypot(dx, dy)
        if pose_delta_cm < MIN_MOTION_SAMPLE_CM:
            continue
        phase_name = str(cur["phase"])
        if float(cur["phase_elapsed_s"]) < phase_ignore.get(phase_name, DEFAULT_ANALYSIS_IGNORE_SECONDS):
            continue
        moving_samples.append(
            {
                "phase": phase_name,
                "cmd_throttle": cur["cmd_throttle"],
                "cmd_steering": cur["cmd_steering"],
                "raw_heading_deg": cur["raw_heading_deg"],
                "motion_bearing_deg": math.degrees(math.atan2(dy, dx)),
                "pose_delta_cm": pose_delta_cm,
            }
        )

    lines.append(f"moving_sample_count={len(moving_samples)}")

    groups = {
        "forward_straight": [
            s for s in moving_samples if str(s["phase"]) == "forward_straight"
        ],
        "reverse_straight": [
            s for s in moving_samples if str(s["phase"]) == "reverse_straight"
        ],
        "forward_left": [
            s for s in moving_samples if str(s["phase"]) == "forward_left"
        ],
        "forward_right": [
            s for s in moving_samples if str(s["phase"]) == "forward_right"
        ],
        "all_forward": [
            s for s in moving_samples if float(s["cmd_throttle"]) > 0.0
        ],
        "all_reverse": [
            s for s in moving_samples if float(s["cmd_throttle"]) < 0.0
        ],
    }

    def ang_err(a: float, b: float) -> float:
        return wrap_angle_deg(a - b)

    for group_name, group_rows in groups.items():
        lines.append(f"[{group_name}] count={len(group_rows)}")
        if not group_rows:
            continue
        scores: list[tuple[float, str]] = []
        for name, fn in transforms.items():
            errs: list[float] = []
            for row in group_rows:
                raw_heading = float(row["raw_heading_deg"])
                motion_bearing = float(row["motion_bearing_deg"])
                forward_bearing = fn(raw_heading)
                expected_motion = forward_bearing
                if float(row["cmd_throttle"]) < 0.0:
                    expected_motion += 180.0
                errs.append(abs(ang_err(motion_bearing, expected_motion)))
            score = sum(errs) / len(errs)
            scores.append((score, name))
        scores.sort()
        for score, name in scores[:4]:
            lines.append(f"{group_name}.{name}.mean_abs_err_deg={score:.3f}")
    return lines


def save_summary(records: list[dict[str, float | str | bool]], out_txt: Path) -> None:
    lines = analyze_records(records)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = RUNS_DIR / f"motion_calibration_{timestamp}.csv"
    summary_path = RUNS_DIR / f"motion_calibration_{timestamp}_summary.txt"

    sock = open_rover_socket()
    records: list[dict[str, float | str | bool]] = []
    try:
        if not wait_for_dust(sock, timeout_seconds=20.0, poll_seconds=0.5):
            raise RuntimeError("DUST is not connected to TSS.")
        run_start = time.monotonic()
        next_sample_time = run_start
        for phase in CALIBRATION_PHASES:
            next_sample_time = run_phase(sock, phase, records, run_start, next_sample_time)
    finally:
        stop_rover(sock)
        close_rover_socket(sock)

    save_csv(records, csv_path)
    save_summary(records, summary_path)
    print(f"Saved calibration CSV: {csv_path}")
    print(f"Saved calibration summary: {summary_path}")


if __name__ == "__main__":
    main()
