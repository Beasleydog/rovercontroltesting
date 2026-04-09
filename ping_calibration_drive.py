from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path

from dumbdrive import REMOTE_SERVER, REMOTE_SERVER_URL
from dumblocate import (
    PING_LOG_INTERVAL_SEC,
    PING_SETTLE_SEC,
    PingSample,
    PingStrengthSampler,
    drive_to_goal_locate,
    fetch_pose_and_telemetry,
    hold_with_ui_updates,
    local_cm_to_raw_world_m,
    raw_world_m_to_local_cm,
    sample_ping,
)
from main import stop_rover
from rover_control import (
    close_rover_socket,
    configure_remote_server,
    fetch_ltv_json,
    open_rover_socket,
    set_lights,
    wait_for_dust,
)


SURVEY_RADIUS_M = 40.0
HEX_RADIUS_M = 30.0
SURVEY_GOAL_REACHED_CM = 2500.0
LAST_KNOWN_ENTRY_RADIUS_CM = 5000.0
DEFAULT_LOOP_COUNT = 12
FIT_SEARCH_RADIUS_M = 120.0
FIT_COARSE_STEP_M = 5.0


class SurveyLogger:
    def __init__(self) -> None:
        root = Path("runs")
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"ping_calibration_drive_{time.strftime('%Y%m%d_%H%M%S')}.tsv"
        self._file = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._file, delimiter="\t")
        self._writer.writerow(
            [
                "iso_time_utc",
                "mono_s",
                "sample_kind",
                "sample_idx",
                "planned_x_m",
                "planned_y_m",
                "rover_x_m",
                "rover_y_m",
                "rover_z_m",
                "heading_deg",
                "goal_error_m",
                "ping_value",
                "predicted_radius_m",
                "fit_a",
                "fit_b",
                "suspected_x_m",
                "suspected_y_m",
                "last_known_x_m",
                "last_known_y_m",
            ]
        )
        self._file.flush()
        print(f"Calibration log: {self.path}")

    def log(
        self,
        *,
        sample_kind: str,
        sample_idx: int,
        planned_x_m: float,
        planned_y_m: float,
        rover_x_m: float,
        rover_y_m: float,
        rover_z_m: float,
        heading_deg: float,
        goal_error_m: float,
        ping_value: float,
        predicted_radius_m: float,
        fit_a: float | None,
        fit_b: float | None,
        suspected_x_m: float | None,
        suspected_y_m: float | None,
        last_known_x_m: float,
        last_known_y_m: float,
    ) -> None:
        def maybe(value: float | None) -> str:
            return "" if value is None else f"{value:.6f}"

        self._writer.writerow(
            [
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                f"{time.monotonic():.3f}",
                sample_kind,
                sample_idx,
                f"{planned_x_m:.3f}",
                f"{planned_y_m:.3f}",
                f"{rover_x_m:.3f}",
                f"{rover_y_m:.3f}",
                f"{rover_z_m:.3f}",
                f"{heading_deg:.3f}",
                f"{goal_error_m:.3f}",
                f"{ping_value:.3f}",
                f"{predicted_radius_m:.3f}",
                maybe(fit_a),
                maybe(fit_b),
                maybe(suspected_x_m),
                maybe(suspected_y_m),
                f"{last_known_x_m:.3f}",
                f"{last_known_y_m:.3f}",
            ]
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def hex_points_cm(center_xy: tuple[float, float], radius_m: float) -> list[tuple[float, float]]:
    radius_cm = float(radius_m) * 100.0
    points: list[tuple[float, float]] = []
    for angle_deg in range(0, 360, 60):
        angle_rad = math.radians(float(angle_deg))
        points.append(
            (
                center_xy[0] + math.cos(angle_rad) * radius_cm,
                center_xy[1] + math.sin(angle_rad) * radius_cm,
            )
        )
    return points


def random_point_in_disk_cm(
    rng: random.Random,
    center_xy: tuple[float, float],
    radius_m: float,
) -> tuple[float, float]:
    radius_cm = float(radius_m) * 100.0
    angle = rng.uniform(0.0, 2.0 * math.pi)
    radial_scale = math.sqrt(rng.random())
    offset_cm = radius_cm * radial_scale
    return (
        center_xy[0] + math.cos(angle) * offset_cm,
        center_xy[1] + math.sin(angle) * offset_cm,
    )


def fit_formula_for_candidate(
    samples: list[PingSample],
    candidate_x_m: float,
    candidate_y_m: float,
) -> tuple[float, float, float]:
    distances = [
        math.hypot(candidate_x_m - sample.rover_x_m, candidate_y_m - sample.rover_y_m)
        for sample in samples
    ]
    if any(distance <= 1e-6 for distance in distances):
        raise ValueError("Candidate is too close to a sample center.")
    mean_ping = sum(sample.ping_value for sample in samples) / len(samples)
    mean_log_distance = sum(math.log(distance) for distance in distances) / len(distances)
    var_ping = sum((sample.ping_value - mean_ping) ** 2 for sample in samples)
    if var_ping <= 1e-12:
        raise ValueError("Ping variance is too small to fit exponential parameters.")
    decay_b = sum(
        (sample.ping_value - mean_ping) * (math.log(distance) - mean_log_distance)
        for sample, distance in zip(samples, distances)
    ) / var_ping
    scale_a = math.exp(mean_log_distance - decay_b * mean_ping)
    if scale_a <= 0.0:
        raise ValueError("Invalid exponential scale.")
    residual_sse = sum(
        (scale_a * math.exp(decay_b * sample.ping_value) - distance) ** 2
        for sample, distance in zip(samples, distances)
    )
    return (scale_a, decay_b, residual_sse)


def fit_location_and_formula(
    samples: list[PingSample],
    initial_x_m: float,
    initial_y_m: float,
) -> tuple[float, float, float, float, float]:
    best: tuple[float, float, float, float, float] | None = None
    coarse_steps = int((2.0 * FIT_SEARCH_RADIUS_M) / FIT_COARSE_STEP_M)
    for ix in range(coarse_steps + 1):
        candidate_x_m = initial_x_m - FIT_SEARCH_RADIUS_M + ix * FIT_COARSE_STEP_M
        for iy in range(coarse_steps + 1):
            candidate_y_m = initial_y_m - FIT_SEARCH_RADIUS_M + iy * FIT_COARSE_STEP_M
            try:
                scale_a, decay_b, residual_sse = fit_formula_for_candidate(
                    samples,
                    candidate_x_m,
                    candidate_y_m,
                )
            except ValueError:
                continue
            if best is None or residual_sse < best[4]:
                best = (candidate_x_m, candidate_y_m, scale_a, decay_b, residual_sse)
    if best is None:
        raise RuntimeError("Could not find a valid location/formula fit from the hexagon samples.")

    best_x_m, best_y_m, best_a, best_b, best_error = best
    for step_m in (2.0, 1.0, 0.5, 0.25, 0.1):
        improved = True
        while improved:
            improved = False
            for dx in (-step_m, 0.0, step_m):
                for dy in (-step_m, 0.0, step_m):
                    candidate_x_m = best_x_m + dx
                    candidate_y_m = best_y_m + dy
                    try:
                        scale_a, decay_b, residual_sse = fit_formula_for_candidate(
                            samples,
                            candidate_x_m,
                            candidate_y_m,
                        )
                    except ValueError:
                        continue
                    if residual_sse < best_error:
                        best_x_m = candidate_x_m
                        best_y_m = candidate_y_m
                        best_a = scale_a
                        best_b = decay_b
                        best_error = residual_sse
                        improved = True
    return (best_x_m, best_y_m, best_a, best_b, best_error)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate ping formula by sampling a hexagon near last known, then looping between the fit estimate and random nearby points."
    )
    parser.add_argument("--count", type=int, default=DEFAULT_LOOP_COUNT, help="Number of estimate/random loop pairs.")
    parser.add_argument("--radius-m", type=float, default=SURVEY_RADIUS_M, help="Random-point radius around last known in meters.")
    parser.add_argument("--hex-radius-m", type=float, default=HEX_RADIUS_M, help="Hexagon radius around last known in meters.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible random points.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    configure_remote_server(REMOTE_SERVER, REMOTE_SERVER_URL)
    sock = open_rover_socket()
    logger: SurveyLogger | None = None
    ping_sampler = PingStrengthSampler(PING_LOG_INTERVAL_SEC)
    try:
        if not wait_for_dust(sock, timeout_seconds=20.0, poll_seconds=0.5):
            raise RuntimeError("DUST is not connected to TSS.")
        set_lights(sock, True)
        logger = SurveyLogger()

        ltv_json = fetch_ltv_json(sock)
        location = ltv_json.get("location", {})
        last_known_x_m = float(location.get("last_known_x", 0.0))
        last_known_y_m = float(location.get("last_known_y", 0.0))
        center_xy = raw_world_m_to_local_cm(last_known_x_m, last_known_y_m)
        print(
            f"Last known LTV location: ({last_known_x_m:.3f}, {last_known_y_m:.3f}) m | "
            f"random radius {args.radius_m:.1f} m | hex radius {args.hex_radius_m:.1f} m | loops {args.count}"
        )

        fit_state = {
            "a": None,
            "b": None,
            "suspected_x_m": None,
            "suspected_y_m": None,
        }

        def make_drive_logger(
            *,
            sample_kind: str,
            sample_idx: int,
            planned_x_m: float,
            planned_y_m: float,
        ):
            def log_drive_sample(
                *,
                phase: str,
                raw_telemetry: dict,
                rover_xyzh: tuple[float, float, float, float],
                goal_xy: tuple[float, float],
                goal_distance_cm: float,
            ) -> None:
                ping_value, sampled_now = ping_sampler.sample(sock)
                if not sampled_now:
                    return
                rover_x_m = float(raw_telemetry.get("rover_pos_x", 0.0))
                rover_y_m = float(raw_telemetry.get("rover_pos_y", 0.0))
                rover_z_m = float(raw_telemetry.get("rover_pos_z", 0.0))
                predicted_radius_m = (
                    float("nan")
                    if fit_state["a"] is None or fit_state["b"] is None
                    else fit_state["a"] * math.exp(fit_state["b"] * ping_value)
                )
                print(
                    f"{sample_kind} ping sample {sample_idx}: "
                    f"rover=({rover_x_m:.3f}, {rover_y_m:.3f}) m | ping={ping_value:.3f}"
                )
                logger.log(
                    sample_kind=sample_kind,
                    sample_idx=sample_idx,
                    planned_x_m=planned_x_m,
                    planned_y_m=planned_y_m,
                    rover_x_m=rover_x_m,
                    rover_y_m=rover_y_m,
                    rover_z_m=rover_z_m,
                    heading_deg=rover_xyzh[3],
                    goal_error_m=goal_distance_cm / 100.0,
                    ping_value=ping_value,
                    predicted_radius_m=predicted_radius_m,
                    fit_a=fit_state["a"],
                    fit_b=fit_state["b"],
                    suspected_x_m=fit_state["suspected_x_m"],
                    suspected_y_m=fit_state["suspected_y_m"],
                    last_known_x_m=last_known_x_m,
                    last_known_y_m=last_known_y_m,
                )
            return log_drive_sample

        def log_stop_sample(
            *,
            sample_kind: str,
            sample_idx: int,
            planned_x_m: float,
            planned_y_m: float,
            goal_xy: tuple[float, float],
        ) -> PingSample:
            x_cm, y_cm, _z_cm, heading_deg, raw_telemetry = fetch_pose_and_telemetry(sock)
            ping_sample = sample_ping(sock, raw_telemetry)
            goal_error_m = math.hypot(goal_xy[0] - x_cm, goal_xy[1] - y_cm) / 100.0
            print(
                f"{sample_kind} stop sample {sample_idx}: "
                f"rover=({ping_sample.rover_x_m:.3f}, {ping_sample.rover_y_m:.3f}) m | "
                f"ping={ping_sample.ping_value:.3f} | goal_error={goal_error_m:.3f} m"
            )
            logger.log(
                sample_kind=sample_kind,
                sample_idx=sample_idx,
                planned_x_m=planned_x_m,
                planned_y_m=planned_y_m,
                rover_x_m=ping_sample.rover_x_m,
                rover_y_m=ping_sample.rover_y_m,
                rover_z_m=float(raw_telemetry.get("rover_pos_z", 0.0)),
                heading_deg=heading_deg,
                goal_error_m=goal_error_m,
                ping_value=ping_sample.ping_value,
                predicted_radius_m=ping_sample.radius_m,
                fit_a=fit_state["a"],
                fit_b=fit_state["b"],
                suspected_x_m=fit_state["suspected_x_m"],
                suspected_y_m=fit_state["suspected_y_m"],
                last_known_x_m=last_known_x_m,
                last_known_y_m=last_known_y_m,
            )
            return ping_sample

        print(
            f"Driving to last known before calibration sweep: "
            f"({last_known_x_m:.3f}, {last_known_y_m:.3f}) m"
        )
        run_state = drive_to_goal_locate(
            sock,
            final_goal_xy=center_xy,
            goal_label=f"{last_known_x_m:.3f}, {last_known_y_m:.3f}",
            viewer=None,
            recorded_obstacle_points=[],
            obstacle_total=0,
            start_time=None,
            step_idx=0,
            total_traveled_cm=0.0,
            goals_reached=0,
            telemetry_callback=None,
            debug_logger=None,
            debug_mode="ping_calibration_drive_last_known",
            goal_reached_cm=SURVEY_GOAL_REACHED_CM,
        )
        if run_state.aborted:
            print("Drive to last known aborted; stopping calibration run.")
            return
        last_known_remaining_cm = math.hypot(
            center_xy[0] - run_state.pose_xyzh[0],
            center_xy[1] - run_state.pose_xyzh[1],
        )
        print(
            f"Distance from last known before hex sweep: "
            f"{last_known_remaining_cm / 100.0:.3f} m"
        )
        if last_known_remaining_cm > LAST_KNOWN_ENTRY_RADIUS_CM:
            raise RuntimeError(
                "Calibration sweep requires the rover to get within 50 m of last known "
                f"before starting hex pings; current distance is {last_known_remaining_cm / 100.0:.3f} m"
            )

        hex_samples: list[PingSample] = []
        for hex_idx, hex_goal_xy in enumerate(hex_points_cm(center_xy, args.hex_radius_m), start=1):
            hex_goal_x_m, hex_goal_y_m = local_cm_to_raw_world_m(*hex_goal_xy)
            print(
                f"Driving to hex ping point {hex_idx}/6: "
                f"({hex_goal_x_m:.3f}, {hex_goal_y_m:.3f}) m"
            )
            hex_logger = make_drive_logger(
                sample_kind="hex_drive",
                sample_idx=hex_idx,
                planned_x_m=hex_goal_x_m,
                planned_y_m=hex_goal_y_m,
            )
            run_state = drive_to_goal_locate(
                sock,
                final_goal_xy=hex_goal_xy,
                goal_label=f"{hex_goal_x_m:.3f}, {hex_goal_y_m:.3f}",
                viewer=run_state.viewer,
                recorded_obstacle_points=run_state.recorded_obstacle_points,
                obstacle_total=run_state.obstacle_total,
                start_time=run_state.start_time,
                step_idx=run_state.step_idx,
                total_traveled_cm=run_state.total_traveled_cm,
                goals_reached=0,
                telemetry_callback=hex_logger,
                debug_logger=None,
                debug_mode=f"ping_calibration_hex_drive_{hex_idx}",
                goal_reached_cm=SURVEY_GOAL_REACHED_CM,
            )
            if run_state.aborted:
                print("Drive to hex point aborted; stopping calibration run.")
                return
            if not hold_with_ui_updates(
                sock,
                viewer=run_state.viewer,
                planner=run_state.planner,
                goal_xy=run_state.goal_xy,
                obstacle_total=run_state.obstacle_total,
                start_time=run_state.start_time,
                total_traveled_cm=run_state.total_traveled_cm,
                duration_s=PING_SETTLE_SEC,
                status=f"Settling at hex point {hex_idx}...",
                telemetry_callback=hex_logger,
                debug_logger=None,
                debug_mode=f"ping_calibration_hex_hold_{hex_idx}",
            ):
                print("Hold at hex point aborted; stopping calibration run.")
                return
            hex_samples.append(
                log_stop_sample(
                    sample_kind="hex_stop",
                    sample_idx=hex_idx,
                    planned_x_m=hex_goal_x_m,
                    planned_y_m=hex_goal_y_m,
                    goal_xy=hex_goal_xy,
                )
            )

        initial_guess_x_m = sum(sample.rover_x_m for sample in hex_samples) / len(hex_samples)
        initial_guess_y_m = sum(sample.rover_y_m for sample in hex_samples) / len(hex_samples)
        suspected_x_m, suspected_y_m, fit_a, fit_b, fit_error = fit_location_and_formula(
            hex_samples,
            initial_guess_x_m,
            initial_guess_y_m,
        )
        fit_state["a"] = fit_a
        fit_state["b"] = fit_b
        fit_state["suspected_x_m"] = suspected_x_m
        fit_state["suspected_y_m"] = suspected_y_m
        suspected_xy = raw_world_m_to_local_cm(suspected_x_m, suspected_y_m)
        print(
            f"Hex fit estimate: ({suspected_x_m:.3f}, {suspected_y_m:.3f}) m | "
            f"A={fit_a:.9f} | B={fit_b:.9f} | residual_sse={fit_error:.6f}"
        )

        current_sample_idx = 0
        for loop_idx in range(1, args.count + 1):
            current_sample_idx += 1
            suspected_goal_x_m, suspected_goal_y_m = local_cm_to_raw_world_m(*suspected_xy)
            print(
                f"Loop {loop_idx}/{args.count}: driving to suspected location "
                f"({suspected_goal_x_m:.3f}, {suspected_goal_y_m:.3f}) m"
            )
            estimate_logger = make_drive_logger(
                sample_kind="drive_to_suspected",
                sample_idx=current_sample_idx,
                planned_x_m=suspected_goal_x_m,
                planned_y_m=suspected_goal_y_m,
            )
            run_state = drive_to_goal_locate(
                sock,
                final_goal_xy=suspected_xy,
                goal_label=f"{suspected_goal_x_m:.3f}, {suspected_goal_y_m:.3f}",
                viewer=run_state.viewer,
                recorded_obstacle_points=run_state.recorded_obstacle_points,
                obstacle_total=run_state.obstacle_total,
                start_time=run_state.start_time,
                step_idx=run_state.step_idx,
                total_traveled_cm=run_state.total_traveled_cm,
                goals_reached=0,
                telemetry_callback=estimate_logger,
                debug_logger=None,
                debug_mode=f"ping_calibration_drive_suspected_{loop_idx}",
                goal_reached_cm=SURVEY_GOAL_REACHED_CM,
            )
            if run_state.aborted:
                print("Drive to suspected location aborted; stopping calibration run.")
                return
            if not hold_with_ui_updates(
                sock,
                viewer=run_state.viewer,
                planner=run_state.planner,
                goal_xy=run_state.goal_xy,
                obstacle_total=run_state.obstacle_total,
                start_time=run_state.start_time,
                total_traveled_cm=run_state.total_traveled_cm,
                duration_s=PING_SETTLE_SEC,
                status=f"Settling at suspected location loop {loop_idx}...",
                telemetry_callback=estimate_logger,
                debug_logger=None,
                debug_mode=f"ping_calibration_hold_suspected_{loop_idx}",
            ):
                print("Hold at suspected location aborted; stopping calibration run.")
                return
            log_stop_sample(
                sample_kind="stop_at_suspected",
                sample_idx=current_sample_idx,
                planned_x_m=suspected_goal_x_m,
                planned_y_m=suspected_goal_y_m,
                goal_xy=suspected_xy,
            )

            current_sample_idx += 1
            random_goal_xy = random_point_in_disk_cm(rng, center_xy, args.radius_m)
            random_goal_x_m, random_goal_y_m = local_cm_to_raw_world_m(*random_goal_xy)
            print(
                f"Loop {loop_idx}/{args.count}: driving to random point "
                f"({random_goal_x_m:.3f}, {random_goal_y_m:.3f}) m"
            )
            random_logger = make_drive_logger(
                sample_kind="drive_to_random",
                sample_idx=current_sample_idx,
                planned_x_m=random_goal_x_m,
                planned_y_m=random_goal_y_m,
            )
            run_state = drive_to_goal_locate(
                sock,
                final_goal_xy=random_goal_xy,
                goal_label=f"{random_goal_x_m:.3f}, {random_goal_y_m:.3f}",
                viewer=run_state.viewer,
                recorded_obstacle_points=run_state.recorded_obstacle_points,
                obstacle_total=run_state.obstacle_total,
                start_time=run_state.start_time,
                step_idx=run_state.step_idx,
                total_traveled_cm=run_state.total_traveled_cm,
                goals_reached=0,
                telemetry_callback=random_logger,
                debug_logger=None,
                debug_mode=f"ping_calibration_drive_random_{loop_idx}",
                goal_reached_cm=SURVEY_GOAL_REACHED_CM,
            )
            if run_state.aborted:
                print("Drive to random point aborted; stopping calibration run.")
                return
            if not hold_with_ui_updates(
                sock,
                viewer=run_state.viewer,
                planner=run_state.planner,
                goal_xy=run_state.goal_xy,
                obstacle_total=run_state.obstacle_total,
                start_time=run_state.start_time,
                total_traveled_cm=run_state.total_traveled_cm,
                duration_s=PING_SETTLE_SEC,
                status=f"Settling at random point loop {loop_idx}...",
                telemetry_callback=random_logger,
                debug_logger=None,
                debug_mode=f"ping_calibration_hold_random_{loop_idx}",
            ):
                print("Hold at random point aborted; stopping calibration run.")
                return
            log_stop_sample(
                sample_kind="stop_at_random",
                sample_idx=current_sample_idx,
                planned_x_m=random_goal_x_m,
                planned_y_m=random_goal_y_m,
                goal_xy=random_goal_xy,
            )
    finally:
        stop_rover(sock)
        close_rover_socket(sock)
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
