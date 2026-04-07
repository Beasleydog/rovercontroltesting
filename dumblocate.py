from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from dumbdrive import (
    FrontendTimingLogger,
    drive_to_goal,
    hold_with_ui_updates,
    make_sanitized_telemetry,
)
from main import (
    POSE_OFFSET_X_CM,
    POSE_OFFSET_Y_CM,
    POSE_UNITS_TO_CM,
    MapWindow,
    parse_pose,
    stop_rover,
)
from rover_control import (
    close_rover_socket,
    extract_json_bytes,
    fetch_rover_json,
    open_rover_socket,
    send_float_command,
    send_get_command,
    set_brakes,
    set_lights,
    set_steering,
    set_throttle,
    wait_for_dust,
)


GET_LTV_JSON = 2
CMD_LTV_PING = 2050
CMD_LTV_PING_UNLIMITED = 2051
USE_UNLIMITED_PING = False
PING_SETTLE_SEC = 1.2
PING_LOG_INTERVAL_SEC = 1.0
PING_RESPONSE_TIMEOUT_SEC = 1.5
PING_RESPONSE_POLL_SEC = 0.05
PING_TRIANGLE_RADIUS_M = 50.0
SECOND_TRILOCATION_RADIUS_M = 5.0
SECOND_TRILOCATION_STRONG_PING_THRESHOLD = -0.75
MAX_DRIVE_SEGMENT_CM = 4000.0
LAST_KNOWN_GOAL_REACHED_CM = 1000.0
PING_MOVE_GOAL_REACHED_CM = 1000.0
FINAL_ESTIMATE_GOAL_REACHED_CM = 100.0
STOP_AT_LAST_KNOWN_ONLY = False
EFFICIENCY_MODE = True
REAL_LTV_LOCATION_M = (-6047.30, -10769.3, 1463.0)


@dataclass(slots=True)
class PingSample:
    rover_x_m: float
    rover_y_m: float
    ping_value: float
    radius_m: float


class LocateMetricsLogger:
    def __init__(self) -> None:
        root = Path("runs")
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"dumblocate_metrics_{time.strftime('%Y%m%d_%H%M%S')}.tsv"
        self._file = self.path.open("w", encoding="utf-8", newline="")
        self._file.write(
            "iso_time_utc\tmono_s\tphase\trover_x_m\trover_y_m\trover_z_m\t"
            "dist_to_real_ltv_m\tping_strength\tgoal_dist_cm\n"
        )
        self._file.flush()
        print(f"Locate metrics log: {self.path}")

    def log(self, *, phase: str, rover_x_m: float, rover_y_m: float, rover_z_m: float, ping_strength: float, goal_dist_cm: float) -> None:
        dx = rover_x_m - REAL_LTV_LOCATION_M[0]
        dy = rover_y_m - REAL_LTV_LOCATION_M[1]
        dz = rover_z_m - REAL_LTV_LOCATION_M[2]
        dist_m = math.sqrt(dx * dx + dy * dy + dz * dz)
        self._file.write(
            "\t".join(
                [
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    f"{time.monotonic():.3f}",
                    phase,
                    f"{rover_x_m:.3f}",
                    f"{rover_y_m:.3f}",
                    f"{rover_z_m:.3f}",
                    f"{dist_m:.3f}",
                    f"{ping_strength:.3f}",
                    f"{goal_dist_cm:.3f}",
                ]
            )
            + "\n"
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class PingStrengthSampler:
    def __init__(self, interval_sec: float) -> None:
        self.interval_sec = float(interval_sec)
        self.last_sample_monotonic: float | None = None
        self.last_strength: float = float("nan")
        self.enabled = True

    def sample(self, sock) -> tuple[float, bool]:
        now = time.monotonic()
        if not self.enabled:
            return (self.last_strength, False)
        if (
            self.last_sample_monotonic is None
            or (now - self.last_sample_monotonic) >= self.interval_sec
        ):
            self.last_strength = current_ping_strength(sock)
            self.last_sample_monotonic = now
            return (self.last_strength, True)
        return (self.last_strength, False)


def ltv_ping_to_meters(ping_value: float) -> float:
    return 14.034819 * math.exp(-0.066163 * float(ping_value))


def fetch_ltv_json(sock) -> dict:
    response = send_get_command(sock, GET_LTV_JSON)
    return json.loads(extract_json_bytes(response).decode("utf-8"))


def read_ltv_signal_strength(sock) -> float:
    ltv_json = fetch_ltv_json(sock)
    signal = ltv_json.get("signal", {})
    return float(signal.get("strength", 0.0))


def raw_world_m_to_local_cm(x_m: float, y_m: float) -> tuple[float, float]:
    x_cm = float(x_m) * POSE_UNITS_TO_CM - POSE_OFFSET_X_CM
    y_cm = float(y_m) * POSE_UNITS_TO_CM - POSE_OFFSET_Y_CM
    return (x_cm, y_cm)


def fetch_pose_and_telemetry(sock) -> tuple[float, float, float, float, dict]:
    rover_json = fetch_rover_json(sock)
    raw_telemetry = rover_json.get("pr_telemetry")
    if not isinstance(raw_telemetry, dict):
        raise RuntimeError("ROVER.json did not contain pr_telemetry")
    telemetry = make_sanitized_telemetry(raw_telemetry)
    x_cm, y_cm, z_cm, heading_deg = parse_pose(telemetry)
    return (x_cm, y_cm, z_cm, heading_deg, raw_telemetry)


def request_ping_and_read_strength(sock) -> float:
    ping_command = CMD_LTV_PING_UNLIMITED if USE_UNLIMITED_PING else CMD_LTV_PING
    previous_strength = read_ltv_signal_strength(sock)
    send_float_command(sock, ping_command, 1.0)
    deadline = time.monotonic() + PING_RESPONSE_TIMEOUT_SEC
    latest_strength = previous_strength
    while time.monotonic() < deadline:
        time.sleep(PING_RESPONSE_POLL_SEC)
        latest_strength = read_ltv_signal_strength(sock)
        if latest_strength != previous_strength:
            return latest_strength
    # print(
    #     f"Warning: ping command {ping_command} did not change LTV signal strength "
    #     f"within {PING_RESPONSE_TIMEOUT_SEC:.2f}s; keeping {latest_strength:.3f}"
    # )
    return latest_strength


def current_ping_strength(sock) -> float:
    return request_ping_and_read_strength(sock)


def sample_ping(sock, raw_telemetry: dict) -> PingSample:
    ping_value = request_ping_and_read_strength(sock)
    radius_m = ltv_ping_to_meters(ping_value)
    rover_x_m = float(raw_telemetry.get("rover_pos_x", 0.0))
    rover_y_m = float(raw_telemetry.get("rover_pos_y", 0.0))
    sample = PingSample(
        rover_x_m=rover_x_m,
        rover_y_m=rover_y_m,
        ping_value=ping_value,
        radius_m=radius_m,
    )
    print(
        f"Ping sample: rover=({sample.rover_x_m:.3f}, {sample.rover_y_m:.3f}) m | "
        f"ping={sample.ping_value:.3f} | radius={sample.radius_m:.3f} m"
    )
    return sample


def trilaterate(
    samples: tuple[PingSample, PingSample, PingSample],
) -> tuple[float, float]:
    s1, s2, s3 = samples
    a11 = 2.0 * (s2.rover_x_m - s1.rover_x_m)
    a12 = 2.0 * (s2.rover_y_m - s1.rover_y_m)
    a21 = 2.0 * (s3.rover_x_m - s1.rover_x_m)
    a22 = 2.0 * (s3.rover_y_m - s1.rover_y_m)
    b1 = (
        s1.radius_m * s1.radius_m
        - s2.radius_m * s2.radius_m
        - s1.rover_x_m * s1.rover_x_m
        + s2.rover_x_m * s2.rover_x_m
        - s1.rover_y_m * s1.rover_y_m
        + s2.rover_y_m * s2.rover_y_m
    )
    b2 = (
        s1.radius_m * s1.radius_m
        - s3.radius_m * s3.radius_m
        - s1.rover_x_m * s1.rover_x_m
        + s3.rover_x_m * s3.rover_x_m
        - s1.rover_y_m * s1.rover_y_m
        + s3.rover_y_m * s3.rover_y_m
    )
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-9:
        raise RuntimeError("Ping geometry degenerate; cannot trilaterate")
    x_m = (b1 * a22 - b2 * a12) / det
    y_m = (a11 * b2 - a21 * b1) / det
    return (x_m, y_m)


def local_cm_to_raw_world_m(x_cm: float, y_cm: float) -> tuple[float, float]:
    x_m = (float(x_cm) + POSE_OFFSET_X_CM) / POSE_UNITS_TO_CM
    y_m = (float(y_cm) + POSE_OFFSET_Y_CM) / POSE_UNITS_TO_CM
    return (x_m, y_m)


def triangle_sample_points_cm(
    center_xy: tuple[float, float],
    radius_m: float,
) -> list[tuple[float, float]]:
    radius_cm = float(radius_m) * 100.0
    points: list[tuple[float, float]] = []
    for angle_deg in (90.0, 210.0, 330.0):
        angle_rad = math.radians(angle_deg)
        points.append(
            (
                center_xy[0] + radius_cm * math.cos(angle_rad),
                center_xy[1] + radius_cm * math.sin(angle_rad),
            )
        )
    return points


def phase_label(phase: str) -> str:
    labels = {
        "to_last_known": "Drive: last known",
        "to_ping_2": "Move for ping 2",
        "to_ping_3": "Move for ping 3",
        "to_estimated": "Drive: trilaterated LTV",
        "done": "Done",
    }
    return labels.get(phase, phase)


def step_toward_goal_cm(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    max_step_cm: float,
) -> tuple[float, float]:
    dx = float(goal_xy[0] - start_xy[0])
    dy = float(goal_xy[1] - start_xy[1])
    dist = math.hypot(dx, dy)
    if dist <= max_step_cm or dist <= 1e-6:
        return goal_xy
    scale = max_step_cm / dist
    return (start_xy[0] + dx * scale, start_xy[1] + dy * scale)


def drive_to_goal_segmented(
    sock,
    *,
    final_goal_xy: tuple[float, float],
    goal_label: str | None,
    viewer,
    recorded_obstacle_points: list[tuple[float, float]],
    obstacle_total: int,
    start_time: float | None,
    step_idx: int,
    total_traveled_cm: float,
    goals_reached: int,
    telemetry_callback=None,
    debug_logger: FrontendTimingLogger | None,
    debug_mode: str,
    goal_reached_cm: float,
):
    run_state = None
    while True:
        if run_state is None:
            start_x, start_y, _z, _heading, _raw = fetch_pose_and_telemetry(sock)
            current_xy = (start_x, start_y)
        else:
            current_xy = (run_state.pose_xyzh[0], run_state.pose_xyzh[1])

        segment_goal_xy = step_toward_goal_cm(
            current_xy,
            final_goal_xy,
            MAX_DRIVE_SEGMENT_CM,
        )
        run_state = drive_to_goal(
            sock,
            goal_xy=segment_goal_xy,
            display_goal_xy=final_goal_xy,
            goal_label=goal_label,
            viewer=viewer,
            recorded_obstacle_points=recorded_obstacle_points,
            obstacle_total=obstacle_total,
            start_time=start_time,
            step_idx=step_idx,
            total_traveled_cm=total_traveled_cm,
            goals_reached=goals_reached,
            goal_reached_cm=goal_reached_cm,
            telemetry_callback=telemetry_callback,
            debug_logger=debug_logger,
            debug_mode=debug_mode,
        )
        if run_state.aborted:
            return run_state
        rover_x, rover_y, _z, _heading = run_state.pose_xyzh
        remaining_cm = math.hypot(final_goal_xy[0] - rover_x, final_goal_xy[1] - rover_y)
        if remaining_cm <= MAX_DRIVE_SEGMENT_CM:
            if segment_goal_xy == final_goal_xy:
                return run_state
            viewer = run_state.viewer
            recorded_obstacle_points = run_state.recorded_obstacle_points
            obstacle_total = run_state.obstacle_total
            start_time = run_state.start_time
            step_idx = run_state.step_idx
            total_traveled_cm = run_state.total_traveled_cm
            continue
        viewer = run_state.viewer
        recorded_obstacle_points = run_state.recorded_obstacle_points
        obstacle_total = run_state.obstacle_total
        start_time = run_state.start_time
        step_idx = run_state.step_idx
        total_traveled_cm = run_state.total_traveled_cm


def drive_to_goal_locate(
    sock,
    *,
    final_goal_xy: tuple[float, float],
    goal_label: str | None,
    viewer,
    recorded_obstacle_points: list[tuple[float, float]],
    obstacle_total: int,
    start_time: float | None,
    step_idx: int,
    total_traveled_cm: float,
    goals_reached: int,
    telemetry_callback=None,
    debug_logger: FrontendTimingLogger | None,
    debug_mode: str,
    goal_reached_cm: float,
):
    if EFFICIENCY_MODE:
        return drive_to_goal(
            sock,
            goal_xy=final_goal_xy,
            display_goal_xy=final_goal_xy,
            goal_label=goal_label,
            viewer=viewer,
            frontend_enabled=False,
            recorded_obstacle_points=recorded_obstacle_points,
            obstacle_total=obstacle_total,
            start_time=start_time,
            step_idx=step_idx,
            total_traveled_cm=total_traveled_cm,
            goals_reached=goals_reached,
            goal_reached_cm=goal_reached_cm,
            replan_on_edge=False,
            replan_only_on_obstacle=True,
            telemetry_callback=telemetry_callback,
            debug_logger=debug_logger,
            debug_mode=debug_mode,
        )
    return drive_to_goal_segmented(
        sock,
        final_goal_xy=final_goal_xy,
        goal_label=goal_label,
        viewer=viewer,
        recorded_obstacle_points=recorded_obstacle_points,
        obstacle_total=obstacle_total,
        start_time=start_time,
        step_idx=step_idx,
        total_traveled_cm=total_traveled_cm,
        goals_reached=goals_reached,
        telemetry_callback=telemetry_callback,
        debug_logger=debug_logger,
        debug_mode=debug_mode,
        goal_reached_cm=goal_reached_cm,
    )


def collect_triangle_ping_samples(
    sock,
    *,
    center_xy: tuple[float, float],
    radius_m: float,
    run_state,
    viewer,
    telemetry_callback,
    debug_logger: FrontendTimingLogger | None,
    debug_prefix: str,
) -> tuple[list[PingSample], object, MapWindow | None, bool]:
    samples: list[PingSample] = []
    triangle_points = triangle_sample_points_cm(center_xy, radius_m)
    for ping_idx, triangle_goal_xy in enumerate(triangle_points, start=1):
        sample_x_m, sample_y_m = local_cm_to_raw_world_m(*triangle_goal_xy)
        print(
            f"Driving to {debug_prefix} ping {ping_idx} triangle point: "
            f"({sample_x_m:.3f}, {sample_y_m:.3f}) m"
        )
        run_state = drive_to_goal_locate(
            sock,
            final_goal_xy=triangle_goal_xy,
            goal_label=f"{sample_x_m:.3f}, {sample_y_m:.3f}",
            viewer=viewer,
            recorded_obstacle_points=run_state.recorded_obstacle_points,
            obstacle_total=run_state.obstacle_total,
            start_time=run_state.start_time,
            step_idx=run_state.step_idx,
            total_traveled_cm=run_state.total_traveled_cm,
            goals_reached=0,
            goal_reached_cm=PING_MOVE_GOAL_REACHED_CM,
            telemetry_callback=telemetry_callback,
            debug_logger=debug_logger,
            debug_mode=f"{debug_prefix}_drive_ping_{ping_idx}",
        )
        viewer = run_state.viewer
        if run_state.aborted:
            return (samples, run_state, viewer, False)

        print(f"Collecting {debug_prefix} ping {ping_idx}...")
        if not hold_with_ui_updates(
            sock,
            viewer=viewer,
            planner=run_state.planner,
            goal_xy=run_state.goal_xy,
            obstacle_total=run_state.obstacle_total,
            start_time=run_state.start_time,
            total_traveled_cm=run_state.total_traveled_cm,
            duration_s=PING_SETTLE_SEC,
            status=f"Collecting {debug_prefix} ping {ping_idx}...",
            telemetry_callback=telemetry_callback,
            debug_logger=debug_logger,
            debug_mode=f"{debug_prefix}_hold_ping_{ping_idx}",
        ):
            return (samples, run_state, viewer, False)
        samples.append(sample_ping(sock, run_state.raw_telemetry))
        if ping_idx < len(triangle_points):
            if not hold_with_ui_updates(
                sock,
                viewer=viewer,
                planner=run_state.planner,
                goal_xy=run_state.goal_xy,
                obstacle_total=run_state.obstacle_total,
                start_time=run_state.start_time,
                total_traveled_cm=run_state.total_traveled_cm,
                duration_s=0.5,
                status="Preparing next move...",
                telemetry_callback=telemetry_callback,
                debug_logger=debug_logger,
                debug_mode=f"{debug_prefix}_hold_after_ping_{ping_idx}",
            ):
                return (samples, run_state, viewer, False)
    return (samples, run_state, viewer, True)


def main() -> None:
    sock = open_rover_socket()
    viewer: MapWindow | None = None
    debug_logger: FrontendTimingLogger | None = None
    metrics_logger: LocateMetricsLogger | None = None
    ping_sampler: PingStrengthSampler | None = None
    run_start_wall = time.time()
    run_completed = False
    print(f"Dumblocate start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run_start_wall))}")
    try:
        debug_logger = FrontendTimingLogger("dumblocate")
        metrics_logger = LocateMetricsLogger()
        ping_sampler = PingStrengthSampler(PING_LOG_INTERVAL_SEC)
        if not wait_for_dust(sock, timeout_seconds=20.0, poll_seconds=0.5):
            raise RuntimeError("DUST is not connected to TSS.")
        set_lights(sock, True)

        def log_metrics(*, phase: str, raw_telemetry: dict, rover_xyzh: tuple[float, float, float, float], goal_xy: tuple[float, float], goal_distance_cm: float) -> None:
            if metrics_logger is None or ping_sampler is None:
                return
            ping_strength, sampled_now = ping_sampler.sample(sock)
            if not sampled_now:
                return
            rover_x_m = float(raw_telemetry.get("rover_pos_x", 0.0))
            rover_y_m = float(raw_telemetry.get("rover_pos_y", 0.0))
            rover_z_m = float(raw_telemetry.get("rover_pos_z", 0.0))
            metrics_logger.log(
                phase=phase,
                rover_x_m=rover_x_m,
                rover_y_m=rover_y_m,
                rover_z_m=rover_z_m,
                ping_strength=ping_strength,
                goal_dist_cm=goal_distance_cm,
            )

        x, y, z, heading, raw_telemetry = fetch_pose_and_telemetry(sock)
        ltv_json = fetch_ltv_json(sock)
        location = ltv_json.get("location", {})
        last_known_x_m = float(location.get("last_known_x", 0.0))
        last_known_y_m = float(location.get("last_known_y", 0.0))
        goal_x, goal_y = raw_world_m_to_local_cm(last_known_x_m, last_known_y_m)
        print(
            f"Last known LTV location: ({last_known_x_m:.3f}, {last_known_y_m:.3f}) m"
        )

        run_state = drive_to_goal_locate(
            sock,
            final_goal_xy=(goal_x, goal_y),
            goal_label=f"{last_known_x_m:.3f}, {last_known_y_m:.3f}",
            viewer=viewer,
            recorded_obstacle_points=[],
            obstacle_total=0,
            start_time=None,
            step_idx=0,
            total_traveled_cm=0.0,
            goals_reached=0,
            goal_reached_cm=LAST_KNOWN_GOAL_REACHED_CM,
            telemetry_callback=log_metrics,
            debug_logger=debug_logger,
            debug_mode="dumblocate_drive_last_known",
        )
        viewer = run_state.viewer
        if run_state.aborted:
            return

        if STOP_AT_LAST_KNOWN_ONLY:
            print("Reached last known LTV location. Stopping here because STOP_AT_LAST_KNOWN_ONLY is enabled.")
            return

        samples: list[PingSample] = []
        last_known_remaining_cm = math.hypot(
            goal_x - run_state.pose_xyzh[0],
            goal_y - run_state.pose_xyzh[1],
        )
        print("Arrived at last known location.")
        print(f"Distance from last known at triangle start: {last_known_remaining_cm:.1f} cm")

        samples, run_state, viewer, ok = collect_triangle_ping_samples(
            sock,
            center_xy=(goal_x, goal_y),
            radius_m=PING_TRIANGLE_RADIUS_M,
            run_state=run_state,
            viewer=viewer,
            telemetry_callback=log_metrics,
            debug_logger=debug_logger,
            debug_prefix="dumblocate_round1",
        )
        if not ok or run_state.aborted:
            return
        est_x_m, est_y_m = trilaterate((samples[0], samples[1], samples[2]))
        print(f"Trilaterated LTV estimate: ({est_x_m:.3f}, {est_y_m:.3f}) m")

        goal_x, goal_y = raw_world_m_to_local_cm(est_x_m, est_y_m)
        if not hold_with_ui_updates(
            sock,
            viewer=viewer,
            planner=run_state.planner,
            goal_xy=(goal_x, goal_y),
            obstacle_total=run_state.obstacle_total,
            start_time=run_state.start_time,
            total_traveled_cm=run_state.total_traveled_cm,
            duration_s=0.5,
            status="Preparing final drive...",
            telemetry_callback=log_metrics,
            debug_logger=debug_logger,
            debug_mode="dumblocate_hold_before_final_drive",
        ):
            return
        run_state = drive_to_goal_locate(
            sock,
            final_goal_xy=(goal_x, goal_y),
            goal_label=f"{est_x_m:.3f}, {est_y_m:.3f}",
            viewer=viewer,
            recorded_obstacle_points=run_state.recorded_obstacle_points,
            obstacle_total=run_state.obstacle_total,
            start_time=run_state.start_time,
            step_idx=run_state.step_idx,
            total_traveled_cm=run_state.total_traveled_cm,
            goals_reached=0,
            goal_reached_cm=FINAL_ESTIMATE_GOAL_REACHED_CM,
            telemetry_callback=log_metrics,
            debug_logger=debug_logger,
            debug_mode="dumblocate_drive_final",
        )
        viewer = run_state.viewer
        if run_state.aborted:
            return

        print("Arrived near trilaterated LTV location.")
        if not hold_with_ui_updates(
            sock,
            viewer=viewer,
            planner=run_state.planner,
            goal_xy=run_state.goal_xy,
            obstacle_total=run_state.obstacle_total,
            start_time=run_state.start_time,
            total_traveled_cm=run_state.total_traveled_cm,
            duration_s=PING_SETTLE_SEC,
            status="Checking estimate ping...",
            telemetry_callback=log_metrics,
            debug_logger=debug_logger,
            debug_mode="dumblocate_hold_verify_estimate",
        ):
            return
        final_estimate_ping = sample_ping(sock, run_state.raw_telemetry)
        print(
            f"Ping at first estimate: {final_estimate_ping.ping_value:.3f} "
            f"(strong-enough threshold {SECOND_TRILOCATION_STRONG_PING_THRESHOLD:.3f})"
        )
        if final_estimate_ping.ping_value < SECOND_TRILOCATION_STRONG_PING_THRESHOLD:
            print("Estimate ping is still too weak. Running second trilateration round.")
            second_center_xy = raw_world_m_to_local_cm(est_x_m, est_y_m)
            second_samples, run_state, viewer, ok = collect_triangle_ping_samples(
                sock,
                center_xy=second_center_xy,
                radius_m=SECOND_TRILOCATION_RADIUS_M,
                run_state=run_state,
                viewer=viewer,
                telemetry_callback=log_metrics,
                debug_logger=debug_logger,
                debug_prefix="dumblocate_round2",
            )
            if not ok or run_state.aborted:
                return
            est_x_m, est_y_m = trilaterate(
                (second_samples[0], second_samples[1], second_samples[2])
            )
            print(f"Second-round trilaterated LTV estimate: ({est_x_m:.3f}, {est_y_m:.3f}) m")
            goal_x, goal_y = raw_world_m_to_local_cm(est_x_m, est_y_m)
            if not hold_with_ui_updates(
                sock,
                viewer=viewer,
                planner=run_state.planner,
                goal_xy=(goal_x, goal_y),
                obstacle_total=run_state.obstacle_total,
                start_time=run_state.start_time,
                total_traveled_cm=run_state.total_traveled_cm,
                duration_s=0.5,
                status="Preparing second final drive...",
                telemetry_callback=log_metrics,
                debug_logger=debug_logger,
                debug_mode="dumblocate_hold_before_second_final_drive",
            ):
                return
            run_state = drive_to_goal_locate(
                sock,
                final_goal_xy=(goal_x, goal_y),
                goal_label=f"{est_x_m:.3f}, {est_y_m:.3f}",
                viewer=viewer,
                recorded_obstacle_points=run_state.recorded_obstacle_points,
                obstacle_total=run_state.obstacle_total,
                start_time=run_state.start_time,
                step_idx=run_state.step_idx,
                total_traveled_cm=run_state.total_traveled_cm,
                goals_reached=0,
                goal_reached_cm=FINAL_ESTIMATE_GOAL_REACHED_CM,
                telemetry_callback=log_metrics,
                debug_logger=debug_logger,
                debug_mode="dumblocate_drive_second_final",
            )
            viewer = run_state.viewer
            if run_state.aborted:
                return
            print("Arrived near second-round trilaterated LTV location.")
        else:
            print("Estimate ping is strong enough. Skipping second trilateration round.")
        run_completed = True

    except KeyboardInterrupt:
        pass
    finally:
        run_end_wall = time.time()
        elapsed_sec = run_end_wall - run_start_wall
        print(f"Dumblocate end: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run_end_wall))}")
        print(
            f"Dumblocate elapsed: {elapsed_sec:.1f}s "
            f"({elapsed_sec / 60.0:.2f} min){' [completed]' if run_completed else ' [stopped early]'}"
        )
        stop_rover(sock)
       
        close_rover_socket(sock)
        if viewer is not None:
            viewer.close()
        if debug_logger is not None:
            debug_logger.close()
        if metrics_logger is not None:
            metrics_logger.close()


if __name__ == "__main__":
    main()
