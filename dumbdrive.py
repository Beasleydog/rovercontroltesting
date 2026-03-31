from __future__ import annotations

import csv
import json
import math
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from html_ui import HtmlCanvasWindow
from main import (
    CONTROL_PERIOD_SEC,
    GOAL_REACHED_CM,
    GRID_CELL_SIZE_CM,
    LIDAR_SENSOR_LAYOUT,
    LIDAR_SENSOR_LABELS,
    PATH_TARGET_LOOKAHEAD_CM,
    POSE_UNITS_TO_CM,
    ROVER_HALF_LENGTH_CM,
    ROVER_HALF_WIDTH_CM,
    TARGET_DX_CM,
    TARGET_DY_CM,
    CELL_OBSTACLE,
    MapWindow,
    choose_drive_command,
    create_planner,
    distance_cm,
    local_to_world_2d,
    parse_lidar,
    parse_pose,
    plan_path_for_following,
    stop_rover,
    select_local_path_target,
)
from rover_control import (
    close_rover_socket,
    fetch_rover_json,
    open_rover_socket,
    sanitize_lidar_scan,
    set_brakes,
    set_steering,
    set_throttle,
    wait_for_dust,
)


GOOD_UI = True
# FRONTEND_REPLAY_LOG_PATH: str | None = "C:/Users/beasl/.stuff/school/claws/rovercontroltesting/runs/dumbdrive_debug_20260315_232929/frontend_state.jsonl"
FRONTEND_REPLAY_LOG_PATH: str | None = None
DUMBDRIVE_GOAL_REACHED_CM = 350.0
STATIONARY_TIMEOUT_SEC = 5
STARTUP_STUCK_GRACE_SEC = 5.0
STATIONARY_MOVE_THRESHOLD_CM = 25.0
STATIONARY_SPEED_THRESHOLD = 0.25
STUCK_REARM_EXIT_DISTANCE_CM = 45.0
NORMAL_DRIVE_THROTTLE = 40.0
MIN_FORWARD_DRIVE_THROTTLE = 20.0
ENABLE_REVERSE_TO_TARGET = False
REVERSE_TO_TARGET_WINDOW_DEG = 20.0
RECOVERY_REVERSE_THROTTLE = -85.0
RECOVERY_REVERSE_SECONDS = 8
RECOVERY_BRAKE_SECONDS = 5
RECOVERY_REVERSE_STEER_GAIN = 0.35
STUCK_OBSTACLE_FORWARD_OFFSET_CM = ROVER_HALF_LENGTH_CM
STUCK_OBSTACLE_BUMPER_ROW_SAMPLES = 7
STUCK_HISTORY_FRAMES = 150
STUCK_HISTORY_RING_MIN_CM = 0.0
STUCK_HISTORY_RING_MAX_CM = 100.0
PLANNER_EDGE_REBUILD_MARGIN_CELLS = 8
RANDOM_GOAL_RANGE_CM = 4000.0
CLEAR_KNOWN_OBSTACLES_ON_GOAL_REACHED = True
CLEAN_LOG_ROOT = Path("cleanlog")
RESET_SCRIPT_PATH = Path(__file__).with_name("reset.py")
RECOVERY_RESET_STALL_TIMEOUT_SEC = 60.0
RECOVERY_RESET_ESCAPE_RADIUS_CM = 150.0
RAW_TSS_TELEMETRY_FIELDS = [
    "rover_pos_x",
    "rover_pos_y",
    "rover_pos_z",
    "heading",
    "pitch",
    "roll",
]


def make_sanitized_telemetry(raw_telemetry: dict) -> dict:
    telemetry = dict(raw_telemetry)
    sanitized_lidar = sanitize_lidar_scan(raw_telemetry.get("lidar"))
    if sanitized_lidar is not None:
        telemetry["lidar"] = sanitized_lidar
    return telemetry


class CleanGoalLogWriter:
    def __init__(self) -> None:
        self.root_dir = CLEAN_LOG_ROOT
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.session_timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.goal_idx = 0
        self.active_file = None
        self.active_path: Path | None = None
        self.active_goal_xy: tuple[float, float] | None = None
        self.active_goal_reason = ""
        self.header = self._build_header()

    @staticmethod
    def _fmt(value: float | int | bool | str) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if math.isnan(value):
                return "nan"
            if math.isinf(value):
                return "inf" if value > 0 else "-inf"
            return f"{value:.3f}"
        return str(value)

    @staticmethod
    def _fmt_raw(value) -> str:
        return json.dumps(value, separators=(",", ":"))

    def _build_header(self) -> list[str]:
        base_columns = [
            "iso_time_utc",
            "elapsed_s",
            "step_idx",
        ]
        raw_telemetry_columns = list(RAW_TSS_TELEMETRY_FIELDS)
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
        return base_columns + raw_telemetry_columns + lidar_columns

    def start_goal(self, goal_x_cm: float, goal_y_cm: float, reason: str) -> None:
        self.close_active_file()
        self.goal_idx += 1
        self.active_goal_xy = (goal_x_cm, goal_y_cm)
        self.active_goal_reason = reason
        self.active_path = self.root_dir / f"dumbdrive_goal_{self.session_timestamp}_{self.goal_idx:03d}.csv"
        self.active_file = self.active_path.open("w", encoding="utf-8", newline="")
        csv.writer(self.active_file).writerow(self.header)
        self.active_file.flush()

    def log_lidar_update(
        self,
        *,
        elapsed_s: float,
        step_idx: int,
        raw_telemetry: dict,
    ) -> None:
        if self.active_file is None:
            raise RuntimeError("Clean goal log file is not open.")
        row = [
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            self._fmt(elapsed_s),
            self._fmt(step_idx),
        ]
        row.extend(self._fmt_raw(raw_telemetry.get(field)) for field in RAW_TSS_TELEMETRY_FIELDS)
        raw_lidar = raw_telemetry.get("lidar", [])
        if isinstance(raw_lidar, list):
            row.extend(self._fmt_raw(value) for value in raw_lidar[: len(LIDAR_SENSOR_LAYOUT)])
            if len(raw_lidar) < len(LIDAR_SENSOR_LAYOUT):
                row.extend("" for _ in range(len(LIDAR_SENSOR_LAYOUT) - len(raw_lidar)))
        else:
            row.extend("" for _ in range(len(LIDAR_SENSOR_LAYOUT)))
        csv.writer(self.active_file).writerow(row)
        self.active_file.flush()

    def close_active_file(self) -> None:
        if self.active_file is not None:
            self.active_file.close()
            self.active_file = None

    def close(self) -> None:
        self.close_active_file()


def build_status(
    recovering: bool,
    obstacle_added: bool,
    path_len: int,
    goals_reached: int,
) -> str:
    if recovering:
        base_status = "Recovering: reverse"
    elif obstacle_added:
        base_status = f"Stuck: marked obstacle + replanned ({path_len} pts)"
    elif path_len <= 0:
        base_status = "No path"
    else:
        base_status = "Running"
    return f"{base_status} | Goals: {goals_reached}"


def forward_edge_samples() -> list[tuple[float, float, int, float]]:
    row_offsets = np.linspace(-ROVER_HALF_WIDTH_CM, ROVER_HALF_WIDTH_CM, STUCK_OBSTACLE_BUMPER_ROW_SAMPLES)
    return [
        (STUCK_OBSTACLE_FORWARD_OFFSET_CM, float(lateral_offset_cm), sample_idx, float(lateral_offset_cm))
        for sample_idx, lateral_offset_cm in enumerate(row_offsets)
    ]


def mark_stuck_obstacles_from_history(
    planner,
    pose_history: deque[tuple[float, float, float, float]],
    stuck_pose_xyzh: tuple[float, float, float, float],
    debug_rows: list[dict[str, float | int | bool | str]] | None = None,
) -> int:
    new_obstacles = 0
    stuck_x, stuck_y, stuck_z, _ = stuck_pose_xyzh
    for hist_x, hist_y, hist_z, hist_heading in pose_history:
        dist_from_stuck_cm = math.sqrt(
            (hist_x - stuck_x) ** 2
            + (hist_y - stuck_y) ** 2
            + (hist_z - stuck_z) ** 2
        )
        if not (STUCK_HISTORY_RING_MIN_CM <= dist_from_stuck_cm <= STUCK_HISTORY_RING_MAX_CM):
            continue
        for local_x_cm, local_y_cm, sample_idx, sample_offset_cm in forward_edge_samples():
            obstacle_x, obstacle_y = local_to_world_2d(
                hist_x,
                hist_y,
                hist_heading,
                local_x_cm,
                local_y_cm,
            )
            obstacle_cell_x, obstacle_cell_y = planner.world_to_cell(obstacle_x, obstacle_y)
            placed = planner.mark_obstacle_world(obstacle_x, obstacle_y, cell_value=CELL_OBSTACLE)
            if debug_rows is not None:
                debug_rows.append(
                    {
                        "stuck_x_cm": stuck_x,
                        "stuck_y_cm": stuck_y,
                        "stuck_z_cm": stuck_z,
                        "hist_x_cm": hist_x,
                        "hist_y_cm": hist_y,
                        "hist_z_cm": hist_z,
                        "hist_heading_deg": hist_heading,
                        "dist_from_stuck_cm": dist_from_stuck_cm,
                        "contact_side": "front",
                        "contact_min_lidar_cm": float("nan"),
                        "bumper_sample_idx": sample_idx,
                        "bumper_lateral_offset_cm": sample_offset_cm,
                        "obstacle_x_cm": obstacle_x,
                        "obstacle_y_cm": obstacle_y,
                        "obstacle_cell_x": obstacle_cell_x,
                        "obstacle_cell_y": obstacle_cell_y,
                        "placed": placed,
                    }
                )
            if placed:
                new_obstacles += 1
    return new_obstacles


def compute_recovery_reverse_steering(
    rover_x_cm: float,
    rover_y_cm: float,
    rover_heading_deg: float,
    target_x_cm: float,
    target_y_cm: float,
) -> float:
    _, forward_steering, _, _ = choose_drive_command(
        rover_x_cm,
        rover_y_cm,
        rover_heading_deg,
        target_x_cm,
        target_y_cm,
    )
    return max(-1.0, min(1.0, -forward_steering * RECOVERY_REVERSE_STEER_GAIN))


def trigger_dust_reset() -> None:
    subprocess.run([sys.executable, str(RESET_SCRIPT_PATH)], check=True)


def should_reverse_to_target(heading_error_deg: float) -> bool:
    if not ENABLE_REVERSE_TO_TARGET:
        return False
    return abs(abs(heading_error_deg) - 180.0) <= REVERSE_TO_TARGET_WINDOW_DEG


def sample_random_goal_xy(
    origin_x_cm: float,
    origin_y_cm: float,
    current_x_cm: float,
    current_y_cm: float,
) -> tuple[float, float]:
    for _ in range(64):
        goal_x_cm = origin_x_cm + random.uniform(-RANDOM_GOAL_RANGE_CM, RANDOM_GOAL_RANGE_CM)
        goal_y_cm = origin_y_cm + random.uniform(-RANDOM_GOAL_RANGE_CM, RANDOM_GOAL_RANGE_CM)
        if distance_cm(current_x_cm, current_y_cm, goal_x_cm, goal_y_cm) > DUMBDRIVE_GOAL_REACHED_CM * 2.0:
            return (goal_x_cm, goal_y_cm)
    return (origin_x_cm + RANDOM_GOAL_RANGE_CM, origin_y_cm + RANDOM_GOAL_RANGE_CM)


def rebuild_planner_with_obstacles(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    obstacle_points_world: list[tuple[float, float]],
    planner_updates_world: list[tuple[str, float, float, float]] | None = None,
):
    planner = create_planner(start_xy, goal_xy)
    for obstacle_x, obstacle_y in obstacle_points_world:
        planner.mark_obstacle_world(obstacle_x, obstacle_y, cell_value=CELL_OBSTACLE)
    for update_kind, update_x, update_y, update_delta in planner_updates_world or []:
        if update_kind == "clearance":
            planner.add_clearance_world(update_x, update_y, update_delta)
        else:
            planner.add_evidence_world(update_x, update_y, update_delta)
    return planner


def compute_live_follow_path(
    planner,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
) -> list[tuple[float, float]]:
    _, follow_path = plan_path_for_following(planner, start_xy, goal_xy)
    return follow_path


def choose_goal_with_path(
    origin_xy: tuple[float, float],
    start_xy: tuple[float, float],
    recorded_obstacle_points: list[tuple[float, float]],
    preferred_goal_xy: tuple[float, float] | None = None,
    max_attempts: int = 24,
):
    candidate_goals: list[tuple[float, float]] = []
    if preferred_goal_xy is not None:
        candidate_goals.append(preferred_goal_xy)
    for _ in range(max_attempts - len(candidate_goals)):
        candidate_goals.append(
            sample_random_goal_xy(
                origin_xy[0],
                origin_xy[1],
                start_xy[0],
                start_xy[1],
            )
        )

    last_planner = None
    for goal_xy in candidate_goals:
        planner = rebuild_planner_with_obstacles(start_xy, goal_xy, recorded_obstacle_points)
        path = compute_live_follow_path(planner, start_xy, goal_xy)
        last_planner = planner
        if path:
            return (goal_xy, planner, path)

    fallback_goal_xy = candidate_goals[-1]
    return (fallback_goal_xy, last_planner, [])


def planner_needs_rebuild(planner, rover_cell: tuple[int, int]) -> bool:
    if not planner.in_bounds(rover_cell):
        return True
    cx, cy = rover_cell
    return (
        cx < PLANNER_EDGE_REBUILD_MARGIN_CELLS
        or cy < PLANNER_EDGE_REBUILD_MARGIN_CELLS
        or cx >= planner.width_cells - PLANNER_EDGE_REBUILD_MARGIN_CELLS
        or cy >= planner.height_cells - PLANNER_EDGE_REBUILD_MARGIN_CELLS
    )


def replay_frontend_log(log_path: Path) -> None:
    if not log_path.exists():
        raise FileNotFoundError(f"Frontend replay log not found: {log_path}")

    frames: list[dict[str, object]] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frames.append(json.loads(line))
    if not frames:
        raise RuntimeError(f"Frontend replay log is empty: {log_path}")

    dummy_planner = SimpleNamespace(config=SimpleNamespace(cell_size_cm=GRID_CELL_SIZE_CM))
    viewer = HtmlCanvasWindow(dummy_planner)
    try:
        replay_start = time.monotonic()
        first_elapsed = float(frames[0].get("elapsed_s", 0.0))
        for frame in frames:
            frame_elapsed = float(frame.get("elapsed_s", 0.0)) - first_elapsed
            delay = frame_elapsed - (time.monotonic() - replay_start)
            if delay > 0.0:
                time.sleep(delay)
            viewer.set_state_snapshot(dict(frame["state"]))
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


def main() -> None:
    if FRONTEND_REPLAY_LOG_PATH:
        replay_frontend_log(Path(FRONTEND_REPLAY_LOG_PATH))
        return

    sock = open_rover_socket()
    viewer: MapWindow | None = None
    clean_log_writer: CleanGoalLogWriter | None = None

    try:
        if not wait_for_dust(sock, timeout_seconds=20.0, poll_seconds=0.5):
            raise RuntimeError("DUST is not connected to TSS.")

        rover_json = fetch_rover_json(sock)
        raw_telemetry = rover_json.get("pr_telemetry")
        if not isinstance(raw_telemetry, dict):
            raise RuntimeError("ROVER.json did not contain pr_telemetry")
        telemetry = make_sanitized_telemetry(raw_telemetry)
        initial_x, initial_y, _, _ = parse_pose(telemetry)
        origin_x = initial_x
        origin_y = initial_y
        goal_x = initial_x + TARGET_DX_CM
        goal_y = initial_y + TARGET_DY_CM

        start_x, start_y, _, heading = parse_pose(telemetry)
        (goal_x, goal_y), planner, path = choose_goal_with_path(
            origin_xy=(origin_x, origin_y),
            start_xy=(start_x, start_y),
            recorded_obstacle_points=[],
            preferred_goal_xy=(goal_x, goal_y),
        )
        if planner is None:
            raise RuntimeError("Failed to create planner for dumbdrive.")
        viewer = HtmlCanvasWindow(planner) if GOOD_UI else MapWindow(planner)
        clean_log_writer = CleanGoalLogWriter()
        clean_log_writer.start_goal(goal_x, goal_y, reason="initial_goal")
        print(f"Dumbdrive clean logs: {clean_log_writer.root_dir}")

        obstacle_total = 0
        recorded_obstacle_points: list[tuple[float, float]] = []
        step_idx = 0
        start_time = time.monotonic()
        stationary_anchor_xy = (start_x, start_y)
        stationary_anchor_time = time.monotonic()
        pose_history: deque[tuple[float, float, float, float]] = deque(maxlen=STUCK_HISTORY_FRAMES)
        pose_history.append((start_x, start_y, 0.0, heading))
        reverse_until: float | None = None
        stuck_detection_armed = False
        stuck_rearm_pending = False
        stuck_rearm_origin_xy: tuple[float, float] | None = None
        recovery_reset_watch_started_at: float | None = None
        recovery_reset_watch_origin_xy: tuple[float, float] | None = None
        recovery_reset_triggered = False
        goals_reached = 0
        status = build_status(recovering=False, obstacle_added=False, path_len=len(path), goals_reached=goals_reached)
        target_x = goal_x
        target_y = goal_y
        waypoint_idx = 0
        waypoint_distance_cm = distance_cm(start_x, start_y, goal_x, goal_y)
        waypoint_distance_sum_cm = 0.0
        waypoint_distance_count = 0
        total_traveled_cm = 0.0

        while True:
            now = time.monotonic()
            rover_json = fetch_rover_json(sock)
            raw_telemetry = rover_json.get("pr_telemetry")
            if not isinstance(raw_telemetry, dict):
                raise RuntimeError("ROVER.json did not contain pr_telemetry")
            telemetry = make_sanitized_telemetry(raw_telemetry)
            x, y, z, heading = parse_pose(telemetry)
            lidar_cm = parse_lidar(telemetry)
            speed = float(raw_telemetry.get("speed", 0.0))
            goal_distance = distance_cm(x, y, goal_x, goal_y)
            logged_goal_x = goal_x
            logged_goal_y = goal_y
            logged_goal_distance = goal_distance
            pose_history.append((x, y, z, heading))
            if pose_history:
                prev_x, prev_y, prev_z, _ = pose_history[-2] if len(pose_history) >= 2 else (x, y, z, heading)
                total_traveled_cm += math.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2 + (z - prev_z) ** 2)
            rover_cell_x, rover_cell_y = planner.world_to_cell(x, y)
            if planner_needs_rebuild(planner, (rover_cell_x, rover_cell_y)):
                planner = rebuild_planner_with_obstacles((x, y), (goal_x, goal_y), recorded_obstacle_points)
                rover_cell_x, rover_cell_y = planner.world_to_cell(x, y)
                if viewer is not None:
                    viewer.view_center_x_cm = x
                    viewer.view_center_y_cm = y
                    viewer._update_scale(planner)

            obstacle_added = False
            obstacle_debug_rows: list[dict[str, float | int | bool | str]] = []
            if reverse_until is not None and now >= reverse_until:
                reverse_until = None
                set_throttle(sock, 0.0)
                set_brakes(sock, True)
                time.sleep(RECOVERY_BRAKE_SECONDS)
                set_brakes(sock, False)
                stuck_detection_armed = False
                stuck_rearm_pending = True
                stationary_anchor_xy = (x, y)
                stationary_anchor_time = time.monotonic()

            startup_grace_elapsed = (now - start_time) >= STARTUP_STUCK_GRACE_SEC
            if reverse_until is None:
                moved_since_anchor_cm = distance_cm(x, y, stationary_anchor_xy[0], stationary_anchor_xy[1])
                movement_detected = moved_since_anchor_cm > STATIONARY_MOVE_THRESHOLD_CM or abs(speed) > STATIONARY_SPEED_THRESHOLD
                if stuck_rearm_pending:
                    dist_from_stuck_origin_cm = (
                        0.0
                        if stuck_rearm_origin_xy is None
                        else distance_cm(x, y, stuck_rearm_origin_xy[0], stuck_rearm_origin_xy[1])
                    )
                    if speed > STATIONARY_SPEED_THRESHOLD and dist_from_stuck_origin_cm >= STUCK_REARM_EXIT_DISTANCE_CM:
                        stuck_rearm_pending = False
                        stuck_rearm_origin_xy = None
                        stuck_detection_armed = True
                        stationary_anchor_xy = (x, y)
                        stationary_anchor_time = now
                        recovery_reset_triggered = False
                        recovery_reset_watch_started_at = None
                        recovery_reset_watch_origin_xy = None
                elif movement_detected and not stuck_detection_armed:
                    stuck_detection_armed = True
                    stationary_anchor_xy = (x, y)
                    stationary_anchor_time = now
                elif movement_detected:
                    stationary_anchor_xy = (x, y)
                    stationary_anchor_time = now
                elif stuck_detection_armed and startup_grace_elapsed and (now - stationary_anchor_time) >= STATIONARY_TIMEOUT_SEC:
                    stuck_pose_xyzh = (x, y, z, heading)
                    obstacle_total += mark_stuck_obstacles_from_history(
                        planner,
                        pose_history,
                        stuck_pose_xyzh,
                        debug_rows=obstacle_debug_rows,
                    )
                    recorded_obstacle_points.extend(
                        (float(row["obstacle_x_cm"]), float(row["obstacle_y_cm"]))
                        for row in obstacle_debug_rows
                        if bool(row["placed"])
                    )
                    reverse_until = now + RECOVERY_REVERSE_SECONDS
                    stuck_detection_armed = False
                    stuck_rearm_pending = False
                    stuck_rearm_origin_xy = (x, y)
                    stationary_anchor_xy = (x, y)
                    stationary_anchor_time = now
                    if recovery_reset_watch_started_at is None:
                        recovery_reset_watch_started_at = now
                        recovery_reset_watch_origin_xy = (x, y)
                    obstacle_added = True

            if (
                not recovery_reset_triggered
                and recovery_reset_watch_started_at is not None
                and recovery_reset_watch_origin_xy is not None
            ):
                recovery_watch_distance_cm = distance_cm(
                    x,
                    y,
                    recovery_reset_watch_origin_xy[0],
                    recovery_reset_watch_origin_xy[1],
                )
                if recovery_watch_distance_cm >= RECOVERY_RESET_ESCAPE_RADIUS_CM:
                    recovery_reset_triggered = False
                    recovery_reset_watch_started_at = None
                    recovery_reset_watch_origin_xy = None
                elif (now - recovery_reset_watch_started_at) >= RECOVERY_RESET_STALL_TIMEOUT_SEC:
                    print("\n" + "=" * 72)
                    print("DUMBDRIVE RESET TRIGGERED: rover remained stuck after recovery.")
                    print(
                        f"Stall window: {now - recovery_reset_watch_started_at:.1f}s | "
                        f"Displacement: {recovery_watch_distance_cm:.1f} cm"
                    )
                    print("=" * 72 + "\n")
                    print(
                        "Recovery stall watchdog triggered reset: "
                        f"moved only {recovery_watch_distance_cm:.1f} cm in "
                        f"{now - recovery_reset_watch_started_at:.1f} s."
                    )
                    stop_rover(sock)
                    trigger_dust_reset()
                    recovery_reset_triggered = True
                    recovery_reset_watch_started_at = None
                    recovery_reset_watch_origin_xy = None
                    reverse_until = None
                    stuck_detection_armed = False
                    stuck_rearm_pending = False
                    stuck_rearm_origin_xy = None
                    stationary_anchor_xy = (x, y)
                    stationary_anchor_time = now

            path = compute_live_follow_path(planner, (x, y), (goal_x, goal_y))
            resampled_goal = False
            next_goal_reason: str | None = None
            if not path:
                (goal_x, goal_y), planner, path = choose_goal_with_path(
                    origin_xy=(origin_x, origin_y),
                    start_xy=(x, y),
                    recorded_obstacle_points=recorded_obstacle_points,
                )
                resampled_goal = True
                next_goal_reason = "replanned_goal"
                goal_distance = distance_cm(x, y, goal_x, goal_y)
                rover_cell_x, rover_cell_y = planner.world_to_cell(x, y)
            if path:
                target_x, target_y, waypoint_idx = select_local_path_target(
                    path_world=path,
                    rover_x_cm=x,
                    rover_y_cm=y,
                    lookahead_cm=PATH_TARGET_LOOKAHEAD_CM,
                )
            else:
                target_x, target_y = goal_x, goal_y
                waypoint_idx = 0

            goal_advanced = False
            if goal_distance <= DUMBDRIVE_GOAL_REACHED_CM:
                goals_reached += 1
                if CLEAR_KNOWN_OBSTACLES_ON_GOAL_REACHED:
                    recorded_obstacle_points.clear()
                    obstacle_total = 0
                (goal_x, goal_y), planner, path = choose_goal_with_path(
                    origin_xy=(origin_x, origin_y),
                    start_xy=(x, y),
                    recorded_obstacle_points=recorded_obstacle_points,
                )
                next_goal_reason = f"after_goal_{goals_reached}"
                if path:
                    target_x, target_y, waypoint_idx = select_local_path_target(
                        path_world=path,
                        rover_x_cm=x,
                        rover_y_cm=y,
                        lookahead_cm=PATH_TARGET_LOOKAHEAD_CM,
                    )
                else:
                    target_x = goal_x
                    target_y = goal_y
                    waypoint_idx = 0
                stationary_anchor_xy = (x, y)
                stationary_anchor_time = now
                stuck_detection_armed = False
                stuck_rearm_pending = False
                stuck_rearm_origin_xy = None
                recovery_reset_triggered = False
                recovery_reset_watch_started_at = None
                recovery_reset_watch_origin_xy = None
                goal_distance = distance_cm(x, y, goal_x, goal_y)
                rover_cell_x, rover_cell_y = planner.world_to_cell(x, y)
                throttle_cmd = 0.0
                steering_cmd = 0.0
                set_brakes(sock, False)
                set_steering(sock, steering_cmd)
                set_throttle(sock, throttle_cmd)
                goal_advanced = True
                status = f"Reached goal {goals_reached} | Goals: {goals_reached}"
            elif reverse_until is not None:
                status = build_status(recovering=True, obstacle_added=obstacle_added, path_len=len(path), goals_reached=goals_reached)
                throttle_cmd = RECOVERY_REVERSE_THROTTLE
                steering_cmd = compute_recovery_reverse_steering(x, y, heading, target_x, target_y)
                set_brakes(sock, False)
                set_steering(sock, steering_cmd)
                set_throttle(sock, throttle_cmd)
            else:
                status = build_status(recovering=False, obstacle_added=obstacle_added, path_len=len(path), goals_reached=goals_reached)
                if resampled_goal:
                    status = f"New goal selected | Goals: {goals_reached}"
                desired_throttle_cmd, steering_cmd, _, heading_error = choose_drive_command(x, y, heading, target_x, target_y)
                if should_reverse_to_target(heading_error):
                    throttle_cmd = -NORMAL_DRIVE_THROTTLE
                    steering_cmd = 0.0
                else:
                    throttle_cmd = desired_throttle_cmd
                    if throttle_cmd > 0.0:
                        throttle_cmd = max(
                            MIN_FORWARD_DRIVE_THROTTLE,
                            min(throttle_cmd, NORMAL_DRIVE_THROTTLE),
                        )
                set_brakes(sock, False)
                set_steering(sock, steering_cmd)
                set_throttle(sock, throttle_cmd)

            waypoint_distance_cm = distance_cm(x, y, target_x, target_y)
            waypoint_distance_sum_cm += waypoint_distance_cm
            waypoint_distance_count += 1
            waypoint_distance_avg_cm = waypoint_distance_sum_cm / max(1, waypoint_distance_count)
            target_cell_x, target_cell_y = planner.world_to_cell(target_x, target_y)
            elapsed_s = now - start_time
            stationary_elapsed_s = now - stationary_anchor_time
            moved_since_anchor_cm = distance_cm(x, y, stationary_anchor_xy[0], stationary_anchor_xy[1])

            draw_kwargs = {
                "planner": planner,
                "rover_xy": (x, y),
                "heading_deg": heading,
                "goal_xy": (goal_x, goal_y),
                "target_xy": (target_x, target_y),
                "path_world": path,
                "status": status,
                "goal_distance_cm": goal_distance,
                "throttle_cmd": throttle_cmd,
                "steering_cmd": steering_cmd,
                "waypoint_idx": waypoint_idx,
                "waypoint_distance_cm": waypoint_distance_cm,
                "waypoint_distance_avg_cm": waypoint_distance_avg_cm,
                "obstacle_total": obstacle_total,
                "lidar_cm": lidar_cm,
            }
            if GOOD_UI:
                draw_kwargs["runtime_elapsed_s"] = elapsed_s
                draw_kwargs["total_traveled_cm"] = total_traveled_cm
                draw_kwargs["stationary_elapsed_s"] = stationary_elapsed_s
                draw_kwargs["throttle_cmd"] = throttle_cmd
                draw_kwargs["steering_cmd"] = steering_cmd
                draw_kwargs["reverse_active"] = reverse_until is not None
                draw_kwargs["raw_rover_xy"] = (
                    float(raw_telemetry.get("rover_pos_x", 0.0)),
                    float(raw_telemetry.get("rover_pos_y", 0.0)),
                )

            if not viewer.draw(**draw_kwargs):
                break
            if clean_log_writer is not None:
                clean_log_writer.log_lidar_update(
                    elapsed_s=elapsed_s,
                    step_idx=step_idx,
                    raw_telemetry=raw_telemetry,
                )
                if next_goal_reason is not None:
                    clean_log_writer.start_goal(goal_x, goal_y, reason=next_goal_reason)

            step_idx += 1
            if goal_advanced:
                time.sleep(0.5)
                continue
            time.sleep(CONTROL_PERIOD_SEC)

    except KeyboardInterrupt:
        pass
    finally:
        stop_rover(sock)
        close_rover_socket(sock)
        if viewer is not None:
            viewer.close()
        if clean_log_writer is not None:
            clean_log_writer.close()


if __name__ == "__main__":
    main()
