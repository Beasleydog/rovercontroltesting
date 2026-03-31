from __future__ import annotations

import csv
import math
import time
from collections import deque

import numpy as np

from dumbdrive import (
    RECOVERY_BRAKE_SECONDS,
    RECOVERY_REVERSE_SECONDS,
    RECOVERY_REVERSE_THROTTLE,
    STARTUP_STUCK_GRACE_SEC,
    STATIONARY_MOVE_THRESHOLD_CM,
    STATIONARY_SPEED_THRESHOLD,
    STATIONARY_TIMEOUT_SEC,
    STUCK_HISTORY_FRAMES,
    compute_recovery_reverse_steering,
    mark_stuck_obstacles_from_history,
    planner_needs_rebuild,
    rebuild_planner_with_obstacles,
)
from main import (
    CONTROL_PERIOD_SEC,
    DEBUG_LOG_ROOT,
    DEBUG_TEXT_LOGGING,
    GOAL_REACHED_CM,
    GRID_CELL_SIZE_CM,
    LIDAR_MAX_RANGE_CM,
    LIDAR_CLEAR_EVIDENCE_DELTA,
    LIDAR_CLEARANCE_COST_DELTA,
    LIDAR_OBSTACLE_EVIDENCE_DELTA,
    LIDAR_OBSTACLE_EVIDENCE_BLOCK_SIZE_CM,
    LIDAR_OBSTACLE_EVIDENCE_EDGE_GAIN,
    LIDAR_OBSTACLE_PROXIMITY_GAIN,
    LIDAR_SENSOR_LAYOUT,
    MapWindow,
    PATH_CLEARANCE_COST_SCALE,
    PATH_CLEARANCE_MAX_VALUE,
    PATH_CLEARANCE_MIN_VALUE,
    PATH_EVIDENCE_COST_SCALE,
    PATH_EVIDENCE_MAX_VALUE,
    PATH_EVIDENCE_MIN_VALUE,
    PATH_TARGET_LOOKAHEAD_CM,
    POSE_UNITS_TO_CM,
    REPLAN_MIN_INTERVAL_SEC,
    TARGET_DX_CM,
    TARGET_DY_CM,
    build_active_path_from_rover,
    choose_drive_command,
    create_planner,
    distance_cm,
    heading_basis_matrix,
    heading_error_deg,
    obstacle_mask_from_inference,
    parse_lidar,
    parse_pose,
    plan_path_for_following,
    select_local_path_target,
    stop_rover,
    update_obstacles_from_lidar,
)
from model import (
    INFERENCE_BACKEND_BUNDLE_CNN,
    configure_inference,
    inferencer_backend,
    ingest_lidar,
    reset_history,
)
from rover_control import (
    close_rover_socket,
    fetch_rover_telemetry,
    open_rover_socket,
    set_brakes,
    set_steering,
    set_throttle,
    wait_for_dust,
)


MAX_FORWARD_THROTTLE = 18.0
MIN_FORWARD_THROTTLE = 18.0
PATH_DIRECTION_LOOKAHEAD_CM = 120.0
# Temporary conservative reverse thresholds while close-range clearance is inflated.
# Wider hysteresis gap makes direction choice stickier once engaged.
PATH_REVERSE_ENGAGE_HEADING_DEG = 125.0
PATH_REVERSE_RELEASE_HEADING_DEG = 75.0
MODEL_HISTORY_MIN_DELTA = 0.0
MODEL_HISTORY_MIN_FRAMES = 4
MODEL_INFERENCE_BACKEND = INFERENCE_BACKEND_BUNDLE_CNN
IGNORE_BACKWARD_FACING_LIDAR_CLASSIFICATIONS = True
BACKWARD_FACING_LIDAR_YAW_THRESHOLD_DEG = 90.0
MIN_LIDAR_READING_CM = 150.0


def suppress_backward_facing_lidar_hits(obstacle_mask: np.ndarray) -> np.ndarray:
    masked = np.asarray(obstacle_mask, dtype=bool).copy()
    for sensor_idx, (_, _, sensor_yaw_deg, _) in enumerate(LIDAR_SENSOR_LAYOUT):
        if sensor_idx >= masked.size:
            break
        if abs(float(sensor_yaw_deg)) > BACKWARD_FACING_LIDAR_YAW_THRESHOLD_DEG:
            masked[sensor_idx] = False
    return masked


def clamp_min_lidar_reading(lidar_cm: np.ndarray) -> np.ndarray:
    lidar_arr = np.asarray(lidar_cm, dtype=np.float32).copy()
    valid_mask = lidar_arr >= 0.0
    lidar_arr[valid_mask] = np.maximum(lidar_arr[valid_mask], float(MIN_LIDAR_READING_CM))
    return lidar_arr


class SmartdriveDebugWriter:
    def __init__(self) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = DEBUG_LOG_ROOT / f"smartdrive_debug_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = (self.run_dir / "config.txt").open("w", encoding="utf-8")
        self.steps_file = (self.run_dir / "steps.txt").open("w", encoding="utf-8")
        self.replans_file = (self.run_dir / "replans.txt").open("w", encoding="utf-8")
        self.lidar_file = (self.run_dir / "lidar_points.txt").open("w", encoding="utf-8")
        self.zero_file = (self.run_dir / "zero_hits.txt").open("w", encoding="utf-8")
        self.stuck_file = (self.run_dir / "stuck_obstacles.txt").open("w", encoding="utf-8")
        self.paths_file = (self.run_dir / "paths.txt").open("w", encoding="utf-8")
        self.replay_path = self.run_dir / f"world_smartdrive_{timestamp}.txt"
        self.replay_file = self.replay_path.open("w", encoding="utf-8", newline="")
        self.replay_writer = csv.writer(self.replay_file)
        self._path_snapshot_idx = 0

        self.replay_writer.writerow(
            [
                "timestep",
                "x_cm",
                "y_cm",
                "z_cm",
                "yaw_deg",
                "basis_xx",
                "basis_xy",
                "basis_xz",
                "basis_yx",
                "basis_yy",
                "basis_yz",
                "basis_zx",
                "basis_zy",
                "basis_zz",
                "teleport_flag",
            ]
        )

        self.steps_file.write(
            "step_idx\telapsed_s\tstatus\treverse_active\tpath_reverse_active\tstuck_detection_armed\tplanner_rebuilt\treplan_attempted\t"
            "raw_path_points\tfollow_path_points\tactive_path_points\t"
            "raw_rover_x\traw_rover_y\traw_rover_z\traw_heading_deg\t"
            "rover_x_cm\trover_y_cm\trover_z_cm\trover_heading_deg\t"
            "raw_pose_delta\tpose_delta_cm\t"
            "telemetry_throttle\ttelemetry_steering\ttelemetry_speed\ttelemetry_brakes\ttelemetry_distance_traveled\tdistance_traveled_delta\t"
            "moved_since_anchor_cm\tstationary_elapsed_s\t"
            "goal_x_cm\tgoal_y_cm\tgoal_dist_cm\ttarget_x_cm\ttarget_y_cm\t"
            "rover_cell_x\trover_cell_y\ttarget_cell_x\ttarget_cell_y\t"
            "waypoint_idx\twaypoint_distance_cm\twaypoint_distance_avg_cm\t"
            "path_heading_error_deg\t"
            "throttle_cmd\tsteering_cmd\tnew_lidar_obstacles\tstuck_obstacles_added\tobstacle_total\n"
        )
        self.replans_file.write(
            "step_idx\telapsed_s\tattempted\ttrigger\tplanner_rebuilt\tprevious_path_points\traw_path_points\tfollow_path_points\tstatus\n"
        )
        self.lidar_file.write(
            "step_idx\telapsed_s\tsensor_idx\tsensor_label\tsensor_enabled\traw_cm\traw_is_zero\tvalid_range_gt0\tvalid_range_ge0\tmask_hit\tpitch_scale\t"
            "planar_hit_cm\thit_world_z_cm\tlow_clearance_hit\tchunk_name\tchunk_hit_count\tchunk_threshold_met\thit_dist_from_rover_cm\twithin_local_radius\t"
            "usable_for_mark\tplaced\tusable_reason\tmap_origin\t"
            "sensor_local_x_cm\tsensor_local_y_cm\tsensor_yaw_deg\tsensor_pitch_deg\t"
            "sensor_world_x_cm\tsensor_world_y_cm\tsensor_cell_x\tsensor_cell_y\t"
            "hit_world_x_cm\thit_world_y_cm\tray_hit_world_x_cm\tray_hit_world_y_cm\thit_cell_x\thit_cell_y\t"
            "center_hit_world_x_cm\tcenter_hit_world_y_cm\tcenter_hit_cell_x\tcenter_hit_cell_y\t"
            "rover_x_cm\trover_y_cm\trover_heading_deg\n"
        )
        self.zero_file.write(
            "step_idx\telapsed_s\tsensor_idx\tsensor_label\traw_cm\tmask_hit\tsensor_world_x_cm\tsensor_world_y_cm\t"
            "sensor_cell_x\tsensor_cell_y\trover_x_cm\trover_y_cm\trover_heading_deg\n"
        )
        self.stuck_file.write(
            "step_idx\telapsed_s\tstuck_x_cm\tstuck_y_cm\tstuck_z_cm\t"
            "hist_x_cm\thist_y_cm\thist_z_cm\thist_heading_deg\t"
            "dist_from_stuck_cm\tcontact_side\tcontact_min_lidar_cm\t"
            "bumper_sample_idx\tbumper_lateral_offset_cm\t"
            "obstacle_x_cm\tobstacle_y_cm\t"
            "obstacle_cell_x\tobstacle_cell_y\tplaced\n"
        )
        self.paths_file.write(
            "snapshot_idx\tstep_idx\telapsed_s\tlabel\tpoint_idx\tx_cm\ty_cm\n"
        )

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

    def write_config(
        self,
        planner,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        model_backend: str,
    ) -> None:
        self.config_file.write(f"created_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        self.config_file.write(f"model_backend={model_backend}\n")
        self.config_file.write(f"pose_units_to_cm={POSE_UNITS_TO_CM}\n")
        self.config_file.write(f"grid_cell_size_cm={GRID_CELL_SIZE_CM}\n")
        self.config_file.write(f"path_target_lookahead_cm={PATH_TARGET_LOOKAHEAD_CM}\n")
        self.config_file.write(f"max_forward_throttle={MAX_FORWARD_THROTTLE}\n")
        self.config_file.write(f"min_forward_throttle={MIN_FORWARD_THROTTLE}\n")
        self.config_file.write(f"recovery_reverse_throttle={RECOVERY_REVERSE_THROTTLE}\n")
        self.config_file.write(f"recovery_reverse_seconds={RECOVERY_REVERSE_SECONDS}\n")
        self.config_file.write(f"recovery_brake_seconds={RECOVERY_BRAKE_SECONDS}\n")
        self.config_file.write(f"stationary_timeout_sec={STATIONARY_TIMEOUT_SEC}\n")
        self.config_file.write(f"startup_stuck_grace_sec={STARTUP_STUCK_GRACE_SEC}\n")
        self.config_file.write(f"stationary_move_threshold_cm={STATIONARY_MOVE_THRESHOLD_CM}\n")
        self.config_file.write(f"stationary_speed_threshold={STATIONARY_SPEED_THRESHOLD}\n")
        self.config_file.write(f"path_direction_lookahead_cm={PATH_DIRECTION_LOOKAHEAD_CM}\n")
        self.config_file.write(f"path_reverse_engage_heading_deg={PATH_REVERSE_ENGAGE_HEADING_DEG}\n")
        self.config_file.write(f"path_reverse_release_heading_deg={PATH_REVERSE_RELEASE_HEADING_DEG}\n")
        self.config_file.write(f"model_history_min_delta={MODEL_HISTORY_MIN_DELTA}\n")
        self.config_file.write(f"model_history_min_frames={MODEL_HISTORY_MIN_FRAMES}\n")
        self.config_file.write(
            f"ignore_backward_facing_lidar_classifications={int(IGNORE_BACKWARD_FACING_LIDAR_CLASSIFICATIONS)}\n"
        )
        self.config_file.write(f"backward_facing_lidar_yaw_threshold_deg={BACKWARD_FACING_LIDAR_YAW_THRESHOLD_DEG}\n")
        self.config_file.write(f"min_lidar_reading_cm={MIN_LIDAR_READING_CM}\n")
        self.config_file.write(f"lidar_obstacle_evidence_delta={LIDAR_OBSTACLE_EVIDENCE_DELTA}\n")
        self.config_file.write(f"lidar_clear_evidence_delta={LIDAR_CLEAR_EVIDENCE_DELTA}\n")
        self.config_file.write(f"lidar_clearance_cost_delta={LIDAR_CLEARANCE_COST_DELTA}\n")
        self.config_file.write(f"lidar_obstacle_proximity_gain={LIDAR_OBSTACLE_PROXIMITY_GAIN}\n")
        self.config_file.write(f"lidar_obstacle_evidence_block_size_cm={LIDAR_OBSTACLE_EVIDENCE_BLOCK_SIZE_CM}\n")
        self.config_file.write(f"lidar_obstacle_evidence_edge_gain={LIDAR_OBSTACLE_EVIDENCE_EDGE_GAIN}\n")
        self.config_file.write(f"path_evidence_cost_scale={PATH_EVIDENCE_COST_SCALE}\n")
        self.config_file.write(f"path_clearance_cost_scale={PATH_CLEARANCE_COST_SCALE}\n")
        self.config_file.write(f"path_evidence_min_value={PATH_EVIDENCE_MIN_VALUE}\n")
        self.config_file.write(f"path_evidence_max_value={PATH_EVIDENCE_MAX_VALUE}\n")
        self.config_file.write(f"path_clearance_min_value={PATH_CLEARANCE_MIN_VALUE}\n")
        self.config_file.write(f"path_clearance_max_value={PATH_CLEARANCE_MAX_VALUE}\n")
        self.config_file.write(f"lidar_max_range_cm={LIDAR_MAX_RANGE_CM}\n")
        self.config_file.write(f"grid_origin_x_cm={planner.origin_x_cm}\n")
        self.config_file.write(f"grid_origin_y_cm={planner.origin_y_cm}\n")
        self.config_file.write(f"grid_width_cells={planner.width_cells}\n")
        self.config_file.write(f"grid_height_cells={planner.height_cells}\n")
        self.config_file.write(f"obstacle_padding_cells={planner.config.obstacle_padding_cells}\n")
        self.config_file.write(f"start_x_cm={start_xy[0]:.3f}\n")
        self.config_file.write(f"start_y_cm={start_xy[1]:.3f}\n")
        self.config_file.write(f"goal_x_cm={goal_xy[0]:.3f}\n")
        self.config_file.write(f"goal_y_cm={goal_xy[1]:.3f}\n")
        self.config_file.write("sensor_layout_idx\tsensor_x_cm\tsensor_y_cm\tsensor_yaw_deg\tsensor_pitch_deg\n")
        for idx, (sx, sy, syaw, spitch) in enumerate(LIDAR_SENSOR_LAYOUT):
            self.config_file.write(f"{idx}\t{sx:.3f}\t{sy:.3f}\t{syaw:.3f}\t{spitch:.3f}\n")
        self.config_file.flush()

    def log_step(self, **values: float | int | bool | str) -> None:
        ordered_keys = [
            "step_idx",
            "elapsed_s",
            "status",
            "reverse_active",
            "path_reverse_active",
            "stuck_detection_armed",
            "planner_rebuilt",
            "replan_attempted",
            "raw_path_points",
            "follow_path_points",
            "active_path_points",
            "raw_rover_x",
            "raw_rover_y",
            "raw_rover_z",
            "raw_heading_deg",
            "rover_x_cm",
            "rover_y_cm",
            "rover_z_cm",
            "rover_heading_deg",
            "raw_pose_delta",
            "pose_delta_cm",
            "telemetry_throttle",
            "telemetry_steering",
            "telemetry_speed",
            "telemetry_brakes",
            "telemetry_distance_traveled",
            "distance_traveled_delta",
            "moved_since_anchor_cm",
            "stationary_elapsed_s",
            "goal_x_cm",
            "goal_y_cm",
            "goal_dist_cm",
            "target_x_cm",
            "target_y_cm",
            "rover_cell_x",
            "rover_cell_y",
            "target_cell_x",
            "target_cell_y",
            "waypoint_idx",
            "waypoint_distance_cm",
            "waypoint_distance_avg_cm",
            "path_heading_error_deg",
            "throttle_cmd",
            "steering_cmd",
            "new_lidar_obstacles",
            "stuck_obstacles_added",
            "obstacle_total",
        ]
        row = "\t".join(self._fmt(values.get(k, "nan")) for k in ordered_keys)
        self.steps_file.write(f"{row}\n")
        self.steps_file.flush()

    def log_replan(
        self,
        step_idx: int,
        elapsed_s: float,
        attempted: bool,
        trigger: str,
        planner_rebuilt: bool,
        previous_path_points: int,
        raw_path_points: int,
        follow_path_points: int,
        status: str,
    ) -> None:
        self.replans_file.write(
            f"{step_idx}\t{elapsed_s:.3f}\t{int(attempted)}\t{trigger}\t{int(planner_rebuilt)}\t"
            f"{previous_path_points}\t{raw_path_points}\t{follow_path_points}\t{status}\n"
        )
        self.replans_file.flush()

    def log_lidar_rows(self, step_idx: int, elapsed_s: float, rows: list[dict[str, float | int | bool | str]]) -> None:
        ordered_keys = [
            "sensor_idx",
            "sensor_label",
            "sensor_enabled",
            "raw_cm",
            "raw_is_zero",
            "valid_range_gt0",
            "valid_range_ge0",
            "mask_hit",
            "pitch_scale",
            "planar_hit_cm",
            "hit_world_z_cm",
            "low_clearance_hit",
            "chunk_name",
            "chunk_hit_count",
            "chunk_threshold_met",
            "hit_dist_from_rover_cm",
            "within_local_radius",
            "usable_for_mark",
            "placed",
            "usable_reason",
            "map_origin",
            "sensor_local_x_cm",
            "sensor_local_y_cm",
            "sensor_yaw_deg",
            "sensor_pitch_deg",
            "sensor_world_x_cm",
            "sensor_world_y_cm",
            "sensor_cell_x",
            "sensor_cell_y",
            "hit_world_x_cm",
            "hit_world_y_cm",
            "ray_hit_world_x_cm",
            "ray_hit_world_y_cm",
            "hit_cell_x",
            "hit_cell_y",
            "center_hit_world_x_cm",
            "center_hit_world_y_cm",
            "center_hit_cell_x",
            "center_hit_cell_y",
            "rover_x_cm",
            "rover_y_cm",
            "rover_heading_deg",
        ]
        for row in rows:
            serialized = [self._fmt(step_idx), self._fmt(elapsed_s)]
            serialized.extend(self._fmt(row.get(key, "nan")) for key in ordered_keys)
            self.lidar_file.write("\t".join(serialized) + "\n")
        self.lidar_file.flush()

    def log_zero_rows(self, step_idx: int, elapsed_s: float, rows: list[dict[str, float | int | bool | str]]) -> None:
        for row in rows:
            if float(row.get("raw_cm", float("nan"))) != 0.0:
                continue
            serialized = [
                self._fmt(step_idx),
                self._fmt(elapsed_s),
                self._fmt(row.get("sensor_idx", "nan")),
                self._fmt(row.get("sensor_label", "nan")),
                self._fmt(row.get("raw_cm", "nan")),
                self._fmt(row.get("mask_hit", "nan")),
                self._fmt(row.get("sensor_world_x_cm", "nan")),
                self._fmt(row.get("sensor_world_y_cm", "nan")),
                self._fmt(row.get("sensor_cell_x", "nan")),
                self._fmt(row.get("sensor_cell_y", "nan")),
                self._fmt(row.get("rover_x_cm", "nan")),
                self._fmt(row.get("rover_y_cm", "nan")),
                self._fmt(row.get("rover_heading_deg", "nan")),
            ]
            self.zero_file.write("\t".join(serialized) + "\n")
        self.zero_file.flush()

    def log_stuck_rows(self, step_idx: int, elapsed_s: float, rows: list[dict[str, float | int | bool | str]]) -> None:
        ordered_keys = [
            "stuck_x_cm",
            "stuck_y_cm",
            "stuck_z_cm",
            "hist_x_cm",
            "hist_y_cm",
            "hist_z_cm",
            "hist_heading_deg",
            "dist_from_stuck_cm",
            "contact_side",
            "contact_min_lidar_cm",
            "bumper_sample_idx",
            "bumper_lateral_offset_cm",
            "obstacle_x_cm",
            "obstacle_y_cm",
            "obstacle_cell_x",
            "obstacle_cell_y",
            "placed",
        ]
        for row in rows:
            serialized = [self._fmt(step_idx), self._fmt(elapsed_s)]
            serialized.extend(self._fmt(row.get(key, "nan")) for key in ordered_keys)
            self.stuck_file.write("\t".join(serialized) + "\n")
        self.stuck_file.flush()

    def log_path_snapshot(
        self,
        step_idx: int,
        elapsed_s: float,
        label: str,
        path_world: list[tuple[float, float]],
    ) -> None:
        snapshot_idx = self._path_snapshot_idx
        self._path_snapshot_idx += 1
        if not path_world:
            self.paths_file.write(
                f"{snapshot_idx}\t{step_idx}\t{elapsed_s:.3f}\t{label}\t-1\tnan\tnan\n"
            )
        else:
            for point_idx, (x_cm, y_cm) in enumerate(path_world):
                self.paths_file.write(
                    f"{snapshot_idx}\t{step_idx}\t{elapsed_s:.3f}\t{label}\t{point_idx}\t{x_cm:.3f}\t{y_cm:.3f}\n"
                )
        self.paths_file.flush()

    def log_replay_frame(
        self,
        timestep: int,
        x_cm: float,
        y_cm: float,
        z_cm: float,
        yaw_deg: float,
        basis: np.ndarray,
        teleport_flag: int = 0,
    ) -> None:
        basis_arr = np.asarray(basis, dtype=np.float32).reshape(3, 3)
        self.replay_writer.writerow(
            [
                int(timestep),
                f"{float(x_cm):.6f}",
                f"{float(y_cm):.6f}",
                f"{float(z_cm):.6f}",
                f"{float(yaw_deg):.6f}",
                f"{float(basis_arr[0, 0]):.6f}",
                f"{float(basis_arr[0, 1]):.6f}",
                f"{float(basis_arr[0, 2]):.6f}",
                f"{float(basis_arr[1, 0]):.6f}",
                f"{float(basis_arr[1, 1]):.6f}",
                f"{float(basis_arr[1, 2]):.6f}",
                f"{float(basis_arr[2, 0]):.6f}",
                f"{float(basis_arr[2, 1]):.6f}",
                f"{float(basis_arr[2, 2]):.6f}",
                int(teleport_flag),
            ]
        )
        self.replay_file.flush()

    def close(self) -> None:
        self.config_file.close()
        self.steps_file.close()
        self.replans_file.close()
        self.lidar_file.close()
        self.zero_file.close()
        self.stuck_file.close()
        self.paths_file.close()
        self.replay_file.close()


def compute_follow_path(
    planner,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
) -> list[tuple[float, float]]:
    _, follow_path = plan_path_for_following(planner, start_xy, goal_xy)
    return follow_path


def collect_placed_obstacle_points(rows: list[dict[str, float | int | bool | str]]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        if bool(row.get("placed")):
            points.append((float(row["hit_world_x_cm"]), float(row["hit_world_y_cm"])))
    return points


def collect_stuck_obstacle_points(rows: list[dict[str, float | int | bool | str]]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        if bool(row.get("placed")):
            points.append((float(row["obstacle_x_cm"]), float(row["obstacle_y_cm"])))
    return points


def select_drive_target(
    path_world: list[tuple[float, float]],
    rover_x_cm: float,
    rover_y_cm: float,
    rover_heading_deg: float,
    fallback_target_xy: tuple[float, float],
) -> tuple[float, float, int, float]:
    if path_world:
        target_x, target_y, waypoint_idx = select_local_path_target(
            path_world=path_world,
            rover_x_cm=rover_x_cm,
            rover_y_cm=rover_y_cm,
            lookahead_cm=PATH_TARGET_LOOKAHEAD_CM,
        )
    else:
        target_x, target_y = fallback_target_xy
        waypoint_idx = 0

    path_heading_x, path_heading_y = target_x, target_y
    if path_world:
        path_heading_x, path_heading_y, _ = select_local_path_target(
            path_world=path_world,
            rover_x_cm=rover_x_cm,
            rover_y_cm=rover_y_cm,
            lookahead_cm=PATH_DIRECTION_LOOKAHEAD_CM,
        )
    path_heading_deg = math.degrees(math.atan2(path_heading_y - rover_y_cm, path_heading_x - rover_x_cm))
    path_heading_error_deg = heading_error_deg(rover_heading_deg, path_heading_deg)
    return (target_x, target_y, waypoint_idx, path_heading_error_deg)


def should_drive_path_in_reverse(path_heading_error_deg: float, reverse_active: bool) -> bool:
    threshold_deg = PATH_REVERSE_RELEASE_HEADING_DEG if reverse_active else PATH_REVERSE_ENGAGE_HEADING_DEG
    return abs(path_heading_error_deg) >= threshold_deg


def clamp_forward_throttle(throttle_cmd: float) -> float:
    return min(MAX_FORWARD_THROTTLE, max(MIN_FORWARD_THROTTLE, throttle_cmd))


def compute_path_reverse_command(
    rover_x_cm: float,
    rover_y_cm: float,
    rover_heading_deg: float,
    target_x_cm: float,
    target_y_cm: float,
) -> tuple[float, float]:
    desired_throttle_cmd, forward_steering_cmd, _, _ = choose_drive_command(
        rover_x_cm,
        rover_y_cm,
        rover_heading_deg,
        target_x_cm,
        target_y_cm,
    )
    throttle_cmd = -clamp_forward_throttle(desired_throttle_cmd)
    steering_cmd = max(-1.0, min(1.0, -forward_steering_cmd))
    return (throttle_cmd, steering_cmd)


def build_status(
    reverse_active: bool,
    path_reverse_active: bool,
    new_obstacles: int,
    stuck_marked: bool,
    path_len: int,
) -> str:
    if reverse_active:
        return "Recovering: reverse"
    if stuck_marked:
        return f"Stuck: marked + replanned ({path_len} pts)"
    if new_obstacles > 0:
        return f"Obstacle hits: +{new_obstacles}"
    if path_reverse_active:
        return "Running: path-reverse"
    if path_len <= 0:
        return "No path"
    return "Running"


def main() -> None:
    sock = open_rover_socket()
    viewer: MapWindow | None = None
    debug_writer: SmartdriveDebugWriter | None = None

    try:
        if not wait_for_dust(sock, timeout_seconds=20.0, poll_seconds=0.5):
            raise RuntimeError("DUST is not connected to TSS.")

        configure_inference(backend=MODEL_INFERENCE_BACKEND)
        reset_history()
        telemetry = fetch_rover_telemetry(sock)
        initial_x, initial_y, _, _ = parse_pose(telemetry)
        goal_x = initial_x + TARGET_DX_CM
        goal_y = initial_y + TARGET_DY_CM

        start_x, start_y, start_z, heading = parse_pose(telemetry)
        planner = create_planner((start_x, start_y), (goal_x, goal_y))
        viewer = MapWindow(planner)
        model_backend = inferencer_backend()
        debug_writer = SmartdriveDebugWriter()
        debug_writer.write_config(
            planner=planner,
            start_xy=(start_x, start_y),
            goal_xy=(goal_x, goal_y),
            model_backend=model_backend,
        )
        print(f"Smartdrive debug logs: {debug_writer.run_dir}")
        print(f"Smartdrive replay file: {debug_writer.replay_path}")
        if DEBUG_TEXT_LOGGING and not (
            model_backend.startswith("gru_pt_loaded") or model_backend.startswith("simple_pt_loaded")
        ):
            print(f"WARNING: lidar classifier backend='{model_backend}'")

        path = compute_follow_path(planner, (start_x, start_y), (goal_x, goal_y))
        if debug_writer is not None:
            debug_writer.log_path_snapshot(step_idx=-1, elapsed_s=0.0, label="initial_follow_path", path_world=path)
            debug_writer.log_replay_frame(
                timestep=0,
                x_cm=start_x,
                y_cm=start_y,
                z_cm=start_z,
                yaw_deg=heading,
                basis=heading_basis_matrix(heading),
                teleport_flag=0,
            )
        recorded_obstacle_points: list[tuple[float, float]] = []
        recorded_planner_updates: list[tuple[str, float, float, float]] = []
        pose_history: deque[tuple[float, float, float, float]] = deque(maxlen=STUCK_HISTORY_FRAMES)
        pose_history.append((start_x, start_y, 0.0, heading))
        stationary_anchor_xy = (start_x, start_y)
        stationary_anchor_time = time.monotonic()
        start_time = time.monotonic()
        last_plan_time = 0.0
        reverse_until: float | None = None
        path_reverse_active = False
        stuck_detection_armed = False
        obstacle_total = 0
        step_idx = 0
        waypoint_distance_sum_cm = 0.0
        waypoint_distance_count = 0
        prev_raw_pose: tuple[float, float, float] | None = None
        prev_scaled_pose: tuple[float, float, float] | None = None
        prev_distance_traveled: float | None = None

        while True:
            now = time.monotonic()
            telemetry = fetch_rover_telemetry(sock)
            raw_x = float(telemetry.get("rover_pos_x", 0.0))
            raw_y = float(telemetry.get("rover_pos_y", 0.0))
            raw_z = float(telemetry.get("rover_pos_z", 0.0))
            raw_heading = float(telemetry.get("heading", 0.0))
            x, y, z, heading = parse_pose(telemetry)
            telemetry_throttle = float(telemetry.get("throttle", 0.0))
            telemetry_steering = float(telemetry.get("steering", 0.0))
            speed = float(telemetry.get("speed", 0.0))
            telemetry_brakes = bool(telemetry.get("brakes", False))
            distance_traveled = float(telemetry.get("distance_traveled", 0.0))
            goal_distance = distance_cm(x, y, goal_x, goal_y)
            lidar_cm = parse_lidar(telemetry)
            lidar_cm = clamp_min_lidar_reading(lidar_cm)
            pose_history.append((x, y, z, heading))
            elapsed_s = now - start_time

            raw_pose_delta = 0.0
            if prev_raw_pose is not None:
                raw_pose_delta = math.sqrt(
                    (raw_x - prev_raw_pose[0]) ** 2
                    + (raw_y - prev_raw_pose[1]) ** 2
                    + (raw_z - prev_raw_pose[2]) ** 2
                )
            pose_delta_cm = 0.0
            if prev_scaled_pose is not None:
                pose_delta_cm = math.sqrt(
                    (x - prev_scaled_pose[0]) ** 2
                    + (y - prev_scaled_pose[1]) ** 2
                    + (z - prev_scaled_pose[2]) ** 2
                )
            distance_traveled_delta = 0.0 if prev_distance_traveled is None else (distance_traveled - prev_distance_traveled)

            planner_rebuilt = False
            rover_cell = planner.world_to_cell(x, y)
            if planner_needs_rebuild(planner, rover_cell):
                planner = rebuild_planner_with_obstacles(
                    (x, y),
                    (goal_x, goal_y),
                    recorded_obstacle_points,
                    recorded_planner_updates,
                )
                planner_rebuilt = True
                rover_cell = planner.world_to_cell(x, y)
                if viewer is not None:
                    viewer.view_center_x_cm = x
                    viewer.view_center_y_cm = y
                    viewer._update_scale(planner)
                path = []

            inference = ingest_lidar(
                lidar_cm=lidar_cm,
                pose_xyz_cm=np.asarray([x, y, z], dtype=np.float32),
                basis=heading_basis_matrix(heading),
                min_history_delta=MODEL_HISTORY_MIN_DELTA,
                min_history_frames=MODEL_HISTORY_MIN_FRAMES,
            )
            obstacle_mask = obstacle_mask_from_inference(inference, len(lidar_cm))
            if IGNORE_BACKWARD_FACING_LIDAR_CLASSIFICATIONS:
                obstacle_mask = suppress_backward_facing_lidar_hits(obstacle_mask)
            lidar_rows: list[dict[str, float | int | bool | str]] = []
            step_evidence_updates: list[tuple[float, float, float]] = []
            step_clearance_updates: list[tuple[float, float, float]] = []
            new_obstacles = update_obstacles_from_lidar(
                planner=planner,
                rover_x_cm=x,
                rover_y_cm=y,
                rover_z_cm=z,
                rover_heading_deg=heading,
                lidar_cm=lidar_cm,
                obstacle_mask=obstacle_mask,
                debug_rows=lidar_rows,
                evidence_updates=step_evidence_updates,
                clearance_updates=step_clearance_updates,
            )
            if step_evidence_updates:
                recorded_planner_updates.extend(("evidence", ux, uy, ud) for ux, uy, ud in step_evidence_updates)
            if step_clearance_updates:
                recorded_planner_updates.extend(("clearance", ux, uy, ud) for ux, uy, ud in step_clearance_updates)
            if new_obstacles > 0:
                obstacle_total += new_obstacles
            if debug_writer is not None:
                debug_writer.log_lidar_rows(step_idx=step_idx, elapsed_s=elapsed_s, rows=lidar_rows)
                debug_writer.log_zero_rows(step_idx=step_idx, elapsed_s=elapsed_s, rows=lidar_rows)

            if reverse_until is not None and now >= reverse_until:
                reverse_until = None
                set_throttle(sock, 0.0)
                set_brakes(sock, True)
                time.sleep(RECOVERY_BRAKE_SECONDS)
                set_brakes(sock, False)
                stationary_anchor_xy = (x, y)
                stationary_anchor_time = time.monotonic()

            stuck_marked = False
            stuck_added = 0
            stuck_rows: list[dict[str, float | int | bool | str]] = []
            if reverse_until is None:
                moved_since_anchor_cm = distance_cm(x, y, stationary_anchor_xy[0], stationary_anchor_xy[1])
                movement_detected = moved_since_anchor_cm > STATIONARY_MOVE_THRESHOLD_CM or abs(speed) > STATIONARY_SPEED_THRESHOLD
                if movement_detected and not stuck_detection_armed:
                    stuck_detection_armed = True
                    stationary_anchor_xy = (x, y)
                    stationary_anchor_time = now
                elif movement_detected:
                    stationary_anchor_xy = (x, y)
                    stationary_anchor_time = now
                elif (
                    stuck_detection_armed
                    and (now - start_time) >= STARTUP_STUCK_GRACE_SEC
                    and (now - stationary_anchor_time) >= STATIONARY_TIMEOUT_SEC
                ):
                    stuck_rows: list[dict[str, float | int | bool | str]] = []
                    stuck_added = mark_stuck_obstacles_from_history(
                        planner,
                        pose_history,
                        (x, y, z, heading),
                        debug_rows=stuck_rows,
                    )
                    if stuck_added > 0:
                        recorded_obstacle_points.extend(collect_stuck_obstacle_points(stuck_rows))
                        obstacle_total += stuck_added
                    reverse_until = now + RECOVERY_REVERSE_SECONDS
                    stationary_anchor_xy = (x, y)
                    stationary_anchor_time = now
                    stuck_marked = True
                    if debug_writer is not None and stuck_rows:
                        debug_writer.log_stuck_rows(step_idx=step_idx, elapsed_s=elapsed_s, rows=stuck_rows)
            else:
                moved_since_anchor_cm = distance_cm(x, y, stationary_anchor_xy[0], stationary_anchor_xy[1])

            previous_path_len = len(path)
            should_replan = (new_obstacles > 0 or stuck_marked or not path) and (now - last_plan_time) >= REPLAN_MIN_INTERVAL_SEC
            raw_path_points = 0
            follow_path_points = previous_path_len
            replan_status = "not_attempted"
            replan_trigger_parts: list[str] = []
            if new_obstacles > 0:
                replan_trigger_parts.append("lidar")
            if stuck_marked:
                replan_trigger_parts.append("stuck")
            if not path:
                replan_trigger_parts.append("empty_path")
            if planner_rebuilt:
                replan_trigger_parts.append("planner_rebuilt")
            if should_replan:
                raw_path, new_path = plan_path_for_following(planner, (x, y), (goal_x, goal_y))
                raw_path_points = len(raw_path)
                follow_path_points = len(new_path)
                if new_path:
                    path = new_path
                    last_plan_time = now
                    replan_status = "replanned"
                    if debug_writer is not None:
                        debug_writer.log_path_snapshot(
                            step_idx=step_idx,
                            elapsed_s=elapsed_s,
                            label="follow_path_replan",
                            path_world=path,
                        )
                elif not path:
                    path = []
                    replan_status = "replan_failed"
                else:
                    replan_status = "replan_kept_old_path"
            if debug_writer is not None:
                debug_writer.log_replan(
                    step_idx=step_idx,
                    elapsed_s=elapsed_s,
                    attempted=should_replan,
                    trigger="|".join(replan_trigger_parts) if replan_trigger_parts else "none",
                    planner_rebuilt=planner_rebuilt,
                    previous_path_points=previous_path_len,
                    raw_path_points=raw_path_points,
                    follow_path_points=follow_path_points,
                    status=replan_status,
                )

            active_path = build_active_path_from_rover(path, x, y)
            target_x, target_y, waypoint_idx, path_heading_error = select_drive_target(
                path_world=active_path,
                rover_x_cm=x,
                rover_y_cm=y,
                rover_heading_deg=heading,
                fallback_target_xy=(goal_x, goal_y),
            )
            if reverse_until is None:
                path_reverse_active = should_drive_path_in_reverse(
                    path_heading_error_deg=path_heading_error,
                    reverse_active=path_reverse_active,
                )
            else:
                path_reverse_active = False

            if goal_distance <= GOAL_REACHED_CM:
                throttle_cmd = 0.0
                steering_cmd = 0.0
                path_reverse_active = False
                status = "Goal reached"
                stop_rover(sock)
            elif reverse_until is not None:
                throttle_cmd = RECOVERY_REVERSE_THROTTLE
                steering_cmd = compute_recovery_reverse_steering(x, y, heading, target_x, target_y)
                status = build_status(
                    reverse_active=True,
                    path_reverse_active=False,
                    new_obstacles=new_obstacles,
                    stuck_marked=stuck_marked,
                    path_len=len(active_path),
                )
                set_brakes(sock, False)
                set_steering(sock, steering_cmd)
                set_throttle(sock, throttle_cmd)
            else:
                if path_reverse_active:
                    throttle_cmd, steering_cmd = compute_path_reverse_command(
                        rover_x_cm=x,
                        rover_y_cm=y,
                        rover_heading_deg=heading,
                        target_x_cm=target_x,
                        target_y_cm=target_y,
                    )
                else:
                    desired_throttle_cmd, steering_cmd, _, _ = choose_drive_command(x, y, heading, target_x, target_y)
                    throttle_cmd = clamp_forward_throttle(desired_throttle_cmd)
                status = build_status(
                    reverse_active=False,
                    path_reverse_active=path_reverse_active,
                    new_obstacles=new_obstacles,
                    stuck_marked=stuck_marked,
                    path_len=len(active_path),
                )
                set_brakes(sock, False)
                set_steering(sock, steering_cmd)
                set_throttle(sock, throttle_cmd)

            waypoint_distance_cm = distance_cm(x, y, target_x, target_y)
            waypoint_distance_sum_cm += waypoint_distance_cm
            waypoint_distance_count += 1
            waypoint_distance_avg_cm = waypoint_distance_sum_cm / max(1, waypoint_distance_count)
            rover_cell = planner.world_to_cell(x, y)
            target_cell_x, target_cell_y = planner.world_to_cell(target_x, target_y)
            stationary_elapsed_s = now - stationary_anchor_time
            if debug_writer is not None:
                debug_writer.log_replay_frame(
                    timestep=step_idx + 1,
                    x_cm=x,
                    y_cm=y,
                    z_cm=z,
                    yaw_deg=heading,
                    basis=heading_basis_matrix(heading),
                    teleport_flag=0,
                )
                debug_writer.log_step(
                    step_idx=step_idx,
                    elapsed_s=elapsed_s,
                    status=status,
                    reverse_active=reverse_until is not None,
                    path_reverse_active=path_reverse_active,
                    stuck_detection_armed=stuck_detection_armed,
                    planner_rebuilt=planner_rebuilt,
                    replan_attempted=should_replan,
                    raw_path_points=raw_path_points,
                    follow_path_points=follow_path_points,
                    active_path_points=len(active_path),
                    raw_rover_x=raw_x,
                    raw_rover_y=raw_y,
                    raw_rover_z=raw_z,
                    raw_heading_deg=raw_heading,
                    rover_x_cm=x,
                    rover_y_cm=y,
                    rover_z_cm=z,
                    rover_heading_deg=heading,
                    raw_pose_delta=raw_pose_delta,
                    pose_delta_cm=pose_delta_cm,
                    telemetry_throttle=telemetry_throttle,
                    telemetry_steering=telemetry_steering,
                    telemetry_speed=speed,
                    telemetry_brakes=telemetry_brakes,
                    telemetry_distance_traveled=distance_traveled,
                    distance_traveled_delta=distance_traveled_delta,
                    moved_since_anchor_cm=moved_since_anchor_cm,
                    stationary_elapsed_s=stationary_elapsed_s,
                    goal_x_cm=goal_x,
                    goal_y_cm=goal_y,
                    goal_dist_cm=goal_distance,
                    target_x_cm=target_x,
                    target_y_cm=target_y,
                    rover_cell_x=rover_cell[0],
                    rover_cell_y=rover_cell[1],
                    target_cell_x=target_cell_x,
                    target_cell_y=target_cell_y,
                    waypoint_idx=waypoint_idx,
                    waypoint_distance_cm=waypoint_distance_cm,
                    waypoint_distance_avg_cm=waypoint_distance_avg_cm,
                    path_heading_error_deg=path_heading_error,
                    throttle_cmd=throttle_cmd,
                    steering_cmd=steering_cmd,
                    new_lidar_obstacles=new_obstacles,
                    stuck_obstacles_added=stuck_added,
                    obstacle_total=obstacle_total,
                )
            if not viewer.draw(
                planner=planner,
                rover_xy=(x, y),
                heading_deg=heading,
                goal_xy=(goal_x, goal_y),
                target_xy=(target_x, target_y),
                path_world=active_path,
                status=f"{status} | err={path_heading_error:.1f}",
                goal_distance_cm=goal_distance,
                throttle_cmd=throttle_cmd,
                steering_cmd=steering_cmd,
                waypoint_idx=waypoint_idx,
                waypoint_distance_cm=waypoint_distance_cm,
                waypoint_distance_avg_cm=waypoint_distance_avg_cm,
                obstacle_total=obstacle_total,
                lidar_cm=lidar_cm,
                lidar_debug_rows=lidar_rows,
            ):
                break

            if status == "Goal reached":
                time.sleep(0.5)
                break

            step_idx += 1
            if DEBUG_TEXT_LOGGING and step_idx % 10 == 0:
                print(
                    f"[smartdrive] step={step_idx} status='{status}' "
                    f"goal_cm={goal_distance:.1f} path_pts={len(active_path)} "
                    f"obs_total={obstacle_total} path_err={path_heading_error:.1f}"
                )
            prev_raw_pose = (raw_x, raw_y, raw_z)
            prev_scaled_pose = (x, y, z)
            prev_distance_traveled = distance_traveled
            time.sleep(CONTROL_PERIOD_SEC)

    except KeyboardInterrupt:
        pass
    finally:
        stop_rover(sock)
        close_rover_socket(sock)
        if debug_writer is not None:
            debug_writer.close()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
