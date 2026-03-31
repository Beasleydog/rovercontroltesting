from __future__ import annotations

import time

import numpy as np

from dumbdrive import (
    CLEAR_KNOWN_OBSTACLES_ON_GOAL_REACHED,
    DUMBDRIVE_GOAL_REACHED_CM,
    GOOD_UI,
    MIN_FORWARD_DRIVE_THROTTLE,
    NORMAL_DRIVE_THROTTLE,
    CleanGoalLogWriter,
    make_sanitized_telemetry,
    planner_needs_rebuild,
    rebuild_planner_with_obstacles,
    sample_random_goal_xy,
    should_reverse_to_target,
)
from html_ui import HtmlCanvasWindow
from main import (
    CONTROL_PERIOD_SEC,
    LIDAR_SENSOR_LAYOUT,
    MapWindow,
    PATH_TARGET_LOOKAHEAD_CM,
    TARGET_DX_CM,
    TARGET_DY_CM,
    choose_drive_command,
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
    fetch_rover_json,
    open_rover_socket,
    set_brakes,
    set_steering,
    set_throttle,
    wait_for_dust,
)



GOOD_UI=False
MODEL_INFERENCE_BACKEND = INFERENCE_BACKEND_BUNDLE_CNN
MODEL_HISTORY_MIN_DELTA = 0.0
MODEL_HISTORY_MIN_FRAMES = 4
USE_RAW_MODEL_CLASSIFICATIONS = False
MODELDRIVE_THROTTLE = 10
INVERT_LIDAR_LEFT_RIGHT = False
IGNORE_BACKWARD_FACING_LIDAR_CLASSIFICATIONS = True
BACKWARD_FACING_LIDAR_YAW_THRESHOLD_DEG = 90.0
MIN_LIDAR_READING_CM = 150.0
TARGET_BEHIND_REVERSE_HEADING_DEG = 90.0

LEFT_RIGHT_SENSOR_SWAP_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 4),
    (1, 3),
    (5, 6),
    (7, 8),
    (9, 12),
    (10, 11),
    (13, 14),
    (15, 16),
)


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


def maybe_invert_lidar_left_right(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values).copy()
    if not INVERT_LIDAR_LEFT_RIGHT:
        return arr
    for left_idx, right_idx in LEFT_RIGHT_SENSOR_SWAP_PAIRS:
        if left_idx >= arr.shape[0] or right_idx >= arr.shape[0]:
            continue
        arr[left_idx], arr[right_idx] = arr[right_idx].copy(), arr[left_idx].copy()
    return arr


def compute_live_follow_path(
    planner,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
) -> list[tuple[float, float]]:
    _, follow_path = plan_path_for_following(planner, start_xy, goal_xy)
    return follow_path


def choose_goal_with_model_path(
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
        planner = rebuild_planner_with_obstacles(
            start_xy,
            goal_xy,
            recorded_obstacle_points,
        )
        path = compute_live_follow_path(planner, start_xy, goal_xy)
        last_planner = planner
        if path:
            return (goal_xy, planner, path)

    fallback_goal_xy = candidate_goals[-1]
    return (fallback_goal_xy, last_planner, [])


def collect_placed_obstacle_points(rows: list[dict[str, float | int | bool | str]]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        if bool(row.get("placed")):
            points.append((float(row["hit_world_x_cm"]), float(row["hit_world_y_cm"])))
    return points


def build_status(
    *,
    new_obstacles: int,
    history_ready: bool,
    path_len: int,
    goals_reached: int,
    resampled_goal: bool,
    reverse_active: bool,
) -> str:
    if resampled_goal:
        return f"New goal selected | Goals: {goals_reached}"
    if not history_ready:
        return f"Model warmup | Goals: {goals_reached}"
    if reverse_active:
        return f"Running: reverse | Goals: {goals_reached}"
    if new_obstacles > 0:
        return f"Model obstacle hits: +{new_obstacles} | Goals: {goals_reached}"
    if path_len <= 0:
        return f"No path | Goals: {goals_reached}"
    return f"Running | Goals: {goals_reached}"


def clamp_forward_throttle(throttle_cmd: float) -> float:
    return max(MIN_FORWARD_DRIVE_THROTTLE, min(float(throttle_cmd), float(MODELDRIVE_THROTTLE)))


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

    target_heading_deg = np.degrees(np.arctan2(target_y - rover_y_cm, target_x - rover_x_cm))
    target_heading_error_deg = heading_error_deg(rover_heading_deg, float(target_heading_deg))
    return (target_x, target_y, waypoint_idx, target_heading_error_deg)


def should_drive_target_in_reverse(target_heading_error_deg: float) -> bool:
    return abs(float(target_heading_error_deg)) > TARGET_BEHIND_REVERSE_HEADING_DEG


def compute_reverse_drive_command(
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
    throttle_cmd = -clamp_forward_throttle(max(abs(desired_throttle_cmd), float(MODELDRIVE_THROTTLE)))
    steering_cmd = max(-1.0, min(1.0, -forward_steering_cmd))
    return (throttle_cmd, steering_cmd)


def main() -> None:
    sock = open_rover_socket()
    viewer: HtmlCanvasWindow | MapWindow | None = None
    clean_log_writer: CleanGoalLogWriter | None = None

    try:
        if not wait_for_dust(sock, timeout_seconds=20.0, poll_seconds=0.5):
            raise RuntimeError("DUST is not connected to TSS.")

        configure_inference(backend=MODEL_INFERENCE_BACKEND)
        reset_history()

        rover_json = fetch_rover_json(sock)
        raw_telemetry = rover_json.get("pr_telemetry")
        if not isinstance(raw_telemetry, dict):
            raise RuntimeError("ROVER.json did not contain pr_telemetry")
        telemetry = make_sanitized_telemetry(raw_telemetry)

        initial_x, initial_y, _, _ = parse_pose(telemetry)
        origin_xy = (initial_x, initial_y)
        preferred_goal_xy = (initial_x + TARGET_DX_CM, initial_y + TARGET_DY_CM)

        start_x, start_y, _, _ = parse_pose(telemetry)
        recorded_obstacle_points: list[tuple[float, float]] = []
        (goal_x, goal_y), planner, path = choose_goal_with_model_path(
            origin_xy=origin_xy,
            start_xy=(start_x, start_y),
            recorded_obstacle_points=recorded_obstacle_points,
            preferred_goal_xy=preferred_goal_xy,
        )
        if planner is None:
            raise RuntimeError("Failed to create planner for modeldrive.")

        viewer = HtmlCanvasWindow(planner) if GOOD_UI else MapWindow(planner)
        clean_log_writer = CleanGoalLogWriter()
        clean_log_writer.start_goal(goal_x, goal_y, reason="initial_goal")
        print(f"Modeldrive clean logs: {clean_log_writer.root_dir}")

        model_backend = inferencer_backend()
        if not model_backend.startswith("bundle_cnn_loaded"):
            print(f"WARNING: lidar classifier backend='{model_backend}'")

        step_idx = 0
        start_time = time.monotonic()
        goals_reached = 0
        status = build_status(
            new_obstacles=0,
            history_ready=False,
            path_len=len(path),
            goals_reached=goals_reached,
            resampled_goal=False,
            reverse_active=False,
        )
        target_x = goal_x
        target_y = goal_y
        waypoint_idx = 0
        waypoint_distance_sum_cm = 0.0
        waypoint_distance_count = 0
        total_traveled_cm = 0.0
        prev_xyz: tuple[float, float, float] | None = None

        while True:
            now = time.monotonic()
            rover_json = fetch_rover_json(sock)
            raw_telemetry = rover_json.get("pr_telemetry")
            if not isinstance(raw_telemetry, dict):
                raise RuntimeError("ROVER.json did not contain pr_telemetry")
            telemetry = make_sanitized_telemetry(raw_telemetry)

            x, y, z, heading = parse_pose(telemetry)
            lidar_cm = clamp_min_lidar_reading(parse_lidar(telemetry))
            goal_distance = distance_cm(x, y, goal_x, goal_y)
            if prev_xyz is not None:
                total_traveled_cm += distance_cm(x, y, prev_xyz[0], prev_xyz[1])
            prev_xyz = (x, y, z)

            rover_cell = planner.world_to_cell(x, y)
            if planner_needs_rebuild(planner, rover_cell):
                planner = rebuild_planner_with_obstacles(
                    (x, y),
                    (goal_x, goal_y),
                    recorded_obstacle_points,
                )
                if viewer is not None:
                    viewer.view_center_x_cm = x
                    viewer.view_center_y_cm = y
                    viewer._update_scale(planner)

            inference = ingest_lidar(
                lidar_cm=lidar_cm,
                pose_xyz_cm=np.asarray([x, y, z], dtype=np.float32),
                basis=heading_basis_matrix(heading),
                min_history_delta=MODEL_HISTORY_MIN_DELTA,
                min_history_frames=MODEL_HISTORY_MIN_FRAMES,
            )
            history_ready = bool(inference.get("history_ready", False))
            obstacle_mask = obstacle_mask_from_inference(inference, len(lidar_cm))
            if not USE_RAW_MODEL_CLASSIFICATIONS and IGNORE_BACKWARD_FACING_LIDAR_CLASSIFICATIONS:
                obstacle_mask = suppress_backward_facing_lidar_hits(obstacle_mask)
            lidar_cm_for_projection = maybe_invert_lidar_left_right(lidar_cm)
            obstacle_mask_for_projection = maybe_invert_lidar_left_right(obstacle_mask)

            lidar_rows: list[dict[str, float | int | bool | str]] = []
            new_obstacles = update_obstacles_from_lidar(
                planner=planner,
                rover_x_cm=x,
                rover_y_cm=y,
                rover_z_cm=z,
                rover_heading_deg=heading,
                lidar_cm=lidar_cm_for_projection,
                obstacle_mask=obstacle_mask_for_projection,
                debug_rows=lidar_rows,
            )
            if new_obstacles > 0:
                recorded_obstacle_points.extend(collect_placed_obstacle_points(lidar_rows))
            planner = rebuild_planner_with_obstacles(
                (x, y),
                (goal_x, goal_y),
                recorded_obstacle_points,
            )
            if viewer is not None:
                viewer.view_center_x_cm = x
                viewer.view_center_y_cm = y
                viewer._update_scale(planner)

            path = compute_live_follow_path(planner, (x, y), (goal_x, goal_y))
            resampled_goal = False
            next_goal_reason: str | None = None
            if not path:
                (goal_x, goal_y), planner, path = choose_goal_with_model_path(
                    origin_xy=origin_xy,
                    start_xy=(x, y),
                    recorded_obstacle_points=recorded_obstacle_points,
                )
                resampled_goal = True
                next_goal_reason = "replanned_goal"
                goal_distance = distance_cm(x, y, goal_x, goal_y)

            target_x, target_y, waypoint_idx, target_heading_error = select_drive_target(
                path_world=path,
                rover_x_cm=x,
                rover_y_cm=y,
                rover_heading_deg=heading,
                fallback_target_xy=(goal_x, goal_y),
            )
            reverse_active = should_drive_target_in_reverse(target_heading_error)

            goal_advanced = False
            if goal_distance <= DUMBDRIVE_GOAL_REACHED_CM:
                goals_reached += 1
                if CLEAR_KNOWN_OBSTACLES_ON_GOAL_REACHED:
                    recorded_obstacle_points.clear()
                (goal_x, goal_y), planner, path = choose_goal_with_model_path(
                    origin_xy=origin_xy,
                    start_xy=(x, y),
                    recorded_obstacle_points=recorded_obstacle_points,
                )
                next_goal_reason = f"after_goal_{goals_reached}"
                if path:
                    target_x, target_y, waypoint_idx, target_heading_error = select_drive_target(
                        path_world=path,
                        rover_x_cm=x,
                        rover_y_cm=y,
                        rover_heading_deg=heading,
                        fallback_target_xy=(goal_x, goal_y),
                    )
                else:
                    target_x, target_y = goal_x, goal_y
                    waypoint_idx = 0
                    target_heading_error = 0.0
                goal_distance = distance_cm(x, y, goal_x, goal_y)
                throttle_cmd = 0.0
                steering_cmd = 0.0
                set_brakes(sock, False)
                set_steering(sock, steering_cmd)
                set_throttle(sock, throttle_cmd)
                status = f"Reached goal {goals_reached} | Goals: {goals_reached}"
                goal_advanced = True
            else:
                status = build_status(
                    new_obstacles=new_obstacles,
                    history_ready=history_ready,
                    path_len=len(path),
                    goals_reached=goals_reached,
                    resampled_goal=resampled_goal,
                    reverse_active=reverse_active,
                )
                if reverse_active:
                    throttle_cmd, steering_cmd = compute_reverse_drive_command(
                        x,
                        y,
                        heading,
                        target_x,
                        target_y,
                    )
                else:
                    desired_throttle_cmd, steering_cmd, _, heading_error = choose_drive_command(
                        x,
                        y,
                        heading,
                        target_x,
                        target_y,
                    )
                    if should_reverse_to_target(heading_error):
                        throttle_cmd = -float(MODELDRIVE_THROTTLE)
                        steering_cmd = 0.0
                    else:
                        throttle_cmd = desired_throttle_cmd
                        if throttle_cmd > 0.0:
                            throttle_cmd = clamp_forward_throttle(throttle_cmd)
                set_brakes(sock, False)
                set_steering(sock, steering_cmd)
                set_throttle(sock, throttle_cmd)

            waypoint_distance_cm = distance_cm(x, y, target_x, target_y)
            waypoint_distance_sum_cm += waypoint_distance_cm
            waypoint_distance_count += 1
            waypoint_distance_avg_cm = waypoint_distance_sum_cm / max(1, waypoint_distance_count)
            elapsed_s = now - start_time

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
                "obstacle_total": len(recorded_obstacle_points),
                "lidar_cm": lidar_cm_for_projection,
                "lidar_debug_rows": lidar_rows,
            }
            if GOOD_UI:
                draw_kwargs["runtime_elapsed_s"] = elapsed_s
                draw_kwargs["total_traveled_cm"] = total_traveled_cm
                draw_kwargs["stationary_elapsed_s"] = 0.0
                draw_kwargs["reverse_active"] = reverse_active
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
