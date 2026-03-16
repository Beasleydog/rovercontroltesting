from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from main import (
    LIDAR_MAX_RANGE_CM,
    LIDAR_SENSOR_LAYOUT,
    ROVER_BODY_LENGTH_CM,
    ROVER_BODY_WIDTH_CM,
    local_to_world_2d,
    parse_lidar,
    parse_pose,
)
from rover_control import close_rover_socket, fetch_rover_telemetry, open_rover_socket, wait_for_dust

try:
    import pygame
except ModuleNotFoundError as exc:
    raise RuntimeError("pygame is required for pointscan.py. Install with: pip install -r requirements.txt") from exc


RUNS_DIR = Path("runs")
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
BG_COLOR = (10, 12, 16)
GRID_COLOR = (36, 52, 68)
POINT_COLOR = (72, 104, 128)
ACTIVE_POINT_COLOR = (130, 240, 255)
PATH_COLOR = (255, 186, 84)
ACTIVE_PATH_COLOR = (255, 120, 72)
ROVER_COLOR = (255, 230, 120)
TEXT_COLOR = (236, 240, 245)
NEAR_PLANE_CM = 10.0
POINTSCAN_DURATION_SEC = 30.0
POINTSCAN_POLL_HZ = 12.0
POINTSCAN_FILE_PATH: Path | None = Path("runs/pointscan_from_smartdrive_20260311_185954/points.csv")
# POINTSCAN_FILE_PATH: Path | None = None
MOVE_SPEED_CM_PER_SEC = 500.0
SHIFT_MULTIPLIER = 3.0
MOUSE_SENSITIVITY_DEG = 0.25
DEFAULT_CAMERA_DISTANCE_CM = 1500.0
DEFAULT_CAMERA_HEIGHT_CM = 700.0
FOV_DEG = 70.0
TOPDOWN_HEIGHT_MARGIN_CM = 600.0


@dataclass(frozen=True)
class ScanPoint:
    xyz_cm: np.ndarray
    sensor_idx: int
    distance_cm: float
    sample_idx: int


@dataclass(frozen=True)
class PoseSample:
    xyz_cm: np.ndarray
    heading_deg: float


@dataclass
class CameraState:
    position_cm: np.ndarray
    yaw_deg: float
    pitch_deg: float
    topdown_locked: bool = False


def rotation_matrix_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    yaw_mat = np.asarray(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    pitch_mat = np.asarray(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float32,
    )
    return yaw_mat @ pitch_mat


def collect_point_scan(duration_sec: float, poll_hz: float) -> tuple[list[ScanPoint], list[PoseSample], Path]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"pointscan_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "points.csv"

    sock = open_rover_socket()
    points: list[ScanPoint] = []
    poses: list[PoseSample] = []
    delay_sec = 0.0 if poll_hz <= 0.0 else (1.0 / poll_hz)
    sample_idx = 0

    try:
        if not wait_for_dust(sock, timeout_seconds=20.0, poll_seconds=0.5):
            raise RuntimeError("DUST is not connected to TSS.")

        end_time = time.monotonic() + max(0.1, float(duration_sec))
        while time.monotonic() < end_time:
            telemetry = fetch_rover_telemetry(sock)
            rover_x_cm, rover_y_cm, rover_z_cm, heading_deg = parse_pose(telemetry)
            poses.append(PoseSample(xyz_cm=np.asarray([rover_x_cm, rover_y_cm, rover_z_cm], dtype=np.float32), heading_deg=heading_deg))
            lidar_cm = parse_lidar(telemetry)

            for sensor_idx, (sx, sy, syaw_deg, spitch_deg) in enumerate(LIDAR_SENSOR_LAYOUT):
                if sensor_idx >= len(lidar_cm):
                    continue
                distance_cm = float(lidar_cm[sensor_idx])
                if not (0.0 < distance_cm <= LIDAR_MAX_RANGE_CM):
                    continue

                sensor_world_x_cm, sensor_world_y_cm = local_to_world_2d(
                    rover_x_cm,
                    rover_y_cm,
                    heading_deg,
                    sx,
                    sy,
                )
                yaw_rad = math.radians(heading_deg + syaw_deg)
                pitch_rad = math.radians(spitch_deg)
                planar_cm = distance_cm * math.cos(pitch_rad)
                hit_x_cm = sensor_world_x_cm + planar_cm * math.cos(yaw_rad)
                hit_y_cm = sensor_world_y_cm + planar_cm * math.sin(yaw_rad)
                hit_z_cm = rover_z_cm + distance_cm * math.sin(pitch_rad)
                points.append(
                    ScanPoint(
                        xyz_cm=np.asarray([hit_x_cm, hit_y_cm, hit_z_cm], dtype=np.float32),
                        sensor_idx=sensor_idx,
                        distance_cm=distance_cm,
                        sample_idx=sample_idx,
                    )
                )

            sample_idx += 1
            if delay_sec > 0.0:
                time.sleep(delay_sec)
    finally:
        close_rover_socket(sock)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "sample_idx",
                "sensor_idx",
                "distance_cm",
                "x_cm",
                "y_cm",
                "z_cm",
                "pose_x_cm",
                "pose_y_cm",
                "pose_z_cm",
                "heading_deg",
            ]
        )
        pose_by_idx = {idx: pose for idx, pose in enumerate(poses)}
        for point in points:
            pose = pose_by_idx.get(point.sample_idx)
            writer.writerow(
                [
                    point.sample_idx,
                    point.sensor_idx,
                    f"{point.distance_cm:.6f}",
                    f"{float(point.xyz_cm[0]):.6f}",
                    f"{float(point.xyz_cm[1]):.6f}",
                    f"{float(point.xyz_cm[2]):.6f}",
                    f"{float(pose.xyz_cm[0]):.6f}" if pose is not None else "",
                    f"{float(pose.xyz_cm[1]):.6f}" if pose is not None else "",
                    f"{float(pose.xyz_cm[2]):.6f}" if pose is not None else "",
                    f"{float(pose.heading_deg):.6f}" if pose is not None else "",
                ]
            )

    return (points, poses, run_dir)


def load_point_scan(csv_path: Path) -> tuple[list[ScanPoint], list[PoseSample], Path]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Pointscan file not found: {csv_path}")

    points: list[ScanPoint] = []
    pose_map: dict[int, PoseSample] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is missing a header row")
        required = {"sample_idx", "sensor_idx", "distance_cm", "x_cm", "y_cm", "z_cm"}
        missing = required.difference(reader.fieldnames)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{csv_path} is missing required columns: {missing_text}")
        for row in reader:
            sample_idx = int(float(row["sample_idx"]))
            point = ScanPoint(
                xyz_cm=np.asarray(
                    [
                        float(row["x_cm"]),
                        float(row["y_cm"]),
                        float(row["z_cm"]),
                    ],
                    dtype=np.float32,
                ),
                sensor_idx=int(float(row["sensor_idx"])),
                distance_cm=float(row["distance_cm"]),
                sample_idx=sample_idx,
            )
            points.append(point)

            pose_x = row.get("pose_x_cm")
            pose_y = row.get("pose_y_cm")
            pose_z = row.get("pose_z_cm")
            heading = row.get("heading_deg")
            if pose_x is None or pose_y is None or pose_z is None or heading is None:
                continue
            if sample_idx not in pose_map:
                pose_map[sample_idx] = PoseSample(
                    xyz_cm=np.asarray(
                        [
                            float(pose_x),
                            float(pose_y),
                            float(pose_z),
                        ],
                        dtype=np.float32,
                    ),
                    heading_deg=float(heading),
                )

    poses = [pose_map[idx] for idx in sorted(pose_map)]
    return (points, poses, csv_path.parent)


def compute_initial_camera(points: list[ScanPoint], poses: list[PoseSample]) -> CameraState:
    if points:
        cloud = np.stack([point.xyz_cm for point in points], axis=0)
        center_cm = cloud.mean(axis=0)
        extent_xy_cm = np.max(np.linalg.norm(cloud[:, :2] - center_cm[:2], axis=1))
    elif poses:
        cloud = np.stack([pose.xyz_cm for pose in poses], axis=0)
        center_cm = cloud.mean(axis=0)
        extent_xy_cm = 600.0
    else:
        center_cm = np.zeros(3, dtype=np.float32)
        extent_xy_cm = 600.0

    distance_cm = max(DEFAULT_CAMERA_DISTANCE_CM, float(extent_xy_cm) * 1.8)
    position_cm = center_cm + np.asarray([-distance_cm, -distance_cm, DEFAULT_CAMERA_HEIGHT_CM], dtype=np.float32)
    return CameraState(position_cm=position_cm.astype(np.float32), yaw_deg=45.0, pitch_deg=25.0, topdown_locked=False)


def compute_topdown_camera(points: list[ScanPoint], poses: list[PoseSample], pose: PoseSample) -> CameraState:
    if points:
        cloud = np.stack([point.xyz_cm for point in points], axis=0)
        min_xyz = cloud.min(axis=0)
        max_xyz = cloud.max(axis=0)
        span_xy = np.max(max_xyz[:2] - min_xyz[:2])
        top_z = float(max_xyz[2])
    else:
        span_xy = 1500.0
        top_z = float(pose.xyz_cm[2])
    height_cm = max(DEFAULT_CAMERA_DISTANCE_CM, span_xy * 0.9 + TOPDOWN_HEIGHT_MARGIN_CM)
    position_cm = np.asarray(
        [
            float(pose.xyz_cm[0]),
            float(pose.xyz_cm[1]),
            top_z + height_cm,
        ],
        dtype=np.float32,
    )
    return CameraState(position_cm=position_cm, yaw_deg=90.0, pitch_deg=89.0, topdown_locked=True)


def project_point(
    point_cm: np.ndarray,
    camera: CameraState,
    focal_px: float,
    width: int,
    height: int,
) -> tuple[float, float, float] | None:
    rel = np.asarray(point_cm, dtype=np.float32) - camera.position_cm
    inv_rot = rotation_matrix_yaw_pitch(camera.yaw_deg, camera.pitch_deg).T
    cam = inv_rot @ rel
    depth = float(cam[0])
    if depth <= NEAR_PLANE_CM:
        return None
    sx = width * 0.5 + focal_px * (float(cam[1]) / depth)
    sy = height * 0.5 - focal_px * (float(cam[2]) / depth)
    return (sx, sy, depth)


def draw_grid(
    screen: pygame.Surface,
    camera: CameraState,
    focal_px: float,
    width: int,
    height: int,
    center_xy_cm: np.ndarray,
) -> None:
    grid_half = 3000
    grid_step = 250
    for gx in range(-grid_half, grid_half + 1, grid_step):
        start = np.asarray([center_xy_cm[0] + gx, center_xy_cm[1] - grid_half, 0.0], dtype=np.float32)
        end = np.asarray([center_xy_cm[0] + gx, center_xy_cm[1] + grid_half, 0.0], dtype=np.float32)
        ps = project_point(start, camera, focal_px, width, height)
        pe = project_point(end, camera, focal_px, width, height)
        if ps is not None and pe is not None:
            pygame.draw.line(screen, GRID_COLOR, ps[:2], pe[:2], width=1)
    for gy in range(-grid_half, grid_half + 1, grid_step):
        start = np.asarray([center_xy_cm[0] - grid_half, center_xy_cm[1] + gy, 0.0], dtype=np.float32)
        end = np.asarray([center_xy_cm[0] + grid_half, center_xy_cm[1] + gy, 0.0], dtype=np.float32)
        ps = project_point(start, camera, focal_px, width, height)
        pe = project_point(end, camera, focal_px, width, height)
        if ps is not None and pe is not None:
            pygame.draw.line(screen, GRID_COLOR, ps[:2], pe[:2], width=1)


def rover_corners_world(pose: PoseSample) -> list[np.ndarray]:
    half_length = ROVER_BODY_LENGTH_CM * 0.5
    half_width = ROVER_BODY_WIDTH_CM * 0.5
    corners_local = [
        np.asarray([half_length, half_width, 0.0], dtype=np.float32),
        np.asarray([half_length, -half_width, 0.0], dtype=np.float32),
        np.asarray([-half_length, -half_width, 0.0], dtype=np.float32),
        np.asarray([-half_length, half_width, 0.0], dtype=np.float32),
    ]
    yaw_rad = math.radians(pose.heading_deg)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    corners_world: list[np.ndarray] = []
    for local in corners_local:
        world_xy = np.asarray(
            [
                pose.xyz_cm[0] + local[0] * cos_yaw - local[1] * sin_yaw,
                pose.xyz_cm[1] + local[0] * sin_yaw + local[1] * cos_yaw,
                pose.xyz_cm[2],
            ],
            dtype=np.float32,
        )
        corners_world.append(world_xy)
    return corners_world


def draw_rover(
    screen: pygame.Surface,
    pose: PoseSample,
    camera: CameraState,
    focal_px: float,
    width: int,
    height: int,
) -> None:
    corners_world = rover_corners_world(pose)
    projected: list[tuple[float, float]] = []
    for corner in corners_world:
        result = project_point(corner, camera, focal_px, width, height)
        if result is None:
            return
        projected.append((result[0], result[1]))
    pygame.draw.polygon(screen, ROVER_COLOR, projected, width=2)

    front_center = pose.xyz_cm + np.asarray(
        [
            math.cos(math.radians(pose.heading_deg)) * (ROVER_BODY_LENGTH_CM * 0.5),
            math.sin(math.radians(pose.heading_deg)) * (ROVER_BODY_LENGTH_CM * 0.5),
            0.0,
        ],
        dtype=np.float32,
    )
    center_proj = project_point(pose.xyz_cm, camera, focal_px, width, height)
    front_proj = project_point(front_center, camera, focal_px, width, height)
    if center_proj is not None and front_proj is not None:
        pygame.draw.line(screen, ROVER_COLOR, center_proj[:2], front_proj[:2], width=2)


def run_viewer(points: list[ScanPoint], poses: list[PoseSample], run_dir: Path) -> None:
    pygame.init()
    try:
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption(f"Point Scan - {run_dir.name}")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", 18)
        camera = compute_initial_camera(points, poses)
        dragging = False

        if points:
            center_xy_cm = np.stack([point.xyz_cm for point in points], axis=0).mean(axis=0)[:2]
        elif poses:
            center_xy_cm = np.stack([pose.xyz_cm for pose in poses], axis=0).mean(axis=0)[:2]
        else:
            center_xy_cm = np.zeros(2, dtype=np.float32)

        while True:
            dt_sec = clock.tick(60) / 1000.0
            width, height = screen.get_size()
            mouse_x, _ = pygame.mouse.get_pos()
            scrub_t = 0.0 if width <= 1 else float(np.clip(mouse_x / max(1, width - 1), 0.0, 1.0))
            current_idx = 0 if not poses else min(len(poses) - 1, int(round(scrub_t * (len(poses) - 1))))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    dragging = True
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    dragging = False
                if event.type == pygame.MOUSEMOTION and dragging:
                    if camera.topdown_locked:
                        continue
                    dx, dy = event.rel
                    camera.yaw_deg += dx * MOUSE_SENSITIVITY_DEG
                    camera.pitch_deg = float(np.clip(camera.pitch_deg - dy * MOUSE_SENSITIVITY_DEG, -89.0, 89.0))
                if event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                    if camera.topdown_locked:
                        camera = compute_initial_camera(points, poses)
                    else:
                        target_pose = poses[current_idx] if poses else PoseSample(np.zeros(3, dtype=np.float32), 0.0)
                        camera = compute_topdown_camera(points, poses, target_pose)

            keys = pygame.key.get_pressed()
            move_speed = MOVE_SPEED_CM_PER_SEC * (SHIFT_MULTIPLIER if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 1.0)
            forward = rotation_matrix_yaw_pitch(camera.yaw_deg, 0.0) @ np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            right = rotation_matrix_yaw_pitch(camera.yaw_deg, 0.0) @ np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
            velocity = np.zeros(3, dtype=np.float32)
            if keys[pygame.K_w]:
                velocity += forward
            if keys[pygame.K_s]:
                velocity -= forward
            if keys[pygame.K_d]:
                velocity += right
            if keys[pygame.K_a]:
                velocity -= right
            if keys[pygame.K_q]:
                velocity -= up
            if keys[pygame.K_e]:
                velocity += up
            norm = float(np.linalg.norm(velocity))
            if norm > 0.0:
                if not camera.topdown_locked:
                    camera.position_cm += velocity / norm * move_speed * dt_sec

            focal_px = (width * 0.5) / math.tan(math.radians(FOV_DEG * 0.5))
            if camera.topdown_locked and poses:
                camera = compute_topdown_camera(points, poses, poses[current_idx])
            screen.fill(BG_COLOR)
            draw_grid(screen, camera, focal_px, width, height, center_xy_cm)

            pose_screen_points: list[tuple[float, float]] = []
            active_pose_screen_points: list[tuple[float, float]] = []
            for pose_idx, pose in enumerate(poses):
                projected = project_point(pose.xyz_cm, camera, focal_px, width, height)
                if projected is not None:
                    pose_screen_points.append((projected[0], projected[1]))
                    if pose_idx <= current_idx:
                        active_pose_screen_points.append((projected[0], projected[1]))
            if len(pose_screen_points) >= 2:
                pygame.draw.lines(screen, PATH_COLOR, False, pose_screen_points, width=2)
            if len(active_pose_screen_points) >= 2:
                pygame.draw.lines(screen, ACTIVE_PATH_COLOR, False, active_pose_screen_points, width=3)

            draw_items: list[tuple[float, tuple[float, float], int, tuple[int, int, int]]] = []
            for point in points:
                projected = project_point(point.xyz_cm, camera, focal_px, width, height)
                if projected is None:
                    continue
                px, py, depth = projected
                if -50.0 <= px <= width + 50.0 and -50.0 <= py <= height + 50.0:
                    is_active_point = point.sample_idx == current_idx
                    radius = 4 if is_active_point else (3 if depth < 1200.0 else 2)
                    color = ACTIVE_POINT_COLOR if is_active_point else POINT_COLOR
                    draw_items.append((depth, (px, py), radius, color))
            draw_items.sort(reverse=True)
            for _, pos, radius, color in draw_items:
                pygame.draw.circle(screen, color, (int(pos[0]), int(pos[1])), radius)

            if poses:
                draw_rover(screen, poses[current_idx], camera, focal_px, width, height)

            hud_lines = [
                f"{run_dir.name}",
                f"points={len(points)} poses={len(poses)}",
                f"scrub={scrub_t * 100.0:5.1f}% frame={current_idx + 1:03d}/{max(1, len(poses)):03d}",
                f"cam=({camera.position_cm[0]:.0f}, {camera.position_cm[1]:.0f}, {camera.position_cm[2]:.0f}) cm",
                f"yaw={camera.yaw_deg:.1f} pitch={camera.pitch_deg:.1f} topdown={'on' if camera.topdown_locked else 'off'}",
                "Move mouse left/right to scrub timeline",
                "Drag LMB to look",
                "V toggle top-down follow",
                "WASD move, Q/E vertical, Shift faster",
            ]
            y = 10
            for line in hud_lines:
                surf = font.render(line, True, TEXT_COLOR)
                screen.blit(surf, (12, y))
                y += 22

            pygame.display.flip()
    finally:
        pygame.quit()


def main() -> None:
    if POINTSCAN_FILE_PATH is not None:
        csv_path = Path(POINTSCAN_FILE_PATH)
        points, poses, run_dir = load_point_scan(csv_path)
        print(f"Loaded point scan: {csv_path}")
    else:
        points, poses, run_dir = collect_point_scan(
            duration_sec=POINTSCAN_DURATION_SEC,
            poll_hz=POINTSCAN_POLL_HZ,
        )
        print(f"Saved point scan: {run_dir / 'points.csv'}")
    run_viewer(points, poses, run_dir)


if __name__ == "__main__":
    main()
