from __future__ import annotations

import math

import numpy as np
import pygame
import pygame.gfxdraw

from main import (
    BG_COLOR,
    GOAL_COLOR,
    LIDAR_HIT_COLOR,
    LOW_CLEARANCE_OBSTACLE_COLOR,
    MAP_BG_COLOR,
    MAP_ZOOM_STEP,
    MAP_MAX_ZOOM,
    MAP_MIN_ZOOM,
    OBSTACLE_COLOR,
    PATH_COLOR,
    TARGET_COLOR,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    CELL_LOW_CLEARANCE_OBSTACLE,
    MapWindow,
    lidar_map_hit_line_color,
    local_to_world_2d,
    planner_clearance_heat_color,
    planner_evidence_heat_color,
)


class CleanMapWindow(MapWindow):
    def __init__(self, planner) -> None:
        super().__init__(planner)
        self.font = pygame.font.SysFont("Segoe UI Semibold", 22)
        self.small_font = pygame.font.SysFont("Segoe UI", 18)
        self.tiny_font = pygame.font.SysFont("Segoe UI", 15)

    @staticmethod
    def _format_distance(distance_cm: float) -> str:
        if not math.isfinite(distance_cm):
            return "--"
        if distance_cm >= 100000.0:
            return f"{distance_cm / 100000.0:.2f} km"
        if distance_cm >= 100.0:
            return f"{distance_cm / 100.0:.1f} m"
        return f"{distance_cm:.0f} cm"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if not math.isfinite(seconds) or seconds < 0.0:
            return "--:--"
        total = int(round(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def _estimate_eta_seconds(
        waypoint_distance_cm: float,
        runtime_elapsed_s: float,
        total_traveled_cm: float,
    ) -> float:
        if runtime_elapsed_s <= 1e-6 or total_traveled_cm <= 1e-6:
            return float("nan")
        avg_speed_cm_s = total_traveled_cm / runtime_elapsed_s
        if avg_speed_cm_s <= 1e-6:
            return float("nan")
        return waypoint_distance_cm / avg_speed_cm_s

    @staticmethod
    def _normalize(x: float, y: float) -> tuple[float, float]:
        length = math.hypot(x, y)
        if length <= 1e-6:
            return (0.0, 0.0)
        return (x / length, y / length)

    @classmethod
    def _rounded_polygon_points(
        cls,
        points: list[tuple[float, float] | tuple[int, int]],
        radius: float,
        arc_steps: int = 10,
    ) -> list[tuple[int, int]]:
        if len(points) < 3:
            return [(int(round(px)), int(round(py))) for px, py in points]

        rounded: list[tuple[int, int]] = []
        count = len(points)
        for idx in range(count):
            prev_pt = points[(idx - 1) % count]
            cur_pt = points[idx]
            next_pt = points[(idx + 1) % count]
            in_vec = (prev_pt[0] - cur_pt[0], prev_pt[1] - cur_pt[1])
            out_vec = (next_pt[0] - cur_pt[0], next_pt[1] - cur_pt[1])
            in_len = math.hypot(*in_vec)
            out_len = math.hypot(*out_vec)
            in_dir = cls._normalize(*in_vec)
            out_dir = cls._normalize(*out_vec)
            dot = max(-0.9999, min(0.9999, in_dir[0] * out_dir[0] + in_dir[1] * out_dir[1]))
            interior_angle = math.acos(dot)
            tangent_dist = radius / math.tan(interior_angle * 0.5)
            max_tangent = min(in_len, out_len) * 0.48
            tangent_dist = min(tangent_dist, max_tangent)
            corner_radius = tangent_dist * math.tan(interior_angle * 0.5)
            if corner_radius <= 1.0:
                rounded.append((int(round(cur_pt[0])), int(round(cur_pt[1]))))
                continue

            start = (cur_pt[0] + in_dir[0] * tangent_dist, cur_pt[1] + in_dir[1] * tangent_dist)
            end = (cur_pt[0] + out_dir[0] * tangent_dist, cur_pt[1] + out_dir[1] * tangent_dist)
            bisector = cls._normalize(in_dir[0] + out_dir[0], in_dir[1] + out_dir[1])
            center_dist = corner_radius / math.sin(interior_angle * 0.5)
            center = (cur_pt[0] + bisector[0] * center_dist, cur_pt[1] + bisector[1] * center_dist)
            start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
            end_angle = math.atan2(end[1] - center[1], end[0] - center[0])
            cross = (start[0] - center[0]) * (end[1] - center[1]) - (start[1] - center[1]) * (end[0] - center[0])
            if cross > 0.0 and end_angle < start_angle:
                end_angle += math.tau
            elif cross < 0.0 and end_angle > start_angle:
                end_angle -= math.tau

            for step in range(arc_steps + 1):
                t = step / arc_steps
                angle = start_angle + (end_angle - start_angle) * t
                px = center[0] + math.cos(angle) * corner_radius
                py = center[1] + math.sin(angle) * corner_radius
                rounded.append((int(round(px)), int(round(py))))
        return rounded

    def _draw_rounded_path(self, path_points: list[tuple[int, int]], width: int) -> None:
        if len(path_points) < 2:
            return
        for start_pt, end_pt in zip(path_points, path_points[1:]):
            pygame.draw.line(self.screen, PATH_COLOR, start_pt, end_pt, width=width)
        radius = max(2, width // 2)
        for point in path_points:
            pygame.draw.circle(self.screen, PATH_COLOR, point, radius)

    def _draw_rounded_arrow(self, planner, rover_xy: tuple[float, float], heading_deg: float) -> None:
        center_x, center_y = self._world_to_screen(planner, rover_xy[0], rover_xy[1])
        icon_scale = 1.0
        triangle_local = [
            (0.0, -28.0),
            (22.0, 18.0),
            (-22.0, 18.0),
        ]
        triangle_screen: list[tuple[float, float]] = []
        heading_rad = math.radians(heading_deg)
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)
        for lx, ly in triangle_local:
            px = (lx * icon_scale) * cos_h - (ly * icon_scale) * sin_h
            py = (lx * icon_scale) * sin_h + (ly * icon_scale) * cos_h
            triangle_screen.append((center_x + px, center_y - py))

        rounded_points = self._rounded_polygon_points(triangle_screen, radius=7.0)
        pygame.gfxdraw.filled_polygon(self.screen, rounded_points, (245, 247, 250))
        pygame.gfxdraw.aapolygon(self.screen, rounded_points, (245, 247, 250))
        pygame.draw.aalines(self.screen, (90, 96, 104), True, rounded_points)

    def _draw_card(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, (22, 22, 23), rect, border_radius=14)
        pygame.draw.rect(self.screen, (74, 77, 82), rect, width=1, border_radius=14)

    def _draw_badge(self, center: tuple[int, int], text: str, fill_color: tuple[int, int, int]) -> None:
        pygame.draw.circle(self.screen, fill_color, center, 14)
        surf = self.tiny_font.render(text, True, (15, 15, 15))
        self.screen.blit(surf, surf.get_rect(center=center))

    def _draw_sidebar(
        self,
        goal_xy: tuple[float, float],
        target_xy: tuple[float, float],
        waypoint_distance_cm: float,
        runtime_elapsed_s: float,
        total_traveled_cm: float,
        obstacle_total: int,
        status: str,
    ) -> None:
        panel = self.panel_rect
        pygame.draw.rect(self.screen, (11, 11, 12), panel, border_radius=18)
        pygame.draw.rect(self.screen, (40, 42, 46), panel, width=1, border_radius=18)

        title = self.font.render("Route", True, (246, 246, 247))
        self.screen.blit(title, (panel.left + 18, panel.top + 16))

        rail_x = panel.left + 30
        goal_y = panel.top + 72
        waypoint_y = panel.top + 150
        pygame.draw.line(self.screen, (72, 74, 78), (rail_x, goal_y), (rail_x, waypoint_y), width=3)

        self._draw_badge((rail_x, goal_y), "G", (245, 205, 74))
        pygame.draw.circle(self.screen, (245, 205, 74), (rail_x, goal_y), 5)
        pygame.draw.circle(self.screen, (18, 102, 173), (rail_x, waypoint_y), 11)
        pygame.draw.circle(self.screen, (39, 129, 214), (rail_x, waypoint_y), 5)

        goal_card = pygame.Rect(panel.left + 52, panel.top + 50, panel.width - 70, 46)
        self._draw_card(goal_card)
        goal_text = self.small_font.render("Goal", True, (235, 238, 242))
        goal_loc = self.tiny_font.render(
            f"{goal_xy[0]:.1f}, {goal_xy[1]:.1f} cm",
            True,
            (148, 152, 160),
        )
        self.screen.blit(goal_text, (goal_card.left + 16, goal_card.top + 8))
        self.screen.blit(goal_loc, (goal_card.left + 16, goal_card.top + 24))

        waypoint_card = pygame.Rect(panel.left + 52, panel.top + 110, panel.width - 70, 152)
        self._draw_card(waypoint_card)
        waypoint_label = self.small_font.render("Waypoint", True, (235, 238, 242))
        self.screen.blit(waypoint_label, (waypoint_card.left + 16, waypoint_card.top + 14))

        eta_seconds = self._estimate_eta_seconds(waypoint_distance_cm, runtime_elapsed_s, total_traveled_cm)
        stats = [
            ("Location", f"{target_xy[0]:.1f}, {target_xy[1]:.1f} cm"),
            ("Distance from rover", self._format_distance(waypoint_distance_cm)),
            ("Navigation time", self._format_duration(eta_seconds)),
        ]
        y = waypoint_card.top + 52
        for label, value in stats:
            label_surf = self.tiny_font.render(label, True, (135, 139, 146))
            value_surf = self.tiny_font.render(value, True, (238, 240, 242))
            self.screen.blit(label_surf, (waypoint_card.left + 16, y))
            value_rect = value_surf.get_rect(midright=(waypoint_card.right - 16, y + 8))
            self.screen.blit(value_surf, value_rect)
            y += 34

        divider_y = waypoint_card.bottom + 18
        pygame.draw.line(
            self.screen,
            (38, 40, 44),
            (panel.left + 16, divider_y),
            (panel.right - 16, divider_y),
            width=1,
        )

        runtime_label = self.tiny_font.render("Run time", True, (135, 139, 146))
        runtime_value = self.small_font.render(self._format_duration(runtime_elapsed_s), True, (238, 240, 242))
        traveled_label = self.tiny_font.render("Traveled", True, (135, 139, 146))
        traveled_value = self.small_font.render(self._format_distance(total_traveled_cm), True, (238, 240, 242))
        obstacle_label = self.tiny_font.render("Obstacles", True, (135, 139, 146))
        obstacle_value = self.small_font.render(str(obstacle_total), True, (238, 240, 242))
        status_label = self.tiny_font.render(status, True, (155, 180, 205))

        footer_top = divider_y + 14
        self.screen.blit(runtime_label, (panel.left + 18, footer_top))
        self.screen.blit(runtime_value, (panel.left + 18, footer_top + 16))
        self.screen.blit(traveled_label, (panel.left + 18, footer_top + 50))
        self.screen.blit(traveled_value, (panel.left + 18, footer_top + 66))
        self.screen.blit(obstacle_label, (panel.left + 18, footer_top + 100))
        self.screen.blit(obstacle_value, (panel.left + 18, footer_top + 116))
        self.screen.blit(status_label, (panel.left + 18, panel.bottom - 28))

    def draw(
        self,
        planner,
        rover_xy: tuple[float, float],
        heading_deg: float,
        goal_xy: tuple[float, float],
        target_xy: tuple[float, float],
        path_world: list[tuple[float, float]],
        status: str,
        goal_distance_cm: float,
        throttle_cmd: float,
        steering_cmd: float,
        waypoint_idx: int,
        waypoint_distance_cm: float,
        waypoint_distance_avg_cm: float,
        obstacle_total: int,
        lidar_cm: np.ndarray,
        lidar_debug_rows: list[dict[str, float | int | bool | str]] | None = None,
        runtime_elapsed_s: float = 0.0,
        total_traveled_cm: float = 0.0,
    ) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.map_rect.collidepoint(event.pos):
                self.dragging_map = True
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging_map = False
            if event.type == pygame.MOUSEMOTION and self.dragging_map:
                self._pan_by_screen_delta(planner, event.rel[0], event.rel[1])
            if event.type == pygame.MOUSEWHEEL:
                mouse_pos = pygame.mouse.get_pos()
                if event.y > 0:
                    self._zoom_at_screen_pos(planner, MAP_ZOOM_STEP ** event.y, mouse_pos)
                elif event.y < 0:
                    self._zoom_at_screen_pos(planner, MAP_ZOOM_STEP ** event.y, mouse_pos)
            if hasattr(pygame, "MULTIGESTURE") and event.type == pygame.MULTIGESTURE:
                if abs(getattr(event, "pinched", 0.0)) > 1e-4:
                    gesture_pos = (
                        int(float(getattr(event, "x", 0.5)) * WINDOW_WIDTH),
                        int(float(getattr(event, "y", 0.5)) * WINDOW_HEIGHT),
                    )
                    self._zoom_at_screen_pos(planner, MAP_ZOOM_STEP ** (event.pinched * 4.0), gesture_pos)

        self.zoom = min(MAP_MAX_ZOOM, max(MAP_MIN_ZOOM, self.zoom))
        self._update_scale(planner)
        self.screen.fill(BG_COLOR)
        pygame.draw.rect(self.screen, MAP_BG_COLOR, self.map_rect, border_radius=18)
        pygame.draw.rect(self.screen, (62, 67, 74), self.map_rect, width=1, border_radius=18)

        map_clip = self.screen.get_clip()
        self.screen.set_clip(self.map_rect)

        for y in range(planner.height_cells):
            for x in range(planner.width_cells):
                clearance_value = float(planner.clearance_grid[y][x])
                clearance_color = planner_clearance_heat_color(planner, clearance_value)
                if clearance_color is None:
                    continue
                cell_min_x = planner.origin_x_cm + x * planner.config.cell_size_cm
                cell_min_y = planner.origin_y_cm + y * planner.config.cell_size_cm
                cell_max_x = cell_min_x + planner.config.cell_size_cm
                cell_max_y = cell_min_y + planner.config.cell_size_cm
                left, top = self._world_to_screen(planner, cell_min_x, cell_max_y)
                right, bottom = self._world_to_screen(planner, cell_max_x, cell_min_y)
                rect = pygame.Rect(min(left, right), min(top, bottom), max(1, abs(right - left)), max(1, abs(bottom - top)))
                surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                surf.fill(clearance_color)
                self.screen.blit(surf, rect.topleft)

        for y in range(planner.height_cells):
            for x in range(planner.width_cells):
                evidence_value = float(planner.evidence_grid[y][x])
                heat_color = planner_evidence_heat_color(planner, evidence_value)
                if heat_color is None:
                    continue
                cell_min_x = planner.origin_x_cm + x * planner.config.cell_size_cm
                cell_min_y = planner.origin_y_cm + y * planner.config.cell_size_cm
                cell_max_x = cell_min_x + planner.config.cell_size_cm
                cell_max_y = cell_min_y + planner.config.cell_size_cm
                left, top = self._world_to_screen(planner, cell_min_x, cell_max_y)
                right, bottom = self._world_to_screen(planner, cell_max_x, cell_min_y)
                rect = pygame.Rect(min(left, right), min(top, bottom), max(1, abs(right - left)), max(1, abs(bottom - top)))
                surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                surf.fill(heat_color)
                self.screen.blit(surf, rect.topleft)

        for y in range(planner.height_cells):
            for x in range(planner.width_cells):
                cell_value = planner.grid[y][x]
                if cell_value == 0:
                    continue
                cell_min_x = planner.origin_x_cm + x * planner.config.cell_size_cm
                cell_min_y = planner.origin_y_cm + y * planner.config.cell_size_cm
                cell_max_x = cell_min_x + planner.config.cell_size_cm
                cell_max_y = cell_min_y + planner.config.cell_size_cm
                left, top = self._world_to_screen(planner, cell_min_x, cell_max_y)
                right, bottom = self._world_to_screen(planner, cell_max_x, cell_min_y)
                rect = pygame.Rect(min(left, right), min(top, bottom), max(1, abs(right - left)), max(1, abs(bottom - top)))
                color = LOW_CLEARANCE_OBSTACLE_COLOR if cell_value == CELL_LOW_CLEARANCE_OBSTACLE else OBSTACLE_COLOR
                pygame.draw.rect(self.screen, color, rect)

        if len(path_world) >= 2:
            path_points = [self._world_to_screen(planner, px, py) for px, py in path_world]
            self._draw_rounded_path(path_points, width=5)

        for row in lidar_debug_rows or []:
            if not bool(row["valid_range_ge0"]):
                continue
            start_px = self._world_to_screen(planner, float(row["sensor_world_x_cm"]), float(row["sensor_world_y_cm"]))
            hit_px = self._world_to_screen(planner, float(row["ray_hit_world_x_cm"]), float(row["ray_hit_world_y_cm"]))
            raw_cm = float(row["raw_cm"])
            hit_color = LIDAR_HIT_COLOR if bool(row["mask_hit"]) else lidar_map_hit_line_color(raw_cm)
            pygame.draw.line(self.screen, hit_color, start_px, hit_px, width=2)

        gx, gy = self._world_to_screen(planner, goal_xy[0], goal_xy[1])
        tx, ty = self._world_to_screen(planner, target_xy[0], target_xy[1])
        pygame.draw.circle(self.screen, GOAL_COLOR, (gx, gy), 8)
        pygame.draw.circle(self.screen, TARGET_COLOR, (tx, ty), 6)
        self._draw_rounded_arrow(planner, rover_xy, heading_deg)

        self.screen.set_clip(map_clip)
        self._draw_sidebar(
            goal_xy=goal_xy,
            target_xy=target_xy,
            waypoint_distance_cm=waypoint_distance_cm,
            runtime_elapsed_s=runtime_elapsed_s,
            total_traveled_cm=total_traveled_cm,
            obstacle_total=obstacle_total,
            status=status,
        )
        pygame.display.flip()
        return True
