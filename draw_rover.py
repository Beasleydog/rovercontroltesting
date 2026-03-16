from __future__ import annotations

import math

try:
    import pygame
    import pygame.gfxdraw
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "pygame is required for draw_rover.py. Install with: pip install -r requirements.txt"
    ) from exc


WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700
BACKGROUND_COLOR = (18, 18, 18)
ROVER_FILL = (248, 248, 248)
ROVER_OUTLINE = (214, 214, 214)
def rotate_point(x: float, y: float, heading_deg: float) -> tuple[float, float]:
    rad = math.radians(heading_deg)
    cos_h = math.cos(rad)
    sin_h = math.sin(rad)
    return (x * cos_h - y * sin_h, x * sin_h + y * cos_h)


def _normalize(x: float, y: float) -> tuple[float, float]:
    length = math.hypot(x, y)
    if length <= 1e-6:
        return (0.0, 0.0)
    return (x / length, y / length)


def rounded_polygon_points(
    points: list[tuple[float, float]],
    radius: float,
    arc_steps: int = 12,
) -> list[tuple[int, int]]:
    if len(points) < 3:
        return [(int(round(x)), int(round(y))) for x, y in points]

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
        in_dir = _normalize(*in_vec)
        out_dir = _normalize(*out_vec)
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
        bisector = _normalize(in_dir[0] + out_dir[0], in_dir[1] + out_dir[1])
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


def draw_triangle_rover(
    screen: pygame.Surface,
    center: tuple[int, int],
    heading_deg: float,
    scale: float = 1.0,
) -> None:
    cx, cy = center
    local_points = [
        (0.0, -150.0),
        (116.0, 92.0),
        (-116.0, 92.0),
    ]

    points: list[tuple[float, float]] = []
    for px, py in local_points:
        rx, ry = rotate_point(px * scale, py * scale, heading_deg)
        points.append((cx + rx, cy + ry))

    rounded_points = rounded_polygon_points(points, radius=26.0 * scale)
    pygame.gfxdraw.filled_polygon(screen, rounded_points, ROVER_FILL)
    pygame.gfxdraw.aapolygon(screen, rounded_points, ROVER_FILL)
    pygame.draw.aalines(screen, ROVER_OUTLINE, True, rounded_points)


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Rover Shape Test")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Segoe UI", 22)

    heading_deg = 0.0
    scale = 1.0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_LEFT:
                    heading_deg += 8.0
                elif event.key == pygame.K_RIGHT:
                    heading_deg -= 8.0
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    scale = min(2.5, scale + 0.05)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    scale = max(0.25, scale - 0.05)

        screen.fill(BACKGROUND_COLOR)
        draw_triangle_rover(
            screen=screen,
            center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2),
            heading_deg=heading_deg,
            scale=scale,
        )

        heading_text = font.render(f"Heading: {heading_deg:.0f} deg", True, (235, 235, 235))
        scale_text = font.render(f"Scale: {scale:.2f}", True, (235, 235, 235))
        help_text = font.render("Left/Right rotate   +/- scale   Esc quit", True, (180, 180, 180))
        screen.blit(heading_text, (24, 22))
        screen.blit(scale_text, (24, 52))
        screen.blit(help_text, (24, WINDOW_HEIGHT - 42))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
