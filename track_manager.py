"""Track loading via PNG image parsing or procedural spline generation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from car import Car

import numpy as np
import pygame

import config


class LapTracker:
    """Anti-exploit lap counter: requires full-circuit distance, angle, and far-point proof."""

    def __init__(self, track: "TrackManager", reference_heading: float) -> None:
        self._track = track
        self.reference_heading = reference_heading
        self.laps = 0
        self._armed = False
        self._was_in_zone = False
        self._distance_outside = 0.0
        self._distance_before_arm = 0.0
        self._angle_progress = 0.0
        self._last_center_angle: float | None = None
        self._max_far_from_finish = 0.0
        self._exited_zone_since_arm = False
        self._lap_distance_required = track.get_lap_distance_required()
        self._lap_far_required = track.get_lap_far_distance_required()

    def reset(self, reference_heading: float | None = None) -> None:
        if reference_heading is not None:
            self.reference_heading = reference_heading
        self.laps = 0
        self._armed = False
        self._was_in_zone = False
        self._reset_lap_progress()

    def _reset_lap_progress(self) -> None:
        self._distance_outside = 0.0
        self._distance_before_arm = 0.0
        self._angle_progress = 0.0
        self._last_center_angle = None
        self._max_far_from_finish = 0.0
        self._exited_zone_since_arm = False

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return (a - b + math.pi) % (2 * math.pi) - math.pi

    def _heading_matches(self, heading: float) -> bool:
        if abs(self._angle_diff(heading, self.reference_heading)) > config.LAP_HEADING_TOLERANCE:
            return False
        hx, hy = math.cos(heading), math.sin(heading)
        rx, ry = self._track.race_direction
        return hx * rx + hy * ry >= config.MIN_FORWARD_DOT

    def _is_forward_entry(self, car: Car) -> bool:
        dx = car.x - car._prev_x
        dy = car.y - car._prev_y
        move_len = math.hypot(dx, dy)
        if move_len < 0.05:
            return car.velocity >= 0.0
        mx, my = dx / move_len, dy / move_len
        rx, ry = self._track.race_direction
        return mx * rx + my * ry >= config.MIN_FORWARD_DOT

    def _lap_requirements_met(self) -> bool:
        return (
            self._distance_outside >= self._lap_distance_required
            and self._angle_progress >= config.MIN_LAP_ANGLE_PROGRESS
            and self._max_far_from_finish >= self._lap_far_required
        )

    @property
    def distance_outside(self) -> float:
        return self._distance_outside

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def progress_pct(self) -> float:
        dist_req = max(self._lap_distance_required, 1.0)
        angle_req = max(config.MIN_LAP_ANGLE_PROGRESS, 0.01)
        far_req = max(self._lap_far_required, 1.0)
        return min(
            100.0,
            100.0
            * (
                0.4 * min(1.0, self._distance_outside / dist_req)
                + 0.35 * min(1.0, self._angle_progress / angle_req)
                + 0.25 * min(1.0, self._max_far_from_finish / far_req)
            ),
        )

    def update(self, car: Car) -> bool:
        nose = car.get_nose()
        center = car.get_center()
        in_zone = self._track.in_finish_zone(*nose) or self._track.in_finish_zone(*center)

        step_dist = math.hypot(car.x - car._prev_x, car.y - car._prev_y)

        if not in_zone:
            self._distance_before_arm += step_dist
            if not self._armed and self._distance_before_arm >= config.MIN_CLEAR_START_DISTANCE:
                self._armed = True
                self._distance_outside = 0.0
                self._angle_progress = 0.0
                self._last_center_angle = None
                self._max_far_from_finish = 0.0

            if self._armed:
                self._distance_outside += step_dist
                center_angle = self._track.angle_from_center(car.x, car.y)
                if self._last_center_angle is not None:
                    self._angle_progress += abs(
                        self._angle_diff(center_angle, self._last_center_angle)
                    )
                self._last_center_angle = center_angle
                self._max_far_from_finish = max(
                    self._max_far_from_finish,
                    self._track.distance_from_finish_center(car.x, car.y),
                )

        exited_zone = not in_zone and self._was_in_zone
        if exited_zone:
            self._exited_zone_since_arm = True

        entered_zone = in_zone and not self._was_in_zone
        completed = False

        if (
            entered_zone
            and self._armed
            and self._exited_zone_since_arm
            and self._heading_matches(car.heading)
            and self._is_forward_entry(car)
            and self._lap_requirements_met()
        ):
            self.laps += 1
            completed = True
            self._reset_lap_progress()
            self._exited_zone_since_arm = False

        self._was_in_zone = in_zone
        return completed


class TrackManager:
    """Loads, queries, and renders a 2D race track."""

    def __init__(self) -> None:
        self.source: str = "png"
        self.surface: pygame.Surface | None = None
        self.width: int = 0
        self.height: int = 0
        self._pixels: np.ndarray | None = None  # (H, W, 3) RGB
        self._road_mask: np.ndarray | None = None
        self._wall_mask: np.ndarray | None = None
        self._start_mask: np.ndarray | None = None
        self._spawn: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._finish_p1: Tuple[float, float] | None = None
        self._finish_p2: Tuple[float, float] | None = None
        self._finish_center: Tuple[float, float] = (0.0, 0.0)
        self._race_direction: Tuple[float, float] = (1.0, 0.0)
        self._track_center: Tuple[float, float] = (0.0, 0.0)
        self._circuit_radius: float = 300.0
        self._circuit_length: float = 1800.0
        self._track_span: float = 600.0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, source: str | Path | None = None) -> None:
        """Load a track from a PNG path or generate one procedurally."""
        if source is None:
            source = config.DEFAULT_TRACK_PATH

        source_str = str(source).lower()
        if source_str in ("procedural", "gen", "random"):
            self._load_procedural()
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(
                    f"Track image not found: {path}. "
                    "Draw a track or run with --track procedural."
                )
            self._load_png(path)

    def _ensure_display(self) -> None:
        """Allow surface.convert() when no game window exists yet."""
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((1, 1))

    def _load_png(self, path: Path) -> None:
        self.source = "png"
        self._ensure_display()
        self.surface = pygame.image.load(str(path)).convert()
        self.width, self.height = self.surface.get_size()
        self._pixels = pygame.surfarray.array3d(self.surface).transpose(1, 0, 2)
        self._build_masks()
        self._compute_track_metrics()
        self._finish_p1, self._finish_p2 = self._compute_finish_line_from_mask()
        self._finish_center = (
            (self._finish_p1[0] + self._finish_p2[0]) / 2,
            (self._finish_p1[1] + self._finish_p2[1]) / 2,
        )
        self._spawn = self._compute_spawn_from_mask()

    def _load_procedural(self) -> None:
        self.source = "procedural"
        self._ensure_display()
        self.width = config.SCREEN_WIDTH
        self.height = config.SCREEN_HEIGHT
        self.surface = pygame.Surface((self.width, self.height))
        self.surface.fill(config.TRACK_COLOR_WALL)
        centerline, finish_pts = self._draw_procedural_track(self.surface)
        self._pixels = pygame.surfarray.array3d(self.surface).transpose(1, 0, 2)
        self._build_masks()
        self._compute_track_metrics()
        self._finish_p1, self._finish_p2 = finish_pts
        self._finish_center = (
            (self._finish_p1[0] + self._finish_p2[0]) / 2,
            (self._finish_p1[1] + self._finish_p2[1]) / 2,
        )
        self._spawn = self._compute_spawn_from_centerline(centerline, finish_pts)

    # ------------------------------------------------------------------
    # Mask building & queries
    # ------------------------------------------------------------------

    def _build_masks(self) -> None:
        assert self._pixels is not None
        h, w, _ = self._pixels.shape

        road_target = np.array(config.TRACK_COLOR_ROAD, dtype=int)
        wall_target = np.array(config.TRACK_COLOR_WALL, dtype=int)
        start_target = np.array(config.TRACK_COLOR_START_FINISH, dtype=int)
        tol = config.COLOR_TOLERANCE

        diff_road = np.max(np.abs(self._pixels.astype(int) - road_target), axis=2)
        diff_wall = np.max(np.abs(self._pixels.astype(int) - wall_target), axis=2)
        diff_start = np.max(np.abs(self._pixels.astype(int) - start_target), axis=2)

        self._road_mask = diff_road <= tol
        self._wall_mask = diff_wall <= tol
        self._start_mask = diff_start <= tol

    def _compute_track_metrics(self) -> None:
        """Estimate circuit size for lap anti-exploit thresholds."""
        assert self._road_mask is not None
        ys, xs = np.where(self._road_mask)
        if len(xs) == 0:
            self._track_center = (self.width / 2, self.height / 2)
            self._circuit_radius = min(self.width, self.height) * 0.3
        else:
            self._track_center = (float(xs.mean()), float(ys.mean()))
            dists = np.hypot(xs - self._track_center[0], ys - self._track_center[1])
            self._circuit_radius = float(np.percentile(dists, 70))

        self._circuit_length = 2 * math.pi * self._circuit_radius
        self._track_span = self._circuit_radius * 2

    @property
    def race_direction(self) -> Tuple[float, float]:
        return self._race_direction

    def get_lap_distance_required(self) -> float:
        return max(
            config.MIN_LAP_DISTANCE_FLOOR,
            self._circuit_length * config.LAP_DISTANCE_FRACTION,
        )

    def get_lap_far_distance_required(self) -> float:
        return self._track_span * config.LAP_FAR_DISTANCE_FRACTION

    def angle_from_center(self, x: float, y: float) -> float:
        cx, cy = self._track_center
        return math.atan2(y - cy, x - cx)

    def distance_from_finish_center(self, x: float, y: float) -> float:
        fx, fy = self._finish_center
        return math.hypot(x - fx, y - fy)

    def _in_bounds(self, x: float, y: float) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _mask_at(self, mask: np.ndarray, x: float, y: float) -> bool:
        if not self._in_bounds(x, y):
            return True  # off-screen treated as wall
        return bool(mask[int(y), int(x)])

    def is_wall(self, x: float, y: float) -> bool:
        assert self._wall_mask is not None
        return self._mask_at(self._wall_mask, x, y)

    def is_road(self, x: float, y: float) -> bool:
        assert self._road_mask is not None
        return self._mask_at(self._road_mask, x, y)

    def is_start_finish(self, x: float, y: float) -> bool:
        assert self._start_mask is not None
        return self._mask_at(self._start_mask, x, y)

    def get_spawn(self) -> Tuple[float, float, float]:
        """Return spawn position and heading (x, y, theta radians)."""
        return self._spawn

    def in_finish_zone(self, x: float, y: float) -> bool:
        """True when the car touches the green line or its proximity band."""
        if self.is_start_finish(x, y):
            return True
        if self._finish_p1 is None or self._finish_p2 is None:
            return False
        return (
            self._distance_to_segment(x, y, self._finish_p1, self._finish_p2)
            <= config.FINISH_ZONE_RADIUS
        )

    @staticmethod
    def _distance_to_segment(
        px: float,
        py: float,
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> float:
        ax, ay = a
        bx, by = b
        abx, aby = bx - ax, by - ay
        ab_len_sq = abx * abx + aby * aby
        if ab_len_sq < 1e-6:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_len_sq))
        cx, cy = ax + t * abx, ay + t * aby
        return math.hypot(px - cx, py - cy)

    # ------------------------------------------------------------------
    # Finish line & spawn
    # ------------------------------------------------------------------

    def _compute_finish_line_from_mask(
        self,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        assert self._start_mask is not None
        ys, xs = np.where(self._start_mask)
        if len(xs) < 2:
            return ((0.0, 0.0), (1.0, 0.0))

        coords = np.column_stack([xs.astype(float), ys.astype(float)])
        mean = coords.mean(axis=0)
        centered = coords - mean
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        projections = centered @ axis
        p1 = (float(coords[int(np.argmin(projections))][0]), float(coords[int(np.argmin(projections))][1]))
        p2 = (float(coords[int(np.argmax(projections))][0]), float(coords[int(np.argmax(projections))][1]))
        return (p1, p2)

    def _heading_from_centerline(
        self,
        centerline: list[Tuple[float, float]],
        index: int,
    ) -> float:
        x0, y0 = centerline[index - 1]
        x1, y1 = centerline[(index + 1) % len(centerline)]
        return math.atan2(y1 - y0, x1 - x0)

    def _compute_spawn_from_centerline(
        self,
        centerline: list[Tuple[float, float]],
        finish_pts: Tuple[Tuple[float, float], Tuple[float, float]],
    ) -> Tuple[float, float, float]:
        self._finish_p1, self._finish_p2 = finish_pts
        hint = self._heading_from_centerline(centerline, 0)
        return self._resolve_spawn(finish_pts, hint_heading=hint)

    def _compute_spawn_from_mask(self) -> Tuple[float, float, float]:
        assert self._road_mask is not None

        p1, p2 = self._finish_p1, self._finish_p2
        if p1 is None or p2 is None:
            p1, p2 = self._compute_finish_line_from_mask()
        self._finish_p1, self._finish_p2 = p1, p2
        return self._resolve_spawn((p1, p2))

    def _resolve_spawn(
        self,
        finish_pts: Tuple[Tuple[float, float], Tuple[float, float]],
        hint_heading: float | None = None,
    ) -> Tuple[float, float, float]:
        """Derive spawn (before line), heading, and race direction from finish geometry."""
        p1, p2 = finish_pts
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2
        lx, ly = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(lx, ly) or 1.0
        tx, ty = lx / length, ly / length
        candidates = [(-ty, tx), (ty, -tx)]

        best: Tuple[float, float, float] | None = None
        best_score = -1.0

        for nx, ny in candidates:
            sx = mx - nx * config.SPAWN_OFFSET_BEFORE_LINE
            sy = my - ny * config.SPAWN_OFFSET_BEFORE_LINE
            ahead = (mx + nx * 80, my + ny * 80)

            if not self.is_road(sx, sy) or not self.is_road(*ahead):
                continue

            heading = math.atan2(ny, nx)
            score = 1.0
            if hint_heading is not None:
                diff = abs((heading - hint_heading + math.pi) % (2 * math.pi) - math.pi)
                score += max(0.0, 1.5 - diff)

            if score > best_score:
                best_score = score
                self._race_direction = (nx, ny)
                best = (sx, sy, heading)

        if best is not None:
            return best

        # Fallback: center of road near line midpoint
        heading = hint_heading if hint_heading is not None else 0.0
        return (mx, my, heading)

    # ------------------------------------------------------------------
    # Procedural track generation
    # ------------------------------------------------------------------

    def _draw_procedural_track(
        self,
        surface: pygame.Surface,
    ) -> Tuple[list[Tuple[float, float]], Tuple[Tuple[float, float], Tuple[float, float]]]:
        rng = np.random.default_rng()
        cx, cy = self.width / 2, self.height / 2
        margin = config.PROCEDURAL_MARGIN
        n = config.PROCEDURAL_CONTROL_POINTS

        base_rx = min(cx, cy) - margin
        base_ry = base_rx * 0.72
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
        points = [
            (
                int(cx + base_rx * math.cos(a) + rng.normal(0, 18)),
                int(cy + base_ry * math.sin(a) + rng.normal(0, 18)),
            )
            for a in angles
        ]

        centerline = self._sample_closed_spline(points, samples=480)
        half_w = config.TRACK_WIDTH / 2

        outer_pts: list[Tuple[int, int]] = []
        inner_pts: list[Tuple[int, int]] = []

        for i, (x, y) in enumerate(centerline):
            x0, y0 = centerline[i - 1]
            x1, y1 = centerline[(i + 1) % len(centerline)]
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / length, dx / length
            outer_pts.append((int(x + nx * half_w), int(y + ny * half_w)))
            inner_pts.append((int(x - nx * half_w), int(y - ny * half_w)))

        # Single ring polygon: outer boundary forward, inner boundary reversed
        ring = outer_pts + inner_pts[::-1]
        if len(ring) >= 3:
            pygame.draw.polygon(surface, config.TRACK_COLOR_ROAD, ring)

        # Start/finish line across track (inner wall to outer wall)
        sx, sy = centerline[0]
        x0, y0 = centerline[-1]
        x1, y1 = centerline[1]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        p1 = (int(sx - nx * half_w), int(sy - ny * half_w))
        p2 = (int(sx + nx * half_w), int(sy + ny * half_w))
        pygame.draw.line(
            surface,
            config.TRACK_COLOR_START_FINISH,
            p1,
            p2,
            config.START_FINISH_LINE_WIDTH,
        )

        return centerline, (p1, p2)

    @staticmethod
    def _sample_closed_spline(
        control_points: list[Tuple[int, int]],
        samples: int = 300,
    ) -> list[Tuple[float, float]]:
        """Catmull-Rom spline through closed control polygon."""
        pts = control_points
        n = len(pts)
        if n < 3:
            return [(float(x), float(y)) for x, y in pts]

        result: list[Tuple[float, float]] = []
        for i in range(n):
            p0 = pts[(i - 1) % n]
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            p3 = pts[(i + 2) % n]
            for t in np.linspace(0, 1, samples // n, endpoint=False):
                t2, t3 = t * t, t * t * t
                x = 0.5 * (
                    (2 * p1[0])
                    + (-p0[0] + p2[0]) * t
                    + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                    + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
                )
                y = 0.5 * (
                    (2 * p1[1])
                    + (-p0[1] + p2[1]) * t
                    + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                    + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
                )
                result.append((x, y))
        return result

    # ------------------------------------------------------------------
    # Rendering & raycasting
    # ------------------------------------------------------------------

    def render(self, surface: pygame.Surface) -> None:
        if self.surface is not None:
            surface.blit(self.surface, (0, 0))

    def raycast(
        self,
        origin: Tuple[float, float],
        angle_rad: float,
        max_distance: float = config.RAY_MAX_DISTANCE,
    ) -> float:
        """Return distance from origin to nearest wall along angle_rad."""
        ox, oy = origin
        step = 2.0
        dist = 0.0
        while dist < max_distance:
            x = ox + math.cos(angle_rad) * dist
            y = oy + math.sin(angle_rad) * dist
            if not self._in_bounds(x, y) or self.is_wall(x, y):
                return dist
            dist += step
        return max_distance
