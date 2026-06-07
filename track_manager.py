"""Track loading via PNG image parsing or procedural spline generation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pygame

import config


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
        self._spawn = self._compute_spawn_from_mask()

    def _load_procedural(self) -> None:
        self.source = "procedural"
        self._ensure_display()
        self.width = config.SCREEN_WIDTH
        self.height = config.SCREEN_HEIGHT
        self.surface = pygame.Surface((self.width, self.height))
        self.surface.fill(config.TRACK_COLOR_WALL)
        self._draw_procedural_track(self.surface)
        self._pixels = pygame.surfarray.array3d(self.surface).transpose(1, 0, 2)
        self._build_masks()
        self._spawn = self._compute_spawn_from_mask()

    # ------------------------------------------------------------------
    # Mask building & queries
    # ------------------------------------------------------------------

    def _color_match(self, pixel: np.ndarray, target: Tuple[int, int, int]) -> bool:
        diff = np.abs(pixel.astype(int) - np.array(target, dtype=int))
        return bool(np.all(diff <= config.COLOR_TOLERANCE))

    def _build_masks(self) -> None:
        assert self._pixels is not None
        h, w, _ = self._pixels.shape
        road = np.zeros((h, w), dtype=bool)
        wall = np.zeros((h, w), dtype=bool)
        start = np.zeros((h, w), dtype=bool)

        road_target = np.array(config.TRACK_COLOR_ROAD, dtype=int)
        wall_target = np.array(config.TRACK_COLOR_WALL, dtype=int)
        start_target = np.array(config.TRACK_COLOR_START_FINISH, dtype=int)
        tol = config.COLOR_TOLERANCE

        diff_road = np.max(np.abs(self._pixels.astype(int) - road_target), axis=2)
        diff_wall = np.max(np.abs(self._pixels.astype(int) - wall_target), axis=2)
        diff_start = np.max(np.abs(self._pixels.astype(int) - start_target), axis=2)

        road = diff_road <= tol
        wall = diff_wall <= tol
        start = diff_start <= tol

        self._road_mask = road
        self._wall_mask = wall
        self._start_mask = start

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

    def crossed_start_finish(
        self,
        prev: Tuple[float, float],
        curr: Tuple[float, float],
    ) -> bool:
        """Detect lap crossing: was off the line before, now on it."""
        px, py = prev
        cx, cy = curr
        was_on = self.is_start_finish(px, py)
        now_on = self.is_start_finish(cx, cy)
        return (not was_on) and now_on

    # ------------------------------------------------------------------
    # Procedural track generation
    # ------------------------------------------------------------------

    def _draw_procedural_track(self, surface: pygame.Surface) -> None:
        rng = np.random.default_rng()
        cx, cy = self.width / 2, self.height / 2
        margin = config.PROCEDURAL_MARGIN
        n = config.PROCEDURAL_CONTROL_POINTS

        angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
        radii = rng.uniform(margin, min(cx, cy) - margin / 2, size=n)
        points = [
            (
                int(cx + r * math.cos(a) + rng.normal(0, 20)),
                int(cy + r * math.sin(a) + rng.normal(0, 20)),
            )
            for a, r in zip(angles, radii)
        ]

        centerline = self._sample_closed_spline(points, samples=400)
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

        pygame.draw.polygon(surface, config.TRACK_COLOR_ROAD, outer_pts)
        pygame.draw.polygon(surface, config.TRACK_COLOR_WALL, inner_pts)

        # Start/finish line across track at first centerline point
        sx, sy = centerline[0]
        x0, y0 = centerline[-1]
        x1, y1 = centerline[1]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        p1 = (int(sx + nx * half_w), int(sy + ny * half_w))
        p2 = (int(sx - nx * half_w), int(sy - ny * half_w))
        pygame.draw.line(surface, config.TRACK_COLOR_START_FINISH, p1, p2, 4)

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

    def _compute_spawn_from_mask(self) -> Tuple[float, float, float]:
        assert self._start_mask is not None
        ys, xs = np.where(self._start_mask)
        if len(xs) == 0:
            # Fallback: center of road mask
            assert self._road_mask is not None
            ys, xs = np.where(self._road_mask)
            if len(xs) == 0:
                return (self.width / 2, self.height / 2, 0.0)
            idx = len(xs) // 2
            return (float(xs[idx]), float(ys[idx]), 0.0)

        cx = float(xs.mean())
        cy = float(ys.mean())

        # Estimate heading from nearby road pixels
        assert self._road_mask is not None
        theta = 0.0
        radius = 30
        best_count = 0
        for angle_deg in range(0, 360, 10):
            rad = math.radians(angle_deg)
            dx, dy = math.cos(rad), math.sin(rad)
            count = 0
            for dist in range(5, radius, 3):
                px = int(cx + dx * dist)
                py = int(cy + dy * dist)
                if self._in_bounds(px, py) and self._road_mask[py, px]:
                    count += 1
            if count > best_count:
                best_count = count
                theta = rad

        return (cx, cy, theta)

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
