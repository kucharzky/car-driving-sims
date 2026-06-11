"""Vehicle kinematics (bicycle model), sensors, and rendering."""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING, Tuple

import cv2
import numpy as np
import pygame

import config

if TYPE_CHECKING:
    from track_manager import TrackManager


class Car:
    """2D kinematic bicycle model with raycasting and vision sensors."""

    def __init__(
        self,
        x: float,
        y: float,
        heading: float = 0.0,
    ) -> None:
        self.x = x
        self.y = y
        self.velocity = 0.0
        self.heading = heading
        self.steer_angle = 0.0
        self._prev_x = x
        self._prev_y = y
        self._ray_distances: list[float] = [1.0] * config.RAY_COUNT
        self._frame_stack: deque[np.ndarray] = deque(maxlen=config.FRAME_STACK)
        self._last_crop_rgb: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def apply_action(self, action: int) -> None:
        """Apply combined steer + throttle action for one simulation step."""
        steer_left = action in (config.ACTION_ACCEL_LEFT, config.ACTION_COAST_LEFT)
        steer_right = action in (config.ACTION_ACCEL_RIGHT, config.ACTION_COAST_RIGHT)
        accelerate = action in (
            config.ACTION_ACCEL_STRAIGHT,
            config.ACTION_ACCEL_LEFT,
            config.ACTION_ACCEL_RIGHT,
        )
        brake = action == config.ACTION_BRAKE

        if steer_left:
            self.steer_angle -= config.STEER_RATE
        elif steer_right:
            self.steer_angle += config.STEER_RATE
        else:
            if self.steer_angle > 0:
                self.steer_angle = max(0.0, self.steer_angle - config.STEER_RATE * 0.5)
            elif self.steer_angle < 0:
                self.steer_angle = min(0.0, self.steer_angle + config.STEER_RATE * 0.5)

        self.steer_angle = max(
            -config.MAX_STEER_ANGLE,
            min(config.MAX_STEER_ANGLE, self.steer_angle),
        )

        if accelerate:
            self.velocity += config.MAX_ACCELERATION
        elif brake:
            self.velocity = max(0.0, self.velocity - config.MAX_BRAKING)
        elif self.velocity > 0:
            self.velocity = max(0.0, self.velocity - config.MAX_BRAKING * 0.15)

        self.velocity = min(config.MAX_SPEED, max(0.0, self.velocity))

    def apply_keyboard(self, keys: pygame.key.ScancodeWrapper) -> None:
        """Update steering and throttle from WASD keys."""
        if keys[pygame.K_a]:
            self.steer_angle -= config.STEER_RATE
        elif keys[pygame.K_d]:
            self.steer_angle += config.STEER_RATE
        else:
            if self.steer_angle > 0:
                self.steer_angle = max(0.0, self.steer_angle - config.STEER_RATE * 0.5)
            elif self.steer_angle < 0:
                self.steer_angle = min(0.0, self.steer_angle + config.STEER_RATE * 0.5)

        self.steer_angle = max(
            -config.MAX_STEER_ANGLE,
            min(config.MAX_STEER_ANGLE, self.steer_angle),
        )

        if keys[pygame.K_w]:
            self.velocity += config.MAX_ACCELERATION
        elif keys[pygame.K_s]:
            self.velocity = max(0.0, self.velocity - config.MAX_BRAKING)
        elif self.velocity > 0:
            self.velocity = max(0.0, self.velocity - config.MAX_BRAKING * 0.15)

        self.velocity = min(config.MAX_SPEED, max(0.0, self.velocity))

    def update(self, dt: float = config.DT) -> None:
        """Advance state using the kinematic bicycle equations."""
        self._prev_x = self.x
        self._prev_y = self.y

        self.x += self.velocity * math.cos(self.heading) * dt * config.FPS
        self.y += self.velocity * math.sin(self.heading) * dt * config.FPS

        if abs(self.velocity) > 1e-4:
            self.heading += (
                (self.velocity / config.WHEELBASE)
                * math.tan(self.steer_angle)
                * dt
                * config.FPS
            )

    def revert_position(self) -> None:
        """Undo the last position update (used on wall collision)."""
        self.x = self._prev_x
        self.y = self._prev_y
        self.velocity = 0.0

    def reset(self, x: float, y: float, heading: float) -> None:
        self.x = x
        self.y = y
        self.heading = heading
        self.velocity = 0.0
        self.steer_angle = 0.0
        self._prev_x = x
        self._prev_y = y
        self._frame_stack.clear()
        self._last_crop_rgb = None

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------

    def update_sensors(self, track: TrackManager) -> None:
        """Refresh ray distances and push a new vision frame onto the stack."""
        self._ray_distances = self.cast_rays(track)
        frame = self.capture_vision_frame(track)
        self._frame_stack.append(frame)

    def cast_rays(self, track: TrackManager) -> list[float]:
        """Cast rays from the nose; return normalized distances in [0, 1]."""
        ox, oy = self.get_nose()
        distances: list[float] = []
        for deg in config.RAY_ANGLES_DEG:
            angle = self.heading + math.radians(deg)
            raw = track.raycast((ox, oy), angle, config.RAY_MAX_DISTANCE)
            distances.append(min(1.0, raw / config.RAY_MAX_DISTANCE))
        return distances

    def get_ray_distances(self) -> list[float]:
        """Latest normalized ray distances."""
        return list(self._ray_distances)

    def _extract_oriented_crop(self, track: TrackManager) -> pygame.Surface:
        """Car-centric top-down crop from the track surface (heading points up)."""
        assert track.surface is not None
        size = config.VISION_CROP_SIZE
        crop = pygame.Surface((size, size))
        crop.fill(config.TRACK_COLOR_WALL)

        half = size // 2
        crop.blit(track.surface, (half - int(self.x), half - int(self.y)))

        rotation = math.degrees(self.heading) - 90.0
        rotated = pygame.transform.rotozoom(crop, -rotation, 1.0)
        centered = pygame.Surface((size, size))
        centered.fill(config.TRACK_COLOR_WALL)
        rect = rotated.get_rect(center=(half, half))
        centered.blit(rotated, rect)
        return centered

    def capture_vision_frame(self, track: TrackManager) -> np.ndarray:
        """OpenCV pipeline: crop → grayscale → 84×84 binary road mask in [0, 1]."""
        crop = self._extract_oriented_crop(track)
        rgb = pygame.surfarray.array3d(crop).transpose(1, 0, 2)
        self._last_crop_rgb = rgb.copy()

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(
            gray,
            (config.VISION_SIZE, config.VISION_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        _, binary = cv2.threshold(resized, 127, 1.0, cv2.THRESH_BINARY)
        return binary.astype(np.float32)

    def get_vision_stack(self) -> np.ndarray:
        """Return stacked frames as (FRAME_STACK, 84, 84) float32."""
        if not self._frame_stack:
            return np.zeros(
                (config.FRAME_STACK, config.VISION_SIZE, config.VISION_SIZE),
                dtype=np.float32,
            )

        frames = list(self._frame_stack)
        while len(frames) < config.FRAME_STACK:
            frames.insert(0, frames[0])
        return np.stack(frames[-config.FRAME_STACK :], axis=0)

    def get_state_vector(self) -> np.ndarray:
        """Concatenated sensor vector: normalized rays (for RL in Phase 3)."""
        return np.array(self._ray_distances, dtype=np.float32)

    @property
    def frame_stack_count(self) -> int:
        return len(self._frame_stack)

    def get_crop_rgb(self) -> np.ndarray | None:
        """Latest oriented crop for debug preview (H, W, 3) RGB."""
        return self._last_crop_rgb

    def get_stack_preview_bgr(self, scale: int = 4) -> np.ndarray | None:
        """2×2 grid of stacked frames for cv2.imshow."""
        stack = self.get_vision_stack()
        if stack.size == 0:
            return None

        tiles = [stack[i] for i in range(config.FRAME_STACK)]
        top = np.hstack([tiles[0], tiles[1]])
        bottom = np.hstack([tiles[2], tiles[3]])
        grid = np.vstack([top, bottom])
        preview = (grid * 255).astype(np.uint8)
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
        edge = config.VISION_SIZE * scale
        return cv2.resize(preview, (edge, edge), interpolation=cv2.INTER_NEAREST)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def get_corners(self) -> list[Tuple[float, float]]:
        """Return the four corners of the car body in world coordinates."""
        half_l = config.CAR_LENGTH / 2
        half_w = config.CAR_WIDTH / 2
        local = [
            (half_l, half_w),
            (half_l, -half_w),
            (-half_l, -half_w),
            (-half_l, half_w),
        ]
        cos_h, sin_h = math.cos(self.heading), math.sin(self.heading)
        corners = []
        for lx, ly in local:
            wx = self.x + lx * cos_h - ly * sin_h
            wy = self.y + lx * sin_h + ly * cos_h
            corners.append((wx, wy))
        return corners

    def get_center(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def get_prev_center(self) -> Tuple[float, float]:
        return (self._prev_x, self._prev_y)

    def get_nose(self) -> Tuple[float, float]:
        """Front bumper position."""
        nx = self.x + (config.CAR_LENGTH / 2) * math.cos(self.heading)
        ny = self.y + (config.CAR_LENGTH / 2) * math.sin(self.heading)
        return (nx, ny)

    def get_prev_nose(self) -> Tuple[float, float]:
        px = self._prev_x + (config.CAR_LENGTH / 2) * math.cos(self.heading)
        py = self._prev_y + (config.CAR_LENGTH / 2) * math.sin(self.heading)
        return (px, py)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, surface: pygame.Surface) -> None:
        corners = self.get_corners()
        points = [(int(px), int(py)) for px, py in corners]
        pygame.draw.polygon(surface, (30, 120, 220), points)
        pygame.draw.polygon(surface, (10, 40, 80), points, 2)

        nose_x, nose_y = self.get_nose()
        pygame.draw.circle(surface, (255, 220, 50), (int(nose_x), int(nose_y)), 4)

    def render_rays(self, surface: pygame.Surface) -> None:
        """Draw LiDAR rays on the main track view."""
        ox, oy = self.get_nose()
        for deg, norm_dist in zip(config.RAY_ANGLES_DEG, self._ray_distances):
            angle = self.heading + math.radians(deg)
            dist = norm_dist * config.RAY_MAX_DISTANCE
            ex = ox + math.cos(angle) * dist
            ey = oy + math.sin(angle) * dist
            color = config.RAY_COLOR_NEAR if norm_dist < 0.35 else config.RAY_COLOR_FAR
            pygame.draw.line(surface, color, (int(ox), int(oy)), (int(ex), int(ey)), 2)
            pygame.draw.circle(surface, color, (int(ex), int(ey)), 4)
