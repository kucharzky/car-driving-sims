"""Vehicle kinematics (bicycle model) and rendering."""

from __future__ import annotations

import math
from typing import Tuple

import pygame

import config


class Car:
    """2D kinematic bicycle model with keyboard input support."""

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

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def apply_keyboard(self, keys: pygame.key.ScancodeWrapper) -> None:
        """Update steering and throttle from WASD keys."""
        if keys[pygame.K_a]:
            self.steer_angle -= config.STEER_RATE
        elif keys[pygame.K_d]:
            self.steer_angle += config.STEER_RATE
        else:
            # Return steering toward center
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
            self.velocity -= config.MAX_BRAKING
        else:
            # Light friction
            if self.velocity > 0:
                self.velocity = max(0.0, self.velocity - config.MAX_BRAKING * 0.15)
            elif self.velocity < 0:
                self.velocity = min(0.0, self.velocity + config.MAX_BRAKING * 0.15)

        self.velocity = max(
            -config.MAX_REVERSE_SPEED,
            min(config.MAX_SPEED, self.velocity),
        )

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

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, surface: pygame.Surface) -> None:
        corners = self.get_corners()
        points = [(int(px), int(py)) for px, py in corners]
        pygame.draw.polygon(surface, (30, 120, 220), points)
        pygame.draw.polygon(surface, (10, 40, 80), points, 2)

        # Nose marker
        nose_x = self.x + (config.CAR_LENGTH / 2) * math.cos(self.heading)
        nose_y = self.y + (config.CAR_LENGTH / 2) * math.sin(self.heading)
        pygame.draw.circle(surface, (255, 220, 50), (int(nose_x), int(nose_y)), 4)
