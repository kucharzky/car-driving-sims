"""Gym-style environment: pygame wrapper, collisions, rewards, and observations."""

from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np
import pygame

import config
from car import Car
from track_manager import LapTracker, TrackManager


def check_collision(car: Car, track: TrackManager) -> bool:
    """Return True if any car corner intersects a wall pixel."""
    for px, py in car.get_corners():
        if track.is_wall(px, py):
            return True
    return False


class Environment:
    """RL environment wrapping track, car physics, sensors, and lap logic."""

    def __init__(
        self,
        track_source: str | None = None,
        render: bool = False,
    ) -> None:
        pygame.init()
        self.track = TrackManager()
        source = track_source or str(config.DEFAULT_TRACK_PATH)
        try:
            self.track.load(source)
        except FileNotFoundError:
            self.track.load("procedural")

        self.render_mode = render
        self.screen: pygame.Surface | None = None
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)

        if render:
            self.screen = pygame.display.set_mode(
                (config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
            )

        self.car = Car(0.0, 0.0, 0.0)
        self.lap_tracker = LapTracker(self.track, 0.0)
        self.steps = 0
        self.episode_reward = 0.0
        self.max_progress_pct = 0.0
        self._prev_progress_pct = 0.0

    def _warmup_sensors(self) -> None:
        for _ in range(config.FRAME_STACK):
            self.car.update_sensors(self.track)

    def get_observation(self) -> np.ndarray:
        """CNN input: (FRAME_STACK, VISION_SIZE, VISION_SIZE) float32."""
        return self.car.get_vision_stack()

    def get_rays(self) -> np.ndarray:
        """Normalized LiDAR distances (RAY_COUNT,) float32."""
        return np.array(self.car.get_ray_distances(), dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Reset episode and return initial observation."""
        sx, sy, heading = self.track.get_spawn()
        self.car.reset(sx, sy, heading)
        self.lap_tracker.reset(heading)
        self.steps = 0
        self.episode_reward = 0.0
        self.max_progress_pct = 0.0
        self._prev_progress_pct = 0.0
        self._warmup_sensors()
        return self.get_observation()

    def _step_distance(self) -> float:
        return math.hypot(self.car.x - self.car._prev_x, self.car.y - self.car._prev_y)

    def _compute_reward(self, crashed: bool, lap_completed: bool) -> float:
        rays = self.car.get_ray_distances()
        min_ray = min(rays) if rays else 1.0
        center_ray = rays[len(rays) // 2] if rays else 1.0
        step_dist = self._step_distance()
        on_road = self.track.is_road(self.car.x, self.car.y)

        reward = 0.0

        # Reward actual path distance on road (works on curves, unlike race-axis projection)
        if on_road and self.car.velocity > 0.1 and step_dist > 0.01:
            reward += config.REWARD_PATH_SCALE * step_dist

        # Discourage hugging walls — uses LiDAR the agent also sees in state
        if min_ray < config.WALL_PROXIMITY_THRESHOLD:
            closeness = 1.0 - (min_ray / config.WALL_PROXIMITY_THRESHOLD)
            reward -= config.REWARD_WALL_PROXIMITY * closeness * closeness

        # Reward making lap progress (dense signal toward completing a circuit)
        progress = self.lap_tracker.progress_pct
        progress_delta = progress - self._prev_progress_pct
        if progress_delta > 0:
            reward += config.REWARD_PROGRESS_DELTA * progress_delta
        self._prev_progress_pct = progress
        self.max_progress_pct = max(self.max_progress_pct, progress)

        # Encourage maintaining speed when path is clear
        if center_ray > config.CLEAR_RAY_THRESHOLD and self.car.velocity > 0.5:
            reward += config.REWARD_SPEED_CLEAR * (self.car.velocity / config.MAX_SPEED)

        # Penalize stopping when nothing is blocking ahead
        if center_ray > config.CLEAR_RAY_THRESHOLD and self.car.velocity < 0.2:
            reward -= config.REWARD_IDLE

        if not on_road:
            reward -= config.REWARD_OFF_ROAD

        if lap_completed:
            reward += config.REWARD_LAP
        if crashed:
            reward += config.REWARD_CRASH

        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Apply action and return (observation, reward, done, info)."""
        self.car.apply_action(action)
        self.car.update()

        crashed = check_collision(self.car, self.track)
        if crashed:
            self.car.revert_position()

        self.car.update_sensors(self.track)
        lap_completed = self.lap_tracker.update(self.car)

        reward = self._compute_reward(crashed, lap_completed)
        self.steps += 1
        self.episode_reward += reward

        done = crashed or self.steps >= config.MAX_EPISODE_STEPS
        obs = self.get_observation()
        info = {
            "crashed": crashed,
            "lap_completed": lap_completed,
            "laps": self.lap_tracker.laps,
            "steps": self.steps,
            "episode_reward": self.episode_reward,
            "max_progress_pct": self.max_progress_pct,
            "velocity": self.car.velocity,
            "min_ray": min(self.car.get_ray_distances()),
            "rays": self.get_rays(),
        }
        return obs, reward, done, info

    def render(self, show_rays: bool = False) -> None:
        if self.screen is None:
            return

        self.screen.fill((40, 40, 40))
        self.track.render(self.screen)
        if show_rays:
            self.car.render_rays(self.screen)
        self.car.render(self.screen)

        hud = [
            f"Steps: {self.steps}  Reward: {self.episode_reward:.1f}  "
            f"Laps: {self.lap_tracker.laps}  Speed: {self.car.velocity:.2f}",
            f"Lap progress: {self.lap_tracker.progress_pct:.0f}%  "
            f"max:{self.max_progress_pct:.0f}%  "
            f"{'ARMED' if self.lap_tracker.armed else 'start'}",
        ]
        for i, line in enumerate(hud):
            text = self.font.render(line, True, (230, 230, 230))
            self.screen.blit(text, (10, 10 + i * 20))

        pygame.display.flip()
        self.clock.tick(config.FPS)

    def close(self) -> None:
        if self.render_mode:
            pygame.quit()
