"""Smoke test for Phase 2 raycasting and vision pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pygame

import config
from car import Car
from track_manager import TrackManager


def main() -> None:
    pygame.init()
    track = TrackManager()
    track.load("assets/track.png")
    sx, sy, heading = track.get_spawn()
    car = Car(sx, sy, heading)

    for _ in range(config.FRAME_STACK + 2):
        car.update_sensors(track)

    rays = car.get_ray_distances()
    stack = car.get_vision_stack()

    assert len(rays) == config.RAY_COUNT, rays
    assert all(0.0 <= r <= 1.0 for r in rays), rays
    assert stack.shape == (config.FRAME_STACK, config.VISION_SIZE, config.VISION_SIZE), stack.shape
    assert stack.dtype.name == "float32"
    assert stack.max() <= 1.0 and stack.min() >= 0.0

    # Nose ray (0 deg) should see road ahead at spawn, not immediate wall
    center_ray = rays[config.RAY_ANGLES_DEG.index(0)]
    assert center_ray > 0.05, f"center ray too short: {center_ray}"

    print("OK rays", [round(r, 2) for r in rays])
    print("OK stack", stack.shape, "road_frac", round(stack.mean(), 2))
    pygame.quit()


if __name__ == "__main__":
    main()
