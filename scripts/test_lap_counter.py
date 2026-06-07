"""Simulate a realistic full lap: cross line, tour track, cross line again."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pygame

from car import Car
from track_manager import LapTracker, TrackManager


def realistic_lap_path(track: TrackManager) -> list[tuple[float, float, float]]:
    sx, sy, sh = track.get_spawn()
    cx, cy = track.width / 2, track.height / 2
    pts: list[tuple[float, float, float]] = []

    for y in range(int(sy), 455, 3):
        pts.append((sx, float(y), sh))

    for i in range(30, 900):
        t = -math.pi / 2 - 0.25 + (2 * math.pi * i / 900)
        t2 = -math.pi / 2 - 0.25 + (2 * math.pi * (i + 1) / 900)
        x = cx + 340 * math.cos(t)
        y = cy + 240 * math.sin(t)
        h = math.atan2(
            cy + 240 * math.sin(t2) - y,
            cx + 340 * math.cos(t2) - x,
        )
        pts.append((x, y, h))

    for y in range(int(sy), 400, 3):
        pts.append((sx, float(y), sh))

    return pts


def drive(track_source: str) -> int:
    pygame.init()
    track = TrackManager()
    track.load(track_source)
    sx, sy, heading = track.get_spawn()
    car = Car(sx, sy, heading)
    lap_tracker = LapTracker(track, heading)

    for x, y, h in realistic_lap_path(track):
        car._prev_x, car._prev_y = car.x, car.y
        car.x, car.y = x, y
        car.heading = h
        lap_tracker.update(car)

    pygame.quit()
    return lap_tracker.laps


if __name__ == "__main__":
    for src in ("assets/track.png", "procedural"):
        print(f"{src}: laps = {drive(src)}")
