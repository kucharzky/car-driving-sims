"""Entry point for the 2D self-driving car simulation."""

from __future__ import annotations

import argparse
import math
import sys

import pygame

import config
from car import Car
from track_manager import TrackManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D Self-Driving Car Simulation")
    parser.add_argument(
        "--mode",
        choices=["manual", "train", "play"],
        default="manual",
        help="Run mode (train/play are Phase 3)",
    )
    parser.add_argument(
        "--track",
        default=str(config.DEFAULT_TRACK_PATH),
        help="Path to PNG track or 'procedural' for generated track",
    )
    return parser.parse_args()


def check_collision(car: Car, track: TrackManager) -> bool:
    """Return True if any car corner intersects a wall pixel."""
    for px, py in car.get_corners():
        if track.is_wall(px, py):
            return True
    return False


def run_manual(track_source: str) -> None:
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Self-Driving Car Sim — Manual Mode")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    track = TrackManager()
    try:
        track.load(track_source)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        print("Falling back to procedural track.", file=sys.stderr)
        track.load("procedural")

    sx, sy, heading = track.get_spawn()
    car = Car(sx, sy, heading)

    running = True
    laps = 0
    on_start_line = track.is_start_finish(sx, sy)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    sx, sy, heading = track.get_spawn()
                    car.reset(sx, sy, heading)
                    on_start_line = True

        keys = pygame.key.get_pressed()
        car.apply_keyboard(keys)
        car.update()

        if check_collision(car, track):
            car.revert_position()

        prev_center = car.get_prev_center()
        curr_center = car.get_center()
        if track.crossed_start_finish(prev_center, curr_center) and not on_start_line:
            laps += 1
        on_start_line = track.is_start_finish(*curr_center)

        screen.fill((40, 40, 40))
        track.render(screen)
        car.render(screen)

        hud_lines = [
            "WASD — drive | R — reset | ESC — quit",
            f"Speed: {car.velocity:.2f}  Steer: {math.degrees(car.steer_angle):.1f}°",
            f"Laps: {laps}  Track: {track.source}",
        ]
        for i, line in enumerate(hud_lines):
            text = font.render(line, True, (230, 230, 230))
            screen.blit(text, (10, 10 + i * 22))

        pygame.display.flip()
        clock.tick(config.FPS)

    pygame.quit()


def main() -> None:
    args = parse_args()

    if args.mode == "manual":
        run_manual(args.track)
    else:
        print(
            f"Mode '{args.mode}' is not implemented yet (Phase 3). "
            "Use --mode manual for now.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
