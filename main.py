"""Entry point for the 2D self-driving car simulation."""

from __future__ import annotations

import argparse
import math
import sys

import cv2
import pygame

import config
from car import Car
from track_manager import LapTracker, TrackManager


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
    parser.add_argument(
        "--debug-sensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show ray overlay and OpenCV debug windows (Phase 2)",
    )
    return parser.parse_args()


def check_collision(car: Car, track: TrackManager) -> bool:
    """Return True if any car corner intersects a wall pixel."""
    for px, py in car.get_corners():
        if track.is_wall(px, py):
            return True
    return False


def render_sensor_hud(
    surface: pygame.Surface,
    font: pygame.font.Font,
    car: Car,
    y_offset: int,
) -> None:
    """Ray distance bars at the bottom of the main window."""
    bar_w = 36
    bar_h = 60
    gap = 8
    total_w = config.RAY_COUNT * bar_w + (config.RAY_COUNT - 1) * gap
    start_x = (config.SCREEN_WIDTH - total_w) // 2
    y = config.SCREEN_HEIGHT - y_offset

    for i, (deg, dist) in enumerate(zip(config.RAY_ANGLES_DEG, car.get_ray_distances())):
        x = start_x + i * (bar_w + gap)
        fill_h = int(dist * bar_h)
        pygame.draw.rect(surface, (50, 50, 50), (x, y, bar_w, bar_h))
        color = (220, 80, 80) if dist < 0.35 else (80, 200, 100)
        pygame.draw.rect(surface, color, (x, y + bar_h - fill_h, bar_w, fill_h))
        label = font.render(str(deg), True, (200, 200, 200))
        surface.blit(label, (x + 4, y + bar_h + 4))


def show_cv_debug_windows(car: Car, show: bool) -> None:
    """OpenCV windows for oriented crop and 4-frame CNN input preview."""
    if not show:
        return

    crop = car.get_crop_rgb()
    if crop is not None:
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        preview = cv2.resize(bgr, (280, 280), interpolation=cv2.INTER_AREA)
        cv2.putText(
            preview,
            "Oriented crop (car up)",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("Vision Crop", preview)

    stack_preview = car.get_stack_preview_bgr(scale=4)
    if stack_preview is not None:
        cv2.putText(
            stack_preview,
            "CNN input (4-frame stack)",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("Vision Stack", stack_preview)

    cv2.waitKey(1)


def run_manual(track_source: str, debug_sensors: bool) -> None:
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Self-Driving Car Sim — Manual Mode")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    small_font = pygame.font.SysFont("consolas", 14)

    track = TrackManager()
    try:
        track.load(track_source)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        print("Falling back to procedural track.", file=sys.stderr)
        track.load("procedural")

    sx, sy, heading = track.get_spawn()
    car = Car(sx, sy, heading)
    lap_tracker = LapTracker(track, heading)
    show_debug = debug_sensors

    running = True
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
                    lap_tracker.reset(heading)
                elif event.key == pygame.K_v:
                    show_debug = not show_debug
                    if not show_debug:
                        cv2.destroyAllWindows()

        keys = pygame.key.get_pressed()
        car.apply_keyboard(keys)
        car.update()

        if check_collision(car, track):
            car.revert_position()

        car.update_sensors(track)
        lap_tracker.update(car)

        screen.fill((40, 40, 40))
        track.render(screen)
        if show_debug:
            car.render_rays(screen)
        car.render(screen)

        hud_lines = [
            "WASD drive | R reset | V toggle sensors | ESC quit",
            f"Speed: {car.velocity:.2f}  Steer: {math.degrees(car.steer_angle):.1f}°  "
            f"Rays: {car.get_ray_distances()}",
            f"Laps: {lap_tracker.laps}  Track: {track.source}  "
            f"{'ARMED' if lap_tracker.armed else 'start'}  "
            f"lap:{lap_tracker.progress_pct:.0f}%  "
            f"stack:{car.frame_stack_count}/{config.FRAME_STACK}",
        ]
        for i, line in enumerate(hud_lines):
            text = font.render(line, True, (230, 230, 230))
            screen.blit(text, (10, 10 + i * 20))

        if show_debug:
            render_sensor_hud(screen, small_font, car, y_offset=100)

        pygame.display.flip()
        show_cv_debug_windows(car, show_debug)
        clock.tick(config.FPS)

    cv2.destroyAllWindows()
    pygame.quit()


def main() -> None:
    args = parse_args()

    if args.mode == "manual":
        run_manual(args.track, args.debug_sensors)
    else:
        print(
            f"Mode '{args.mode}' is not implemented yet (Phase 3). "
            "Use --mode manual for now.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
