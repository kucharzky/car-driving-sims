"""Entry point for the 2D self-driving car simulation."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import pygame

import config
from agent import Agent
from car import Car
from environment import Environment, check_collision
from track_manager import LapTracker, TrackManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D Self-Driving Car Simulation")
    parser.add_argument(
        "--mode",
        choices=["manual", "train", "play"],
        default="manual",
        help="Run mode",
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
        help="Show ray overlay and OpenCV debug windows (manual/play)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Training episodes (train mode)",
    )
    parser.add_argument(
        "--render-train",
        action="store_true",
        help="Render the simulation while training (slower)",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(config.DEFAULT_CHECKPOINT),
        help="Path to save/load agent weights (.pth)",
    )
    return parser.parse_args()


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
        cv2.imshow("Vision Crop", preview)

    stack_preview = car.get_stack_preview_bgr(scale=4)
    if stack_preview is not None:
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


def run_train(
    track_source: str,
    episodes: int,
    render: bool,
    checkpoint: str,
) -> None:
    env = Environment(track_source, render=render)
    agent = Agent()
    checkpoint_path = Path(checkpoint)

    if checkpoint_path.exists():
        try:
            agent.load(checkpoint_path)
            print(f"Resumed from checkpoint: {checkpoint_path}")
        except RuntimeError as exc:
            print(
                f"Could not load {checkpoint_path} (architecture changed): {exc}",
                file=sys.stderr,
            )
            print("Starting fresh training run.", file=sys.stderr)

    best_score = float("-inf")
    best_progress = 0.0
    print(f"Training on {track_source} for {episodes} episodes ({agent.device})")

    try:
        for episode in range(1, episodes + 1):
            state = env.reset()
            rays = env.get_rays()
            done = False
            losses: list[float] = []

            while not done:
                if render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt

                action = agent.select_action(state, rays)
                next_state, reward, done, info = env.step(action)
                next_rays = info["rays"]
                agent.remember(state, rays, action, reward, next_state, next_rays, done)

                loss = agent.train_updates()
                if loss is not None:
                    losses.append(loss)

                state = next_state
                rays = next_rays

                if render:
                    env.render(show_rays=True)

            agent.decay_epsilon()

            # Save by lap progress first, then episode reward (avoids "park near wall" optimum)
            score = info["max_progress_pct"] * 1000.0 + env.episode_reward
            if score > best_score:
                best_score = score
                best_progress = info["max_progress_pct"]
                agent.save(checkpoint_path)

            if episode % config.SAVE_EVERY_EPISODES == 0:
                periodic = config.CHECKPOINT_DIR / f"agent_ep{episode:04d}.pth"
                agent.save(periodic)

            if episode % config.TRAIN_LOG_EVERY == 0 or episode == 1:
                avg_loss = sum(losses) / len(losses) if losses else 0.0
                print(
                    f"Ep {episode:4d} | reward {env.episode_reward:8.1f} | "
                    f"progress {info['max_progress_pct']:5.1f}% | "
                    f"best_prog {best_progress:5.1f}% | laps {info['laps']} | "
                    f"steps {info['steps']} | eps {agent.epsilon:.3f} | "
                    f"loss {avg_loss:.4f} | buffer {len(agent.memory)} | "
                    f"min_ray {info.get('min_ray', 0):.2f}"
                )

    except KeyboardInterrupt:
        print("\nTraining interrupted — saving checkpoint.")
        agent.save(checkpoint_path)
    finally:
        env.close()

    print(f"Done. Best progress: {best_progress:.1f}% -> {checkpoint_path}")


def run_play(track_source: str, checkpoint: str, debug_sensors: bool) -> None:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    env = Environment(track_source, render=True)
    agent = Agent()
    agent.load(checkpoint_path)
    agent.epsilon = config.EPSILON_END

    pygame.display.set_caption("Self-Driving Car Sim — Play Mode")
    print(f"Loaded agent from {checkpoint_path}")

    running = True
    try:
        state = env.reset()
        rays = env.get_rays()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        state = env.reset()
                        rays = env.get_rays()
                    elif event.key == pygame.K_v:
                        debug_sensors = not debug_sensors
                        if not debug_sensors:
                            cv2.destroyAllWindows()

            action = agent.select_action(state, rays, epsilon=0.0)
            state, _reward, done, info = env.step(action)
            rays = info["rays"]

            if done:
                state = env.reset()

            env.render(show_rays=debug_sensors)
            if debug_sensors:
                show_cv_debug_windows(env.car, True)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        env.close()


def main() -> None:
    args = parse_args()

    if args.mode == "manual":
        run_manual(args.track, args.debug_sensors)
    elif args.mode == "train":
        run_train(args.track, args.episodes, args.render_train, args.checkpoint)
    elif args.mode == "play":
        run_play(args.track, args.checkpoint, args.debug_sensors)


if __name__ == "__main__":
    main()
