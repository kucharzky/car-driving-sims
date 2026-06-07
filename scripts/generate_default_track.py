"""Generate the default assets/track.png sample map."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pygame

import config

OUTPUT = config.ASSETS_DIR / "track.png"


def main() -> None:
    pygame.init()
    config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    surface = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    surface.fill(config.TRACK_COLOR_WALL)

    cx, cy = config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2
    outer_rx, outer_ry = 380, 280
    inner_rx, inner_ry = 300, 200

    # Road ring: outer ellipse filled white, inner ellipse punched out black
    pygame.draw.ellipse(
        surface,
        config.TRACK_COLOR_ROAD,
        (cx - outer_rx, cy - outer_ry, outer_rx * 2, outer_ry * 2),
    )
    pygame.draw.ellipse(
        surface,
        config.TRACK_COLOR_WALL,
        (cx - inner_rx, cy - inner_ry, inner_rx * 2, inner_ry * 2),
    )

    # Start/finish line spans across the track (inner wall → outer wall).
    # At the right side of the oval the centerline runs vertically, so the
    # line is horizontal — perpendicular to the driving direction.
    angle = 0.0  # rightmost point on ellipse
    ox = cx + outer_rx * math.cos(angle)
    oy = cy + outer_ry * math.sin(angle)
    ix = cx + inner_rx * math.cos(angle)
    iy = cy + inner_ry * math.sin(angle)
    pygame.draw.line(
        surface,
        config.TRACK_COLOR_START_FINISH,
        (int(ix), int(iy)),
        (int(ox), int(oy)),
        config.START_FINISH_LINE_WIDTH,
    )

    pygame.image.save(surface, str(OUTPUT))
    pygame.quit()
    print(f"Saved default track to {OUTPUT}")


if __name__ == "__main__":
    main()
