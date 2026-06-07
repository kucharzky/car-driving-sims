"""Generate the default assets/track.png sample map."""

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

    # Outer boundary (road fill between two ellipses)
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

    # Start/finish line on the right straight
    line_x = cx + (outer_rx + inner_rx) // 2
    y_top = cy - (outer_ry + inner_ry) // 4
    y_bot = cy + (outer_ry + inner_ry) // 4
    pygame.draw.line(
        surface,
        config.TRACK_COLOR_START_FINISH,
        (line_x, y_top),
        (line_x, y_bot),
        6,
    )

    pygame.image.save(surface, str(OUTPUT))
    pygame.quit()
    print(f"Saved default track to {OUTPUT}")


if __name__ == "__main__":
    main()
