#!/usr/bin/env python3
"""Overlay each trial's classification symbols onto its generated image.

Usage:
    python src/add_classification_icons.py --config configs/generation.yaml

Which trials.csv columns hold the symbols (and how many there are) comes
from config["icons"]["columns"]/["positions"], so this script works for a
future CSV with a different number of symbol columns — only
configs/generation.yaml needs to change, not this file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from prompts import load_trials
from utils import append_manifest_row, ensure_dir, load_config

_POSITION_ANCHORS = {
    "top_left": lambda w, h, badge, margin: (margin, margin),
    "top_right": lambda w, h, badge, margin: (w - badge - margin, margin),
    "bottom_left": lambda w, h, badge, margin: (margin, h - badge - margin),
    "bottom_right": lambda w, h, badge, margin: (w - badge - margin, h - badge - margin),
}


def _load_default_font(font_size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(font_size)
    except TypeError:
        # Older Pillow: load_default() takes no size argument.
        return ImageFont.load_default()


def _font_codepoints(font_path: str) -> set[int] | None:
    """Return the set of Unicode codepoints `font_path` has glyphs for.

    Returns None if fontTools isn't installed to introspect it — callers
    should then assume the font covers everything asked of it.
    """
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return None
    return set(TTFont(font_path).getBestCmap())


def build_font_resolver(font_paths: list[str] | str | None, font_size: int):
    """Build a function symbol -> PIL font.

    No single font covers every kind of symbol a CSV might use — e.g. an
    emoji-only font like Noto Emoji has no "?" or "●" glyph, and Pillow's
    own default font is missing plenty of Unicode symbol blocks too (it
    silently renders a "missing glyph" box instead of erroring). So
    `font_paths` is tried in order, and each symbol is routed to the first
    font whose glyph coverage includes every character in it; Pillow's
    default font is the final fallback.
    """
    if isinstance(font_paths, str):
        font_paths = [font_paths]

    default_font = _load_default_font(font_size)
    candidates: list[tuple[ImageFont.ImageFont, set[int] | None]] = []
    for font_path in font_paths or []:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            print(f"warning: could not load font {font_path!r}, skipping it")
            continue
        covered = _font_codepoints(font_path)
        if covered is None:
            print(
                f"warning: fontTools not installed, cannot check glyph coverage of "
                f"{font_path!r}; using it for every symbol"
            )
        candidates.append((font, covered))

    def resolve(symbol: str) -> ImageFont.ImageFont:
        for font, covered in candidates:
            if covered is None or (symbol and all(ord(ch) in covered for ch in symbol)):
                return font
        return default_font

    return resolve


def overlay_icons(
    image: Image.Image,
    symbols: list[str],
    positions: list[str],
    fonts: list[ImageFont.ImageFont],
    *,
    margin: int = 24,
) -> Image.Image:
    """Draw each symbol in a white badge at its configured corner of `image`."""
    if not (len(symbols) == len(positions) == len(fonts)):
        raise ValueError("symbols, positions, and fonts must be the same length")

    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)

    for symbol, position, font in zip(symbols, positions, fonts):
        badge_diameter = max(getattr(font, "size", margin * 2) * 1.6, 1)
        x, y = _POSITION_ANCHORS[position](image.width, image.height, badge_diameter, margin)
        draw.ellipse(
            [x, y, x + badge_diameter, y + badge_diameter],
            fill="white", outline="black", width=3,
        )
        center = (x + badge_diameter / 2, y + badge_diameter / 2)
        try:
            draw.text(center, symbol, font=font, fill="black", anchor="mm", embedded_color=True)
        except TypeError:
            # Older Pillow: draw.text() has no embedded_color argument.
            draw.text(center, symbol, font=font, fill="black", anchor="mm")

    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/generation.yaml"),
        help="Path to the pipeline config (default: configs/generation.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    csv_cfg = config["csv"]
    icons_cfg = config["icons"]
    paths_cfg = config["paths"]

    resolve_font = build_font_resolver(icons_cfg.get("font_paths"), icons_cfg.get("font_size", 96))
    generated_dir = Path(paths_cfg["generated_dir"])
    classified_dir = ensure_dir(Path(paths_cfg["classified_dir"]))
    manifest_path = Path(paths_cfg["classification_manifest"])

    trials = load_trials(
        Path(paths_cfg["trials_csv"]),
        id_column=csv_cfg["id_column"],
        prompt_columns=csv_cfg["prompt_columns"],
        prompt_separator=csv_cfg.get("prompt_separator", " "),
    )

    for trial in trials:
        image_path = generated_dir / f"{trial.trial_id}.png"
        if not image_path.exists():
            print(f"warning: no generated image for trial {trial.trial_id!r}, skipping")
            continue

        symbols = [trial.row[column] for column in icons_cfg["columns"]]
        fonts = [resolve_font(symbol) for symbol in symbols]
        with Image.open(image_path) as source:
            image = overlay_icons(
                source,
                symbols,
                icons_cfg["positions"],
                fonts,
                margin=icons_cfg.get("margin", 24),
            )

        out_path = classified_dir / f"{trial.trial_id}.png"
        image.convert("RGB").save(out_path)
        append_manifest_row(manifest_path, {**trial.row, "image_path": str(out_path)})
        print(f"[{trial.trial_id}] saved {out_path}")


if __name__ == "__main__":
    main()
