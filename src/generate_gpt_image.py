#!/usr/bin/env python3
"""Generate images with OpenAI's gpt-image-1.5, optionally style-conditioned
on reference images.

Install once:
    pip install --upgrade openai python-dotenv

Requires OPENAI_API_KEY in the environment, or a .env file in the repo root
with OPENAI_API_KEY=... (never committed — see .gitignore).

CLI example:
    python generate_gpt_image.py \
      --prompt "A single vivid royal-blue handbag, centered on a plain background" \
      --reference-image ../assets/templates/example_image1.png \
      --reference-image ../assets/templates/example_image2.png \
      --output outputs/blue_bag.png

`build_client()` / `generate_image()` are also imported by
generate_trials.py so the client is created once and reused across trials.
"""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()

DEFAULT_MODEL = "gpt-image-1.5"

# images.edit() treats input images as content to edit, not a style
# reference — without this framing, a prompt that needs N distinct objects
# (e.g. "two boxes") can get the model to just reuse the N reference images
# as those objects instead of drawing new ones. Prepending this makes the
# reference-only role explicit.
_STYLE_ONLY_PREFIX = (
    "The attached reference images are an ILLUSTRATION STYLE GUIDE ONLY "
    "(line weight, shading, color palette, background treatment) — do not "
    "reproduce their subjects, objects, scenes, or composition in the "
    "output. Depict only the following, as entirely new original content "
    "rendered in that style:\n\n"
)


def build_client() -> OpenAI:
    """Create an OpenAI client. Reads OPENAI_API_KEY from the environment."""
    return OpenAI()


def generate_image(
    client: OpenAI,
    prompt: str,
    *,
    reference_images: list[Path] | None = None,
    model: str = DEFAULT_MODEL,
    size: str = "1024x1024",
    quality: str = "medium",
    background: str = "opaque",
) -> Image.Image:
    """Generate one image, optionally conditioned on 1+ reference images for style.

    With `reference_images`, this calls the images.edit endpoint (which
    accepts up to 16 input images) rather than images.generate, since
    passing reference images is how the GPT image models do
    style-conditioned generation. GPT image models always return
    base64-encoded output (no URL option).
    """
    if reference_images:
        files = [open(path, "rb") for path in reference_images]
        try:
            response = client.images.edit(
                image=files,
                prompt=_STYLE_ONLY_PREFIX + prompt,
                model=model,
                size=size,
                quality=quality,
                background=background,
                n=1,
            )
        finally:
            for f in files:
                f.close()
    else:
        response = client.images.generate(
            prompt=prompt, model=model, size=size, quality=quality, background=background, n=1,
        )

    image_bytes = base64.b64decode(response.data[0].b64_json)
    return Image.open(io.BytesIO(image_bytes))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one image with OpenAI's gpt-image-1.5."
    )
    prompt_source = parser.add_mutually_exclusive_group(required=True)
    prompt_source.add_argument("--prompt", help="Text prompt for the image.")
    prompt_source.add_argument(
        "--prompt-file", type=Path, help="UTF-8 text file containing the prompt."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("gpt_image_output.png"),
        help="Output PNG path (default: gpt_image_output.png).",
    )
    parser.add_argument(
        "--reference-image", action="append", dest="reference_images", type=Path,
        help="Reference image for style conditioning (repeatable, up to 16).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--quality", default="medium")
    parser.add_argument("--background", default="opaque", choices=["opaque", "transparent", "auto"])
    return parser.parse_args()


def get_prompt(args: argparse.Namespace) -> str:
    prompt = args.prompt
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("The prompt cannot be empty.")
    return prompt


def main() -> None:
    args = parse_args()
    prompt = get_prompt(args)

    client = build_client()
    image = generate_image(
        client, prompt,
        reference_images=args.reference_images,
        model=args.model, size=args.size, quality=args.quality, background=args.background,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Saved image to {args.output}")


if __name__ == "__main__":
    main()
