#!/usr/bin/env python3
"""Generate images with Google's Gemini image models (e.g. Nano Banana 2 /
gemini-3.1-flash-image), optionally style-conditioned on reference images.

Install once:
    pip install --upgrade google-genai python-dotenv

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment, or a .env
file in the repo root with GEMINI_API_KEY=... (never committed — see
.gitignore). Get a free key at https://aistudio.google.com/apikey. Free-tier
model availability varies by account — check https://ai.dev/rate-limit for
which model(s) your key actually has free quota for.

CLI example:
    python generate_gemini_image.py \
      --prompt "A single vivid royal-blue handbag, centered on a plain background" \
      --reference-image ../assets/templates/example_image1.png \
      --reference-image ../assets/templates/example_image2.png \
      --output outputs/blue_bag.png

`build_client()` / `generate_image()` are also imported by
generate_trials.py so the client is created once and reused across trials.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()

DEFAULT_MODEL = "gemini-3.1-flash-image"


def build_client() -> genai.Client:
    """Create a Gemini client. Reads GEMINI_API_KEY/GOOGLE_API_KEY from the environment."""
    return genai.Client()


def generate_image(
    client: genai.Client,
    prompt: str,
    *,
    reference_images: list[Path] | None = None,
    model: str = DEFAULT_MODEL,
) -> Image.Image:
    """Generate one image, optionally conditioned on 1+ reference images for style.

    Passing `reference_images` (e.g. the cartoon style templates) lets the
    model see the actual target style rather than only a text description
    of it.
    """
    contents = [prompt]
    for path in reference_images or []:
        contents.append(types.Part.from_bytes(data=path.read_bytes(), mime_type="image/png"))

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )

    for part in response.parts:
        if part.inline_data:
            return Image.open(io.BytesIO(part.inline_data.data))

    raise RuntimeError(f"Gemini returned no image for prompt: {prompt!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one image with a Gemini image model (e.g. Nano Banana)."
    )
    prompt_source = parser.add_mutually_exclusive_group(required=True)
    prompt_source.add_argument("--prompt", help="Text prompt for the image.")
    prompt_source.add_argument(
        "--prompt-file", type=Path, help="UTF-8 text file containing the prompt."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("gemini_output.png"),
        help="Output PNG path (default: gemini_output.png).",
    )
    parser.add_argument(
        "--reference-image", action="append", dest="reference_images", type=Path,
        help="Reference image for style conditioning (repeatable).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
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
        client, prompt, reference_images=args.reference_images, model=args.model,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Saved image to {args.output}")


if __name__ == "__main__":
    main()
