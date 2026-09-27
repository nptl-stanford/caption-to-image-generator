#!/usr/bin/env python3
"""Generate text-to-image samples with FLUX.2 [klein] 4B.

Install once (a recent Diffusers build is required):
    pip install --upgrade git+https://github.com/huggingface/diffusers.git
    pip install --upgrade transformers accelerate safetensors sentencepiece

CLI example:
    python generate_flux2_klein.py \
      --prompt "A single vivid royal-blue handbag, centered on a plain background" \
      --output outputs/blue_bag.png --seed 42

`build_pipeline()` / `generate_image()` are also imported by
generate_trials.py so the model is loaded once and reused across trials.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image


MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


def build_pipeline(
    model_id: str = MODEL_ID, cpu_offload: bool = True, device: str = "cuda"
) -> Flux2KleinPipeline:
    """Load the FLUX.2 [klein] pipeline once, ready for repeated generate_image() calls.

    `device="cpu"` works but is much slower (minutes per image instead of
    seconds) since this is a 4B-parameter model with no CUDA kernels to fall
    back on; it's meant as a fallback when no GPU is available, not routine use.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "This script requires an NVIDIA CUDA GPU. The model card lists about 13 GB "
            "of VRAM as the minimum target. Pass device='cpu' to run on CPU instead "
            "(much slower)."
        )
    # The official model card recommends bfloat16 and CPU offloading to reduce VRAM use;
    # neither applies when running on CPU, where everything already stays in RAM.
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    if device == "cpu":
        pipe.to("cpu")
    elif cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def generate_image(
    pipe: Flux2KleinPipeline,
    prompt: str,
    *,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
    guidance_scale: float = 1.0,
    device: str = "cuda",
) -> Image.Image:
    """Run one FLUX.2 [klein] inference and return the resulting PIL image."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")
    generator = torch.Generator(device=device).manual_seed(seed)
    return pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one image with FLUX.2 [klein] 4B."
    )
    prompt_source = parser.add_mutually_exclusive_group(required=True)
    prompt_source.add_argument("--prompt", help="Text prompt for the image.")
    prompt_source.add_argument(
        "--prompt-file", type=Path, help="UTF-8 text file containing the prompt."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("flux2_output.png"),
        help="Output PNG path (default: flux2_output.png).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--no-cpu-offload", action="store_true",
        help="Keep the full pipeline on the GPU; requires more VRAM.",
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default="cuda",
        help="Device to run on (default: cuda). CPU works but is much slower.",
    )
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

    pipe = build_pipeline(cpu_offload=not args.no_cpu_offload, device=args.device)
    image = generate_image(
        pipe,
        prompt,
        seed=args.seed,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Saved image to {args.output}")


if __name__ == "__main__":
    main()
