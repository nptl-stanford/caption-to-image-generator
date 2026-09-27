#!/usr/bin/env python3
"""Generate one image per trial in a trials CSV, named <trial_id>.png.

Usage:
    python src/generate_trials.py --config configs/generation.yaml

Which CSV columns hold the id and the prompt text comes from
config["csv"], so this script works for any trials CSV — only
configs/generation.yaml needs to change, not this file.

Three independent generation backends are supported via config["generation"]["backend"]
(or --backend): "flux" (self-hosted FLUX.2-klein-4B, generate_flux2_klein.py),
"gemini" (Google's Gemini image API, generate_gemini_image.py), and "openai"
(OpenAI's gpt-image-1.5, generate_gpt_image.py). Each backend's dependencies
(torch/diffusers vs. google-genai vs. openai) are only imported when that
backend is actually selected.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from prompts import load_trials
from utils import append_manifest_row, ensure_dir, load_config

_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 5


def _generate_with_retry(generate_fn, *args, **kwargs):
    """Call `generate_fn`, retrying transient failures (rate limits, network
    blips) with exponential backoff. Raises on the final attempt so callers
    can decide whether to skip the trial.
    """
    backoff = _RETRY_BACKOFF_SECONDS
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return generate_fn(*args, **kwargs)
        except Exception as exc:
            if attempt == _MAX_ATTEMPTS:
                raise
            print(f"warning: generation failed (attempt {attempt}/{_MAX_ATTEMPTS}): {exc}; retrying in {backoff}s")
            time.sleep(backoff)
            backoff *= 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/generation.yaml"),
        help="Path to the pipeline config (default: configs/generation.yaml).",
    )
    parser.add_argument(
        "--trial-id", action="append", dest="trial_ids",
        help="Only generate this trial_id (repeatable). Default: generate every trial.",
    )
    parser.add_argument(
        "--backend", choices=["flux", "gemini", "openai"],
        help="Override config['generation']['backend'].",
    )
    return parser.parse_args()


def _save_and_log(image, trial, generated_dir: Path, manifest_path: Path, prompt: str) -> None:
    out_path = generated_dir / f"{trial.trial_id}.png"
    image.save(out_path)
    append_manifest_row(manifest_path, {**trial.row, "prompt": prompt, "image_path": str(out_path)})
    print(f"[{trial.trial_id}] saved {out_path}")


def _run_flux(trials, config: dict, style_suffix: str, generated_dir: Path, manifest_path: Path) -> None:
    from generate_flux2_klein import build_pipeline, generate_image

    gen_cfg = config["generation"]
    device = gen_cfg.get("device", "cuda")
    pipe = build_pipeline(
        model_id=config["model"]["id"],
        cpu_offload=gen_cfg.get("cpu_offload", True),
        device=device,
    )

    for trial in trials:
        prompt = f"{trial.prompt}, {style_suffix}" if style_suffix else trial.prompt
        try:
            image = _generate_with_retry(
                generate_image,
                pipe,
                prompt,
                seed=config["model"].get("seed", 0),
                width=gen_cfg.get("width", 1024),
                height=gen_cfg.get("height", 1024),
                steps=gen_cfg.get("steps", 4),
                guidance_scale=gen_cfg.get("guidance_scale", 1.0),
                device=device,
            )
        except Exception as exc:
            print(f"error: trial {trial.trial_id!r} failed after {_MAX_ATTEMPTS} attempts: {exc}; skipping")
            continue
        _save_and_log(image, trial, generated_dir, manifest_path, prompt)


def _run_gemini(trials, config: dict, style_suffix: str, generated_dir: Path, manifest_path: Path) -> None:
    from generate_gemini_image import build_client, generate_image

    gemini_cfg = config["generation"].get("gemini", {})
    client = build_client()
    model = gemini_cfg.get("model", "gemini-2.5-flash-image")
    reference_images = [Path(p) for p in gemini_cfg.get("style_reference_images", [])]

    for trial in trials:
        prompt = f"{trial.prompt}. {style_suffix}" if style_suffix else trial.prompt
        try:
            image = _generate_with_retry(
                generate_image, client, prompt, reference_images=reference_images, model=model,
            )
        except Exception as exc:
            print(f"error: trial {trial.trial_id!r} failed after {_MAX_ATTEMPTS} attempts: {exc}; skipping")
            continue
        _save_and_log(image, trial, generated_dir, manifest_path, prompt)


def _run_openai(trials, config: dict, style_suffix: str, generated_dir: Path, manifest_path: Path) -> None:
    from generate_gpt_image import build_client, generate_image

    openai_cfg = config["generation"].get("openai", {})
    client = build_client()
    model = openai_cfg.get("model", "gpt-image-1.5")
    size = openai_cfg.get("size", "1024x1024")
    quality = openai_cfg.get("quality", "medium")
    background = openai_cfg.get("background", "opaque")
    reference_images = [Path(p) for p in openai_cfg.get("style_reference_images", [])]

    for trial in trials:
        prompt = f"{trial.prompt}. {style_suffix}" if style_suffix else trial.prompt
        try:
            image = _generate_with_retry(
                generate_image, client, prompt, reference_images=reference_images,
                model=model, size=size, quality=quality, background=background,
            )
        except Exception as exc:
            print(f"error: trial {trial.trial_id!r} failed after {_MAX_ATTEMPTS} attempts: {exc}; skipping")
            continue
        _save_and_log(image, trial, generated_dir, manifest_path, prompt)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    csv_cfg = config["csv"]
    gen_cfg = config["generation"]
    paths_cfg = config["paths"]
    style_suffix = (gen_cfg.get("style_suffix") or "").strip()

    generated_dir = ensure_dir(Path(paths_cfg["generated_dir"]))
    manifest_path = Path(paths_cfg["generation_manifest"])

    trials = load_trials(
        Path(paths_cfg["trials_csv"]),
        id_column=csv_cfg["id_column"],
        prompt_columns=csv_cfg["prompt_columns"],
        prompt_separator=csv_cfg.get("prompt_separator", " "),
    )
    if args.trial_ids:
        wanted = set(args.trial_ids)
        trials = (trial for trial in trials if trial.trial_id in wanted)

    backend = args.backend or gen_cfg.get("backend", "flux")
    if backend == "flux":
        _run_flux(trials, config, style_suffix, generated_dir, manifest_path)
    elif backend == "gemini":
        _run_gemini(trials, config, style_suffix, generated_dir, manifest_path)
    elif backend == "openai":
        _run_openai(trials, config, style_suffix, generated_dir, manifest_path)
    else:
        raise ValueError(f"Unknown generation.backend: {backend!r}")


if __name__ == "__main__":
    main()
