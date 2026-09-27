# image-captioning-data-generator

Goal: standardize text-to-image generation for image-captioning tasks.

## Pipeline

1. Put your trials in a CSV under `data/input/` (this run uses
   `image_captioning_constrain_language.csv`) and point
   `configs/generation.yaml`'s `paths.trials_csv` at it.
2. `python src/generate_trials.py --config configs/generation.yaml [--backend flux|gemini|openai]`
   builds a prompt per trial from `csv.prompt_columns` (+ a fixed
   `generation.style_suffix` so images match the cartoon reference style in
   `assets/templates/`), generates it, and saves it to
   `<generated_dir>/<trial_id>.png`. Every row is logged to
   `generation_manifest.csv`.
3. *(Optional)* `python src/add_classification_icons.py --config configs/generation.yaml`
   draws each trial's classification symbols (from `icons.columns`, e.g.
   `focus_symbol`, `response_symbol`) in a badge at the configured corner
   (`icons.positions`) of its generated image, and saves the result to
   `<classified_dir>/<trial_id>.png`. Every row is logged to
   `classification_manifest.csv`.

Generated/classified images and manifests are regenerable outputs, so
`paths.*` in the config point at `$SCRATCH` by default, not this git repo.
`data/input/`, `data/generated/`, `data/classified/`, and `data/manifests/`
are all gitignored (only empty via a `.gitkeep`) — trials CSVs, generated
images, and manifests all stay local/on `$SCRATCH` and are never committed.

## Generation backends

`generation.backend` in `configs/generation.yaml` (or `--backend` on
`generate_trials.py`) picks between three independent, separately-installable
backends. In plain terms, the practical choice is mostly cost vs. GPU access:

- **`flux`** — self-hosted FLUX.2-klein-4B (`src/generate_flux2_klein.py`).
  Runs on this cluster's GPU (or CPU, `generation.device: cpu`, much
  slower). **Free** — no API key, no per-image cost — but needs a GPU
  allocation and a multi-GB model download, and style consistency relies
  only on `generation.style_suffix` (a text description) since
  FLUX.2-klein-4B doesn't take reference images. **Status: implemented but
  not yet tried end-to-end on this cluster** — treat the first real run as
  a test, not a known-good path.
- **`gemini`** — Google's Gemini image API, e.g. Nano Banana
  (`src/generate_gemini_image.py`). Fast, no GPU needed, has a free tier.
  Requires `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in the environment or a
  `.env` file in the repo root (gitignored, never committed) — get a free
  key at https://aistudio.google.com/apikey. Unlike FLUX, Gemini image
  models accept real reference images, so `generation.gemini.style_reference_images`
  (the two templates in `assets/templates/`) are passed in directly for
  style conditioning, not just described in text.
- **`openai`** — OpenAI's `gpt-image-1.5` (`src/generate_gpt_image.py`).
  Also fast, no GPU needed, and also takes the reference images in
  `assets/templates/` directly for style conditioning. Requires
  `OPENAI_API_KEY` in the environment or a `.env` file in the repo root
  (gitignored, never committed). **Costs real money per image generated
  — this backend bills your OpenAI account, it has no free tier.** **Status:
  this is the backend actually used to generate and review the trials in
  this repo so far, and it's producing good results** — it's the
  recommended default until `flux` has been tried and confirmed working.

`requirements.txt` is split by backend with comments — skip the FLUX section
(torch/diffusers, large) if you only use `gemini`/`openai`, and vice versa.

## Making it general across CSVs

Most trials CSVs will only need an id column and one or two text columns
(e.g. `sentence`, `image_generation_note`). Some, like this one, also carry
per-trial classification symbols (`focus_symbol`, `response_symbol`). The
scripts don't hardcode any of this — everything CSV-shape-specific lives in
`configs/generation.yaml`:

- `csv.id_column` / `csv.prompt_columns` — which columns identify a trial
  and which ones get joined into the generation prompt.
- `icons.columns` / `icons.positions` — which columns hold symbols to draw,
  and which corner each goes in. Set both to `[]` for a CSV with no symbols
  to classify, or add a third entry if a future CSV has three.

For a new CSV with a different schema, edit the config — not `src/`.

## Icon rendering

Symbols are drawn with Pillow directly from the CSV values (e.g. `?`, `●`,
or emoji like `🎨`/`📍`/`🏃`), not from static icon image files. Plain
glyphs render fine with Pillow's default font; pictographic emoji use the
bundled `assets/fonts/NotoEmoji.ttf` (Google, OFL-licensed) —
`add_classification_icons.py` checks per-symbol glyph coverage and falls
back to Pillow's default font for anything that font doesn't have.

## Layout

```
configs/            Pipeline configuration (model, csv schema, icon placement)
data/input/         Source trial CSVs (gitignored — bring your own)
assets/templates/   Cartoon-style reference images (style guide; also fed to the gemini/openai backends)
assets/fonts/       Bundled emoji/symbol font for icon rendering
src/                Pipeline scripts and shared helpers
tests/              Tests for the pipeline
```

`data/input/`, `data/manifests/`, `data/generated/`, `data/classified/`
exist as empty, gitignored placeholders (each holds just a `.gitkeep`) —
actual trial CSVs and outputs go to `$SCRATCH` per the config defaults
above, and are never committed to this repo.
