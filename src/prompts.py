"""Load per-trial prompts from a trials CSV (e.g. data/input/*.csv)."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence


@dataclass
class Trial:
    trial_id: str
    prompt: str
    row: dict[str, str] = field(default_factory=dict)


def load_trials(
    csv_path: Path,
    id_column: str = "trial_id",
    prompt_columns: Sequence[str] = ("sentence", "image_generation_note"),
    prompt_separator: str = " ",
) -> Iterator[Trial]:
    """Read a trials CSV and yield one Trial per row.

    Only `id_column` and `prompt_columns` need to exist. Every other column
    (e.g. focus_symbol, response_symbol, tense_aspect, ...) is passed through
    untouched on `Trial.row`, so a future CSV with a different schema only
    needs a different `configs/generation.yaml`, not a code change here.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = prompt_separator.join(
                row[column].strip() for column in prompt_columns if row.get(column)
            )
            yield Trial(trial_id=row[id_column], prompt=prompt, row=row)
