"""Load the committed golden dataset of ground-truth-labelled headlines.

The golden set is a JSONL file (one record per line). Only `text` and
`ground_truth` are required for evaluation; the remaining fields are provenance
metadata captured during error-analysis annotation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import GOLDEN_PATH, LABEL2ID


@dataclass
class GoldenHeadline:
    text: str
    ground_truth: str
    rationale: str | None = None
    ticker: str | None = None
    source: str | None = None


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenHeadline]:
    """Read the golden JSONL into a list of GoldenHeadline.

    Raises FileNotFoundError if the dataset has not been built yet, and ValueError
    on malformed rows or labels outside the known class set.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Golden dataset not found at {path}. Build it first via the "
            "error-discovery workflow (see plan / README Part 3)."
        )

    out: list[GoldenHeadline] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc

            text = rec.get("text")
            label = rec.get("ground_truth")
            if not text or label not in LABEL2ID:
                raise ValueError(
                    f"{path}:{lineno} needs a non-empty 'text' and a 'ground_truth' "
                    f"in {sorted(LABEL2ID)}; got text={text!r}, ground_truth={label!r}."
                )
            out.append(
                GoldenHeadline(
                    text=text,
                    ground_truth=label,
                    rationale=rec.get("rationale"),
                    ticker=rec.get("ticker"),
                    source=rec.get("source"),
                )
            )

    if not out:
        raise ValueError(f"Golden dataset at {path} is empty.")
    return out
