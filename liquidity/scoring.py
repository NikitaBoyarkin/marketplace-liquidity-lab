"""Score the estimators against the planted answer key.

This is the **only** module in the package allowed to open
``_truth/ground_truth.json``. Every analysis module stays truth-free; the leak
test enforces that the reader set is exactly ``{"scoring.py"}``, so widening it
is a visible, reviewed edit rather than a quiet import.

The point of separating the scorer from the analysis is that the audit table in
``README.md`` becomes reproducible with one command:

    uv run python -m liquidity.scoring

Without this module the table's ``true`` and ``error`` columns could only be
rebuilt by opening the answer key by hand, which is exactly what the analysis is
forbidden from doing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from liquidity import config
from liquidity.bias import compare, load_cohort

PROJECT_ROOT = config.PROJECT_ROOT
TRUTH_PATH = PROJECT_ROOT / "_truth" / "ground_truth.json"

COLUMNS = (
    "category",
    "cohort_size",
    "true",
    "naive",
    "naive_error",
    "corrected",
    "corrected_error",
    "naive_ci_low",
    "naive_ci_high",
    "corrected_ci_low",
    "corrected_ci_high",
)


def load_truth(truth_path: Path | str = TRUTH_PATH) -> dict[str, Any]:
    """Read the frozen answer key. Only this module does, by design."""
    return json.loads(Path(truth_path).read_text(encoding="utf-8"))


def truth_by_category(truth: dict[str, Any] | None = None) -> dict[str, float]:
    """Latent sell-through per category plus the platform aggregate."""
    truth = load_truth() if truth is None else truth
    latent = truth["latent"]
    values = dict(latent["sell_through_true_by_category"])
    values["platform"] = float(latent["platform_true"])
    return values


def scored_table(
    df: pd.DataFrame | None = None,
    truth: dict[str, Any] | None = None,
    n_iters: int = 300,
) -> pd.DataFrame:
    """Estimates next to the truth they are scored against.

    ``naive`` and ``corrected`` come from ``bias.compare`` (observable data only);
    ``true`` comes from the answer key. Errors are computed from unrounded
    values, so hand-subtracting the printed columns can differ in the last digit.
    """
    df = load_cohort() if df is None else df
    truth = load_truth() if truth is None else truth

    estimates = compare(df, n_iters=n_iters)
    true = truth_by_category(truth)

    table = estimates.copy()
    table["true"] = table["category"].map(true)
    table["naive_error"] = table["naive"] - table["true"]
    table["corrected_error"] = table["corrected"] - table["true"]
    return table[list(COLUMNS)]


def main() -> None:
    """Print the audit table: estimates, truth, and both errors."""
    table = scored_table()
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
