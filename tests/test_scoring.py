"""The README audit table must be reproducible with one command.

The claim under test is not "the estimator is good" (that is
`test_bias_recovery.py`) but "the numbers printed in `README.md` are what the
scorer prints". Without this test the table could drift from the code while every
other test stayed green — the exact failure mode `liquidity/scoring.py` exists to
prevent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from liquidity import scoring

README = Path(__file__).resolve().parent.parent / "README.md"
TABLE_HEADER = "| Category | True | Naive | Naive error | Corrected | Corrected error |"


def _readme_rows() -> dict[str, list[float]]:
    """Parse the markdown audit table into {category: [five numbers]}."""
    text = README.read_text(encoding="utf-8")
    start = text.index(TABLE_HEADER)
    rows: dict[str, list[float]] = {}
    for line in text[start:].splitlines():
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 6 or cells[0] in {"Category", "---"} or set(cells[0]) <= {"-", ":"}:
            continue
        name = cells[0].replace("**", "").replace("`", "").strip()
        rows[name] = [
            float(cell.replace("**", "").replace("−", "-").replace("+", "")) for cell in cells[1:]
        ]
    return rows


@pytest.fixture(scope="module")
def table():
    return scoring.scored_table()


def test_readme_table_is_parsed_at_all():
    """Guards the parser: a silently empty table would make the next test vacuous."""
    rows = _readme_rows()
    assert set(rows) == {"goods", "services", "realty", "auto", "jobs", "platform"}


def test_readme_numbers_match_the_scorer(table):
    rows = _readme_rows()
    by_category = table.set_index("category")
    cols = ["true", "naive", "naive_error", "corrected", "corrected_error"]

    mismatches: list[str] = []
    for category, values in rows.items():
        computed = [round(float(by_category.loc[category, col]), 4) for col in cols]
        printed = [round(value, 4) for value in values]
        if computed != printed:
            mismatches.append(f"{category}: README {printed} vs scorer {computed}")
    assert not mismatches, "README table drifted from the scorer: " + "; ".join(mismatches)


def test_scorer_prints_every_column_it_promises(table):
    assert list(table.columns) == list(scoring.COLUMNS)


def test_platform_truth_is_the_weighted_aggregate(table):
    """The platform row is the planted aggregate, not a re-derived mean."""
    truth = scoring.load_truth()
    platform = table.set_index("category").loc["platform"]
    assert platform["true"] == pytest.approx(truth["latent"]["platform_true"], abs=1e-9)
    assert platform["naive_error"] == pytest.approx(platform["naive"] - platform["true"], abs=1e-9)


def test_h1a_arithmetic_is_reproducible_from_the_scorer(table):
    """The falsification of H1a is arithmetic over the table, not prose."""
    per_category = table[table["category"] != "platform"]
    platform_error = abs(float(table.set_index("category").loc["platform", "naive_error"]))
    mean_abs_category_error = float(per_category["naive_error"].abs().mean())
    assert platform_error > mean_abs_category_error, "H1a would then hold, and README is wrong"
    assert platform_error == pytest.approx(0.0657, abs=1e-4)
    assert mean_abs_category_error == pytest.approx(0.0544, abs=1e-4)


def test_readme_does_not_claim_the_analysis_cli_prints_the_truth():
    """The truth columns come from the scorer; `liquidity.bias` must not be sold as printing them."""
    text = README.read_text(encoding="utf-8")
    assert "python -m liquidity.scoring" in text, "README must point at the scorer"
    assert (
        re.search(r"`uv run python -m liquidity\.bias` prints next to every estimate", text) is None
    )
