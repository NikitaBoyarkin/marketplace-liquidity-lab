"""Figures are artefacts with claims, so both halves get checked.

A PNG in a repository can be three things: a rendering of the current numbers, a
rendering of last week's numbers, or a screenshot of something else entirely. The
distinction is invisible to a reader, which is why this module exists.

Two mechanisms:

* **The sidecar.** `reports/figures/figure_data.json` holds every number each PNG drew.
  A fresh build is compared against the committed file, so a figure that no longer
  matches the code fails the suite instead of quietly aging.
* **The values.** Each builder returns what it plotted; the assertions below compare
  those values to `liquidity.metrics`, `liquidity.monetization` and `liquidity.scoring`
  directly, so a figure cannot draw something the module does not say.

PNG bytes are *not* compared: rasterisation differs between the maintainer's macOS and
a Linux runner, so a byte gate would fail for reasons that have nothing to do with the
numbers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.image as mpimg
import pytest

from liquidity import bias, config, metrics, monetization, scoring

ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT / "reports" / "figures"
SIDECAR = FIGURE_DIR / "figure_data.json"
REPORT = ROOT / "reports" / "report.md"
README = ROOT / "README.md"

EXPECTED_FIGURES = {
    "01_sell_through_audit",
    "02_error_vs_tolerance",
    "03_separator_sensitivity",
    "05_boost_waterfall",
    "04_incrementality_vs_mde",
    "06_position_mix",
}


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    """Build every figure once, into a temporary directory (no repo side effects)."""
    out_dir = tmp_path_factory.mktemp("figures")
    return out_dir, scoring.build_figures(out_dir)


# --------------------------------------------------------------------------- #
# the files
# --------------------------------------------------------------------------- #
def test_every_figure_is_rendered(built: tuple[Path, dict]) -> None:
    """Six figures, each a real raster at print resolution — not a stub or a 1px placeholder."""
    out_dir, payload = built
    assert set(payload) == EXPECTED_FIGURES

    for name in EXPECTED_FIGURES:
        path = out_dir / f"{name}.png"
        assert path.exists(), f"{name}.png was not written"
        assert path.stat().st_size > 30_000, f"{name}.png is suspiciously small"
        image = mpimg.imread(path)
        height, width = image.shape[:2]
        assert width >= 1200 and height >= 700, f"{name}.png is {width}x{height}, too small to read"


def test_the_payload_sidecar_is_written_next_to_the_figures(built: tuple[Path, dict]) -> None:
    out_dir, payload = built
    sidecar = out_dir / "figure_data.json"
    assert sidecar.exists()
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    assert document["figures"] == payload
    assert document["generated_by"].startswith("uv run python -m liquidity.scoring")


def test_committed_sidecar_matches_a_fresh_build(built: tuple[Path, dict]) -> None:
    """The freshness gate: a PNG whose numbers no longer match the code is stale.

    PNG bytes cannot be diffed across platforms, but the numbers can. Rerun
    `make figures` after a change and commit the sidecar with the images.
    """
    _, payload = built
    assert SIDECAR.exists(), "reports/figures/figure_data.json is missing — run `make figures`"
    committed = json.loads(SIDECAR.read_text(encoding="utf-8"))
    assert committed["figures"] == payload, (
        "the committed figures are stale: rerun `make figures` and commit "
        "reports/figures/ (PNGs + figure_data.json)"
    )


# --------------------------------------------------------------------------- #
# the numbers each figure draws
# --------------------------------------------------------------------------- #
def test_observable_figures_quote_the_modules(built: tuple[Path, dict]) -> None:
    _, payload = built
    frames = metrics.load()
    displacement = monetization.guardrails(frames)["displacement"]

    waterfall = payload["05_boost_waterfall"]
    assert waterfall["slot_arithmetic_pct"] == round(
        displacement["organic_impressions_per_session"]["relative_change"] * 100, 6
    )
    assert waterfall["conversion_pct"] == round(
        displacement["contacts_per_organic_impression"]["relative_change"] * 100, 6
    )
    assert waterfall["total_pct"] == round(
        displacement["organic_contacts_per_session"]["relative_change"] * 100, 6
    )
    assert 0.0 < waterfall["slot_share_of_drop"] < 1.0

    incrementality = monetization.holdout_incrementality(frames)
    forest = payload["04_incrementality_vs_mde"]
    assert forest["incrementality"] == round(incrementality["incrementality"], 6)
    assert forest["ci_95_low"] == round(incrementality["ci_95_low"], 6)
    assert forest["mde_80"] == round(incrementality["mde_80"], 6)
    assert forest["effect_to_mde"] == round(
        abs(incrementality["incrementality"]) / incrementality["mde_80"], 6
    )

    mix = payload["06_position_mix"]
    for row in displacement["by_position"]:
        key = f"position_{row['position_bucket']}_relative_change"
        assert mix[key] == round(row["relative_change"], 6), key


def test_audit_figure_quotes_the_scorer(built: tuple[Path, dict]) -> None:
    """The truth overlay must draw the same cells the printed audit table prints."""
    _, payload = built
    table = scoring.scored_table().set_index("category")
    audit = payload["01_sell_through_audit"]

    for category, cells in audit.items():
        for column in ("naive", "corrected", "true", "naive_error", "corrected_error"):
            assert cells[column] == round(float(table.loc[category, column]), 6), (
                f"{category}.{column} drifted from scored_table()"
            )


def test_error_figure_states_the_tolerance_it_draws(built: tuple[Path, dict]) -> None:
    _, payload = built
    errors = payload["02_error_vs_tolerance"]
    assert errors["tolerance_pp"] == round(config.TOL_RECOVERY * 100, 6)
    assert errors["worst_naive_error_pp"] > errors["tolerance_pp"] * 3
    assert errors["worst_corrected_error_pp"] <= errors["tolerance_pp"]


# --------------------------------------------------------------------------- #
# the claim the sensitivity figure makes
# --------------------------------------------------------------------------- #
def test_separator_sweep_restores_the_module_constant(built: tuple[Path, dict]) -> None:
    """A sweep that leaks its override would silently corrupt every later estimate."""
    assert bias.SOLD_LAG_MAX == 4.5, f"SOLD_LAG_MAX was left at {bias.SOLD_LAG_MAX} by a figure run"


def test_sensitivity_figure_supports_its_own_caption(built: tuple[Path, dict]) -> None:
    """The title claims recovery inside the band and failure outside it. Check both."""
    _, payload = built
    sweep = payload["03_separator_sensitivity"]
    tolerance = sweep["tolerance_pp"]

    # Inside the band no category may miss; outside it, most must — the same bar
    # `tests/test_separator_sensitivity.py` holds the estimator to.
    low, high = sweep["recovery_band_days"]
    assert sweep["recovery_band_worst_error_pp"] <= tolerance

    by_cutoff = sweep["failing_categories_by_cutoff"]
    inside = [count for cutoff, count in by_cutoff.items() if low <= float(cutoff) <= high]
    outside = [count for cutoff, count in by_cutoff.items() if not low <= float(cutoff) <= high]
    assert max(inside) == 0
    assert min(outside) >= 3

    # The band the estimator tests assert (4 d and 6 d) must sit inside the empirical one,
    # otherwise the figure and `tests/test_separator_sensitivity.py` describe different worlds.
    for cutoff in sweep["declared_band_cutoffs_ok"]:
        assert low <= cutoff <= high


# --------------------------------------------------------------------------- #
# the links a reader actually clicks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("document", [REPORT, README], ids=["report", "readme"])
def test_embedded_images_exist(document: Path) -> None:
    """A broken image link is a claim with nothing behind it."""
    text = document.read_text(encoding="utf-8")
    targets = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    local = [target for target in targets if not target.startswith(("http://", "https://"))]
    missing = [target for target in local if not (document.parent / target).resolve().exists()]
    assert not missing, f"{document.name} links images that do not exist: {missing}"


def test_report_embeds_the_audit_and_the_guardrail_figures() -> None:
    """The two figures that carry the decision must be in the evidence document."""
    text = REPORT.read_text(encoding="utf-8")
    for name in ("01_sell_through_audit", "05_boost_waterfall", "04_incrementality_vs_mde"):
        assert name in text, f"reports/report.md never embeds {name}.png"
