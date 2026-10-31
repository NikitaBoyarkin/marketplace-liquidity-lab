"""Does the mixture correction recover the latent sell-through?

These tests may read the planted answer key — that is the point of the separation:
`liquidity/bias.py` cannot, and everything it estimates is scored here against it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from liquidity import bias, config

TOL = config.TOL_RECOVERY
TRUTH_FILE = Path(__file__).resolve().parent.parent / "_truth" / "ground_truth.json"


@pytest.fixture(scope="module")
def truth() -> dict:
    doc = json.loads(TRUTH_FILE.read_text(encoding="utf-8"))
    return doc["latent"]


@pytest.fixture(scope="module")
def cohort():
    return bias.load_cohort()


@pytest.fixture(scope="module")
def estimates(cohort):
    return bias.corrected_sell_through(cohort, use_cache=False), bias.naive_sell_through(cohort)


def test_every_category_recovers_within_tolerance(estimates, truth):
    corrected, _ = estimates
    true = truth["sell_through_true_by_category"]
    errors = {c: corrected[c] - true[c] for c in true}
    print("\ncategory  true    corrected  error")
    for c, err in sorted(errors.items()):
        print(f"{c:9s} {true[c]:.4f}  {corrected[c]:.4f}   {err:+.4f}")
    bad = {c: round(e, 4) for c, e in errors.items() if abs(e) > TOL}
    assert not bad, f"categories outside +/-{TOL}: {bad}"


def test_bootstrap_interval_covers_the_truth_in_at_least_four_categories(cohort, truth):
    ci = bias.bootstrap_ci(
        cohort, bias.corrected_sell_through, n_iters=120, alpha=config.BOOTSTRAP_ALPHA
    )
    true = truth["sell_through_true_by_category"]
    covered = [c for c in true if ci[c][0] <= true[c] <= ci[c][1]]
    for c in sorted(true):
        print(f"{c:9s} ci=({ci[c][0]:.4f}, {ci[c][1]:.4f}) true={true[c]:.4f}")
    assert len(covered) >= 4, f"only {covered} covered; intervals: {ci}"


def test_platform_correction_beats_the_proxy(estimates, truth):
    corrected, naive = estimates
    err_corrected = abs(corrected["platform"] - truth["platform_true"])
    err_naive = abs(naive["platform"] - truth["platform_true"])
    print(
        f"\nplatform naive={naive['platform']:.4f} corrected={corrected['platform']:.4f} "
        f"true={truth['platform_true']:.4f} | err naive={err_naive:.4f} "
        f"corrected={err_corrected:.4f}"
    )
    assert err_corrected < err_naive


def test_compare_table_shape(cohort):
    table = bias.compare(cohort, n_iters=40)
    assert list(table.columns) == [
        "category",
        "cohort_size",
        "naive",
        "corrected",
        "naive_ci_low",
        "naive_ci_high",
    ]
    assert table["category"].iloc[-1] == "platform"
    assert table["cohort_size"].iloc[-1] == int(len(cohort))
    assert (table["naive_ci_low"] <= table["naive"]).all()
    assert (table["naive"] <= table["naive_ci_high"]).all()
