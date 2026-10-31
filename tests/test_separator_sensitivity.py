"""The correction is only as good as its separator, and this test says so out loud.

The EM recovery in `bias.py` rests on one structural assumption: a sale shows up as a replied
contact followed by a removal within a few days. That assumption is also how the generator
decides which listings sold, so "the correction recovers the truth" is a specification check,
not an independent validation. These tests bind the limitation to a mechanism — if someone
widens the cutoff or drops the separator and the suite stays green, the README is lying.
"""

from __future__ import annotations

import json

import pytest

from liquidity import bias, config

TOL = config.TOL_RECOVERY
# Wide enough that `replied_short` collapses into plain `has_replied_contact`, i.e. the
# separator the preregistration actually specified.
NO_SEPARATOR = 1e9


@pytest.fixture(scope="module")
def scored() -> tuple[dict[str, float], dict[str, float]]:
    truth = json.loads((config.PROJECT_ROOT / "_truth" / "ground_truth.json").read_text())
    return truth["latent"]["sell_through_true_by_category"], truth["latent"]["platform_true"]


def _errors(corrected: dict[str, float], truth: dict[str, float]) -> dict[str, float]:
    return {cat: corrected[cat] - truth[cat] for cat in truth}


def test_without_the_separator_the_correction_fails(monkeypatch, scored) -> None:
    truth, _ = scored
    cohort = bias.load_cohort()
    monkeypatch.setattr(bias, "SOLD_LAG_MAX", NO_SEPARATOR)

    errors = _errors(bias.corrected_sell_through(cohort, use_cache=False), truth)
    failing = [cat for cat, err in errors.items() if abs(err) > TOL]

    assert len(failing) >= 3, (
        "dropping the short-lag separator should break the recovery; "
        f"it did not, so the separator carries no weight: {errors}"
    )


@pytest.mark.parametrize("cutoff", [4.0, 6.0])
def test_recovery_holds_across_the_working_band(monkeypatch, scored, cutoff: float) -> None:
    truth, _ = scored
    cohort = bias.load_cohort()
    monkeypatch.setattr(bias, "SOLD_LAG_MAX", cutoff)

    errors = _errors(bias.corrected_sell_through(cohort, use_cache=False), truth)

    assert all(abs(err) <= TOL for err in errors.values()), errors


def test_the_separator_cutoff_is_not_generic(monkeypatch, scored) -> None:
    """A cutoff far from the working band must degrade the estimate, otherwise the
    parameter would not be load-bearing and the sensitivity claim above would be empty."""
    truth, _ = scored
    cohort = bias.load_cohort()
    monkeypatch.setattr(bias, "SOLD_LAG_MAX", 2.0)

    errors = _errors(bias.corrected_sell_through(cohort, use_cache=False), truth)

    assert max(abs(err) for err in errors.values()) > TOL
