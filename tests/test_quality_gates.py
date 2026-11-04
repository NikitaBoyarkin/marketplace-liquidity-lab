"""Fault injection for every quality gate.

A gate that has never failed is indistinguishable from a gate that cannot fail. The two
gates this suite replaced both belonged to that class: one asserted ``distinct_pairs <=
rows`` (true for any frame) and one compared two config constants with each other. So
each gate now carries an injection case here: mutate one frame in a way the gate exists to
catch, and require the gate to go red.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from liquidity import config, etl, quality

Raw = dict[str, pd.DataFrame]


@pytest.fixture(scope="module")
def raw() -> Raw:
    return etl.load_raw()


def _copy(raw: Raw) -> Raw:
    return {name: frame.copy() for name, frame in raw.items()}


def _break_referential(raw: Raw) -> Raw:
    frames = _copy(raw)
    frames["impressions"].loc[frames["impressions"].index[0], "search_id"] = -1
    return frames


def _break_temporal(raw: Raw) -> Raw:
    frames = _copy(raw)
    removed = frames["listings"]["status"].fillna("") == "removed"
    index = frames["listings"].index[removed][0]
    frames["listings"].loc[index, "removal_day"] = (
        frames["listings"].loc[index, "publish_day"] - 1.0
    )
    return frames


def _break_duplicates(raw: Raw) -> Raw:
    frames = _copy(raw)
    goods = frames["listings"]["category_id"] == "goods"
    frames["listings"].loc[goods, "object_id"] = 0  # object_id is int64; 0 collides with a real duplicate group
    return frames


def _break_contact_density(raw: Raw) -> Raw:
    frames = _copy(raw)
    contacts = frames["contacts"]
    contacts.loc[contacts.index[:10], "buyer_id"] = 1
    contacts.loc[contacts.index[:10], "listing_id"] = contacts.loc[contacts.index[0], "listing_id"]
    return frames


def _break_window(raw: Raw) -> Raw:
    frames = _copy(raw)
    frames["searches"]["day"] = np.minimum(frames["searches"]["day"], 10.0)
    return frames


def _break_holdout(raw: Raw) -> Raw:
    frames = _copy(raw)
    searches = frames["searches"]
    searches["in_holdout"] = 0
    searches.loc[searches.index[: int(len(searches) * 0.01)], "in_holdout"] = 1
    return frames


def _break_domains(raw: Raw) -> Raw:
    frames = _copy(raw)
    impressions = frames["impressions"]
    impressions.loc[impressions.index[0], "position"] = 0
    impressions.loc[impressions.index[1], "is_promoted"] = 2
    frames["listings"].loc[frames["listings"].index[0], "price_rub"] = 0.0
    frames["contacts"].loc[frames["contacts"].index[0], "seller_replied"] = 7
    return frames


def _break_funnel(raw: Raw) -> Raw:
    frames = _copy(raw)
    searches = frames["searches"]
    searches.loc[searches.index[0], "n_results"] = searches.loc[searches.index[0], "n_results"] + 5
    return frames


INJECTIONS: dict[str, Callable[[Raw], Raw]] = {
    "referential_integrity": _break_referential,
    "temporal_order": _break_temporal,
    "duplicate_rate": _break_duplicates,
    "contact_density": _break_contact_density,
    "window_coverage": _break_window,
    "holdout_share": _break_holdout,
    "value_domains": _break_domains,
    "funnel_consistency": _break_funnel,
}


def test_every_gate_has_an_injection_case(raw: Raw) -> None:
    """The registry is the mechanism: a new gate without a case fails this test."""
    registered = {result.name for result in quality.run_all(raw)}
    assert registered == set(INJECTIONS), (
        "every quality gate needs a fault-injection case in this file; "
        f"gates={sorted(registered)} cases={sorted(INJECTIONS)}"
    )


@pytest.mark.parametrize("gate", sorted(INJECTIONS))
def test_gate_goes_red_on_its_own_injected_defect(gate: str, raw: Raw) -> None:
    results = {result.name: result for result in quality.run_all(INJECTIONS[gate](raw))}

    assert results[gate].passed is False, f"{gate} stayed green on a defect it exists to catch"
    assert results[gate].detail, "a failing gate must say what it measured"


@pytest.mark.parametrize("gate", sorted(INJECTIONS))
def test_gate_is_green_on_the_real_log(gate: str, raw: Raw) -> None:
    results = {result.name: result for result in quality.run_all(raw)}
    assert results[gate].passed is True, f"{gate} fails on the frozen log: {results[gate].detail}"


def test_gate_thresholds_live_in_config() -> None:
    """No module keeps its own threshold: that is how the axis silently diverges."""
    source = Path(quality.__file__).read_text(encoding="utf-8")

    assert "MAX_DUPLICATE_SHARE = 0.2" not in source
    for name in (
        "config.MAX_DUPLICATE_SHARE",
        "config.CONTACT_REPEAT_MAX",
        "config.HOLDOUT_SHARE",
        "config.HOLDOUT_SHARE_TOLERANCE",
    ):
        assert name in source, f"quality.py must read {name}"


def test_contact_density_measures_repeats_not_row_counts(raw: Raw) -> None:
    """The invariant must be about the data: repeats per pair, not rows >= distinct."""
    result = {r.name: r for r in quality.run_all(raw)}["contact_density"]

    assert result.detail["max_contacts_per_pair"] <= config.CONTACT_REPEAT_MAX
    assert (
        result.detail["distinct_pairs"] + result.detail["repeated_pairs"] == result.detail["rows"]
    )
    assert result.detail["max_allowed"] == config.CONTACT_REPEAT_MAX


def test_window_coverage_reads_the_observed_horizon(raw: Raw) -> None:
    """The horizon comes from the log, so truncating the export fails the gate."""
    result = {r.name: r for r in quality.run_all(raw)}["window_coverage"]

    assert result.detail["observed_horizon_day"] == pytest.approx(
        raw["searches"]["day"].max(), abs=1e-9
    )
    truncated = _break_window(raw)
    broken = {r.name: r for r in quality.run_all(truncated)}["window_coverage"]
    assert broken.detail["violations"] > 0
    assert broken.detail["observed_horizon_day"] < broken.detail["latest_required_end"]
