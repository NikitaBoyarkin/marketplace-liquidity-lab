"""Behavioural tests for the monetisation leg.

Two estimators answer "did paid visibility create contacts?". The naive pre/post
says yes; the randomised holdout says the gains were relocated, not created, and
the organic guardrail is dirty. Every number asserted here was fixed before the
modules were written.
"""

from __future__ import annotations

import pandas as pd
import pytest

from liquidity import metrics, monetization

TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def raw() -> dict[str, pd.DataFrame]:
    return metrics.load()


# --------------------------------------------------------------------------- #
# The causal contrast
# --------------------------------------------------------------------------- #
def test_incrementality_is_negative_and_not_significant(raw: dict[str, pd.DataFrame]) -> None:
    """The sign is the finding; the interval is the caveat.

    This test used to pin the point estimate into ``[-0.030, -0.010]``, which demanded a
    "material" effect from a p = 0.31 readout. Sign, interval coverage and significance
    are the claims the data can support; the width is asserted separately, against the
    MDE, in ``test_power_is_reported_and_the_effect_is_below_the_mde``.
    """
    result = monetization.holdout_incrementality(raw)

    assert result["incrementality"] < 0, "displacement outweighs the promoted listing's gain"
    assert result["incrementality"] == pytest.approx(
        result["contacts_per_session_treated"] - result["contacts_per_session_holdout"],
        abs=TOLERANCE,
    )
    assert result["ci_95_low"] <= result["incrementality"] <= result["ci_95_high"]
    assert 0.0 <= result["welch_p_value"] <= 1.0
    assert result["welch_p_value"] > 0.05, "the effect is not significant; do not report it as proven"
    assert result["significant_at_alpha"] is False
    assert result["sessions_treated"] > result["sessions_holdout"]


def test_power_is_reported_and_the_effect_is_below_the_mde(
    raw: dict[str, pd.DataFrame],
) -> None:
    """A readout smaller than its own MDE cannot distinguish "no effect" from "effect"."""
    result = monetization.holdout_incrementality(raw)

    assert result["mde_80"] > 0
    assert result["standard_error"] > 0
    assert result["mde_80"] == pytest.approx(2.801585 * result["standard_error"], rel=1e-4)
    assert abs(result["incrementality"]) < result["mde_80"]
    assert result["underpowered_for_its_own_effect"] is True
    assert 0.05 <= result["mde_relative_to_holdout"] <= 0.09


def test_srm_gate_passes_on_the_real_split_and_fails_on_drift(raw: dict[str, pd.DataFrame]) -> None:
    """The 10% holdout is a design assumption; a silent drift must be visible."""
    result = monetization.holdout_incrementality(raw)
    srm = result["srm"]

    assert srm["planned_holdout_share"] == pytest.approx(0.10)
    assert srm["observed_holdout_share"] == pytest.approx(0.1002, abs=1e-3)
    assert srm["p_value"] > 0.01
    assert srm["passes_at_alpha"] is True

    drifted = monetization.sample_ratio_check(treated_sessions=98_000, holdout_sessions=2_000)
    assert drifted["passes_at_alpha"] is False, "a 2% holdout against a planned 10% must fail"


def test_incrementality_rates_match_the_frozen_log(raw: dict[str, pd.DataFrame]) -> None:
    result = monetization.holdout_incrementality(raw)
    assert result["contacts_per_session_treated"] == pytest.approx(0.4581, abs=2e-4)
    assert result["contacts_per_session_holdout"] == pytest.approx(0.4697, abs=2e-4)


def test_zero_result_sessions_are_excluded_from_the_denominator() -> None:
    searches = pd.DataFrame(
        {
            "search_id": [1, 2],
            "in_holdout": [0, 1],
            "n_results": [10, 0],
        }
    )
    contacts = pd.DataFrame(
        {
            "contact_id": [1],
            "search_id": [1],
            "listing_id": [5],
            "buyer_id": [9],
            "is_promoted": [0],
            "seller_replied": [1],
            "day": [1.0],
        }
    )
    frame = monetization.session_contacts({"searches": searches, "contacts": contacts})
    assert frame["search_id"].tolist() == [1], "a session with no results cannot deliver a contact"
    assert frame["contacts"].tolist() == [1.0]


# --------------------------------------------------------------------------- #
# The guardrails
# --------------------------------------------------------------------------- #
def test_organic_contacts_fall_when_promoted_slots_are_shown(
    raw: dict[str, pd.DataFrame],
) -> None:
    guard = monetization.guardrails(raw)["displacement"]["organic_contacts_per_session"]

    assert guard["treated"] < guard["holdout"]
    drop = (guard["holdout"] - guard["treated"]) / guard["holdout"]
    assert 0.07 <= drop <= 0.14, f"organic contacts fell {drop:.1%}, expected 7-14%"
    assert guard["relative_change"] < 0
    assert guard["ci_95_high"] < 0, "the fall is distinguishable from zero"


def test_displacement_is_decomposed_into_slots_and_conversion(
    raw: dict[str, pd.DataFrame],
) -> None:
    """Contacts per session falls for two reasons; the guardrail must not merge them."""
    displacement = monetization.guardrails(raw)["displacement"]

    slots = displacement["organic_impressions_per_session"]
    conversion = displacement["contacts_per_organic_impression"]

    assert slots["relative_change"] < 0, "a promoted slot occupies an organic one"
    assert conversion["relative_change"] < 0
    assert conversion["ci_95_high"] < 0, "conversion per impression also falls, not just the count"
    share = displacement["slot_arithmetic_share_of_drop"]
    assert 0.30 <= share <= 0.55, f"slot arithmetic explains {share:.0%} of the drop"
    # Both drops are needed to reconstruct the headline: neither replaces the other.
    assert slots["treated"] > 10 and conversion["treated"] > 0.02


def test_top_organic_positions_absorb_the_displacement(raw: dict[str, pd.DataFrame]) -> None:
    """The stated mechanism is "organic is pushed down", so the mix must show it."""
    mix = {row["position_bucket"]: row for row in monetization.guardrails(raw)["displacement"]["by_position"]}

    assert set(mix) == {"1-3", "4-6", "7-10", "11-20"}
    top = mix["1-3"]["relative_change"]
    assert top < -0.15, f"top-3 organic impressions change {top:.1%}, expected a large fall"
    for bucket in ("4-6", "7-10", "11-20"):
        assert mix[bucket]["relative_change"] > top, "lower buckets suffer less than the top"


def test_reply_rate_has_a_causal_and_a_descriptive_version(raw: dict[str, pd.DataFrame]) -> None:
    """Paid vs unpaid compares different listings; only the holdout contrast is causal."""
    guard = monetization.guardrails(raw)
    causal = guard["reply_rate_causal"]
    descriptive = guard["reply_rate_descriptive"]

    assert descriptive["paid"] == pytest.approx(0.5662, abs=2e-3)
    assert descriptive["unpaid"] == pytest.approx(0.6436, abs=2e-3)
    assert descriptive["absolute_change"] < -0.05

    # The causal version is flat: on the same listings, suppression barely moves replies.
    assert abs(causal["absolute_change"]) < 0.01
    assert causal["ci_95_low"] < 0 < causal["ci_95_high"]
    assert causal["contacts_treated"] > causal["contacts_holdout"]


def test_guardrails_are_reported_as_absolute_numbers(raw: dict[str, pd.DataFrame]) -> None:
    guard = monetization.guardrails(raw)

    for key in ("treated", "holdout", "absolute_change", "relative_change", "ci_95_low", "ci_95_high"):
        assert key in guard["displacement"]["organic_contacts_per_session"]
        assert key in guard["displacement"]["contacts_per_organic_impression"]
        assert key in guard["reply_rate_causal"]
    for key in ("paid", "unpaid", "absolute_change", "relative_change"):
        assert key in guard["reply_rate_descriptive"]


def test_the_two_contrasts_are_not_mixed_under_one_label(raw: dict[str, pd.DataFrame]) -> None:
    """Regression guard: the old table labelled a paid/unpaid cross-section as holdout."""
    guard = monetization.guardrails(raw)

    assert "seller_reply_rate" not in guard
    assert "unpaid_contacts_per_session" not in guard
    assert set(guard) == {"displacement", "reply_rate_causal", "reply_rate_descriptive"}


# --------------------------------------------------------------------------- #
# The naive estimator and its confound
# --------------------------------------------------------------------------- #
def test_naive_prepost_is_positive_because_of_selection(raw: dict[str, pd.DataFrame]) -> None:
    frame = monetization.naive_prepost(raw)

    assert frame.attrs["lift_ratio"] > 1.0, "the boosted listing does gain contacts"
    assert frame.attrs["aggregate_post"] > frame.attrs["aggregate_pre"]
    assert frame.attrs["n_listings"] == int(
        raw["listings"]["boost_purchased"].sum()
    )
    assert len(frame) == frame.attrs["n_listings"]
    assert (frame["lift"] == frame["contacts_post"] - frame["contacts_pre"]).all()


def test_naive_and_holdout_disagree_in_sign(raw: dict[str, pd.DataFrame]) -> None:
    naive_ratio = monetization.naive_prepost(raw).attrs["lift_ratio"]
    incrementality = monetization.holdout_incrementality(raw)["incrementality"]

    assert naive_ratio > 1.0
    assert incrementality < 0


# --------------------------------------------------------------------------- #
# The decision rule belongs to the preregistration
# --------------------------------------------------------------------------- #
def test_preregistration_declares_the_guardrail_condition() -> None:
    rules = monetization.read_decision_rules()
    outcomes = [rule["outcome"] for rule in rules]

    assert rules, "the decision-rule table must be parseable"
    assert any("unpaid contacts" in outcome.lower() for outcome in outcomes)
    assert any("H2b < 0" in outcome for outcome in outcomes)


def test_decision_rules_come_from_the_document_not_the_code() -> None:
    document = monetization.PREREGISTRATION_PATH.read_text(encoding="utf-8")
    rules = monetization.read_decision_rules()

    for rule in rules:
        assert rule["outcome"] in document
        assert rule["decision"] in document


def test_decision_rolls_back_on_a_dirty_guardrail(raw: dict[str, pd.DataFrame]) -> None:
    verdict = monetization.decision(raw)

    assert verdict["guardrail_dirty"] is True
    recommendation = verdict["recommendation"].lower()
    assert "roll back" in recommendation or "cap" in recommendation
    assert verdict["reason"]
    assert verdict["preregistration"].endswith("preregistration.md")
    # The decision carries the decomposition, so a reader can see how much of the
    # "dirty guardrail" is arithmetic rather than a conversion effect.
    assert verdict["guardrail_decomposition"]["slot_arithmetic_share_of_drop"] > 0
    assert verdict["applied_rules"]
    # The guardrail row is the rule that must dominate.
    assert "unpaid contacts" in verdict["applied_rules"][0]["outcome"].lower()
    # The primary is negative, so the pricing/cap rule is reported alongside it.
    assert any("cap slots per serp" in rule["decision"].lower() for rule in verdict["applied_rules"])
