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
def test_incrementality_is_negative_and_material(raw: dict[str, pd.DataFrame]) -> None:
    result = monetization.holdout_incrementality(raw)

    assert result["incrementality"] < 0, "displacement outweighs the promoted listing's gain"
    assert -0.030 <= result["incrementality"] <= -0.010
    assert result["incrementality"] == pytest.approx(
        result["contacts_per_session_treated"] - result["contacts_per_session_holdout"],
        abs=TOLERANCE,
    )
    # The bootstrap CI must bracket the point estimate it is built on. The holdout
    # arm is only ~10% of sessions, so the interval is wide -- the sign of the point
    # estimate is the finding, not the significance of the interval.
    assert result["ci_95_low"] <= result["incrementality"] <= result["ci_95_high"]
    assert 0.0 <= result["welch_p_value"] <= 1.0
    assert result["sessions_treated"] > result["sessions_holdout"]


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
def test_unpaid_contacts_fall_when_promoted_slots_are_shown(
    raw: dict[str, pd.DataFrame],
) -> None:
    guard = monetization.guardrails(raw)["unpaid_contacts_per_session"]

    assert guard["treated"] < guard["holdout"]
    drop = (guard["holdout"] - guard["treated"]) / guard["holdout"]
    assert 0.07 <= drop <= 0.14, f"unpaid contacts fell {drop:.1%}, expected 7-14%"
    assert guard["relative_change"] < 0


def test_paid_traffic_converts_to_a_conversation_less_often(
    raw: dict[str, pd.DataFrame],
) -> None:
    reply = monetization.guardrails(raw)["seller_reply_rate"]

    assert reply["paid"] < reply["unpaid"]
    assert reply["paid"] == pytest.approx(0.5662, abs=2e-3)
    assert reply["unpaid"] == pytest.approx(0.6436, abs=2e-3)


def test_guardrails_are_reported_as_absolute_numbers(raw: dict[str, pd.DataFrame]) -> None:
    guard = monetization.guardrails(raw)

    for key in ("treated", "holdout", "absolute_change", "relative_change"):
        assert key in guard["unpaid_contacts_per_session"]
    for key in ("paid", "unpaid", "absolute_change", "relative_change"):
        assert key in guard["seller_reply_rate"]


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
    assert verdict["applied_rules"]
    # The guardrail row is the rule that must dominate.
    assert "unpaid contacts" in verdict["applied_rules"][0]["outcome"].lower()
    # The primary is negative, so the pricing/cap rule is reported alongside it.
    assert any("cap slots per serp" in rule["decision"].lower() for rule in verdict["applied_rules"])
