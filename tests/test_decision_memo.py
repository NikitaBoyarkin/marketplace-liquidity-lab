"""The decision memo must quote the code, not yesterday's run.

`docs/decision-memo.md` is the artefact a product owner reads instead of the notebooks.
If its numbers drift from `liquidity/monetization.py` and `liquidity/scoring.py`, the
memo becomes a second, silently stale source of truth — the exact failure this repo
spends its tests on. These tests are the binding.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from liquidity import metrics, monetization, scoring

MEMO = Path(__file__).resolve().parent.parent / "docs" / "decision-memo.md"
REQUIRED_SECTIONS = (
    "## Вопрос",
    "## Рекомендация",
    "## Доказательство",
    "## Цена",
    "## Порог отката",
    "## Владелец и следующий шаг",
    "## Что изменит решение",
)


@pytest.fixture(scope="module")
def memo_text() -> str:
    return MEMO.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def normalized(memo_text: str) -> str:
    return memo_text.replace("−", "-").replace("\u00a0", " ")


@pytest.fixture(scope="module")
def raw() -> dict:
    return metrics.load()


def test_memo_exists_and_carries_every_required_section(memo_text: str) -> None:
    assert MEMO.exists()
    for section in REQUIRED_SECTIONS:
        assert section in memo_text, f"missing section {section!r}"


def test_memo_quotes_the_live_decision(raw: dict, normalized: str) -> None:
    verdict = monetization.decision(raw)
    assert "откат" in verdict["recommendation"].lower() or "roll back" in verdict["recommendation"]
    # The memo states the recommendation in its own words, but it must also name the
    # rule that produced it, and that rule text comes from the preregistration.
    assert verdict["applied_rules"][0]["outcome"] in normalized
    assert verdict["guardrail_dirty"] is True


def test_memo_numbers_match_the_code(raw: dict, normalized: str) -> None:
    incrementality = monetization.holdout_incrementality(raw)
    displacement = monetization.guardrails(raw)["displacement"]

    expected = {
        "incrementality": f"{incrementality['incrementality']:.4f}",
        "mde_80": f"{incrementality['mde_80']:.4f}",
        "mde_relative": f"{incrementality['mde_relative_to_holdout']:.1%}".replace("-", ""),
        "organic_contacts_relative": f"{abs(displacement['organic_contacts_per_session']['relative_change']):.1%}",
        "slot_share": f"{displacement['slot_arithmetic_share_of_drop']:.0%}",
        "org_impressions_relative": f"{abs(displacement['organic_impressions_per_session']['relative_change']):.1%}",
        "conversion_relative": f"{abs(displacement['contacts_per_organic_impression']['relative_change']):.1%}",
    }
    missing = {name: value for name, value in expected.items() if value not in normalized}
    assert not missing, f"memo is stale against the code: {missing}"


def test_memo_position_number_matches_the_guardrail(raw: dict, normalized: str) -> None:
    mix = {
        row["position_bucket"]: row
        for row in monetization.guardrails(raw)["displacement"]["by_position"]
    }
    top = f"{abs(mix['1-3']['relative_change']):.1%}"
    assert top in normalized, f"top-3 fall is {top}, memo says something else"
    assert mix["1-3"]["relative_change"] < min(
        mix[bucket]["relative_change"] for bucket in ("4-6", "7-10", "11-20")
    )


def test_memo_sell_through_numbers_match_the_scorer(normalized: str) -> None:
    table = scoring.scored_table().set_index("category")
    platform = table.loc["platform"]
    per_category = table.drop(index="platform")

    assert f"{platform['naive']:.4f}" in normalized
    assert f"{platform['true']:.4f}" in normalized
    assert f"{platform['naive_error']:.4f}" in normalized
    assert f"{per_category['naive_error'].abs().mean():.4f}" in normalized
    assert f"{per_category['corrected_error'].abs().max():.4f}" in normalized


def test_memo_does_not_invent_a_slot_limit(normalized: str) -> None:
    """The cap is the one number the log cannot produce; the memo must say so."""
    assert "лимит слотов в этом логе не измерялся" in normalized.lower()
    # And it must separate what was measured from what is recommended.
    assert "0 / 1 / 2 / 3 промо-слота" in normalized, (
        "the pilot design is the only place slots appear"
    )


def test_memo_refuses_to_price_the_effect_in_rubles(normalized: str) -> None:
    assert "биллинг" in normalized
    assert "руб/день" in normalized
    assert "цена контакта" in normalized


def test_every_command_named_in_the_memo_resolves(normalized: str) -> None:
    modules = set(re.findall(r"python -m (liquidity\.[a-z_]+)", normalized))
    assert modules, "the memo must name at least one reproducible command"
    for name in sorted(modules):
        importlib.import_module(name)
