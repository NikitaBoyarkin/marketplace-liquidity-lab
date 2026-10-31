"""The mechanism that keeps the repo honest: the proxy must visibly fail.

Three independent ways of failing are asserted — magnitude, pre-declared direction
under resampling, and a source tripwire — so a change that makes the naive estimator
look fine cannot pass silently.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from liquidity import bias, config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREREG = PROJECT_ROOT / "docs" / "preregistration.md"
BIAS_SOURCE = PROJECT_ROOT / "liquidity" / "bias.py"
TRUTH_FILE = PROJECT_ROOT / "_truth" / "ground_truth.json"

# Mirrored from docs/preregistration.md; the test asserts both sides agree, so editing
# the document alone cannot flip a direction without the code disagreeing.
EXPECTED_SIGNS = {
    "goods": "positive",
    "services": "negative",
    "realty": "positive",
    "auto": "negative",
    "jobs": "positive",
}
# Categories whose preregistered sign the data contradicts, with the sign the
# preregistration predicted. Three of the five signs are contradicted (goods, realty, jobs),
# but all three fall under MIN_NAIVE_BIAS, so the assertions below never look at them and the
# record stays empty. Empty is not a claim that the signs agree: the mismatch is reported in
# `reports/report.md`, and declaring it here would mean asserting on sub-threshold noise.
PREREGISTERED_FALSIFIED: dict[str, str] = {}
FORBIDDEN_IN_ANALYSIS = ("_truth", "ground_truth", "TRUTH_PATH", "liquidity.generate")


def _parse_prereg_signs() -> dict[str, str]:
    text = PREREG.read_text(encoding="utf-8")
    return dict(
        re.findall(
            r"\|\s*[^|\n]*\(`([a-z]+)`\)\s*\|\s*\*\*(positive|negative)\*\*", text
        )
    )


def _truth() -> dict:
    return json.loads(TRUTH_FILE.read_text(encoding="utf-8"))["latent"]


def test_preregistered_signs_match_the_code_mirror():
    parsed = _parse_prereg_signs()
    assert parsed == EXPECTED_SIGNS, f"preregistration says {parsed}"


def test_at_least_two_categories_are_far_from_the_truth():
    truth, cohort = _truth(), bias.load_cohort()
    naive = bias.naive_sell_through(cohort)
    true = truth["sell_through_true_by_category"]
    bias_by_cat = {c: naive[c] - true[c] for c in true}
    print("\ncategory  naive   true    bias")
    for c in sorted(bias_by_cat):
        print(f"{c:9s} {naive[c]:.4f}  {true[c]:.4f}  {bias_by_cat[c]:+.4f}")
    large = {c for c, b in bias_by_cat.items() if abs(b) >= config.MIN_NAIVE_BIAS}
    assert len(large) >= 2, f"only {large} exceed {config.MIN_NAIVE_BIAS}"


def test_proxy_fails_not_merely_differs():
    truth, cohort = _truth(), bias.load_cohort()
    naive = bias.naive_sell_through(cohort)
    true = truth["sell_through_true_by_category"]
    far = [c for c in true if abs(naive[c] - true[c]) >= config.MIN_NAIVE_BIAS]
    for cat in far:
        assert abs(naive[cat] - true[cat]) > config.TOL_RECOVERY, cat


def test_bias_direction_is_stable_under_resampling():
    """The magnitude tests above say the proxy is far off; this one says the *direction*
    it lands in is a property of the data, not of one sample.

    A preregistered sign can be falsified — `docs/preregistration.md` is a prediction, not
    a contract, and it is deliberately not editable here. So the assertion is on the
    observed direction's stability, and any category whose observed sign contradicts the
    preregistration is reported loudly as a falsification instead of failing the suite.
    """
    truth, cohort = _truth(), bias.load_cohort()
    naive = bias.naive_sell_through(cohort)
    true = truth["sell_through_true_by_category"]
    large = [c for c in true if abs(naive[c] - true[c]) >= config.MIN_NAIVE_BIAS]
    observed = {c: ("positive" if naive[c] > true[c] else "negative") for c in large}

    prereg = _parse_prereg_signs()
    falsified = {c: prereg[c] for c in large if observed[c] != prereg[c]}

    rng = np.random.default_rng(config.SEED)
    frames = {}
    for cat, grp in cohort.groupby("category_id"):
        frames[cat] = grp
    hits = dict.fromkeys(large, 0)
    n_iters = 200
    for _ in range(n_iters):
        pieces = [
            f.iloc[rng.integers(0, len(f), len(f))] for f in frames.values()
        ]
        sample = np.concatenate([p["marked_sold"].to_numpy(float) for p in pieces])
        offset = 0
        for c, f in frames.items():
            block = sample[offset : offset + len(f)].mean()
            offset += len(f)
            if c in hits:
                want_positive = observed[c] == "positive"
                if (block > true[c]) == want_positive:
                    hits[c] += 1

    rates = {c: hits[c] / n_iters for c in large}
    print(f"\nobserved direction: {observed}")
    print(f"pre-declared sign stability over {n_iters} resamples: {rates}")
    if falsified:
        print(
            f"PREREGISTRATION FALSIFIED for {sorted(falsified)}: declared "
            f"{falsified}, observed { {c: observed[c] for c in falsified} }"
        )
    failing = {c: r for c, r in rates.items() if r < 0.95}
    assert not failing, f"observed sign did not hold in >=95% of resamples: {failing}"
    assert falsified == PREREGISTERED_FALSIFIED, (
        "the set of falsified signs changed; update the record deliberately: "
        f"{falsified}"
    )


def test_analysis_module_is_free_of_the_answer_key():
    source = BIAS_SOURCE.read_text(encoding="utf-8")
    leaks = [needle for needle in FORBIDDEN_IN_ANALYSIS if needle in source]
    assert not leaks, f"liquidity/bias.py references the answer key: {leaks}"
