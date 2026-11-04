"""Monetisation leg: does buying visibility create contacts, or relocate them?

Two estimators answer the same question and disagree:

* ``naive_prepost`` -- a boosted listing's contacts before vs after the purchase.
  This is the number an analyst reaches for first, and it is confounded: boosts are
  bought disproportionately by listings with weak organic position, so the
  "before" window is a low baseline by construction.
* ``holdout_incrementality`` -- contacts per search session, treated sessions vs
  the randomised holdout where promoted slots were suppressed. That is the causal
  contrast.

``guardrails`` reports what the primary metric hides: contacts on *unpaid*
listings, and whether paid traffic turns into a real conversation. ``decision``
applies the rule declared in ``docs/preregistration.md`` before the analysis ran.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from liquidity import config
from liquidity.metrics import contact_pairs, load

PROJECT_ROOT = config.PROJECT_ROOT
PREREGISTRATION_PATH = PROJECT_ROOT / "docs" / "preregistration.md"

# A relative fall smaller than this is treated as noise rather than a dirty
# guardrail. The preregistration says "unpaid contacts fall"; 1% is the smallest
# drop we are willing to call a fall on this sample size.
GUARDRAIL_TOLERANCE = 0.01
# Bootstrap settings are declared once, in config: this module used to run 1 000 draws at
# seed 7 while config said 2 000 at seed 42, and nothing reconciled the two.
ALPHA = config.BOOTSTRAP_ALPHA
N_BOOTSTRAP = config.SESSION_BOOTSTRAP_ITERS
BOOTSTRAP_SEED = config.SESSION_BOOTSTRAP_SEED

# Power arithmetic for the minimum detectable effect. Two-sided alpha 0.05, power 0.80:
# MDE = (z_{1-alpha/2} + z_{power}) * SE. Reported next to the estimate because a
# "negative, not significant" readout is uninformative when the effect is below the MDE.
Z_ALPHA_2S = 1.959963984540054
Z_POWER_80 = 0.8416212335729143


# --------------------------------------------------------------------------- #
# Shared session-level primitive
# --------------------------------------------------------------------------- #
def session_contacts(df: dict[str, pd.DataFrame], paid: bool | None = None) -> pd.DataFrame:
    """Contacts per search session, one row per session.

    Only sessions that returned results are kept. A zero-result session has no
    listing to contact, so including it measures a supply failure, not the effect
    of promoted slots, and dilutes both arms by the same unrelated amount.

    The counting unit here is the pair *as delivered by one session*
    (``search_id``, ``buyer_id``, ``listing_id``). It differs from the global
    ``(buyer_id, listing_id)`` pair used by the listing-level liquidity metrics:
    when a buyer re-contacts the same listing in a later session, the listing-level
    metric counts it once, while the session-level rate counts the contact in the
    session that produced it. Both are one contact, never a row count.

    ``paid=True`` counts contacts on promoted listings, ``paid=False`` on organic
    ones, ``None`` counts all.
    """
    sessions = df["searches"]
    contacts = df["contacts"]
    if paid is not None:
        contacts = contacts[contacts["is_promoted"].eq(1 if paid else 0)]

    counts = (
        contacts.drop_duplicates(["search_id", "buyer_id", "listing_id"])
        .groupby("search_id")
        .size()
        .rename("contacts")
    )

    served = sessions[sessions["n_results"] > 0][["search_id", "in_holdout"]]
    return served.merge(counts, on="search_id", how="left").fillna({"contacts": 0})


def _split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    treated = frame.loc[frame["in_holdout"] == 0, "contacts"].to_numpy(dtype=float)
    holdout = frame.loc[frame["in_holdout"] == 1, "contacts"].to_numpy(dtype=float)
    return treated, holdout


def _bootstrap_diff(
    treated: np.ndarray, holdout: np.ndarray, n_bootstrap: int, seed: int
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``mean(treated) - mean(holdout)``.

    Sampling is batched so a 35k-session arm does not allocate one giant index
    matrix per replicate.
    """
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap)
    batch = min(64, n_bootstrap)
    done = 0
    while done < n_bootstrap:
        size = min(batch, n_bootstrap - done)
        idx_treated = rng.integers(0, treated.size, size=(size, treated.size))
        idx_holdout = rng.integers(0, holdout.size, size=(size, holdout.size))
        diffs[done : done + size] = treated[idx_treated].mean(axis=1) - holdout[idx_holdout].mean(
            axis=1
        )
        done += size
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _bootstrap_rate_diff(
    num_treated: np.ndarray,
    den_treated: np.ndarray,
    num_holdout: np.ndarray,
    den_holdout: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``sum(num)/sum(den)`` treated minus holdout.

    A ratio of totals, not a mean of ratios: an arm's contact rate is contacts over
    exposures, so resampling must carry both the numerator and the denominator of
    the same unit. Used for contacts per organic impression and for the reply rate.
    """
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap)
    batch = min(32, n_bootstrap)
    done = 0
    while done < n_bootstrap:
        size = min(batch, n_bootstrap - done)
        it = rng.integers(0, num_treated.size, size=(size, num_treated.size))
        ih = rng.integers(0, num_holdout.size, size=(size, num_holdout.size))
        with np.errstate(invalid="ignore", divide="ignore"):
            diffs[done : done + size] = num_treated[it].sum(axis=1) / den_treated[it].sum(
                axis=1
            ) - num_holdout[ih].sum(axis=1) / den_holdout[ih].sum(axis=1)
        done += size
    return float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))


def session_frame(df: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per served session: exposures and contacts, split by paid/organic.

    The exposure columns are what make displacement decomposable. Contacts per
    session can fall because fewer organic slots are shown (arithmetic) or because
    an organic slot converts less often (a real effect); without the exposure
    denominator those two collapse into one number.
    """
    sessions = df["searches"]
    served = sessions.loc[sessions["n_results"] > 0, ["search_id", "in_holdout"]].copy()

    impressions = df["impressions"]
    organic_impressions = (
        impressions.loc[impressions["is_promoted"].eq(0)].groupby("search_id").size()
    )
    promoted_impressions = (
        impressions.loc[impressions["is_promoted"].eq(1)].groupby("search_id").size()
    )

    frame = served
    for name, series in (
        ("organic_impressions", organic_impressions),
        ("promoted_impressions", promoted_impressions),
    ):
        frame = frame.merge(series.rename(name), on="search_id", how="left")

    for name, paid in (("organic_contacts", False), ("promoted_contacts", True)):
        counts = session_contacts(df, paid=paid).rename(columns={"contacts": name})
        frame = frame.merge(counts[["search_id", name]], on="search_id", how="left")

    numeric = [
        "organic_impressions",
        "promoted_impressions",
        "organic_contacts",
        "promoted_contacts",
    ]
    return frame.fillna(dict.fromkeys(numeric, 0))


def _ratio_block(
    frame: pd.DataFrame,
    numerator: str,
    denominator: str,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Treated vs holdout for ``numerator / denominator``, with a bootstrap interval."""
    treated = frame[frame["in_holdout"] == 0]
    holdout = frame[frame["in_holdout"] == 1]

    def rate(arm: pd.DataFrame) -> float:
        denominator_sum = float(arm[denominator].sum())
        return float(arm[numerator].sum()) / denominator_sum if denominator_sum else float("nan")

    treated_rate = rate(treated)
    holdout_rate = rate(holdout)
    ci_low, ci_high = _bootstrap_rate_diff(
        treated[numerator].to_numpy(dtype=float),
        treated[denominator].to_numpy(dtype=float),
        holdout[numerator].to_numpy(dtype=float),
        holdout[denominator].to_numpy(dtype=float),
        n_bootstrap,
        seed,
    )
    delta = treated_rate - holdout_rate
    return {
        "treated": treated_rate,
        "holdout": holdout_rate,
        "absolute_change": delta,
        "relative_change": delta / holdout_rate if holdout_rate else float("nan"),
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
    }


def _position_mix(df: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Organic impressions per session by SERP position bucket, per arm (descriptive).

    Answers the only question an analyst would ask about the displacement story
    without a counterfactual: are organic listings actually pushed down, and does
    the mix of positions a session sees change?
    """
    sessions = df["searches"]
    served = sessions.loc[sessions["n_results"] > 0, ["search_id", "in_holdout"]]
    organic = df["impressions"].loc[df["impressions"]["is_promoted"].eq(0)].copy()
    organic["bucket"] = pd.cut(
        organic["position"],
        bins=[0, 3, 6, 10, 20],
        labels=["1-3", "4-6", "7-10", "11-20"],
    )
    merged = organic.merge(served, on="search_id", how="inner")
    counts = merged.groupby(["bucket", "in_holdout"], observed=True).size().unstack(fill_value=0)
    arm_sessions = served["in_holdout"].value_counts()

    rows: list[dict[str, Any]] = []
    for bucket in counts.index:
        treated = float(counts.loc[bucket].get(0, 0)) / float(arm_sessions.get(0, 1))
        holdout = float(counts.loc[bucket].get(1, 0)) / float(arm_sessions.get(1, 1))
        rows.append(
            {
                "position_bucket": str(bucket),
                "treated": treated,
                "holdout": holdout,
                "relative_change": (treated - holdout) / holdout if holdout else float("nan"),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# 1. The naive estimator
# --------------------------------------------------------------------------- #
def naive_prepost(
    df: dict[str, pd.DataFrame], pre_days: float = 7.0, post_days: float = 7.0
) -> pd.DataFrame:
    """Contacts per boosted listing before vs after the boost purchase.

    Returns one row per boosted listing with ``contacts_pre``, ``contacts_post``
    and ``lift``. Aggregate figures live on ``frame.attrs``:

    ``lift_ratio``      sum(post) / sum(pre) across boosted listings
    ``aggregate_pre``   mean contacts per listing in the pre window
    ``aggregate_post``  mean contacts per listing in the post window
    ``n_listings``      number of boosted listings
    """
    listings = df["listings"]
    boosted = listings[listings["boost_purchased"] == 1]

    pairs = contact_pairs(df).merge(
        boosted[["listing_id", "boost_start_day"]], on="listing_id", how="inner"
    )
    elapsed = pairs["day"] - pairs["boost_start_day"]

    pre = pairs[(elapsed >= -pre_days) & (elapsed < 0)].groupby("listing_id").size()
    post = pairs[(elapsed >= 0) & (elapsed < post_days)].groupby("listing_id").size()

    frame = pd.DataFrame(
        {
            "contacts_pre": pre,
            "contacts_post": post,
        }
    ).reindex(boosted["listing_id"].to_numpy(), fill_value=0)
    frame = frame.fillna(0).astype(int)
    frame["lift"] = frame["contacts_post"] - frame["contacts_pre"]
    frame.index.name = "listing_id"

    total_pre = int(frame["contacts_pre"].sum())
    total_post = int(frame["contacts_post"].sum())
    frame.attrs["lift_ratio"] = float(total_post / total_pre) if total_pre else float("inf")
    frame.attrs["aggregate_pre"] = (
        float(frame["contacts_pre"].mean()) if len(frame) else float("nan")
    )
    frame.attrs["aggregate_post"] = (
        float(frame["contacts_post"].mean()) if len(frame) else float("nan")
    )
    frame.attrs["n_listings"] = int(len(frame))
    return frame


# --------------------------------------------------------------------------- #
# 2. The causal contrast
# --------------------------------------------------------------------------- #
def holdout_incrementality(
    df: dict[str, pd.DataFrame], n_bootstrap: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    """Contacts per session, treated sessions minus randomised holdout sessions.

    Returns the two rates, their difference, a Welch t-test on the per-session means
    and a percentile bootstrap CI on the same difference. It also returns the two
    numbers that decide whether the contrast can answer the question at all: the
    minimum detectable effect at 80% power, and the sample-ratio check against the
    planned 10% holdout.
    """
    frame = session_contacts(df)
    treated, holdout = _split(frame)

    treated_rate = float(treated.mean())
    holdout_rate = float(holdout.mean())
    t_stat, p_value = stats.ttest_ind(treated, holdout, equal_var=False)
    ci_low, ci_high = _bootstrap_diff(treated, holdout, n_bootstrap, seed)

    standard_error = float(
        np.sqrt(treated.var(ddof=1) / treated.size + holdout.var(ddof=1) / holdout.size)
    )
    mde_80 = float((Z_ALPHA_2S + Z_POWER_80) * standard_error)
    incrementality = treated_rate - holdout_rate

    return {
        "contacts_per_session_treated": treated_rate,
        "contacts_per_session_holdout": holdout_rate,
        "incrementality": incrementality,
        "relative_change": incrementality / holdout_rate if holdout_rate else float("nan"),
        "sessions_treated": int(treated.size),
        "sessions_holdout": int(holdout.size),
        "welch_t": float(t_stat),
        "welch_p_value": float(p_value),
        "significant_at_alpha": bool(p_value < ALPHA),
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "standard_error": standard_error,
        "mde_80": mde_80,
        "mde_relative_to_holdout": mde_80 / holdout_rate if holdout_rate else float("nan"),
        "underpowered_for_its_own_effect": bool(abs(incrementality) < mde_80),
        "srm": sample_ratio_check(int(treated.size), int(holdout.size)),
    }


def sample_ratio_check(
    treated_sessions: int, holdout_sessions: int, planned_holdout_share: float | None = None
) -> dict[str, Any]:
    """Chi-square check that the arms split as planned.

    A 10% holdout that silently becomes 5% fails no existing test; it only widens the
    interval. This turns that silent failure into a reported number.
    """
    planned = config.HOLDOUT_SHARE if planned_holdout_share is None else planned_holdout_share
    total = treated_sessions + holdout_sessions
    observed_holdout = holdout_sessions / total if total else float("nan")
    expected_holdout = total * planned
    expected_treated = total * (1.0 - planned)

    chi_square = 0.0
    for observed, expected in (
        (holdout_sessions, expected_holdout),
        (treated_sessions, expected_treated),
    ):
        if expected > 0:
            chi_square += (observed - expected) ** 2 / expected
    p_value = float(stats.chi2.sf(chi_square, df=1)) if total else float("nan")

    return {
        "planned_holdout_share": planned,
        "observed_holdout_share": observed_holdout,
        "expected_holdout_sessions": expected_holdout,
        "chi_square": float(chi_square),
        "p_value": p_value,
        "passes_at_alpha": bool(p_value > 0.01),
    }


# --------------------------------------------------------------------------- #
# 3. Guardrails
# --------------------------------------------------------------------------- #
def guardrails(df: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """What the primary readout hides, split by what each contrast can support.

    Three blocks, deliberately not mixed:

    * ``displacement`` -- organic contacts per session, decomposed into the number of
      organic slots a session is shown (arithmetic: a promoted slot occupies one) and
      the conversion of an organic impression (the part that could be a real effect).
      The position mix sits next to it because the stated mechanism is "organic is
      pushed down the SERP".
    * ``reply_rate_causal`` -- reply rate on **organic** contacts, treated sessions vs
      the randomised holdout. Same listings, same demand, different slots: the only
      version of this guardrail that carries a causal reading.
    * ``reply_rate_descriptive`` -- paid vs unpaid contacts. Different listings, and a
      boosted listing is a weak listing by construction. Reported because it was
      preregistered, labelled for what it is.
    """
    frame = session_frame(df)

    displacement: dict[str, Any] = {
        "organic_contacts_per_session": _ratio_block(
            frame.assign(exposure=1.0), "organic_contacts", "exposure"
        ),
        "organic_impressions_per_session": _ratio_block(
            frame.assign(exposure=1.0), "organic_impressions", "exposure"
        ),
        "contacts_per_organic_impression": _ratio_block(
            frame, "organic_contacts", "organic_impressions"
        ),
        "by_position": _position_mix(df),
    }
    slot_arithmetic = displacement["organic_impressions_per_session"]["relative_change"]
    conversion_shift = displacement["contacts_per_organic_impression"]["relative_change"]
    displacement["slot_arithmetic_share_of_drop"] = (
        slot_arithmetic / (slot_arithmetic + conversion_shift)
        if (slot_arithmetic + conversion_shift) != 0
        else float("nan")
    )

    pairs = contact_pairs(df)
    organic = pairs.merge(df["searches"][["search_id", "in_holdout"]], on="search_id", how="inner")
    organic = organic.loc[organic["is_promoted"].eq(0)]
    reply_causal = _ratio_block(organic.assign(exposure=1.0), "seller_replied", "exposure")
    reply_causal["contacts_treated"] = int((organic["in_holdout"] == 0).sum())
    reply_causal["contacts_holdout"] = int((organic["in_holdout"] == 1).sum())

    reply_rates = pairs.groupby("is_promoted")["seller_replied"].mean()
    paid_reply = float(reply_rates.get(1, float("nan")))
    unpaid_reply = float(reply_rates.get(0, float("nan")))

    return {
        "displacement": displacement,
        "reply_rate_causal": reply_causal,
        "reply_rate_descriptive": {
            "paid": paid_reply,
            "unpaid": unpaid_reply,
            "absolute_change": paid_reply - unpaid_reply,
            "relative_change": (paid_reply - unpaid_reply) / unpaid_reply
            if unpaid_reply
            else float("nan"),
        },
    }


# --------------------------------------------------------------------------- #
# 4. Decision rule, read from the preregistration
# --------------------------------------------------------------------------- #
def _strip_markup(cell: str) -> str:
    return cell.replace("**", "").replace("`", "").strip()


def read_decision_rules(path: Path | str = PREREGISTRATION_PATH) -> list[dict[str, str]]:
    """Parse the decision-rule table out of the preregistration document.

    The table lives under ``## Decision rules ...`` and has two columns, outcome
    and decision. Parsing it -- rather than restating it in code -- is what keeps
    the recommendation bound to the rule that was written before the data ran.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.strip().lower().startswith("## decision rules")),
        None,
    )
    if start is None:
        raise ValueError(f"no decision-rule section in {path}")

    rules: list[dict[str, str]] = []
    header_seen = False
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        if not stripped.startswith("|"):
            continue
        cells = [_strip_markup(cell) for cell in stripped.strip("|").split("|")]
        if len(cells) < 2 or set(cells[0]) <= set("-: "):
            continue
        if cells[0].lower() == "outcome":
            header_seen = True
            continue
        if header_seen:
            rules.append({"outcome": cells[0], "decision": cells[1]})

    if not rules:
        raise ValueError(f"no decision-rule rows in {path}")
    return rules


def _find_rule(rules: list[dict[str, str]], needle: str) -> dict[str, str]:
    for rule in rules:
        if needle.lower() in rule["outcome"].lower():
            return rule
    raise ValueError(f"preregistration has no decision rule matching {needle!r}")


def decision(
    df: dict[str, pd.DataFrame], preregistration_path: Path | str = PREREGISTRATION_PATH
) -> dict[str, Any]:
    """Recommendation under the preregistered rule.

    The guardrail dominates: if unpaid-listing contacts fall, the recommendation
    is to roll back whatever the primary says. Otherwise the sign of the primary
    picks between the two monetisation rules.
    """
    incrementality = holdout_incrementality(df)
    guard = guardrails(df)
    unpaid = guard["displacement"]["organic_contacts_per_session"]
    guardrail_dirty = unpaid["relative_change"] < -GUARDRAIL_TOLERANCE

    rules = read_decision_rules(preregistration_path)
    applied: list[dict[str, str]] = []

    if guardrail_dirty:
        applied.append(_find_rule(rules, "unpaid contacts"))
    if incrementality["incrementality"] < 0:
        applied.append(_find_rule(rules, "H2b < 0"))
    elif not guardrail_dirty:
        applied.append(_find_rule(rules, "H2b ≥ 0"))

    primary = applied[0]
    reason = "; ".join(f"{rule['outcome']} -> {rule['decision']}" for rule in applied)

    return {
        "incrementality": incrementality["incrementality"],
        "guardrail_dirty": bool(guardrail_dirty),
        "recommendation": primary["decision"],
        "reason": reason,
        "applied_rules": applied,
        "guardrail_relative_change": unpaid["relative_change"],
        "guardrail_decomposition": {
            key: guard["displacement"][key]
            for key in (
                "organic_impressions_per_session",
                "contacts_per_organic_impression",
                "slot_arithmetic_share_of_drop",
            )
        },
        "preregistration": str(Path(preregistration_path)),
        "incrementality_detail": incrementality,
        "guardrails": guard,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run() -> dict[str, Any]:
    """Run all four monetisation analyses on the raw log."""
    frames = load()
    return {
        "naive_prepost": naive_prepost(frames),
        "holdout_incrementality": holdout_incrementality(frames),
        "guardrails": guardrails(frames),
        "decision": decision(frames),
    }


if __name__ == "__main__":
    output = run()

    print("=== 1. Naive pre/post (confounded) ===")
    naive = output["naive_prepost"]
    print(naive.head(10).to_string())
    print(
        f"listings={naive.attrs['n_listings']} "
        f"pre={naive.attrs['aggregate_pre']:.4f} "
        f"post={naive.attrs['aggregate_post']:.4f} "
        f"lift_ratio={naive.attrs['lift_ratio']:.4f}"
    )

    print("\n=== 2. Holdout incrementality ===")
    inc = output["holdout_incrementality"]
    for key in (
        "contacts_per_session_treated",
        "contacts_per_session_holdout",
        "incrementality",
        "relative_change",
        "welch_t",
        "welch_p_value",
        "ci_95_low",
        "ci_95_high",
        "sessions_treated",
        "sessions_holdout",
        "standard_error",
        "mde_80",
        "mde_relative_to_holdout",
        "underpowered_for_its_own_effect",
    ):
        print(f"{key:32s} {inc[key]}")
    print(f"{'srm_observed_holdout_share':32s} {inc['srm']['observed_holdout_share']:.5f}")
    print(f"{'srm_p_value':32s} {inc['srm']['p_value']:.5f}")
    print(f"{'srm_passes_at_alpha':32s} {inc['srm']['passes_at_alpha']}")

    print("\n=== 3. Guardrails ===")
    guard = output["guardrails"]
    for family in ("displacement", "reply_rate_causal", "reply_rate_descriptive"):
        print(family)
        values = guard[family]
        for key, value in values.items():
            if key == "by_position":
                continue
            print(f"  {key:38s} {value}")
        if "by_position" in values:
            print("  organic impressions per session by position bucket")
            print(pd.DataFrame(values["by_position"]).to_string(index=False))

    print("\n=== 4. Decision ===")
    verdict = output["decision"]
    print(f"incrementality   {verdict['incrementality']:.5f}")
    print(f"guardrail_dirty  {verdict['guardrail_dirty']}")
    print(f"recommendation   {verdict['recommendation']}")
    print(f"reason           {verdict['reason']}")
