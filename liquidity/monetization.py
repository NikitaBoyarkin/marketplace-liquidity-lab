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

from liquidity.metrics import contact_pairs, load

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREREGISTRATION_PATH = PROJECT_ROOT / "docs" / "preregistration.md"

# A relative fall smaller than this is treated as noise rather than a dirty
# guardrail. The preregistration says "unpaid contacts fall"; 1% is the smallest
# drop we are willing to call a fall on this sample size.
GUARDRAIL_TOLERANCE = 0.01
ALPHA = 0.05
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 7


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
        diffs[done : done + size] = (
            treated[idx_treated].mean(axis=1) - holdout[idx_holdout].mean(axis=1)
        )
        done += size
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


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

    pre = (
        pairs[(elapsed >= -pre_days) & (elapsed < 0)].groupby("listing_id").size()
    )
    post = (
        pairs[(elapsed >= 0) & (elapsed < post_days)].groupby("listing_id").size()
    )

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
    frame.attrs["aggregate_pre"] = float(frame["contacts_pre"].mean()) if len(frame) else float("nan")
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

    Returns the two rates, their difference, a Welch t-test on the per-session
    means and a percentile bootstrap CI on the same difference.
    """
    frame = session_contacts(df)
    treated, holdout = _split(frame)

    treated_rate = float(treated.mean())
    holdout_rate = float(holdout.mean())
    t_stat, p_value = stats.ttest_ind(treated, holdout, equal_var=False)
    ci_low, ci_high = _bootstrap_diff(treated, holdout, n_bootstrap, seed)

    return {
        "contacts_per_session_treated": treated_rate,
        "contacts_per_session_holdout": holdout_rate,
        "incrementality": treated_rate - holdout_rate,
        "relative_change": (treated_rate - holdout_rate) / holdout_rate if holdout_rate else float("nan"),
        "sessions_treated": int(treated.size),
        "sessions_holdout": int(holdout.size),
        "welch_t": float(t_stat),
        "welch_p_value": float(p_value),
        "significant_at_alpha": bool(p_value < ALPHA),
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
    }


# --------------------------------------------------------------------------- #
# 3. Guardrails
# --------------------------------------------------------------------------- #
def guardrails(df: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """What the primary readout hides: organic contacts and reply quality.

    * unpaid contacts per session, treated vs holdout -- promoted slots push every
      organic listing down the SERP, so this is where displacement shows up.
    * ``seller_reply_rate`` on paid vs unpaid contacts -- paid traffic is only
      valuable if it starts a conversation.
    """
    unpaid = session_contacts(df, paid=False)
    unpaid_treated, unpaid_holdout = _split(unpaid)
    unpaid_treated_rate = float(unpaid_treated.mean())
    unpaid_holdout_rate = float(unpaid_holdout.mean())
    unpaid_delta = unpaid_treated_rate - unpaid_holdout_rate

    pairs = contact_pairs(df)
    reply_rates = pairs.groupby("is_promoted")["seller_replied"].mean()
    paid_reply = float(reply_rates.get(1, float("nan")))
    unpaid_reply = float(reply_rates.get(0, float("nan")))

    return {
        "unpaid_contacts_per_session": {
            "treated": unpaid_treated_rate,
            "holdout": unpaid_holdout_rate,
            "absolute_change": unpaid_delta,
            "relative_change": unpaid_delta / unpaid_holdout_rate
            if unpaid_holdout_rate
            else float("nan"),
            "sessions_treated": int(unpaid_treated.size),
            "sessions_holdout": int(unpaid_holdout.size),
        },
        "seller_reply_rate": {
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
    unpaid = guard["unpaid_contacts_per_session"]
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
    for key, value in output["holdout_incrementality"].items():
        print(f"{key:34s} {value}")

    print("\n=== 3. Guardrails ===")
    for family, values in output["guardrails"].items():
        print(family)
        for key, value in values.items():
            print(f"  {key:22s} {value}")

    print("\n=== 4. Decision ===")
    verdict = output["decision"]
    print(f"incrementality   {verdict['incrementality']:.5f}")
    print(f"guardrail_dirty  {verdict['guardrail_dirty']}")
    print(f"recommendation   {verdict['recommendation']}")
    print(f"reason           {verdict['reason']}")
