"""Data-quality gates for the raw marketplace event log.

Each check returns a :class:`CheckResult`; ``run_all`` returns the list. A check passes
only when its invariant holds across the whole log. Thresholds live in ``liquidity.config``
alongside every other threshold in the repo.

Two rules govern this module, because a green gate that cannot turn red is worse than no
gate at all:

* **A gate must depend on the data, not on the constants that define it.** The old
  ``contact_uniqueness`` asserted ``distinct_pairs <= rows`` (true for any frame) and the
  old ``window_coverage`` compared two config constants with each other.
* **Every gate has a fault-injection test** in ``tests/test_quality_gates.py``: mutate one
  frame, expect this gate to fail. Without that test the gate is decoration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from liquidity import config
from liquidity.etl import load_raw

TIME_TOL_DAYS = 1e-6


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one quality gate: a name, a verdict and its measurements."""

    name: str
    passed: bool
    detail: dict[str, object]


def _unresolved(child: pd.Series, parent: pd.Series) -> int:
    """Count child keys missing from the parent key set."""
    return int((~child.isin(parent)).sum())


def check_referential_integrity(raw: dict[str, pd.DataFrame]) -> CheckResult:
    """Every foreign key in the event log must resolve to a known parent row."""
    listings = raw["listings"]
    searches = raw["searches"]
    impressions = raw["impressions"]
    views = raw["views"]
    contacts = raw["contacts"]

    unresolved = {
        "listings.seller_id": _unresolved(listings["seller_id"], raw["sellers"]["seller_id"]),
        "listings.category_id": _unresolved(
            listings["category_id"], raw["categories"]["category_id"]
        ),
        "listings.region_id": _unresolved(listings["region_id"], raw["regions"]["region_id"]),
        "searches.category_id": _unresolved(
            searches["category_id"], raw["categories"]["category_id"]
        ),
        "searches.region_id": _unresolved(searches["region_id"], raw["regions"]["region_id"]),
        "impressions.search_id": _unresolved(impressions["search_id"], searches["search_id"]),
        "impressions.listing_id": _unresolved(impressions["listing_id"], listings["listing_id"]),
        "views.impression_id": _unresolved(views["impression_id"], impressions["impression_id"]),
        "contacts.search_id": _unresolved(contacts["search_id"], searches["search_id"]),
        "contacts.listing_id": _unresolved(contacts["listing_id"], listings["listing_id"]),
        "contacts.impression_id": _unresolved(
            contacts["impression_id"], impressions["impression_id"]
        ),
        "contacts.view_id": _unresolved(contacts["view_id"], views["view_id"]),
    }
    broken = {key: value for key, value in unresolved.items() if value}
    return CheckResult("referential_integrity", not broken, {"unresolved": unresolved})


def check_temporal_order(raw: dict[str, pd.DataFrame]) -> CheckResult:
    """Event times must be monotone along their own chain and inside the horizon.

    A listing cannot be contacted or removed before it is published, a removal cannot
    land after the horizon, a reply lag cannot be negative, and a contact must carry the
    day of the session that produced it.
    """
    listings = raw["listings"]
    contacts = raw["contacts"]
    searches = raw["searches"]

    removed = listings[listings["status"].fillna("") == "removed"]
    removal_before_publish = int((removed["removal_day"] < removed["publish_day"]).sum())
    removal_after_horizon = int((removed["removal_day"] > config.N_DAYS).sum())

    replied = contacts[contacts["seller_replied"] == 1]
    negative_reply_lag = int((replied["reply_lag_hours"] < 0).sum())

    search_day = contacts["search_id"].map(searches.set_index("search_id")["day"])
    off_chain = int((~np.isclose(contacts["day"], search_day, atol=TIME_TOL_DAYS, equal_nan=False)).sum())

    publish_day = contacts["listing_id"].map(listings.set_index("listing_id")["publish_day"])
    contact_lag = contacts["day"] - publish_day
    pre_publish = int((contact_lag < 0).sum())
    share = float(pre_publish / len(contacts)) if len(contacts) else 0.0

    passed = (
        removal_before_publish == 0
        and removal_after_horizon == 0
        and negative_reply_lag == 0
        and off_chain == 0
        and pre_publish == 0
    )
    detail = {
        "removal_before_publish": removal_before_publish,
        "removal_after_horizon": removal_after_horizon,
        "negative_reply_lag": negative_reply_lag,
        "contact_off_its_session": off_chain,
        "pre_publish_contacts": pre_publish,
        "pre_publish_contact_share": round(share, 6),
    }
    return CheckResult("temporal_order", passed, detail)


def check_duplicate_rate(raw: dict[str, pd.DataFrame]) -> CheckResult:
    """Duplicate share per category must stay inside the configured band."""
    listings = raw["listings"]
    duplicated = listings.duplicated("object_id", keep=False)
    share = duplicated.groupby(listings["category_id"]).mean()

    by_category = {str(key): round(float(value), 6) for key, value in share.items()}
    offending = {
        key: value
        for key, value in by_category.items()
        if not (0.0 <= value <= config.MAX_DUPLICATE_SHARE)
    }
    detail = {
        "share_by_category": by_category,
        "max_allowed": config.MAX_DUPLICATE_SHARE,
        "duplicated_rows": int(duplicated.sum()),
        "listings": int(len(listings)),
    }
    return CheckResult("duplicate_rate", not offending, detail)


def check_contact_density(raw: dict[str, pd.DataFrame]) -> CheckResult:
    """One (buyer, listing) pair may be re-contacted, but not indefinitely.

    Replaces a tautology: the old check asserted ``distinct_pairs <= rows``, which holds
    for every frame by construction. This one fails when the log double-counts contacts,
    which is the defect that inflates every contact-rate metric downstream.
    """
    contacts = raw["contacts"]
    repeats = contacts.groupby(["buyer_id", "listing_id"]).size()
    worst = int(repeats.max()) if len(repeats) else 0
    over_limit = int((repeats > config.CONTACT_REPEAT_MAX).sum())

    detail = {
        "rows": int(len(contacts)),
        "distinct_pairs": int(len(repeats)),
        "repeated_pairs": int(len(contacts)) - int(len(repeats)),
        "max_contacts_per_pair": worst,
        "max_allowed": config.CONTACT_REPEAT_MAX,
        "pairs_over_limit": over_limit,
    }
    return CheckResult("contact_density", over_limit == 0, detail)


def check_window_coverage(raw: dict[str, pd.DataFrame]) -> CheckResult:
    """Every cohort listing must be observable for the full window in the *observed* data.

    The horizon is read from the log (the latest session day), not from
    ``config.N_DAYS``: comparing two constants can never fail, while a truncated export
    silently censors the cohort and flatters the sell-through proxy.
    """
    listings = raw["listings"]
    cohort = listings[listings["publish_day"] < config.COHORT_DAYS]
    required_end = cohort["publish_day"] + config.LIQUIDITY_WINDOW_DAYS
    observed_horizon = float(raw["searches"]["day"].max())
    slack = observed_horizon - required_end
    violations = int((slack < 0).sum())

    detail = {
        "cohort_listings": int(len(cohort)),
        "observed_horizon_day": round(observed_horizon, 6),
        "configured_horizon_day": config.N_DAYS,
        "window_days": config.LIQUIDITY_WINDOW_DAYS,
        "latest_required_end": round(float(required_end.max()), 6) if len(cohort) else None,
        "min_slack_days": round(float(slack.min()), 6) if len(cohort) else None,
        "violations": violations,
    }
    return CheckResult("window_coverage", violations == 0, detail)


def check_holdout_share(raw: dict[str, pd.DataFrame]) -> CheckResult:
    """The randomised holdout must stay at its planned share.

    A 10% arm that drifts to 6% fails no other check; it just widens every interval and
    under-powers the contrast that the monetisation decision rests on.
    """
    sessions = raw["searches"]
    served = sessions[sessions["n_results"] > 0]
    share = float(served["in_holdout"].mean()) if len(served) else float("nan")
    drift = abs(share - config.HOLDOUT_SHARE)

    detail = {
        "served_sessions": int(len(served)),
        "observed_holdout_share": round(share, 6),
        "planned_holdout_share": config.HOLDOUT_SHARE,
        "tolerance": config.HOLDOUT_SHARE_TOLERANCE,
        "drift": round(drift, 6),
    }
    return CheckResult("holdout_share", drift <= config.HOLDOUT_SHARE_TOLERANCE, detail)


def check_value_domains(raw: dict[str, pd.DataFrame]) -> CheckResult:
    """Columns must stay inside their declared domains.

    Every analysis in this repo assumes these: a zero position breaks the SERP logic, a
    non-positive price breaks every "price vs market" signal, and a flag outside {0,1}
    turns ``groupby("is_promoted")`` into a silent partial split.
    """
    impressions = raw["impressions"]
    views = raw["views"]
    listings = raw["listings"]
    contacts = raw["contacts"]
    searches = raw["searches"]

    binary_violations = {
        "impressions.is_promoted": int((~impressions["is_promoted"].isin([0, 1])).sum()),
        "views.is_promoted": int((~views["is_promoted"].isin([0, 1])).sum()),
        "contacts.is_promoted": int((~contacts["is_promoted"].isin([0, 1])).sum()),
        "contacts.seller_replied": int((~contacts["seller_replied"].isin([0, 1])).sum()),
        "listings.boost_purchased": int((~listings["boost_purchased"].isin([0, 1])).sum()),
        "searches.in_holdout": int((~searches["in_holdout"].isin([0, 1])).sum()),
        "searches.zero_result": int((~searches["zero_result"].isin([0, 1])).sum()),
    }
    violations = {
        "impressions.position": int((impressions["position"] < 1).sum()),
        "views.position": int((views["position"] < 1).sum()),
        "listings.price_rub": int((listings["price_rub"] <= 0).sum()),
        "searches.n_results": int((searches["n_results"] < 0).sum()),
        "searches.day": int(
            ((searches["day"] < 0) | (searches["day"] > config.N_DAYS)).sum()
        ),
        "contacts.day": int(((contacts["day"] < 0) | (contacts["day"] > config.N_DAYS)).sum()),
    }
    broken = {key: value for key, value in (violations | binary_violations).items() if value}
    return CheckResult(
        "value_domains",
        not broken,
        {"violations": violations, "binary_violations": binary_violations},
    )


def check_funnel_consistency(raw: dict[str, pd.DataFrame]) -> CheckResult:
    """The funnel must be monotone and the page-size counter must match the log.

    ``n_results`` is the denominator of every per-session rate in the monetisation leg; if
    it disagrees with the impression log, the two legs of the report describe different
    datasets.
    """
    searches = raw["searches"]
    impressions = raw["impressions"]
    views = raw["views"]
    contacts = raw["contacts"]

    per_search = impressions.groupby("search_id").size()
    declared = searches.set_index("search_id")["n_results"]
    mismatched = int((per_search.reindex(declared.index, fill_value=0) != declared).sum())

    served_without_impressions = int(
        ((searches["n_results"] > 0) & (~searches["search_id"].isin(per_search.index))).sum()
    )
    row_counts = {
        "impressions": int(len(impressions)),
        "views": int(len(views)),
        "contacts": int(len(contacts)),
    }
    monotone = row_counts["contacts"] <= row_counts["views"] <= row_counts["impressions"]

    detail = {
        "n_results_mismatches": mismatched,
        "served_sessions_without_impressions": served_without_impressions,
        "row_counts": row_counts,
        "monotone": monotone,
    }
    return CheckResult(
        "funnel_consistency",
        mismatched == 0 and served_without_impressions == 0 and monotone,
        detail,
    )


CHECKS = (
    check_referential_integrity,
    check_temporal_order,
    check_duplicate_rate,
    check_contact_density,
    check_window_coverage,
    check_holdout_share,
    check_value_domains,
    check_funnel_consistency,
)


def run_all(raw: dict[str, pd.DataFrame] | None = None) -> list[CheckResult]:
    """Run every check, loading the raw layer once when no frames are supplied."""
    data = load_raw() if raw is None else raw
    return [check(data) for check in CHECKS]


if __name__ == "__main__":
    results = run_all()
    print(f"{'check':24s} {'verdict':8s} measurements")
    print("-" * 96)
    for result in results:
        verdict = "PASS" if result.passed else "FAIL"
        print(f"{result.name:24s} {verdict:8s} {result.detail}")
    failed = [result.name for result in results if not result.passed]
    print("-" * 96)
    print("all checks passed" if not failed else f"failed: {', '.join(failed)}")
