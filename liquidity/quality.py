"""Data-quality gates for the raw marketplace event log.

Each check returns a :class:`CheckResult`; ``run_all`` returns the list. A check passes
only when its invariant holds across the whole log. Thresholds live in this module
because ``liquidity.config`` is frozen for this axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from liquidity import config
from liquidity.etl import load_raw

MAX_DUPLICATE_SHARE = 0.20
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
        if not (0.0 <= value <= MAX_DUPLICATE_SHARE)
    }
    detail = {
        "share_by_category": by_category,
        "max_allowed": MAX_DUPLICATE_SHARE,
        "duplicated_rows": int(duplicated.sum()),
        "listings": int(len(listings)),
    }
    return CheckResult("duplicate_rate", not offending, detail)


def check_contact_uniqueness(raw: dict[str, pd.DataFrame]) -> CheckResult:
    """A (listing, buyer) pair may repeat across sessions, but only as extra rows."""
    contacts = raw["contacts"]
    rows = int(len(contacts))
    distinct_pairs = int(len(contacts.drop_duplicates(["listing_id", "buyer_id"])))
    detail = {
        "rows": rows,
        "distinct_pairs": distinct_pairs,
        "repeated_pairs": rows - distinct_pairs,
    }
    return CheckResult("contact_uniqueness", distinct_pairs <= rows, detail)


def check_window_coverage(raw: dict[str, pd.DataFrame]) -> CheckResult:
    """Every cohort listing must be observable for the full liquidity window."""
    listings = raw["listings"]
    cohort = listings[listings["publish_day"] < config.COHORT_DAYS]
    required_end = cohort["publish_day"] + config.LIQUIDITY_WINDOW_DAYS
    slack = config.N_DAYS - required_end
    violations = int((slack < 0).sum())

    detail = {
        "cohort_listings": int(len(cohort)),
        "horizon_end_day": config.N_DAYS,
        "window_days": config.LIQUIDITY_WINDOW_DAYS,
        "latest_required_end": round(float(required_end.max()), 6) if len(cohort) else None,
        "min_slack_days": round(float(slack.min()), 6) if len(cohort) else None,
        "violations": violations,
    }
    return CheckResult("window_coverage", violations == 0, detail)


CHECKS = (
    check_referential_integrity,
    check_temporal_order,
    check_duplicate_rate,
    check_contact_uniqueness,
    check_window_coverage,
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
