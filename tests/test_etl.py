"""Tests for the raw -> staging -> marts pipeline and the quality gates."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from liquidity import config, etl, quality

EXPECTED_MART_ROWS = {
    "dim_region": 6,
    "dim_category": 5,
    "dim_listing": 6000,
    "fact_search": 40000,
}


@pytest.fixture(scope="module")
def raw() -> dict[str, pd.DataFrame]:
    return etl.load_raw()


@pytest.fixture(scope="module")
def staging(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return etl.build_staging(raw)


@pytest.fixture(scope="module")
def mart_counts(staging: dict[str, pd.DataFrame]) -> dict[str, int]:
    return etl.build_marts(staging)


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="float64") for column in columns})


def _listings(publish_days: list[float]) -> pd.DataFrame:
    size = len(publish_days)
    return pd.DataFrame(
        {
            "listing_id": np.arange(size, dtype="int64"),
            "object_id": np.arange(size, dtype="int64"),
            "seller_id": [10] * size,
            "seller_type": ["private"] * size,
            "category_id": ["goods"] * size,
            "region_id": ["msk"] * size,
            "price_rub": [1000.0] * size,
            "quality_score": [0.5] * size,
            "relevance_static": [0.5] * size,
            "publish_day": publish_days,
            "boost_purchased": [0] * size,
            "boost_start_day": [np.inf] * size,
            "boost_end_day": [np.inf] * size,
            "publish_ts": ["2026-01-07"] * size,
            "status": ["active"] * size,
            "removal_reason": [None] * size,
            "removal_day": [np.nan] * size,
            "status_ts": [None] * size,
        }
    )


def _synthetic_raw() -> dict[str, pd.DataFrame]:
    """One listing, buyer 7 contacts it twice in two sessions; buyer 8 once."""
    listing_id = 1
    search_ids = [100, 200, 300]
    buyers = [7, 7, 8]
    days = [10.0, 12.0, 14.0]

    impressions = pd.DataFrame(
        {
            "impression_id": [0, 1, 2],
            "search_id": search_ids,
            "listing_id": [listing_id] * 3,
            "position": [1, 1, 1],
            "is_promoted": [0, 1, 0],
            "buyer_id": buyers,
            "day": days,
            "ts": ["t"] * 3,
        }
    )
    views = impressions.drop(columns=["ts"]).copy()
    views.insert(0, "view_id", [0, 1, 2])
    contacts = pd.DataFrame(
        {
            "contact_id": [0, 1, 2],
            "view_id": [0, 1, 2],
            "impression_id": [0, 1, 2],
            "search_id": search_ids,
            "listing_id": [listing_id] * 3,
            "position": [1, 1, 1],
            "is_promoted": [0, 1, 0],
            "buyer_id": buyers,
            "day": days,
            "channel": ["call", "message", "call"],
            "seller_replied": [1, 0, 1],
            "reply_lag_hours": [2.0, np.nan, 3.0],
            "ts": ["t"] * 3,
        }
    )
    searches = pd.DataFrame(
        {
            "search_id": search_ids,
            "buyer_id": buyers,
            "day": days,
            "category_id": ["goods"] * 3,
            "region_id": ["msk"] * 3,
            "in_holdout": [0, 0, 0],
            "n_results": [1, 1, 1],
            "zero_result": [0, 0, 0],
            "ts": ["t"] * 3,
        }
    )
    listings = _listings([2.0])
    listings["listing_id"] = listing_id
    listings["object_id"] = listing_id
    return {
        "listings": listings,
        "searches": searches,
        "impressions": impressions,
        "views": views,
        "contacts": contacts,
    }


def test_staging_row_counts_match_raw(
    raw: dict[str, pd.DataFrame], staging: dict[str, pd.DataFrame]
) -> None:
    assert len(staging["listings"]) == len(raw["listings"])
    assert len(staging["listing_demand"]) == len(raw["listings"])
    assert len(staging["search_demand"]) == len(raw["searches"])


def test_staging_keeps_one_row_per_listing(staging: dict[str, pd.DataFrame]) -> None:
    assert staging["listings"]["listing_id"].is_unique
    assert staging["listing_demand"]["listing_id"].is_unique


def test_listing_demand_has_expected_columns(staging: dict[str, pd.DataFrame]) -> None:
    expected = {
        "n_impressions",
        "n_views",
        "n_contacts",
        "n_sessions",
        "first_contact_day",
        "first_contact_hours",
        "has_contact",
        "has_replied_contact",
        "first_replied_day",
        "last_replied_day",
    }
    assert expected <= set(staging["listing_demand"].columns)


def test_n_contacts_is_one_for_same_buyer_in_two_sessions() -> None:
    staging = etl.build_staging(_synthetic_raw(), persist=False)
    demand = staging["listing_demand"].set_index("listing_id").loc[1]

    assert demand["n_contacts"] == 2
    assert demand["n_sessions"] == 3
    assert demand["first_contact_day"] == pytest.approx(10.0)
    assert demand["first_contact_hours"] == pytest.approx((10.0 - 2.0) * 24)
    assert bool(demand["has_replied_contact"])
    assert demand["first_replied_day"] == pytest.approx(10.0)
    assert demand["last_replied_day"] == pytest.approx(14.0)


def test_n_contacts_uses_distinct_pairs_on_real_data(
    raw: dict[str, pd.DataFrame], staging: dict[str, pd.DataFrame]
) -> None:
    got = (
        staging["listing_demand"]
        .set_index("listing_id")["n_contacts"]
        .astype("int64")
        .sort_index()
    )
    expected = (
        raw["contacts"]
        .groupby("listing_id")["buyer_id"]
        .nunique()
        .reindex(got.index, fill_value=0)
        .rename("n_contacts")
        .astype("int64")
    )
    pd.testing.assert_series_equal(got, expected, check_index=True)
    assert int(got.sum()) == len(raw["contacts"].drop_duplicates(["listing_id", "buyer_id"]))


def test_raw_repeated_pairs_are_counted_once(
    raw: dict[str, pd.DataFrame], staging: dict[str, pd.DataFrame]
) -> None:
    pairs = raw["contacts"].groupby(["listing_id", "buyer_id"]).size()
    repeated = pairs[pairs > 1]
    if repeated.empty:
        pytest.skip("raw contacts hold no repeated (buyer, listing) pair on this data")
    listing_id, buyer_id = repeated.index[0]
    assert int(pairs.loc[(listing_id, buyer_id)]) > 1

    demand = staging["listing_demand"].set_index("listing_id")
    on_listing = raw["contacts"].loc[raw["contacts"]["listing_id"] == listing_id, "buyer_id"]
    assert int(demand.loc[listing_id, "n_contacts"]) == int(on_listing.nunique())


def test_in_cohort_boundary_at_exactly_cohort_days() -> None:
    raw = {
        "listings": _listings([0.0, 29.999, 30.0, 45.0]),
        "impressions": _empty(["impression_id", "search_id", "listing_id", "is_promoted", "day"]),
        "views": _empty(["view_id", "listing_id"]),
        "contacts": _empty(["contact_id", "listing_id", "buyer_id", "search_id", "day", "seller_replied"]),
        "searches": _empty(["search_id", "day"]),
    }
    staging = etl.build_staging(raw, persist=False)
    got = staging["listings"].sort_values("listing_id")["in_cohort"].tolist()
    assert got == [True, True, False, False]


def test_mart_tables_exist_with_expected_rows(mart_counts: dict[str, int]) -> None:
    assert mart_counts == EXPECTED_MART_ROWS

    connection = duckdb.connect(str(config.DB_PATH), read_only=True)
    try:
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        assert set(EXPECTED_MART_ROWS) <= tables
        for name, rows in EXPECTED_MART_ROWS.items():
            observed = connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            assert int(observed) == rows
    finally:
        connection.close()


def test_build_marts_is_idempotent(
    staging: dict[str, pd.DataFrame], mart_counts: dict[str, int]
) -> None:
    assert etl.build_marts(staging) == mart_counts
    assert etl.build_marts(staging) == mart_counts


def test_every_quality_check_passes_on_real_data(raw: dict[str, pd.DataFrame]) -> None:
    results = quality.run_all(raw)
    failed = {result.name: result.detail for result in results if not result.passed}
    assert not failed, f"failing quality checks: {failed}"
    assert len(results) == 5


def test_quality_check_names_and_uniqueness_detail(raw: dict[str, pd.DataFrame]) -> None:
    results = {result.name: result for result in quality.run_all(raw)}
    assert set(results) == {
        "referential_integrity",
        "temporal_order",
        "duplicate_rate",
        "contact_uniqueness",
        "window_coverage",
    }

    uniqueness = results["contact_uniqueness"]
    assert uniqueness.detail["rows"] == len(raw["contacts"])
    assert uniqueness.detail["distinct_pairs"] <= uniqueness.detail["rows"]

    duplicates = results["duplicate_rate"]
    shares = duplicates.detail["share_by_category"]
    assert set(shares) == set(raw["categories"]["category_id"])
    assert all(0.0 <= value <= quality.MAX_DUPLICATE_SHARE for value in shares.values())

    temporal = results["temporal_order"]
    assert temporal.detail["removal_before_publish"] == 0
    assert temporal.detail["removal_after_horizon"] == 0
    assert temporal.detail["negative_reply_lag"] == 0
