"""Behavioural tests for the liquidity metrics.

The fixtures use hand-built frames so the cohort boundary and the contact-counting
convention are pinned by facts, not by whatever the generator happened to produce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from liquidity import metrics

LISTING_COLUMNS = ["listing_id", "object_id", "category_id", "region_id", "publish_day"]
CONTACT_COLUMNS = [
    "contact_id",
    "search_id",
    "listing_id",
    "buyer_id",
    "is_promoted",
    "seller_replied",
    "day",
]
VIEW_COLUMNS = ["listing_id", "search_id", "day"]


def _frames(
    listings: pd.DataFrame,
    contacts: pd.DataFrame | None = None,
    views: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    return {
        "listings": listings,
        "contacts": contacts if contacts is not None else pd.DataFrame(columns=CONTACT_COLUMNS),
        "views": views if views is not None else pd.DataFrame(columns=VIEW_COLUMNS),
    }


@pytest.fixture(scope="module")
def raw() -> dict[str, pd.DataFrame]:
    return metrics.load()


# --------------------------------------------------------------------------- #
# Cohort definition
# --------------------------------------------------------------------------- #
def test_cohort_boundary_is_strictly_less_than_30_days() -> None:
    listings = pd.DataFrame(
        {
            "listing_id": [1, 2, 3],
            "object_id": [1, 2, 3],
            "category_id": ["goods", "goods", "goods"],
            "region_id": ["msk", "msk", "msk"],
            "publish_day": [29.99, 30.0, 45.0],
        }
    )
    frame = metrics.liquidity_by_cohort(_frames(listings))

    platform = frame[(frame["category_id"] == metrics.PLATFORM) & (frame["region_id"] == "all")]
    assert len(platform) == 1
    assert int(platform["cohort_size"].iloc[0]) == 1, "publish_day == 30 must be excluded"


# --------------------------------------------------------------------------- #
# Contact counting
# --------------------------------------------------------------------------- #
def test_contact_metric_counts_distinct_pairs_not_rows() -> None:
    listings = pd.DataFrame(
        {
            "listing_id": [1, 2],
            "object_id": [1, 2],
            "category_id": ["goods", "goods"],
            "region_id": ["msk", "msk"],
            "publish_day": [0.0, 0.0],
        }
    )
    contacts = pd.DataFrame(
        {
            "contact_id": [1, 2, 3],
            "search_id": [10, 11, 12],
            "listing_id": [1, 1, 2],
            "buyer_id": [7, 7, 7],
            "is_promoted": [0, 0, 1],
            "seller_replied": [1, 1, 0],
            "day": [1.0, 2.0, 3.0],
        }
    )
    frames = _frames(listings, contacts)

    pairs = metrics.contact_pairs(frames)
    assert len(pairs) == 2, "buyer 7 contacting listing 1 twice is one contact"

    frame = metrics.liquidity_by_cohort(frames)
    platform = frame[(frame["category_id"] == metrics.PLATFORM) & (frame["region_id"] == "all")]
    assert platform["listings_with_contact"].iloc[0] == pytest.approx(1.0)


def test_contact_metric_ignores_contacts_outside_the_observation_window() -> None:
    listings = pd.DataFrame(
        {
            "listing_id": [1],
            "object_id": [1],
            "category_id": ["goods"],
            "region_id": ["msk"],
            "publish_day": [0.0],
        }
    )
    contacts = pd.DataFrame(
        {
            "contact_id": [1],
            "search_id": [10],
            "listing_id": [1],
            "buyer_id": [7],
            "is_promoted": [0],
            "seller_replied": [1],
            "day": [31.0],
        }
    )
    frame = metrics.liquidity_by_cohort(_frames(listings, contacts))
    platform = frame[(frame["category_id"] == metrics.PLATFORM) & (frame["region_id"] == "all")]
    assert platform["listings_with_contact"].iloc[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Real-data invariants
# --------------------------------------------------------------------------- #
def test_shares_are_probabilities(raw: dict[str, pd.DataFrame]) -> None:
    frame = metrics.liquidity_by_cohort(raw)
    for column in ("listings_with_contact", "dead_listing_share", "duplicate_share"):
        assert frame[column].between(0.0, 1.0).all(), column


def test_time_to_first_contact_is_positive(raw: dict[str, pd.DataFrame]) -> None:
    frame = metrics.liquidity_by_cohort(raw)
    contacts = frame.dropna(subset=["median_time_to_first_contact_hours"])
    assert (contacts["median_time_to_first_contact_hours"] >= 0).all()


def test_quadrant_uses_only_the_declared_labels(raw: dict[str, pd.DataFrame]) -> None:
    frame = metrics.demand_sufficiency(raw)
    assert set(frame["quadrant"]).issubset(set(metrics.QUADRANTS))
    assert frame["quadrant"].notna().all()
    assert len(metrics.QUADRANTS) == 4


def test_zero_result_rate_is_split_by_population_tier(
    raw: dict[str, pd.DataFrame],
) -> None:
    frame = metrics.zero_result_rate(raw)
    tiers = frame[frame["region_id"] == metrics.ALL_REGIONS]
    assert len(tiers) >= 3, "each population tier needs its own row"
    assert frame["zero_result_share"].between(0.0, 1.0).all()
    # Thin supply (tier 3) must fail more searches than the largest cities.
    tier_share = tiers.loc[tiers["population_tier"] != "all"].set_index("population_tier")[
        "zero_result_share"
    ]
    assert tier_share["3"] > tier_share["1"]


# --------------------------------------------------------------------------- #
# SQL is not decorative
# --------------------------------------------------------------------------- #
def test_sql_matches_pandas_on_cohort_contact_share(raw: dict[str, pd.DataFrame]) -> None:
    pandas_frame = metrics.liquidity_by_cohort(raw)
    pandas_category = pandas_frame[
        (pandas_frame["category_id"] != metrics.PLATFORM)
        & (pandas_frame["region_id"] == metrics.ALL_REGIONS)
    ]
    sql_frame = metrics.cohort_contact_share_sql()

    merged = pandas_category.merge(sql_frame, on="category_id", suffixes=("_pandas", "_sql"))
    assert len(merged) == len(sql_frame) == 5

    for column in ("cohort_size", "listings_with_contact", "dead_listing_share"):
        assert np.allclose(
            merged[f"{column}_pandas"], merged[f"{column}_sql"], atol=1e-9
        ), column


# --------------------------------------------------------------------------- #
# Mix shift
# --------------------------------------------------------------------------- #
def test_platform_row_is_never_returned_without_the_category_split(
    raw: dict[str, pd.DataFrame],
) -> None:
    cohort_frame = metrics.liquidity_by_cohort(raw)
    assert (cohort_frame["category_id"] == metrics.PLATFORM).any()
    per_category = cohort_frame[cohort_frame["category_id"] != metrics.PLATFORM]
    assert per_category["category_id"].nunique() == 5

    demand_frame = metrics.demand_sufficiency(raw)
    assert (demand_frame["category_id"] == metrics.PLATFORM).any()
    per_category_demand = demand_frame[demand_frame["category_id"] != metrics.PLATFORM]
    assert per_category_demand["category_id"].nunique() == 5
