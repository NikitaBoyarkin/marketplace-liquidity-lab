"""Contract tests for the generator: determinism, invariants, and the shape of the raw data.

These tests are cheap — they read what is already on disk rather than regenerating it.
Regeneration takes minutes, so determinism is asserted on a single entity builder with the
same seed twice, not on a full re-run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from liquidity import config, generate

RAW = config.RAW_DIR


@pytest.fixture(scope="module")
def raw() -> dict[str, pd.DataFrame]:
    names = ["regions", "categories", "sellers", "listings", "searches", "impressions", "views", "contacts"]
    missing = [n for n in names if not (RAW / f"{n}.parquet").exists()]
    if missing:
        pytest.skip(f"raw data missing ({missing}); run `uv run python -m liquidity.generate` first")
    return {n: pd.read_parquet(RAW / f"{n}.parquet") for n in names}


def test_generator_is_seeded() -> None:
    a = generate.build_sellers(np.random.default_rng(config.SEED))
    b = generate.build_sellers(np.random.default_rng(config.SEED))
    pd.testing.assert_frame_equal(a, b)


def test_all_tables_have_rows(raw: dict[str, pd.DataFrame]) -> None:
    for name, df in raw.items():
        assert len(df) > 0, f"{name} is empty"


def test_every_contact_follows_a_publish(raw: dict[str, pd.DataFrame]) -> None:
    publish = raw["listings"].set_index("listing_id")["publish_day"]
    lag = raw["contacts"]["day"] - raw["contacts"]["listing_id"].map(publish)
    assert (lag >= 0).all(), "a contact precedes the listing it is about"


def test_removal_never_precedes_publish(raw: dict[str, pd.DataFrame]) -> None:
    df = raw["listings"]
    removed = df["removal_day"].notna()
    assert (df.loc[removed, "removal_day"] >= df.loc[removed, "publish_day"]).all()


def test_no_removal_after_the_data_window(raw: dict[str, pd.DataFrame]) -> None:
    df = raw["listings"]
    assert (df["removal_day"].dropna() <= config.N_DAYS).all()


def test_no_engagement_event_follows_a_removal(raw: dict[str, pd.DataFrame]) -> None:
    """A removed listing is off the shelf: nothing may be shown, viewed or contacted after."""
    removal = raw["listings"].set_index("listing_id")["removal_day"]
    for name in ("impressions", "views", "contacts"):
        df = raw[name]
        removed_day = df["listing_id"].map(removal)
        late = int(((removed_day.notna()) & (df["day"] - removed_day > 1e-6)).sum())
        assert late == 0, f"{name} contains {late} events after the listing's removal"


def test_cohort_has_a_full_observation_window(raw: dict[str, pd.DataFrame]) -> None:
    cohort = raw["listings"][raw["listings"]["publish_day"] < config.COHORT_DAYS]
    assert len(cohort) > 0
    assert config.N_DAYS - config.COHORT_DAYS >= config.LIQUIDITY_WINDOW_DAYS


def test_the_platform_never_sees_the_sale(raw: dict[str, pd.DataFrame]) -> None:
    for table in raw.values():
        assert not {"true_sale", "sold", "sale_day"} & set(table.columns)


def test_duplicates_exist_and_share_an_object_id(raw: dict[str, pd.DataFrame]) -> None:
    df = raw["listings"]
    dup = df["object_id"].duplicated(keep=False)
    assert dup.any(), "no duplicates generated — the dedup metric would be meaningless"
    per_cat = df.groupby("category_id")["object_id"].apply(lambda s: s.duplicated(keep=False).mean())
    assert (per_cat < 0.25).all(), f"duplicate share implausible: {per_cat.to_dict()}"


def test_sold_marking_is_imperfect(raw: dict[str, pd.DataFrame]) -> None:
    """If the proxy were accurate the whole repo would be pointless — assert it is not."""
    df = raw["listings"]
    cohort = df[df["publish_day"] < config.COHORT_DAYS]
    marked = (cohort["removal_reason"] == "sold").mean()
    assert 0.05 < marked < 0.60, f"marked-sold share {marked:.3f} is implausible"


def test_boost_is_endogenous(raw: dict[str, pd.DataFrame]) -> None:
    """Boosts must be bought disproportionally by weak listings, otherwise the naive
    pre/post estimator would not be confounded and the second leg would be trivial."""
    df = raw["listings"]
    boosted = df[df["boost_purchased"] == 1]
    organic = df[df["boost_purchased"] == 0]
    assert len(boosted) > 100
    assert boosted["quality_score"].mean() < organic["quality_score"].mean()


def test_holdout_is_within_the_declared_range(raw: dict[str, pd.DataFrame]) -> None:
    share = raw["searches"]["in_holdout"].mean()
    assert abs(share - config.HOLDOUT_SHARE) < 0.02, f"holdout share {share:.3f}"


def test_positions_are_one_based_and_bounded(raw: dict[str, pd.DataFrame]) -> None:
    imp = raw["impressions"]
    assert imp["position"].min() >= 1
    assert imp["position"].max() <= generate.RESULTS_PER_SEARCH


def test_promoted_slots_never_appear_in_holdout_sessions(raw: dict[str, pd.DataFrame]) -> None:
    holdout_ids = set(raw["searches"].loc[raw["searches"]["in_holdout"] == 1, "search_id"])
    imp = raw["impressions"]
    leaked = imp[(imp["is_promoted"] == 1) & imp["search_id"].isin(holdout_ids)]
    assert leaked.empty, f"{len(leaked)} promoted impressions leaked into the holdout"
