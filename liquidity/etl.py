"""Reproducible raw -> staging -> marts pipeline.

Raw parquet is read once, enriched into staging frames (written to ``data/staging``),
then materialised into a DuckDB database that downstream analysis queries without
touching the raw layer. Every mart table is rebuilt with ``CREATE OR REPLACE`` so the
pipeline can be re-run any number of times without accumulating rows.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from liquidity import config

STAGING_TABLES = ("listings", "listing_demand", "search_demand")
MART_TABLES = ("dim_region", "dim_category", "dim_listing", "fact_search")

CONTACT_COUNT_COLUMNS = ("n_impressions", "n_views", "n_contacts", "n_sessions")


def load_raw(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Read every parquet file in the raw directory, keyed by file stem."""
    directory = Path(raw_dir) if raw_dir is not None else config.RAW_DIR
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found in {directory}")
    return {path.stem: pd.read_parquet(path) for path in files}


def _enrich_listings(listings: pd.DataFrame) -> pd.DataFrame:
    """Add cohort, duplicate, age and removal flags to the listing dimension."""
    out = listings.copy()
    out["in_cohort"] = (out["publish_day"] < config.COHORT_DAYS).astype(bool)
    out["is_duplicate"] = out.duplicated("object_id", keep=False)
    # Age at the end of the observation horizon, not at the listing's own last event.
    out["age_days"] = config.N_DAYS - out["publish_day"]
    out["removed"] = (out["status"].fillna("") == "removed").astype(bool)
    out["marked_sold"] = (out["removal_reason"].fillna("") == "sold").astype(bool)
    out["days_to_removal"] = out["removal_day"] - out["publish_day"]
    return out


def _listing_demand(
    listings: pd.DataFrame,
    impressions: pd.DataFrame,
    views: pd.DataFrame,
    contacts: pd.DataFrame,
) -> pd.DataFrame:
    """One row per listing with exposure and contact demand aggregates."""
    index = pd.Index(listings["listing_id"], name="listing_id")
    demand = listings[["listing_id", "publish_day"]].set_index("listing_id")

    demand["n_impressions"] = (
        impressions.groupby("listing_id").size().reindex(index, fill_value=0)
    )
    demand["n_views"] = views.groupby("listing_id").size().reindex(index, fill_value=0)
    demand["n_sessions"] = (
        impressions.groupby("listing_id")["search_id"].nunique().reindex(index, fill_value=0)
    )

    contact_groups = contacts.groupby("listing_id")
    # Distinct buyers, not contact rows: the same buyer may message a listing in
    # several sessions and still be one unit of demand.
    demand["n_contacts"] = contact_groups["buyer_id"].nunique().reindex(index)
    demand["first_contact_day"] = contact_groups["day"].min().reindex(index)
    demand["first_contact_hours"] = 24.0 * (demand["first_contact_day"] - demand["publish_day"])
    demand["has_contact"] = demand["first_contact_day"].notna()

    replied = contacts[contacts["seller_replied"] == 1].groupby("listing_id")["day"]
    demand["first_replied_day"] = replied.min().reindex(index)
    demand["last_replied_day"] = replied.max().reindex(index)
    demand["has_replied_contact"] = demand["first_replied_day"].notna()

    for column in CONTACT_COUNT_COLUMNS:
        demand[column] = demand[column].fillna(0).astype("int64")

    return demand.reset_index()[
        [
            "listing_id",
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
        ]
    ]


def _search_demand(
    searches: pd.DataFrame,
    impressions: pd.DataFrame,
    contacts: pd.DataFrame,
) -> pd.DataFrame:
    """One row per search session with contact and promotion aggregates."""
    index = pd.Index(searches["search_id"], name="search_id")
    demand = searches[["search_id"]].set_index("search_id")

    demand["n_contacts"] = (
        contacts.groupby("search_id").size().reindex(index, fill_value=0).astype("int64")
    )
    promoted = (
        impressions[impressions["is_promoted"] == 1]
        .groupby("search_id")
        .size()
        .reindex(index, fill_value=0)
        .astype("int64")
    )
    demand["n_promoted_impressions"] = promoted
    demand["has_promoted"] = promoted > 0

    return demand.reset_index()[["search_id", "n_contacts", "n_promoted_impressions", "has_promoted"]]


def build_staging(
    raw: dict[str, pd.DataFrame],
    staging_dir: Path | None = None,
    *,
    persist: bool = True,
) -> dict[str, pd.DataFrame]:
    """Enrich raw tables into staging frames and write them to ``staging_dir``.

    Dimension and search sources are passed through unchanged so ``build_marts`` can
    consume a single staging dict; only the three enriched tables are persisted.
    """
    listings = _enrich_listings(raw["listings"])
    listing_demand = _listing_demand(
        listings, raw["impressions"], raw["views"], raw["contacts"]
    )
    search_demand = _search_demand(raw["searches"], raw["impressions"], raw["contacts"])

    staging = {
        "listings": listings,
        "listing_demand": listing_demand,
        "search_demand": search_demand,
    }
    for passthrough in ("regions", "categories", "searches"):
        if passthrough in raw:
            staging[passthrough] = raw[passthrough]

    if persist:
        directory = Path(staging_dir) if staging_dir is not None else config.STAGING_DIR
        directory.mkdir(parents=True, exist_ok=True)
        for name in STAGING_TABLES:
            staging[name].to_parquet(directory / f"{name}.parquet", index=False)

    return staging


def _mart_frames(staging: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    dim_listing = staging["listings"].merge(
        staging["listing_demand"], on="listing_id", how="left", validate="one_to_one"
    )
    fact_search = staging["searches"].merge(
        staging["search_demand"], on="search_id", how="left", validate="one_to_one"
    )
    return {
        "dim_region": staging["regions"],
        "dim_category": staging["categories"],
        "dim_listing": dim_listing,
        "fact_search": fact_search,
    }


def build_marts(
    staging: dict[str, pd.DataFrame], db_path: Path | None = None
) -> dict[str, int]:
    """Materialise the four mart tables into DuckDB and return their row counts."""
    frames = _mart_frames(staging)
    path = Path(db_path) if db_path is not None else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(str(path))
    try:
        for name, frame in frames.items():
            connection.register(f"_source_{name}", frame)
        # Consume each result explicitly: DuckDB rejects further work on a connection
        # while a previous execute() result is still open.
        for name in frames:
            connection.execute(
                f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _source_{name}"
            ).fetchall()
        counts = {
            name: int(connection.execute(f"SELECT count(*) FROM {name}").fetchall()[0][0])
            for name in frames
        }
    finally:
        connection.close()
    return counts


def run() -> dict[str, int]:
    """Run the whole pipeline and return row counts for every produced table."""
    raw = load_raw()
    staging = build_staging(raw)
    marts = build_marts(staging)

    counts = {f"raw_{name}": len(frame) for name, frame in raw.items()}
    counts.update({f"staging_{name}": len(staging[name]) for name in STAGING_TABLES})
    counts.update({f"mart_{name}": value for name, value in marts.items()})
    return counts


if __name__ == "__main__":
    for key, value in run().items():
        print(f"{key:28s} {value:>8d}")
