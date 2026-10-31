"""Liquidity metrics built from the observable classifieds event log.

The platform records searches, impressions, views, contacts and self-reported
removals. It never observes a sale, so every number here is a *proxy* for
liquidity, not a transaction count.

Three conventions are load-bearing and are applied everywhere:

* **cohort** -- listings with ``publish_day < COHORT_WINDOW_DAYS``. A cohort is
  fixed by publish time, never by "still active at the end of the window":
  conditioning on survival flatters every liquidity number.
* **contact** -- one distinct ``(buyer_id, listing_id)`` pair. Repeat contacts by
  the same buyer on the same listing count once.
* **no platform average without the split** -- every frame carries a
  ``platform`` rollup row *next to* the per-category rows, so a single number can
  never be quoted on its own while the mix shift stays hidden.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SQL_DIR = PROJECT_ROOT / "sql"

# --- Cohort definition -------------------------------------------------------
COHORT_WINDOW_DAYS = 30.0

PLATFORM = "platform"
ALL_REGIONS = "all"

# --- Demand-sufficiency thresholds -------------------------------------------
# Fixed before the quadrant labels were looked at, so the labels cannot be fitted
# to a story. A "cell" is one (category, region); a category rollup pools six
# regions and therefore needs its own, much higher floor.
DEMAND_FLOOR_SEARCHES_PER_DAY_CELL = 8.0
DEMAND_FLOOR_SEARCHES_PER_DAY_CATEGORY = 60.0
HIGH_ZERO_RESULT_SHARE = 0.05
LOW_VIEW_RATE = 0.10
LOW_CONTACT_RATE = 0.10

QUADRANTS = ("no_demand", "no_views", "no_contact", "liquid")


# --------------------------------------------------------------------------- #
# Loading and shared primitives
# --------------------------------------------------------------------------- #
def load(raw_dir: Path | str = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Load every raw parquet table into a dict keyed by table name."""
    raw = Path(raw_dir)
    return {path.stem: pd.read_parquet(path) for path in sorted(raw.glob("*.parquet"))}


def contact_pairs(df: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Collapse the contacts log to one row per distinct ``(buyer_id, listing_id)``.

    The first contact row of each pair is kept, so the pair carries the session,
    position and reply status of the moment the conversation started.
    """
    contacts = df["contacts"]
    return contacts.sort_values("contact_id").drop_duplicates(["buyer_id", "listing_id"])


def run_sql(name: str, raw_dir: Path | str = RAW_DIR, **params: Any) -> pd.DataFrame:
    """Execute a SQL file from ``sql/`` over the raw parquet files.

    ``{{RAW}}`` is replaced by the raw-data directory (``read_parquet`` needs a
    literal path) and any other ``{{KEY}}`` by the matching keyword argument.
    """
    sql = (SQL_DIR / name).read_text(encoding="utf-8")
    sql = sql.replace("{{RAW}}", Path(raw_dir).as_posix())
    for key, value in params.items():
        sql = sql.replace("{{" + key + "}}", str(value))
    return duckdb.sql(sql).df()


# --------------------------------------------------------------------------- #
# Cohort liquidity
# --------------------------------------------------------------------------- #
def _cohort_stats(part: pd.DataFrame) -> dict[str, Any]:
    """Summary statistics for any slice of the per-listing cohort frame."""
    size = len(part)
    if size == 0:
        return {
            "cohort_size": 0,
            "listings_with_contact": float("nan"),
            "dead_listing_share": float("nan"),
            "median_time_to_first_contact_hours": float("nan"),
            "p90_time_to_first_contact_hours": float("nan"),
            "duplicate_share": float("nan"),
        }
    ttfc = part["time_to_first_contact_hours"]
    has_ttfc = ttfc.notna().any()
    return {
        "cohort_size": int(size),
        "listings_with_contact": float(part["has_contact"].mean()),
        "dead_listing_share": float(part["n_views"].eq(0).mean()),
        "median_time_to_first_contact_hours": float(ttfc.median()) if has_ttfc else float("nan"),
        "p90_time_to_first_contact_hours": float(ttfc.quantile(0.9)) if has_ttfc else float("nan"),
        "duplicate_share": float(part["is_duplicate"].mean()),
    }


def _per_listing_cohort(
    df: dict[str, pd.DataFrame], window_days: float
) -> pd.DataFrame:
    """One row per cohort listing with the primitives the shares are built from."""
    listings = df["listings"]
    cohort = listings[listings["publish_day"] < window_days]
    publish = cohort.set_index("listing_id")["publish_day"]
    cohort_ids = publish.index

    pairs = contact_pairs(df)
    pairs = pairs[pairs["listing_id"].isin(cohort_ids)].copy()
    pairs["publish_day"] = pairs["listing_id"].map(publish)
    # Contacts outside the observation window belong to a later state of the
    # listing and would silently lengthen the measured time-to-contact.
    pairs = pairs[pairs["day"] < pairs["publish_day"] + window_days]
    first_contact_day = pairs.groupby("listing_id")["day"].min()
    ttfc = (first_contact_day - publish.reindex(first_contact_day.index)) * 24.0

    views = df["views"]
    views = views[views["listing_id"].isin(cohort_ids)].copy()
    views["publish_day"] = views["listing_id"].map(publish)
    views = views[views["day"] < views["publish_day"] + window_days]
    n_views = views.groupby("listing_id").size()

    # Duplicates are a property of the whole catalogue, not of the cohort: the
    # sibling listing may live outside the cohort and still be a duplicate.
    duplicate_objects = listings.groupby("object_id").size().gt(1)

    per_listing = pd.DataFrame(index=cohort_ids)
    per_listing["category_id"] = cohort.set_index("listing_id")["category_id"]
    per_listing["region_id"] = cohort.set_index("listing_id")["region_id"]
    per_listing["has_contact"] = per_listing.index.isin(first_contact_day.index)
    per_listing["n_views"] = n_views.reindex(per_listing.index).fillna(0)
    per_listing["time_to_first_contact_hours"] = ttfc.reindex(per_listing.index)
    per_listing["is_duplicate"] = (
        cohort.set_index("listing_id")["object_id"].map(duplicate_objects).astype(bool)
    )
    return per_listing


def liquidity_by_cohort(
    df: dict[str, pd.DataFrame], window_days: float = COHORT_WINDOW_DAYS
) -> pd.DataFrame:
    """Liquidity of the publish cohort, per category x region plus rollups.

    Columns: ``cohort_size``, ``listings_with_contact``, ``dead_listing_share``,
    ``median_time_to_first_contact_hours``, ``p90_time_to_first_contact_hours``,
    ``duplicate_share``. Rollup rows use ``region_id == "all"`` and
    ``category_id == "platform"``.
    """
    per_listing = _per_listing_cohort(df, window_days)
    rows: list[dict[str, Any]] = []

    for (category, region), part in per_listing.groupby(["category_id", "region_id"], sort=True):
        rows.append({"category_id": category, "region_id": region, **_cohort_stats(part)})

    for category, part in per_listing.groupby("category_id", sort=True):
        rows.append({"category_id": category, "region_id": ALL_REGIONS, **_cohort_stats(part)})

    for region, part in per_listing.groupby("region_id", sort=True):
        rows.append({"category_id": PLATFORM, "region_id": region, **_cohort_stats(part)})

    rows.append(
        {
            "category_id": PLATFORM,
            "region_id": ALL_REGIONS,
            **_cohort_stats(per_listing),
        }
    )

    columns = [
        "category_id",
        "region_id",
        "cohort_size",
        "listings_with_contact",
        "dead_listing_share",
        "median_time_to_first_contact_hours",
        "p90_time_to_first_contact_hours",
        "duplicate_share",
    ]
    return pd.DataFrame(rows, columns=columns)


def cohort_contact_share_sql(
    raw_dir: Path | str = RAW_DIR, window_days: float = COHORT_WINDOW_DAYS
) -> pd.DataFrame:
    """Cohort contact share per category, straight from ``sql/cohort_liquidity.sql``."""
    return run_sql("cohort_liquidity.sql", raw_dir, WINDOW_DAYS=window_days)


# --------------------------------------------------------------------------- #
# Demand sufficiency
# --------------------------------------------------------------------------- #
def _quadrant(
    searches_per_day: float,
    zero_result_share: float,
    impressions: float,
    view_rate: float,
    contact_rate: float,
    searches_floor: float,
) -> str:
    """Map a cell's funnel onto one of four liquidity failure states.

    The thresholds are the module constants above; the checks are ordered from the
    earliest failure to the latest, because a cell that already has no demand
    cannot meaningfully be "low-contact".
    """
    if searches_per_day < searches_floor and zero_result_share > HIGH_ZERO_RESULT_SHARE:
        return "no_demand"
    if impressions > 0 and view_rate < LOW_VIEW_RATE:
        return "no_views"
    if view_rate > 0 and contact_rate < LOW_CONTACT_RATE:
        return "no_contact"
    return "liquid"


def demand_sufficiency(df: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Searches, zero-result share and funnel rates per category x region + rollups.

    ``quadrant`` labels each cell ``no_demand`` / ``no_views`` / ``no_contact`` /
    ``liquid``. Category rollups use the category search floor, cells use the
    (lower) cell floor; both are module constants.
    """
    searches = df["searches"]
    sessions = searches[["search_id", "category_id", "region_id", "zero_result"]]

    impressions = df["impressions"].merge(
        sessions[["search_id", "category_id", "region_id"]], on="search_id", how="left"
    )
    views = df["views"].merge(
        sessions[["search_id", "category_id", "region_id"]], on="search_id", how="left"
    )
    pairs = contact_pairs(df).merge(
        sessions[["search_id", "category_id", "region_id"]], on="search_id", how="left"
    )

    keys = ["category_id", "region_id"]
    cells = pd.DataFrame(
        {
            "searches": searches.groupby(keys).size(),
            "zero_results": searches.groupby(keys)["zero_result"].sum(),
            "impressions": impressions.groupby(keys).size(),
            "views": views.groupby(keys).size(),
            # pairs are already one row per (buyer_id, listing_id) globally
            "contacts": pairs.groupby(keys).size(),
        }
    ).fillna(0)

    data_days = float(searches["day"].max() - searches["day"].min()) or 1.0

    def _summarise(category: str, region: str, part: pd.DataFrame, floor: float) -> dict[str, Any]:
        n_searches = float(part["searches"].sum())
        n_impressions = float(part["impressions"].sum())
        n_views = float(part["views"].sum())
        n_contacts = float(part["contacts"].sum())
        zero_share = float(part["zero_results"].sum() / n_searches) if n_searches else float("nan")
        imp_per_search = n_impressions / n_searches if n_searches else float("nan")
        view_rate = n_views / n_impressions if n_impressions else float("nan")
        contact_rate = n_contacts / n_views if n_views else float("nan")
        per_day = n_searches / data_days
        return {
            "category_id": category,
            "region_id": region,
            "searches": int(n_searches),
            "zero_result_share": zero_share,
            "search_to_impression_rate": imp_per_search,
            "impressions": int(n_impressions),
            "views": int(n_views),
            "view_rate": view_rate,
            "contact_rate": contact_rate,
            "searches_per_day": per_day,
            "quadrant": _quadrant(
                per_day, zero_share, n_impressions, view_rate, contact_rate, floor
            ),
        }

    rows: list[dict[str, Any]] = []
    for (category, region), part in cells.groupby(level=keys, sort=True):
        rows.append(
            _summarise(category, region, part, DEMAND_FLOOR_SEARCHES_PER_DAY_CELL)
        )

    for category, part in cells.groupby(level="category_id", sort=True):
        rows.append(
            _summarise(category, ALL_REGIONS, part, DEMAND_FLOOR_SEARCHES_PER_DAY_CATEGORY)
        )

    for region, part in cells.groupby(level="region_id", sort=True):
        rows.append(
            _summarise(PLATFORM, region, part, DEMAND_FLOOR_SEARCHES_PER_DAY_CATEGORY)
        )

    rows.append(
        _summarise(PLATFORM, ALL_REGIONS, cells, DEMAND_FLOOR_SEARCHES_PER_DAY_CATEGORY)
    )

    columns = [
        "category_id",
        "region_id",
        "searches",
        "zero_result_share",
        "search_to_impression_rate",
        "impressions",
        "views",
        "view_rate",
        "contact_rate",
        "searches_per_day",
        "quadrant",
    ]
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------- #
# Zero-result rate
# --------------------------------------------------------------------------- #
def zero_result_rate(df: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Zero-result share by region population tier, with the per-region detail.

    The tier rows come from ``sql/zero_result.sql``. Region rows are attached so
    the tier average can never be quoted without the regions that produced it.
    """
    tier_frame = run_sql("zero_result.sql")
    tier_frame = tier_frame.assign(region_id=ALL_REGIONS)
    tier_frame["population_tier"] = tier_frame["population_tier"].astype(str)

    merged = df["searches"].merge(df["regions"], on="region_id", how="left")
    region_frame = (
        merged.groupby(["population_tier", "region_id"])
        .agg(searches=("search_id", "size"), zero_results=("zero_result", "sum"))
        .reset_index()
    )
    region_frame["population_tier"] = region_frame["population_tier"].astype(str)
    region_frame["zero_result_share"] = (
        region_frame["zero_results"] / region_frame["searches"]
    )

    overall = pd.DataFrame(
        [
            {
                "population_tier": "all",
                "region_id": ALL_REGIONS,
                "searches": int(len(merged)),
                "zero_results": int(merged["zero_result"].sum()),
                "zero_result_share": float(merged["zero_result"].mean()),
            }
        ]
    )

    columns = ["population_tier", "region_id", "searches", "zero_results", "zero_result_share"]
    return pd.concat(
        [
            tier_frame[columns],
            overall[columns],
            region_frame[columns],
        ],
        ignore_index=True,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the raw log and return the cohort, demand and zero-result frames."""
    frames = load()
    return (
        liquidity_by_cohort(frames),
        demand_sufficiency(frames),
        zero_result_rate(frames),
    )


if __name__ == "__main__":
    cohort_frame, demand_frame, zero_frame = run()
    print("=== Cohort liquidity (publish_day < 30) ===")
    print(cohort_frame.to_string(index=False))
    print("\n=== Demand sufficiency (per category x region) ===")
    print(demand_frame.to_string(index=False))
    print("\n=== Zero-result rate by population tier ===")
    print(zero_frame.to_string(index=False))
