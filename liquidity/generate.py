"""Synthetic classifieds generator — the only producer of latent truth.

Visible tables go to `data/raw/`. Latent quantities go to `_truth/ground_truth.json`
and are imported by tests only; `tests/test_no_truth_leak.py` fails if any analysis
module imports this file or the truth directory.

Domain assumption: the transaction happens off-platform. The platform therefore never
sees `true_sale`. Everything downstream — the whole point of this repo — is an attempt
to recover it from views, contacts and removals.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from liquidity import config

ROOT = config.PROJECT_ROOT
RAW_DIR = config.RAW_DIR
TRUTH_PATH = ROOT / "_truth" / "ground_truth.json"

# --- planted parameters. Must never be imported by analysis. --------------
REGIONS = [
    ("msk", "Москва", 1, 0.30, 1.00),
    ("spb", "Санкт-Петербург", 1, 0.15, 0.92),
    ("ekb", "Екатеринбург", 2, 0.12, 0.78),
    ("nov", "Новосибирск", 2, 0.11, 0.74),
    ("kzn", "Казань", 2, 0.10, 0.72),
    ("reg", "Прочие регионы", 3, 0.22, 0.55),
]

CATEGORIES = {
    "goods": {"name": "Товары", "share": 0.25, "true_sale_rate": 0.34,
              "p_mark_sold_given_sale": 0.92, "p_false_sold": 0.18, "dup_rate": 0.09,
              "log_price_mu": 8.9, "log_price_sigma": 1.15},
    "services": {"name": "Услуги", "share": 0.20, "true_sale_rate": 0.22,
                 "p_mark_sold_given_sale": 0.60, "p_false_sold": 0.05, "dup_rate": 0.04,
                 "log_price_mu": 8.2, "log_price_sigma": 1.05},
    "realty": {"name": "Недвижимость", "share": 0.15, "true_sale_rate": 0.14,
               "p_mark_sold_given_sale": 0.70, "p_false_sold": 0.10, "dup_rate": 0.03,
               "log_price_mu": 13.6, "log_price_sigma": 0.85},
    "auto": {"name": "Автомобили", "share": 0.30, "true_sale_rate": 0.26,
             "p_mark_sold_given_sale": 0.55, "p_false_sold": 0.02, "dup_rate": 0.05,
             "log_price_mu": 13.1, "log_price_sigma": 0.95},
    "jobs": {"name": "Работа", "share": 0.10, "true_sale_rate": 0.30,
             "p_mark_sold_given_sale": 0.80, "p_false_sold": 0.12, "dup_rate": 0.06,
             "log_price_mu": 11.2, "log_price_sigma": 0.70},
}

CAT_IDS = list(CATEGORIES)
CAT_P = np.array([CATEGORIES[c]["share"] for c in CAT_IDS], dtype=float)
CAT_P = CAT_P / CAT_P.sum()

N_SELECTED_SELLERS = 3000
N_BUYERS = 9000
N_LISTINGS = 6000
N_SEARCHES = 40000
RESULTS_PER_SEARCH = 20
P_REMOVED_IF_UNSOLD = 0.45
BOOST_SHARE = 0.18
PROMOTED_SCORE_BONUS = 1.60


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _ts(day: float) -> str:
    base = pd.Timestamp(config.DATA_START)
    return (base + pd.to_timedelta(float(day), unit="D")).isoformat()


def build_sellers(rng: np.random.Generator) -> pd.DataFrame:
    n = N_SELECTED_SELLERS
    seller_type = rng.choice(["private", "pro", "business"], size=n, p=[0.62, 0.26, 0.12])
    region_ids = np.array([r[0] for r in REGIONS])
    region_p = np.array([r[3] for r in REGIONS])
    region_p = region_p / region_p.sum()

    rank_weights = 1.0 / np.arange(1, n + 1) ** 1.05
    activity = rank_weights[rng.permutation(n)]
    activity = activity / activity.sum()

    return pd.DataFrame({
        "seller_id": np.arange(n),
        "seller_type": seller_type,
        "region_id": rng.choice(region_ids, size=n, p=region_p),
        "tenure_days": rng.integers(0, 900, size=n),
        "verified": (rng.random(n) < np.where(seller_type == "private", 0.35, 0.88)).astype(int),
        "activity_weight": activity,
    })


def build_listings(rng: np.random.Generator, sellers: pd.DataFrame) -> pd.DataFrame:
    n = N_LISTINGS
    cat_idx = rng.choice(len(CAT_IDS), size=n, p=CAT_P)
    category_id = np.array(CAT_IDS)[cat_idx]

    seller_idx = rng.choice(N_SELECTED_SELLERS, size=n, p=sellers["activity_weight"].to_numpy())
    seller = sellers.iloc[seller_idx].reset_index(drop=True)

    mu = np.array([CATEGORIES[c]["log_price_mu"] for c in category_id])
    sigma = np.array([CATEGORIES[c]["log_price_sigma"] for c in category_id])
    price = np.exp(rng.normal(mu, sigma)).round(-1)

    tenure = seller["tenure_days"].to_numpy() / 900.0
    verified = seller["verified"].to_numpy()
    type_bonus = np.where(seller["seller_type"].to_numpy() == "private", -0.05, 0.06)
    quality = np.clip(rng.beta(2.2, 3.0, size=n) + 0.10 * tenure + 0.05 * verified + type_bonus, 0.02, 0.99)

    publish_day = rng.random(n) * config.N_DAYS
    relevance = np.clip(0.55 * quality + 0.25 * rng.random(n) + 0.20 * np.exp(-publish_day / 60.0), 0.01, 1.0)

    # Boost purchases are endogenous: weak organic listings buy visibility.
    rank_pct = pd.Series(relevance).rank(pct=True).to_numpy()
    p_boost = _sigmoid(-4.2 * (rank_pct - 0.45)) * np.where(seller["seller_type"].to_numpy() == "private", 0.55, 1.0)
    p_boost = np.clip(p_boost * BOOST_SHARE * 2.4, 0.0, 0.75)
    boost_bought = (rng.random(n) < p_boost) & (publish_day < config.N_DAYS - 8)
    boost_start = np.where(boost_bought, np.minimum(publish_day + rng.integers(3, 26, size=n), config.N_DAYS - 3), np.inf)
    boost_end = boost_start + rng.integers(7, 31, size=n)

    listings = pd.DataFrame({
        "listing_id": np.arange(n),
        "object_id": np.arange(n),
        "seller_id": seller["seller_id"].to_numpy(),
        "seller_type": seller["seller_type"].to_numpy(),
        "category_id": category_id,
        "region_id": seller["region_id"].to_numpy(),
        "price_rub": price,
        "quality_score": quality.round(4),
        "relevance_static": relevance.round(4),
        "publish_day": publish_day.round(5),
        "boost_purchased": boost_bought.astype(int),
        "boost_start_day": boost_start,
        "boost_end_day": boost_end,
    })
    listings["publish_ts"] = [_ts(d) for d in listings["publish_day"]]

    # Duplicates: same object_id reused by a different seller in the same category/region.
    dup_rate = np.array([CATEGORIES[c]["dup_rate"] for c in category_id])
    dup_rows = rng.random(n) < dup_rate
    for idx in np.flatnonzero(dup_rows)[: int(dup_rows.sum())]:
        pool = np.flatnonzero(
            (listings["category_id"].to_numpy() == category_id[idx])
            & (listings["region_id"].to_numpy() == listings["region_id"].to_numpy()[idx])
            & (np.arange(n) < idx)
        )
        if pool.size:
            listings.loc[idx, "object_id"] = listings.loc[rng.choice(pool), "object_id"]
    return listings


def build_searches(rng: np.random.Generator, listings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = N_SEARCHES
    cat_idx = rng.choice(len(CAT_IDS), size=n, p=CAT_P)
    category_id = np.array(CAT_IDS)[cat_idx]
    region_ids = np.array([r[0] for r in REGIONS])
    region_p = np.array([r[3] for r in REGIONS])
    region_p = region_p / region_p.sum()
    region_id = rng.choice(region_ids, size=n, p=region_p)

    searches = pd.DataFrame({
        "search_id": np.arange(n),
        "buyer_id": rng.integers(0, N_BUYERS, size=n),
        "day": (rng.random(n) * config.N_DAYS).round(5),
        "category_id": category_id,
        "region_id": region_id,
        "in_holdout": (rng.random(n) < config.HOLDOUT_SHARE).astype(int),
    })

    pools: dict[tuple[str, str], np.ndarray] = {}
    for (c, r), g in listings.groupby(["category_id", "region_id"], sort=False):
        pools[(c, str(r))] = g.index.to_numpy()

    tier = {r[0]: r[2] for r in REGIONS}
    thin = np.array([tier[r] for r in region_id])
    p_zero = np.where(thin == 3, 0.09, np.where(thin == 2, 0.035, 0.012))
    zero_mask = rng.random(n) < p_zero

    imp_search, imp_listing, imp_pos, imp_promoted = [], [], [], []
    relev = listings["relevance_static"].to_numpy()
    pubday = listings["publish_day"].to_numpy()
    b_start = listings["boost_start_day"].to_numpy()
    b_end = listings["boost_end_day"].to_numpy()

    for (c, r), g in searches.groupby(["category_id", "region_id"], sort=False):
        pool = pools.get((c, str(r)))
        if pool is None or pool.size == 0:
            continue
        rows = g.index.to_numpy()
        keep = rows[~zero_mask[rows]]
        if keep.size == 0:
            continue
        m = keep.size
        k = min(RESULTS_PER_SEARCH, pool.size)
        day = searches["day"].to_numpy()[keep][:, None]
        rand = rng.random((m, pool.size))
        rand[pubday[pool][None, :] > day] = 2.0   # a listing cannot be shown before it exists
        order = np.argsort(rand, axis=1)[:, :k]
        cand = pool[order]                       # (m, k)
        hold = searches["in_holdout"].to_numpy()[keep][:, None]

        fresh = np.exp(-np.clip(day - pubday[cand], 0, None) / 45.0)
        active = (b_start[cand] <= day) & (day < b_end[cand]) & (hold == 0)
        score = 0.55 * relev[cand] + 0.45 * fresh + 0.30 * rng.random((m, k))
        score = score + PROMOTED_SCORE_BONUS * active
        pos = np.argsort(-score, axis=1)

        s_rep = np.repeat(keep, k).reshape(m, k)
        l_rep = cand
        p_rep = np.empty_like(pos)
        p_rep[np.arange(m)[:, None], pos] = np.arange(1, k + 1)
        prom_rep = active
        imp_search.append(s_rep.ravel())
        imp_listing.append(l_rep.ravel())
        imp_pos.append(p_rep.ravel())
        imp_promoted.append(prom_rep.ravel())

    if not imp_search:
        raise RuntimeError("no impressions generated — listing pools too thin")

    impressions = pd.DataFrame({
        "search_id": np.concatenate(imp_search),
        "listing_id": np.concatenate(imp_listing),
        "position": np.concatenate(imp_pos).astype(int),
        "is_promoted": np.concatenate(imp_promoted).astype(int),
    })
    impressions["buyer_id"] = searches["buyer_id"].to_numpy()[impressions["search_id"].to_numpy()]
    impressions["day"] = searches["day"].to_numpy()[impressions["search_id"].to_numpy()]

    # Drop any residual pre-publication impression and re-rank so positions stay contiguous.
    impressions = impressions[
        pubday[impressions["listing_id"].to_numpy()] <= impressions["day"].to_numpy()
    ].reset_index(drop=True)
    impressions = impressions.sort_values(["search_id", "position"], kind="stable").reset_index(drop=True)
    impressions["position"] = impressions.groupby("search_id").cumcount() + 1
    impressions.insert(0, "impression_id", np.arange(len(impressions)))

    served = impressions.groupby("search_id").size()
    searches["n_results"] = searches["search_id"].map(served).fillna(0).astype(int)
    searches["zero_result"] = (searches["n_results"] == 0).astype(int)
    searches["ts"] = [_ts(d) for d in searches["day"]]
    return searches, impressions


def build_engagement(
    rng: np.random.Generator,
    impressions: pd.DataFrame,
    listings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Serve views and contacts from impressions.

    Returns the impressions alongside views and contacts. Every view and contact inherits
    its parent impression's day, which is what lets the removal cut-off be applied to one
    table and hold for all three.
    """
    q = listings["quality_score"].to_numpy()[impressions["listing_id"].to_numpy()]
    pos = impressions["position"].to_numpy()
    promoted = impressions["is_promoted"].to_numpy()

    p_view = np.clip(0.55 * np.exp(-(pos - 1) / 6.5) * (0.55 + 0.90 * q), 0.0, 1.0)
    viewed = rng.random(len(impressions)) < p_view
    views = impressions[viewed].copy()
    views.insert(0, "view_id", np.arange(len(views)))

    vq = q[viewed]
    vprom = promoted[viewed]
    p_contact = np.clip(0.05 + 0.30 * vq + 0.02 * vprom, 0.005, 0.60)
    contacted = rng.random(len(views)) < p_contact
    contacts = views[contacted].copy()
    contacts.insert(0, "contact_id", np.arange(len(contacts)))

    seller_type = listings["seller_type"].to_numpy()[contacts["listing_id"].to_numpy()]
    cq = q[viewed][contacted]
    responsiveness = np.clip(0.28 + 0.55 * cq + np.where(seller_type == "private", 0.0, 0.10), 0.05, 0.97)
    replied = rng.random(len(contacts)) < responsiveness
    contacts["channel"] = rng.choice(["call", "message"], size=len(contacts), p=[0.42, 0.58])
    contacts["seller_replied"] = replied.astype(int)
    contacts["reply_lag_hours"] = np.where(replied, rng.exponential(6.0, size=len(contacts)).round(3), np.nan)
    contacts["ts"] = [_ts(d) for d in contacts["day"]]
    views["ts"] = [_ts(d) for d in views["day"]]
    impressions["ts"] = [_ts(d) for d in impressions["day"]]
    return impressions, views, contacts


SOLD_REPLY_LAG_MAX = 4.5


def _compute_sold(
    rng: np.random.Generator,
    listings: pd.DataFrame,
    contacts: pd.DataFrame,
    removal_schedule: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose which listings sold, from replied contacts inside the sale window.

    Returns ``(sold, has_replied, first_replied, sale_lag)``. A listing can only sell if
    a seller replied inside ``LIQUIDITY_WINDOW_DAYS`` of publish. When a removal schedule
    is supplied the sale must also fit it: the listing is actually removed, and the
    removal follows the first reply by at most ``SOLD_REPLY_LAG_MAX`` days. That lag is
    the only honest separator the audit estimator is allowed to observe.
    """
    n = len(listings)
    replied = contacts[contacts["seller_replied"] == 1]
    # The sale window is measured from the FIRST seller reply: a listing is sold if a
    # conversation happened within 30 days of publish. Using the last reply would
    # exclude every early-published listing that kept receiving contacts afterwards.
    first_replied_day = replied.groupby("listing_id")["day"].min()
    has_replied = np.zeros(n, dtype=bool)
    first_replied = np.full(n, np.inf)
    idx = first_replied_day.index.to_numpy()
    has_replied[idx] = True
    first_replied[idx] = first_replied_day.to_numpy()

    pubday = listings["publish_day"].to_numpy()
    sale_lag = rng.integers(0, 5, size=n)
    eligible = has_replied & (
        np.clip(first_replied - pubday, 0, None) + sale_lag <= config.LIQUIDITY_WINDOW_DAYS
    )
    if removal_schedule is not None:
        both = np.isfinite(removal_schedule) & np.isfinite(first_replied)
        lag_from_reply = np.full(n, np.inf)
        lag_from_reply[both] = removal_schedule[both] - first_replied[both]
        eligible &= (
            np.isfinite(removal_schedule)
            & (lag_from_reply >= 0)
            & (lag_from_reply <= SOLD_REPLY_LAG_MAX)
        )

    q = listings["quality_score"].to_numpy()
    score = 0.45 * q + 0.30 * rng.random(n) + np.where(has_replied, 0.55, 0.0)
    score = np.where(eligible, score, -np.inf)

    cat = listings["category_id"].to_numpy()
    sold = np.zeros(n, dtype=bool)
    for c in CAT_IDS:
        rows = np.flatnonzero(cat == c)
        target = int(round(CATEGORIES[c]["true_sale_rate"] * rows.size))
        order = rows[np.argsort(-score[rows])]
        take = order[np.isfinite(score[order])][:target]
        sold[take] = True
    return sold, has_replied, first_replied, sale_lag


def provisional_removal(
    rng: np.random.Generator,
    listings: pd.DataFrame,
    contacts: pd.DataFrame,
) -> np.ndarray:
    """Pass-one removal schedule, derived before the engagement cut-off exists.

    Sold listings are taken down 0-4 days after the first replied contact; unsold
    listings the seller bothers to remove leave 1-60 days after publish, independent of
    any contact. ``inf`` marks a listing that stays active for the whole window.
    """
    n = len(listings)
    sold, _, first_replied, sale_lag = _compute_sold(rng, listings, contacts)
    pubday = listings["publish_day"].to_numpy()
    sale_day = np.clip(first_replied - pubday, 0, None) + sale_lag
    removed_unsold = (~sold) & (rng.random(n) < P_REMOVED_IF_UNSOLD)
    removal_offset = np.where(sold, sale_day, rng.uniform(1.0, 60.0, size=n))
    removal_day = pubday + removal_offset
    removed = (sold | removed_unsold) & (removal_day <= config.N_DAYS)
    return np.where(removed, removal_day, np.inf)


def cut_after_removal(df: pd.DataFrame, schedule: np.ndarray) -> pd.DataFrame:
    """Drop every row whose event day falls after its listing's removal day."""
    keep = df["day"].to_numpy() <= schedule[df["listing_id"].to_numpy()]
    return df[keep].reset_index(drop=True)


def resolve_outcomes(
    rng: np.random.Generator,
    listings: pd.DataFrame,
    contacts: pd.DataFrame,
    removal_schedule: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Finalise sold / removed status against the removal schedule and the visible marks.

    Passing a schedule makes the outcome a function of the pass-two (cut) engagement log;
    without one the schedule is derived here, which is what a single-pass caller gets.
    """
    n = len(listings)
    if removal_schedule is None:
        removal_schedule = provisional_removal(rng, listings, contacts)

    sold, has_replied, _, _ = _compute_sold(rng, listings, contacts, removal_schedule)
    removed = np.isfinite(removal_schedule)
    removal_day = np.where(removed, removal_schedule, np.nan)
    cat = listings["category_id"].to_numpy()

    # visible removal reason
    p_ts = np.array([CATEGORIES[c]["p_mark_sold_given_sale"] for c in cat])
    p_fs = np.array([CATEGORIES[c]["p_false_sold"] for c in cat])
    r = rng.random(n)
    other = rng.choice(["withdrawn", "expired", "violation"], size=n, p=[0.58, 0.32, 0.10])
    reason_sold = np.where(sold, r < p_ts, r < p_fs)
    visible_reason = np.where(reason_sold, "sold", other)

    listings["status"] = np.where(removed, "removed", "active")
    listings["removal_reason"] = np.where(removed, visible_reason, None)
    listings["removal_day"] = np.where(removed, removal_day, np.nan)
    listings["status_ts"] = [
        _ts(d) if np.isfinite(d) else None for d in listings["removal_day"].to_numpy()
    ]

    cohort = listings["publish_day"].to_numpy() < config.COHORT_DAYS
    latent = {}
    proxy = {}
    confusion = {}
    for c in CAT_IDS:
        m = cohort & (cat == c)
        removed_m = m & removed
        latent[c] = round(float(sold[m].mean()), 6)
        proxy[c] = round(float((listings["removal_reason"].to_numpy()[m] == "sold").mean()), 6)
        true_sold = sold & removed_m
        false_sold = (~sold) & removed_m
        confusion[c] = {
            "cohort_size": int(m.sum()),
            "sold_share_marked_sold": round(float((listings["removal_reason"].to_numpy()[true_sold] == "sold").mean()), 6) if true_sold.any() else None,
            "unsold_share_marked_sold": round(float((listings["removal_reason"].to_numpy()[false_sold] == "sold").mean()), 6) if false_sold.any() else None,
        }

    weights = np.array([CATEGORIES[c]["share"] for c in CAT_IDS])
    weights = weights / weights.sum()
    platform_true = float(sum(w * latent[c] for c, w in zip(CAT_IDS, weights, strict=True)))
    platform_proxy = float(sum(w * proxy[c] for c, w in zip(CAT_IDS, weights, strict=True)))

    return listings, {
        "latent": {
            "sell_through_true_by_category": latent,
            "sell_through_proxy_by_category": proxy,
            "removal_confusion_by_category": confusion,
            "platform_true": round(platform_true, 6),
            "platform_proxy": round(platform_proxy, 6),
        },
        "boost_share_realized": round(float(listings["boost_purchased"].mean()), 6),
        "reply_rate_by_sold": {
            "sold": round(float(np.mean(has_replied[sold])), 6) if sold.any() else None,
            "unsold": round(float(np.mean(has_replied[~sold])), 6) if (~sold).any() else None,
        },
    }


def summarise_experiment(searches: pd.DataFrame, contacts: pd.DataFrame) -> dict:
    counts = contacts.groupby("search_id").size()
    s = searches.copy()
    s["n_contacts"] = s["search_id"].map(counts).fillna(0).astype(int)

    served = s[s["n_results"] > 0]
    hold_served = served[served["in_holdout"] == 1]
    treat_served = served[served["in_holdout"] == 0]
    holdout_contacts = float(hold_served["n_contacts"].sum())
    treated_contacts = float(treat_served["n_contacts"].sum())

    unpaid = contacts[contacts["is_promoted"] == 0]
    paid = contacts[contacts["is_promoted"] == 1]
    return {
        "treated_sessions": int(len(treat_served)),
        "holdout_sessions": int(len(hold_served)),
        "contacts_per_treated_session": round(treated_contacts / max(len(treat_served), 1), 6),
        "contacts_per_holdout_session": round(holdout_contacts / max(len(hold_served), 1), 6),
        "incrementality_per_session": round(
            treated_contacts / max(len(treat_served), 1) - holdout_contacts / max(len(hold_served), 1), 6
        ),
        "unpaid_contacts_per_treated_session": round(
            float(unpaid["search_id"].isin(treat_served["search_id"]).sum()) / max(len(treat_served), 1), 6
        ),
        "unpaid_contacts_per_holdout_session": round(
            float(unpaid["search_id"].isin(hold_served["search_id"]).sum()) / max(len(hold_served), 1), 6
        ),
        "paid_contacts": int(len(paid)),
        "unpaid_contacts": int(len(unpaid)),
        "reply_rate_paid": round(float(paid["seller_replied"].mean()), 6) if len(paid) else None,
        "reply_rate_unpaid": round(float(unpaid["seller_replied"].mean()), 6) if len(unpaid) else None,
    }


def main() -> None:
    rng = np.random.default_rng(config.SEED)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)

    sellers = build_sellers(rng)
    listings = build_listings(rng, sellers)
    searches, impressions = build_searches(rng, listings)

    # Pass one: engagement with no removal cut-off. Its only job is to fix the day each
    # listing leaves the window — the pipeline is circular, so this is the provisional
    # leg of a two-pass approximation, not a fixed point.
    impressions, views, contacts = build_engagement(
        np.random.default_rng(config.SEED + 1), impressions, listings
    )
    schedule = provisional_removal(
        np.random.default_rng(config.SEED + 2), listings, contacts
    )

    # Pass two: re-serve the same draws with the removal cut-off applied. Views and
    # contacts inherit their impression's day, so cutting each table by the schedule
    # yields exactly the log a listing would have produced had it stopped being shown on
    # its removal day — with no engagement event left after the removal.
    impressions = cut_after_removal(impressions, schedule)
    views = cut_after_removal(views, schedule)
    contacts = cut_after_removal(contacts, schedule)
    # Ids are left as generated: a view's parent impression has the same day and listing,
    # so it is cut with it, and every surviving foreign key still resolves.

    served = impressions.groupby("search_id").size()
    searches["n_results"] = searches["search_id"].map(served).fillna(0).astype(int)
    searches["zero_result"] = (searches["n_results"] == 0).astype(int)

    listings, truth = resolve_outcomes(
        np.random.default_rng(config.SEED + 4), listings, contacts, schedule
    )

    experiment = summarise_experiment(searches, contacts)

    out = {
        "sellers": sellers.drop(columns=["activity_weight"]),
        "categories": pd.DataFrame(
            [{"category_id": c, "category_name": v["name"]} for c, v in CATEGORIES.items()]
        ),
        "regions": pd.DataFrame(
            [{"region_id": r[0], "region_name": r[1], "population_tier": r[2]} for r in REGIONS]
        ),
        "listings": listings,
        "searches": searches,
        "impressions": impressions,
        "views": views,
        "contacts": contacts,
    }
    for name, df in out.items():
        df.to_parquet(RAW_DIR / f"{name}.parquet", index=False)

    truth_doc = {
        "seed": config.SEED,
        "data_start": config.DATA_START,
        "cohort_days": config.COHORT_DAYS,
        "liquidity_window_days": config.LIQUIDITY_WINDOW_DAYS,
        **truth,
        "experiment": experiment,
        "expected_signs": {
            c: ("positive" if truth["latent"]["sell_through_proxy_by_category"][c] >
                truth["latent"]["sell_through_true_by_category"][c] else "negative")
            for c in CAT_IDS
        },
    }
    TRUTH_PATH.write_text(json.dumps(truth_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"listings={len(listings)} searches={len(searches)} impressions={len(impressions)} "
          f"views={len(views)} contacts={len(contacts)}")
    print(f"platform true={truth['latent']['platform_true']:.4f} "
          f"proxy={truth['latent']['platform_proxy']:.4f}")
    for c in CAT_IDS:
        t = truth["latent"]["sell_through_true_by_category"][c]
        p = truth["latent"]["sell_through_proxy_by_category"][c]
        print(f"  {c:9s} true={t:.4f} proxy={p:.4f} err={p - t:+.4f}")
    print(f"incrementality/session={experiment['incrementality_per_session']:+.4f}")


if __name__ == "__main__":
    main()
