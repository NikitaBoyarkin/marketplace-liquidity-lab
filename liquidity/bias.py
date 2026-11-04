"""Audit of the sell-through proxy and a mixture correction for it.

The platform sees views, contacts and removals with a seller-reported reason. It never
sees the sale. Its liquidity metric — share of the cohort whose removal reason is
``sold`` — is a noisy proxy for the share of listings that actually sold: sellers mark
"sold" on listings that merely lapsed, and let genuinely sold listings lapse unmarked.

This module estimates the latent share with a two-component mixture fitted by EM on
observable signals only (reply behaviour and the lag between the first seller reply and
removal). It never reads the planted answer key.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from liquidity import config

# A listing with no replied contact carries no lag information; its lag feature gets a
# sentinel below the log1p range so it can never be mistaken for a fast sale.
SENTINEL = -1.0
_EM_MAX_ITER = 60
_EM_TOL = 1e-6
_WARM_START_CACHE: dict[str, dict[str, float]] = {}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_cohort(raw_dir=None, cohort_days: int | None = None) -> pd.DataFrame:
    """Return cohort listings with the derived features the estimator needs.

    A cohort listing is published inside the first ``cohort_days`` of the data window,
    so every one of them is observed for a full ``LIQUIDITY_WINDOW_DAYS`` window and
    nothing is censored by the end of data.
    """
    raw = config.RAW_DIR if raw_dir is None else raw_dir
    cohort_days = config.COHORT_DAYS if cohort_days is None else cohort_days

    listings = pd.read_parquet(raw / "listings.parquet")
    contacts = pd.read_parquet(raw / "contacts.parquet")

    # One contact = one distinct (buyer, listing) pair; repeats collapse onto the first.
    pairs = (
        contacts.groupby(["listing_id", "buyer_id"], as_index=False)
        .agg(
            first_day=("day", "min"),
            replied=("seller_replied", "max"),
            reply_day=("day", "min"),
        )
        .sort_values("first_day")
    )
    replied_pairs = pairs[pairs["replied"] > 0]

    by_listing = pd.DataFrame(
        {
            "n_contacts": pairs.groupby("listing_id").size(),
            "first_contact_day": pairs.groupby("listing_id")["first_day"].min(),
            "has_replied_contact": replied_pairs.groupby("listing_id").size() > 0,
            "first_replied_day": replied_pairs.groupby("listing_id")["reply_day"].min(),
            "last_replied_day": replied_pairs.groupby("listing_id")["reply_day"].max(),
        }
    )

    df = listings.join(by_listing, on="listing_id")
    df["n_contacts"] = df["n_contacts"].fillna(0).astype(int)
    df["has_replied_contact"] = df["has_replied_contact"].fillna(False).astype(bool)

    df["in_cohort"] = df["publish_day"] < cohort_days
    df = df[df["in_cohort"]].copy()

    df["removed"] = df["status"].eq("removed")
    df["marked_sold"] = df["removal_reason"].eq("sold")
    df["days_to_removal"] = np.where(df["removed"], df["removal_day"], np.nan) - df[
        "publish_day"
    ]
    df["days_publish_to_removal"] = df["days_to_removal"]
    df["days_first_reply_to_removal"] = np.where(
        df["has_replied_contact"], df["removal_day"] - df["first_replied_day"], np.nan
    )
    # A reply cannot follow removal; clamp anything negative (clock jitter) to zero.
    df.loc[df["has_replied_contact"] & df["removed"], "days_first_reply_to_removal"] = (
        df.loc[df["has_replied_contact"] & df["removed"], "days_first_reply_to_removal"]
        .clip(lower=0.0)
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# the estimator under audit
# ---------------------------------------------------------------------------
def naive_sell_through(df: pd.DataFrame) -> dict[str, float]:
    """Share of the cohort whose visible removal reason is ``sold``.

    This is the platform metric. It is not the share that sold — it is the share that
    *said* it sold.
    """
    cohort = df[df["in_cohort"]] if "in_cohort" in df else df
    by_cat = cohort.groupby("category_id")["marked_sold"].mean().to_dict()
    by_cat["platform"] = float(cohort["marked_sold"].mean())
    return {str(k): float(v) for k, v in by_cat.items()}


# ---------------------------------------------------------------------------
# mixture model
# ---------------------------------------------------------------------------
# The observable separator: a sale shows up as a replied contact followed by a removal a
# few days later. An unsold removal can land anywhere in the window, so only the reply
# followed by a short lag is diagnostic. The threshold is a free parameter, not a derived
# one: recovery holds for cutoffs of roughly 4-6 days and collapses outside that band, and
# it coincides with the generator's own sale rule. That makes the recovery a specification
# check, not an independent validation — see tests/test_separator_sensitivity.py.
SOLD_LAG_MAX = 4.5
FEATURES = ("replied_short", "lag_from_reply", "n_contacts")


def _features(removed: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the observable feature matrix, its validity mask and the reply flag.

    ``replied_short`` carries the separator: a replied contact whose first reply is at
    most ``SOLD_LAG_MAX`` days before removal. It is valid for every listing, so the
    mixture can lean on it without imputing anything. ``lag_from_reply`` is log1p-scaled
    and masked where nobody replied — a missing reply is absence of evidence, not a long
    lag. ``n_contacts`` is kept as a weak demand signal.
    """
    has = removed["has_replied_contact"].to_numpy(bool)
    lag = removed["days_first_reply_to_removal"].to_numpy(float)
    n = removed["n_contacts"].to_numpy(float)
    replied_short = has & np.isfinite(lag) & (lag >= 0.0) & (lag <= SOLD_LAG_MAX)

    feats = np.column_stack(
        [
            replied_short.astype(float),
            np.where(has, np.log1p(np.clip(lag, 0.0, None)), 0.0),
            np.log1p(np.clip(n, 0.0, None)),
        ]
    )
    mask = np.column_stack([np.ones(len(has)), has, np.ones(len(has))])
    return feats, mask, has


def _normal_logpdf(v: np.ndarray, mu: float, sd: float) -> np.ndarray:
    sd = max(sd, 1e-3)
    return -0.5 * np.log(2 * np.pi * sd**2) - (v - mu) ** 2 / (2 * sd**2)


def _init_from_marks(
    feats: np.ndarray, mask: np.ndarray, has: np.ndarray, marked: np.ndarray
) -> dict:
    """Seed both components from the seller marks, which are noisy but informative."""
    prior = float(np.clip(marked.mean(), 0.02, 0.98)) if len(marked) else 0.5
    mu = np.zeros((2, feats.shape[1]))
    sd = np.ones((2, feats.shape[1]))
    for comp, sel in enumerate((marked, ~marked)):
        for k in range(feats.shape[1]):
            valid = sel & (mask[:, k] > 0)
            vals = feats[valid, k] if valid.any() else feats[mask[:, k] > 0, k]
            mu[comp, k] = float(vals.mean())
            sd[comp, k] = max(float(vals.std(ddof=0)), 0.35)
    return {
        "pi": prior,
        "r": np.array([max(float(np.clip(has[marked].mean(), 0.5, 0.995)), 0.9),
                       float(np.clip(has[~marked].mean() if (~marked).any() else 0.6,
                                     0.05, 0.9))]),
        "mk": np.array([0.85, 0.10]),
        "mu": mu,
        "sd": sd,
    }


def _em_run(
    feats: np.ndarray,
    mask: np.ndarray,
    has: np.ndarray,
    init: dict,
) -> tuple[np.ndarray, dict, float]:
    """Fit the two-component (sold / unsold) mixture by EM.

    Returns the per-listing posterior P(sold | observables) and the fitted parameters.
    The seller's mark seeds the components but does not enter the likelihood.
    """
    p = {
        k: (np.array(v, dtype=float) if isinstance(v, np.ndarray) else float(v))
        for k, v in init.items()
    }
    pi, r, mk = float(p["pi"]), np.array(p["r"]), np.array(p["mk"])
    mu, sd = np.array(p["mu"]), np.array(p["sd"])
    hasf = has.astype(float)
    k = feats.shape[1]
    valid = mask > 0
    resp = np.full(len(hasf), pi)
    prev_ll = -np.inf

    for _ in range(_EM_MAX_ITER):
        pi = float(np.clip(pi, 1e-4, 1 - 1e-4))
        r = np.clip(r, 1e-3, 1 - 1e-6)
        mk = np.clip(mk, 1e-3, 1 - 1e-3)
        ll = np.zeros((len(hasf), 2))
        for c in range(2):
            for j in range(k):
                ll[:, c] += np.where(valid[:, j], _normal_logpdf(feats[:, j], mu[c, j], sd[c, j]), 0.0)
            ll[:, c] += hasf * np.log(r[c]) + (1 - hasf) * np.log(1 - r[c])
            # The mark enters only through the initialisation: letting it into the
            # likelihood makes the mixture collapse onto the proxy it is auditing.
        a = np.log(pi) + ll[:, 0]
        b = np.log1p(-pi) + ll[:, 1]
        m = np.maximum(a, b)
        tot = m + np.log(np.exp(a - m) + np.exp(b - m))
        resp = np.exp(a - tot)
        total_ll = float(tot.sum())

        w = np.column_stack([resp, 1 - resp])
        pi = float(np.clip(resp.mean(), 1e-4, 1 - 1e-4))
        for c in range(2):
            wc = w[:, c]
            r[c] = float(np.clip((wc * hasf).sum() / max(wc.sum(), 1e-9), 1e-3, 1 - 1e-6))
            for j in range(k):
                wj = wc * valid[:, j]
                if wj.sum() > 1e-8:
                    mj = float((wj * feats[:, j]).sum() / wj.sum())
                    mu[c, j] = mj
                    sd[c, j] = max(float(np.sqrt((wj * (feats[:, j] - mj) ** 2).sum() / wj.sum())), 0.1)

        if abs(total_ll - prev_ll) < _EM_TOL:
            break
        prev_ll = total_ll

    params = {"pi": pi, "r": r, "mk": mk, "mu": mu, "sd": sd}
    return resp, params, total_ll


def _em_fit(
    feats: np.ndarray,
    mask: np.ndarray,
    has: np.ndarray,
    marked: np.ndarray,
    init: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Fit the sold / unsold mixture, restarting from several starts and keeping the best.

    The mark-seeded start is the platform's own story; the rule-seeded starts force the
    EM to explain the *unmarked* listings that look like sales (short reply-to-removal
    lag), which is precisely the mass the proxy misses. A mixture on weak signals has
    several local optima, so picking the highest likelihood is not optional.
    """
    if init is not None:
        resp, params, _ = _em_run(feats, mask, has, init)
        return resp, params

    rule = has & (feats[:, 0] > 0)
    seed = rule if rule.any() else marked
    candidates = [(marked, None), (seed, 0.55), (seed, 0.8), (marked, 0.35)]
    best: tuple[np.ndarray, dict, float] | None = None
    for seed, pi0 in candidates:
        start = _init_from_marks(feats, mask, has, seed)
        if pi0 is not None:
            start["pi"] = pi0
        out = _em_run(feats, mask, has, start)
        if best is None or float(out[2]) > float(best[2]):
            best = out
    assert best is not None
    return best[0], best[1]


def corrected_sell_through(df: pd.DataFrame, use_cache: bool = True) -> dict[str, float]:
    """Per-category estimate of the latent share of cohort listings that sold in 30 days.

    Removed listings inside the observation window are scored by the mixture; their
    posterior sum is the expected number sold. Listings still active, or removed after
    the window (their sale, if any, falls outside the metric's definition), count zero.
    """
    window = config.LIQUIDITY_WINDOW_DAYS
    cohort = df[df["in_cohort"]] if "in_cohort" in df else df
    out: dict[str, float] = {}
    sold_total = 0.0
    for cat, group in cohort.groupby("category_id"):
        key = str(cat)
        removed = group[group["removed"] & (group["days_publish_to_removal"] <= window)]
        if len(removed) == 0:
            out[key] = 0.0
            continue
        feats, mask, has = _features(removed)
        init = _WARM_START_CACHE.get(key) if use_cache else None
        resp, params = _em_fit(feats, mask, has, removed["marked_sold"].to_numpy(bool), init)
        if use_cache:
            _WARM_START_CACHE[key] = params
        expected_sold = float(resp.sum())
        sold_total += expected_sold
        out[key] = expected_sold / len(group)
    out["platform"] = sold_total / len(cohort)
    return out


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------
def bootstrap_ci(
    df: pd.DataFrame,
    estimator,
    n_iters: int | None = None,
    alpha: float | None = None,
    seed: int = config.SEED,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CI, resampling listings within category.

    Resampling inside a category keeps each category's size and composition fixed, so
    the interval reflects listing-level noise, not category mix.
    """
    n_iters = config.BOOTSTRAP_ITERS if n_iters is None else int(n_iters)
    alpha = config.BOOTSTRAP_ALPHA if alpha is None else float(alpha)
    cohort = df[df["in_cohort"]] if "in_cohort" in df else df

    rng = np.random.default_rng(seed)
    frames = {c: cohort[cohort["category_id"] == c] for c in cohort["category_id"].unique()}

    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_iters):
        pieces = [
            frames[c].iloc[rng.integers(0, len(frames[c]), len(frames[c]))] for c in frames
        ]
        sample = pd.concat(pieces, ignore_index=True)
        est = estimator(sample)
        for key, val in est.items():
            draws[key].append(val)

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        k: (float(np.percentile(v, lo_q)), float(np.percentile(v, hi_q)))
        for k, v in draws.items()
    }


# ---------------------------------------------------------------------------
# comparison table
# ---------------------------------------------------------------------------
def compare(df: pd.DataFrame, n_iters: int = 300) -> pd.DataFrame:
    """One row per category plus a platform row: proxy, correction, both intervals.

    The naive estimate has always shipped an interval and the corrected one did not,
    which made the *worse* estimator look like the more measured one. Both are resampled
    the same way now (same listings, same draws) so the two columns are comparable.
    """
    naive = naive_sell_through(df)
    corrected = corrected_sell_through(df)
    ci = bootstrap_ci(df, naive_sell_through, n_iters=n_iters)
    corrected_ci = bootstrap_ci(df, corrected_sell_through, n_iters=n_iters)

    cohort = df[df["in_cohort"]] if "in_cohort" in df else df
    sizes = cohort.groupby("category_id").size().to_dict()
    rows = [
        {
            "category": cat,
            "cohort_size": int(sizes[cat]),
            "naive": naive[cat],
            "corrected": corrected[cat],
            "naive_ci_low": ci[cat][0],
            "naive_ci_high": ci[cat][1],
            "corrected_ci_low": corrected_ci[cat][0],
            "corrected_ci_high": corrected_ci[cat][1],
        }
        for cat in sorted(sizes)
    ]
    rows.append(
        {
            "category": "platform",
            "cohort_size": int(len(cohort)),
            "naive": naive["platform"],
            "corrected": corrected["platform"],
            "naive_ci_low": ci["platform"][0],
            "naive_ci_high": ci["platform"][1],
            "corrected_ci_low": corrected_ci["platform"][0],
            "corrected_ci_high": corrected_ci["platform"][1],
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    """Print the per-category audit table."""
    df = load_cohort()
    table = compare(df)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
