"""Score the estimators against the planted answer key.

This is the **only** module in the package allowed to open
``_truth/ground_truth.json``. Every analysis module stays truth-free; the leak
test enforces that the reader set is exactly ``{"scoring.py"}``, so widening it
is a visible, reviewed edit rather than a quiet import.

The point of separating the scorer from the analysis is that the audit table in
``README.md`` becomes reproducible with one command:

    uv run python -m liquidity.scoring

Without this module the table's ``true`` and ``error`` columns could only be
rebuilt by opening the answer key by hand, which is exactly what the analysis is
forbidden from doing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from liquidity import bias, config
from liquidity.bias import compare, load_cohort
from liquidity.figures import (
    FIGURES_DIR,
    PALETTE,
    apply_style,
    observable_payload,
    save_figure,
    source_note,
)

PROJECT_ROOT = config.PROJECT_ROOT
TRUTH_PATH = PROJECT_ROOT / "_truth" / "ground_truth.json"

COLUMNS = (
    "category",
    "cohort_size",
    "true",
    "naive",
    "naive_error",
    "corrected",
    "corrected_error",
    "naive_ci_low",
    "naive_ci_high",
    "corrected_ci_low",
    "corrected_ci_high",
)


def load_truth(truth_path: Path | str = TRUTH_PATH) -> dict[str, Any]:
    """Read the frozen answer key. Only this module does, by design."""
    return json.loads(Path(truth_path).read_text(encoding="utf-8"))


def truth_by_category(truth: dict[str, Any] | None = None) -> dict[str, float]:
    """Latent sell-through per category plus the platform aggregate."""
    truth = load_truth() if truth is None else truth
    latent = truth["latent"]
    values = dict(latent["sell_through_true_by_category"])
    values["platform"] = float(latent["platform_true"])
    return values


def scored_table(
    df: pd.DataFrame | None = None,
    truth: dict[str, Any] | None = None,
    n_iters: int = 300,
) -> pd.DataFrame:
    """Estimates next to the truth they are scored against.

    ``naive`` and ``corrected`` come from ``bias.compare`` (observable data only);
    ``true`` comes from the answer key. Errors are computed from unrounded
    values, so hand-subtracting the printed columns can differ in the last digit.
    """
    df = load_cohort() if df is None else df
    truth = load_truth() if truth is None else truth

    estimates = compare(df, n_iters=n_iters)
    true = truth_by_category(truth)

    table = estimates.copy()
    table["true"] = table["category"].map(true)
    table["naive_error"] = table["naive"] - table["true"]
    table["corrected_error"] = table["corrected"] - table["true"]
    return table[list(COLUMNS)]


# ---------------------------------------------------------------------------
# figures — the part of the story only the key can tell
# ---------------------------------------------------------------------------
# These live here, not in `figures.py`, for the same reason `scored_table` does: drawing
# the truth requires reading the key, and the leak test keeps that reader set at exactly
# this module. A figure module that quietly opened the key would make the audit's central
# claim ("the analysis never saw the answer") false in a way no reader could see.
SWEEP_CUTOFFS = (1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0)

CATEGORY_COLORS = ("#0a0070", "#4a4470", "#8076d7", "#ff8569", "#a63d21")


def separator_sweep(
    df: pd.DataFrame | None = None, cutoffs: tuple[float, ...] = SWEEP_CUTOFFS
) -> pd.DataFrame:
    """Per-category signed error as the reply-to-removal cutoff moves.

    The cutoff is the estimator's one free parameter: a sale is a replied contact
    followed by a removal within ``SOLD_LAG_MAX`` days. Sweeping it shows where the
    correction works and where it stops working — which is the difference between
    "the method recovers truth" and "the method recovers truth when the separator is
    right". The module constant is restored in a ``finally`` block so a sweep can
    never leak into a later estimator call.
    """
    df = load_cohort() if df is None else df
    truth = truth_by_category()
    original = bias.SOLD_LAG_MAX
    rows: list[dict[str, float]] = []
    try:
        for cutoff in cutoffs:
            bias.SOLD_LAG_MAX = float(cutoff)
            estimate = bias.corrected_sell_through(df, use_cache=False)
            row = {"cutoff_days": float(cutoff)}
            row.update({name: estimate[name] - value for name, value in truth.items()})
            rows.append(row)
    finally:
        bias.SOLD_LAG_MAX = original
    return pd.DataFrame(rows)


def plot_sell_through_audit(table: pd.DataFrame, out_dir: Path | None = None) -> dict[str, Any]:
    """Naive vs corrected sell-through per category, with the latent truth marked.

    Two estimates and one key on the same row: the gap between the platforms's
    self-report and the truth is the finding, and the gap between the corrected read
    and the truth is the method's residual.
    """
    apply_style()
    frame = table.set_index("category")
    order = [name for name in frame.index if name != "platform"] + ["platform"]
    positions = np.arange(len(order))[::-1].astype(float)

    fig, ax = plt.subplots(figsize=(10, 6.0), layout="constrained")
    for position, name in zip(positions, order, strict=True):
        row = frame.loc[name]
        ax.plot(
            [row["naive_ci_low"], row["naive_ci_high"]],
            [position + 0.15] * 2,
            color=PALETTE["ink_soft"],
            linewidth=1.4,
            zorder=2,
        )
        ax.plot(
            [row["corrected_ci_low"], row["corrected_ci_high"]],
            [position - 0.15] * 2,
            color=PALETTE["ink"],
            linewidth=1.4,
            zorder=2,
        )
        ax.plot(
            row["naive"],
            position + 0.15,
            marker="o",
            markersize=8,
            markerfacecolor=PALETTE["ground"],
            markeredgecolor=PALETTE["ink_soft"],
            markeredgewidth=1.8,
            zorder=3,
        )
        ax.plot(
            row["corrected"],
            position - 0.15,
            marker="o",
            markersize=8,
            color=PALETTE["ink"],
            zorder=3,
        )
        ax.plot(
            [row["true"], row["true"]],
            [position - 0.34, position + 0.34],
            color=PALETTE["accent"],
            linewidth=3.4,
            solid_capstyle="butt",
            zorder=4,
        )

    separator = positions[order.index("platform")] + 0.55
    ax.axhline(separator, color=PALETTE["hairline"], linewidth=1.0, linestyle=(0, (2, 4)))

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [
            f"{name}"
            if name == "platform"
            else f"{name}  (n={int(frame.loc[name, 'cohort_size']):,})"
            for name in order
        ]
    )
    ax.set_xlabel("Sell-through in the 30-day window (share of the publish cohort)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}%"))
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(order) - 0.3)

    platform_error = frame.loc["platform", "naive_error"] * 100
    worst_error = frame["naive_error"].abs().max() * 100
    ax.set_title(
        f"The platform's own number is {abs(platform_error):.1f} pp low — "
        f"{worst_error:.1f} pp in the worst category"
    )

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="-",
            color=PALETTE["ink_soft"],
            markerfacecolor=PALETTE["ground"],
            markeredgewidth=1.8,
            label="Platform proxy (removal reason = sold)",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="-",
            color=PALETTE["ink"],
            label="EM correction (observable signals only)",
        ),
        plt.Line2D(
            [],
            [],
            marker="|",
            linestyle="none",
            color=PALETTE["accent"],
            markersize=16,
            markeredgewidth=3.4,
            label="Latent truth (answer key)",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=3)
    source_note(
        ax,
        "bars: 95% percentile bootstrap, 300 draws · liquidity.scoring.scored_table() · "
        "truth read only by this module",
    )
    save_figure(fig, "01_sell_through_audit", out_dir)

    return {
        name: {
            "naive": round(float(frame.loc[name, "naive"]), 6),
            "corrected": round(float(frame.loc[name, "corrected"]), 6),
            "true": round(float(frame.loc[name, "true"]), 6),
            "naive_error": round(float(frame.loc[name, "naive_error"]), 6),
            "corrected_error": round(float(frame.loc[name, "corrected_error"]), 6),
        }
        for name in order
    }


def plot_error_vs_tolerance(table: pd.DataFrame, out_dir: Path | None = None) -> dict[str, Any]:
    """Signed error of both estimators against the ±3 pp recovery tolerance."""
    apply_style()
    order = [name for name in table["category"] if name != "platform"] + ["platform"]
    frame = table.set_index("category").loc[order]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
    ax.axhspan(
        -config.TOL_RECOVERY * 100,
        config.TOL_RECOVERY * 100,
        color=PALETTE["hairline"],
        alpha=0.16,
        linewidth=0,
        zorder=1,
    )
    for edge in (-config.TOL_RECOVERY * 100, config.TOL_RECOVERY * 100):
        ax.axhline(edge, color=PALETTE["hairline"], linewidth=1.0, zorder=2)

    width = 0.36
    naive = frame["naive_error"].to_numpy() * 100
    corrected = frame["corrected_error"].to_numpy() * 100
    ax.bar(
        x - width / 2,
        naive,
        width,
        color=PALETTE["ground"],
        edgecolor=PALETTE["ink_soft"],
        linewidth=1.6,
        hatch="///",
        label="Platform proxy",
        zorder=3,
    )
    ax.bar(x + width / 2, corrected, width, color=PALETTE["ink"], label="EM correction", zorder=3)

    for index, (bad, good) in enumerate(zip(naive, corrected, strict=True)):
        ax.annotate(
            f"{bad:+.1f}",
            (index - width / 2, bad),
            xytext=(0, -6 if bad < 0 else 4),
            textcoords="offset points",
            ha="center",
            va="top" if bad < 0 else "bottom",
            fontsize=9.5,
            color=PALETTE["ink_soft"],
        )
        ax.annotate(
            f"{good:+.1f}",
            (index + width / 2, good),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=PALETTE["ink"],
        )

    ax.axhline(0, color=PALETTE["ink"], linewidth=1.2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(min(naive.min(), corrected.min()) - 1.8, max(naive.max(), corrected.max()) + 1.4)
    ax.set_ylabel("Estimate − latent truth, percentage points")
    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", visible=False)

    worst_naive = float(np.abs(naive).max())
    worst_corrected = float(np.abs(corrected).max())
    ax.set_title(
        f"The proxy misses by up to {worst_naive:.1f} pp; the correction stays inside "
        f"±{config.TOL_RECOVERY * 100:.0f} pp everywhere",
        fontsize=13,
    )
    ax.annotate(
        f"shaded band: ±{config.TOL_RECOVERY * 100:.0f} pp recovery tolerance",
        (0.62, config.TOL_RECOVERY * 100 * 0.82),
        ha="left",
        va="bottom",
        fontsize=9,
        color=PALETTE["ink_soft"],
    )
    fig.legend(loc="outside lower center", ncols=2)
    source_note(
        ax,
        "positive = over-reporting, negative = under-reporting · "
        "liquidity.config.TOL_RECOVERY · liquidity.scoring.scored_table()",
    )
    save_figure(fig, "02_error_vs_tolerance", out_dir)

    return {
        "tolerance_pp": round(config.TOL_RECOVERY * 100, 6),
        "worst_naive_error_pp": round(worst_naive, 6),
        "worst_corrected_error_pp": round(worst_corrected, 6),
        "corrected_errors_pp": {
            name: round(float(value), 6) for name, value in zip(order, corrected, strict=True)
        },
    }


def plot_separator_sensitivity(
    out_dir: Path | None = None, df: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Absolute error by cutoff: the estimator's one free parameter, swept.

    The band is derived from the sweep rather than declared: the recovery band is the
    contiguous run of cutoffs around the best one where no category misses the
    tolerance. That matters because the band is wider than the 4-6 days the suite
    asserts — 7 and 8 days also recover — and the figure should not claim otherwise.
    """
    sweep = separator_sweep(df)
    categories = [column for column in sweep.columns if column not in ("cutoff_days", "platform")]
    tolerance = config.TOL_RECOVERY * 100

    error = sweep[categories].abs() * 100
    worst = error.max(axis=1)
    failing = (error > tolerance).sum(axis=1)
    cutoffs = sweep["cutoff_days"].to_numpy()
    best_index = int(np.argmin(worst.to_numpy()))

    low_index = best_index
    while low_index > 0 and failing.iloc[low_index - 1] == 0:
        low_index -= 1
    high_index = best_index
    while high_index < len(cutoffs) - 1 and failing.iloc[high_index + 1] == 0:
        high_index += 1
    band_low, band_high = float(cutoffs[low_index]), float(cutoffs[high_index])

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5.6), layout="constrained")
    ax.axvspan(band_low, band_high, color=PALETTE["hairline"], alpha=0.16, linewidth=0, zorder=1)

    for color, name in zip(CATEGORY_COLORS, categories, strict=True):
        ax.plot(
            cutoffs,
            error[name],
            color=color,
            linewidth=1.4,
            alpha=0.85,
            marker="o",
            markersize=3.5,
            label=name,
            zorder=2,
        )
    ax.plot(
        cutoffs,
        worst,
        color=PALETTE["ink"],
        linewidth=2.6,
        label="worst of the five",
        zorder=3,
    )
    ax.axhline(tolerance, color=PALETTE["accent"], linewidth=1.6, linestyle=(0, (4, 3)), zorder=4)
    ax.annotate(
        f"±{tolerance:.0f} pp tolerance",
        (4.6, tolerance),
        xytext=(0, 6),
        textcoords="offset points",
        ha="left",
        fontsize=9.5,
        color=PALETTE["ink"],
    )
    ax.axvline(4.5, color=PALETTE["ink_soft"], linewidth=1.1, linestyle=":", zorder=3)
    ax.annotate(
        "the generator's own\nsale rule (4.5 d)",
        (4.5, ax.get_ylim()[1] * 0.70),
        xytext=(-6, 0),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=PALETTE["ink_soft"],
    )
    ax.annotate(
        f"recovery band\n{band_low:.0f}–{band_high:.0f} days",
        ((band_low + band_high) / 2, ax.get_ylim()[1] * 0.90),
        ha="center",
        fontsize=9,
        color=PALETTE["ink_soft"],
    )

    ax.set_xlabel("Reply-to-removal cutoff (days): the separator that defines a sale")
    ax.set_ylabel("|corrected − latent truth|, percentage points")
    ax.set_title(
        f"The correction only works inside a {band_low:.0f}–{band_high:.0f} day separator band",
        fontsize=13,
    )
    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncols=2, fontsize=9)
    source_note(
        ax,
        "each point refits the EM mixture with liquidity.bias.SOLD_LAG_MAX overridden · "
        "the constant is restored after the sweep · liquidity.scoring.separator_sweep()",
    )
    save_figure(fig, "03_separator_sensitivity", out_dir)

    return {
        "tolerance_pp": round(tolerance, 6),
        "categories": len(categories),
        "best_cutoff_days": round(float(cutoffs[best_index]), 6),
        "best_worst_error_pp": round(float(worst.iloc[best_index]), 6),
        "recovery_band_days": [band_low, band_high],
        "recovery_band_worst_error_pp": round(float(worst[low_index : high_index + 1].max()), 6),
        "failing_categories_by_cutoff": {
            f"{cutoff:g}": int(count) for cutoff, count in zip(cutoffs, failing, strict=True)
        },
        "declared_band_cutoffs_ok": [4.0, 6.0],
    }


def build_figures(out_dir: Path | str | None = None) -> dict[str, Any]:
    """Render every figure and write the payload each PNG drew.

    The sidecar is the freshness mechanism: PNG bytes depend on the platform's font
    rasteriser, so a byte diff is not a portable gate, but every plotted number is
    portable. ``tests/test_figures.py`` rebuilds this payload and fails when the
    committed one drifts from the code — that is what makes a stale PNG visible.
    """
    target = Path(FIGURES_DIR if out_dir is None else out_dir)
    table = scored_table()
    payload: dict[str, Any] = {
        "01_sell_through_audit": plot_sell_through_audit(table, target),
        "02_error_vs_tolerance": plot_error_vs_tolerance(table, target),
        "03_separator_sensitivity": plot_separator_sensitivity(target),
    }
    payload.update(observable_payload(out_dir=target))

    document = {
        "generated_by": "uv run python -m liquidity.scoring --figures",
        "note": "every number each PNG draws; a PNG is a rendering of these values",
        "figures": payload,
    }
    (target / "figure_data.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    """Print the audit table, or render the figures with ``--figures``."""
    parser = argparse.ArgumentParser(description="Score the estimators against the planted key.")
    parser.add_argument(
        "--figures",
        nargs="?",
        const=str(FIGURES_DIR),
        default=None,
        metavar="DIR",
        help="render every figure (truth overlay + observable) and write figure_data.json",
    )
    args = parser.parse_args()

    if args.figures:
        payload = build_figures(args.figures)
        print(f"figures written to {args.figures}")
        for name in sorted(payload):
            print(f"  {name}.png")
        print(f"  {'figure_data.json'}")
        return

    table = scored_table()
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
