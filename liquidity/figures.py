"""Figures for the observable half of the audit.

Two rules govern this module, and both are load-bearing:

* **It is truth-free.** Nothing here computes or draws the latent answer key; the
  truth-overlay figures live in the scorer, the single module allowed to open the
  key. The leak guard scans this file like any other analysis module, so a truth
  reference added here turns the suite red — the boundary is enforced, not promised.
* **Every builder returns the numbers it drew.** A figure whose values cannot be
  compared to the module it illustrates is decoration. Tests compare them, and the
  scorer's ``--figures`` flag writes the same numbers to ``reports/figures/figure_data.json``
  so a stale PNG is visible in a diff.

Run the observable figures with::

    uv run python -m liquidity.figures
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: figures are files, never windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from liquidity import config, metrics, monetization

FIGURES_DIR = config.PROJECT_ROOT / "reports" / "figures"

# Palette read off the portfolio site's design tokens (`src/styles/global.css`):
# a cream ground with deep-indigo ink and coral reserved for the treated arm. One
# palette, no second theme — a figure that only works on one background is a figure
# that will be re-made as a screenshot somewhere.
PALETTE = {
    "ground": "#f8f2d8",
    "ink": "#0a0070",
    "ink_soft": "#4a4470",
    "hairline": "#8076d7",
    "accent": "#ff8569",
    "accent_ink": "#1a1a1a",
    "grid": "#8076d7",
}

DPI = 300


def apply_style() -> None:
    """House style. Idempotent: every builder calls it, no global setup order."""
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["ground"],
            "axes.facecolor": PALETTE["ground"],
            "savefig.facecolor": PALETTE["ground"],
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "figure.dpi": 130,
            "font.family": ["DejaVu Sans", "sans-serif"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 11,
            "axes.labelcolor": PALETTE["ink"],
            "axes.edgecolor": PALETTE["ink"],
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.30,
            "grid.linewidth": 0.7,
            "xtick.color": PALETTE["ink_soft"],
            "ytick.color": PALETTE["ink_soft"],
            "xtick.labelcolor": PALETTE["ink"],
            "ytick.labelcolor": PALETTE["ink"],
            "legend.frameon": False,
            "legend.fontsize": 10,
        }
    )


def save_figure(fig: plt.Figure, name: str, out_dir: Path | None = None) -> Path:
    """Write ``name.png`` at print resolution and release the figure."""
    out_dir = FIGURES_DIR if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        metadata={"Software": None},
    )
    plt.close(fig)
    return path


def source_note(ax: plt.Axes, text: str) -> None:
    """Small provenance line under the axes, in axes coordinates."""
    ax.text(
        0.0,
        -0.16,
        text,
        transform=ax.transAxes,
        fontsize=8.5,
        color=PALETTE["ink_soft"],
        va="top",
        ha="left",
    )


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


# --------------------------------------------------------------------------- #
# 04 — where the organic-contacts drop comes from
# --------------------------------------------------------------------------- #
def boost_waterfall(
    frames: dict[str, pd.DataFrame] | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    """Decompose the organic-contacts drop into slot arithmetic and conversion.

    The two components are relative changes, so they add: fewer organic slots per
    session (-4.8%) plus lower contacts per organic impression (-6.5%) is the
    -10.9% the guardrail reports. The split is the product decision — one part is
    geometry, the other is behaviour.
    """
    frames = metrics.load() if frames is None else frames
    displacement = monetization.guardrails(frames)["displacement"]

    slot = displacement["organic_impressions_per_session"]["relative_change"]
    conversion = displacement["contacts_per_organic_impression"]["relative_change"]
    total = displacement["organic_contacts_per_session"]["relative_change"]
    slot_share = displacement["slot_arithmetic_share_of_drop"]

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5.6), layout="constrained")

    steps = [
        ("Fewer organic slots\nper session", slot * 100, PALETTE["ink_soft"]),
        ("Contacts per organic\nimpression", conversion * 100, PALETTE["accent"]),
        ("Organic contacts\nper session", total * 100, PALETTE["ink"]),
    ]
    width = 0.55
    running = 0.0

    def _flow_bar(index: int, low: float, high: float, color: str, label: str) -> None:
        """One bar spanning ``low..high`` with its value printed past the far end."""
        ax.bar(
            index,
            abs(high - low),
            bottom=min(low, high),
            width=width,
            color=color,
            edgecolor=PALETTE["ink"],
            linewidth=0.8,
            zorder=2,
        )
        at, dy = (min(low, high), -20) if high <= 0 else (max(low, high), 12)
        ax.annotate(
            label,
            (index, at),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            fontsize=12,
            fontweight="bold",
            color=PALETTE["ink"],
        )

    for index, (_label, value, color) in enumerate(steps[:-1]):
        _flow_bar(index, running, running + value, color, f"{value:+.1f}%")
        ax.plot(
            [index + width / 2, index + 1 - width / 2],
            [running + value, running + value],
            color=PALETTE["ink_soft"],
            linewidth=1.0,
            linestyle=(0, (3, 3)),
            zorder=1,
        )
        running += value

    total_pct = total * 100
    ax.bar(
        len(steps) - 1,
        abs(total_pct),
        bottom=min(0.0, total_pct),
        width=width,
        color=PALETTE["ink"],
        edgecolor=PALETTE["ink"],
        linewidth=0.8,
        zorder=2,
    )
    ax.annotate(
        f"{total_pct:+.1f}%",
        (len(steps) - 1, total_pct),
        xytext=(0, -20),
        textcoords="offset points",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["ink"],
    )

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([label for label, _, _ in steps])
    ax.set_ylabel("Change vs randomised holdout, % of holdout level")
    ax.set_title("Paid slots cut organic contacts 10.9% — 42% of that is one displaced slot")
    ax.set_ylim(min(0.0, total * 100) - 2.5, 1.0)
    ax.axhline(0, color=PALETTE["ink"], linewidth=1.2, zorder=3)
    ax.grid(axis="x", visible=False)
    source_note(
        ax,
        f"treated {displacement['organic_contacts_per_session']['treated']:.4f} vs "
        f"holdout {displacement['organic_contacts_per_session']['holdout']:.4f} organic contacts "
        f"per session · liquidity.monetization.guardrails · 95% CI "
        f"[{displacement['organic_contacts_per_session']['ci_95_low'] * 100:+.1f}%, "
        f"{displacement['organic_contacts_per_session']['ci_95_high'] * 100:+.1f}%]",
    )
    save_figure(fig, "05_boost_waterfall", out_dir)

    return {
        "slot_arithmetic_pct": _round(slot * 100),
        "conversion_pct": _round(conversion * 100),
        "total_pct": _round(total * 100),
        "slot_share_of_drop": _round(slot_share),
        "treated": _round(displacement["organic_contacts_per_session"]["treated"]),
        "holdout": _round(displacement["organic_contacts_per_session"]["holdout"]),
    }


# --------------------------------------------------------------------------- #
# 05 — the effect next to the design's own resolution
# --------------------------------------------------------------------------- #
def incrementality_vs_mde(
    frames: dict[str, pd.DataFrame] | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    """Forest plot of the causal contrast, scaled against the MDE.

    A point estimate with a p-value invites the wrong reading when the design could
    not have detected the effect either way. Drawing the MDE as a band makes the
    sentence "negative but not proven, and underpowered for its own effect" visible.
    """
    frames = metrics.load() if frames is None else frames
    incrementality = monetization.holdout_incrementality(frames)

    point = incrementality["incrementality"]
    low = incrementality["ci_95_low"]
    high = incrementality["ci_95_high"]
    mde = incrementality["mde_80"]

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")

    ax.axvspan(-mde, mde, color=PALETTE["hairline"], alpha=0.18, linewidth=0, zorder=1)
    for edge in (-mde, mde):
        ax.axvline(edge, color=PALETTE["hairline"], linewidth=1.0, zorder=1)
    ax.axvline(0, color=PALETTE["ink"], linewidth=1.4, zorder=2)

    ax.hlines(0, low, high, color=PALETTE["ink"], linewidth=2.0, zorder=3)
    ax.plot(
        [low, high],
        [0, 0],
        linestyle="none",
        marker="|",
        markersize=13,
        color=PALETTE["ink"],
        zorder=3,
    )
    ax.plot(
        point,
        0,
        marker="o",
        markersize=12,
        color=PALETTE["accent"],
        markeredgecolor=PALETTE["ink"],
        markeredgewidth=1.2,
        zorder=4,
    )

    ax.annotate(
        f"estimate {point:+.4f}\n95% CI [{low:+.4f}, {high:+.4f}]\nWelch p = "
        f"{incrementality['welch_p_value']:.3f}",
        (point, 0),
        xytext=(0, 26),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        color=PALETTE["ink"],
    )
    ax.annotate(
        f"MDE at 80% power  ±{mde:.4f}\n({incrementality['mde_relative_to_holdout'] * 100:.1f}% "
        f"of the holdout rate)",
        (mde, 0),
        xytext=(-8, -26),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=9.5,
        color=PALETTE["ink_soft"],
    )

    ax.set_yticks([])
    ax.set_ylim(-0.55, 0.62)
    ax.set_xlim(min(low, -mde) * 1.35, max(high, mde) * 1.35)
    ax.set_xlabel("Contacts per search session, treated minus randomised holdout")
    ax.set_title(
        f"The effect is {abs(point) / mde * 100:.0f}% of the smallest effect "
        "this design could detect"
    )
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    source_note(
        ax,
        f"{incrementality['sessions_treated']:,} treated vs "
        f"{incrementality['sessions_holdout']:,} holdout sessions · "
        "liquidity.monetization.holdout_incrementality",
    )
    save_figure(fig, "04_incrementality_vs_mde", out_dir)

    return {
        "incrementality": _round(point),
        "ci_95_low": _round(low),
        "ci_95_high": _round(high),
        "mde_80": _round(mde),
        "relative_change": _round(incrementality["relative_change"]),
        "welch_p_value": _round(incrementality["welch_p_value"]),
        "effect_to_mde": _round(abs(point) / mde),
    }


# --------------------------------------------------------------------------- #
# 06 — where on the SERP the displacement lands
# --------------------------------------------------------------------------- #
def position_mix(
    frames: dict[str, pd.DataFrame] | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    """Organic impressions per session by position bucket, treated vs holdout.

    The aggregate says organic reach fell. This says where: the top of the SERP
    loses a fifth of its organic impressions while positions 4-20 barely move, which
    is what makes a per-SERP slot cap rather than a global rollback the lever.
    """
    frames = metrics.load() if frames is None else frames
    rows = monetization.guardrails(frames)["displacement"]["by_position"]
    frame = pd.DataFrame(rows)

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5.2), layout="constrained")

    y = np.arange(len(frame))
    height = 0.34
    ax.barh(
        y - height / 2,
        frame["holdout"],
        height=height,
        color=PALETTE["ink"],
        label="Randomised holdout (10%)",
        zorder=2,
    )
    ax.barh(
        y + height / 2,
        frame["treated"],
        height=height,
        color=PALETTE["accent"],
        edgecolor=PALETTE["ink"],
        linewidth=0.7,
        label="Promoted slots shown (treated)",
        zorder=2,
    )
    for index, row in enumerate(frame.itertuples()):
        ax.annotate(
            f"{row.relative_change * 100:+.1f}%",
            (max(row.treated, row.holdout), index),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=PALETTE["ink"],
        )

    ax.set_yticks(y)
    ax.set_yticklabels([f"positions {bucket}" for bucket in frame["position_bucket"]])
    ax.invert_yaxis()
    ax.set_xlabel("Organic impressions per search session")
    ax.set_xlim(0, frame[["treated", "holdout"]].to_numpy().max() * 1.22)
    ax.set_title(
        "Promoted slots eat the top of the SERP: positions 1–3 lose 21.6% of organic impressions",
        fontsize=13,
    )
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right")
    source_note(
        ax,
        "same sessions, same listings — only the slot mix differs · "
        "liquidity.monetization.guardrails()['displacement']['by_position']",
    )
    save_figure(fig, "06_position_mix", out_dir)

    return {
        f"position_{row.position_bucket}_relative_change": _round(row.relative_change)
        for row in frame.itertuples()
    }


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def observable_payload(
    frames: dict[str, pd.DataFrame] | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    """Build the three observable figures and return every number they drew."""
    frames = metrics.load() if frames is None else frames
    return {
        "05_boost_waterfall": boost_waterfall(frames, out_dir),
        "04_incrementality_vs_mde": incrementality_vs_mde(frames, out_dir),
        "06_position_mix": position_mix(frames, out_dir),
    }


def main() -> None:
    payload = observable_payload()
    print(f"figures written to {FIGURES_DIR}")
    for name in sorted(payload):
        print(f"  {name}.png")
    for name, values in payload.items():
        print(f"\n{name}")
        for key, value in values.items():
            print(f"  {key:48s} {value}")


if __name__ == "__main__":
    main()
