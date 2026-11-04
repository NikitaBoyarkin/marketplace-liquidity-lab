"""The analytical report must quote the code, and the README must reach the report.

`reports/report.md` is the evidence-classed deliverable: every number in it is supposed to
be one a reader can re-derive. Two failure modes are covered here — the report drifting from
the code, and the report becoming unreachable from the front page (it spent its whole life
unlinked from the README).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from liquidity import config, metrics

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "reports" / "report.md"
README = ROOT / "README.md"
PREREG = ROOT / "docs" / "preregistration.md"


@pytest.fixture(scope="module")
def report_text() -> str:
    return REPORT.read_text(encoding="utf-8").replace("\u00a0", " ")


@pytest.fixture(scope="module")
def frames() -> dict:
    return metrics.load()


def test_report_carries_the_liquidity_block(report_text: str) -> None:
    """The repository is called a liquidity lab; the report must show liquidity."""
    lowered = report_text.lower()
    for needle in (
        "listings with \u22651 contact",
        "dead listings",
        "hours to first contact",
        "zero-result share",
        "quadrant",
    ):
        assert needle in lowered, f"report never mentions {needle!r}"


def test_report_platform_liquidity_row_matches_the_code(report_text: str, frames: dict) -> None:
    table = metrics.liquidity_by_cohort(frames)
    row = table[(table["category_id"] == "platform") & (table["region_id"] == "all")].iloc[0]

    expected = [
        f"{row['listings_with_contact']:.4f}",
        f"{row['dead_listing_share']:.4f}",
        f"{row['median_time_to_first_contact_hours']:.1f}",
        f"{row['p90_time_to_first_contact_hours']:.1f}",
        f"{row['duplicate_share']:.4f}",
    ]
    missing = [value for value in expected if value not in report_text]
    assert not missing, f"report platform row is stale against metrics: {missing}"


def test_report_window_comparison_matches_the_code(report_text: str, frames: dict) -> None:
    windows = metrics.liquidity_by_window(frames)
    for window, frame in windows.items():
        row = frame[(frame["category_id"] == "platform") & (frame["region_id"] == "all")].iloc[0]
        for column in ("listings_with_contact", "dead_listing_share"):
            value = f"{row[column]:.4f}"
            assert value in report_text, f"W={window} {column}={value} is missing from the report"


def test_report_zero_result_shares_match_the_code(report_text: str, frames: dict) -> None:
    zero = metrics.zero_result_rate(frames)
    overall = zero[(zero["population_tier"] == "all") & (zero["region_id"] == "all")].iloc[0]
    assert f"{overall['zero_result_share']:.2%}" in report_text


def test_report_states_the_quadrant_is_degenerate(report_text: str, frames: dict) -> None:
    """A four-state metric with two unreachable states is a finding, not a footnote."""
    quadrants = metrics.demand_sufficiency(frames)["quadrant"].value_counts().to_dict()
    assert quadrants.get("no_views", 0) == 0
    assert quadrants.get("no_contact", 0) == 0
    assert "unreachable" in report_text
    assert "38" in report_text and "no_demand" in report_text


def test_quadrant_thresholds_are_recorded_in_the_preregistration() -> None:
    """They live in config; the preregistration addendum makes them auditable."""
    prereg = PREREG.read_text(encoding="utf-8")
    for name in (
        "DEMAND_FLOOR_SEARCHES_PER_DAY_CELL",
        "DEMAND_FLOOR_SEARCHES_PER_DAY_CATEGORY",
        "HIGH_ZERO_RESULT_SHARE",
        "LOW_VIEW_RATE",
        "LOW_CONTACT_RATE",
    ):
        assert name in prereg, f"{name} is not recorded in the preregistration addendum"
        assert getattr(config, name) is not None

    plain = prereg.replace("*", "").lower()
    assert "addendum" in plain
    assert "post-hoc" in plain or "not part of the original document" in plain


def test_readme_links_to_the_report() -> None:
    """The 23 KB deliverable must be one click from the front page."""
    readme = README.read_text(encoding="utf-8")
    assert "reports/report.md" in readme
