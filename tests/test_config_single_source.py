"""One source of truth for thresholds, windows and bootstrap settings.

The failure this file guards against is silent: `metrics` used to keep its own
``COHORT_WINDOW_DAYS = 30.0`` while ``config.COHORT_DAYS`` also said 30, and
``monetization`` ran its own bootstrap at 1 000 draws / seed 7 while config said
2 000 / 42. Nothing crashed; the two halves of the report just stopped describing the
same dataset the moment anyone edited one side. These tests notice.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from liquidity import bias, config, metrics, monetization

LIQUIDITY_DIR = Path(__file__).resolve().parent.parent / "liquidity"
REPO_ROOT = LIQUIDITY_DIR.parent


def _module_sources() -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in LIQUIDITY_DIR.glob("*.py")
        if path.name != "__init__.py"
    }


def test_no_module_redefines_a_config_constant() -> None:
    """A name assigned at module level outside config.py is a second source of truth."""
    offenders: list[str] = []
    for name, source in _module_sources().items():
        if name == "config.py":
            continue
        pattern = r"^([A-Z][A-Z0-9_]{3,})\s*(?::[^=]+)?=\s*(?P<rhs>.+)$"
        for match in re.finditer(pattern, source, re.MULTILINE):
            constant = match.group(1)
            rhs = match.group("rhs")
            if not hasattr(config, constant):
                continue
            if rhs.startswith("config."):
                continue  # an alias, not a second source of truth
            offenders.append(f"{name}: {constant} = {rhs.strip()}")
    assert not offenders, "constant redefined outside config: " + "; ".join(offenders)


def test_every_config_constant_is_read_somewhere() -> None:
    """A key nothing reads is not a switch. This is the antidote to dormant thresholds.

    A constant is live if it is named more than once anywhere in the package or the tests:
    once in its own definition, once by a reader. ``MARTS_DIR`` counts because ``DB_PATH``
    composes it inside config; ``SHORT_WINDOW_DAYS`` used to be named once and nowhere else,
    which is exactly the case this test exists to catch.
    """
    sources = [
        path.read_text(encoding="utf-8")
        for path in list(LIQUIDITY_DIR.glob("*.py")) + list((REPO_ROOT / "tests").glob("*.py"))
    ]
    unused = [
        name
        for name in dir(config)
        if name.isupper() and sum(len(re.findall(rf"\b{name}\b", text)) for text in sources) < 2
    ]
    assert not unused, f"config constants nobody reads: {unused}"


def test_metrics_reads_the_cohort_window_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proves the dependency is live: change the config value, the cohort changes."""
    frames = metrics.load()
    default_size = int(metrics.liquidity_by_cohort(frames)["cohort_size"].iloc[-1])

    monkeypatch.setattr(config, "COHORT_DAYS", 15)
    shrunk = int(metrics.liquidity_by_cohort(frames)["cohort_size"].iloc[-1])
    assert shrunk < default_size, "metrics ignored config.COHORT_DAYS"

    monkeypatch.setattr(config, "COHORT_DAYS", 45)
    widened = int(metrics.liquidity_by_cohort(frames)["cohort_size"].iloc[-1])
    assert widened > default_size


def test_cohort_and_observation_window_are_separate_numbers() -> None:
    """A shorter window censors observation; it must not shrink the cohort."""
    frames = metrics.load()
    short = metrics.liquidity_by_window(frames)[config.SHORT_WINDOW_DAYS].iloc[-1]
    full = metrics.liquidity_by_window(frames)[config.LIQUIDITY_WINDOW_DAYS].iloc[-1]

    assert short["cohort_size"] == full["cohort_size"]
    assert short["listings_with_contact"] < full["listings_with_contact"], (
        "a 7-day window cannot reach the 30-day contact share"
    )


def test_monetization_reads_bootstrap_settings_from_config() -> None:
    assert monetization.N_BOOTSTRAP == config.SESSION_BOOTSTRAP_ITERS
    assert monetization.BOOTSTRAP_SEED == config.SESSION_BOOTSTRAP_SEED
    assert monetization.ALPHA == config.BOOTSTRAP_ALPHA
    assert monetization.N_BOOTSTRAP != config.BOOTSTRAP_ITERS, (
        "session-level and category-level draws are different procedures"
    )


def test_corrected_estimate_ships_its_own_interval() -> None:
    """The worse estimator used to be the only one with an interval."""
    table = bias.compare(bias.load_cohort(), n_iters=60)

    assert {"corrected_ci_low", "corrected_ci_high"} <= set(table.columns)
    assert (table["corrected_ci_low"] <= table["corrected"]).all()
    assert (table["corrected"] <= table["corrected_ci_high"]).all()
    assert (table["naive_ci_low"] <= table["naive"]).all()
