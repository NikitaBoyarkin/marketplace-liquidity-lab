"""Fail once, clearly, when the generated log is missing.

`data/raw/` is gitignored, so a fresh clone has no parquet. Without this hook every test
module fails separately with `FileNotFoundError` — on a clean clone that was 61 red errors
and no hint about the fix. One actionable error is better than a wall of them.
"""

from __future__ import annotations

import pytest

from liquidity import config as liquidity_config

REQUIRED = ("listings", "searches", "impressions", "views", "contacts")


def pytest_configure(config: pytest.Config) -> None:
    raw_dir = liquidity_config.RAW_DIR
    missing = [name for name in REQUIRED if not (raw_dir / f"{name}.parquet").exists()]
    if missing:
        raise pytest.UsageError(
            "raw data is missing: "
            + ", ".join(f"data/raw/{name}.parquet" for name in missing)
            + "\nRun `make data` (or `make all`) first — it regenerates the log from seed "
            + f"{liquidity_config.SEED} in a few seconds. `data/raw/` is gitignored on purpose."
        )
