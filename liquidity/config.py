"""Single place for every threshold, window and path.

Nothing here may reveal a latent quantity of the data generator: planted
parameters live in `liquidity/generate.py`, and analysis modules are forbidden
from importing that file (enforced by the leak test in `tests/`).
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
STAGING_DIR = PROJECT_ROOT / "data" / "staging"
MARTS_DIR = PROJECT_ROOT / "data" / "marts"
DB_PATH = MARTS_DIR / "liquidity.duckdb"
REPORTS_DIR = PROJECT_ROOT / "reports"

SEED = 42

# --- time ---------------------------------------------------------------
DATA_START = "2026-01-05"
N_DAYS = 90
COHORT_DAYS = 30          # cohort = listings published in the first COHORT_DAYS
LIQUIDITY_WINDOW_DAYS = 30  # observation window per listing, measured from publish
SHORT_WINDOW_DAYS = 7

# --- experiment ---------------------------------------------------------
HOLDOUT_SHARE = 0.10      # share of search sessions where promoted slots are suppressed

# --- metric thresholds --------------------------------------------------
MIN_VIEWS_FOR_CONTACT_RATE = 100
BOOTSTRAP_ITERS = 2000
BOOTSTRAP_ALPHA = 0.05

# --- assertion tolerances (used by tests, not by analysis) --------------
TOL_RECOVERY = 0.03       # corrected estimate must land within this of the planted value
MIN_NAIVE_BIAS = 0.04     # naive estimate must be at least this far from truth
