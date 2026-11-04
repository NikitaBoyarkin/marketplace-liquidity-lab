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
SQL_DIR = PROJECT_ROOT / "sql"

SEED = 42

# --- time ---------------------------------------------------------------
DATA_START = "2026-01-05"
N_DAYS = 90
COHORT_DAYS = 30          # cohort = listings published in the first COHORT_DAYS
LIQUIDITY_WINDOW_DAYS = 30  # observation window per listing, measured from publish
SHORT_WINDOW_DAYS = 7

# --- experiment ---------------------------------------------------------
HOLDOUT_SHARE = 0.10      # share of search sessions where promoted slots are suppressed
HOLDOUT_SHARE_TOLERANCE = 0.01  # observed holdout share must sit within ±1 pp of the plan

# --- bootstrap ----------------------------------------------------------
# Two bootstrap procedures with different units and different costs. Both are named
# here so a module cannot quietly invent its own iteration count (the way
# `monetization` used to: 1 000 draws at seed 7 while this file said 2 000 at seed 42).
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_ITERS = 2000          # category-level: draws one category's listings
BOOTSTRAP_SEED = SEED
SESSION_BOOTSTRAP_ITERS = 1000  # session-level: draws 38k sessions, ~50x the work per draw
SESSION_BOOTSTRAP_SEED = 7

# --- demand-sufficiency quadrant ----------------------------------------
# Recorded in docs/preregistration.md as a dated addendum: they were declared in code,
# not before the labels were computed, and the report says so.
DEMAND_FLOOR_SEARCHES_PER_DAY_CELL = 8.0
DEMAND_FLOOR_SEARCHES_PER_DAY_CATEGORY = 60.0
HIGH_ZERO_RESULT_SHARE = 0.05
LOW_VIEW_RATE = 0.10
LOW_CONTACT_RATE = 0.10

# --- data-quality gates -------------------------------------------------
CONTACT_REPEAT_MAX = 3      # repeats of one (buyer, listing) pair before a gate fails
MAX_DUPLICATE_SHARE = 0.20  # per-category share of listings sharing one object_id

# --- metric thresholds --------------------------------------------------
# MIN_VIEWS_FOR_CONTACT_RATE lived here and was read by nothing; a minimum-sample gate
# needs its own pre-declaration, not a dormant constant. Removed rather than wired up.

# --- assertion tolerances (used by tests, not by analysis) --------------
TOL_RECOVERY = 0.03       # corrected estimate must land within this of the planted value
MIN_NAIVE_BIAS = 0.04     # naive estimate must be at least this far from truth
