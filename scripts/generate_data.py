"""Entry point: regenerate the synthetic dataset.

Deterministic — same seed produces byte-identical parquet files.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liquidity.generate import main  # noqa: E402

if __name__ == "__main__":
    main()
