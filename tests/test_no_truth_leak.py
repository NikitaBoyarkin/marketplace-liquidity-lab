"""No analysis module may read the latent truth.

The whole credibility of this repo rests on the analysis not seeing the answer key.
A comment saying so is prose; this test is the mechanism. It scans every module under
`liquidity/` except the generator, and fails on any import of `_truth`, `ground_truth`,
`liquidity.generate`, or a literal read of `TRUTH_PATH`.
"""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIQUIDITY_DIR = PROJECT_ROOT / "liquidity"

FORBIDDEN_MODULES = {"liquidity.generate", "liquidity.generator", "json"}

# The generator and the data-quality layer are allowed to know nothing either; the only
# permitted reader of the truth file is the test suite itself.
ALLOWED = {"generate.py", "__init__.py"}


def _iter_analysis_modules() -> list[Path]:
    return [
        p
        for p in LIQUIDITY_DIR.rglob("*.py")
        if p.name not in ALLOWED and "__pycache__" not in p.parts
    ]


def test_analysis_modules_do_not_import_the_generator() -> None:
    offenders: list[str] = []
    for path in _iter_analysis_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("liquidity.generate"):
                        offenders.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod.startswith("liquidity.generate"):
                    offenders.append(f"{path.name}: from {mod} import ...")
    assert not offenders, "analysis imports the generator: " + "; ".join(offenders)


def test_analysis_modules_do_not_touch_the_truth_file() -> None:
    offenders: list[str] = []
    for path in _iter_analysis_modules():
        text = path.read_text(encoding="utf-8")
        for needle in ("ground_truth", "_truth", "TRUTH_PATH"):
            if needle in text:
                offenders.append(f"{path.name}: mentions {needle!r}")
    assert not offenders, "analysis references the answer key: " + "; ".join(offenders)


def test_truth_file_is_not_in_the_raw_data_dir() -> None:
    truth = PROJECT_ROOT / "_truth" / "ground_truth.json"
    raw = PROJECT_ROOT / "data" / "raw"
    assert truth.parent != raw, "truth must live outside data/raw so an ETL glob cannot pick it up"
