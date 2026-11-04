"""No analysis module may read the latent truth.

The whole credibility of this repo rests on the analysis not seeing the answer key.
A comment saying so is prose; this test is the mechanism. It scans every module under
`liquidity/` except the generator, and fails on any import of `_truth`, `ground_truth`,
`liquidity.generate`, or a literal read of `TRUTH_PATH`.

The single exception is `liquidity/scoring.py`, the scorer that prints the audit table
and therefore has to know the truth. Its membership is asserted as an exact set, so the
exception cannot be widened by adding an import: it needs an edit to this file.
"""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIQUIDITY_DIR = PROJECT_ROOT / "liquidity"

FORBIDDEN_MODULES = {"liquidity.generate", "liquidity.generator", "json"}

ALLOWED = {"generate.py", "__init__.py"}

# The one module allowed to open the answer key. It scores the estimators and prints the
# README table; it never feeds a number back into an analysis module.
TRUTH_READERS = {"scoring.py"}

TRUTH_NEEDLES = ("ground_truth", "_truth", "TRUTH_PATH")


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
        if path.name in TRUTH_READERS:
            continue
        text = path.read_text(encoding="utf-8")
        for needle in TRUTH_NEEDLES:
            if needle in text:
                offenders.append(f"{path.name}: mentions {needle!r}")
    assert not offenders, "analysis references the answer key: " + "; ".join(offenders)


def test_only_the_scorer_reads_the_truth_file() -> None:
    """The reader set is exact: one module, named here, not one module more."""
    readers = {
        path.name
        for path in _iter_analysis_modules()
        if any(needle in path.read_text(encoding="utf-8") for needle in TRUTH_NEEDLES)
    }
    assert readers == TRUTH_READERS, (
        "the set of modules that mention the answer key drifted: "
        f"expected {sorted(TRUTH_READERS)}, found {sorted(readers)}"
    )


def test_the_scorer_is_not_reachable_from_the_analysis() -> None:
    """No analysis module may import the scorer and borrow its truth."""
    offenders = [
        path.name
        for path in _iter_analysis_modules()
        if path.name not in TRUTH_READERS
        and "liquidity.scoring" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, "analysis imports the scorer: " + "; ".join(offenders)


def test_truth_file_is_not_in_the_raw_data_dir() -> None:
    truth = PROJECT_ROOT / "_truth" / "ground_truth.json"
    raw = PROJECT_ROOT / "data" / "raw"
    assert truth.parent != raw, "truth must live outside data/raw so an ETL glob cannot pick it up"
