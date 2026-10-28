"""Verify preserved artifacts and saved notebook execution without network access.

The default checks committed evidence and needs no CSV downloads. --with-data
also fingerprints local sources and repeats the historical cleaning comparison.
This verifies saved execution evidence; use execute_notebooks.py for a fresh run.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
from itertools import permutations
import json
from pathlib import Path
import re
import sys

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "1f00286346c6a4549059bf804994c3765a73a156"
# Original README is intentionally updated; only these five artifacts are pinned.
ORIGINAL_SHA256 = {
    "602_Regression_Model_Abhinav&Devika.ipynb": "0bdac7d3d4196029edf26449cb4c342b8fcc6b5f70e8d80bef18af1d2a9bbcd4",
    "602_Clustering_&_Classification_Model_Abhinav&Devika.ipynb": "004c9eda8655183ad76fea0691ec44c3164d6e65be9c1f029afc04427599474e",
    "602_Classification_Model_Abhinav&Devika.ipynb": "b10dc78bd13cc705517bafaa85e4d0deedeb9ca6bc0d80ae9c1d84bede85b75a",
    "602 executive  summary -  Devika  & Abhinav.pdf": "92a961f77faef12ba0052c4a2ff3a9c872fef164d4ae41ebc7d2307d2d74a556",
    "DATA 602 FINAL PROJECT PPT[1].pdf": "3b5ac7998c98418dd5336ea134440d6b4e02ca1a8baa8a89113e2d3138161e4a",
}
NOTEBOOKS = (
    "notebooks/01_data_audit.ipynb",
    "notebooks/02_capacity_clustering.ipynb",
    "notebooks/03_supervised_evaluation.ipynb",
    "notebooks/04_historical_replay.ipynb",
)
CLUSTER_NOTEBOOK = "602_Clustering_&_Classification_Model_Abhinav&Devika.ipynb"
MEAN_COLUMNS = ("BED_LIC", "BED_AVL", "BED_STF", "Amount")
FORBIDDEN_CODE = re.compile(r"\b(?:pip\d*|gdown|colab)\b", re.IGNORECASE)
SUMMARY_MARKER = "Cluster Summary (mean values):"


class VerificationError(ValueError):
    """A committed artifact or local data source failed verification."""


def _require(condition, message):
    if not condition:
        raise VerificationError(message)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_from_snapshot(root):
    snapshot = _json(root / "reports/historical/original_saved_text_outputs.json")
    summaries = [text for entry in snapshot[CLUSTER_NOTEBOOK]
                 for text in entry["saved_text_outputs"] if SUMMARY_MARKER in text]
    _require(len(summaries) == 1, "Expected one historical K-Means mean table in saved text outputs")
    # Bind the extracted evidence to the preserved original notebook, whose full
    # bytes are checked above, rather than trusting an editable JSON transcription.
    original = _json(root / CLUSTER_NOTEBOOK)
    source_summaries = ["".join(output.get("text", []))
                        for cell in original["cells"] for output in cell.get("outputs", [])
                        if SUMMARY_MARKER in "".join(output.get("text", []))]
    _require(source_summaries == summaries, "Saved K-Means summary text differs from original notebook")
    return summaries[0]


def _printed_precision_matches(actual, printed):
    """Accept half a unit in the last printed decimal place, including exponents.

    The original beds have six decimal places; Amount has six decimal places in
    scientific notation. A blanket relative tolerance would weaken one of these.
    """
    try:
        value, reference = Decimal(str(actual)), Decimal(printed)
    except InvalidOperation:
        return False
    if not value.is_finite() or not reference.is_finite():
        return False
    half_printed_unit = Decimal("0.5").scaleb(reference.as_tuple().exponent)
    return abs(value - reference) <= half_printed_unit


def verify_cluster_means(root):
    """Compare original displayed means with replay, independent of label IDs."""
    summary = _summary_from_snapshot(root)
    printed = [line.split()[1:] for line in summary.splitlines()
               if re.match(r"^\s*\d+\s+", line)]
    _require(len(printed) == 3 and all(len(row) == 4 for row in printed),
             "Historical K-Means summary must contain three four-feature mean rows")
    with (root / "reports/historical/replayed_cluster_means.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames == ["legacy_cluster", *MEAN_COLUMNS], "Unexpected replayed cluster mean columns")
        actual = list(reader)
    _require(len(actual) == 3 and len({row["legacy_cluster"] for row in actual}) == 3,
             "Replayed K-Means means must contain three distinct clusters")
    matches = any(all(all(_printed_precision_matches(row[column], token)
                             for column, token in zip(MEAN_COLUMNS, expected))
                         for row, expected in zip(order, printed))
                  for order in permutations(actual))
    _require(matches, "Replayed K-Means means disagree with historical displayed precision")


def _verify_notebooks(root, source_manifest):
    manifest = _json(root / "reports/execution_manifest.json")
    records = manifest.get("notebooks", [])
    _require(isinstance(records, list) and all(isinstance(item, dict) for item in records),
             "Notebook execution manifest must contain a list of records")
    paths = [item.get("notebook") for item in records]
    _require(len(paths) == len(NOTEBOOKS) and set(paths) == set(NOTEBOOKS),
             "Execution manifest must contain exactly the four improved notebooks, once each")
    _require({path.relative_to(root).as_posix() for path in (root / "notebooks").glob("*.ipynb")} == set(NOTEBOOKS),
             "Expected exactly the four improved notebooks on disk")
    _require(manifest.get("raw_sha256") == source_manifest["raw"]["sha256"],
             "Execution manifest raw fingerprint differs from source manifest")
    for record in records:
        path = root / record["notebook"]
        _require(_sha256(path) == record.get("sha256"), f"Notebook fingerprint mismatch: {record['notebook']}")
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        _require(bool(cells), f"No executed code cells: {record['notebook']}")
        _require([cell.execution_count for cell in cells] == list(range(1, len(cells) + 1)),
                 f"Nonsequential execution counters: {record['notebook']}")
        _require(record.get("executed_code_cells") == len(cells) and record.get("errors") == 0,
                 f"Execution manifest counts disagree: {record['notebook']}")
        for cell in cells:
            _require(not any(output.output_type == "error" for output in cell.get("outputs", [])),
                     f"Saved error output: {record['notebook']}")
            _require(not FORBIDDEN_CODE.search(cell.source),
                     f"Install/download/Colab code in improved notebook: {record['notebook']}")


def _verify_data(root, source_manifest):
    import pandas as pd

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from hospital_ml.data import LEGACY_COLUMNS, historical_cleaning, read_raw

    source = source_manifest["raw"]
    raw_path = root / source["path"]
    _require(raw_path.is_file(), "Raw data missing; fetch sources before using --with-data")
    _require(_sha256(raw_path) == source["sha256"], "Raw data fingerprint mismatch")
    raw = read_raw(raw_path)
    _require(list(raw.shape) == source["observed_shape"], "Raw data shape differs from fingerprinted source manifest")
    cleaned, _ = historical_cleaning(raw)
    historical = source_manifest["historical_cleaned"]
    historical_path = root / historical["path"]
    if not historical_path.exists():
        return f"Raw source verified; historical cleaning replay produced {len(cleaned):,} rows (shared cleaned CSV absent)"
    _require(_sha256(historical_path) == historical["sha256"], "Historical cleaned data fingerprint mismatch")
    downloaded = pd.read_csv(historical_path, low_memory=False)
    _require(list(downloaded.shape) == historical["observed_shape"], "Historical cleaned data shape differs from source manifest")
    try:
        pd.testing.assert_frame_equal(cleaned[LEGACY_COLUMNS], downloaded[LEGACY_COLUMNS], check_dtype=False)
    except AssertionError as error:
        raise VerificationError("Historical cleaning replay differs from the shared cleaned CSV") from error
    return f"Local source fingerprints and exact eight-column historical cleaning match verified ({len(raw):,} -> {len(cleaned):,} rows)"


def verify_repository(root=ROOT, *, with_data=False):
    root = Path(root)
    for name, expected in ORIGINAL_SHA256.items():
        _require(_sha256(root / name) == expected, f"Original artifact changed from {SOURCE_COMMIT}: {name}")
    source_manifest = _json(root / "data/source_manifest.json")
    _verify_notebooks(root, source_manifest)
    verify_cluster_means(root)
    messages = ["Five original artifacts unchanged; historical README excluded intentionally",
                "Four valid improved notebooks: sequential counters, zero saved errors, matching execution hashes",
                "Historical replayed K-Means means match original displayed precision, allowing label permutations"]
    if with_data:
        messages.append(_verify_data(root, source_manifest))
    else:
        messages.append("Local CSV checks skipped; use --with-data to verify source files and cleaning equality")
    return messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-data", action="store_true", help="Also verify local source CSVs and repeat historical cleaning")
    arguments = parser.parse_args()
    try:
        for message in verify_repository(with_data=arguments.with_data):
            print(f"PASS: {message}")
    except (OSError, ValueError, KeyError, TypeError, nbformat.ValidationError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
