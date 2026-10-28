"""Artifact guards detect changed originals, stale execution, and altered results."""
import csv
import hashlib
import json
import shutil

import nbformat
import pytest

from scripts.verify_artifacts import (
    NOTEBOOKS, ORIGINAL_SHA256, ROOT, VerificationError, verify_repository,
)


@pytest.fixture
def evidence_copy(tmp_path):
    # Deliberately omit CSVs: CI must verify committed evidence without downloads.
    for relative in [*ORIGINAL_SHA256, *NOTEBOOKS, "data/source_manifest.json",
                     "reports/execution_manifest.json",
                     "reports/historical/original_saved_text_outputs.json",
                     "reports/historical/replayed_cluster_means.csv"]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return tmp_path


def _change_notebook(root, mutation, *, refresh_manifest=False):
    path = root / NOTEBOOKS[0]
    notebook = nbformat.read(path, as_version=4)
    mutation(next(cell for cell in notebook.cells if cell.cell_type == "code"))
    nbformat.write(notebook, path)
    if refresh_manifest:
        manifest_path = root / "reports/execution_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["notebooks"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))


def test_default_verification_needs_no_source_csvs_and_allows_label_permutation(evidence_copy):
    path = evidence_copy / "reports/historical/replayed_cluster_means.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for label, row in enumerate(reversed(rows)):
            writer.writerow({**row, "legacy_cluster": str(label)})
    assert len(verify_repository(evidence_copy)) == 4
    with pytest.raises(VerificationError, match="Raw data missing"):
        verify_repository(evidence_copy, with_data=True)


def test_original_change_is_rejected(evidence_copy):
    path = evidence_copy / next(iter(ORIGINAL_SHA256))
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(VerificationError, match="Original artifact changed"):
        verify_repository(evidence_copy)


def test_stale_execution_hash_is_rejected(evidence_copy):
    _change_notebook(evidence_copy, lambda cell: cell.update(source=cell.source + "\n# changed"))
    with pytest.raises(VerificationError, match="Notebook fingerprint mismatch"):
        verify_repository(evidence_copy)


@pytest.mark.parametrize(("mutation", "message"), [
    (lambda cell: cell.update(execution_count=99), "Nonsequential execution counters"),
    (lambda cell: cell.outputs.append(nbformat.v4.new_output("error", ename="ValueError", evalue="bad", traceback=[])), "Saved error output"),
    (lambda cell: cell.update(source=cell.source + "\n%pip install package"), "Install/download/Colab code"),
])
def test_execution_guards_still_apply_if_manifest_hash_is_refreshed(evidence_copy, mutation, message):
    _change_notebook(evidence_copy, mutation, refresh_manifest=True)
    with pytest.raises(VerificationError, match=message):
        verify_repository(evidence_copy)


def test_duplicate_manifest_path_is_rejected(evidence_copy):
    path = evidence_copy / "reports/execution_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["notebooks"][1] = manifest["notebooks"][0]
    path.write_text(json.dumps(manifest))
    with pytest.raises(VerificationError, match="exactly the four improved notebooks"):
        verify_repository(evidence_copy)


def test_cluster_mean_outside_original_display_precision_is_rejected(evidence_copy):
    path = evidence_copy / "reports/historical/replayed_cluster_means.csv"
    text = path.read_text()
    path.write_text(text.replace("68.60107595779382", "68.60108"))
    with pytest.raises(VerificationError, match="disagree with historical displayed precision"):
        verify_repository(evidence_copy)
