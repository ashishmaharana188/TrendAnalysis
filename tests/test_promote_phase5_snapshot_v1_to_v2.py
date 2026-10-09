from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from promote_phase5_snapshot_v1_to_v2 import directory_digest, promote_snapshot, sha256_file, validate_snapshot

LABELS = ("UP", "SIDEWAYS", "DOWN")
A = {"UP": 62.0, "SIDEWAYS": 23.0, "DOWN": 15.0}
B = {"UP": 14.0, "SIDEWAYS": 26.0, "DOWN": 60.0}
COMBINED = {"UP": 38.0, "SIDEWAYS": 30.0, "DOWN": 32.0}


def _write_rows(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _make_snapshot(root: Path, rows: list[dict], *, version: str = "1") -> Path:
    root.mkdir(parents=True)
    _write_rows(root / "prediction_folds.jsonl.gz", rows)
    _write_rows(root / "panel_observations.jsonl.gz", [{"date": "2026-01-01", "value": 1}])
    (root / "market_daily.parquet").write_bytes(b"fixture-placeholder-for-migration-test")
    manifest = {
        "schema_version": version,
        "phase5_baseline_version": "PHASE5_8",
        "status": "FROZEN",
        "ticker": "RELIANCE",
        "benchmark": "Nifty_50",
        "analysis_timeframe": "6M",
        "holding_period_months": 1.0,
        "prediction_fold_count": len(rows),
        "panel_observation_count": 1,
        "artifact_sha256": {
            name: sha256_file(root / name)
            for name in ("prediction_folds.jsonl.gz", "panel_observations.jsonl.gz", "market_daily.parquet")
        },
        "immutability": {"overwrite_allowed": False, "purpose": "PHASE5_DEVELOPMENT_BASELINE"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "snapshot.sha256").write_text(directory_digest(root) + "\n", encoding="ascii")
    return root


def _base_row(day: str, *, vectors: bool = False, actual: str = "DOWN") -> dict:
    row = {
        "prediction_date": day,
        "actual_class": actual,
        "probabilities_pct": dict(COMBINED),
        "predicted_trend": "UP",
        "method_a_trend": "UP",
        "method_b_trend": "DOWN",
    }
    if vectors:
        row["method_a_probabilities_pct"] = dict(A)
        row["method_b_probabilities_pct"] = dict(B)
    return row


def test_promotes_existing_exact_vectors_without_touching_source(tmp_path: Path) -> None:
    source = _make_snapshot(
        tmp_path / "RELIANCE_Nifty_50_6M_1M_v1",
        [_base_row("2026-08-24", vectors=True)],
    )
    original_manifest = (source / "manifest.json").read_bytes()
    original_fold_bytes = (source / "prediction_folds.jsonl.gz").read_bytes()

    destination = promote_snapshot(source)
    manifest, rows = validate_snapshot(destination)

    assert destination.name == "RELIANCE_Nifty_50_6M_1M_v2"
    assert manifest["schema_version"] == "2"
    assert manifest["migration"]["prediction_recalculated"] is False
    assert rows[0]["method_a_probabilities_pct"] == A
    assert rows[0]["method_b_probabilities_pct"] == B
    assert (source / "manifest.json").read_bytes() == original_manifest
    assert (source / "prediction_folds.jsonl.gz").read_bytes() == original_fold_bytes


def test_refuses_to_invent_missing_method_vectors(tmp_path: Path) -> None:
    source = _make_snapshot(tmp_path / "source_v1", [_base_row("2026-08-24")])
    with pytest.raises(ValueError, match="no valid exact probability vectors"):
        promote_snapshot(source)
    assert not (tmp_path / "source_v2").exists()


def test_merges_method_only_snapshots_by_prediction_date(tmp_path: Path) -> None:
    source = _make_snapshot(
        tmp_path / "combined_v1",
        [_base_row("2026-08-24"), _base_row("2026-08-25", actual="UP")],
    )
    method_a = _make_snapshot(
        tmp_path / "method_a_v1",
        [
            {"prediction_date": "2026-08-24", "actual_class": "DOWN", "probabilities_pct": A},
            {"prediction_date": "2026-08-25", "actual_class": "UP", "probabilities_pct": A},
        ],
    )
    method_b = _make_snapshot(
        tmp_path / "method_b_v1",
        [
            {"prediction_date": "2026-08-24", "actual_class": "DOWN", "probabilities_pct": B},
            {"prediction_date": "2026-08-25", "actual_class": "UP", "probabilities_pct": B},
        ],
    )

    destination = promote_snapshot(source, method_a_snapshot=method_a, method_b_snapshot=method_b)
    manifest, rows = validate_snapshot(destination)
    assert manifest["schema_version"] == "2"
    assert rows[0]["method_a_probabilities_pct"] == A
    assert rows[0]["method_b_probabilities_pct"] == B
    assert rows[1]["method_a_probabilities_pct"] == A
    assert rows[1]["method_b_probabilities_pct"] == B
    assert manifest["migration"]["method_a_vectors_available"] == 2
    assert manifest["migration"]["method_b_vectors_available"] == 2


def test_refuses_mismatched_sidecar_actual_class(tmp_path: Path) -> None:
    source = _make_snapshot(tmp_path / "combined_v1", [_base_row("2026-08-24", actual="DOWN")])
    method_a = _make_snapshot(
        tmp_path / "method_a_v1",
        [{"prediction_date": "2026-08-24", "actual_class": "UP", "probabilities_pct": A}],
    )
    method_b = _make_snapshot(
        tmp_path / "method_b_v1",
        [{"prediction_date": "2026-08-24", "actual_class": "DOWN", "probabilities_pct": B}],
    )
    with pytest.raises(ValueError, match="Actual-class mismatch"):
        promote_snapshot(source, method_a_snapshot=method_a, method_b_snapshot=method_b)
