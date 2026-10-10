from __future__ import annotations

import json
from pathlib import Path

import pytest

from phase6_2_complete_validation import build_steps, snapshot_tree_digest


def test_plan_contains_all_consolidated_diagnostic_stages(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    snapshot = tmp_path / "frozen_snapshot"
    snapshot.mkdir()
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    steps = build_steps(repo_root, snapshot, output_dir, bootstrap_replicates=100)
    names = [step.name for step in steps]

    assert names == [
        "probability_construction",
        "method_probability_skill",
        "probability_surface_decomposition",
        "joint_smoothing_temperature_selection",
        "confidence_diagnostics",
        "selective_prediction",
    ]
    assert all(step.argv[0] for step in steps)
    assert all(str(snapshot) in step.argv for step in steps)
    assert (output_dir / "json").is_dir()


def test_plan_requires_unsmoothed_reference_alpha(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    with pytest.raises(ValueError, match="include 0.0"):
        build_steps(repo_root, snapshot, output_dir, alphas=(0.001, 0.01))


def test_plan_requires_reference_temperature_cap(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    with pytest.raises(ValueError, match="include the reference cap 4.0"):
        build_steps(repo_root, snapshot, output_dir, temperature_max_values=(8.0, 16.0))


def test_snapshot_digest_is_stable_and_detects_mutation(tmp_path: Path) -> None:
    snapshot = tmp_path / "frozen_snapshot"
    snapshot.mkdir()
    source = snapshot / "manifest.json"
    source.write_text(json.dumps({"status": "FROZEN"}), encoding="utf-8")

    before = snapshot_tree_digest(snapshot)
    after_no_change = snapshot_tree_digest(snapshot)
    assert before == after_no_change

    source.write_text(json.dumps({"status": "CHANGED"}), encoding="utf-8")
    after_change = snapshot_tree_digest(snapshot)
    assert before != after_change
