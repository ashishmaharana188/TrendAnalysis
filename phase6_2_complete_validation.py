from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]
    json_outputs: tuple[str, ...] = ()


def snapshot_tree_digest(snapshot: Path) -> dict[str, str]:
    """Hash every file in the frozen snapshot to verify this runner is read-only."""
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Snapshot directory not found: {snapshot}")
    result: dict[str, str] = {}
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result[str(path.relative_to(snapshot)).replace("\\", "/")] = digest
    return result


def build_steps(
    repo_root: Path,
    snapshot: Path,
    output_dir: Path,
    *,
    min_calibration_observations: int = 30,
    block_length: int = 5,
    bootstrap_replicates: int = 1000,
    development_fraction: float = 0.70,
    alphas: Sequence[float] = (0.0, 0.0001, 0.001, 0.005, 0.01, 0.02),
    temperature_max_values: Sequence[float] = (4.0, 8.0, 16.0),
) -> list[Step]:
    """Build one deterministic execution plan for all consolidated diagnostics."""
    py = sys.executable
    snap = str(snapshot)
    logs = output_dir / "json"
    logs.mkdir(parents=True, exist_ok=True)

    def command(script: str, *args: Any) -> tuple[str, ...]:
        return (py, str(repo_root / script), snap, *(str(arg) for arg in args))

    alpha_args = [format(float(value), ".12g") for value in alphas]
    max_args = [format(float(value), ".12g") for value in temperature_max_values]
    if not any(abs(float(value) - 4.0) < 1e-12 for value in temperature_max_values):
        raise ValueError("temperature_max_values must include the reference cap 4.0")
    if not any(abs(float(value)) < 1e-15 for value in alphas):
        raise ValueError("alphas must include 0.0 as the unsmoothed reference")

    return [
        Step(
            "probability_construction",
            command(
                "phase6_2_probability_construction_audit.py",
                "--top-cases", 10,
                "--json-out", logs / "probability_construction.json",
            ),
            (str(logs / "probability_construction.json"),),
        ),
        Step(
            "method_probability_skill",
            command(
                "phase6_2_method_probability_audit.py",
                "--min-calibration-observations", min_calibration_observations,
                "--block-length", block_length,
                "--bootstrap-replicates", bootstrap_replicates,
            ),
        ),
        Step(
            "probability_surface_decomposition",
            command(
                "phase6_2_probability_surface_audit.py",
                "--min-calibration-observations", min_calibration_observations,
                "--block-length", block_length,
                "--bootstrap-replicates", bootstrap_replicates,
            ),
        ),
        Step(
            "joint_smoothing_temperature_selection",
            command(
                "phase6_2_joint_calibration_selection.py",
                "--alphas", *alpha_args,
                "--development-fraction", development_fraction,
                "--min-calibration-observations", min_calibration_observations,
                "--min-development-fitted-observations", 20,
                "--temperature-min", 0.25,
                "--temperature-max-values", *max_args,
                "--reference-temperature-max", 4,
                "--reference-grid-size", 161,
                "--block-length", block_length,
                "--bootstrap-replicates", bootstrap_replicates,
                "--json-out", logs / "joint_calibration_selection.json",
            ),
            (str(logs / "joint_calibration_selection.json"),),
        ),
        Step(
            "confidence_diagnostics",
            command(
                "phase6_2_diagnostics_from_snapshot.py",
                "--min-calibration-observations", min_calibration_observations,
            ),
        ),
        Step(
            "selective_prediction",
            command(
                "phase6_selective_from_snapshot.py",
                "--min-calibration-observations", min_calibration_observations,
            ),
        ),
    ]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _assessment(
    *,
    snapshot_before: dict[str, str],
    snapshot_after: dict[str, str],
    steps: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    construction = _load_json(output_dir / "json" / "probability_construction.json") or {}
    joint = _load_json(output_dir / "json" / "joint_calibration_selection.json") or {}
    all_ok = all(step["return_code"] == 0 for step in steps)
    unchanged = snapshot_before == snapshot_after
    methods = construction.get("methods", {})
    return {
        "diagnostic_run_status": "PASS" if all_ok and unchanged else "FAIL",
        "all_diagnostic_steps_passed": all_ok,
        "frozen_snapshot_unchanged": unchanged,
        "snapshot_files_before": len(snapshot_before),
        "snapshot_files_after": len(snapshot_after),
        "snapshot_changed_paths": sorted(
            path for path in set(snapshot_before) | set(snapshot_after)
            if snapshot_before.get(path) != snapshot_after.get(path)
        ),
        "step_statuses": {step["name"]: ("PASS" if step["return_code"] == 0 else "FAIL") for step in steps},
        "probability_construction": {
            "combined_availability": methods.get("Combined", {}).get("availability_status"),
            "method_a_availability": methods.get("Method A", {}).get("availability_status"),
            "method_b_availability": methods.get("Method B", {}).get("availability_status"),
            "method_a_builder_count_trace_status": methods.get("Method A", {}).get("source_count_trace_status"),
            "method_b_builder_count_trace_status": methods.get("Method B", {}).get("source_count_trace_status"),
            "combined_formula_reconstruction": construction.get("combined_reconstruction", {}).get("status"),
            "combined_formula_mismatches": construction.get("combined_reconstruction", {}).get("folds_mismatched"),
        },
        "joint_calibration_selection": {
            "development_fraction": (
                joint.get("split_index_global", 0) / joint.get("folds", 1)
                if joint.get("folds") else None
            ),
            "holdout_start_date": joint.get("split_date_exclusive_development"),
            "selected_parameters_by_surface": {
                name: details.get("selected_parameters")
                for name, details in joint.get("surfaces", {}).items()
            } if isinstance(joint.get("surfaces"), dict) else {},
            "warning": (
                "The holdout is post-hoc relative to prior model-development work. "
                "Do not call it a fresh untouched model-validation sample or promote parameters based on this report alone."
            ),
        },
        "forecast_skill_verdict": (
            "SEE_METHOD_PROBABILITY_SKILL_LOG: diagnostic completion is not itself a model-skill pass. "
            "Judge forecast skill from paired scores and their intervals against stored per-fold baselines."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run all Phase 6.2 frozen-snapshot probability diagnostics and sensitivity checks "
            "in one auditable report; no Phase 5.8/OLAP recomputation."
        )
    )
    parser.add_argument("snapshot", help="Frozen Phase 5.8 artifact directory")
    parser.add_argument("--output-root", default="artifacts/phase6_2_reports")
    parser.add_argument("--min-calibration-observations", type=int, default=30)
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--development-fraction", type=float, default=0.70)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.0001, 0.001, 0.005, 0.01, 0.02])
    parser.add_argument("--temperature-max-values", nargs="+", type=float, default=[4.0, 8.0, 16.0])
    args = parser.parse_args()
    if args.min_calibration_observations < 1:
        parser.error("--min-calibration-observations must be >= 1")
    if args.block_length < 1 or args.bootstrap_replicates < 100:
        parser.error("--block-length must be >= 1 and --bootstrap-replicates must be >= 100")
    if not 0.5 <= args.development_fraction < 1.0:
        parser.error("--development-fraction must be in [0.5, 1.0)")

    repo_root = Path(__file__).resolve().parent
    snapshot = Path(args.snapshot).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / f"complete_validation_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)

    try:
        from analysis.phase5_snapshot import load_phase5_snapshot
        loaded_snapshot = load_phase5_snapshot(str(snapshot))
        manifest = loaded_snapshot.manifest
        if manifest.get("status") != "FROZEN":
            raise RuntimeError(f"Snapshot status is not FROZEN: {manifest.get('status')!r}")
    except Exception as exc:
        print(f"PREFLIGHT FAILED: unable to load a frozen Phase 5.8 snapshot: {type(exc).__name__}: {exc}")
        return 2

    before = snapshot_tree_digest(snapshot)
    steps = build_steps(
        repo_root,
        snapshot,
        output_dir,
        min_calibration_observations=args.min_calibration_observations,
        block_length=args.block_length,
        bootstrap_replicates=args.bootstrap_replicates,
        development_fraction=args.development_fraction,
        alphas=args.alphas,
        temperature_max_values=args.temperature_max_values,
    )

    full_report: list[str] = [
        "PHASE 6.2 COMPLETE FROZEN-SNAPSHOT VALIDATION",
        f"Run ID (UTC): {run_id}",
        f"Snapshot: {snapshot}",
        f"Snapshot status: {manifest.get('status')}",
        f"Schema version: {manifest.get('schema_version')}",
        f"Fold records: {len(loaded_snapshot.prediction_folds)}",
        "Protocol: FROZEN_PHASE5_8_ARTIFACT; NO_OLAP_RECOMPUTATION",
        f"Output directory: {output_dir}",
        "",
    ]
    step_results: list[dict[str, Any]] = []

    for index, step in enumerate(steps, start=1):
        start = time.perf_counter()
        result = subprocess.run(
            list(step.argv),
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        elapsed = time.perf_counter() - start
        log_path = output_dir / f"{index:02d}_{step.name}.log"
        combined_output = result.stdout
        if result.stderr:
            combined_output += ("\n--- STDERR ---\n" if combined_output else "") + result.stderr
        log_path.write_text(combined_output, encoding="utf-8")

        status = "PASS" if result.returncode == 0 else "FAIL"
        step_results.append({
            "name": step.name,
            "status": status,
            "return_code": result.returncode,
            "elapsed_seconds": round(elapsed, 3),
            "log_file": log_path.name,
            "json_outputs": [name for name in step.json_outputs if Path(name).exists()],
            "argv": list(step.argv),
        })
        full_report.extend([
            "=" * 86,
            f"STEP {index}/{len(steps)}: {step.name} | {status} | {elapsed:.2f}s",
            f"Log: {log_path.name}",
            "=" * 86,
            combined_output.rstrip(),
            "",
        ])
        print(f"[{index}/{len(steps)}] {step.name}: {status} ({elapsed:.2f}s)")
        if result.returncode:
            print(f"  See: {log_path}")

    after = snapshot_tree_digest(snapshot)
    assessment = _assessment(
        snapshot_before=before,
        snapshot_after=after,
        steps=step_results,
        output_dir=output_dir,
    )
    summary = {
        "run_id_utc": run_id,
        "snapshot": str(snapshot),
        "snapshot_status": manifest.get("status"),
        "snapshot_schema_version": manifest.get("schema_version"),
        "snapshot_fold_count": len(loaded_snapshot.prediction_folds),
        "output_dir": str(output_dir),
        "steps": step_results,
        "assessment": assessment,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    full_report.extend([
        "=" * 86,
        "FINAL CONSOLIDATED ASSESSMENT",
        "=" * 86,
        json.dumps(assessment, indent=2, sort_keys=True),
        "",
    ])
    (output_dir / "complete_validation_report.txt").write_text("\n".join(full_report) + "\n", encoding="utf-8")

    print(f"\nFinal diagnostic status: {assessment['diagnostic_run_status']}")
    print(f"Frozen snapshot unchanged: {assessment['frozen_snapshot_unchanged']}")
    print(f"Complete report: {output_dir / 'complete_validation_report.txt'}")
    print(f"Machine-readable summary: {output_dir / 'summary.json'}")
    return 0 if assessment["diagnostic_run_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
