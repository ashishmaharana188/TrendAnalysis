from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

import numpy as np

from analysis.outcome_labels import OUTCOME_CLASSES
from analysis.phase5_snapshot import load_phase5_snapshot
from analysis.phase6_probability_metrics import (
    as_date,
    calibrate_with_mature_labels,
    compare_probabilities_paired,
    normalize_distribution,
    print_paired_comparison,
    print_score_line,
    score_probabilities,
)


@dataclass(frozen=True)
class Fold:
    prediction_date: date
    outcome_end_date: date | None
    actual_class: str
    combined_probabilities_pct: dict[str, float]
    baseline_probabilities_pct: dict[str, float] | None
    method_a_probabilities_pct: dict[str, float] | None
    method_b_probabilities_pct: dict[str, float] | None
    method_a_trend: str | None
    method_b_trend: str | None
    method_a_evidence_score: float | None
    method_b_evidence_score: float | None
    method_a_effective_sample_size: float | None
    method_b_effective_sample_size: float | None
    method_a_sample_count: int | None
    method_b_sample_count: int | None
    method_a_variables: tuple[str, ...]
    method_b_variables: tuple[str, ...]
    method_a_condition: tuple[str, ...]
    method_b_condition: tuple[str, ...]
    method_a_selection_status: str
    method_b_selection_status: str
    method_a_selection_metadata: dict[str, Any]
    method_b_selection_metadata: dict[str, Any]


def _dist(raw: Any) -> dict[str, float] | None:
    return normalize_distribution(raw)


def load_folds(snapshot_root: str) -> list[Fold]:
    snapshot = load_phase5_snapshot(snapshot_root)
    rows: list[Fold] = []
    for row in snapshot.prediction_folds:
        actual = row.get("actual_class")
        combined = _dist(row.get("probabilities_pct"))
        if actual not in OUTCOME_CLASSES or combined is None:
            continue
        a_meta = row.get("method_a_selection_metadata") or {}
        b_meta = row.get("method_b_selection_metadata") or {}
        rows.append(
            Fold(
                prediction_date=as_date(row.get("prediction_date")),
                outcome_end_date=as_date(row.get("outcome_end_date")),
                actual_class=actual,
                combined_probabilities_pct=combined,
                baseline_probabilities_pct=_dist(row.get("baseline_probabilities_pct")),
                method_a_probabilities_pct=_dist(row.get("method_a_probabilities_pct")),
                method_b_probabilities_pct=_dist(row.get("method_b_probabilities_pct")),
                method_a_trend=row.get("method_a_trend"),
                method_b_trend=row.get("method_b_trend"),
                method_a_evidence_score=(
                    float(row["method_a_evidence_score"])
                    if row.get("method_a_evidence_score") is not None else None
                ),
                method_b_evidence_score=(
                    float(row["method_b_evidence_score"])
                    if row.get("method_b_evidence_score") is not None else None
                ),
                method_a_effective_sample_size=(
                    float(row["method_a_effective_sample_size"])
                    if row.get("method_a_effective_sample_size") is not None else None
                ),
                method_b_effective_sample_size=(
                    float(row["method_b_effective_sample_size"])
                    if row.get("method_b_effective_sample_size") is not None else None
                ),
                method_a_sample_count=(
                    int(row["method_a_sample_count"])
                    if row.get("method_a_sample_count") is not None else None
                ),
                method_b_sample_count=(
                    int(row["method_b_sample_count"])
                    if row.get("method_b_sample_count") is not None else None
                ),
                method_a_variables=tuple(row.get("method_a_variables") or ()),
                method_b_variables=tuple(row.get("method_b_variables") or ()),
                method_a_condition=tuple(row.get("method_a_condition") or ()),
                method_b_condition=tuple(row.get("method_b_condition") or ()),
                method_a_selection_status=str(a_meta.get("selection_status", "MISSING")),
                method_b_selection_status=str(b_meta.get("selection_status", "MISSING")),
                method_a_selection_metadata=dict(a_meta),
                method_b_selection_metadata=dict(b_meta),
            )
        )
    return sorted(rows, key=lambda row: row.prediction_date)


def _availability_status(present: int, total: int) -> str:
    if present == 0:
        return "UNAVAILABLE"
    if present < total:
        return "PARTIAL"
    return "COMPLETE"


def _mean_selected(folds: Sequence[Fold], field: str) -> float | None:
    values = [getattr(row, field) for row in folds if getattr(row, field) is not None]
    return float(np.mean(values)) if values else None


def _print_method_report(
    name: str,
    folds: list[Fold],
    prob_field: str,
    status_field: str | None,
    *,
    min_calibration_observations: int,
    block_length: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    actuals = [row.actual_class for row in folds]
    dates = [row.prediction_date for row in folds]
    end_dates = [row.outcome_end_date for row in folds]
    raw = [getattr(row, prob_field) for row in folds]
    baseline = [row.baseline_probabilities_pct for row in folds]
    available_indices = [i for i, probs in enumerate(raw) if probs is not None]
    total = len(folds)
    print(f"{name} probability surface:")
    print(f"  vectors present: {len(available_indices)}/{total}")
    print(f"  availability status: {_availability_status(len(available_indices), total)}")
    status_values = (
        [getattr(row, status_field) for row in folds] if status_field is not None
        else ["FORECAST_AVAILABLE" if getattr(row, prob_field) is not None else "NO_FORECAST" for row in folds]
    )
    print(f"  selection statuses: {dict(Counter(status_values))}")
    if not available_indices:
        print("  STATUS: no usable probability vectors were stored for this method.")
        return {"raw": raw, "calibrated": [None] * len(folds), "calibration": None, "available_indices": []}

    method_calibration = calibrate_with_mature_labels(
        dates,
        end_dates,
        actuals,
        raw,
        min_calibration_observations=min_calibration_observations,
    )
    calibrated = list(method_calibration.probabilities_pct)
    status_counts = Counter(method_calibration.statuses[i] for i in available_indices)
    fitted_indices = [i for i in available_indices if method_calibration.statuses[i] == "FITTED"]
    training_sizes = [method_calibration.mature_training_observations[i] for i in available_indices]
    fitted_temperatures = [method_calibration.temperatures[i] for i in fitted_indices]

    selected_actual = [actuals[i] for i in available_indices]
    selected_raw = [raw[i] for i in available_indices]
    selected_calibrated = [calibrated[i] for i in available_indices]
    print(f"  prediction dates: {dates[available_indices[0]]} through {dates[available_indices[-1]]}")
    print(f"  calibration protocol: PREQUENTIAL_ONLY_OUTCOMES_MATURED_BEFORE_PREDICTION")
    print(f"  calibration statuses: {dict(status_counts)}")
    print(
        "  mature calibration history sizes: "
        f"min={min(training_sizes)}, median={float(np.median(training_sizes)):.1f}, max={max(training_sizes)}"
    )
    if fitted_temperatures:
        print(
            f"  fitted temperatures: mean={float(np.mean(fitted_temperatures)):.4f}, "
            f"median={float(np.median(fitted_temperatures)):.4f}"
        )
    else:
        print("  fitted temperatures: none; no fold had sufficient mature history")

    raw_metrics = score_probabilities(selected_actual, selected_raw)
    calibrated_metrics = score_probabilities(selected_actual, selected_calibrated)
    print_score_line("raw", raw_metrics)
    print_score_line("prequential calibrated (all available folds)", calibrated_metrics)
    print("  calibrated mean class probabilities:")
    for label in OUTCOME_CLASSES:
        print(
            f"    {label}: raw={raw_metrics['mean_probability_pct'][label]:.2f}% "
            f"calibrated={calibrated_metrics['mean_probability_pct'][label]:.2f}% "
            f"observed={raw_metrics['observed_class_frequency_pct'][label]:.2f}% "
            f"calibrated_bias={calibrated_metrics['probability_bias_pct_points'][label]:+.2f}pp"
        )

    paired_indices = [i for i in available_indices if baseline[i] is not None]
    print(f"  stored fold-baseline coverage on this method's sample: {len(paired_indices)}/{len(available_indices)}")
    if paired_indices:
        paired_actual = [actuals[i] for i in paired_indices]
        paired_baseline = [baseline[i] for i in paired_indices]
        paired_raw = [raw[i] for i in paired_indices]
        paired_calibrated = [calibrated[i] for i in paired_indices]
        baseline_metrics = score_probabilities(paired_actual, paired_baseline)
        print_score_line("stored per-fold baseline (paired sample)", baseline_metrics)
        print_paired_comparison(
            "raw model versus stored per-fold baseline",
            compare_probabilities_paired(
                paired_actual, paired_raw, paired_baseline,
                block_length=block_length,
                replicates=bootstrap_replicates,
                seed=seed,
            ),
        )
        print_paired_comparison(
            "calibrated model versus stored per-fold baseline",
            compare_probabilities_paired(
                paired_actual, paired_calibrated, paired_baseline,
                block_length=block_length,
                replicates=bootstrap_replicates,
                seed=seed + 100,
            ),
        )
    else:
        baseline_metrics = None
        print("  paired baseline comparison: unavailable; no comparable stored baseline vectors")

    fitted_paired = [i for i in fitted_indices if baseline[i] is not None]
    if fitted_paired:
        fitted_actual = [actuals[i] for i in fitted_paired]
        fitted_raw = [raw[i] for i in fitted_paired]
        fitted_cal = [calibrated[i] for i in fitted_paired]
        fitted_base = [baseline[i] for i in fitted_paired]
        print(f"  fitted-only matched calibration subset: n={len(fitted_paired)}")
        print_score_line("raw on fitted-only subset", score_probabilities(fitted_actual, fitted_raw))
        print_score_line("calibrated on fitted-only subset", score_probabilities(fitted_actual, fitted_cal))
        print_score_line("stored baseline on fitted-only subset", score_probabilities(fitted_actual, fitted_base))
        print_paired_comparison(
            "calibrated model versus baseline, fitted-only subset",
            compare_probabilities_paired(
                fitted_actual, fitted_cal, fitted_base,
                block_length=block_length,
                replicates=bootstrap_replicates,
                seed=seed + 200,
            ),
        )
    else:
        print("  fitted-only matched calibration subset: none with a stored baseline")

    print("  scoring definitions: Brier is multiclass sum-of-squared-error per fold; ECE is top-label ECE with 10 bins.")
    print("  uncertainty: circular moving-block bootstrap of paired per-fold score differences; positive gains favor the method.")
    return {
        "raw": raw,
        "calibrated": calibrated,
        "calibration": method_calibration,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "baseline_metrics": baseline_metrics,
        "available_indices": available_indices,
    }


def _print_gate_diagnostics(folds: list[Fold]) -> None:
    print("Selection-gate diagnostics (stored fold metadata):")
    for method_name, metadata_field, status_field in (
        ("Method A", "method_a_selection_metadata", "method_a_selection_status"),
        ("Method B", "method_b_selection_metadata", "method_b_selection_status"),
    ):
        metadata_rows = [getattr(row, metadata_field) for row in folds]
        gates = [meta.get("gate_metadata") for meta in metadata_rows if isinstance(meta.get("gate_metadata"), dict)]
        candidates = [int(gate["candidate_evaluation_count"]) for gate in gates if gate.get("candidate_evaluation_count") is not None]
        discovered = [int(gate["discovery_candidate_count"]) for gate in gates if gate.get("discovery_candidate_count") is not None]
        mt = [gate.get("multiple_testing") for gate in gates if isinstance(gate.get("multiple_testing"), dict)]
        accepted = [int(item["accepted_candidate_count"]) for item in mt if item.get("accepted_candidate_count") is not None]
        configured = [int(item["permutations_configured"]) for item in mt if item.get("permutations_configured") is not None]
        run = [int(item["permutations_run"]) for item in mt if item.get("permutations_run") is not None]
        alphas = sorted({float(item["alpha"]) for item in mt if item.get("alpha") is not None})
        adjusted = [float(item["selected_adjusted_p_value"]) for item in mt if item.get("selected_adjusted_p_value") is not None]
        print(f"  {method_name}:")
        print(f"    selection statuses: {dict(Counter(getattr(row, status_field) for row in folds))}")
        print(f"    folds with gate metadata: {len(gates)}/{len(folds)}")
        if candidates:
            print(f"    candidate evaluations per fold: mean={float(np.mean(candidates)):.1f}, min={min(candidates)}, max={max(candidates)}")
        if discovered:
            print(f"    discovery candidates per fold: mean={float(np.mean(discovered)):.1f}, min={min(discovered)}, max={max(discovered)}")
        if accepted:
            print(f"    accepted candidates after multiple-testing control: total={sum(accepted)}, folds_with_any={sum(value > 0 for value in accepted)}/{len(accepted)}")
        if alphas:
            print(f"    multiple-testing alpha values: {alphas}")
        if configured:
            print(f"    permutations configured: min={min(configured)}, max={max(configured)}")
        if run:
            print(f"    permutations run: min={min(run)}, max={max(run)}")
        if adjusted:
            print(f"    selected adjusted p-value: min={min(adjusted):.4f}, median={float(np.median(adjusted)):.4f}, max={max(adjusted):.4f}")
        else:
            print("    selected adjusted p-value: none recorded for selected candidates")


def _print_common_overlap(
    folds: list[Fold],
    method_a: dict[str, Any],
    method_b: dict[str, Any],
    combined: dict[str, Any],
    *,
    block_length: int,
    bootstrap_replicates: int,
) -> None:
    a_raw, b_raw = method_a["raw"], method_b["raw"]
    c_raw = combined["raw"]
    a_cal, b_cal, c_cal = method_a["calibrated"], method_b["calibrated"], combined["calibrated"]
    common = [
        i for i in range(len(folds))
        if a_raw[i] is not None and b_raw[i] is not None and c_raw[i] is not None
        and folds[i].baseline_probabilities_pct is not None
        and a_cal[i] is not None and b_cal[i] is not None and c_cal[i] is not None
    ]
    print("COMMON OVERLAP: COMBINED, METHOD A AND METHOD B")
    print(f"  same-fold sample: {len(common)}/{len(folds)}")
    if not common:
        print("  STATUS: no common sample available")
        return
    actual = [folds[i].actual_class for i in common]
    baseline = [folds[i].baseline_probabilities_pct for i in common]
    raw_maps = {
        "combined raw": [c_raw[i] for i in common],
        "Method A raw": [a_raw[i] for i in common],
        "Method B raw": [b_raw[i] for i in common],
    }
    cal_maps = {
        "combined calibrated": [c_cal[i] for i in common],
        "Method A calibrated": [a_cal[i] for i in common],
        "Method B calibrated": [b_cal[i] for i in common],
    }
    for label, maps in (("raw", raw_maps), ("prequential calibrated", cal_maps)):
        print(f"  {label} forecasts on identical folds:")
        for name, probabilities in maps.items():
            print_score_line(name, score_probabilities(actual, probabilities))
    print_paired_comparison(
        "Method A calibrated versus Method B calibrated (same folds; positive favors A)",
        compare_probabilities_paired(
            actual, cal_maps["Method A calibrated"], cal_maps["Method B calibrated"],
            block_length=block_length, replicates=bootstrap_replicates, seed=20261010,
        ),
    )
    print_paired_comparison(
        "Method A calibrated versus combined calibrated (same folds; positive favors A)",
        compare_probabilities_paired(
            actual, cal_maps["Method A calibrated"], cal_maps["combined calibrated"],
            block_length=block_length, replicates=bootstrap_replicates, seed=20261110,
        ),
    )
    print_paired_comparison(
        "Method B calibrated versus combined calibrated (same folds; positive favors B)",
        compare_probabilities_paired(
            actual, cal_maps["Method B calibrated"], cal_maps["combined calibrated"],
            block_length=block_length, replicates=bootstrap_replicates, seed=20261210,
        ),
    )
    print("  Note: each prequential calibrator used only that model's past labels whose outcome-end dates had matured by each forecast date.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the exact frozen Phase 5.8 probability vectors, method availability, "
            "mature-label prequential calibration, paired baseline skill and block-bootstrap uncertainty."
        )
    )
    parser.add_argument("snapshot")
    parser.add_argument("--min-calibration-observations", type=int, default=30)
    parser.add_argument("--block-length", type=int, default=5,
                        help="Moving-block length in evaluated folds; run sensitivity checks at several values.")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    if args.block_length < 1 or args.bootstrap_replicates < 100:
        parser.error("--block-length must be >=1 and --bootstrap-replicates must be >=100")

    folds = load_folds(args.snapshot)
    if not folds:
        raise RuntimeError("Snapshot contains no usable folds.")
    actuals = [row.actual_class for row in folds]
    dates = [row.prediction_date for row in folds]
    ends = [row.outcome_end_date for row in folds]

    print("PHASE 6.2 METHOD-A/B PROBABILITY AND FORECAST-SKILL AUDIT")
    print(f"Snapshot: {args.snapshot}")
    print("Snapshot status: FROZEN")
    print(f"Folds: {len(folds)}")
    print(f"Dates: {dates[0]} through {dates[-1]}")
    print("Protocol: FROZEN_FOLDS; NO_OLAP_RECOMPUTATION")
    print("Calibration: prequential temperature scaling with outcome maturity gate")
    print(f"Minimum mature calibration observations: {args.min_calibration_observations}")
    print(f"Moving-block bootstrap: block_length={args.block_length} folds, replicates={args.bootstrap_replicates}")
    print("")

    combined = _print_method_report(
        "Combined model", folds, "combined_probabilities_pct", None,
        min_calibration_observations=args.min_calibration_observations,
        block_length=args.block_length, bootstrap_replicates=args.bootstrap_replicates, seed=20261010,
    )
    print("")
    method_a = _print_method_report(
        "Method A", folds, "method_a_probabilities_pct", "method_a_selection_status",
        min_calibration_observations=args.min_calibration_observations,
        block_length=args.block_length, bootstrap_replicates=args.bootstrap_replicates, seed=20262010,
    )
    print("")
    method_b = _print_method_report(
        "Method B", folds, "method_b_probabilities_pct", "method_b_selection_status",
        min_calibration_observations=args.min_calibration_observations,
        block_length=args.block_length, bootstrap_replicates=args.bootstrap_replicates, seed=20263010,
    )

    print("")
    _print_common_overlap(
        folds, method_a, method_b, combined,
        block_length=args.block_length,
        bootstrap_replicates=args.bootstrap_replicates,
    )

    print("")
    _print_gate_diagnostics(folds)
    print("\nSelected-relationship metadata:")
    for name, status_field, score_field, ess_field, count_field, vars_field, cond_field in (
        ("Method A", "method_a_selection_status", "method_a_evidence_score", "method_a_effective_sample_size", "method_a_sample_count", "method_a_variables", "method_a_condition"),
        ("Method B", "method_b_selection_status", "method_b_evidence_score", "method_b_effective_sample_size", "method_b_sample_count", "method_b_variables", "method_b_condition"),
    ):
        statuses = Counter(getattr(row, status_field) for row in folds)
        selected = [row for row in folds if getattr(row, vars_field)]
        print(f"  {name} selection statuses: {dict(statuses)}")
        print(f"  {name} selected relationships recorded: {len(selected)}/{len(folds)}")
        if selected:
            print(f"    mean evidence score={_mean_selected(selected, score_field):.6f}")
            print(f"    mean effective sample size={_mean_selected(selected, ess_field):.3f}")
            print(f"    mean supporting sample count={_mean_selected(selected, count_field):.3f}")
            unique = len({(getattr(row, vars_field), getattr(row, cond_field)) for row in selected})
            print(f"    unique selected relationship signatures={unique}")

    print("")
    print("Interpretation guardrails:")
    print("  - All forecasts, outcomes, baseline vectors and method vectors are read from the frozen Phase 5.8 snapshot.")
    print("  - The stored baseline is evaluated fold-by-fold; this script does not replace it with a retrospective class-frequency baseline.")
    print("  - Positive paired log-loss/Brier gains mean the candidate model has lower loss than its paired reference.")
    print("  - Block-bootstrap intervals are uncertainty diagnostics, not proof of causality or trading profitability.")
    print("  - Repeat with alternate --block-length values to assess sensitivity; the default of five folds is a documented heuristic.")


if __name__ == "__main__":
    main()
