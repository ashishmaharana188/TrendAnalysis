from __future__ import annotations

"""One chronological, maturity-gated joint smoothing/temperature experiment.

This script uses frozen Phase 5.8 folds only. It selects each surface's
(alpha, temperature_max) on the development period using labels that matured
strictly before the holdout starts, then evaluates the selected pair on the
later period. During that later period, the calibrator updates only from
previous outcomes that have matured strictly before each prediction date.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_CLASSES = ("UP", "SIDEWAYS", "DOWN")
_DEFAULT_ALPHAS = (0.0, 0.0001, 0.001, 0.005, 0.01, 0.02)
_DEFAULT_TEMP_MAX = (4.0, 8.0, 16.0)


def _grid_size_for_cap(
    temperature_min: float,
    temperature_max: float,
    reference_max: float,
    reference_grid_size: int,
) -> int:
    ratio = math.log(temperature_max / temperature_min) / math.log(reference_max / temperature_min)
    return max(3, int(reference_grid_size * ratio))


def _calibrate_surface(
    rows: Sequence[Any],
    source_field: str,
    *,
    alpha: float,
    temperature_min: float,
    temperature_max: float,
    temperature_grid_size: int,
    min_calibration_observations: int,
) -> dict[str, Any]:
    from analysis.phase6_probability_metrics import calibrate_with_mature_labels
    from phase6_2_probability_smoothing_sensitivity import smooth_distribution

    dates = [row.prediction_date for row in rows]
    end_dates = [row.outcome_end_date for row in rows]
    actual = [row.actual_class for row in rows]
    raw = [smooth_distribution(getattr(row, source_field), alpha) for row in rows]
    calibrated = calibrate_with_mature_labels(
        dates,
        end_dates,
        actual,
        raw,
        min_calibration_observations=min_calibration_observations,
        temperature_min=temperature_min,
        temperature_max=temperature_max,
        temperature_grid_size=temperature_grid_size,
    )
    return {
        "raw": raw,
        "calibration": calibrated,
        "temperature_grid_size": temperature_grid_size,
    }


def _available_surface_rows(rows: Sequence[Any], source_field: str) -> list[Any]:
    """Return rows with a valid stored distribution for ``source_field``.

    Keep the availability gate in one testable helper. The consolidated runner
    calls this before any per-surface calibration or scoring begins.
    """
    from analysis.phase6_probability_metrics import normalize_distribution

    return [
        row
        for row in rows
        if normalize_distribution(getattr(row, source_field, None)) is not None
    ]


def _idx_for_dates(rows: Sequence[Any], predicate) -> list[int]:
    return [index for index, row in enumerate(rows) if predicate(row)]


def _score_indices(actual: Sequence[str], vectors: Sequence[dict[str, float] | None], indices: Sequence[int]) -> dict[str, Any] | None:
    from analysis.phase6_probability_metrics import score_probabilities

    if not indices or any(vectors[index] is None for index in indices):
        return None
    typed = [vectors[index] for index in indices]
    return score_probabilities(
        [actual[index] for index in indices],
        [item for item in typed if item is not None],
    )


def _filtered_score(actual: Sequence[str], vectors: Sequence[dict[str, float] | None], indices: Sequence[int]) -> dict[str, Any] | None:
    valid = [index for index in indices if vectors[index] is not None and actual[index] in _CLASSES]
    return _score_indices(actual, vectors, valid) if valid else None


def _compare(
    actual: Sequence[str],
    left: Sequence[dict[str, float] | None],
    right: Sequence[dict[str, float] | None],
    indices: Sequence[int],
    *,
    block_length: int,
    replicates: int,
    seed: int,
) -> dict[str, Any] | None:
    from analysis.phase6_probability_metrics import compare_probabilities_paired

    common = [
        index for index in indices
        if actual[index] in _CLASSES and left[index] is not None and right[index] is not None
    ]
    if not common:
        return None
    left_vectors = [left[index] for index in common]
    right_vectors = [right[index] for index in common]
    return compare_probabilities_paired(
        [actual[index] for index in common],
        [item for item in left_vectors if item is not None],
        [item for item in right_vectors if item is not None],
        block_length=block_length,
        replicates=replicates,
        seed=seed,
    )


def _summary_score(name: str, actual: Sequence[str], vectors: Sequence[dict[str, float] | None], indices: Sequence[int]) -> dict[str, Any] | None:
    from analysis.phase6_probability_metrics import print_score_line

    result = _filtered_score(actual, vectors, indices)
    if result is not None:
        print_score_line(name, result)
    else:
        print(f"  {name}: no scoreable observations")
    return result


def _analyze_surface(
    name: str,
    rows: Sequence[Any],
    source_field: str,
    *,
    global_split_date,
    alphas: Sequence[float],
    temperature_max_values: Sequence[float],
    temperature_min: float,
    reference_temperature_max: float,
    reference_grid_size: int,
    min_calibration_observations: int,
    min_development_fitted_observations: int,
    block_length: int,
    bootstrap_replicates: int,
    seed_base: int,
) -> dict[str, Any]:
    from analysis.phase6_probability_metrics import (
        normalize_distribution,
        print_paired_comparison,
        summarize_temperature_boundaries,
    )

    actual = [row.actual_class for row in rows]
    dates = [row.prediction_date for row in rows]
    ends = [row.outcome_end_date for row in rows]
    source = [normalize_distribution(getattr(row, source_field)) for row in rows]
    available_indices = [i for i, vector in enumerate(source) if vector is not None]
    development_indices = [i for i, row in enumerate(rows) if row.prediction_date < global_split_date]
    holdout_indices = [i for i, row in enumerate(rows) if row.prediction_date >= global_split_date]
    holdout_start = global_split_date

    candidate_rows: list[dict[str, Any]] = []
    candidate_cache: dict[tuple[float, float], dict[str, Any]] = {}
    for alpha_index, alpha in enumerate(alphas):
        for cap_index, temperature_max in enumerate(temperature_max_values):
            grid_size = _grid_size_for_cap(
                temperature_min,
                float(temperature_max),
                reference_temperature_max,
                reference_grid_size,
            )
            candidate = _calibrate_surface(
                rows,
                source_field,
                alpha=float(alpha),
                temperature_min=temperature_min,
                temperature_max=float(temperature_max),
                temperature_grid_size=grid_size,
                min_calibration_observations=min_calibration_observations,
            )
            candidate_cache[(float(alpha), float(temperature_max))] = candidate
            cal = candidate["calibration"]
            eligible_selection_indices = [
                index for index in development_indices
                if source[index] is not None
                and cal.statuses[index] == "FITTED"
                and ends[index] is not None
                and ends[index] < holdout_start
                and cal.probabilities_pct[index] is not None
            ]
            score = _score_indices(actual, cal.probabilities_pct, eligible_selection_indices)
            eligible = score is not None and score["observations"] >= min_development_fitted_observations
            row = {
                "alpha": float(alpha),
                "temperature_max": float(temperature_max),
                "temperature_grid_size": grid_size,
                "selection_observations": len(eligible_selection_indices),
                "selection_log_loss": score["log_loss"] if score else None,
                "selection_brier_score": score["brier_score"] if score else None,
                "eligible": eligible,
                "selection_rule": "minimum maturity-eligible development fitted-only log loss; tie-break by smaller alpha then smaller temperature cap",
            }
            candidate_rows.append(row)

    eligible_candidates = [row for row in candidate_rows if row["eligible"]]
    if not eligible_candidates:
        raise RuntimeError(
            f"{name}: no joint alpha/temperature candidate has at least "
            f"{min_development_fitted_observations} mature fitted development observations."
        )
    chosen_row = min(
        eligible_candidates,
        key=lambda item: (
            float(item["selection_log_loss"]),
            float(item["alpha"]),
            float(item["temperature_max"]),
        ),
    )
    chosen_alpha = float(chosen_row["alpha"])
    chosen_cap = float(chosen_row["temperature_max"])
    chosen = candidate_cache[(chosen_alpha, chosen_cap)]
    selected_raw = chosen["raw"]
    selected_cal = chosen["calibration"].probabilities_pct

    # Diagnostic references: unsmoothed with the selected cap, and unsmoothed
    # with the current cap 4. These are not used to select the chosen pair.
    cap_grid = _grid_size_for_cap(temperature_min, chosen_cap, reference_temperature_max, reference_grid_size)
    unsmoothed_same_cap = _calibrate_surface(
        rows,
        source_field,
        alpha=0.0,
        temperature_min=temperature_min,
        temperature_max=chosen_cap,
        temperature_grid_size=cap_grid,
        min_calibration_observations=min_calibration_observations,
    )
    unsmoothed_ref_cap = _calibrate_surface(
        rows,
        source_field,
        alpha=0.0,
        temperature_min=temperature_min,
        temperature_max=reference_temperature_max,
        temperature_grid_size=reference_grid_size,
        min_calibration_observations=min_calibration_observations,
    )
    baseline = [normalize_distribution(row.baseline_probabilities_pct) for row in rows]

    holdout_available = [i for i in holdout_indices if selected_raw[i] is not None]
    holdout_baseline_paired = [i for i in holdout_indices if selected_cal[i] is not None and baseline[i] is not None]
    if not holdout_available:
        raise RuntimeError(f"{name}: no forecasts available in holdout period.")

    print(f"\n{name}")
    print(f"  vectors available: {len(available_indices)}/{len(rows)}")
    print(f"  development rows: {len(development_indices)}; holdout rows: {len(holdout_indices)}")
    print(f"  joint development selection cutoff: outcomes must mature strictly before {holdout_start}")
    print("  Joint development candidates (fitted forecasts with mature labels only):")
    for candidate in candidate_rows:
        metric = candidate["selection_log_loss"]
        print(
            f"    alpha={candidate['alpha']:g}, T_max={candidate['temperature_max']:g}, "
            f"grid={candidate['temperature_grid_size']}, n={candidate['selection_observations']}, "
            f"log_loss={metric:.5f}" if metric is not None else
            f"    alpha={candidate['alpha']:g}, T_max={candidate['temperature_max']:g}, "
            f"grid={candidate['temperature_grid_size']}, n={candidate['selection_observations']}, log_loss=n/a",
            end="",
        )
        print(" [ELIGIBLE]" if candidate["eligible"] else " [INELIGIBLE]")
    print(f"  Selected joint pair from development only: alpha={chosen_alpha:g}, T_max={chosen_cap:g}")

    cal = chosen["calibration"]
    status_by_period = {
        "development": dict(Counter(cal.statuses[i] for i in development_indices if source[i] is not None)),
        "holdout": dict(Counter(cal.statuses[i] for i in holdout_indices if source[i] is not None)),
    }
    boundary_by_period = {}
    for period_name, period_indices in (("development", development_indices), ("holdout", holdout_indices)):
        selected_temperatures = [cal.temperatures[i] for i in period_indices]
        selected_statuses = [cal.statuses[i] for i in period_indices]
        boundary_by_period[period_name] = summarize_temperature_boundaries(
            selected_temperatures,
            selected_statuses,
            temperature_min=temperature_min,
            temperature_max=chosen_cap,
        )

    print("  Holdout scores:")
    scores = {
        "selected_alpha_raw_smoothed": _summary_score(name + " selected raw-smoothed", actual, selected_raw, holdout_indices),
        "selected_alpha_prequential_calibrated": _summary_score(name + " selected prequential calibrated", actual, selected_cal, holdout_indices),
        "unsmoothed_raw": _summary_score(name + " unsmoothed raw", actual, unsmoothed_same_cap["raw"], holdout_indices),
        "unsmoothed_calibrated_same_cap": _summary_score(name + " unsmoothed calibrated at selected cap", actual, unsmoothed_same_cap["calibration"].probabilities_pct, holdout_indices),
        "unsmoothed_calibrated_cap_4": _summary_score(name + " unsmoothed calibrated at cap 4", actual, unsmoothed_ref_cap["calibration"].probabilities_pct, holdout_indices),
        "stored_baseline": _summary_score(name + " stored baseline", actual, baseline, holdout_baseline_paired),
    }

    comparisons = {
        "selected_calibrated_vs_unsmoothed_calibrated_same_cap": _compare(
            actual, selected_cal, unsmoothed_same_cap["calibration"].probabilities_pct, holdout_indices,
            block_length=block_length, replicates=bootstrap_replicates, seed=seed_base + 1,
        ),
        "selected_calibrated_vs_unsmoothed_calibrated_cap_4": _compare(
            actual, selected_cal, unsmoothed_ref_cap["calibration"].probabilities_pct, holdout_indices,
            block_length=block_length, replicates=bootstrap_replicates, seed=seed_base + 2,
        ),
        "selected_calibrated_vs_stored_baseline": _compare(
            actual, selected_cal, baseline, holdout_indices,
            block_length=block_length, replicates=bootstrap_replicates, seed=seed_base + 3,
        ),
    }
    print("  Paired holdout comparisons (positive log-loss/Brier gain favors candidate):")
    for title, result in comparisons.items():
        if result is None:
            print(f"    {title}: unavailable")
        else:
            print_paired_comparison(title.replace("_", " "), result)

    return {
        "status": "EVALUATED" if len(holdout_available) >= 10 else "TOO_FEW_HOLDOUT_OBSERVATIONS_FOR_ROBUST_INFERENCE",
        "surface": name,
        "available_observations": len(available_indices),
        "development_rows": len(development_indices),
        "holdout_rows": len(holdout_indices),
        "holdout_available_observations": len(holdout_available),
        "development_start_date": str(rows[development_indices[0]].prediction_date) if development_indices else None,
        "development_end_date": str(rows[development_indices[-1]].prediction_date) if development_indices else None,
        "holdout_start_date": str(holdout_start),
        "holdout_end_date": str(rows[holdout_indices[-1]].prediction_date) if holdout_indices else None,
        "outcome_maturity_cutoff_exclusive": str(holdout_start),
        "selected_parameters": {
            "alpha": chosen_alpha,
            "temperature_min": temperature_min,
            "temperature_max": chosen_cap,
            "temperature_grid_size": chosen["temperature_grid_size"],
            "selection_metric": "prequential calibrated log loss on development fitted rows with outcomes matured strictly before holdout start",
            "selected_development_log_loss": chosen_row["selection_log_loss"],
            "selected_development_brier_score": chosen_row["selection_brier_score"],
            "selected_development_observations": chosen_row["selection_observations"],
            "all_candidates": candidate_rows,
        },
        "calibration_status_counts": status_by_period,
        "temperature_boundary_diagnostics": boundary_by_period,
        "holdout_scores": scores,
        "paired_comparisons": comparisons,
        "holdout_calibration_protocol": "PREQUENTIAL_PRIOR_OUTCOMES_MATURED_STRICTLY_BEFORE_CURRENT_FORECAST; ALPHA_AND_CAP_FIXED_FROM_DEVELOPMENT_SELECTION",
        "warning": "This is a post-hoc sensitivity study of previously examined Phase 5.8 folds, not a fresh untouched evaluation of the original model-development pipeline.",
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Jointly select smoothing alpha and temperature cap on development-only matured labels; evaluate once on later folds. No OLAP."
    )
    parser.add_argument("snapshot")
    parser.add_argument("--alphas", nargs="+", type=float, default=list(_DEFAULT_ALPHAS))
    parser.add_argument("--temperature-min", type=float, default=0.25)
    parser.add_argument("--temperature-max-values", nargs="+", type=float, default=list(_DEFAULT_TEMP_MAX))
    parser.add_argument("--reference-temperature-max", type=float, default=4.0)
    parser.add_argument("--reference-grid-size", type=int, default=161)
    parser.add_argument("--development-fraction", type=float, default=0.70)
    parser.add_argument("--min-calibration-observations", type=int, default=30)
    parser.add_argument("--min-development-fitted-observations", type=int, default=20)
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    alphas = tuple(sorted(set(float(value) for value in args.alphas)))
    maxima = tuple(sorted(set(float(value) for value in args.temperature_max_values)))
    if not any(abs(value) < 1e-15 for value in alphas):
        parser.error("--alphas must include 0.0 as the unsmoothed reference")
    if any(not math.isfinite(value) or value < 0.0 for value in alphas):
        parser.error("all alpha values must be finite and non-negative")
    if not 0.0 < args.temperature_min < args.reference_temperature_max:
        parser.error("require 0 < --temperature-min < --reference-temperature-max")
    if args.reference_temperature_max not in maxima or any(value <= args.temperature_min for value in maxima):
        parser.error("temperature maximums must include the reference cap and exceed temperature minimum")
    if args.reference_grid_size < 3 or args.min_calibration_observations < 1 or args.min_development_fitted_observations < 1:
        parser.error("grid size and minimum observation counts must be positive (grid size >=3)")
    if args.block_length < 1 or args.bootstrap_replicates < 100:
        parser.error("block length must be >=1 and bootstrap replicates >=100")
    if not 0.5 <= args.development_fraction < 1.0:
        parser.error("development fraction must be in [0.5, 1.0)")

    from phase6_2_method_probability_audit import load_folds
    from phase6_2_probability_smoothing_sensitivity import probability_zero_diagnostics

    rows = load_folds(args.snapshot)
    if len(rows) < 20:
        raise RuntimeError(f"Only {len(rows)} combined-usable folds; insufficient for joint sensitivity.")
    dates = [row.prediction_date for row in rows]
    split_index = min(max(int(math.floor(len(rows) * args.development_fraction)), 1), len(rows) - 1)
    split_date = rows[split_index].prediction_date
    # Use the same calendar cut for all surfaces. Method availability only
    # changes each method's n, not the chronological split boundary.
    combined_zero_audit = probability_zero_diagnostics(rows, "combined_probabilities_pct", top_cases=10)
    a_zero_audit = probability_zero_diagnostics(rows, "method_a_probabilities_pct", top_cases=10, selection_status_field="method_a_selection_status")
    b_zero_audit = probability_zero_diagnostics(rows, "method_b_probabilities_pct", top_cases=10, selection_status_field="method_b_selection_status")

    print("PHASE 6.2 CONSOLIDATED JOINT SMOOTHING / TEMPERATURE VALIDATION")
    print(f"Snapshot: {args.snapshot}")
    print("Snapshot status: FROZEN")
    print(f"Total combined-usable folds: {len(rows)}")
    print(f"Dates: {dates[0]} through {dates[-1]}")
    print("Protocol: FROZEN_FOLDS; NO_OLAP_RECOMPUTATION")
    print("Calibrator: prequential temperature scaling with strict outcome-maturity gate")
    print(f"Development: {split_index} global folds before {split_date}; holdout: {len(rows)-split_index} folds from {split_date}")
    print(f"Smoothing alpha candidates: {alphas}")
    print(f"Temperature max candidates: {maxima}; minimum={args.temperature_min:g}; reference cap={args.reference_temperature_max:g}")
    print(f"Moving-block bootstrap: block_length={args.block_length}; replicates={args.bootstrap_replicates}")
    print("WARNING: these historical holdout folds were already inspected in prior Phase 6.2 experiments; results are post-hoc sensitivity, not a fresh untouched model-development holdout.")

    print("\nProbability-zero audit, before any smoothing:")
    for name, audit in (("Combined", combined_zero_audit), ("Method A", a_zero_audit), ("Method B", b_zero_audit)):
        true = audit.get("true_class_probability") or {}
        print(
            f"  {name}: availability={audit.get('status')} "
            f"{audit.get('available_observations', 0)}/{audit.get('total_rows', 0)}, "
            f"exact zero entries={audit.get('exact_zero_entries_total', 0)}, "
            f"true-class zeros={true.get('exact_zero_true_class_count', 0)} "
            f"({true.get('exact_zero_true_class_pct', 0.0):.2f}%)"
        )

    surfaces = (
        ("Combined model", "combined_probabilities_pct", 20261011),
        ("Method A", "method_a_probabilities_pct", 20261012),
        ("Method B", "method_b_probabilities_pct", 20261013),
    )
    reports = {}
    for name, field, seed in surfaces:
        available_rows = _available_surface_rows(rows, field)
        if not available_rows:
            reports[name] = {"status": "UNAVAILABLE", "available_observations": 0}
            print(f"\n{name}: no usable frozen probability vectors; skipped without reconstruction.")
            continue
        try:
            reports[name] = _analyze_surface(
                name,
                available_rows,
                field,
                global_split_date=split_date,
                alphas=alphas,
                temperature_max_values=maxima,
                temperature_min=args.temperature_min,
                reference_temperature_max=args.reference_temperature_max,
                reference_grid_size=args.reference_grid_size,
                min_calibration_observations=args.min_calibration_observations,
                min_development_fitted_observations=args.min_development_fitted_observations,
                block_length=args.block_length,
                bootstrap_replicates=args.bootstrap_replicates,
                seed_base=seed,
            )
        except RuntimeError as exc:
            reports[name] = {"status": "NOT_EVALUATED", "reason": str(exc), "available_observations": len(available_rows)}
            print(f"\n{name}: NOT EVALUATED: {exc}")

    report = {
        "snapshot": args.snapshot,
        "status": "FROZEN",
        "folds": len(rows),
        "date_start": str(dates[0]),
        "date_end": str(dates[-1]),
        "split_index_global": split_index,
        "split_date_exclusive_development": str(split_date),
        "alphas": list(alphas),
        "temperature_min": args.temperature_min,
        "temperature_max_values": list(maxima),
        "reference_temperature_max": args.reference_temperature_max,
        "reference_grid_size": args.reference_grid_size,
        "minimum_calibration_observations": args.min_calibration_observations,
        "minimum_development_fitted_observations": args.min_development_fitted_observations,
        "block_length": args.block_length,
        "bootstrap_replicates": args.bootstrap_replicates,
        "probability_zero_audits": {"Combined": combined_zero_audit, "Method A": a_zero_audit, "Method B": b_zero_audit},
        "surfaces": reports,
        "guardrails": [
            "Each pair (alpha, temperature cap) is selected only on development forecasts with fitted calibration and outcomes mature strictly before holdout start.",
            "During holdout, alpha and the cap are fixed; only the prequential calibrator updates using previous outcomes matured before the current prediction date.",
            "The inspected holdout is post-hoc relative to earlier analysis and is not a fresh untouched test of the original model-development pipeline.",
            "A parameter pair's lower log loss does not imply better directional accuracy or profitability.",
            "No stored Phase 5.8 probability, outcome, selection field, manifest or parquet is changed; no OLAP query is issued.",
        ],
    }
    safe = _json_safe(report)
    if args.json_out:
        destination = Path(args.json_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(safe, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nJSON report written: {destination}")
    print("\nInterpretation guardrails:")
    for item in report["guardrails"]:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
