from __future__ import annotations

"""Read-only prequential calibration temperature-bound sensitivity audit.

This script reuses the frozen Phase 5.8 probability vectors. For each candidate
maximum temperature it refits only on past labels whose outcomes matured before
the current prediction date. It never runs relationship discovery or OLAP.

The full-period comparison is a sensitivity diagnostic, not permission to select
a calibration range after looking at the same evaluation outcomes.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from analysis.outcome_labels import OUTCOME_CLASSES
from analysis.phase6_probability_metrics import (
    calibrate_with_mature_labels,
    compare_probabilities_paired,
    normalize_distribution,
    print_paired_comparison,
    print_score_line,
    score_probabilities,
    summarize_temperature_boundaries,
)
from phase6_2_method_probability_audit import load_folds


METHODS = (
    ("Combined model", "combined_probabilities_pct"),
    ("Method A", "method_a_probabilities_pct"),
    ("Method B", "method_b_probabilities_pct"),
)
DEFAULT_MAXIMA = (4.0, 8.0, 16.0)
DEFAULT_REFERENCE_MAX = 4.0
DEFAULT_MIN = 0.25
DEFAULT_GRID_SIZE_AT_REFERENCE = 161


def grid_size_for_maximum(
    temperature_min: float,
    temperature_max: float,
    *,
    reference_max: float = DEFAULT_REFERENCE_MAX,
    reference_grid_size: int = DEFAULT_GRID_SIZE_AT_REFERENCE,
) -> int:
    """Keep approximately constant log-temperature grid spacing across caps."""
    if not 0.0 < temperature_min < temperature_max:
        raise ValueError("Require 0 < temperature_min < temperature_max.")
    if not 0.0 < temperature_min < reference_max:
        raise ValueError("reference_max must be greater than temperature_min.")
    if reference_grid_size < 3:
        raise ValueError("reference_grid_size must be >= 3.")
    reference_intervals = reference_grid_size - 1
    scaled_intervals = reference_intervals * (
        math.log(temperature_max / temperature_min)
        / math.log(reference_max / temperature_min)
    )
    return max(3, int(round(scaled_intervals)) + 1)


def _score_dict(
    rows: list[Any],
    indices: list[int],
    probability_rows: list[dict[str, float] | None],
) -> dict[str, Any] | None:
    if not indices:
        return None
    probabilities = [probability_rows[index] for index in indices]
    if any(item is None for item in probabilities):
        raise ValueError("Scoring indices include unavailable probability vectors.")
    return score_probabilities(
        [rows[index].actual_class for index in indices],
        [item for item in probabilities if item is not None],
    )


def _print_boundary_summary(boundary: dict[str, Any]) -> None:
    fitted = boundary["fitted_observations"]
    print(
        f"  fitted calibrators={fitted}; "
        f"mean T={boundary['temperature_mean'] if fitted else float('nan'):.4f}; "
        f"median T={boundary['temperature_median'] if fitted else float('nan'):.4f}"
    )
    print(
        "  lower-bound hits: "
        f"{boundary['lower_boundary_hits']}/{fitted} "
        f"({boundary['lower_boundary_hit_rate_pct'] if fitted else float('nan'):.2f}%)"
    )
    print(
        "  upper-bound hits: "
        f"{boundary['upper_boundary_hits']}/{fitted} "
        f"({boundary['upper_boundary_hit_rate_pct'] if fitted else float('nan'):.2f}%)"
    )
    print(
        "  any-boundary hits: "
        f"{boundary['any_boundary_hits']}/{fitted} "
        f"({boundary['any_boundary_hit_rate_pct'] if fitted else float('nan'):.2f}%)"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _run_one_method(
    rows: list[Any],
    method_name: str,
    probability_field: str,
    *,
    temperature_min: float,
    maximums: tuple[float, ...],
    reference_max: float,
    reference_grid_size: int,
    min_calibration_observations: int,
    block_length: int,
    bootstrap_replicates: int,
    seed_base: int,
) -> dict[str, Any]:
    dates = [row.prediction_date for row in rows]
    ends = [row.outcome_end_date for row in rows]
    actual = [row.actual_class for row in rows]
    raw = [
        normalize_distribution(getattr(row, probability_field))
        for row in rows
    ]
    baseline = [normalize_distribution(row.baseline_probabilities_pct) for row in rows]
    available_indices = [
        index for index, probability in enumerate(raw) if probability is not None
    ]
    baseline_paired_indices = [
        index for index in available_indices if baseline[index] is not None
    ]

    range_results: dict[float, dict[str, Any]] = {}
    print(f"\n{method_name}")
    print(f"  forecast vectors available: {len(available_indices)}/{len(rows)}")
    print(f"  stored baseline paired coverage: {len(baseline_paired_indices)}/{len(rows)}")
    if not available_indices:
        print("  STATUS: no usable forecasts stored; sensitivity cannot be estimated.")
        return {"available": 0, "ranges": {}, "range_comparisons": {}}

    for maximum in maximums:
        grid_size = grid_size_for_maximum(
            temperature_min,
            maximum,
            reference_max=reference_max,
            reference_grid_size=reference_grid_size,
        )
        calibration = calibrate_with_mature_labels(
            dates,
            ends,
            actual,
            raw,
            min_calibration_observations=min_calibration_observations,
            temperature_min=temperature_min,
            temperature_max=maximum,
            temperature_grid_size=grid_size,
        )
        calibrated = list(calibration.probabilities_pct)
        usable = [
            index for index, value in enumerate(calibrated)
            if raw[index] is not None and value is not None
        ]
        baseline_paired = [
            index for index in usable if baseline[index] is not None
        ]
        fitted = [
            index for index in usable
            if calibration.statuses[index] == "FITTED"
        ]
        fitted_baseline = [
            index for index in fitted if baseline[index] is not None
        ]
        boundary = summarize_temperature_boundaries(
            calibration.temperatures,
            calibration.statuses,
            temperature_min=temperature_min,
            temperature_max=maximum,
        )

        print(f"\n  Temperature range [{temperature_min:g}, {maximum:g}] ")
        print(
            f"  grid points={grid_size}; status counts="
            f"{dict(Counter(calibration.statuses))}"
        )
        print(
            "  mature calibration history: "
            f"min={min(calibration.mature_training_observations)}, "
            f"median={float(np.median(calibration.mature_training_observations)):.1f}, "
            f"max={max(calibration.mature_training_observations)}"
        )
        _print_boundary_summary(boundary)

        raw_all = _score_dict(rows, usable, raw)
        cal_all = _score_dict(rows, usable, calibrated)
        base_all = _score_dict(
            rows,
            baseline_paired,
            [item if item is not None else None for item in baseline],
        )
        if raw_all is not None:
            print_score_line("raw, all available folds", raw_all)
        if cal_all is not None:
            print_score_line("prequential calibrated, all available folds", cal_all)
        if base_all is not None:
            print_score_line("stored baseline, same available folds", base_all)
        raw_vs_baseline = None
        calibrated_vs_baseline = None
        if baseline_paired:
            raw_vs_baseline = compare_probabilities_paired(
                [actual[index] for index in baseline_paired],
                [raw[index] for index in baseline_paired if raw[index] is not None],
                [baseline[index] for index in baseline_paired if baseline[index] is not None],
                block_length=block_length,
                replicates=bootstrap_replicates,
                seed=seed_base + int(maximum * 100),
            )
            calibrated_vs_baseline = compare_probabilities_paired(
                [actual[index] for index in baseline_paired],
                [calibrated[index] for index in baseline_paired if calibrated[index] is not None],
                [baseline[index] for index in baseline_paired if baseline[index] is not None],
                block_length=block_length,
                replicates=bootstrap_replicates,
                seed=seed_base + 1000 + int(maximum * 100),
            )
            print_paired_comparison("raw forecast versus stored baseline", raw_vs_baseline)
            print_paired_comparison("calibrated forecast versus stored baseline", calibrated_vs_baseline)

        fitted_raw = _score_dict(rows, fitted, raw)
        fitted_cal = _score_dict(rows, fitted, calibrated)
        fitted_base = _score_dict(
            rows,
            fitted_baseline,
            [item if item is not None else None for item in baseline],
        )
        print(f"  fitted-only forecast subset: n={len(fitted)}")
        if fitted_raw is not None:
            print_score_line("raw, fitted-only subset", fitted_raw)
        if fitted_cal is not None:
            print_score_line("calibrated, fitted-only subset", fitted_cal)
        print(f"  fitted-only baseline-paired subset: n={len(fitted_baseline)}")
        if fitted_base is not None:
            print_score_line("stored baseline, fitted-only matched subset", fitted_base)
        fitted_calibrated_vs_baseline = None
        if fitted_baseline:
            fitted_calibrated_vs_baseline = compare_probabilities_paired(
                [actual[index] for index in fitted_baseline],
                [calibrated[index] for index in fitted_baseline if calibrated[index] is not None],
                [baseline[index] for index in fitted_baseline if baseline[index] is not None],
                block_length=block_length,
                replicates=bootstrap_replicates,
                seed=seed_base + 2000 + int(maximum * 100),
            )
            print_paired_comparison(
                "calibrated forecast versus stored baseline, fitted-only matched subset",
                fitted_calibrated_vs_baseline,
            )

        range_results[maximum] = {
            "temperature_min": temperature_min,
            "temperature_max": maximum,
            "temperature_grid_size": grid_size,
            "availability": len(available_indices),
            "usable_calibrated": len(usable),
            "baseline_paired": len(baseline_paired),
            "status_counts": dict(Counter(calibration.statuses)),
            "mature_history_min": min(calibration.mature_training_observations),
            "mature_history_median": float(np.median(calibration.mature_training_observations)),
            "mature_history_max": max(calibration.mature_training_observations),
            "boundary": boundary,
            "raw_all_available": raw_all,
            "calibrated_all_available": cal_all,
            "baseline_same_available": base_all,
            "raw_vs_baseline": raw_vs_baseline,
            "calibrated_vs_baseline": calibrated_vs_baseline,
            "fitted_only": {
                "observations": len(fitted),
                "baseline_paired_observations": len(fitted_baseline),
                "raw": fitted_raw,
                "calibrated": fitted_cal,
                "baseline_on_matched_subset": fitted_base,
                "calibrated_vs_baseline_on_matched_subset": fitted_calibrated_vs_baseline,
            },
            "calibrated_probabilities": calibrated,
            "calibration_statuses": list(calibration.statuses),
            "temperatures": list(calibration.temperatures),
            "available_indices": usable,
            "fitted_indices": fitted,
            "fitted_baseline_indices": fitted_baseline,
        }

    print("\n  Range sensitivity comparisons (common fitted folds; positive favors wider cap)")
    comparisons: dict[str, Any] = {}
    if reference_max not in range_results:
        raise ValueError("Reference maximum must be present in temperature maximum candidates.")
    for maximum in maximums:
        if maximum == reference_max:
            continue
        reference = range_results[reference_max]
        candidate = range_results[maximum]
        common_fitted = sorted(
            set(reference["fitted_indices"]) & set(candidate["fitted_indices"])
        )
        key = f"max_{maximum:g}_versus_{reference_max:g}"
        print(f"    candidate max={maximum:g} versus reference max={reference_max:g}: n={len(common_fitted)}")
        if not common_fitted:
            comparisons[key] = {"observations": 0, "comparison": None}
            print("      no common fitted observations")
            continue
        comparison = compare_probabilities_paired(
            [actual[index] for index in common_fitted],
            [candidate["calibrated_probabilities"][index] for index in common_fitted],
            [reference["calibrated_probabilities"][index] for index in common_fitted],
            block_length=block_length,
            replicates=bootstrap_replicates,
            seed=seed_base + 9000 + int(maximum * 100),
        )
        print_paired_comparison(
            f"max {maximum:g} versus max {reference_max:g}", comparison
        )
        comparisons[key] = {"observations": len(common_fitted), "comparison": comparison}

    # Drop aligned per-fold arrays from compact summaries? Keep them in JSON so
    # sensitivity runs can be independently inspected and reproduced.
    return {
        "available": len(available_indices),
        "ranges": range_results,
        "range_comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only prequential calibration boundary and temperature-cap sensitivity "
            "using the frozen Phase 5.8 snapshot."
        )
    )
    parser.add_argument("snapshot", help="Frozen Phase 5.8 snapshot directory; no OLAP queries are executed.")
    parser.add_argument("--temperature-min", type=float, default=DEFAULT_MIN)
    parser.add_argument(
        "--temperature-max-values", nargs="+", type=float, default=list(DEFAULT_MAXIMA),
        help="Candidate maximum temperatures; defaults to current 4 and sensitivity caps 8 and 16.",
    )
    parser.add_argument("--reference-temperature-max", type=float, default=DEFAULT_REFERENCE_MAX)
    parser.add_argument("--grid-size-at-reference", type=int, default=DEFAULT_GRID_SIZE_AT_REFERENCE)
    parser.add_argument("--min-calibration-observations", type=int, default=30)
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--json-out", help="Optional path to write the full sensitivity result as JSON.")
    args = parser.parse_args()

    maximums = tuple(sorted(set(float(value) for value in args.temperature_max_values)))
    if not 0.0 < args.temperature_min < args.reference_temperature_max:
        parser.error("Require 0 < --temperature-min < --reference-temperature-max.")
    if args.reference_temperature_max not in maximums:
        parser.error("--reference-temperature-max must appear in --temperature-max-values.")
    if any(value <= args.temperature_min for value in maximums):
        parser.error("Every candidate maximum must exceed --temperature-min.")
    if args.grid_size_at_reference < 3:
        parser.error("--grid-size-at-reference must be >= 3.")
    if args.min_calibration_observations < 1:
        parser.error("--min-calibration-observations must be >= 1.")
    if args.block_length < 1 or args.bootstrap_replicates < 100:
        parser.error("--block-length must be >=1 and --bootstrap-replicates must be >=100.")

    rows = load_folds(args.snapshot)
    if not rows:
        raise RuntimeError("Snapshot contains no usable folds.")

    dates = [row.prediction_date for row in rows]
    print("PHASE 6.2 CALIBRATION BOUNDARY AND RANGE SENSITIVITY")
    print(f"Snapshot: {args.snapshot}")
    print("Snapshot status: FROZEN")
    print(f"Folds: {len(rows)}")
    print(f"Dates: {dates[0]} through {dates[-1]}")
    print("Protocol: FROZEN_FOLDS; NO_OLAP_RECOMPUTATION")
    print("Calibration: prequential; only outcomes matured strictly before each prediction date")
    print(f"Minimum mature calibration observations: {args.min_calibration_observations}")
    print(f"Temperature minimum: {args.temperature_min:g}")
    print(f"Candidate maximums: {list(maximums)}; reference max={args.reference_temperature_max:g}")
    print(
        "Grid density: approximately matched in log-temperature space; "
        f"reference grid size={args.grid_size_at_reference}"
    )
    print(f"Moving-block bootstrap: block_length={args.block_length}; replicates={args.bootstrap_replicates}")
    print("\nWARNING: this is a post-hoc sensitivity report. Do not choose the production cap from the full evaluation period without a separate, predeclared holdout.")

    report: dict[str, Any] = {
        "snapshot": args.snapshot,
        "status": "FROZEN",
        "folds": len(rows),
        "date_start": str(dates[0]),
        "date_end": str(dates[-1]),
        "folds_metadata": [
            {
                "index": index,
                "prediction_date": str(row.prediction_date),
                "outcome_end_date": str(row.outcome_end_date) if row.outcome_end_date is not None else None,
                "actual_class": row.actual_class,
                "method_a_available": row.method_a_probabilities_pct is not None,
                "method_b_available": row.method_b_probabilities_pct is not None,
            }
            for index, row in enumerate(rows)
        ],
        "protocol": "PREQUENTIAL_OUTCOME_MATURITY_GATED_TEMPERATURE_BOUND_SENSITIVITY",
        "temperature_min": args.temperature_min,
        "temperature_max_values": list(maximums),
        "reference_temperature_max": args.reference_temperature_max,
        "grid_size_at_reference": args.grid_size_at_reference,
        "minimum_calibration_observations": args.min_calibration_observations,
        "block_length": args.block_length,
        "bootstrap_replicates": args.bootstrap_replicates,
        "methods": {},
    }
    for method_index, (method_name, field) in enumerate(METHODS):
        result = _run_one_method(
            rows,
            method_name,
            field,
            temperature_min=args.temperature_min,
            maximums=maximums,
            reference_max=args.reference_temperature_max,
            reference_grid_size=args.grid_size_at_reference,
            min_calibration_observations=args.min_calibration_observations,
            block_length=args.block_length,
            bootstrap_replicates=args.bootstrap_replicates,
            seed_base=20261010 + method_index * 100000,
        )
        # JSON outputs should be useful as archival reports but compact enough to
        # avoid duplicating all per-fold probabilities. Keep temperatures/statuses
        # and metrics, omit calibrated vectors and index lists after comparisons.
        compact_ranges: dict[str, Any] = {}
        for maximum, range_result in result.get("ranges", {}).items():
            compact = dict(range_result)
            compact.pop("calibrated_probabilities", None)
            compact.pop("available_indices", None)
            compact.pop("fitted_indices", None)
            compact.pop("fitted_baseline_indices", None)
            compact_ranges[str(maximum)] = _json_safe(compact)
        report["methods"][method_name] = {
            "available": result.get("available", 0),
            "ranges": compact_ranges,
            "range_comparisons": _json_safe(result.get("range_comparisons", {})),
        }

    if args.json_out:
        destination = Path(args.json_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
        print(f"\nJSON report written: {destination}")

    print("\nInterpretation guardrails:")
    print("  - Only stored fold forecasts, actual classes, outcome-end dates and baseline vectors are read.")
    print("  - A temperature at the cap is a diagnostic that the searched optimum may lie beyond the tested range.")
    print("  - A wider cap is not automatically better; paired scoring and baseline comparisons must be reviewed.")
    print("  - Sensitivity comparisons are not a replacement for a separate model-selection holdout.")
    print("  - No Phase 5.8 calculation, relationship selection, or OLAP query is repeated.")


if __name__ == "__main__":
    main()
