from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any

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
class SnapshotFold:
    prediction_date: date
    probabilities_pct: dict[str, float]
    actual_class: str
    predicted_trend: str
    method_a_trend: str | None
    method_b_trend: str | None
    baseline_probabilities_pct: dict[str, float]
    method_a_probabilities_pct: dict[str, float] | None = None
    method_b_probabilities_pct: dict[str, float] | None = None
    outcome_end_date: date | None = None


def _date(value: Any) -> date:
    result = as_date(value)
    if result is None:
        raise ValueError("Prediction date is required.")
    return result


def _distribution(raw: Any) -> dict[str, float] | None:
    return normalize_distribution(raw)


def _nested_method_distribution(row: dict[str, Any], method: str) -> dict[str, float] | None:
    candidates = [
        row.get(f"method_{method.lower()}_probabilities_pct"),
        row.get(f"{method.lower()}_probabilities_pct"),
    ]
    nested = row.get(f"method_{method.lower()}")
    if isinstance(nested, dict):
        candidates.extend([nested.get("probabilities_pct"), nested.get("calibrated_probabilities_pct")])
    output = row.get(f"method_{method.lower()}_output")
    if isinstance(output, dict):
        candidates.append(output.get("probabilities_pct"))
    for candidate in candidates:
        result = _distribution(candidate)
        if result is not None:
            return result
    return None


def load_folds(snapshot_root: str) -> list[SnapshotFold]:
    snapshot = load_phase5_snapshot(snapshot_root)
    rows: list[SnapshotFold] = []
    for row in snapshot.prediction_folds:
        actual = row.get("actual_class")
        probs = _distribution(row.get("probabilities_pct"))
        baseline = _distribution(row.get("baseline_probabilities_pct"))
        if actual not in OUTCOME_CLASSES or probs is None or baseline is None:
            continue
        rows.append(
            SnapshotFold(
                prediction_date=_date(row.get("prediction_date")),
                outcome_end_date=as_date(row.get("outcome_end_date")),
                probabilities_pct=probs,
                actual_class=actual,
                predicted_trend=str(row.get("predicted_trend", "NO_CLEAR_TREND")),
                method_a_trend=(row.get("method_a_trend") if row.get("method_a_trend") in OUTCOME_CLASSES else None),
                method_b_trend=(row.get("method_b_trend") if row.get("method_b_trend") in OUTCOME_CLASSES else None),
                baseline_probabilities_pct=baseline,
                method_a_probabilities_pct=_nested_method_distribution(row, "a"),
                method_b_probabilities_pct=_nested_method_distribution(row, "b"),
            )
        )
    rows.sort(key=lambda item: item.prediction_date)
    return rows


def _accuracy(pred: list[str], actual: list[str]) -> float:
    return sum(p == a for p, a in zip(pred, actual)) / len(actual) * 100.0 if actual else 0.0


def _class_counts(values: list[str]) -> dict[str, int]:
    return {label: values.count(label) for label in OUTCOME_CLASSES}


def _matrix(pred: list[str], actual: list[str]) -> dict[str, dict[str, int]]:
    return {
        actual_label: {
            predicted_label: sum(a == actual_label and p == predicted_label for p, a in zip(pred, actual))
            for predicted_label in OUTCOME_CLASSES
        }
        for actual_label in OUTCOME_CLASSES
    }


def _mean_distribution(rows: list[SnapshotFold], field: str = "probabilities_pct") -> dict[str, float]:
    return {label: float(np.mean([getattr(row, field)[label] for row in rows])) for label in OUTCOME_CLASSES}


def _mean_baseline(rows: list[SnapshotFold]) -> dict[str, float]:
    return {label: float(np.mean([row.baseline_probabilities_pct[label] for row in rows])) for label in OUTCOME_CLASSES}


def _actual_distribution(rows: list[SnapshotFold]) -> dict[str, float]:
    return {label: sum(row.actual_class == label for row in rows) / len(rows) * 100.0 for label in OUTCOME_CLASSES}


def _group_name(row: SnapshotFold) -> str:
    a, b = row.method_a_trend, row.method_b_trend
    if a and b:
        return "BOTH_AGREE" if a == b else "BOTH_CONFLICT"
    if a:
        return "A_ONLY"
    if b:
        return "B_ONLY"
    return "NEITHER"


def _print_surface(title: str, rows: list[SnapshotFold]) -> None:
    if not rows:
        print(f"  {title}: n=0")
        return
    mean = _mean_distribution(rows)
    observed = _actual_distribution(rows)
    baseline = _mean_baseline(rows)
    print(f"  {title}: n={len(rows)}")
    for label in OUTCOME_CLASSES:
        print(
            f"    {label:8s}: combined={mean[label]:6.2f}% baseline={baseline[label]:6.2f}% "
            f"observed={observed[label]:6.2f}% bias={mean[label]-observed[label]:+6.2f}pp"
        )
    argmax = [max(OUTCOME_CLASSES, key=lambda label: row.probabilities_pct[label]) for row in rows]
    actual = [row.actual_class for row in rows]
    print(f"    argmax counts: {_class_counts(argmax)}")
    print(f"    argmax accuracy: {_accuracy(argmax, actual):.2f}%")
    print(f"    confusion matrix (actual rows, predicted columns): {_matrix(argmax, actual)}")


def _surface_status(present: int, total: int) -> str:
    """Report partial availability accurately instead of treating it as absent."""
    if present <= 0:
        return "UNAVAILABLE"
    if present < total:
        return "PARTIAL"
    return "COMPLETE"


def _exact_method_surface_available(rows: list[SnapshotFold]) -> tuple[bool, bool]:
    return (
        bool(rows) and all(row.method_a_probabilities_pct is not None for row in rows),
        bool(rows) and all(row.method_b_probabilities_pct is not None for row in rows),
    )


def _print_combined_score_report(
    rows: list[SnapshotFold],
    *,
    min_calibration_observations: int,
    block_length: int,
    bootstrap_replicates: int,
) -> None:
    dates = [row.prediction_date for row in rows]
    ends = [row.outcome_end_date for row in rows]
    actual = [row.actual_class for row in rows]
    raw = [row.probabilities_pct for row in rows]
    baseline = [row.baseline_probabilities_pct for row in rows]
    calibration = calibrate_with_mature_labels(
        dates, ends, actual, raw,
        min_calibration_observations=min_calibration_observations,
    )
    calibrated = [item for item in calibration.probabilities_pct if item is not None]
    print("Combined-model proper scoring (fold-stored baseline):")
    print_score_line("raw combined", score_probabilities(actual, raw))
    print_score_line("maturity-gated prequential calibrated", score_probabilities(actual, calibrated))
    print_score_line("stored per-fold baseline", score_probabilities(actual, baseline))
    print(f"  calibration statuses: {dict(Counter(calibration.statuses))}")
    print(f"  mature training history sizes: min={min(calibration.mature_training_observations)}, "
          f"median={float(np.median(calibration.mature_training_observations)):.1f}, "
          f"max={max(calibration.mature_training_observations)}")
    print_paired_comparison(
        "raw combined versus stored baseline",
        compare_probabilities_paired(
            actual, raw, baseline,
            block_length=block_length,
            replicates=bootstrap_replicates,
            seed=20261010,
        ),
    )
    print_paired_comparison(
        "calibrated combined versus stored baseline",
        compare_probabilities_paired(
            actual, calibrated, baseline,
            block_length=block_length,
            replicates=bootstrap_replicates,
            seed=20261110,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Phase 6.2 decomposition of the frozen combined forecast by A/B availability, "
            "agreement and conflict, with honest partial-vector status and scoring metrics."
        )
    )
    parser.add_argument("snapshot")
    parser.add_argument("--min-calibration-observations", type=int, default=30)
    parser.add_argument("--block-length", type=int, default=5,
                        help="Moving-block length in evaluated folds; test sensitivity at alternative values.")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    if args.block_length < 1 or args.bootstrap_replicates < 100:
        parser.error("--block-length must be >=1 and --bootstrap-replicates must be >=100")

    rows = load_folds(args.snapshot)
    if not rows:
        raise RuntimeError("Snapshot contains no usable folds.")

    groups: dict[str, list[SnapshotFold]] = {
        name: [] for name in ("BOTH_AGREE", "BOTH_CONFLICT", "A_ONLY", "B_ONLY", "NEITHER")
    }
    for row in rows:
        groups[_group_name(row)].append(row)

    a_present = sum(row.method_a_probabilities_pct is not None for row in rows)
    b_present = sum(row.method_b_probabilities_pct is not None for row in rows)
    a_complete, b_complete = _exact_method_surface_available(rows)

    print("PHASE 6.2 METHOD-A/B PROBABILITY-SURFACE DECOMPOSITION")
    print(f"Snapshot: {args.snapshot}")
    print("Snapshot status: FROZEN")
    print(f"Folds: {len(rows)}")
    print(f"Dates: {rows[0].prediction_date} through {rows[-1].prediction_date}")
    print("Protocol: FROZEN_PHASE5_8_METHOD_AVAILABILITY_AND_SURFACE_DECOMPOSITION")
    print("Calibration protocol: PREQUENTIAL_ONLY_OUTCOMES_MATURED_BEFORE_PREDICTION")
    print("")

    _print_combined_score_report(
        rows,
        min_calibration_observations=args.min_calibration_observations,
        block_length=args.block_length,
        bootstrap_replicates=args.bootstrap_replicates,
    )

    print("")
    print("Combined probability surface reference:")
    raw_mean = _mean_distribution(rows)
    baseline_mean = _mean_baseline(rows)
    actual_freq = _actual_distribution(rows)
    for label in OUTCOME_CLASSES:
        print(
            f"  {label:8s}: raw={raw_mean[label]:6.2f}% baseline={baseline_mean[label]:6.2f}% "
            f"observed={actual_freq[label]:6.2f}% raw-bias={raw_mean[label]-actual_freq[label]:+6.2f}pp"
        )

    print("")
    print("Decomposition by Method A/B directional availability:")
    for name in ("BOTH_AGREE", "BOTH_CONFLICT", "A_ONLY", "B_ONLY", "NEITHER"):
        _print_surface(name, groups[name])

    print("")
    print("Method trend diagnostics within decomposition groups:")
    for name in ("BOTH_AGREE", "BOTH_CONFLICT", "A_ONLY", "B_ONLY", "NEITHER"):
        group = groups[name]
        if not group:
            continue
        a_rows = [row for row in group if row.method_a_trend in OUTCOME_CLASSES]
        b_rows = [row for row in group if row.method_b_trend in OUTCOME_CLASSES]
        print(f"  {name}:")
        if a_rows:
            print(f"    Method A n={len(a_rows)} accuracy={_accuracy([row.method_a_trend for row in a_rows], [row.actual_class for row in a_rows]):.2f}% counts={_class_counts([row.method_a_trend for row in a_rows])}")
        if b_rows:
            print(f"    Method B n={len(b_rows)} accuracy={_accuracy([row.method_b_trend for row in b_rows], [row.actual_class for row in b_rows]):.2f}% counts={_class_counts([row.method_b_trend for row in b_rows])}")

    print("")
    print("Stored Method A/B probability vectors:")
    print(f"  Method A: {a_present}/{len(rows)}; STATUS={_surface_status(a_present, len(rows))}")
    print(f"  Method B: {b_present}/{len(rows)}; STATUS={_surface_status(b_present, len(rows))}")
    if a_complete and b_complete:
        print("  Exact method-level vectors are COMPLETE for both methods.")
    elif a_present or b_present:
        print("  Exact method-level vectors are PARTIALLY available; available vectors are still auditable.")
        print("  Missing outputs remain missing and are not inferred from direction labels.")
    else:
        print("  Exact method-level vectors are UNAVAILABLE in this snapshot.")
    print("  A missing method vector is distinct from an absent selected relationship; consult selection metadata for the reason.")
    print("")
    print("Interpretation guardrails:")
    print("  - Grouped probability surfaces are descriptive, with subgroup sample sizes shown.")
    print("  - The stored per-fold baseline is scored fold by fold, not replaced by realized aggregate frequencies.")
    print("  - The moving-block interval is a dependence-aware uncertainty diagnostic; test sensitivity to block length.")
    print("  - No Phase 5.8 calculation, relationship search, threshold or decision rule is recomputed.")


if __name__ == "__main__":
    main()
