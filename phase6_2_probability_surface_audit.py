from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from analysis.phase5_snapshot import load_phase5_snapshot
from analysis.outcome_labels import OUTCOME_CLASSES
from analysis.phase6_calibration import calibrate_phase5_folds


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


def _date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _distribution(raw: Any) -> dict[str, float] | None:
    if not isinstance(raw, dict):
        return None
    try:
        values = {label: max(float(raw.get(label, 0.0)), 0.0) for label in OUTCOME_CLASSES}
    except (TypeError, ValueError):
        return None
    total = sum(values.values())
    if not np.isfinite(total) or total <= 0:
        return None
    return {label: values[label] / total * 100.0 for label in OUTCOME_CLASSES}


def _nested_method_distribution(row: dict[str, Any], method: str) -> dict[str, float] | None:
    # Future-compatible: accept either nested method summary or explicit fold-level fields.
    candidates = [
        row.get(f"method_{method.lower()}_probabilities_pct"),
        row.get(f"{method.lower()}_probabilities_pct"),
    ]
    nested = row.get(f"method_{method.lower()}")
    if isinstance(nested, dict):
        candidates.extend(
            [
                nested.get("probabilities_pct"),
                nested.get("calibrated_probabilities_pct"),
            ]
        )
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
        p: {a: sum(pp == p and aa == a for pp, aa in zip(pred, actual) for _ in [0]) for a in OUTCOME_CLASSES}
        for p in OUTCOME_CLASSES
    }


def _mean_distribution(rows: list[SnapshotFold], field: str = "probabilities_pct") -> dict[str, float]:
    return {
        label: float(np.mean([getattr(row, field)[label] for row in rows]))
        for label in OUTCOME_CLASSES
    }


def _mean_baseline(rows: list[SnapshotFold]) -> dict[str, float]:
    return {
        label: float(np.mean([row.baseline_probabilities_pct[label] for row in rows]))
        for label in OUTCOME_CLASSES
    }


def _actual_distribution(rows: list[SnapshotFold]) -> dict[str, float]:
    return {
        label: sum(row.actual_class == label for row in rows) / len(rows) * 100.0
        for label in OUTCOME_CLASSES
    }


def _group_name(row: SnapshotFold) -> str:
    a = row.method_a_trend
    b = row.method_b_trend
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
            f"    {label:8s}: combined={mean[label]:6.2f}% "
            f"baseline={baseline[label]:6.2f}% observed={observed[label]:6.2f}% "
            f"bias={mean[label]-observed[label]:+6.2f}pp"
        )
    argmax = [max(OUTCOME_CLASSES, key=lambda label: row.probabilities_pct[label]) for row in rows]
    actual = [row.actual_class for row in rows]
    print(f"    argmax counts: {_class_counts(argmax)}")
    print(f"    argmax accuracy: {_accuracy(argmax, actual):.2f}%")


def _exact_method_surface_available(rows: list[SnapshotFold]) -> tuple[bool, bool]:
    return (
        all(row.method_a_probabilities_pct is not None for row in rows),
        all(row.method_b_probabilities_pct is not None for row in rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6.2 read-only decomposition of the frozen Phase 5.8 probability surface "
            "by Method A/B availability, agreement, conflict, and optional stored method vectors."
        )
    )
    parser.add_argument("snapshot")
    parser.add_argument("--min-calibration-observations", type=int, default=30)
    args = parser.parse_args()

    rows = load_folds(args.snapshot)
    if not rows:
        raise RuntimeError("Snapshot contains no usable folds.")

    calibrated = calibrate_phase5_folds(
        rows,
        min_calibration_observations=args.min_calibration_observations,
    )
    calibrated_by_date = {item.prediction_date: item for item in calibrated.predictions}

    actual = [row.actual_class for row in rows]
    raw_mean = _mean_distribution(rows)
    baseline_mean = _mean_baseline(rows)
    actual_freq = _actual_distribution(rows)
    calibrated_mean = {
        label: float(np.mean([calibrated_by_date[row.prediction_date].calibrated_probabilities_pct[label] for row in rows]))
        for label in OUTCOME_CLASSES
    }

    groups: dict[str, list[SnapshotFold]] = {name: [] for name in ("BOTH_AGREE", "BOTH_CONFLICT", "A_ONLY", "B_ONLY", "NEITHER")}
    for row in rows:
        groups[_group_name(row)].append(row)

    a_exact, b_exact = _exact_method_surface_available(rows)
    a_present = sum(row.method_a_probabilities_pct is not None for row in rows)
    b_present = sum(row.method_b_probabilities_pct is not None for row in rows)

    print("PHASE 6.2 METHOD-A/B PROBABILITY-SURFACE DECOMPOSITION")
    print(f"Snapshot: {args.snapshot}")
    print("Snapshot status: FROZEN")
    print(f"Folds: {len(rows)}")
    print("Protocol: FROZEN_PHASE5_8_METHOD_AVAILABILITY_AND_SURFACE_DECOMPOSITION")

    print("")
    print("Combined probability surface reference:")
    for label in OUTCOME_CLASSES:
        print(
            f"  {label:8s}: raw={raw_mean[label]:6.2f}% calibrated={calibrated_mean[label]:6.2f}% "
            f"baseline={baseline_mean[label]:6.2f}% observed={actual_freq[label]:6.2f}% "
            f"raw-bias={raw_mean[label]-actual_freq[label]:+6.2f}pp"
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
        a_preds = [row.method_a_trend for row in group if row.method_a_trend in OUTCOME_CLASSES]
        a_actual = [row.actual_class for row in group if row.method_a_trend in OUTCOME_CLASSES]
        b_preds = [row.method_b_trend for row in group if row.method_b_trend in OUTCOME_CLASSES]
        b_actual = [row.actual_class for row in group if row.method_b_trend in OUTCOME_CLASSES]
        print(f"  {name}:")
        if a_preds:
            print(f"    Method A n={len(a_preds)} accuracy={_accuracy(a_preds, a_actual):.2f}% counts={_class_counts(a_preds)}")
        if b_preds:
            print(f"    Method B n={len(b_preds)} accuracy={_accuracy(b_preds, b_actual):.2f}% counts={_class_counts(b_preds)}")

    print("")
    print("Stored Method A/B probability vectors:")
    print(f"  Method A vectors present: {a_present}/{len(rows)}")
    print(f"  Method B vectors present: {b_present}/{len(rows)}")
    if a_exact and b_exact:
        print("  STATUS: exact method-level probability decomposition is available.")
        for method_name, field in (("Method A", "method_a_probabilities_pct"), ("Method B", "method_b_probabilities_pct")):
            method_rows = [row for row in rows if getattr(row, field) is not None]
            mean = {
                label: float(np.mean([getattr(row, field)[label] for row in method_rows]))
                for label in OUTCOME_CLASSES
            }
            print(f"  {method_name} mean raw probabilities: {mean}")
    else:
        print("  STATUS: exact method-level probability decomposition is NOT available in this snapshot.")
        print("  REASON: Phase 5.8 snapshot v1 stores method trends but not fold-level Method A/B probabilities.")
        print("  CONSEQUENCE: this audit will not infer Method A/B probabilities from trend labels.")
        print("  ACTION: a future snapshot schema can retain those vectors; that would require a new Phase 5.8 snapshot.")

    print("")
    print("Interpretation guardrails:")
    print("  - The grouped combined surfaces are descriptive and use only frozen Phase 5.8 fold outputs.")
    print("  - Method trend availability identifies where each method contributed directionally, not its probability mass.")
    print("  - No Phase 5.8 calculation, relationship search, threshold, or decision rule is changed.")


if __name__ == "__main__":
    main()
