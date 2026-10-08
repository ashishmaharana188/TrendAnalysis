from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from analysis.phase5_snapshot import load_phase5_snapshot
from analysis.outcome_labels import OUTCOME_CLASSES
from analysis.phase6_calibration import calibrate_phase5_folds, CalibrationFold


@dataclass(frozen=True)
class SnapshotFold:
    prediction_date: date
    probabilities_pct: dict[str, float]
    actual_class: str
    predicted_trend: str
    method_a_trend: str | None
    method_b_trend: str | None
    baseline_probabilities_pct: dict[str, float]


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
                method_a_trend=row.get("method_a_trend"),
                method_b_trend=row.get("method_b_trend"),
                baseline_probabilities_pct=baseline,
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
        p: {a: sum(pp == p and aa == a for pp, aa in zip(pred, actual)) for a in OUTCOME_CLASSES}
        for p in OUTCOME_CLASSES
    }


def _mean_distribution(rows: list[SnapshotFold]) -> dict[str, float]:
    return {
        label: float(np.mean([row.probabilities_pct[label] for row in rows]))
        for label in OUTCOME_CLASSES
    }


def _mean_baseline(rows: list[SnapshotFold]) -> dict[str, float]:
    return {
        label: float(np.mean([row.baseline_probabilities_pct[label] for row in rows]))
        for label in OUTCOME_CLASSES
    }


def _method_stats(rows: list[SnapshotFold], field: str) -> tuple[list[str], list[str]]:
    predictions: list[str] = []
    actual: list[str] = []
    for row in rows:
        value = getattr(row, field)
        if value in OUTCOME_CLASSES:
            predictions.append(value)
            actual.append(row.actual_class)
    return predictions, actual


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase 5.8 probability surface, method trends, and baseline-relative bias."
    )
    parser.add_argument("snapshot")
    parser.add_argument("--min-calibration-observations", type=int, default=30)
    args = parser.parse_args()

    rows = load_folds(args.snapshot)
    if not rows:
        raise RuntimeError("Snapshot contains no usable folds.")

    actual = [row.actual_class for row in rows]
    combined_pred = [
        row.predicted_trend if row.predicted_trend in OUTCOME_CLASSES else "NO_CLEAR_TREND"
        for row in rows
    ]

    combined_clear = [
        pred for pred in combined_pred if pred in OUTCOME_CLASSES
    ]
    combined_clear_actual = [
        row.actual_class for row, pred in zip(rows, combined_pred)
        if pred in OUTCOME_CLASSES
    ]

    raw_mean = _mean_distribution(rows)
    baseline_mean = _mean_baseline(rows)
    actual_freq = {
        label: actual.count(label) / len(actual) * 100.0
        for label in OUTCOME_CLASSES
    }

    calibration_folds = [
        CalibrationFold(
            prediction_date=row.prediction_date,
            probabilities_pct=row.probabilities_pct,
            actual_class=row.actual_class,
        )
        for row in rows
    ]
    calibrated = calibrate_phase5_folds(
        calibration_folds,
        min_calibration_observations=args.min_calibration_observations,
    )
    calibrated_mean = {
        label: float(np.mean([
            pred.calibrated_probabilities_pct[label]
            for pred in calibrated.predictions
        ]))
        for label in OUTCOME_CLASSES
    }

    print("PHASE 6.2 PROBABILITY-SURFACE / METHOD AUDIT")
    print(f"Snapshot: {args.snapshot}")
    print("Snapshot status: FROZEN")
    print(f"Folds: {len(rows)}")
    print("Protocol: FROZEN_PHASE5_8_PROBABILITY_AND_METHOD_DIAGNOSTICS")

    print("")
    print("Combined probability surface:")
    for label in OUTCOME_CLASSES:
        raw = raw_mean[label]
        cal = calibrated_mean[label]
        obs = actual_freq[label]
        base = baseline_mean[label]
        print(
            f"  {label:8s}: raw={raw:6.2f}%, "
            f"calibrated={cal:6.2f}%, "
            f"baseline={base:6.2f}%, "
            f"observed={obs:6.2f}%, "
            f"raw-vs-observed={raw-obs:+6.2f}pp"
        )

    print("")
    print("Combined decision:")
    print(f"  clear decisions: {len(combined_clear)}")
    print(f"  NO_CLEAR_TREND: {len(rows)-len(combined_clear)}")
    print(f"  clear-decision accuracy: {_accuracy(combined_clear, combined_clear_actual):.2f}%")
    print(f"  clear decision counts: {_class_counts(combined_clear)}")
    print(f"  probability-argmax counts: ", end="")
    argmax = [max(OUTCOME_CLASSES, key=lambda x: row.probabilities_pct[x]) for row in rows]
    print(_class_counts(argmax))

    print("")
    for name, field in (("Method A", "method_a_trend"), ("Method B", "method_b_trend")):
        pred, labels = _method_stats(rows, field)
        print(f"{name}:")
        print(f"  available folds: {len(pred)}")
        print(f"  accuracy: {_accuracy(pred, labels):.2f}%")
        print(f"  class counts: {_class_counts(pred)}")
        print(f"  actual counts in available folds: {_class_counts(labels)}")
        if pred:
            m = _matrix(pred, labels)
            for p in OUTCOME_CLASSES:
                print(
                    f"  {p:8s}: UP={m[p]['UP']:3d}, "
                    f"SIDEWAYS={m[p]['SIDEWAYS']:3d}, DOWN={m[p]['DOWN']:3d}"
                )

    print("")
    agreement = 0
    conflict = 0
    only_a = 0
    only_b = 0
    neither = 0
    for row in rows:
        a = row.method_a_trend if row.method_a_trend in OUTCOME_CLASSES else None
        b = row.method_b_trend if row.method_b_trend in OUTCOME_CLASSES else None
        if a and b:
            if a == b:
                agreement += 1
            else:
                conflict += 1
        elif a:
            only_a += 1
        elif b:
            only_b += 1
        else:
            neither += 1
    print("Method availability/agreement:")
    print(f"  both directional/clear: {agreement + conflict}")
    print(f"  method agreement: {agreement}")
    print(f"  method conflict: {conflict}")
    print(f"  Method A only: {only_a}")
    print(f"  Method B only: {only_b}")
    print(f"  neither clear: {neither}")

    print("")
    print("Interpretation flags:")
    up_bias = raw_mean["UP"] - actual_freq["UP"]
    side_bias = raw_mean["SIDEWAYS"] - actual_freq["SIDEWAYS"]
    if up_bias >= 10.0 and side_bias <= -5.0:
        print("  PROBABILITY_CLASS_BIAS: material UP overprediction and SIDEWAYS underprediction.")
    else:
        print("  PROBABILITY_CLASS_BIAS: no large predefined UP/SIDEWAYS bias pattern detected.")

    if calibrated_mean["UP"] > actual_freq["UP"] + 10.0:
        print("  CALIBRATION_DID_NOT_REMOVE_UP_BIAS: calibrated mean UP remains materially above observed UP.")
    else:
        print("  CALIBRATION_DID_REMOVE_MATERIAL_UP_BIAS: calibrated UP is within 10pp of observed UP.")

    if combined_pred.count("UP") > len(rows) * 0.5:
        print("  DECISION_UP_HEAVY: more than half of clear decisions are UP.")
    else:
        print("  DECISION_UP_HEAVY: not majority-UP.")

    print("")
    print("Method matrices above are descriptive only. No Phase 5.8 calculation is changed.")


if __name__ == "__main__":
    main()
