from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date

from analysis.phase5_snapshot import load_phase5_snapshot
from analysis.phase6_calibration import CalibratedPrediction
from analysis.phase6_probability_metrics import as_date, calibrate_with_mature_labels
from analysis.phase6_selective import evaluate_selective_predictions


@dataclass(frozen=True)
class SnapshotFold:
    prediction_date: date
    outcome_end_date: date | None
    probabilities_pct: dict[str, float]
    actual_class: str


def load_prediction_folds(snapshot_root: str) -> list[SnapshotFold]:
    snapshot = load_phase5_snapshot(snapshot_root)
    folds: list[SnapshotFold] = []
    for row in snapshot.prediction_folds:
        actual = row.get("actual_class")
        probabilities = row.get("probabilities_pct")
        prediction_date = row.get("prediction_date")
        if actual not in {"UP", "SIDEWAYS", "DOWN"}:
            continue
        if not isinstance(probabilities, dict) or not prediction_date:
            continue
        folds.append(
            SnapshotFold(
                prediction_date=as_date(prediction_date),
                outcome_end_date=as_date(row.get("outcome_end_date")),
                probabilities_pct={
                    label: float(probabilities.get(label, 0.0))
                    for label in ("UP", "SIDEWAYS", "DOWN")
                },
                actual_class=actual,
            )
        )
    folds.sort(key=lambda item: item.prediction_date)
    if not folds:
        raise RuntimeError("Snapshot contains no usable prediction folds.")
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate selective prediction using frozen folds and only mature labels for calibration."
    )
    parser.add_argument("snapshot", help="Path to frozen Phase 5.8 snapshot directory.")
    parser.add_argument("--min-calibration-observations", type=int, default=30)
    args = parser.parse_args()

    snapshot = load_phase5_snapshot(args.snapshot)
    folds = load_prediction_folds(args.snapshot)
    calibration = calibrate_with_mature_labels(
        [item.prediction_date for item in folds],
        [item.outcome_end_date for item in folds],
        [item.actual_class for item in folds],
        [item.probabilities_pct for item in folds],
        min_calibration_observations=args.min_calibration_observations,
    )
    predictions = [
        CalibratedPrediction(
            prediction_date=fold.prediction_date,
            raw_probabilities_pct=fold.probabilities_pct,
            calibrated_probabilities_pct=calibration.probabilities_pct[index],
            actual_class=fold.actual_class,
            temperature=(calibration.temperatures[index] if calibration.temperatures[index] is not None else 1.0),
            calibration_status=calibration.statuses[index],
        )
        for index, fold in enumerate(folds)
        if calibration.probabilities_pct[index] is not None
    ]
    result = evaluate_selective_predictions(predictions)

    print("PHASE 6.2 SELECTIVE PREDICTION")
    print(f"Snapshot: {args.snapshot}")
    print(f"Snapshot status: {snapshot.manifest['status']}")
    print(f"Phase 5.8 folds: {len(folds)}")
    print(f"Protocol: {result.protocol}")
    print("Calibration protocol: PREQUENTIAL_ONLY_OUTCOMES_MATURED_BEFORE_PREDICTION")
    print(f"Calibration statuses: {dict(Counter(calibration.statuses))}")
    print("Mature calibration history sizes: "
          f"min={min(calibration.mature_training_observations)}, "
          f"median={sorted(calibration.mature_training_observations)[len(folds)//2]}, "
          f"max={max(calibration.mature_training_observations)}")
    print(f"Baseline majority accuracy: {result.baseline_accuracy_pct:.2f}%")

    print("")
    print("Confidence thresholds:")
    for item in result.thresholds:
        print(
            f"  >= {item.threshold_pct:5.1f}%: selected {item.selected_observations:3d}/{item.total_observations} "
            f"({item.coverage_pct:6.2f}% coverage), accuracy {item.accuracy_pct:6.2f}%, "
            f"lift {item.accuracy_lift_pct_points:+6.2f}pp, mean confidence {item.mean_confidence_pct:6.2f}%"
        )

    print("")
    print("Confidence buckets:")
    for item in result.confidence_buckets:
        print(
            f"  {item.lower_bound_pct:5.1f}% - {item.upper_bound_pct:5.1f}%: n={item.observations:3d}, "
            f"coverage={item.coverage_pct:6.2f}%, accuracy={item.accuracy_pct:6.2f}%, "
            f"mean confidence={item.mean_confidence_pct:6.2f}%, "
            f"gap={item.confidence_minus_accuracy_pct_points:+6.2f}pp"
        )

    print("")
    improving = result.improving_thresholds
    print(f"Thresholds above retrospective majority reference: {len(improving)}")
    for item in improving:
        print(f"  >= {item.threshold_pct:.1f}% -> accuracy {item.accuracy_pct:.2f}% at {item.coverage_pct:.2f}% coverage")
    if not improving:
        print("SELECTIVE GATE: NO PREDECLARED THRESHOLD BEATS BASELINE")
    else:
        print("SELECTIVE GATE: PROMISING THRESHOLDS IDENTIFIED")
    print("Interpretation note: threshold table is descriptive; the selected threshold's final chronological period is evaluated separately in selected_threshold.")
    print(f"Selected-threshold evaluation: {result.selected_threshold.as_dict()}")


if __name__ == "__main__":
    main()
