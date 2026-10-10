from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date

from analysis.phase5_snapshot import load_phase5_snapshot
from analysis.phase6_calibration import CalibratedPrediction
from analysis.phase6_probability_metrics import as_date, calibrate_with_mature_labels
from analysis.phase6_2_diagnostics import diagnose_confidence


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
        actual_class = row.get("actual_class")
        probabilities = row.get("probabilities_pct")
        prediction_date = row.get("prediction_date")
        if actual_class not in {"UP", "SIDEWAYS", "DOWN"}:
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
                actual_class=actual_class,
            )
        )
    folds.sort(key=lambda item: item.prediction_date)
    if not folds:
        raise RuntimeError("Snapshot contains no usable prediction folds.")
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose confidence ranking and class bias using only labels mature at each forecast date."
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
    result = diagnose_confidence(predictions)

    print("PHASE 6.2 CONFIDENCE / CLASS DIAGNOSTICS")
    print(f"Snapshot: {args.snapshot}")
    print(f"Snapshot status: {snapshot.manifest['status']}")
    print(f"Phase 5.8 folds: {len(folds)}")
    print(f"Protocol: {result.protocol}")
    print("Calibration protocol: PREQUENTIAL_ONLY_OUTCOMES_MATURED_BEFORE_PREDICTION")
    print(f"Calibration statuses: {dict(__import__('collections').Counter(calibration.statuses))}")
    print("Mature calibration history sizes: "
          f"min={min(calibration.mature_training_observations)}, "
          f"median={sorted(calibration.mature_training_observations)[len(folds)//2]}, "
          f"max={max(calibration.mature_training_observations)}")
    print(f"Overall accuracy: {result.overall_accuracy_pct:.2f}%")
    print(f"Baseline majority accuracy: {result.baseline_majority_accuracy_pct:.2f}%")
    corr = result.confidence_accuracy_spearman
    print("Confidence vs correctness Spearman: " + (f"{corr:+.4f}" if corr is not None else "n/a"))
    print(f"Mean confidence, correct:   {result.correct_confidence_mean_pct:.2f}%")
    print(f"Mean confidence, incorrect: {result.incorrect_confidence_mean_pct:.2f}%")
    print(
        "Correct-minus-incorrect confidence gap: "
        f"{result.confidence_gap_correct_minus_incorrect_pct_points:+.2f}pp"
    )
    print(f"High-confidence ranking useful: {'YES' if result.high_confidence_is_better else 'NO'}")

    print("")
    print("Confidence rank curve:")
    for item in result.confidence_rank_curve:
        lift = f"{item.lift_pct_points:+.2f}pp" if item.lift_pct_points is not None else "n/a"
        print(
            f"  top {item.population_fraction_pct:5.1f}%: n={item.observations:3d}, "
            f"accuracy={item.accuracy_pct if item.accuracy_pct is not None else 0.0:6.2f}%, "
            f"lift={lift:>8}, "
            f"mean confidence={item.mean_confidence_pct if item.mean_confidence_pct is not None else 0.0:6.2f}%"
        )

    print("")
    print("Predicted-class diagnostics:")
    for item in result.class_diagnostics:
        accuracy = item.accuracy_pct if item.accuracy_pct is not None else 0.0
        confidence = item.mean_confidence_pct if item.mean_confidence_pct is not None else 0.0
        print(
            f"  {item.predicted_class:8s}: n={item.observations:3d}, accuracy={accuracy:6.2f}%, "
            f"mean confidence={confidence:6.2f}%, actual [UP={item.actual_up_pct:5.1f}%, "
            f"SIDEWAYS={item.actual_sideways_pct:5.1f}%, DOWN={item.actual_down_pct:5.1f}%]"
        )


if __name__ == "__main__":
    main()
