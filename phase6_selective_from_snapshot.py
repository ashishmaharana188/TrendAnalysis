from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date

from analysis.phase5_snapshot import load_phase5_snapshot
from analysis.phase6_calibration import calibrate_phase5_folds
from analysis.phase6_selective import evaluate_selective_predictions


@dataclass(frozen=True)
class SnapshotFold:
    prediction_date: date
    probabilities_pct: dict[str, float]
    actual_class: str


def _as_date(value: str) -> date:
    return date.fromisoformat(value)


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
                prediction_date=_as_date(str(prediction_date)),
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
        description="Evaluate selective prediction from a frozen Phase 5.8 snapshot."
    )
    parser.add_argument("snapshot", help="Path to a frozen Phase 5.8 snapshot directory.")
    parser.add_argument(
        "--min-calibration-observations",
        type=int,
        default=30,
    )
    args = parser.parse_args()

    snapshot = load_phase5_snapshot(args.snapshot)
    folds = load_prediction_folds(args.snapshot)
    calibration = calibrate_phase5_folds(
        folds,
        min_calibration_observations=args.min_calibration_observations,
    )
    result = evaluate_selective_predictions(calibration.predictions)

    print("PHASE 6.2 SELECTIVE PREDICTION")
    print(f"Snapshot: {args.snapshot}")
    print(f"Snapshot status: {snapshot.manifest['status']}")
    print(f"Phase 5.8 folds: {len(folds)}")
    print(f"Protocol: {result.protocol}")
    print(f"Baseline majority accuracy: {result.baseline_accuracy_pct:.2f}%")
    print("")
    print("Confidence thresholds:")
    for item in result.thresholds:
        print(
            f"  >= {item.threshold_pct:5.1f}%: "
            f"selected {item.selected_observations:3d}/{item.total_observations} "
            f"({item.coverage_pct:6.2f}% coverage), "
            f"accuracy {item.accuracy_pct:6.2f}%, "
            f"lift {item.accuracy_lift_pct_points:+6.2f}pp, "
            f"mean confidence {item.mean_confidence_pct:6.2f}%"
        )

    print("")
    print("Confidence buckets:")
    for item in result.confidence_buckets:
        print(
            f"  {item.lower_bound_pct:5.1f}% - {item.upper_bound_pct:5.1f}%: "
            f"n={item.observations:3d}, "
            f"coverage={item.coverage_pct:6.2f}%, "
            f"accuracy={item.accuracy_pct:6.2f}%, "
            f"mean confidence={item.mean_confidence_pct:6.2f}%, "
            f"gap={item.confidence_minus_accuracy_pct_points:+6.2f}pp"
        )

    print("")
    improving = result.improving_thresholds
    print(f"Thresholds above baseline: {len(improving)}")
    for item in improving:
        print(
            f"  >= {item.threshold_pct:.1f}% -> "
            f"accuracy {item.accuracy_pct:.2f}% at {item.coverage_pct:.2f}% coverage"
        )

    if not improving:
        print("SELECTIVE GATE: NO PREDECLARED THRESHOLD BEATS BASELINE")
    else:
        print("SELECTIVE GATE: PROMISING THRESHOLDS IDENTIFIED")


if __name__ == "__main__":
    main()
