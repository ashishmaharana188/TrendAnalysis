from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date

from analysis.phase5_snapshot import load_phase5_snapshot
from analysis.phase6_calibration import calibrate_phase5_folds


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
        if not isinstance(probabilities, dict):
            continue
        if not prediction_date:
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
        description="Run Phase 6 calibration from a frozen Phase 5.8 snapshot."
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

    result = calibrate_phase5_folds(
        folds,
        min_calibration_observations=args.min_calibration_observations,
    )

    print("PHASE 6 SNAPSHOT CALIBRATION")
    print(f"Snapshot: {args.snapshot}")
    print(f"Snapshot status: {snapshot.manifest['status']}")
    print(f"Phase 5.8 folds: {len(folds)}")
    print(f"Protocol: {result.protocol}")
    print(f"Raw log loss: {result.metrics.raw_log_loss:.6f}")
    print(f"Calibrated log loss: {result.metrics.log_loss:.6f}")
    print(f"Raw Brier: {result.metrics.raw_brier_score:.6f}")
    print(f"Calibrated Brier: {result.metrics.brier_score:.6f}")
    print(f"Raw accuracy: {result.metrics.raw_accuracy_pct:.2f}%")
    print(f"Calibrated accuracy: {result.metrics.accuracy_pct:.2f}%")
    print(
        f"Raw ECE: {result.metrics.raw_expected_calibration_error_pct:.4f}%"
    )
    print(
        f"Calibrated ECE: {result.metrics.expected_calibration_error_pct:.4f}%"
    )


if __name__ == "__main__":
    main()

