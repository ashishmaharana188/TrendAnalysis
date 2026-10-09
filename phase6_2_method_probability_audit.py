from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from analysis.outcome_labels import OUTCOME_CLASSES
from analysis.phase5_snapshot import load_phase5_snapshot
from analysis.phase6_calibration import CalibrationObservation, fit_temperature


@dataclass(frozen=True)
class Fold:
    prediction_date: date
    actual_class: str
    combined_probabilities_pct: dict[str, float]
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


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _dist(raw: Any) -> dict[str, float] | None:
    if not isinstance(raw, dict):
        return None
    try:
        values = {label: max(float(raw.get(label, 0.0)), 0.0) for label in OUTCOME_CLASSES}
    except (TypeError, ValueError):
        return None
    total = float(sum(values.values()))
    if not np.isfinite(total) or total <= 0.0:
        return None
    return {label: values[label] / total * 100.0 for label in OUTCOME_CLASSES}


def load_folds(snapshot_root: str) -> list[Fold]:
    snapshot = load_phase5_snapshot(snapshot_root)
    rows: list[Fold] = []
    for row in snapshot.prediction_folds:
        actual = row.get("actual_class")
        combined = _dist(row.get("probabilities_pct"))
        if actual not in OUTCOME_CLASSES or combined is None:
            continue
        rows.append(
            Fold(
                prediction_date=_as_date(row.get("prediction_date")),
                actual_class=actual,
                combined_probabilities_pct=combined,
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
            )
        )
    return sorted(rows, key=lambda row: row.prediction_date)


def _mean_distribution(folds: list[Fold], field: str) -> dict[str, float] | None:
    values = [getattr(row, field) for row in folds if getattr(row, field) is not None]
    if not values:
        return None
    return {
        label: float(np.mean([dist[label] for dist in values]))
        for label in OUTCOME_CLASSES
    }


def _mean_selected(folds: list[Fold], field: str) -> float | None:
    values = [getattr(row, field) for row in folds if getattr(row, field) is not None]
    return float(np.mean(values)) if values else None


def _method_calibration(folds: list[Fold], field: str) -> dict[str, Any] | None:
    values = [(row.prediction_date, getattr(row, field), row.actual_class) for row in folds]
    values = [item for item in values if item[1] is not None]
    if not values:
        return None

    history: list[CalibrationObservation] = []
    calibrated_means: list[dict[str, float]] = []
    temperatures: list[float] = []
    statuses: dict[str, int] = {}
    for prediction_date, probabilities, actual in values:
        calibration = fit_temperature(history, min_observations=30)
        calibrated = calibration.apply(probabilities)
        calibrated_means.append(calibrated)
        temperatures.append(calibration.temperature)
        statuses[calibration.status] = statuses.get(calibration.status, 0) + 1
        history.append(
            CalibrationObservation(
                prediction_date=prediction_date,
                probabilities_pct=probabilities,
                actual_class=actual,
            )
        )

    return {
        "observations": len(values),
        "mean_calibrated": {
            label: float(np.mean([item[label] for item in calibrated_means]))
            for label in OUTCOME_CLASSES
        },
        "mean_temperature": float(np.mean(temperatures)),
        "statuses": statuses,
    }


def _report(name: str, folds: list[Fold], prob_field: str) -> None:
    available = [row for row in folds if getattr(row, prob_field) is not None]
    print(f"{name} probability surface:")
    print(f"  exact vectors present: {len(available)}/{len(folds)}")
    if not available:
        print("  STATUS: unavailable in this snapshot")
        return

    mean = _mean_distribution(folds, prob_field)
    observed = {
        label: sum(row.actual_class == label for row in available) / len(available) * 100.0
        for label in OUTCOME_CLASSES
    }
    print(
        "  raw mean: "
        + ", ".join(f"{label}={mean[label]:.2f}%" for label in OUTCOME_CLASSES)
    )
    print(
        "  observed: "
        + ", ".join(f"{label}={observed[label]:.2f}%" for label in OUTCOME_CLASSES)
    )
    print(
        "  bias: "
        + ", ".join(
            f"{label}={mean[label] - observed[label]:+.2f}pp"
            for label in OUTCOME_CLASSES
        )
    )
    calibration = _method_calibration(folds, prob_field)
    print(
        "  calibrated mean: "
        + ", ".join(
            f"{label}={calibration['mean_calibrated'][label]:.2f}%"
            for label in OUTCOME_CLASSES
        )
    )
    print(f"  mean temperature: {calibration['mean_temperature']:.4f}")
    print(f"  calibration statuses: {calibration['statuses']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit exact Phase 5.8 Method A/B probability vectors and selection metadata."
    )
    parser.add_argument("snapshot")
    args = parser.parse_args()

    folds = load_folds(args.snapshot)
    if not folds:
        raise RuntimeError("Snapshot contains no usable folds.")

    print("PHASE 6.2 METHOD-A/B EXACT PROBABILITY / SELECTION-REUSE AUDIT")
    print(f"Snapshot: {args.snapshot}")
    print("Snapshot status: FROZEN")
    print(f"Folds: {len(folds)}")
    print("Protocol: EXACT_STORED_METHOD_SURFACES_NO_OLAP_RECOMPUTATION")
    print("")

    _report("Method A", folds, "method_a_probabilities_pct")
    print("")
    _report("Method B", folds, "method_b_probabilities_pct")

    print("")
    print("Selected-relationship metadata:")
    for name, score_field, ess_field, count_field, vars_field, cond_field in (
        ("Method A", "method_a_evidence_score", "method_a_effective_sample_size", "method_a_sample_count", "method_a_variables", "method_a_condition"),
        ("Method B", "method_b_evidence_score", "method_b_effective_sample_size", "method_b_sample_count", "method_b_variables", "method_b_condition"),
    ):
        selected = [row for row in folds if getattr(row, vars_field)]
        print(f"  {name}: selected relationships recorded={len(selected)}/{len(folds)}")
        if selected:
            print(f"    mean evidence score={_mean_selected(folds, score_field):.6f}")
            print(f"    mean effective sample size={_mean_selected(folds, ess_field):.3f}")
            print(f"    mean supporting sample count={_mean_selected(folds, count_field):.3f}")
            unique_relationships = len({(getattr(row, vars_field), getattr(row, cond_field)) for row in selected})
            print(f"    unique selected relationship signatures={unique_relationships}")

    print("")
    print("Selection-reuse protocol note:")
    print("  Phase 5.8 selects each relationship on the inner chronological holdout.")
    print("  The selected relationship's retained support is then used to construct Method A/B probabilities.")
    print("  This audit does not change that protocol; it makes its resulting probability surface observable.")


if __name__ == "__main__":
    main()
