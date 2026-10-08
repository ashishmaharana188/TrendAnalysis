from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from analysis.phase5_snapshot import load_phase5_snapshot
from analysis.outcome_labels import OUTCOME_CLASSES


def _normalise(raw: dict[str, Any]) -> dict[str, float]:
    values = {label: max(float(raw.get(label, 0.0)), 0.0) for label in OUTCOME_CLASSES}
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("Probability distribution has no positive mass.")
    return {label: value / total * 100.0 for label, value in values.items()}


def _accuracy(predicted: list[str], actual: list[str]) -> float:
    if not actual:
        return 0.0
    return sum(p == a for p, a in zip(predicted, actual)) / len(actual) * 100.0


def _majority_accuracy(actual: list[str]) -> float:
    if not actual:
        return 0.0
    return max(actual.count(label) for label in OUTCOME_CLASSES) / len(actual) * 100.0


def _counts(values: list[str | None]) -> dict[str, int]:
    return {
        label: sum(value == label for value in values)
        for label in OUTCOME_CLASSES
    }


@dataclass(frozen=True)
class DecisionAudit:
    all_folds: int
    actual_classified_folds: int
    current_clear_folds: int
    current_no_clear_folds: int
    current_accuracy_on_clear_folds_pct: float
    current_coverage_pct: float
    current_decided_hit_count: int
    argmax_accuracy_all_folds_pct: float
    argmax_accuracy_on_clear_folds_pct: float
    baseline_majority_accuracy_all_folds_pct: float
    baseline_majority_accuracy_clear_folds_pct: float
    current_vs_argmax_disagreements_on_clear_folds: int
    sideways_argmax_all_folds: int
    sideways_argmax_and_current_abstain: int
    sideways_argmax_but_current_directional: int
    current_class_counts: dict[str, int]
    argmax_class_counts: dict[str, int]
    actual_class_counts: dict[str, int]
    mean_probability_pct: dict[str, float]
    probability_minus_actual_frequency_pct_points: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "all_folds": self.all_folds,
            "actual_classified_folds": self.actual_classified_folds,
            "current_clear_folds": self.current_clear_folds,
            "current_no_clear_folds": self.current_no_clear_folds,
            "current_accuracy_on_clear_folds_pct": self.current_accuracy_on_clear_folds_pct,
            "current_coverage_pct": self.current_coverage_pct,
            "current_decided_hit_count": self.current_decided_hit_count,
            "argmax_accuracy_all_folds_pct": self.argmax_accuracy_all_folds_pct,
            "argmax_accuracy_on_clear_folds_pct": self.argmax_accuracy_on_clear_folds_pct,
            "baseline_majority_accuracy_all_folds_pct": self.baseline_majority_accuracy_all_folds_pct,
            "baseline_majority_accuracy_clear_folds_pct": self.baseline_majority_accuracy_clear_folds_pct,
            "current_vs_argmax_disagreements_on_clear_folds": self.current_vs_argmax_disagreements_on_clear_folds,
            "sideways_argmax_all_folds": self.sideways_argmax_all_folds,
            "sideways_argmax_and_current_abstain": self.sideways_argmax_and_current_abstain,
            "sideways_argmax_but_current_directional": self.sideways_argmax_but_current_directional,
            "current_class_counts": dict(self.current_class_counts),
            "argmax_class_counts": dict(self.argmax_class_counts),
            "actual_class_counts": dict(self.actual_class_counts),
            "mean_probability_pct": dict(self.mean_probability_pct),
            "probability_minus_actual_frequency_pct_points": dict(self.probability_minus_actual_frequency_pct_points),
        }


def run_audit(snapshot_root: str) -> DecisionAudit:
    snapshot = load_phase5_snapshot(snapshot_root)
    rows = list(snapshot.prediction_folds)

    actual: list[str] = []
    current_clear: list[str] = []
    current_clear_actual: list[str] = []
    argmax: list[str] = []
    clear_argmax: list[str] = []

    current_values: list[str | None] = []
    probabilities: list[dict[str, float]] = []

    disagreement = 0
    sideways_argmax_all = 0
    sideways_argmax_abstain = 0
    sideways_argmax_directional = 0

    for row in rows:
        actual_class = row.get("actual_class")
        current = row.get("predicted_trend")
        raw_probs = row.get("probabilities_pct")

        if actual_class not in OUTCOME_CLASSES or not isinstance(raw_probs, dict):
            continue

        probs = _normalise(raw_probs)
        top_class = max(OUTCOME_CLASSES, key=lambda label: probs[label])

        actual.append(actual_class)
        argmax.append(top_class)
        probabilities.append(probs)

        if current in OUTCOME_CLASSES:
            current_values.append(current)
            current_clear.append(current)
            current_clear_actual.append(actual_class)
            clear_argmax.append(top_class)
            if current != top_class:
                disagreement += 1
        else:
            current_values.append(None)

        if top_class == "SIDEWAYS":
            sideways_argmax_all += 1
            if current not in OUTCOME_CLASSES:
                sideways_argmax_abstain += 1
            else:
                sideways_argmax_directional += 1

    if not actual:
        raise RuntimeError("Snapshot contains no usable classified prediction folds.")

    mean_probability = {
        label: sum(row[label] for row in probabilities) / len(probabilities)
        for label in OUTCOME_CLASSES
    }
    actual_counts = _counts(actual)
    actual_frequency = {
        label: actual_counts[label] / len(actual) * 100.0
        for label in OUTCOME_CLASSES
    }

    clear_accuracy = _accuracy(current_clear, current_clear_actual)
    clear_n = len(current_clear)

    return DecisionAudit(
        all_folds=len(actual),
        actual_classified_folds=len(actual),
        current_clear_folds=clear_n,
        current_no_clear_folds=len(actual) - clear_n,
        current_accuracy_on_clear_folds_pct=clear_accuracy,
        current_coverage_pct=clear_n / len(actual) * 100.0,
        current_decided_hit_count=sum(
            prediction == outcome
            for prediction, outcome in zip(current_clear, current_clear_actual)
        ),
        argmax_accuracy_all_folds_pct=_accuracy(argmax, actual),
        argmax_accuracy_on_clear_folds_pct=_accuracy(clear_argmax, current_clear_actual),
        baseline_majority_accuracy_all_folds_pct=_majority_accuracy(actual),
        baseline_majority_accuracy_clear_folds_pct=_majority_accuracy(current_clear_actual),
        current_vs_argmax_disagreements_on_clear_folds=disagreement,
        sideways_argmax_all_folds=sideways_argmax_all,
        sideways_argmax_and_current_abstain=sideways_argmax_abstain,
        sideways_argmax_but_current_directional=sideways_argmax_directional,
        current_class_counts=_counts(current_values),
        argmax_class_counts=_counts(argmax),
        actual_class_counts=actual_counts,
        mean_probability_pct=mean_probability,
        probability_minus_actual_frequency_pct_points={
            label: mean_probability[label] - actual_frequency[label]
            for label in OUTCOME_CLASSES
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the full Phase 5.8 decision layer without dropping NO_CLEAR_TREND folds."
    )
    parser.add_argument("snapshot")
    args = parser.parse_args()

    result = run_audit(args.snapshot)

    print("PHASE 5.8 DECISION-LAYER AUDIT V2")
    print(f"Snapshot: {args.snapshot}")
    print("Snapshot status: FROZEN")
    print(f"All usable classified folds: {result.all_folds}")
    print(f"Current clear decisions: {result.current_clear_folds}")
    print(f"Current NO_CLEAR_TREND abstentions: {result.current_no_clear_folds}")
    print(f"Current decision accuracy on decided folds: {result.current_accuracy_on_clear_folds_pct:.2f}%")
    print(f"Current decision coverage: {result.current_coverage_pct:.2f}%")
    print(f"Probability-argmax accuracy on all folds: {result.argmax_accuracy_all_folds_pct:.2f}%")
    print(f"Probability-argmax accuracy on current clear folds: {result.argmax_accuracy_on_clear_folds_pct:.2f}%")
    print(f"Baseline majority accuracy on all folds: {result.baseline_majority_accuracy_all_folds_pct:.2f}%")
    print(f"Baseline majority accuracy on current clear folds: {result.baseline_majority_accuracy_clear_folds_pct:.2f}%")
    print(f"Current vs argmax disagreements on decided folds: {result.current_vs_argmax_disagreements_on_clear_folds}")
    print(f"SIDEWAYS probability argmax on all folds: {result.sideways_argmax_all_folds}")
    print(f"SIDEWAYS argmax with current abstention: {result.sideways_argmax_and_current_abstain}")
    print(f"SIDEWAYS argmax turned into directional decision: {result.sideways_argmax_but_current_directional}")

    print("")
    print("Current decision counts (None = NO_CLEAR_TREND):")
    for label in OUTCOME_CLASSES:
        print(f"  {label:8s}: {result.current_class_counts[label]:3d}")
    print(f"  NO_CLEAR: {result.current_no_clear_folds:3d}")

    print("")
    print("Probability-argmax class counts:")
    for label in OUTCOME_CLASSES:
        print(f"  {label:8s}: {result.argmax_class_counts[label]:3d}")

    print("")
    print("Actual class counts:")
    for label in OUTCOME_CLASSES:
        print(f"  {label:8s}: {result.actual_class_counts[label]:3d}")

    print("")
    print("Mean predicted probabilities vs observed frequencies:")
    for label in OUTCOME_CLASSES:
        observed = result.actual_class_counts[label] / result.all_folds * 100.0
        bias = result.probability_minus_actual_frequency_pct_points[label]
        print(
            f"  {label:8s}: predicted={result.mean_probability_pct[label]:6.2f}%, "
            f"observed={observed:6.2f}%, bias={bias:+6.2f}pp"
        )


if __name__ == "__main__":
    main()
