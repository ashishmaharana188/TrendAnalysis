from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# This overlay test exercises Phase 6 code without requiring the rest of the
# full TrendAnalysis dependency graph. The real repository already provides
# analysis.outcome_labels.
stub = types.ModuleType("analysis.outcome_labels")
stub.OUTCOME_CLASSES = ("UP", "SIDEWAYS", "DOWN")
stub.OutcomeClass = str
sys.modules["analysis.outcome_labels"] = stub

from analysis.phase6_calibration import CalibratedPrediction
from analysis.phase6_selective import evaluate_selective_predictions


def _prediction(day: int, probs: dict[str, float], actual: str) -> CalibratedPrediction:
    return CalibratedPrediction(
        prediction_date=date(2026, 1, 1) + timedelta(days=day),
        raw_probabilities_pct=probs,
        calibrated_probabilities_pct=probs,
        actual_class=actual,
        temperature=1.0,
        calibration_status="FITTED",
    )


def test_fixed_threshold_is_selective_and_above_baseline() -> None:
    predictions = [
        _prediction(0, {"UP": 80, "SIDEWAYS": 15, "DOWN": 5}, "UP"),
        _prediction(1, {"UP": 75, "SIDEWAYS": 20, "DOWN": 5}, "UP"),
        _prediction(2, {"UP": 70, "SIDEWAYS": 20, "DOWN": 10}, "DOWN"),
        _prediction(3, {"UP": 45, "SIDEWAYS": 35, "DOWN": 20}, "SIDEWAYS"),
        _prediction(4, {"UP": 42, "SIDEWAYS": 40, "DOWN": 18}, "UP"),
        _prediction(5, {"UP": 40, "SIDEWAYS": 35, "DOWN": 25}, "DOWN"),
    ]

    result = evaluate_selective_predictions(
        predictions,
        thresholds_pct=(40.0, 70.0),
        development_fraction=4 / 6,
    )

    assert result.observations == 6
    assert result.baseline_accuracy_pct == 50.0

    at_70 = next(item for item in result.thresholds if item.threshold_pct == 70.0)
    assert at_70.selected_observations == 3
    assert at_70.accuracy_pct == 66.66666666666666
    assert at_70.improves_over_baseline


def test_chronological_selection_uses_final_period_only_for_evaluation() -> None:
    predictions = [
        _prediction(0, {"UP": 90, "SIDEWAYS": 5, "DOWN": 5}, "UP"),
        _prediction(1, {"UP": 85, "SIDEWAYS": 10, "DOWN": 5}, "UP"),
        _prediction(2, {"UP": 80, "SIDEWAYS": 15, "DOWN": 5}, "UP"),
        _prediction(3, {"UP": 75, "SIDEWAYS": 20, "DOWN": 5}, "UP"),
        _prediction(4, {"UP": 72, "SIDEWAYS": 20, "DOWN": 8}, "DOWN"),
        _prediction(5, {"UP": 50, "SIDEWAYS": 35, "DOWN": 15}, "SIDEWAYS"),
        _prediction(6, {"UP": 70, "SIDEWAYS": 20, "DOWN": 10}, "DOWN"),
        _prediction(7, {"UP": 65, "SIDEWAYS": 25, "DOWN": 10}, "UP"),
        _prediction(8, {"UP": 60, "SIDEWAYS": 25, "DOWN": 15}, "UP"),
        _prediction(9, {"UP": 55, "SIDEWAYS": 30, "DOWN": 15}, "DOWN"),
    ]

    result = evaluate_selective_predictions(
        predictions,
        thresholds_pct=(40.0, 60.0, 70.0),
        development_fraction=0.6,
        minimum_development_coverage_pct=10.0,
    )

    selected = result.selected_threshold
    assert selected.selection_status == "SELECTED_FROM_DEVELOPMENT"
    assert selected.threshold_pct == 70.0
    assert selected.development_observations == 6
    assert selected.final_observations == 4
    assert selected.final_coverage_pct == 50.0
    assert selected.final_accuracy_pct == 0.0


def test_threshold_uses_calibrated_confidence() -> None:
    predictions = [
        _prediction(0, {"UP": 55, "SIDEWAYS": 25, "DOWN": 20}, "UP"),
        _prediction(1, {"UP": 49, "SIDEWAYS": 30, "DOWN": 21}, "DOWN"),
    ]

    result = evaluate_selective_predictions(
        predictions,
        thresholds_pct=(50.0,),
        development_fraction=0.5,
    )
    item = result.thresholds[0]

    assert item.selected_observations == 1
    assert item.accuracy_pct == 100.0
    assert item.mean_confidence_pct == 55.0
