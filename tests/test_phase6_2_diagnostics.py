from datetime import date
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.phase6_calibration import CalibratedPrediction
from analysis.phase6_2_diagnostics import diagnose_confidence


def _p(day: int, up: float, side: float, down: float, actual: str) -> CalibratedPrediction:
    return CalibratedPrediction(
        prediction_date=date(2026, 1, day),
        raw_probabilities_pct={"UP": up, "SIDEWAYS": side, "DOWN": down},
        calibrated_probabilities_pct={"UP": up, "SIDEWAYS": side, "DOWN": down},
        actual_class=actual,
        temperature=1.0,
        calibration_status="FITTED",
    )


def test_confidence_ranking_positive():
    rows = [
        _p(1, 90, 5, 5, "UP"),
        _p(2, 85, 10, 5, "UP"),
        _p(3, 80, 10, 10, "UP"),
        _p(4, 40, 35, 25, "DOWN"),
        _p(5, 38, 35, 27, "SIDEWAYS"),
        _p(6, 36, 34, 30, "DOWN"),
    ]
    result = diagnose_confidence(rows)
    assert result.confidence_accuracy_spearman is not None
    assert result.confidence_accuracy_spearman > 0
    assert result.correct_confidence_mean_pct > result.incorrect_confidence_mean_pct


def test_confidence_ranking_inversion():
    # All rows are predicted UP. The low-confidence predictions are correct,
    # while the high-confidence predictions are wrong. This creates a genuine
    # negative confidence/correctness rank relationship rather than a degenerate
    # all-incorrect sample where Spearman correlation is undefined.
    rows = [
        _p(1, 40, 35, 25, "UP"),
        _p(2, 38, 35, 27, "UP"),
        _p(3, 36, 34, 30, "UP"),
        _p(4, 90, 5, 5, "DOWN"),
        _p(5, 85, 10, 5, "SIDEWAYS"),
        _p(6, 80, 10, 10, "DOWN"),
    ]
    result = diagnose_confidence(rows)
    assert result.confidence_accuracy_spearman is not None
    assert result.confidence_accuracy_spearman < 0
    assert result.correct_confidence_mean_pct < result.incorrect_confidence_mean_pct


def test_class_diagnostics_conservation():
    rows = [
        _p(1, 70, 20, 10, "UP"),
        _p(2, 60, 30, 10, "DOWN"),
        _p(3, 20, 70, 10, "SIDEWAYS"),
    ]
    result = diagnose_confidence(rows)
    assert sum(item.observations for item in result.class_diagnostics) == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
