from __future__ import annotations

from datetime import date, timedelta

import pytest

from analysis.phase6_probability_metrics import (
    calibrate_with_mature_labels,
    compare_probabilities_paired,
    moving_block_bootstrap_ci,
    normalize_distribution,
    score_probabilities,
)


def test_normalize_distribution_rejects_invalid_probability_mass() -> None:
    assert normalize_distribution({"UP": 20, "SIDEWAYS": 30, "DOWN": 50}) == pytest.approx(
        {"UP": 20, "SIDEWAYS": 30, "DOWN": 50}
    )
    assert normalize_distribution({"UP": -1, "SIDEWAYS": 50, "DOWN": 51}) is None
    assert normalize_distribution({"UP": 0, "SIDEWAYS": 0, "DOWN": 0}) is None
    assert normalize_distribution({"UP": float("nan"), "SIDEWAYS": 50, "DOWN": 50}) is None


def test_score_probabilities_returns_proper_scores_and_class_diagnostics() -> None:
    actual = ["UP", "DOWN", "SIDEWAYS"]
    predictions = [
        {"UP": 80, "SIDEWAYS": 10, "DOWN": 10},
        {"UP": 10, "SIDEWAYS": 10, "DOWN": 80},
        {"UP": 15, "SIDEWAYS": 70, "DOWN": 15},
    ]
    result = score_probabilities(actual, predictions)
    assert result["observations"] == 3
    assert result["accuracy_pct"] == pytest.approx(100.0)
    assert result["log_loss"] > 0
    assert result["brier_score"] > 0
    assert result["top_label_ece_pct"] == pytest.approx(100.0 * (0.2 * 2 / 3 + 0.3 / 3))
    assert result["confusion_matrix_actual_rows_predicted_columns"]["UP"]["UP"] == 1
    assert result["observed_class_frequency_pct"] == pytest.approx(
        {"UP": 100 / 3, "SIDEWAYS": 100 / 3, "DOWN": 100 / 3}
    )


def test_calibration_uses_only_previous_outcomes_mature_before_current_date() -> None:
    dates = [
        date(2026, 1, 1), date(2026, 1, 6), date(2026, 1, 11),
        date(2026, 1, 16), date(2026, 2, 1),
    ]
    end_dates = [
        date(2026, 1, 31), date(2026, 2, 5), date(2026, 2, 10),
        date(2026, 2, 15), date(2026, 3, 3),
    ]
    actual = ["UP", "DOWN", "SIDEWAYS", "DOWN", "UP"]
    forecasts = [
        {"UP": 70, "SIDEWAYS": 20, "DOWN": 10},
        {"UP": 10, "SIDEWAYS": 20, "DOWN": 70},
        {"UP": 20, "SIDEWAYS": 60, "DOWN": 20},
        {"UP": 20, "SIDEWAYS": 20, "DOWN": 60},
        {"UP": 50, "SIDEWAYS": 30, "DOWN": 20},
    ]

    first = calibrate_with_mature_labels(
        dates, end_dates, actual, forecasts, min_calibration_observations=1
    )
    changed_unmatured_labels = list(actual)
    changed_unmatured_labels[1] = "UP"
    changed_unmatured_labels[2] = "UP"
    changed_unmatured_labels[3] = "UP"
    second = calibrate_with_mature_labels(
        dates, end_dates, changed_unmatured_labels, forecasts,
        min_calibration_observations=1,
    )

    # On Feb 1 only the Jan 1 outcome ended before the forecast date. The
    # Jan 6/11/16 labels have not matured and must not affect its calibrator.
    assert first.mature_training_observations == (0, 0, 0, 0, 1)
    assert first.statuses[:4] == (
        "NO_CALIBRATION_DATA", "NO_CALIBRATION_DATA",
        "NO_CALIBRATION_DATA", "NO_CALIBRATION_DATA",
    )
    assert first.statuses[4] == "FITTED"
    assert first.temperatures[4] == pytest.approx(second.temperatures[4])
    assert first.probabilities_pct[4] == pytest.approx(second.probabilities_pct[4])


def test_block_bootstrap_is_deterministic_and_flags_small_samples() -> None:
    values = [float(i % 3) for i in range(30)]
    result_a = moving_block_bootstrap_ci(values, block_length=5, replicates=200, seed=42)
    result_b = moving_block_bootstrap_ci(values, block_length=5, replicates=200, seed=42)
    assert result_a == result_b
    assert result_a["status"] == "ESTIMATED_CIRCULAR_MOVING_BLOCK_BOOTSTRAP"
    assert result_a["ci95_lower"] <= result_a["estimate"] <= result_a["ci95_upper"]

    small = moving_block_bootstrap_ci(values[:9], block_length=5, replicates=200)
    assert small["ci95_lower"] is None
    assert small["status"] == "NOT_REPORTED_FEWER_THAN_10_OBSERVATIONS"


def test_paired_model_comparison_is_zero_for_identical_predictions() -> None:
    actual = ["UP", "SIDEWAYS", "DOWN"] * 10
    probabilities = [
        {"UP": 60, "SIDEWAYS": 25, "DOWN": 15},
        {"UP": 20, "SIDEWAYS": 60, "DOWN": 20},
        {"UP": 15, "SIDEWAYS": 20, "DOWN": 65},
    ] * 10
    result = compare_probabilities_paired(
        actual, probabilities, probabilities, block_length=5, replicates=200, seed=7
    )
    for item in result.values():
        assert item["estimate"] == pytest.approx(0.0)
        assert item["ci95_lower"] == pytest.approx(0.0)
        assert item["ci95_upper"] == pytest.approx(0.0)
