from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from phase6_2_probability_smoothing_sensitivity import (
    chronological_split_indices,
    probability_zero_diagnostics,
    run_sensitivity,
    smooth_distribution,
)


def test_alpha_zero_preserves_distribution() -> None:
    raw = {"UP": 25.0, "SIDEWAYS": 25.0, "DOWN": 50.0}
    result = smooth_distribution(raw, 0.0)
    assert result == pytest.approx(raw)


def test_positive_alpha_removes_exact_zero_without_changing_total_mass() -> None:
    raw = {"UP": 0.0, "SIDEWAYS": 60.0, "DOWN": 40.0}
    result = smooth_distribution(raw, 0.001)
    assert result is not None
    assert result["UP"] > 0.0
    assert result["UP"] == pytest.approx(0.001 / 1.003 * 100.0)
    assert sum(result.values()) == pytest.approx(100.0)
    assert result["SIDEWAYS"] > result["DOWN"] > result["UP"]


def test_smoothing_rejects_negative_alpha_and_unusable_vector() -> None:
    with pytest.raises(ValueError):
        smooth_distribution({"UP": 20, "SIDEWAYS": 30, "DOWN": 50}, -0.01)
    assert smooth_distribution({"UP": 0, "SIDEWAYS": 0, "DOWN": 0}, 0.01) is None


def test_probability_zero_diagnostics_counts_exact_and_near_zero_true_class_mass() -> None:
    rows = [
        SimpleNamespace(
            prediction_date=date(2026, 1, 1),
            actual_class="UP",
            probabilities_pct={"UP": 0.0, "SIDEWAYS": 60.0, "DOWN": 40.0},
            method_a_selection_status="SELECTED",
        ),
        SimpleNamespace(
            prediction_date=date(2026, 1, 6),
            actual_class="DOWN",
            probabilities_pct={"UP": 99.99999, "SIDEWAYS": 0.00001, "DOWN": 0.0},
            method_a_selection_status="SELECTED",
        ),
    ]
    result = probability_zero_diagnostics(
        rows,
        "probabilities_pct",
        near_zero_thresholds=(0.0, 1e-6),
        selection_status_field="method_a_selection_status",
    )
    assert result["status"] == "COMPLETE"
    assert result["available_observations"] == 2
    assert result["exact_zero_entries_total"] == 2
    assert result["exact_zero_entries_by_class"] == {"UP": 1, "SIDEWAYS": 0, "DOWN": 1}
    assert result["true_class_probability"]["exact_zero_true_class_count"] == 2
    assert result["near_zero_counts"][0]["true_class_probability_at_or_below_count"] == 2
    assert result["near_zero_counts"][1]["true_class_probability_at_or_below_count"] == 2
    assert len(result["lowest_probability_cases"]) == 2
    assert result["selection_status_counts"] == {"SELECTED": 2}


def test_chronological_split_is_ordered_and_validates_fraction() -> None:
    development, holdout = chronological_split_indices(10, 0.6)
    assert development == list(range(6))
    assert holdout == list(range(6, 10))
    with pytest.raises(ValueError):
        chronological_split_indices(10, 1.0)


def _make_rows(count: int = 45):
    rows = []
    start = date(2022, 1, 1)
    classes = ("UP", "SIDEWAYS", "DOWN")
    for index in range(count):
        actual = classes[index % 3]
        probs = {label: 0.0 for label in classes}
        alternatives = [label for label in classes if label != actual]
        if index % 5 == 0:
            probs[alternatives[0]] = 65.0
            probs[alternatives[1]] = 35.0
        else:
            probs[actual] = 70.0
            probs[alternatives[0]] = 20.0
            probs[alternatives[1]] = 10.0
        date_value = start + timedelta(days=index * 5)
        # Outcome maturity is 21 calendar days after its forecast date, so the
        # last few development labels are not available at the holdout boundary.
        row = SimpleNamespace(
            prediction_date=date_value,
            outcome_end_date=date_value + timedelta(days=21),
            actual_class=actual,
            combined_probabilities_pct=dict(probs),
            baseline_probabilities_pct={"UP": 34.0, "SIDEWAYS": 33.0, "DOWN": 33.0},
            method_a_probabilities_pct=(dict(probs) if index >= 3 else None),
            method_b_probabilities_pct=(dict(probs) if index >= 6 else None),
            method_a_selection_status=("SELECTED" if index >= 3 else "NO_SELECTION"),
            method_b_selection_status=("SELECTED" if index >= 6 else "NO_SELECTION"),
        )
        rows.append(row)
    return rows


def test_end_to_end_selection_excludes_unmatured_development_labels(capsys) -> None:
    rows = _make_rows()
    report = run_sensitivity(
        rows,
        alphas=(0.0, 0.001, 0.01),
        development_fraction=0.6,
        min_calibration_observations=3,
        min_development_fitted_observations=5,
        temperature_min=0.25,
        temperature_max=4.0,
        temperature_grid_size=11,
        near_zero_thresholds=(0.0, 1e-6, 1e-3),
        block_length=2,
        bootstrap_replicates=100,
        seed=123,
        top_cases=3,
    )
    capsys.readouterr()

    assert report["snapshot_folds"] == 45
    assert report["development_index_split"] == 27
    # Split occurs at 2022-05-16 (index 27); rows i=0..22 mature strictly
    # before that date, while later development labels are excluded from alpha selection.
    assert report["development_outcomes_matured_before_selection_cutoff"] == 23

    for model_name in ("Combined model", "Method A", "Method B"):
        model = report["methods"][model_name]
        assert model["selection"]["selected_alpha"] in {0.0, 0.001, 0.01}
        assert model["selection"]["selection_outcome_maturity_cutoff_exclusive"] == report["holdout_start_date"]
        assert model["selection"]["development_outcomes_matured_before_holdout_observations"] <= model["selection"]["development_available_observations"]
        if model_name == "Method A":
            assert model["raw_probability_audit"]["selection_status_counts"]["NO_SELECTION"] == 3
        if model_name == "Method B":
            assert model["raw_probability_audit"]["selection_status_counts"]["NO_SELECTION"] == 6
        assert model["holdout"]["selected_alpha_prequential_calibrated"]["observations"] > 0
        assert "selected_calibrated_vs_stored_baseline" in model["holdout"]["paired_comparisons"]
