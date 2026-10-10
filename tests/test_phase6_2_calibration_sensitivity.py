from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

import analysis.phase6_probability_metrics as metrics
from analysis.phase6_probability_metrics import (
    calibrate_with_mature_labels,
    summarize_temperature_boundaries,
)
from phase6_2_calibration_sensitivity import grid_size_for_maximum


def test_boundary_summary_uses_only_fitted_calibrators() -> None:
    result = summarize_temperature_boundaries(
        [None, 0.25, 1.0, 4.0, 4.000001, 1.0],
        [
            "NO_CALIBRATION_DATA",
            "FITTED",
            "FITTED",
            "FITTED",
            "FITTED",
            "INSUFFICIENT_CALIBRATION_DATA",
        ],
        temperature_min=0.25,
        temperature_max=4.0,
    )

    assert result["fitted_observations"] == 4
    assert result["lower_boundary_hits"] == 1
    assert result["upper_boundary_hits"] == 2
    assert result["lower_boundary_hit_rate_pct"] == pytest.approx(25.0)
    assert result["upper_boundary_hit_rate_pct"] == pytest.approx(50.0)
    assert result["any_boundary_hits"] == 3
    assert result["any_boundary_hit_rate_pct"] == pytest.approx(75.0)


def test_calibration_sensitivity_passes_candidate_temperature_range(monkeypatch) -> None:
    seen: list[dict[str, float | int]] = []

    class FakeTemperature:
        def __init__(self, history, *, min_observations, temperature_min, temperature_max, grid_size):
            seen.append({
                "history": len(history),
                "min_observations": min_observations,
                "temperature_min": temperature_min,
                "temperature_max": temperature_max,
                "grid_size": grid_size,
            })
            self.status = "FITTED" if len(history) >= min_observations else (
                "NO_CALIBRATION_DATA" if not history else "INSUFFICIENT_CALIBRATION_DATA"
            )
            self.temperature = temperature_max if self.status == "FITTED" else 1.0

        def apply(self, probabilities):
            return dict(probabilities)

    monkeypatch.setattr(metrics, "fit_temperature", FakeTemperature)

    dates = [date(2026, 1, 1), date(2026, 1, 5), date(2026, 2, 1)]
    ends = [date(2026, 1, 2), date(2026, 1, 20), date(2026, 2, 10)]
    actual = ["UP", "DOWN", "SIDEWAYS"]
    forecasts = [
        {"UP": 70, "SIDEWAYS": 20, "DOWN": 10},
        {"UP": 10, "SIDEWAYS": 20, "DOWN": 70},
        {"UP": 20, "SIDEWAYS": 60, "DOWN": 20},
    ]

    result = calibrate_with_mature_labels(
        dates,
        ends,
        actual,
        forecasts,
        min_calibration_observations=1,
        temperature_min=0.25,
        temperature_max=8.0,
        temperature_grid_size=201,
    )

    assert result.statuses == (
        "NO_CALIBRATION_DATA",
        "FITTED",
        "FITTED",
    )
    assert result.mature_training_observations == (0, 1, 2)
    assert result.temperatures == pytest.approx((1.0, 8.0, 8.0))
    assert seen
    assert all(item["temperature_max"] == 8.0 for item in seen)
    assert all(item["grid_size"] == 201 for item in seen)


def test_grid_sizes_preserve_log_temperature_resolution() -> None:
    assert grid_size_for_maximum(0.25, 4.0, reference_max=4.0, reference_grid_size=161) == 161
    assert grid_size_for_maximum(0.25, 8.0, reference_max=4.0, reference_grid_size=161) == 201
    assert grid_size_for_maximum(0.25, 16.0, reference_max=4.0, reference_grid_size=161) == 241


def test_boundary_summary_handles_no_fitted_rows() -> None:
    result = summarize_temperature_boundaries(
        [1.0, None],
        ["INSUFFICIENT_CALIBRATION_DATA", "METHOD_UNAVAILABLE"],
        temperature_min=0.25,
        temperature_max=8.0,
    )
    assert result["fitted_observations"] == 0
    assert result["boundary_status"] == "NO_FITTED_CALIBRATORS"
    assert result["upper_boundary_hit_rate_pct"] is None
