from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa

from analysis.phase5_snapshot import export_phase5_snapshot, load_phase5_snapshot
from phase6_3_snapshot_tools import trace_audit


@dataclass(frozen=True)
class SyntheticConfig:
    ticker: str = "SYNTHETIC"
    benchmark: str = "SYNTHETIC_BENCHMARK"
    analysis_timeframe: str = "6M"
    holding_period_months: float = 1.0
    entry_mode: str = "next_trading_day"


@dataclass(frozen=True)
class SyntheticObservation:
    as_of_date: date = date(2026, 1, 5)
    target: str = "SYNTHETIC"
    scope: str = "company"
    states: dict[str, str] | None = None
    stock_return_pct: float = 1.5
    benchmark_return_pct: float = 0.5
    relative_return_pct: float = 1.0
    outcome_end_date: date = date(2026, 2, 5)

    def __post_init__(self):
        if self.states is None:
            object.__setattr__(self, "states", {"company.market.price": "Rising / High"})


@dataclass(frozen=True)
class SyntheticFold:
    prediction_date: date = date(2026, 1, 5)

    def as_dict(self) -> dict[str, Any]:
        method_a = {
            "method": "A",
            "probabilities_pct": {"UP": 67.0, "SIDEWAYS": 21.0, "DOWN": 12.0},
            "sample_count": 100,
            "effective_sample_size": 100.0,
            "class_counts": {"UP": 67, "SIDEWAYS": 21, "DOWN": 12},
            "weighted_class_counts": {"UP": 67.0, "SIDEWAYS": 21.0, "DOWN": 12.0},
            "probability_basis": "synthetic_method_a_unit_weight_empirical",
            "limited": False,
            "limitations": [],
        }
        method_b = {
            "method": "B",
            "probabilities_pct": {"UP": 14.0, "SIDEWAYS": 26.0, "DOWN": 60.0},
            "sample_count": 14,
            "effective_sample_size": 11.0,
            "class_counts": {"UP": 2, "SIDEWAYS": 4, "DOWN": 8},
            "weighted_class_counts": {"UP": 14.0, "SIDEWAYS": 26.0, "DOWN": 60.0},
            "probability_basis": "synthetic_method_b_weighted_empirical",
            "exact_condition_count": 2,
            "weight_concentration": 0.12,
            "stability_score": 0.7,
            "limited": False,
            "limitations": [],
        }
        selection_a = {"method": "A", "selection_status": "SELECTED"}
        selection_b = {"method": "B", "selection_status": "SELECTED"}
        return {
            "prediction_date": self.prediction_date.isoformat(),
            "actual_return_pct": 1.5,
            "actual_class": "UP",
            "predicted_trend": "UP",
            "observed_trend": "UP",
            "probabilities_pct": {"UP": 60.0, "SIDEWAYS": 25.0, "DOWN": 15.0},
            "baseline_probabilities_pct": {"UP": 34.0, "SIDEWAYS": 33.0, "DOWN": 33.0},
            "method_a_trend": "UP",
            "method_b_trend": "DOWN",
            "method_a_probabilities_pct": dict(method_a["probabilities_pct"]),
            "method_b_probabilities_pct": dict(method_b["probabilities_pct"]),
            "method_a_output": method_a,
            "method_b_output": method_b,
            "method_a_selection_metadata": selection_a,
            "method_b_selection_metadata": selection_b,
            "selection_metadata": {
                "recording_contract_version": 2,
                "selection_mode": "synthetic_roundtrip_test",
                "method_a": {"prediction_method_output": method_a, **selection_a},
                "method_b": {"prediction_method_output": method_b, **selection_b},
                "nested_selection_fold": None,
                "nested_run_summary": {},
            },
            "outcome_thresholds": {"limited": False},
            "prediction_provenance": {"clean": True},
            "prediction_result": {"trend": "UP", "probabilities_pct": {"UP": 60.0, "SIDEWAYS": 25.0, "DOWN": 15.0}},
            "current_states": {"company.market.price": "Rising / High"},
            "benchmark_return_pct": 0.5,
            "relative_return_pct": 1.0,
            "outcome_end_date": "2026-02-05",
            "fold_recording_contract_version": 2,
        }


@dataclass(frozen=True)
class SyntheticResult:
    prediction_folds: tuple[SyntheticFold, ...]
    panel_observations: tuple[SyntheticObservation, ...]
    candidate_predictions: int = 1
    evaluated_predictions: int = 1
    selection_candidate_evaluations: int = 1
    selection_validated_predictions: int = 1
    multiple_testing_controlled_folds: int = 1
    latest_validated_prediction_date: date = date(2026, 1, 5)
    latest_market_date: date = date(2026, 3, 1)
    first_prediction_date: date = date(2026, 1, 5)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_predictions": self.candidate_predictions,
            "evaluated_predictions": self.evaluated_predictions,
            "latest_validated_prediction_date": self.latest_validated_prediction_date.isoformat(),
            "latest_market_date": self.latest_market_date.isoformat(),
        }


def _market_table(instrument_close: float) -> pa.Table:
    return pa.table({
        "report_date": pa.array([date(2026, 1, 5)], type=pa.date32()),
        "open": [instrument_close - 0.5],
        "high": [instrument_close + 0.5],
        "low": [instrument_close - 1.0],
        "close": [instrument_close],
        "volume": [1000.0],
    })


def test_phase63_trace_survives_export_gzip_load_contract_and_audit(tmp_path: Path) -> None:
    destination = tmp_path / "synthetic_phase63_snapshot"
    config = SyntheticConfig()
    result = SyntheticResult(
        prediction_folds=(SyntheticFold(),),
        panel_observations=(SyntheticObservation(),),
    )
    export_phase5_snapshot(
        result,
        config,
        destination,
        repository_root=Path.cwd(),
        ticker_market_loader=lambda _ticker: _market_table(100.0),
        benchmark_market_loader=lambda _benchmark: _market_table(200.0),
    )

    # Loading validates the manifest, compressed JSONL fold artifact and v2 contract.
    loaded = load_phase5_snapshot(destination)
    assert loaded.manifest["fold_recording_contract_version"] == 2
    assert loaded.prediction_count == 1
    fold = loaded.prediction_folds[0]
    assert fold["method_a_output"]["class_counts"] == {"UP": 67, "SIDEWAYS": 21, "DOWN": 12}
    assert fold["method_a_output"]["weighted_class_counts"] == {"UP": 67.0, "SIDEWAYS": 21.0, "DOWN": 12.0}
    assert fold["method_a_output"]["probability_basis"] == "synthetic_method_a_unit_weight_empirical"
    assert fold["method_b_output"]["class_counts"] == {"UP": 2, "SIDEWAYS": 4, "DOWN": 8}
    assert fold["method_b_output"]["weighted_class_counts"] == {"UP": 14.0, "SIDEWAYS": 26.0, "DOWN": 60.0}
    assert fold["method_b_output"]["probability_basis"] == "synthetic_method_b_weighted_empirical"
    assert fold["method_b_output"]["exact_condition_count"] == 2
    assert fold["method_b_output"]["weight_concentration"] == 0.12
    assert fold["method_b_output"]["stability_score"] == 0.7

    # Audit ingestion re-loads the compressed artifact and checks all traces.
    report = trace_audit(str(destination))
    for method in ("A", "B"):
        summary = report["method_summary"][method]
        assert summary["folds_missing_required_trace_fields"] == 0
        assert summary["folds_with_probability_reconstruction_mismatch"] == 0
