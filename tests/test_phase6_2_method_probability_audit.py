from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pyarrow as pa
import pytest

from analysis.phase5_snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    export_phase5_snapshot,
    load_phase5_snapshot,
)
from analysis.real_prediction_validation import PredictionFoldResult
from phase6_2_method_probability_audit import load_folds


METHOD_A = {"UP": 67.0, "SIDEWAYS": 21.0, "DOWN": 12.0}
METHOD_B = {"UP": 14.0, "SIDEWAYS": 26.0, "DOWN": 60.0}
COMBINED = {"UP": 45.0, "SIDEWAYS": 30.0, "DOWN": 25.0}
BASELINE = {"UP": 34.0, "SIDEWAYS": 33.0, "DOWN": 33.0}


@dataclass(frozen=True)
class FakeConfig:
    ticker: str = "RELIANCE"
    benchmark: str = "Nifty_50"
    analysis_timeframe: str = "6M"
    holding_period_months: float = 1.0
    entry_mode: str = "next_trading_day"


@dataclass(frozen=True)
class FakeObservation:
    as_of_date: date = date(2026, 8, 24)
    target: str = "RELIANCE"
    scope: str = "company"
    states: dict[str, str] | None = None
    stock_return_pct: float = 1.5
    benchmark_return_pct: float = 0.5
    relative_return_pct: float = 1.0
    outcome_end_date: date = date(2026, 9, 24)

    def __post_init__(self) -> None:
        if self.states is None:
            object.__setattr__(self, "states", {"company.market.price": "Rising / High"})


@dataclass(frozen=True)
class FakeResult:
    prediction_folds: tuple[PredictionFoldResult, ...]
    panel_observations: tuple[FakeObservation, ...]
    candidate_predictions: int = 1
    evaluated_predictions: int = 1
    selection_candidate_evaluations: int = 1
    selection_validated_predictions: int = 1
    multiple_testing_controlled_folds: int = 1
    latest_validated_prediction_date: date = date(2026, 8, 24)
    latest_market_date: date = date(2026, 10, 1)
    first_prediction_date: date = date(2026, 8, 24)

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_predictions": self.candidate_predictions,
            "evaluated_predictions": self.evaluated_predictions,
            "latest_validated_prediction_date": self.latest_validated_prediction_date,
            "latest_market_date": self.latest_market_date,
        }


def _fold(
    method_a: dict[str, float] | None = METHOD_A,
    method_b: dict[str, float] | None = METHOD_B,
) -> PredictionFoldResult:
    return PredictionFoldResult(
        prediction_date=date(2026, 8, 24),
        observed_trend="UP",
        actual_return_pct=1.5,
        actual_class="UP",
        predicted_trend="UP",
        conviction="MODERATE",
        probabilities_pct=dict(COMBINED),
        baseline_probabilities_pct=dict(BASELINE),
        expected_return_pct=0.8,
        training_observations=100,
        method_a_trend="UP" if method_a else None,
        method_b_trend="DOWN" if method_b else None,
        method_agreement=False,
        limited=False,
        provenance_clean=True,
        hit=True,
        trade_eligible=True,
        trade_reason="TEST_FIXTURE",
        method_a_probabilities_pct=dict(method_a or {}),
        method_b_probabilities_pct=dict(method_b or {}),
    )


def _market_loader(instrument: str) -> pa.Table:
    return pa.table(
        {
            "report_date": [date(2026, 10, 1)],
            "open": [100.0 if instrument == "RELIANCE" else 200.0],
            "high": [101.0 if instrument == "RELIANCE" else 201.0],
            "low": [99.0 if instrument == "RELIANCE" else 199.0],
            "close": [100.5 if instrument == "RELIANCE" else 200.5],
            "volume": [1000.0],
        }
    )


def _export_snapshot(destination: Path, method_a=METHOD_A, method_b=METHOD_B) -> Path:
    result = FakeResult(
        prediction_folds=(_fold(method_a, method_b),),
        panel_observations=(FakeObservation(),),
    )
    return export_phase5_snapshot(
        result,
        FakeConfig(),
        destination,
        repository_root=Path(__file__).resolve().parents[1],
        ticker_market_loader=_market_loader,
        benchmark_market_loader=_market_loader,
    )


def test_fold_result_serializes_exact_method_probability_vectors() -> None:
    row = _fold().as_dict()
    assert row["method_a_probabilities_pct"] == METHOD_A
    assert row["method_b_probabilities_pct"] == METHOD_B


def test_method_probability_vectors_survive_snapshot_round_trip_and_reach_audit() -> None:
    with tempfile.TemporaryDirectory() as temp:
        snapshot_path = _export_snapshot(Path(temp) / "RELIANCE_Nifty_50_6M_1M_v2")
        snapshot = load_phase5_snapshot(snapshot_path)

        assert snapshot.manifest["schema_version"] == SNAPSHOT_SCHEMA_VERSION == "2"
        assert snapshot.manifest["status"] == "FROZEN"
        assert len(snapshot.prediction_folds) == 1

        serialized = snapshot.prediction_folds[0]
        assert serialized["method_a_probabilities_pct"] == METHOD_A
        assert serialized["method_b_probabilities_pct"] == METHOD_B

        audit_folds = load_folds(str(snapshot_path))
        assert len(audit_folds) == 1
        # The audit loader normalizes distributions; tolerate harmless float drift.
        assert audit_folds[0].method_a_probabilities_pct == pytest.approx(METHOD_A)
        assert audit_folds[0].method_b_probabilities_pct == pytest.approx(METHOD_B)


def test_unavailable_method_is_not_given_fabricated_probabilities() -> None:
    with tempfile.TemporaryDirectory() as temp:
        snapshot_path = _export_snapshot(
            Path(temp) / "RELIANCE_Nifty_50_6M_1M_v2",
            method_a=METHOD_A,
            method_b=None,
        )
        snapshot = load_phase5_snapshot(snapshot_path)
        row = snapshot.prediction_folds[0]
        assert row["method_a_probabilities_pct"] == METHOD_A
        assert row["method_b_probabilities_pct"] == {}

        audit_folds = load_folds(str(snapshot_path))
        assert audit_folds[0].method_a_probabilities_pct == METHOD_A
        assert audit_folds[0].method_b_probabilities_pct is None


def test_v1_manifest_remains_loadable_after_schema_v2_upgrade() -> None:
    with tempfile.TemporaryDirectory() as temp:
        snapshot_path = _export_snapshot(Path(temp) / "snapshot_v2")
        manifest_path = snapshot_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = "1"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        loaded = load_phase5_snapshot(snapshot_path)
        assert loaded.manifest["schema_version"] == "1"
        assert loaded.prediction_folds[0]["method_a_probabilities_pct"] == METHOD_A
        assert loaded.prediction_folds[0]["method_b_probabilities_pct"] == METHOD_B
