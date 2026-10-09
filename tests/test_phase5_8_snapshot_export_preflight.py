from __future__ import annotations

import json
import sys
import tempfile
import types
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The delivery package intentionally contains the snapshot component, not the
# full TrendAnalysis repository. Provide the one constant needed by the module
# so this test remains completely OLAP-free.
outcome_stub = types.ModuleType("analysis.outcome_labels")
outcome_stub.OUTCOME_CLASSES = ("UP", "SIDEWAYS", "DOWN")
sys.modules.setdefault("analysis.outcome_labels", outcome_stub)

from analysis.phase5_snapshot import (  # noqa: E402
    default_snapshot_path,
    export_phase5_snapshot,
    load_phase5_snapshot,
)


@dataclass(frozen=True)
class FakeConfig:
    ticker: str = "RELIANCE"
    benchmark: str = "Nifty_50"
    analysis_timeframe: str = "6M"
    holding_period_months: float = 1.0
    entry_mode: str = "next_trading_day"


@dataclass(frozen=True)
class FakeObservation:
    as_of_date: date
    target: str
    scope: str
    states: dict[str, str]
    stock_return_pct: float
    benchmark_return_pct: float
    relative_return_pct: float
    outcome_end_date: date


@dataclass(frozen=True)
class FakeFold:
    prediction_date: date
    actual_return_pct: float
    actual_class: str
    predicted_trend: str
    observed_trend: str
    probabilities_pct: dict[str, float]

    def as_dict(self):
        return {
            "prediction_date": self.prediction_date.isoformat(),
            "actual_return_pct": self.actual_return_pct,
            "actual_class": self.actual_class,
            "predicted_trend": self.predicted_trend,
            "observed_trend": self.observed_trend,
            "probabilities_pct": self.probabilities_pct,
        }


@dataclass(frozen=True)
class FakeResult:
    prediction_folds: tuple[FakeFold, ...]
    panel_observations: tuple[FakeObservation, ...]
    candidate_predictions: int = 1
    evaluated_predictions: int = 1
    selection_candidate_evaluations: int = 1
    selection_validated_predictions: int = 1
    multiple_testing_controlled_folds: int = 1
    latest_validated_prediction_date: date = date(2026, 8, 24)
    latest_market_date: date = date(2026, 10, 1)
    first_prediction_date: date = date(2026, 8, 24)

    def as_dict(self):
        return {
            "candidate_predictions": self.candidate_predictions,
            "evaluated_predictions": self.evaluated_predictions,
            "latest_validated_prediction_date": self.latest_validated_prediction_date,
            "latest_market_date": self.latest_market_date,
        }


def ticker_loader(_ticker: str) -> pa.Table:
    return pa.table({
        "ticker": ["RELIANCE"],
        "report_date": [date(2026, 8, 24)],
        "open": [100.0],
        "high": [101.0],
        "low": [99.0],
        "close": [100.5],
        "volume": [1000.0],
    })


def empty_benchmark_loader(_benchmark: str) -> pa.Table:
    return pa.table({
        "report_date": pa.array([], type=pa.date32()),
        "close": pa.array([], type=pa.float64()),
    })


def nonempty_benchmark_loader(_benchmark: str) -> pa.Table:
    return pa.table({
        "indicator": ["Nifty_50"],
        "report_date": [date(2026, 8, 24)],
        "open": [200.0],
        "high": [201.0],
        "low": [199.0],
        "close": [200.5],
        "volume": [2000.0],
    })


def fake_result() -> FakeResult:
    observation = FakeObservation(
        as_of_date=date(2026, 8, 24),
        target="RELIANCE",
        scope="company",
        states={"company.market.price": "Rising / High"},
        stock_return_pct=1.5,
        benchmark_return_pct=0.5,
        relative_return_pct=1.0,
        outcome_end_date=date(2026, 9, 24),
    )
    fold = FakeFold(
        prediction_date=date(2026, 8, 24),
        actual_return_pct=1.5,
        actual_class="UP",
        predicted_trend="UP",
        observed_trend="UP",
        probabilities_pct={"UP": 60.0, "SIDEWAYS": 25.0, "DOWN": 15.0},
    )
    return FakeResult((fold,), (observation,))


def main() -> None:
    config = FakeConfig()
    result = fake_result()

    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        destination = base / "empty-benchmark"
        exported = export_phase5_snapshot(
            result,
            config,
            destination,
            repository_root=REPO_ROOT,
            ticker_market_loader=ticker_loader,
            benchmark_market_loader=empty_benchmark_loader,
        )
        loaded = load_phase5_snapshot(exported)
        assert loaded.market_daily.num_rows == 1

        destination2 = base / "nonempty-benchmark"
        exported2 = export_phase5_snapshot(
            result,
            config,
            destination2,
            repository_root=REPO_ROOT,
            ticker_market_loader=ticker_loader,
            benchmark_market_loader=nonempty_benchmark_loader,
        )
        loaded2 = load_phase5_snapshot(exported2)
        assert loaded2.market_daily.num_rows == 2
        assert loaded2.market_daily["instrument"].to_pylist() == ["Nifty_50", "RELIANCE"]

        assert default_snapshot_path(config).name == (
            "RELIANCE_Nifty_50_6M_1M_v2"
        )

    print("PHASE 5.8 SNAPSHOT EXPORT PREFLIGHT: PASS")
    print("Empty benchmark schema case: PASS")
    print("Non-empty benchmark schema case: PASS")
    print("Snapshot round-trip: PASS")
    print("No OLAP query executed.")


if __name__ == "__main__":
    main()
