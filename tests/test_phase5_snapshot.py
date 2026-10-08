from __future__ import annotations

import gzip
import json
import sys
import tempfile
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.phase5_snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    default_snapshot_path,
    load_phase5_snapshot,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)

        manifest = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "phase5_baseline_version": "5.8",
            "status": "FROZEN",
            "ticker": "RELIANCE",
            "benchmark": "Nifty_50",
            "analysis_timeframe": "6M",
            "holding_period_months": 1.0,
            "entry_mode": "next_trading_day",
            "candidate_predictions": 1,
            "evaluated_predictions": 1,
            "selection_candidate_evaluations": 42804369,
            "selection_validated_predictions": 189,
            "multiple_testing_controlled_folds": 208,
            "latest_validated_prediction_date": "2026-08-24",
            "latest_market_date": "2026-10-01",
            "prediction_fold_count": 1,
            "panel_observation_count": 1,
            "outcome_classes": ["UP", "SIDEWAYS", "DOWN"],
            "artifact_sha256": {},
            "immutability": {"overwrite_allowed": False},
        }

        folds_path = root / "prediction_folds.jsonl.gz"
        panel_path = root / "panel_observations.jsonl.gz"
        market_path = root / "market_daily.parquet"

        fold = {
            "prediction_date": "2026-08-24",
            "observed_trend": "UP",
            "actual_return_pct": 1.5,
            "actual_class": "UP",
            "predicted_trend": "UP",
            "probabilities_pct": {"UP": 60.0, "SIDEWAYS": 25.0, "DOWN": 15.0},
        }
        panel = {
            "as_of_date": "2026-08-24",
            "target": "RELIANCE",
            "scope": "company",
            "states": {"company.market.price": "Rising / High"},
            "stock_return_pct": 1.5,
            "benchmark_return_pct": 0.4,
            "relative_return_pct": 1.1,
            "outcome_end_date": "2026-09-24",
        }

        with gzip.open(folds_path, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(fold) + "\n")
        with gzip.open(panel_path, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(panel) + "\n")

        market = pa.table(
            {
                "instrument": ["RELIANCE", "Nifty_50"],
                "date": pa.array([date(2026, 8, 24), date(2026, 8, 24)], type=pa.date32()),
                "open": [100.0, 200.0],
                "high": [101.0, 201.0],
                "low": [99.0, 199.0],
                "close": [100.5, 200.5],
                "volume": [1000.0, 2000.0],
            }
        )
        pq.write_table(market, market_path)
        manifest["artifact_sha256"] = {
            path.name: __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            for path in (folds_path, panel_path, market_path)
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        snapshot = load_phase5_snapshot(
            root,
            expected_ticker="RELIANCE",
            expected_benchmark="Nifty_50",
            expected_analysis_timeframe="6M",
            expected_holding_period_months=1.0,
        )

        assert snapshot.manifest["status"] == "FROZEN"
        assert snapshot.prediction_count == 1
        assert len(snapshot.panel_observations) == 1
        assert snapshot.market_daily.num_rows == 2

        path = default_snapshot_path(
            type(
                "Config",
                (),
                {
                    "ticker": "RELIANCE",
                    "benchmark": "Nifty_50",
                    "analysis_timeframe": "6M",
                    "holding_period_months": 1.0,
                },
            )()
        )
        assert path.name == "RELIANCE_Nifty_50_6M_1M_v1"

    print("PHASE 5 SNAPSHOT TEST: PASS")
    print("Manifest validation: PASS")
    print("Fold JSONL validation: PASS")
    print("Panel JSONL validation: PASS")
    print("Market Parquet validation: PASS")
    print("Frozen/immutable contract: PASS")


if __name__ == "__main__":
    main()
