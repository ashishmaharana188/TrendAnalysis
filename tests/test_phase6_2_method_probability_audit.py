from datetime import date

from analysis.phase5_snapshot import SNAPSHOT_SCHEMA_VERSION, default_snapshot_path
from analysis.real_prediction_validation import PredictionFoldResult


def _fold(**overrides):
    values = {
        "prediction_date": date(2026, 8, 24),
        "observed_trend": "UP",
        "actual_return_pct": 1.5,
        "actual_class": "UP",
        "predicted_trend": "UP",
        "conviction": "MODERATE",
        "probabilities_pct": {"UP": 45.0, "SIDEWAYS": 30.0, "DOWN": 25.0},
        "baseline_probabilities_pct": {"UP": 33.33, "SIDEWAYS": 33.34, "DOWN": 33.33},
        "expected_return_pct": 0.7,
        "training_observations": 120,
        "method_a_trend": "UP",
        "method_b_trend": "DOWN",
        "method_agreement": False,
        "limited": False,
        "provenance_clean": True,
        "hit": True,
        "trade_eligible": True,
        "trade_reason": "DIRECTIONAL_AGREEMENT",
        "method_a_probabilities_pct": {"UP": 52.0, "SIDEWAYS": 28.0, "DOWN": 20.0},
        "method_b_probabilities_pct": {"UP": 38.0, "SIDEWAYS": 32.0, "DOWN": 30.0},
    }
    values.update(overrides)
    return PredictionFoldResult(**values)


def test_fold_serializes_exact_method_probability_vectors():
    payload = _fold().as_dict()
    assert payload["method_a_probabilities_pct"] == {
        "UP": 52.0, "SIDEWAYS": 28.0, "DOWN": 20.0
    }
    assert payload["method_b_probabilities_pct"] == {
        "UP": 38.0, "SIDEWAYS": 32.0, "DOWN": 30.0
    }
    # Instrumentation must not replace or recalculate the combined vector.
    assert payload["probabilities_pct"] == {
        "UP": 45.0, "SIDEWAYS": 30.0, "DOWN": 25.0
    }


def test_unavailable_method_serializes_as_empty_vector_not_inferred_data():
    payload = _fold(
        method_a_trend=None,
        method_b_trend=None,
        method_a_probabilities_pct={},
        method_b_probabilities_pct={},
    ).as_dict()
    assert payload["method_a_probabilities_pct"] == {}
    assert payload["method_b_probabilities_pct"] == {}


def test_new_snapshots_use_schema_v2_and_default_folder_suffix():
    assert SNAPSHOT_SCHEMA_VERSION == "2"

    class Config:
        ticker = "RELIANCE"
        benchmark = "Nifty_50"
        analysis_timeframe = "6M"
        holding_period_months = 1.0

    assert default_snapshot_path(Config()).name == "RELIANCE_Nifty_50_6M_1M_v2"
