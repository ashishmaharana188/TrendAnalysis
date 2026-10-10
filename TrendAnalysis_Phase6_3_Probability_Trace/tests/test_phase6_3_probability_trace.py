from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from analysis.decision import DecisionResult
from analysis.prediction import PredictionEngine


BASELINE = {"UP": 33.0, "SIDEWAYS": 34.0, "DOWN": 33.0}


def _fake_probability(*, method_b: bool):
    values = {
        "probabilities_pct": {"UP": 50.0, "SIDEWAYS": 25.0, "DOWN": 25.0},
        "expected_return_pct": 1.2,
        "baseline_return_pct": 0.2,
        "return_lift_pct": 1.0,
        "evidence_score": 0.8,
        "reliability": 0.9,
        "sample_count": 4,
        "effective_sample_size": 4.0,
        "stable": True,
        "variables": ("company.market.price",),
        "condition": ("company.market.price=Rising",),
        "class_counts": {"UP": 2, "SIDEWAYS": 1, "DOWN": 1},
        "weighted_class_counts": {"UP": 2.0, "SIDEWAYS": 1.0, "DOWN": 1.0},
        "probability_basis": "synthetic_weighted_class_share_test",
        "limited": False,
        "limitations": (),
    }
    if method_b:
        values.update({
            "exact_condition_count": 2,
            "weight_concentration": 0.25,
            "stability_score": 0.8,
        })
    return SimpleNamespace(**values)


def _decision():
    return DecisionResult(
        trend="UP",
        selected_class="UP",
        probability_pct=50.0,
        baseline_probability_pct=33.0,
        lift_pct=17.0,
        margin_pct=25.0,
        uncertainty_pct=50.0,
        effective_sample_size=4.0,
        reason="synthetic test decision",
        limited=False,
    )


def _assert_trace(output: dict, *, method_b: bool) -> None:
    assert output["class_counts"] == {"UP": 2, "SIDEWAYS": 1, "DOWN": 1}
    assert output["weighted_class_counts"] == {"UP": 2.0, "SIDEWAYS": 1.0, "DOWN": 1.0}
    assert output["probability_basis"] == "synthetic_weighted_class_share_test"
    if method_b:
        assert output["exact_condition_count"] == 2
        assert output["weight_concentration"] == 0.25
        assert output["stability_score"] == 0.8
    else:
        assert output["exact_condition_count"] is None
        assert output["weight_concentration"] is None
        assert output["stability_score"] is None


def test_method_a_builder_trace_survives_method_prediction_wrapper():
    ranking = SimpleNamespace(method_results=[SimpleNamespace(score=1.0)])
    with patch("analysis.prediction.build_method_a_probability", return_value=_fake_probability(method_b=False)), patch(
        "analysis.prediction.decide_baseline_relative", return_value=_decision()
    ):
        method = PredictionEngine()._method_a_prediction(ranking, None, BASELINE)
    assert method is not None
    _assert_trace(method.as_dict(), method_b=False)


def test_method_b_builder_trace_survives_method_prediction_wrapper():
    best = SimpleNamespace(score=1.0, supporting_observations=[("2020-01-01", 1.0)])
    ranking = SimpleNamespace(method_results=[best])
    with patch("analysis.prediction.build_method_b_probability", return_value=_fake_probability(method_b=True)), patch(
        "analysis.prediction.decide_baseline_relative", return_value=_decision()
    ):
        method = PredictionEngine()._method_prediction(ranking, [], None, BASELINE)
    assert method is not None
    _assert_trace(method.as_dict(), method_b=True)
