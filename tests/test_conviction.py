from __future__ import annotations

from analysis.conviction import assess_conviction
from analysis.decision import DecisionResult
from analysis.method_combination import CombinationMethodInput, combine_method_probabilities


def _method(
    name: str,
    trend: str,
    *,
    stable: bool = True,
    limited: bool = False,
    evidence: float = 0.8,
    reliability: float = 0.8,
    effective_n: float = 25.0,
):
    return CombinationMethodInput(
        method=name,
        probabilities_pct={"UP": 70.0, "SIDEWAYS": 20.0, "DOWN": 10.0} if trend == "UP" else {"UP": 10.0, "SIDEWAYS": 20.0, "DOWN": 70.0},
        expected_return_pct=4.0 if trend == "UP" else -4.0,
        evidence_score=evidence,
        reliability=reliability,
        effective_sample_size=effective_n,
        stable=stable,
        trend=trend,
        limited=limited,
    )


def _decision(trend: str, *, limited: bool = False, lift: float = 12.0, margin: float = 25.0):
    return DecisionResult(
        trend=trend,
        selected_class=trend if trend in {"UP", "SIDEWAYS", "DOWN"} else None,
        probability_pct=70.0,
        baseline_probability_pct=33.0,
        lift_pct=lift,
        margin_pct=margin,
        uncertainty_pct=5.0,
        effective_sample_size=25.0,
        reason="test",
        limited=limited,
    )


def _combination(*methods):
    return combine_method_probabilities(list(methods))


def test_strong_requires_agreement_stability_and_validation() -> None:
    result = assess_conviction(
        trend="UP",
        combination=_combination(_method("A", "UP"), _method("B", "UP")),
        decision=_decision("UP"),
        validated_evidence=True,
    )
    assert result.conviction == "STRONG"
    assert result.method_agreement is True
    assert result.all_usable_methods_stable is True


def test_without_validated_evidence_strong_is_capped_at_moderate() -> None:
    result = assess_conviction(
        trend="UP",
        combination=_combination(_method("A", "UP"), _method("B", "UP")),
        decision=_decision("UP"),
        validated_evidence=False,
    )
    assert result.conviction == "MODERATE"


def test_unstable_method_blocks_strong() -> None:
    result = assess_conviction(
        trend="UP",
        combination=_combination(_method("A", "UP", stable=True), _method("B", "UP", stable=False)),
        decision=_decision("UP"),
        validated_evidence=True,
    )
    assert result.conviction == "MODERATE"
    assert result.all_usable_methods_stable is False


def test_single_directional_method_caps_at_moderate() -> None:
    result = assess_conviction(
        trend="UP",
        combination=_combination(_method("A", "UP")),
        decision=_decision("UP"),
        validated_evidence=True,
    )
    assert result.conviction == "MODERATE"
    assert result.directional_method_count == 1


def test_sideways_is_low_conviction_not_strong() -> None:
    method = CombinationMethodInput(
        method="A",
        probabilities_pct={"UP": 10.0, "SIDEWAYS": 80.0, "DOWN": 10.0},
        expected_return_pct=0.0,
        evidence_score=0.9,
        reliability=0.9,
        effective_sample_size=100.0,
        stable=True,
        trend="SIDEWAYS",
    )
    result = assess_conviction(
        trend="SIDEWAYS",
        combination=_combination(method),
        decision=_decision("SIDEWAYS"),
        validated_evidence=True,
    )
    assert result.conviction == "LOW"


def test_method_conflict_forces_none() -> None:
    result = assess_conviction(
        trend="NO_CLEAR_TREND",
        combination=_combination(_method("A", "UP"), _method("B", "DOWN")),
        decision=_decision("NO_CLEAR_TREND"),
        validated_evidence=True,
    )
    assert result.conviction == "NONE"
    assert result.method_conflict is True


def test_limited_evidence_forces_none() -> None:
    result = assess_conviction(
        trend="UP",
        combination=_combination(_method("A", "UP", limited=True)),
        decision=_decision("NO_CLEAR_TREND", limited=True),
        validated_evidence=True,
    )
    assert result.conviction == "NONE"
    assert result.decision_cleared is False


def test_uncleared_decision_forces_none_even_with_consensus() -> None:
    result = assess_conviction(
        trend="UP",
        combination=_combination(_method("A", "UP"), _method("B", "UP")),
        decision=_decision("NO_CLEAR_TREND"),
        validated_evidence=True,
    )
    assert result.conviction == "NONE"


def test_conviction_is_not_probability() -> None:
    result = assess_conviction(
        trend="UP",
        combination=_combination(_method("A", "UP"), _method("B", "UP")),
        decision=_decision("UP"),
        validated_evidence=False,
    )
    data = result.as_dict()
    assert "probability" not in data
    assert "confidence" not in data
    assert result.conviction == "MODERATE"
