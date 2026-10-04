from __future__ import annotations

from analysis.method_combination import (
    CombinationMethodInput,
    combine_method_probabilities,
)


def _method(
    name: str,
    probabilities: dict[str, float],
    *,
    expected_return: float = 0.0,
    evidence: float = 0.8,
    reliability: float = 0.8,
    effective_n: float = 25.0,
    stable: bool = True,
    trend: str = "NO_CLEAR_TREND",
    limited: bool = False,
):
    return CombinationMethodInput(
        method=name,
        probabilities_pct=probabilities,
        expected_return_pct=expected_return,
        evidence_score=evidence,
        reliability=reliability,
        effective_sample_size=effective_n,
        stable=stable,
        trend=trend,
        limited=limited,
    )


def test_combination_uses_evidence_adaptive_weights() -> None:
    result = combine_method_probabilities(
        [
            _method("A", {"UP": 80.0, "SIDEWAYS": 10.0, "DOWN": 10.0}, evidence=0.9, reliability=0.9, effective_n=100, trend="UP"),
            _method("B", {"UP": 20.0, "SIDEWAYS": 20.0, "DOWN": 60.0}, evidence=0.3, reliability=0.5, effective_n=9, trend="DOWN"),
        ]
    )
    assert result.limited is False
    assert result.method_weights["A"] > result.method_weights["B"]
    assert result.method_weight_shares_pct["A"] > result.method_weight_shares_pct["B"]
    assert abs(sum(result.probabilities_pct.values()) - 100.0) < 1e-9
    assert result.method_conflict is True


def test_combination_is_not_fixed_half_and_half() -> None:
    result = combine_method_probabilities(
        [
            _method("A", {"UP": 90.0, "SIDEWAYS": 5.0, "DOWN": 5.0}, evidence=0.95, reliability=0.95, effective_n=100, trend="UP"),
            _method("B", {"UP": 10.0, "SIDEWAYS": 10.0, "DOWN": 80.0}, evidence=0.2, reliability=0.4, effective_n=4, trend="DOWN"),
        ]
    )
    assert result.method_weight_shares_pct["A"] != 50.0
    assert result.method_weight_shares_pct["B"] != 50.0


def test_combination_ignores_limited_method() -> None:
    result = combine_method_probabilities(
        [
            _method("A", {"UP": 70.0, "SIDEWAYS": 20.0, "DOWN": 10.0}, trend="UP"),
            _method("B", {"UP": 1.0, "SIDEWAYS": 1.0, "DOWN": 98.0}, limited=True, trend="DOWN"),
        ]
    )
    assert result.usable_methods == ("A",)
    assert result.probabilities_pct == {"UP": 70.0, "SIDEWAYS": 20.0, "DOWN": 10.0}
    assert result.method_conflict is False


def test_combination_rejects_duplicate_method() -> None:
    result = combine_method_probabilities(
        [
            _method("A", {"UP": 60.0, "SIDEWAYS": 30.0, "DOWN": 10.0}, trend="UP"),
            _method("A", {"UP": 10.0, "SIDEWAYS": 20.0, "DOWN": 70.0}, trend="DOWN"),
        ]
    )
    assert result.limited is False
    assert result.usable_methods == ("A",)
    assert any("Duplicate usable prediction method" in item for item in result.limitations)


def test_combination_fails_closed_when_all_weights_zero() -> None:
    result = combine_method_probabilities(
        [
            _method("A", {"UP": 60.0, "SIDEWAYS": 30.0, "DOWN": 10.0}, evidence=0.0, reliability=0.0),
            _method("B", {"UP": 10.0, "SIDEWAYS": 20.0, "DOWN": 70.0}, evidence=0.0, reliability=0.0),
        ]
    )
    assert result.limited is True
    assert result.probabilities_pct == {"UP": 0.0, "SIDEWAYS": 0.0, "DOWN": 0.0}


def test_combination_preserves_conflict_even_when_weights_differ() -> None:
    result = combine_method_probabilities(
        [
            _method("A", {"UP": 95.0, "SIDEWAYS": 3.0, "DOWN": 2.0}, evidence=0.99, reliability=0.99, effective_n=200, trend="UP"),
            _method("B", {"UP": 2.0, "SIDEWAYS": 3.0, "DOWN": 95.0}, evidence=0.1, reliability=0.2, effective_n=4, trend="DOWN"),
        ]
    )
    assert result.method_conflict is True
    assert result.method_agreement is False
    assert result.directional_methods == ("A", "B")


def test_combination_records_weighting_rule_and_stability() -> None:
    result = combine_method_probabilities(
        [
            _method("A", {"UP": 60.0, "SIDEWAYS": 25.0, "DOWN": 15.0}, trend="UP", stable=True),
            _method("B", {"UP": 65.0, "SIDEWAYS": 20.0, "DOWN": 15.0}, trend="UP", stable=False),
        ]
    )
    assert "evidence_score" in result.weighting_rule
    assert result.stable_methods == ("A",)


def test_malformed_distribution_is_excluded_and_audited() -> None:
    result = combine_method_probabilities(
        [
            _method("A", {"UP": 60.0, "SIDEWAYS": 30.0, "DOWN": 5.0}, trend="UP"),
            _method("B", {"UP": 20.0, "SIDEWAYS": 20.0, "DOWN": 60.0}, trend="DOWN"),
        ]
    )
    assert result.usable_methods == ("B",)
    assert any("does not sum to 100%" in item for item in result.limitations)
