from __future__ import annotations

from datetime import date
from math import nan

from analysis.method_a_prediction import build_method_a_probability
from analysis.outcome_labels import learn_outcome_thresholds
from analysis.relationship import RelationshipResult


def _relationship(
    returns: tuple[float, ...],
    weights: tuple[float, ...] | None = None,
    sample_count: int | None = None,
    method: str = "A",
) -> RelationshipResult:
    support_weights = weights if weights is not None else tuple(1.0 for _ in returns)
    return RelationshipResult(
        method=method,
        variables=("company.market.price",),
        condition=("company.market.price~=Rising / High",),
        sample_count=len(returns) if sample_count is None else sample_count,
        mean_return_pct=sum(returns) / len(returns),
        median_return_pct=sorted(returns)[len(returns) // 2],
        baseline_mean_return_pct=0.5,
        lift_pct=sum(returns) / len(returns) - 0.5,
        positive_rate_pct=sum(value > 0 for value in returns) / len(returns) * 100,
        effect_strength=0.5,
        reliability=0.7,
        score=0.35,
        stable=True,
        supporting_observations=tuple(
            (date(2020, 1, index + 1), value) for index, value in enumerate(returns)
        ),
        supporting_weights=support_weights,
    )


def test_method_a_uses_exact_support_distribution() -> None:
    # Thresholds are learned from the complete historical return distribution.
    thresholds = learn_outcome_thresholds(
        [-6, -5, -4, -1, 0, 1, 4, 5, 6],
        min_observations=9,
    )
    relationship = _relationship((-5.0, 0.0, 6.0, 5.0, -4.0, 1.0))

    result = build_method_a_probability(relationship, thresholds)

    assert not result.limited
    assert result.sample_count == 6
    assert result.effective_sample_size == 6.0
    assert result.class_counts == {"UP": 2, "SIDEWAYS": 2, "DOWN": 2}
    assert result.weighted_class_counts == {"UP": 2.0, "SIDEWAYS": 2.0, "DOWN": 2.0}
    assert all(abs(result.probabilities_pct[label] - (100.0 / 3.0)) < 1e-9 for label in ("UP", "SIDEWAYS", "DOWN"))
    assert abs(sum(result.probabilities_pct.values()) - 100.0) < 1e-9


def test_method_a_weighting_is_explicit_and_auditable() -> None:
    thresholds = learn_outcome_thresholds(
        [-6, -5, -4, -1, 0, 1, 4, 5, 6],
        min_observations=9,
    )
    relationship = _relationship(
        (-5.0, 6.0, 5.0),
        weights=(1.0, 3.0, 1.0),
    )

    result = build_method_a_probability(relationship, thresholds)

    assert not result.limited
    assert result.weighted_class_counts == {"UP": 4.0, "SIDEWAYS": 0.0, "DOWN": 1.0}
    assert result.probabilities_pct["UP"] == 80.0
    assert result.probabilities_pct["DOWN"] == 20.0
    assert result.probabilities_pct["SIDEWAYS"] == 0.0
    assert abs(result.effective_sample_size - 25 / 11) < 1e-9


def test_method_a_fails_closed_on_malformed_support() -> None:
    thresholds = learn_outcome_thresholds(
        [-6, -5, -4, -1, 0, 1, 4, 5, 6],
        min_observations=9,
    )

    mismatched = _relationship((-5.0, 6.0), weights=(1.0,))
    result = build_method_a_probability(mismatched, thresholds)
    assert result.limited
    assert any("mismatched lengths" in item for item in result.limitations)

    invalid = _relationship((-5.0, nan, 6.0))
    result = build_method_a_probability(invalid, thresholds)
    assert result.limited
    assert any("invalid return or weight" in item for item in result.limitations)


def test_method_a_rejects_non_method_a_relationship() -> None:
    thresholds = learn_outcome_thresholds(
        [-6, -5, -4, -1, 0, 1, 4, 5, 6],
        min_observations=9,
    )
    relationship = _relationship((-5.0, 0.0, 6.0), method="B")

    try:
        build_method_a_probability(relationship, thresholds)
    except ValueError as exc:
        assert "received method 'B'" in str(exc)
    else:
        raise AssertionError("Expected Method A probability builder to reject Method B input")


if __name__ == "__main__":
    test_method_a_uses_exact_support_distribution()
    test_method_a_weighting_is_explicit_and_auditable()
    test_method_a_fails_closed_on_malformed_support()
    test_method_a_rejects_non_method_a_relationship()
    print("PHASE 5.2 METHOD A PROBABILITY TEST: PASS")
