from __future__ import annotations

from datetime import date
from math import nan

from analysis.method_b_prediction import build_method_b_probability
from analysis.outcome_labels import learn_outcome_thresholds
from analysis.relationship import RelationshipResult


def _relationship(
    returns: tuple[float, ...],
    weights: tuple[float, ...] | None = None,
    sample_count: int | None = None,
    method: str = "B",
    exact_condition_count: int = 2,
) -> RelationshipResult:
    support_weights = weights if weights is not None else tuple(1.0 for _ in returns)
    mean = sum(returns) / len(returns)
    return RelationshipResult(
        method=method,
        variables=("company.market.price", "macro.test"),
        condition=("company.market.price~=Rising / High", "macro.test~=High"),
        sample_count=len(returns) if sample_count is None else sample_count,
        mean_return_pct=mean,
        median_return_pct=sorted(returns)[len(returns) // 2],
        baseline_mean_return_pct=0.5,
        lift_pct=mean - 0.5,
        positive_rate_pct=sum(value > 0 for value in returns) / len(returns) * 100,
        effect_strength=0.5,
        reliability=0.7,
        score=0.35,
        stable=True,
        weighted_mean_return_pct=mean,
        weighted_positive_rate_pct=sum(value > 0 for value in returns) / len(returns) * 100,
        effective_sample_size=1.0,
        weight_concentration=0.5,
        supporting_observations=tuple(
            (date(2020, 1, index + 1), value) for index, value in enumerate(returns)
        ),
        supporting_weights=support_weights,
        stability_score=0.8,
        exact_condition_count=exact_condition_count,
    )


def test_method_b_uses_full_weighted_support() -> None:
    thresholds = learn_outcome_thresholds(
        [-6, -5, -4, -1, 0, 1, 4, 5, 6],
        min_observations=9,
    )
    relationship = _relationship(
        (-5.0, 6.0, 5.0),
        weights=(1.0, 3.0, 1.0),
        exact_condition_count=1,
    )

    result = build_method_b_probability(relationship, thresholds)

    assert not result.limited
    assert result.sample_count == 3
    assert result.exact_condition_count == 1
    assert result.weighted_class_counts == {"UP": 4.0, "SIDEWAYS": 0.0, "DOWN": 1.0}
    assert result.probabilities_pct["UP"] == 80.0
    assert result.probabilities_pct["DOWN"] == 20.0
    assert result.probabilities_pct["SIDEWAYS"] == 0.0
    assert abs(result.effective_sample_size - 25 / 11) < 1e-9
    assert result.probability_basis.startswith("method_b_")


def test_exact_match_count_does_not_replace_weighted_population() -> None:
    thresholds = learn_outcome_thresholds(
        [-6, -5, -4, -1, 0, 1, 4, 5, 6],
        min_observations=9,
    )
    relationship = _relationship(
        (-5.0, 0.0, 6.0, 5.0),
        weights=(1.0, 1.0, 1.0, 1.0),
        exact_condition_count=1,
    )

    result = build_method_b_probability(relationship, thresholds)

    assert result.sample_count == 4
    assert result.exact_condition_count == 1
    assert result.class_counts == {"UP": 2, "SIDEWAYS": 1, "DOWN": 1}
    assert all(abs(result.probabilities_pct[label] - expected) < 1e-9 for label, expected in {
        "UP": 50.0,
        "SIDEWAYS": 25.0,
        "DOWN": 25.0,
    }.items())


def test_method_b_fails_closed_on_malformed_support() -> None:
    thresholds = learn_outcome_thresholds(
        [-6, -5, -4, -1, 0, 1, 4, 5, 6],
        min_observations=9,
    )
    mismatched = _relationship((-5.0, 6.0), weights=(1.0,))
    result = build_method_b_probability(mismatched, thresholds)
    assert result.limited
    assert any("mismatched lengths" in item for item in result.limitations)

    invalid = _relationship((-5.0, nan, 6.0))
    result = build_method_b_probability(invalid, thresholds)
    assert result.limited
    assert any("invalid return or weight" in item for item in result.limitations)


def test_method_b_rejects_non_method_b_relationship() -> None:
    thresholds = learn_outcome_thresholds(
        [-6, -5, -4, -1, 0, 1, 4, 5, 6],
        min_observations=9,
    )
    relationship = _relationship((-5.0, 0.0, 6.0), method="A")

    try:
        build_method_b_probability(relationship, thresholds)
    except ValueError as exc:
        assert "received method 'A'" in str(exc)
    else:
        raise AssertionError("Expected Method B probability builder to reject Method A input")


def test_method_b_requires_distinct_thresholds() -> None:
    thresholds = learn_outcome_thresholds([1.0] * 12, min_observations=9)
    relationship = _relationship((-1.0, 0.0, 1.0))
    result = build_method_b_probability(relationship, thresholds)
    assert result.limited
    assert "boundaries" in result.limitations[0]


if __name__ == "__main__":
    for fn in (
        test_method_b_uses_full_weighted_support,
        test_exact_match_count_does_not_replace_weighted_population,
        test_method_b_fails_closed_on_malformed_support,
        test_method_b_rejects_non_method_b_relationship,
        test_method_b_requires_distinct_thresholds,
    ):
        fn()
    print("PHASE 5.3 METHOD B PROBABILITY TEST: PASS")
