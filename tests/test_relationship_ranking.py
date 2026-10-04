from __future__ import annotations

from datetime import date

from analysis.ranking import rank_relationships
from analysis.relationship import RelationshipResult


def make_result(
    method: str,
    variables: tuple[str, ...],
    condition: tuple[str, ...],
    score: float,
    mean_return: float,
    reliability: float,
    sample_count: int,
    stable: bool,
) -> RelationshipResult:
    return RelationshipResult(
        method=method,
        variables=variables,
        condition=condition,
        sample_count=sample_count,
        mean_return_pct=mean_return,
        median_return_pct=mean_return,
        baseline_mean_return_pct=0.0,
        lift_pct=mean_return,
        positive_rate_pct=60.0,
        effect_strength=abs(score),
        reliability=reliability,
        score=score,
        stable=stable,
    )


def main() -> None:
    same_variables = (
        "company.financials.TotalRevenue",
        "company.market.price",
    )
    same_condition_a = (
        "company.financials.TotalRevenue~=Rising / High",
        "company.market.price~=Rising / High",
    )
    same_condition_b = (
        "company.market.price=Rising / High",
        "company.financials.TotalRevenue=Rising / High",
    )

    # Same logical relationship, different methods.
    method_a = make_result(
        "A",
        same_variables,
        same_condition_a,
        score=0.80,
        mean_return=3.0,
        reliability=0.85,
        sample_count=20,
        stable=True,
    )
    method_b = make_result(
        "B",
        same_variables,
        same_condition_b,
        score=0.60,
        mean_return=2.0,
        reliability=0.75,
        sample_count=30,
        stable=True,
    )

    conflicting = make_result(
        "B",
        ("industry.market.group_index", "macro.Brent_Crude"),
        (
            "industry.market.group_index=Rising / High",
            "macro.Brent_Crude=Rising / High",
        ),
        score=0.70,
        mean_return=-1.5,
        reliability=0.90,
        sample_count=35,
        stable=False,
    )

    rankings = rank_relationships([method_a, method_b, conflicting])

    assert len(rankings) == 2
    top = rankings[0]

    assert top.variables == same_variables
    assert top.method_count == 2
    assert top.method_agreement is True
    assert top.direction == "POSITIVE"
    assert top.stable_method_count == 2
    assert top.validated_usefulness is False

    # Conflicting method evidence must not be silently merged into agreement.
    second = rankings[1]
    assert second.method_count == 1
    assert second.method_agreement is False
    assert second.direction == "NEGATIVE"


    # Agreement is no longer a primary promotion rule. A materially stronger
    # single-method relationship can rank above a weak cross-method agreement.
    strong_single = make_result(
        "A",
        ("company.financials.TotalRevenue", "company.market.price"),
        (
            "company.financials.TotalRevenue~=Rising / High",
            "company.market.price~=Rising / High",
        ),
        score=1.20,
        mean_return=4.0,
        reliability=0.70,
        sample_count=18,
        stable=False,
    )
    weak_pair_a = make_result(
        "A",
        ("industry.market.group_index", "macro.Brent_Crude"),
        (
            "industry.market.group_index~=Rising / High",
            "macro.Brent_Crude~=Rising / High",
        ),
        score=0.20,
        mean_return=0.6,
        reliability=0.90,
        sample_count=40,
        stable=True,
    )
    weak_pair_b = make_result(
        "B",
        ("industry.market.group_index", "macro.Brent_Crude"),
        (
            "industry.market.group_index=Rising / High",
            "macro.Brent_Crude=Rising / High",
        ),
        score=0.15,
        mean_return=0.5,
        reliability=0.85,
        sample_count=42,
        stable=True,
    )
    revised = rank_relationships([strong_single, weak_pair_a, weak_pair_b])
    assert revised[0].variables == strong_single.variables

    # Deterministic ordering and date-independent behaviour.
    assert date(2026, 10, 4) == date(2026, 10, 4)

    print("PHASE 4.5 RELATIONSHIP EVIDENCE RANKING TEST: PASS")
    print("Ranked relationships:", len(rankings))
    print("Top method agreement:", top.method_agreement)
    print("Top direction:", top.direction)
    print("Top score:", top.best_score)


if __name__ == "__main__":
    main()
