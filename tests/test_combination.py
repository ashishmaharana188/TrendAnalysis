from __future__ import annotations

from datetime import date, timedelta

from analysis.combination import discover_higher_order_combinations
from analysis.relationship import (
    HistoricalRelationshipObservation,
    RelationshipDiscoveryEngine,
)


def main() -> None:
    base = date(2022, 1, 1)
    history = []

    for index in range(72):
        as_of = base + timedelta(days=index)

        # First 24 rows deliberately represent a recurring three-way state
        # with positive subsequent outcomes. The remaining rows provide a
        # contrasting background distribution.
        if index < 24:
            financial = "Rising / High"
            market = "Rising / High"
            industry = "Rising / High"
            ret = 4.0
        elif index % 3 == 0:
            financial = "Rising / High"
            market = "Falling / Low"
            industry = "Falling / Low"
            ret = -2.0
        else:
            financial = "Falling / Low"
            market = "Falling / Low"
            industry = "Falling / Low"
            ret = -1.0

        history.append(
            HistoricalRelationshipObservation(
                as_of_date=as_of,
                target="RELIANCE",
                scope="company",
                states={
                    "company.financials.TotalRevenue": financial,
                    "company.market.price": market,
                    "industry.market.group_index": industry,
                },
                stock_return_pct=ret,
                benchmark_return_pct=1.0,
                relative_return_pct=ret - 1.0,
            )
        )

    current = {
        "company.financials.TotalRevenue": "Rising / High",
        "company.market.price": "Rising / High",
        "industry.market.group_index": "Rising / High",
    }

    engine = RelationshipDiscoveryEngine(
        max_order=2,
        min_observations=5,
    )

    result = discover_higher_order_combinations(
        current_states=current,
        history=history,
        engine=engine,
    )

    assert result.parent_pair_count >= 1
    assert result.selected_pair_count >= 1
    assert result.candidate_count >= 1
    assert result.evaluated_count >= 1
    assert result.rankings

    top = result.rankings[0]
    assert len(top.variables) == 3
    assert top.direction == "POSITIVE"
    assert top.best_score > 0

    assert any(
        set(item) == {
            "company.financials.TotalRevenue",
            "company.market.price",
            "industry.market.group_index",
        }
        for item in result.candidates
    )

    print("PHASE 4.6 ADAPTIVE COMBINATION DISCOVERY TEST: PASS")
    print("Parent pair relationships:", result.parent_pair_count)
    print("Selected seed pairs:", result.selected_pair_count)
    print("3-variable candidates:", result.candidate_count)
    print("Evaluated method results:", result.evaluated_count)
    print("Top direction:", top.direction)
    print("Top score:", round(top.best_score, 4))


if __name__ == "__main__":
    main()
