from __future__ import annotations

from datetime import date, timedelta

from analysis.adaptive_combination import discover_adaptive_higher_order
from analysis.relationship import (
    HistoricalRelationshipObservation,
    RelationshipDiscoveryEngine,
)


def main() -> None:
    base = date(2021, 1, 1)
    history = []

    for index in range(96):
        as_of = base + timedelta(days=index)

        if index < 32:
            states = {
                "company.financials.TotalRevenue": "Rising / High",
                "company.market.price": "Rising / High",
                "industry.market.group_index": "Rising / High",
                "sector.market.group_index": "Rising / High",
            }
            ret = 5.0
        elif index % 2 == 0:
            states = {
                "company.financials.TotalRevenue": "Rising / High",
                "company.market.price": "Rising / High",
                "industry.market.group_index": "Rising / High",
                "sector.market.group_index": "Falling / Low",
            }
            ret = 1.0
        else:
            states = {
                "company.financials.TotalRevenue": "Falling / Low",
                "company.market.price": "Falling / Low",
                "industry.market.group_index": "Falling / Low",
                "sector.market.group_index": "Falling / Low",
            }
            ret = -2.0

        history.append(
            HistoricalRelationshipObservation(
                as_of_date=as_of,
                target="RELIANCE",
                scope="company",
                states=states,
                stock_return_pct=ret,
                benchmark_return_pct=1.0,
                relative_return_pct=ret - 1.0,
            )
        )

    current = {
        "company.financials.TotalRevenue": "Rising / High",
        "company.market.price": "Rising / High",
        "industry.market.group_index": "Rising / High",
        "sector.market.group_index": "Rising / High",
    }

    engine = RelationshipDiscoveryEngine(
        max_order=2,
        min_observations=5,
    )

    result = discover_adaptive_higher_order(
        current_states=current,
        history=history,
        engine=engine,
    )

    assert result.rankings
    assert result.steps
    assert result.highest_order >= 3
    assert all(step.retained_count >= 0 for step in result.steps)

    # With four structural families there can be no 5-variable candidate,
    # so discovery must terminate from structure or evidence, not an invented
    # hard maximum.
    assert result.highest_order <= 4
    assert result.steps[-1].stopped
    assert result.stopped_reason

    print("PHASE 4.7 ADAPTIVE HIGHER-ORDER STOPPING TEST: PASS")
    print("Highest order:", result.highest_order)
    print("Expansion steps:", len(result.steps))
    for step in result.steps:
        print(
            f"Order {step.parent_order}->{step.child_order}: "
            f"candidates={step.candidate_count}, "
            f"evaluated={step.evaluated_count}, "
            f"retained={step.retained_count}, "
            f"best_gain={step.best_incremental_gain:.4f}, "
            f"stopped={step.stopped}"
        )
    print("Stopped reason:", result.stopped_reason)


if __name__ == "__main__":
    main()
