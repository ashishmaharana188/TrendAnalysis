from __future__ import annotations

from datetime import date, timedelta

from analysis.relationship import (
    HistoricalRelationshipObservation,
    RelationshipDiscoveryEngine,
    flatten_phase3_states,
)


def main() -> None:
    observations = []
    base = date(2024, 1, 1)

    for index in range(30):
        as_of = base + timedelta(days=index)

        if index % 3 == 0:
            revenue_state = "Rising / High"
            market_state = "Rising / High"
            ret = 4.0
        elif index % 2 == 0:
            revenue_state = "Rising / High"
            market_state = "Falling / Low"
            ret = 1.5
        else:
            revenue_state = "Falling / Low"
            market_state = "Falling / Low"
            ret = -2.0

        observations.append(
            HistoricalRelationshipObservation(
                as_of_date=as_of,
                target="RELIANCE",
                scope="company",
                states={
                    "company.financials.TotalRevenue": revenue_state,
                    "company.market.price": market_state,
                    "industry.market.group_index": (
                        "Rising / High"
                        if index % 4 < 2
                        else "Falling / Low"
                    ),
                    "macro.Brent_Crude": (
                        "Rising / High"
                        if index % 5 == 0
                        else "Stable / Mid"
                    ),
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
        "macro.Brent_Crude": "Stable / Mid",
    }

    engine = RelationshipDiscoveryEngine(
        max_order=2,
        min_observations=5,
    )

    result = engine.discover(
        current_states=current,
        observations=observations,
        cutoff_date=date(2024, 1, 31),
    )

    assert result["method_a"]
    assert result["method_b"]

    assert all(
        observation.as_of_date < date(2024, 1, 31)
        for observation in engine.prepare_history(
            observations,
            date(2024, 1, 31),
        )
    )

    pair_results = [
        item
        for item in result["method_b"]
        if len(item.variables) == 2
    ]

    assert pair_results

    flat = flatten_phase3_states(
        {
            "company": {
                "market": {
                    "price": {
                        "state": "Rising / High"
                    }
                }
            }
        }
    )

    assert flat == {
        "company.market.price": "Rising / High"
    }

    print("PHASE 4 RELATIONSHIP DISCOVERY TEST: PASS")
    print("Method A relationships:", len(result["method_a"]))
    print("Method B relationships:", len(result["method_b"]))


if __name__ == "__main__":
    main()
