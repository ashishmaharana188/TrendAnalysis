from __future__ import annotations

from datetime import date, timedelta

from analysis.relationship import (
    HistoricalRelationshipObservation,
    RelationshipDiscoveryEngine,
)


def main() -> None:
    base = date(2024, 1, 1)
    history = []

    for index in range(36):
        as_of = base + timedelta(days=index)

        # Three state dimensions form two structural families. The first
        # 24 rows contain intentionally similar states with positive outcomes.
        if index < 24:
            market = "Rising / High"
            financial = "Rising / High"
            macro = "Stable / Mid"
            ret = 3.0
        elif index % 2 == 0:
            market = "Falling / Low"
            financial = "Falling / Low"
            macro = "Rising / High"
            ret = -2.0
        else:
            market = "Rising / High"
            financial = "Falling / Low"
            macro = "Falling / Low"
            ret = 0.5

        history.append(
            HistoricalRelationshipObservation(
                as_of_date=as_of,
                target="RELIANCE",
                scope="company",
                states={
                    "company.market.price": market,
                    "company.financials.TotalRevenue": financial,
                    "macro.Brent_Crude": macro,
                },
                stock_return_pct=ret,
                benchmark_return_pct=1.0,
                relative_return_pct=ret - 1.0,
            )
        )

    engine = RelationshipDiscoveryEngine(
        max_order=2,
        min_observations=5,
    )

    current = {
        "company.market.price": "Rising / High",
        "company.financials.TotalRevenue": "Rising / High",
        "macro.Brent_Crude": "Stable / Mid",
    }

    results = engine.method_a_similar_states(current, history)

    assert len(results) == 1
    result = results[0]

    assert result.method == "A"
    assert result.sample_count >= 5
    assert result.mean_return_pct > 0
    assert result.reliability > 0
    assert result.score > 0
    assert result.stable is True
    assert all("~=" in condition for condition in result.condition)

    # The adaptive neighbourhood must be smaller than the complete history
    # for this sample, proving Method A is actually selecting comparable cases.
    assert result.sample_count < len(history)

    print("PHASE 4.3 METHOD A ADAPTIVE SIMILARITY TEST: PASS")
    print("Selected comparable states:", result.sample_count)
    print("Mean matched return:", round(result.mean_return_pct, 4))
    print("Similarity-adjusted score:", round(result.score, 4))
    print("Stable:", result.stable)


if __name__ == "__main__":
    main()
