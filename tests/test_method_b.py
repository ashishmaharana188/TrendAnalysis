from __future__ import annotations

from datetime import date, timedelta

from analysis.relationship import (
    HistoricalRelationshipObservation,
    RelationshipDiscoveryEngine,
)


def main() -> None:
    base = date(2023, 1, 1)
    history = []

    for index in range(60):
        as_of = base + timedelta(days=index)
        current_like = index < 24

        if current_like:
            market = "Rising / High"
            financial = "Rising / High"
            ret = 4.0
        else:
            market = "Falling / Low"
            financial = "Falling / Low"
            ret = -2.0 if index % 3 else 0.5

        history.append(
            HistoricalRelationshipObservation(
                as_of_date=as_of,
                target="RELIANCE",
                scope="company",
                states={
                    "company.market.price": market,
                    "company.financials.TotalRevenue": financial,
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
    }

    results = engine.method_b_conditioned_distribution(
        current_states=current,
        history=history,
    )

    assert results
    pair = next(item for item in results if len(item.variables) == 2)

    # Method B keeps the full distribution, so its effective sample size
    # must be materially larger than the minimum matching-condition sample.
    assert pair.sample_count >= 20
    assert pair.weighted_mean_return_pct is not None
    assert pair.weighted_positive_rate_pct is not None
    assert pair.effective_sample_size is not None
    assert pair.effective_sample_size >= pair.sample_count
    assert pair.weight_concentration is not None
    assert pair.weight_concentration < 0.5
    assert pair.mean_return_pct > 0
    assert pair.score > 0

    print("PHASE 4.4 METHOD B ADAPTIVE DISTRIBUTION TEST: PASS")
    print("Conditioned relationship samples:", pair.sample_count)
    print("Weighted mean return:", round(pair.weighted_mean_return_pct, 4))
    print("Weighted positive rate:", round(pair.weighted_positive_rate_pct, 4))
    print("Effective sample size:", round(pair.effective_sample_size, 4))
    print("Weight concentration:", round(pair.weight_concentration, 4))


if __name__ == "__main__":
    main()
