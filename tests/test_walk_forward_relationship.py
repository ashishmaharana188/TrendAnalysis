from __future__ import annotations

from datetime import date, timedelta

from analysis.relationship import HistoricalRelationshipObservation, RelationshipDiscoveryEngine
from analysis.walk_forward_relationship import validate_walk_forward_relationships


def main() -> None:
    observations = []
    base = date(2020, 1, 1)

    # Persistent structural relationship:
    # company financials = Rising / High AND company market price = Rising / High
    # tends to be followed by positive returns.
    for index in range(60):
        as_of = base + timedelta(days=index)
        bullish_state = index % 2 == 0
        revenue = "Rising / High" if bullish_state else "Falling / Low"
        price = "Rising / High" if bullish_state else "Falling / Low"
        ret = 2.0 if bullish_state else -1.5

        observations.append(
            HistoricalRelationshipObservation(
                as_of_date=as_of,
                target="TEST",
                scope="company",
                states={
                    "company.financials.TotalRevenue": revenue,
                    "company.market.price": price,
                },
                stock_return_pct=ret,
                benchmark_return_pct=0.5,
                relative_return_pct=ret - 0.5,
            )
        )

    engine = RelationshipDiscoveryEngine(
        max_order=2,
        min_observations=5,
    )

    result = validate_walk_forward_relationships(
        observations=observations,
        engine=engine,
    )

    assert result.leakage_violations == 0
    assert result.skipped_insufficient_history > 0
    assert result.method_a.directional_predictions > 0
    assert result.method_b.directional_predictions > 0
    assert result.method_a.directional_hit_rate_pct >= 80.0
    assert result.method_b.directional_hit_rate_pct >= 80.0
    assert result.combined.directional_hit_rate_pct >= 80.0

    # Explicitly verify that the first usable fold trained only on prior dates.
    first_fold = result.folds[0]
    assert first_fold.training_observations >= 5
    assert first_fold.prediction_date > base

    print("PHASE 4.8 WALK-FORWARD RELATIONSHIP VALIDATION TEST: PASS")
    print("Evaluated folds:", len(result.folds))
    print("Method A hit rate:", round(result.method_a.directional_hit_rate_pct, 2))
    print("Method B hit rate:", round(result.method_b.directional_hit_rate_pct, 2))
    print("Combined hit rate:", round(result.combined.directional_hit_rate_pct, 2))
    print("Leakage violations:", result.leakage_violations)


if __name__ == "__main__":
    main()
