from __future__ import annotations

from datetime import date, timedelta

from analysis.outcome_labels import classify_return, learn_outcome_thresholds
from analysis.prediction import MethodPrediction, PredictionEngine
from analysis.relationship import HistoricalRelationshipObservation, RelationshipDiscoveryEngine


def test_phase5_excludes_future_labels() -> None:
    base = date(2026, 1, 1)
    rows = [
        HistoricalRelationshipObservation(
            as_of_date=base,
            target="TEST",
            scope="company",
            states={"company.market.price": "Rising"},
            stock_return_pct=5.0,
            benchmark_return_pct=0.0,
            relative_return_pct=5.0,
            outcome_end_date=base + timedelta(days=5),
        ),
        HistoricalRelationshipObservation(
            as_of_date=base + timedelta(days=1),
            target="TEST",
            scope="company",
            states={"company.market.price": "Rising"},
            stock_return_pct=50.0,
            benchmark_return_pct=0.0,
            relative_return_pct=50.0,
            outcome_end_date=base + timedelta(days=20),
        ),
    ]
    engine = PredictionEngine(
        relationship_engine=RelationshipDiscoveryEngine(max_order=1, min_observations=1),
        min_threshold_observations=3,
    )
    # The future-ending label must not enter training. The result therefore
    # remains limited because only one completed outcome exists.
    result = engine.predict(
        target="TEST",
        current_states={"company.market.price": "Rising"},
        observations=rows,
        prediction_date=base + timedelta(days=10),
    )
    assert result.training_observations == 1
    assert result.limited is True



def main() -> None:
    base = date(2020, 1, 1)
    history = []

    # A persistent signal with three naturally separated return regimes.
    # The test checks the machinery, not market performance.
    regimes = [
        ("Rising / High", "Rising / High", 6.0),
        ("Stable / Mid", "Stable / Mid", 0.5),
        ("Falling / Low", "Falling / Low", -5.0),
    ]

    for index in range(90):
        market, financial, ret = regimes[index % 3]
        history.append(
            HistoricalRelationshipObservation(
                as_of_date=base + timedelta(days=index),
                target="TEST",
                scope="company",
                states={
                    "company.market.price": market,
                    "company.financials.TotalRevenue": financial,
                },
                stock_return_pct=ret,
                benchmark_return_pct=0.0,
                relative_return_pct=ret,
                outcome_end_date=base + timedelta(days=index + 1),
            )
        )

    thresholds = learn_outcome_thresholds([row.stock_return_pct for row in history])
    assert not thresholds.limited
    assert thresholds.lower_pct < thresholds.upper_pct
    assert classify_return(6.0, thresholds) == "UP"
    assert classify_return(-5.0, thresholds) == "DOWN"

    engine = PredictionEngine(
        relationship_engine=RelationshipDiscoveryEngine(
            max_order=2,
            min_observations=5,
        )
    )

    result = engine.predict(
        target="TEST",
        current_states={
            "company.market.price": "Rising / High",
            "company.financials.TotalRevenue": "Rising / High",
        },
        observations=history,
        prediction_date=base + timedelta(days=80),
        analysis_timeframe="6M",
        holding_period_months=1.0,
        benchmark="Nifty_50",
        entry_mode="next_trading_day",
    )

    # 79 rows have outcomes fully completed before the cutoff; the 80th row
    # ends exactly on the cutoff and is correctly excluded by the strict temporal gate.
    assert result.training_observations == 79
    assert any("not fully realized before the prediction cutoff" in item for item in result.limitations)
    assert result.analysis_timeframe == "6M"
    assert result.holding_period_months == 1.0
    assert result.benchmark == "Nifty_50"
    assert result.entry_mode == "next_trading_day"
    assert sum(result.probabilities_pct.values()) == pytest_approx(100.0)
    assert result.method_b is not None
    assert result.method_b.limited is False
    assert sum(result.method_b.probabilities_pct.values()) == pytest_approx(100.0)
    assert result.trend in {"UP", "SIDEWAYS", "DOWN", "NO_CLEAR_TREND"}
    assert result.conviction in {"STRONG", "MODERATE", "LOW", "NONE"}
    # Without an explicit OOS validation gate, STRONG conviction is forbidden.
    if result.trend in {"UP", "DOWN"} and result.method_a is not None and result.method_b is not None:
        assert result.conviction != "STRONG"


    conflict_a = MethodPrediction(
        method="A",
        trend="UP",
        probabilities_pct={"UP": 70.0, "SIDEWAYS": 20.0, "DOWN": 10.0},
        baseline_probabilities_pct={"UP": 33.0, "SIDEWAYS": 34.0, "DOWN": 33.0},
        expected_return_pct=5.0,
        baseline_return_pct=0.5,
        return_lift_pct=4.5,
        evidence_score=0.5,
        reliability=0.8,
        sample_count=20,
        effective_sample_size=20.0,
        stable=True,
        variables=("x",),
        condition=("x=Rising",),
    )
    conflict_b = MethodPrediction(
        method="B",
        trend="DOWN",
        probabilities_pct={"UP": 10.0, "SIDEWAYS": 20.0, "DOWN": 70.0},
        baseline_probabilities_pct={"UP": 33.0, "SIDEWAYS": 34.0, "DOWN": 33.0},
        expected_return_pct=-4.0,
        baseline_return_pct=0.5,
        return_lift_pct=-4.5,
        evidence_score=0.6,
        reliability=0.9,
        sample_count=30,
        effective_sample_size=30.0,
        stable=True,
        variables=("y",),
        condition=("y=Falling",),
    )
    conflict_trend, conflict_conviction, _probs, _expected, _agreement = engine._combine(
        conflict_a, conflict_b, {"UP": 33.0, "SIDEWAYS": 34.0, "DOWN": 33.0}
    )
    assert conflict_trend == "NO_CLEAR_TREND"
    assert conflict_conviction == "NONE"

    test_phase5_excludes_future_labels()

    print("PHASE 5 PREDICTION ENGINE TEST: PASS")
    print("Trend:", result.trend)
    print("Conviction:", result.conviction)
    print("Probabilities:", {k: round(v, 2) for k, v in result.probabilities_pct.items()})
    print("Method A:", result.method_a.trend if result.method_a else None)
    print("Method B:", result.method_b.trend if result.method_b else None)


def pytest_approx(value: float):
    # Tiny local approximation helper to keep this test dependency-free.
    class Approx:
        def __eq__(self, other):
            return abs(float(other) - value) < 1e-9
    return Approx()


if __name__ == "__main__":
    main()
