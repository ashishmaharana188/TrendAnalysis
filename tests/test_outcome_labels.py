from __future__ import annotations

from datetime import date, timedelta

from analysis.outcome_labels import (
    class_counts,
    classify_return,
    filter_completed_outcomes,
    learn_outcome_thresholds,
    learn_thresholds_from_observations,
)
from analysis.relationship import HistoricalRelationshipObservation


def _obs(
    *,
    target: str,
    as_of: date,
    outcome_end: date | None,
    ret: float | None,
) -> HistoricalRelationshipObservation:
    return HistoricalRelationshipObservation(
        as_of_date=as_of,
        target=target,
        scope="company",
        states={"company.market.price": "Rising / High"},
        stock_return_pct=ret,  # type: ignore[arg-type]
        benchmark_return_pct=0.0,
        relative_return_pct=ret,
        outcome_end_date=outcome_end,
    )


def main() -> None:
    base = date(2020, 1, 1)

    # ------------------------------------------------------------
    # 1. Basic empirical tertiles and boundary semantics.
    # ------------------------------------------------------------
    returns = list(range(-15, 16))
    thresholds = learn_outcome_thresholds(returns, min_observations=9)
    assert not thresholds.limited
    assert thresholds.lower_pct < thresholds.upper_pct
    assert classify_return(thresholds.lower_pct - 0.001, thresholds) == "DOWN"
    assert classify_return(thresholds.lower_pct, thresholds) == "SIDEWAYS"
    assert classify_return(thresholds.upper_pct, thresholds) == "SIDEWAYS"
    assert classify_return(thresholds.upper_pct + 0.001, thresholds) == "UP"

    counts = class_counts(returns, thresholds)
    assert sum(counts.values()) == len(returns)
    assert all(value > 0 for value in counts.values())

    # ------------------------------------------------------------
    # 2. Invalid numeric inputs never become a label.
    # ------------------------------------------------------------
    invalid_case = learn_outcome_thresholds(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, float("nan"), float("inf")],
        min_observations=9,
    )
    assert not invalid_case.limited
    assert invalid_case.sample_count == 9
    assert invalid_case.invalid_count == 2

    # ------------------------------------------------------------
    # 3. Too little usable history is explicitly LIMITED.
    # ------------------------------------------------------------
    limited = learn_outcome_thresholds([1.0, 2.0, 3.0], min_observations=9)
    assert limited.limited
    assert limited.sample_count == 3
    assert limited.limitation is not None

    # ------------------------------------------------------------
    # 4. Collapsed quantiles are LIMITED instead of fabricated.
    # ------------------------------------------------------------
    collapsed = learn_outcome_thresholds([1.0] * 20, min_observations=9)
    assert collapsed.limited
    assert collapsed.lower_pct == collapsed.upper_pct == 1.0

    # ------------------------------------------------------------
    # 5. Point-in-time filter deliberately breaks future/incomplete leakage.
    # ------------------------------------------------------------
    observations = [
        _obs(
            target="TEST",
            as_of=base + timedelta(days=1),
            outcome_end=base + timedelta(days=10),
            ret=1.0,
        ),
        _obs(
            target="TEST",
            as_of=base + timedelta(days=2),
            outcome_end=base + timedelta(days=20),
            ret=2.0,
        ),
        _obs(
            target="TEST",
            as_of=base + timedelta(days=3),
            outcome_end=None,
            ret=3.0,
        ),
        _obs(
            target="OTHER",
            as_of=base + timedelta(days=1),
            outcome_end=base + timedelta(days=10),
            ret=99.0,
        ),
        _obs(
            target="TEST",
            as_of=base + timedelta(days=40),
            outcome_end=base + timedelta(days=45),
            ret=50.0,
        ),
        _obs(
            target="TEST",
            as_of=base + timedelta(days=4),
            outcome_end=base + timedelta(days=9),
            ret=float("nan"),
        ),
        _obs(
            target="TEST",
            as_of=base + timedelta(days=5),
            outcome_end=base + timedelta(days=9),
            ret=None,
        ),
    ]
    cutoff = base + timedelta(days=20)
    eligible, audit = filter_completed_outcomes(observations, "TEST", cutoff)
    assert len(eligible) == 1
    assert eligible[0].stock_return_pct == 1.0
    assert audit.excluded_target == 1
    assert audit.excluded_future_state == 1
    assert audit.excluded_incomplete_outcome == 1
    assert audit.excluded_unknown_outcome_end == 1
    assert audit.excluded_non_finite_return == 1
    assert audit.excluded_missing_return == 1

    # Same filter + threshold learning must use the same eligible universe.
    learned, learned_audit = learn_thresholds_from_observations(
        observations,
        target="TEST",
        cutoff_date=cutoff,
        min_observations=9,
    )
    assert learned.limited
    assert learned.sample_count == 1
    assert learned_audit.eligible_count == 1

    # ------------------------------------------------------------
    # 6. Extreme future/incomplete outcomes cannot move learned thresholds.
    # ------------------------------------------------------------
    training_rows = [
        _obs(
            target="TEST",
            as_of=base + timedelta(days=index),
            outcome_end=base + timedelta(days=100 + index),
            ret=float(index - 5),
        )
        for index in range(9)
    ]
    safe_cutoff = base + timedelta(days=50)
    safe_thresholds, safe_audit = learn_thresholds_from_observations(
        training_rows,
        target="TEST",
        cutoff_date=safe_cutoff,
        min_observations=9,
    )
    assert safe_thresholds.limited
    assert safe_audit.eligible_count == 0

    completed_rows = [
        _obs(
            target="TEST",
            as_of=base + timedelta(days=index),
            outcome_end=base + timedelta(days=20),
            ret=float(index - 5),
        )
        for index in range(9)
    ]
    completed_cutoff = base + timedelta(days=30)
    thresholds_without_future, _ = learn_thresholds_from_observations(
        completed_rows,
        target="TEST",
        cutoff_date=completed_cutoff,
        min_observations=9,
    )
    poisoned_rows = completed_rows + [
        _obs(
            target="TEST",
            as_of=completed_cutoff - timedelta(days=1),
            outcome_end=completed_cutoff + timedelta(days=1),
            ret=10_000.0,
        )
    ]
    thresholds_with_future, poison_audit = learn_thresholds_from_observations(
        poisoned_rows,
        target="TEST",
        cutoff_date=completed_cutoff,
        min_observations=9,
    )
    assert thresholds_without_future.as_dict() == thresholds_with_future.as_dict()
    assert poison_audit.excluded_incomplete_outcome == 1

    print("PHASE 5.1 OUTCOME CLASSIFICATION TEST: PASS")
    print("Thresholds:", round(thresholds.lower_pct, 4), round(thresholds.upper_pct, 4))
    print("Class counts:", counts)
    print("Point-in-time eligible rows:", audit.eligible_count)
    print("Limited-case correctly flagged:", limited.limited)


if __name__ == "__main__":
    main()
