from analysis.real_olap_validation import RealOLAPValidationConfig
from analysis.real_prediction_validation import (
    print_real_olap_prediction_report,
    validate_real_olap_predictions,
)


def main() -> None:
    config = RealOLAPValidationConfig(
        ticker="RELIANCE",
        benchmark="Nifty_50",
        analysis_timeframe="6M",
        holding_period_months=1.0,
        entry_mode="next_trading_day",
        max_folds=250,
        step_trading_days=5,
        min_training_observations=12,
        hardened_validation=True,
        performance_cache=True,
        progress_logging=True,
        progress_every=5,
        adaptive_relationship_search=False,
        relationship_progress_every_candidates=1000,
        relationship_candidate_batch_size=512,
    )

    result = validate_real_olap_predictions(config)
    print_real_olap_prediction_report(result)

    assert result.candidate_predictions > 0
    assert result.evaluated_predictions > 0
    assert result.latest_validated_prediction_date is not None
    assert result.latest_market_date is not None
    assert result.skipped_provenance_failed == 0

    print("PHASE 5.8 REAL OLAP PREDICTION VALIDATION: PASS")
    print("IMPORTANT: This is diagnostic Phase 5 validation; Phase 6 calibration/OOS performance is still required.")


if __name__ == "__main__":
    main()


def test_hardened_selection_uses_support_retaining_engine(monkeypatch):
    """
    Regression guard for the Phase 5.8 zero-selection bug.

    Discovery is intentionally memory-light and may omit raw support. Inner
    selection, however, feeds search-wide permutation/FDR and therefore must
    receive an engine with support retention enabled.
    """
    from datetime import date, timedelta

    from analysis.relationship import (
        HistoricalRelationshipObservation,
        RelationshipResult,
        RelationshipDiscoveryEngine,
    )
    from analysis.walk_forward_relationship import (
        _holdout_select,
        TestingFamily,
    )

    base = date(2025, 1, 1)
    history = [
        HistoricalRelationshipObservation(
            as_of_date=base + timedelta(days=i),
            target="TEST",
            scope="company",
            states={"company.market.price": "Rising / High"},
            stock_return_pct=1.0 if i % 2 == 0 else -0.5,
            benchmark_return_pct=0.0,
            relative_return_pct=1.0 if i % 2 == 0 else -0.5,
            outcome_end_date=base + timedelta(days=i),
        )
        for i in range(12)
    ]

    candidate = RelationshipResult(
        method="A",
        variables=("company.market.price",),
        condition=("company.market.price~=Rising / High",),
        sample_count=5,
        mean_return_pct=0.4,
        median_return_pct=1.0,
        baseline_mean_return_pct=0.25,
        lift_pct=0.15,
        positive_rate_pct=60.0,
        effect_strength=0.5,
        reliability=0.8,
        score=0.4,
        stable=True,
        parameter_provenance=(
            ("state_relevance_frequency", base, base + timedelta(days=5)),
            ("method_a_adaptive_neighbor_count", base, base + timedelta(days=5)),
        ),
    )

    class SupportAwareEngine(RelationshipDiscoveryEngine):
        def method_a_similar_states(
            self,
            current_states,
            history,
            candidate_sets=None,
        ):
            return [
                RelationshipResult(
                    method="A",
                    variables=("company.market.price",),
                    condition=("company.market.price~=Rising / High",),
                    sample_count=5,
                    mean_return_pct=0.4,
                    median_return_pct=1.0,
                    baseline_mean_return_pct=0.25,
                    lift_pct=0.15,
                    positive_rate_pct=60.0,
                    effect_strength=0.5,
                    reliability=0.8,
                    score=0.4,
                    stable=True,
                    supporting_observations=tuple(
                        (row.as_of_date, float(row.stock_return_pct))
                        for row in history[:5]
                    ),
                    supporting_weights=(1.0,) * 5,
                    parameter_provenance=(
                        ("state_relevance_frequency", base, base + timedelta(days=5)),
                        ("method_a_adaptive_neighbor_count", base, base + timedelta(days=5)),
                    ),
                )
            ]

    engine = SupportAwareEngine(
        max_order=1,
        min_observations=5,
        retain_supporting_data=True,
    )

    family = TestingFamily(
        target="TEST",
        scope="company",
        method="A",
        prediction_date=base + timedelta(days=20),
    )

    selected, evaluations, controlled = _holdout_select(
        "A",
        [candidate],
        {"company.market.price": "Rising / High"},
        history,
        engine,
        family,
        alpha=0.10,
    )

    assert evaluations == 1
    assert controlled is True
    assert selected is None or selected.variables == candidate.variables


def test_holdout_selection_fails_closed_when_support_retention_is_disabled():
    from datetime import date
    from analysis.relationship import RelationshipDiscoveryEngine
    from analysis.walk_forward_relationship import _holdout_select, TestingFamily

    engine = RelationshipDiscoveryEngine(
        max_order=1,
        min_observations=1,
        retain_supporting_data=False,
    )

    family = TestingFamily(
        target="TEST",
        scope="company",
        method="A",
        prediction_date=date(2025, 1, 10),
    )

    candidate = object()

    try:
        _holdout_select(
            "A",
            [candidate],
            {},
            [object()],
            engine,
            family,
            alpha=0.10,
        )
    except RuntimeError as exc:
        assert "retain_supporting_data=True" in str(exc)
    else:
        raise AssertionError("Expected nested selection support-retention guard to fire.")


def test_search_adjusted_pvalues_are_not_bh_recorrected(monkeypatch):
    """
    Regression guard for double-correction of search-adjusted p-values.

    The search-wide max-statistic permutation output is already adjusted for
    the full candidate family. With one genuinely extreme candidate and a
    large family, direct alpha-thresholding should accept it. Re-running BH
    over those already-adjusted values would incorrectly reject it.
    """
    from datetime import date

    import analysis.walk_forward_relationship as walk_forward
    from analysis.hardening_4_10 import SearchAdjustedPermutationResult, TestingFamily
    from analysis.relationship import (
        HistoricalRelationshipObservation,
        RelationshipDiscoveryEngine,
        RelationshipResult,
    )

    candidate_count = 1000
    candidates = []
    for index in range(candidate_count):
        feature = f"feature_{index}"
        candidates.append(
            RelationshipResult(
                method="A",
                variables=(feature,),
                condition=(f"{feature}=state",),
                sample_count=5,
                mean_return_pct=1.0,
                median_return_pct=1.0,
                baseline_mean_return_pct=0.0,
                lift_pct=1.0,
                positive_rate_pct=100.0,
                effect_strength=1.0,
                reliability=1.0,
                score=1.0,
                stable=True,
                supporting_observations=((date(2025, 1, 1), 1.0),),
                supporting_weights=(1.0,),
            )
        )

    class StubEngine(RelationshipDiscoveryEngine):
        def method_a_similar_states(self, current_states, history, candidate_sets=None):
            return list(candidates)

    engine = StubEngine(
        max_order=1,
        min_observations=1,
        retain_supporting_data=True,
    )

    history = [
        HistoricalRelationshipObservation(
            as_of_date=date(2025, 1, 1),
            target="TEST",
            scope="company",
            states={f"feature_{i}": "state" for i in range(candidate_count)},
            stock_return_pct=1.0,
            benchmark_return_pct=0.0,
            relative_return_pct=1.0,
            outcome_end_date=date(2025, 1, 2),
        )
    ]

    fake_result = SearchAdjustedPermutationResult(
        family_id="TEST|company|A|2025-02-01",
        candidate_count=candidate_count,
        permutation_count=199,
        raw_p_values=(float("nan"),) * candidate_count,
        max_statistic_p_values=(0.005,) + (1.0,) * (candidate_count - 1),
    )

    monkeypatch.setattr(
        walk_forward,
        "search_adjusted_permutation_p_values",
        lambda *args, **kwargs: fake_result,
    )

    family = TestingFamily(
        target="TEST",
        scope="company",
        method="A",
        prediction_date=date(2025, 2, 1),
    )

    selected, evaluations, controlled = walk_forward._holdout_select(
        "A",
        candidates,
        {f"feature_{i}": "state" for i in range(candidate_count)},
        history,
        engine,
        family,
        alpha=0.10,
    )

    assert evaluations == candidate_count
    assert controlled is True
    assert selected is not None
    assert selected.variables == ("feature_0",)
