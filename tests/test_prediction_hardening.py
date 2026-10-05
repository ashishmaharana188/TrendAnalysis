from __future__ import annotations

from datetime import date, timedelta

from analysis.outcome_labels import OutcomeThresholds
from analysis.prediction_hardening import (
    PredictionProvenanceAudit,
    ValidationEvidenceAudit,
    audit_prediction_provenance,
)
from analysis.ranking import RelationshipRanking
from analysis.relationship import HistoricalRelationshipObservation, RelationshipResult


def _obs(as_of: date, outcome_end: date, ret: float = 1.0) -> HistoricalRelationshipObservation:
    return HistoricalRelationshipObservation(
        as_of_date=as_of,
        target="TEST",
        scope="company",
        states={"company.market.price": "Rising"},
        stock_return_pct=ret,
        benchmark_return_pct=0.0,
        relative_return_pct=ret,
        outcome_end_date=outcome_end,
    )


def _ranking(
    *,
    parameter_provenance=(
        ("state_relevance_frequency", date(2020, 1, 1), date(2026, 1, 10)),
    ),
    supporting_observations=((date(2026, 1, 1), 1.0),),
) -> RelationshipRanking:
    result = RelationshipResult(
        method="A",
        variables=("company.market.price",),
        condition=("company.market.price=Rising",),
        sample_count=len(supporting_observations),
        mean_return_pct=1.0,
        median_return_pct=1.0,
        baseline_mean_return_pct=0.0,
        lift_pct=1.0,
        positive_rate_pct=100.0,
        effect_strength=1.0,
        reliability=0.8,
        score=0.8,
        stable=True,
        supporting_observations=supporting_observations,
        supporting_weights=tuple(1.0 for _ in supporting_observations),
        parameter_provenance=parameter_provenance,
    )
    return RelationshipRanking(
        variables=result.variables,
        condition=result.condition,
        method_results=(result,),
        method_count=1,
        method_agreement=False,
        direction="POSITIVE",
        best_score=result.score,
        reliability=result.reliability,
        sample_count=result.sample_count,
        stable_method_count=1,
        rank_key=(abs(result.score),),
        candidate_count=1,
    )


def _thresholds() -> OutcomeThresholds:
    return OutcomeThresholds(lower_pct=0.0, upper_pct=2.0, sample_count=3)


def test_clean_prediction_provenance_passes() -> None:
    cutoff = date(2026, 2, 1)
    history = [
        _obs(date(2026, 1, 1), date(2026, 1, 5), -1.0),
        _obs(date(2026, 1, 10), date(2026, 1, 15), 1.0),
        _obs(date(2026, 1, 20), date(2026, 1, 25), 3.0),
    ]
    thresholds = _thresholds()
    audit = audit_prediction_provenance(
        observations=history,
        target="TEST",
        cutoff_date=cutoff,
        training_history=history,
        thresholds=thresholds,
        ranked_a=[_ranking()],
        ranked_b=[],
    )
    assert audit.clean
    assert audit.temporal_violations == 0
    assert audit.relationship_parameter_missing == 0
    assert audit.relationship_parameter_unknown == 0


def test_future_relationship_parameter_fails_closed() -> None:
    cutoff = date(2026, 2, 1)
    history = [_obs(date(2026, 1, 1), date(2026, 1, 10))] * 3
    future = _ranking(
        parameter_provenance=(
            ("future_fit", date(2026, 1, 1), date(2026, 2, 2)),
        )
    )
    audit = audit_prediction_provenance(
        observations=history,
        target="TEST",
        cutoff_date=cutoff,
        training_history=history,
        thresholds=_thresholds(),
        ranked_a=[future],
        ranked_b=[],
    )
    assert not audit.clean
    assert audit.relationship_parameter_temporal_violations == 1


def test_missing_parameter_provenance_fails_closed() -> None:
    cutoff = date(2026, 2, 1)
    history = [_obs(date(2026, 1, i), date(2026, 1, i + 1)) for i in (1, 5, 10)]
    missing = _ranking(parameter_provenance=())
    audit = audit_prediction_provenance(
        observations=history,
        target="TEST",
        cutoff_date=cutoff,
        training_history=history,
        thresholds=_thresholds(),
        ranked_a=[missing],
        ranked_b=[],
    )
    assert not audit.clean
    assert audit.relationship_parameter_missing == 1


def test_future_support_observation_fails_closed() -> None:
    cutoff = date(2026, 2, 1)
    history = [_obs(date(2026, 1, i), date(2026, 1, i + 1)) for i in (1, 5, 10)]
    future_support = _ranking(
        supporting_observations=((date(2026, 2, 2), 99.0),),
    )
    audit = audit_prediction_provenance(
        observations=history,
        target="TEST",
        cutoff_date=cutoff,
        training_history=history,
        thresholds=_thresholds(),
        ranked_a=[future_support],
        ranked_b=[],
    )
    assert not audit.clean
    assert audit.relationship_support_temporal_violations == 1


def test_thresholds_fitted_at_cutoff_are_unsafe() -> None:
    cutoff = date(2026, 2, 1)
    history = [
        _obs(date(2026, 1, 1), date(2026, 2, 1), -1.0),
        _obs(date(2026, 1, 5), date(2026, 1, 15), 1.0),
        _obs(date(2026, 1, 10), date(2026, 1, 20), 3.0),
    ]
    audit = audit_prediction_provenance(
        observations=history,
        target="TEST",
        cutoff_date=cutoff,
        training_history=history,
        thresholds=_thresholds(),
        ranked_a=[],
        ranked_b=[],
    )
    assert not audit.clean
    assert audit.threshold_temporal_violation is True


def test_validated_evidence_requires_clean_oos_artifact() -> None:
    cutoff = date(2026, 2, 1)
    history = [_obs(date(2026, 1, i), date(2026, 1, i + 1)) for i in (1, 5, 10)]
    audit = audit_prediction_provenance(
        observations=history,
        target="TEST",
        cutoff_date=cutoff,
        training_history=history,
        thresholds=_thresholds(),
        ranked_a=[_ranking()],
        ranked_b=[],
        validated_evidence=True,
        validation_evidence=None,
    )
    assert not audit.clean
    assert audit.validation_evidence_clean is False


def test_validation_artifact_reaching_cutoff_is_unsafe() -> None:
    cutoff = date(2026, 2, 1)
    history = [_obs(date(2026, 1, i), date(2026, 1, i + 1)) for i in (1, 5, 10)]
    evidence = ValidationEvidenceAudit(
        source="phase6-preview",
        validation_cutoff_date=cutoff,
        evaluated_folds=20,
        leakage_violations=0,
        outer_test_isolated=True,
        parameter_timing_safe=True,
        multiple_testing_controlled=True,
        selection_validated_predictions=5,
    )
    audit = audit_prediction_provenance(
        observations=history,
        target="TEST",
        cutoff_date=cutoff,
        training_history=history,
        thresholds=_thresholds(),
        ranked_a=[_ranking()],
        ranked_b=[],
        validated_evidence=True,
        validation_evidence=evidence,
    )
    assert evidence.clean
    assert not audit.clean
    assert audit.validation_evidence_temporal_violation is True


def test_prediction_provenance_is_auditable() -> None:
    audit = PredictionProvenanceAudit(
        target="TEST",
        cutoff_date=date(2026, 2, 1),
        source_count=3,
        training_count=3,
        latest_training_state_date=date(2026, 1, 10),
        latest_training_outcome_end_date=date(2026, 1, 20),
        excluded_target=0,
        excluded_future_state=0,
        excluded_incomplete_outcome=0,
        excluded_unknown_outcome_end=0,
        excluded_missing_return=0,
        excluded_non_finite_return=0,
        threshold_sample_count=3,
        threshold_fit_end_date=date(2026, 1, 20),
        threshold_temporal_violation=False,
        threshold_count_mismatch=False,
        relationship_candidate_count=1,
        relationship_parameter_count=1,
        relationship_parameter_missing=0,
        relationship_parameter_unknown=0,
        relationship_parameter_temporal_violations=0,
        relationship_support_temporal_violations=0,
        validated_evidence_asserted=False,
        validation_evidence_supplied=False,
        validation_evidence_clean=False,
        validation_evidence_temporal_violation=False,
    )
    data = audit.as_dict()
    assert data["clean"] is True
    assert data["threshold_fit_end_date"] == date(2026, 1, 20)
    assert data["temporal_violations"] == 0



def test_selected_relationships_bypass_in_sample_rediscovery() -> None:
    from analysis.prediction import PredictionEngine
    from analysis.relationship import RelationshipDiscoveryEngine, RelationshipResult
    from analysis.ranking import rank_relationships

    class FailingDiscoveryEngine(RelationshipDiscoveryEngine):
        def discover(self, current_states, observations, cutoff_date):  # type: ignore[override]
            raise AssertionError("in-sample discovery must not run for selected relationships")

    base = date(2025, 1, 1)
    history = [
        _obs(
            base + timedelta(days=index * 10),
            base + timedelta(days=index * 10 + 3),
            ret=float((index % 3) - 1),
        )
        for index in range(20)
    ]
    selected = RelationshipResult(
        method="B",
        variables=("company.market.price",),
        condition=("company.market.price=Rising",),
        sample_count=5,
        mean_return_pct=2.0,
        median_return_pct=2.0,
        baseline_mean_return_pct=0.0,
        lift_pct=2.0,
        positive_rate_pct=80.0,
        effect_strength=0.8,
        reliability=0.8,
        score=0.8,
        stable=True,
        supporting_observations=tuple(
            (item.as_of_date, float(item.stock_return_pct)) for item in history[-5:]
        ),
        supporting_weights=(1.0,) * 5,
        parameter_provenance=(
            ("state_relevance_frequency", history[0].as_of_date, history[-5].outcome_end_date),
            ("method_b_similarity_scale", history[0].as_of_date, history[-5].outcome_end_date),
        ),
    )
    ranking = rank_relationships([selected])[0]
    engine = PredictionEngine(
        relationship_engine=FailingDiscoveryEngine(max_order=1, min_observations=5),
        min_threshold_observations=3,
    )
    result = engine.predict(
        target="TEST",
        current_states={"company.market.price": "Rising"},
        observations=history,
        prediction_date=date(2026, 1, 1),
        selected_relationships={"A": None, "B": ranking},
    )
    assert result.method_b is not None
    assert result.method_b.variables == ("company.market.price",)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("PHASE 5.7 PROVENANCE HARDENING TEST: PASS")


def test_prediction_engine_surfaces_clean_provenance_audit() -> None:
    from analysis.prediction import PredictionEngine
    from analysis.relationship import RelationshipDiscoveryEngine

    base = date(2025, 1, 1)
    history = [
        _obs(
            base + timedelta(days=index * 10),
            base + timedelta(days=index * 10 + 3),
            ret=float((index % 3) - 1),
        )
        for index in range(30)
    ]
    engine = PredictionEngine(
        relationship_engine=RelationshipDiscoveryEngine(max_order=1, min_observations=5),
    )
    result = engine.predict(
        target="TEST",
        current_states={"company.market.price": "Rising"},
        observations=history,
        prediction_date=date(2026, 1, 1),
    )
    assert result.provenance_audit is not None
    assert result.provenance_audit.clean is True
    payload = result.as_dict()
    assert payload["provenance_audit"]["clean"] is True


def test_prediction_engine_suppresses_future_relationship_provenance() -> None:
    from analysis.prediction import PredictionEngine
    from analysis.relationship import RelationshipDiscoveryEngine

    class UnsafeDiscoveryEngine(RelationshipDiscoveryEngine):
        def discover(self, current_states, observations, cutoff_date):  # type: ignore[override]
            unsafe = RelationshipResult(
                method="A",
                variables=("company.market.price",),
                condition=("company.market.price=Rising",),
                sample_count=10,
                mean_return_pct=2.0,
                median_return_pct=2.0,
                baseline_mean_return_pct=0.0,
                lift_pct=2.0,
                positive_rate_pct=80.0,
                effect_strength=1.0,
                reliability=0.8,
                score=0.8,
                stable=True,
                supporting_observations=((date(2025, 6, 1), 2.0),),
                supporting_weights=(1.0,),
                parameter_provenance=(
                    ("future_fit", date(2025, 1, 1), date(2026, 1, 1)),
                ),
            )
            return {"method_a": [unsafe], "method_b": []}

    base = date(2025, 1, 1)
    history = [
        _obs(
            base + timedelta(days=index * 5),
            base + timedelta(days=index * 5 + 2),
            ret=float((index % 5) - 2),
        )
        for index in range(20)
    ]
    engine = PredictionEngine(
        relationship_engine=UnsafeDiscoveryEngine(max_order=1, min_observations=5),
    )
    result = engine.predict(
        target="TEST",
        current_states={"company.market.price": "Rising"},
        observations=history,
        prediction_date=date(2025, 10, 1),
    )
    assert result.provenance_audit is not None
    assert result.provenance_audit.clean is False
    assert result.limited is True
    assert result.trend == "NO_CLEAR_TREND"
    assert result.conviction == "NONE"
    assert result.method_a is None
    assert "fit at or after" in " ".join(result.limitations)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("PHASE 5.7 PROVENANCE HARDENING TEST: PASS")
