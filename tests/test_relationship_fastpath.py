from datetime import date, timedelta

from analysis.relationship import HistoricalRelationshipObservation, RelationshipDiscoveryEngine
class _EngineAdapter:
    min_observations = 5
    min_state_coverage_pct = 75.0
    min_family_coverage_pct = 75.0
    stability_sem_multiplier = 2.0
    candidate_batch_size = 8
    progress_every_candidates = 1000
    retain_supporting_data = True
    progress_callback = None

    @staticmethod
    def _progress(_message):
        return None


from analysis.relationship_fastpath import (
    method_a_prepared,
    method_b_prepared,
    prepare_relationship_context,
)


def _history(n: int = 24):
    base = date(2020, 1, 1)
    rows = []
    for i in range(n):
        rows.append(
            HistoricalRelationshipObservation(
                as_of_date=base + timedelta(days=i),
                target="T",
                scope="S",
                states={
                    "company.market.price": "Rising / High" if i % 3 else "Falling / Low",
                    "macro.rate": "High" if i % 4 else "Low",
                },
                stock_return_pct=float((i % 7) - 3),
                benchmark_return_pct=None,
                relative_return_pct=None,
                outcome_end_date=base + timedelta(days=i + 1),
            )
        )
    return rows


def _assert_equivalent(left, right):
    assert left.method == right.method
    assert left.variables == right.variables
    assert left.condition == right.condition
    assert left.sample_count == right.sample_count
    assert left.stable == right.stable
    assert left.exact_condition_count == right.exact_condition_count
    numeric_fields = (
        "mean_return_pct", "median_return_pct", "baseline_mean_return_pct",
        "lift_pct", "positive_rate_pct", "effect_strength", "reliability",
        "score", "state_coverage_pct", "family_coverage_pct", "stability_score",
    )
    for name in numeric_fields:
        a = getattr(left, name)
        b = getattr(right, name)
        assert abs(a - b) <= 1e-12, f"{name}: {a!r} != {b!r}"


def test_prepared_method_b_matches_reference_engine():
    history = _history()
    current = dict(history[-1].states)
    reference_engine = RelationshipDiscoveryEngine(max_order=2, min_observations=5)
    engine = _EngineAdapter()
    candidates = [
        ("company.market.price",),
        ("macro.rate",),
    ]

    reference = reference_engine.method_b_conditioned_distribution(current, history)
    reference = [r for r in reference if r.variables in candidates]

    context = prepare_relationship_context(current, history)
    fast = method_b_prepared(engine, current, history, candidates, context)

    assert {r.variables for r in fast} == {r.variables for r in reference}
    reference_by_vars = {r.variables: r for r in reference}
    for item in fast:
        _assert_equivalent(item, reference_by_vars[item.variables])


def test_prepared_method_a_matches_reference_for_same_candidate():
    history = _history()
    current = dict(history[-1].states)
    reference_engine = RelationshipDiscoveryEngine(max_order=2, min_observations=5)
    engine = _EngineAdapter()
    candidate = tuple(sorted(current))

    reference = reference_engine.method_a_similar_states(current, history)
    context = prepare_relationship_context(current, history)
    fast = method_a_prepared(engine, current, history, [candidate], context)

    assert len(reference) == 1
    assert len(fast) == 1
    _assert_equivalent(fast[0], reference[0])


def test_prepared_context_reuse_is_supported():
    history = _history()
    current = dict(history[-1].states)
    engine = _EngineAdapter()
    context = prepare_relationship_context(current, history)
    candidates = [("company.market.price",), ("macro.rate",)]

    method_a_prepared(engine, current, history, candidates, context)
    method_b_prepared(engine, current, history, candidates, context)


def test_compact_support_matches_materialized_support():
    from analysis.relationship_fastpath import materialize_selected_support
    history = _history()
    current = dict(history[-1].states)
    reference_engine = RelationshipDiscoveryEngine(max_order=2, min_observations=5)
    engine = _EngineAdapter()
    candidate = tuple(sorted(current))
    context = prepare_relationship_context(current, history)

    reference = next(item for item in reference_engine.method_a_similar_states(current, history) if item.variables == candidate)
    compact = method_a_prepared(engine, current, history, [candidate], context, compact_support=True)
    assert len(compact) == 1
    assert compact[0].supporting_observations == ()
    assert compact[0].supporting_weights == ()
    materialized = materialize_selected_support(compact[0], current, context, compact.support_matrix[0])
    assert materialized.supporting_observations == reference.supporting_observations
    assert materialized.supporting_weights == reference.supporting_weights


def test_compact_method_b_materialization_matches_reference():
    from analysis.relationship_fastpath import materialize_selected_support
    history = _history()
    current = dict(history[-1].states)
    reference_engine = RelationshipDiscoveryEngine(max_order=2, min_observations=5)
    engine = _EngineAdapter()
    candidate = ("macro.rate",)
    context = prepare_relationship_context(current, history)

    reference = next(item for item in reference_engine.method_b_conditioned_distribution(current, history) if item.variables == candidate)
    compact = method_b_prepared(engine, current, history, [candidate], context, compact_support=True)
    materialized = materialize_selected_support(compact[0], current, context, compact.support_matrix[0])
    assert materialized.supporting_observations == reference.supporting_observations
    assert materialized.supporting_weights == reference.supporting_weights
