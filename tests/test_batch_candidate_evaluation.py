from datetime import date, timedelta

from analysis.relationship import HistoricalRelationshipObservation, RelationshipDiscoveryEngine
from analysis.relationship_4_3_backup import RelationshipDiscoveryEngine as LegacyRelationshipDiscoveryEngine
from analysis.relationship_graph import candidate_feature_sets


def _history():
    features = [
        "company.market.price",
        "institutional.FII_FPI.cash_net_value",
        "derivatives.options.oi_pcr",
        "macro.US_VIX",
    ]
    states = ["Rising", "Falling", "Stable", "High", "Low"]
    rows = []
    for i in range(18):
        rows.append(
            HistoricalRelationshipObservation(
                as_of_date=date(2025, 1, 1) + timedelta(days=i),
                target="TEST",
                scope="company",
                states={feature: states[(i + j) % len(states)] for j, feature in enumerate(features)},
                stock_return_pct=float((i % 7) - 3),
                benchmark_return_pct=0.1,
                relative_return_pct=float((i % 7) - 3) - 0.1,
                outcome_end_date=date(2025, 1, 20) + timedelta(days=i),
            )
        )
    return features, rows


def _fields():
    return (
        "sample_count",
        "mean_return_pct",
        "median_return_pct",
        "baseline_mean_return_pct",
        "lift_pct",
        "positive_rate_pct",
        "effect_strength",
        "reliability",
        "score",
        "stable",
        "state_coverage_pct",
        "family_coverage_pct",
        "stability_score",
    )


def test_batch_method_a_matches_legacy_single_candidate_results():
    features, history = _history()
    current = {feature: history[-1].states[feature] for feature in features}
    candidates = candidate_feature_sets(features, 3)[:20]

    engine = RelationshipDiscoveryEngine(max_order=3, min_observations=5, progress_callback=None)
    batched = engine.method_a_similar_states(current, history, candidate_sets=candidates)

    legacy = LegacyRelationshipDiscoveryEngine(max_order=3, min_observations=5)
    expected = {}
    for candidate in candidates:
        result = legacy.method_a_similar_states({feature: current[feature] for feature in candidate}, history)
        for item in result:
            if tuple(sorted(item.variables)) == tuple(sorted(candidate)):
                expected[tuple(sorted(candidate))] = item

    actual = {tuple(sorted(item.variables)): item for item in batched}
    assert set(actual) == set(expected)
    for key in expected:
        for field in _fields():
            assert getattr(actual[key], field) == getattr(expected[key], field)


def test_batch_method_b_matches_batched_single_candidate_execution():
    features, history = _history()
    current = {feature: history[-1].states[feature] for feature in features}
    candidates = candidate_feature_sets(features, 3)[:20]

    engine = RelationshipDiscoveryEngine(max_order=3, min_observations=5, progress_callback=None)
    batched = engine.method_b_conditioned_distribution(current, history, candidate_sets=candidates)

    # The same engine implementation is asked to evaluate one candidate at a
    # time here. This isolates the candidate-batching optimization from the
    # underlying Method B equations.
    expected = {}
    for candidate in candidates:
        result = engine.method_b_conditioned_distribution(
            {feature: current[feature] for feature in candidate},
            history,
        )
        for item in result:
            if tuple(sorted(item.variables)) == tuple(sorted(candidate)):
                expected[tuple(sorted(candidate))] = item

    actual = {tuple(sorted(item.variables)): item for item in batched}
    assert set(actual) == set(expected)
    for key in expected:
        for field in _fields():
            left = getattr(actual[key], field)
            right = getattr(expected[key], field)
            if isinstance(left, float):
                assert abs(left - right) < 1e-12
            else:
                assert left == right
