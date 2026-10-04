from datetime import date, timedelta

from analysis.relationship import RelationshipDiscoveryEngine, _candidate_feature_sets, _family
from analysis.relationship_graph import feature_lineage, compatible_pair, structurally_connected
from analysis.walk_forward_relationship import _discover_all_candidate_relationships
from analysis.adaptive_combination import discover_adaptive_higher_order
from analysis.relationship import HistoricalRelationshipObservation


def _history(n: int = 18):
    rows = []
    base = date(2025, 1, 1)
    for i in range(n):
        d = base + timedelta(days=i)
        states = {
            "institutional.FII_FPI.cash_net_value": "Rising" if i % 2 else "Falling",
            "derivatives.options.oi_pcr": "High" if i % 3 else "Low",
            "microstructure.delivery_percentage": "High" if i % 4 else "Low",
            "company.market.price": "Rising" if i % 2 else "Falling",
        }
        rows.append(
            HistoricalRelationshipObservation(
                as_of_date=d,
                target="TEST",
                scope="company",
                states=states,
                stock_return_pct=1.0 if i % 2 else -0.5,
                benchmark_return_pct=0.2,
                relative_return_pct=0.8 if i % 2 else -0.7,
                outcome_end_date=d + timedelta(days=1),
            )
        )
    return rows


def test_nested_flow_features_resolve_to_canonical_families():
    assert _family("institutional.FII_FPI.cash_net_value") == "institutional"
    assert _family("derivatives.options.oi_pcr") == "derivatives"
    assert _family("microstructure.delivery_percentage") == "microstructure"
    assert _family("trade_events.net_event_volume") == "trade_events"
    assert _family("company.market.price") == "company.market"


def test_duplicate_lineage_is_not_treated_as_independent_evidence():
    assert feature_lineage("derivatives.options.oi_pcr") == feature_lineage("microstructure.oi_pcr")
    assert not compatible_pair("derivatives.options.oi_pcr", "microstructure.oi_pcr")


def test_level_one_and_two_candidates_include_flow_pairs():
    features = [
        "institutional.FII_FPI.cash_net_value",
        "derivatives.options.oi_pcr",
        "company.market.price",
        "microstructure.delivery_percentage",
    ]
    candidates = _candidate_feature_sets(features, 2)
    assert ("institutional.FII_FPI.cash_net_value",) in candidates
    assert ("derivatives.options.oi_pcr",) in candidates
    assert tuple(sorted(("institutional.FII_FPI.cash_net_value", "company.market.price"))) in candidates
    assert tuple(sorted(("institutional.FII_FPI.cash_net_value", "derivatives.options.oi_pcr"))) in candidates
    assert tuple(sorted(("derivatives.options.oi_pcr", "microstructure.delivery_percentage"))) in candidates


def test_level_three_candidates_include_flow_combinations():
    features = [
        "institutional.FII_FPI.cash_net_value",
        "derivatives.options.oi_pcr",
        "company.market.price",
        "microstructure.delivery_percentage",
    ]
    candidates = _candidate_feature_sets(features, 3)
    assert tuple(sorted((
        "institutional.FII_FPI.cash_net_value",
        "derivatives.options.oi_pcr",
        "company.market.price",
    ))) in candidates
    assert tuple(sorted((
        "institutional.FII_FPI.cash_net_value",
        "microstructure.delivery_percentage",
        "company.market.price",
    ))) in candidates
    assert structurally_connected((
        "institutional.FII_FPI.cash_net_value",
        "derivatives.options.oi_pcr",
        "company.market.price",
    ))


def test_adaptive_higher_order_uses_flow_graph_and_respects_max_order():
    states = {
        "institutional.FII_FPI.cash_net_value": "Rising",
        "derivatives.options.oi_pcr": "High",
        "company.market.price": "Rising",
    }
    history = _history()
    engine = RelationshipDiscoveryEngine(max_order=3, min_observations=5)
    result = discover_adaptive_higher_order(states, history, engine)
    candidates = {
        tuple(sorted(r.variables))
        for ranking in result.rankings
        for r in ranking.method_results
    }
    assert tuple(sorted(states)) in candidates
    assert result.highest_order <= 3


def test_walk_forward_discovery_runs_method_a_and_b_at_flow_levels():
    states = _history(1)[0].states
    history = _history(18)
    engine = RelationshipDiscoveryEngine(max_order=3, min_observations=5)
    results, counts = _discover_all_candidate_relationships(states, history, engine)
    assert counts.get(1, 0) > 0
    assert counts.get(2, 0) > 0
    assert counts.get(3, 0) > 0
    methods = {item.method for item in results}
    assert methods == {"A", "B"}
    flow_results = [
        item for item in results
        if any(feature.startswith(("institutional.", "derivatives.", "microstructure.")) for feature in item.variables)
    ]
    assert flow_results


def test_method_b_vectorized_exhaustive_statistics_match_scalar_reference():
    import math
    from statistics import median

    current = {
        "institutional.FII_FPI.cash_net_value": "Rising",
        "derivatives.options.oi_pcr": "High",
        "company.market.price": "Rising",
    }
    history = _history(30)
    engine = RelationshipDiscoveryEngine(max_order=3, min_observations=5, candidate_batch_size=8)
    actual = engine.method_b_conditioned_distribution(current, history)
    actual_map = {tuple(item.variables): item for item in actual}

    candidate = tuple(sorted((
        "institutional.FII_FPI.cash_net_value",
        "derivatives.options.oi_pcr",
        "company.market.price",
    )))
    result = actual_map[candidate]

    # Scalar reference for one candidate, mirroring the Phase 4.4/4.10
    # Method-B equations exactly.
    relevance = {}
    total = len(history)
    frequencies = {}
    for row in history:
        for feature, state in row.states.items():
            frequencies[(feature, state)] = frequencies.get((feature, state), 0) + 1
    for key, frequency in frequencies.items():
        relevance[key] = math.log1p(total / max(frequency, 1))

    feature_weights = {
        feature: relevance.get((feature, current[feature]), 1.0)
        for feature in candidate
    }
    scored = []
    for row in history:
        compared_weight = 0.0
        observed_count = 0
        matches = 0
        for feature in candidate:
            observed = row.states.get(feature)
            weight = max(feature_weights[feature], 0.0)
            if observed is None:
                continue
            compared_weight += weight
            observed_count += 1
            if observed == current[feature]:
                matches += 1
        total_weight = sum(feature_weights.values())
        state_coverage = compared_weight / total_weight * 100.0
        family_coverage = observed_count / len(candidate) * 100.0
        if state_coverage >= engine.min_state_coverage_pct and family_coverage >= engine.min_family_coverage_pct:
            scored.append((row, matches / len(candidate), state_coverage, family_coverage))

    similarities = [item[1] for item in scored]
    positive = [value for value in similarities if value > 0.0]
    scale = median(positive) if positive else 1.0
    scale = max(scale, 1e-9)
    max_similarity = max(similarities)
    weighted_rows = [(row, math.exp((similarity - max_similarity) / scale)) for row, similarity, _, _ in scored]
    weight_total = sum(weight for _, weight in weighted_rows)
    outcomes = [row.stock_return_pct for row, _ in weighted_rows]
    weights = [weight for _, weight in weighted_rows]
    weighted_mean = sum(value * weight for value, weight in zip(outcomes, weights)) / weight_total
    positive_rate = sum(weight for value, weight in zip(outcomes, weights) if value > 0.0) / weight_total * 100.0
    ess = weight_total * weight_total / sum(weight * weight for weight in weights)
    baseline = [row.stock_return_pct for row in history]
    baseline_mean = sum(baseline) / len(baseline)
    baseline_dispersion = math.sqrt(sum((value - baseline_mean) ** 2 for value in baseline) / len(baseline))
    lift = weighted_mean - baseline_mean
    effect = abs(lift) / baseline_dispersion if baseline_dispersion > 0 else 0.0
    reliability = min(1.0, math.sqrt(ess / len(baseline)))
    score = effect * reliability

    assert math.isclose(result.mean_return_pct, weighted_mean, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result.positive_rate_pct, positive_rate, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result.effective_sample_size or 0.0, ess, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result.score, score, rel_tol=1e-12, abs_tol=1e-12)
    assert len(actual_map) > 0


def test_method_b_phase5_uses_exhaustive_search_when_flagged():
    current = _history(1)[0].states
    history = _history(25)
    engine = RelationshipDiscoveryEngine(
        max_order=3,
        min_observations=5,
        adaptive_higher_order=False,
        candidate_batch_size=8,
    )
    result = engine.method_b_conditioned_distribution(current, history)
    candidate_count = len(_candidate_feature_sets(current, 3))
    assert candidate_count > 0
    # The engine evaluates the same exhaustive candidate universe; results may
    # be fewer because minimum-observation/coverage rules can reject candidates.
    # This assertion therefore targets search enumeration, not result count.
    assert any(len(item.variables) == 3 for item in result)



def test_exhaustive_search_can_omit_candidate_support_and_materialize_selected_result():
    history = _history(24)
    current = history[-1].states
    engine = RelationshipDiscoveryEngine(
        max_order=3,
        min_observations=5,
        candidate_batch_size=16,
        retain_supporting_data=False,
    )
    results = engine.method_b_conditioned_distribution(current, history)
    assert results
    assert all(not item.supporting_observations for item in results)
    selected = max(results, key=lambda item: abs(item.score))
    materialized = engine.materialize_support(selected, history)
    assert materialized.supporting_observations
    assert len(materialized.supporting_observations) == materialized.sample_count
    assert len(materialized.supporting_weights) == materialized.sample_count
