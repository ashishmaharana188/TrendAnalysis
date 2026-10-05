from __future__ import annotations

from datetime import date, timedelta

from analysis.hardening_4_10 import (
    audit_parameter_timing,
    audit_state_surface,
    benjamini_hochberg,
    categorical_similarity,
    control_multiple_testing,
    split_stability,
)
from analysis.walk_forward_relationship import validate_walk_forward_relationships_hardened
from analysis.relationship import HistoricalRelationshipObservation, RelationshipDiscoveryEngine


def _family(feature: str) -> str:
    parts = feature.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]


def test_similarity_missingness_cannot_help() -> None:
    current = {
        "company.market.price": "Rising",
        "company.financials.revenue": "Rising",
        "industry.market.index": "Rising",
        "sector.market.index": "Rising",
    }
    weights = {feature: 1.0 for feature in current}
    complete = categorical_similarity(
        current,
        dict(current),
        weights,
        family_lookup=_family,
    )
    missing_one = dict(current)
    missing_one.pop("industry.market.index")
    partial = categorical_similarity(
        current,
        missing_one,
        weights,
        family_lookup=_family,
    )
    assert complete.score > partial.score
    assert partial.state_coverage_pct == 75.0
    assert partial.family_coverage_pct == 75.0


def test_stability_uses_magnitude_consistency() -> None:
    stable = split_stability([2.0, 2.1, 2.0, 2.2, 2.1, 2.0])
    unstable = split_stability([2.0, 2.1, 10.0, 11.0, 10.5, 11.5])
    assert stable.stable is True
    assert stable.stability_score > unstable.stability_score
    assert unstable.stable is False


def test_multiple_testing_control() -> None:
    adjusted, accepted = benjamini_hochberg([0.001, 0.01, 0.8, 0.9], alpha=0.10)
    assert adjusted[0] <= adjusted[1]
    assert 0 in accepted
    assert 2 not in accepted
    controlled = control_multiple_testing([0.001, 0.01, 0.8], candidate_count=20, alpha=0.10)
    assert controlled.candidate_count == 20
    assert controlled.tested_count == 3


def test_parameter_and_surface_audits() -> None:
    parameter_audit = audit_parameter_timing(
        [("zscore", date(2020, 1, 1), date(2026, 10, 2)),
         ("future_fit", date(2026, 9, 1), date(2026, 10, 4))],
        date(2026, 10, 3),
    )
    assert parameter_audit[0].temporal_violation is False
    assert parameter_audit[1].temporal_violation is True

    surface = audit_state_surface(
        {
            "company.market.price": "Rising",
            "industry.market.group_index": "Rising",
        },
        ["company.market", "company.financials", "industry.market", "sector.market", "macro"],
    )
    assert surface.coverage_pct == 40.0
    assert "macro" in surface.missing_families


def test_hardened_walk_forward_purges_overlapping_labels() -> None:
    base = date(2020, 1, 1)
    rows: list[HistoricalRelationshipObservation] = []
    for index in range(90):
        as_of = base + timedelta(days=index)
        positive_state = index % 3 != 0
        states = {
            "company.market.price": "Rising" if positive_state else "Falling",
            "company.financials.revenue": "Rising" if positive_state else "Falling",
        }
        ret = 3.0 if positive_state else -2.0
        rows.append(
            HistoricalRelationshipObservation(
                as_of_date=as_of,
                target="RELIANCE",
                scope="company",
                states=states,
                stock_return_pct=ret,
                benchmark_return_pct=1.0,
                relative_return_pct=ret - 1.0,
                outcome_end_date=as_of + timedelta(days=30),
            )
        )

    engine = RelationshipDiscoveryEngine(max_order=2, min_observations=5)
    result = validate_walk_forward_relationships_hardened(
        rows,
        engine,
        min_training_observations=12,
        selection_fraction=0.30,
        purge_overlapping_labels=True,
    )
    assert result.leakage_violations == 0
    assert result.purged_training_observations > 0
    assert result.unknown_overlap_observations == 0



def test_vectorized_search_permutation_matches_scalar_max_stat() -> None:
    from analysis.hardening_4_10 import search_adjusted_permutation_p_values

    base = date(2021, 1, 1)
    universe = [
        (base + timedelta(days=index), float((index % 7) - 3))
        for index in range(30)
    ]
    candidate_weights = []
    for offset in range(10):
        candidate_weights.append([
            1.0 if index % 5 == offset % 5 else 0.0
            for index in range(30)
        ])

    scalar = search_adjusted_permutation_p_values(
        [(universe, weights) for weights in candidate_weights],
        0.0,
        family_id="vector-test",
        universe_observations=universe,
        permutations=29,
        seed=17,
        compute_raw_p_values=True,
    )
    vectorized = search_adjusted_permutation_p_values(
        [(universe, weights) for weights in candidate_weights],
        0.0,
        family_id="vector-test",
        universe_observations=universe,
        permutations=29,
        seed=17,
        compute_raw_p_values=False,
    )
    assert vectorized.max_statistic_p_values == scalar.max_statistic_p_values


def main() -> None:
    test_similarity_missingness_cannot_help()
    test_stability_uses_magnitude_consistency()
    test_multiple_testing_control()
    test_parameter_and_surface_audits()
    test_hardened_walk_forward_purges_overlapping_labels()
    test_search_adjusted_permutation_accounts_for_candidate_search()
    test_fdr_isolated_by_explicit_testing_family()
    test_parameter_manifest_rejects_missing_learned_quantity()
    test_outer_test_outcome_does_not_change_selection()
    print("PHASE 4.10 METHODOLOGICAL HARDENING TEST: PASS")


def test_search_adjusted_permutation_accounts_for_candidate_search() -> None:
    from random import Random
    from analysis.hardening_4_10 import search_adjusted_permutation_p_values

    base = date(2020, 1, 1)
    universe = [
        (base + timedelta(days=index), 5.0 if index < 5 else -1.0)
        for index in range(100)
    ]
    baseline = sum(value for _day, value in universe) / len(universe)

    candidate_weights = [[
        1.0 if index in {0, 1, 50, 51, 52} else 0.0
        for index in range(100)
    ]]
    rng = Random(4)
    for _ in range(49):
        chosen = set(rng.sample(range(100), 5))
        candidate_weights.append([
            1.0 if index in chosen else 0.0
            for index in range(100)
        ])

    result = search_adjusted_permutation_p_values(
        [(universe, weights) for weights in candidate_weights],
        baseline,
        family_id="RELIANCE|company|A|2026-10-04",
        universe_observations=universe,
        permutations=199,
        seed=1,
    )

    assert result.candidate_count == 50
    assert result.max_statistic_p_values[0] >= result.raw_p_values[0]
    assert result.max_statistic_p_values[0] > 0.10



def test_search_permutation_reruns_discovery_and_ranking() -> None:
    from analysis.hardening_4_10 import search_procedure_permutation_p_value

    base = date(2020, 1, 1)
    rows = []
    for index in range(36):
        as_of = base + timedelta(days=index)
        positive = index % 2 == 0
        rows.append(
            HistoricalRelationshipObservation(
                as_of_date=as_of,
                target="RELIANCE",
                scope="company",
                states={
                    "company.market.price": "Rising" if positive else "Falling",
                    "company.financials.revenue": "Rising" if positive else "Falling",
                },
                stock_return_pct=4.0 if positive else -3.0,
                benchmark_return_pct=1.0,
                relative_return_pct=(4.0 if positive else -3.0) - 1.0,
            )
        )

    class CountingEngine(RelationshipDiscoveryEngine):
        def __init__(self):
            super().__init__(max_order=2, min_observations=5)
            self.a_calls = 0
            self.b_calls = 0

        def method_a_similar_states(self, current_states, history):
            self.a_calls += 1
            return super().method_a_similar_states(current_states, history)

        def method_b_conditioned_distribution(self, current_states, history):
            self.b_calls += 1
            return super().method_b_conditioned_distribution(current_states, history)

    engine = CountingEngine()
    result = search_procedure_permutation_p_value(
        current_states={
            "company.market.price": "Rising",
            "company.financials.revenue": "Rising",
        },
        history=rows,
        engine=engine,
        method="COMBINED",
        family_id="RELIANCE|company|COMBINED|2026-10-04",
        permutations=19,
        seed=7,
    )

    # One original search + one complete search per permutation, for both methods.
    assert engine.a_calls == 20
    assert engine.b_calls == 20
    assert result.permutation_count == 19
    assert result.observed_statistic > 0.0
    assert 0.0 < result.empirical_p_value <= 1.0

def test_fdr_isolated_by_explicit_testing_family() -> None:
    from analysis.hardening_4_10 import control_multiple_testing_by_family

    result = control_multiple_testing_by_family(
        {
            "RELIANCE|company|A|2026-10-04": [0.01, 0.02],
            "TCS|company|A|2026-10-04": [0.01, 0.02],
        },
        alpha=0.10,
    )
    assert set(result) == {
        "RELIANCE|company|A|2026-10-04",
        "TCS|company|A|2026-10-04",
    }
    assert result["RELIANCE|company|A|2026-10-04"].tested_count == 2
    assert result["RELIANCE|company|A|2026-10-04"].adjusted_p_values == result["TCS|company|A|2026-10-04"].adjusted_p_values


def test_parameter_manifest_rejects_missing_learned_quantity() -> None:
    from analysis.hardening_4_10 import audit_parameter_manifest

    audit = audit_parameter_manifest(
        ["normalization", "similarity_scale", "frequency_estimate"],
        [
            ("similarity_scale", date(2020, 1, 1), date(2026, 10, 2)),
            ("frequency_estimate", date(2020, 1, 1), date(2026, 10, 2)),
        ],
        date(2026, 10, 3),
    )
    assert audit.complete is False
    assert "normalization" in audit.missing_names


def test_outer_test_outcome_does_not_change_selection() -> None:
    base = date(2020, 1, 1)
    common_rows = []
    for index in range(89):
        as_of = base + timedelta(days=index)
        positive_state = index % 3 != 0
        common_rows.append(
            HistoricalRelationshipObservation(
                as_of_date=as_of,
                target="RELIANCE",
                scope="company",
                states={
                    "company.market.price": "Rising" if positive_state else "Falling",
                    "company.financials.revenue": "Rising" if positive_state else "Falling",
                },
                stock_return_pct=3.0 if positive_state else -2.0,
                benchmark_return_pct=1.0,
                relative_return_pct=(3.0 if positive_state else -2.0) - 1.0,
                outcome_end_date=as_of + timedelta(days=10),
            )
        )

    final_base = base + timedelta(days=89)
    final_state = {
        "company.market.price": "Rising",
        "company.financials.revenue": "Rising",
    }
    test_positive = HistoricalRelationshipObservation(
        as_of_date=final_base,
        target="RELIANCE",
        scope="company",
        states=final_state,
        stock_return_pct=9.0,
        benchmark_return_pct=1.0,
        relative_return_pct=8.0,
        outcome_end_date=final_base + timedelta(days=30),
    )
    test_negative = HistoricalRelationshipObservation(
        as_of_date=final_base,
        target="RELIANCE",
        scope="company",
        states=final_state,
        stock_return_pct=-9.0,
        benchmark_return_pct=1.0,
        relative_return_pct=-10.0,
        outcome_end_date=final_base + timedelta(days=30),
    )

    engine = RelationshipDiscoveryEngine(max_order=2, min_observations=5)
    result_positive = validate_walk_forward_relationships_hardened(
        [*common_rows, test_positive],
        engine,
        min_training_observations=12,
        selection_fraction=0.30,
        purge_overlapping_labels=True,
    )
    result_negative = validate_walk_forward_relationships_hardened(
        [*common_rows, test_negative],
        engine,
        min_training_observations=12,
        selection_fraction=0.30,
        purge_overlapping_labels=True,
    )

    last_positive = result_positive.folds[-1]
    last_negative = result_negative.folds[-1]
    assert (last_positive.method_a or {}).get("variables") == (last_negative.method_a or {}).get("variables")
    assert (last_positive.method_b or {}).get("variables") == (last_negative.method_b or {}).get("variables")
    assert (last_positive.method_a or {}).get("direction") == (last_negative.method_a or {}).get("direction")
    assert (last_positive.method_b or {}).get("direction") == (last_negative.method_b or {}).get("direction")


def main() -> None:
    test_similarity_missingness_cannot_help()
    test_stability_uses_magnitude_consistency()
    test_multiple_testing_control()
    test_parameter_and_surface_audits()
    test_hardened_walk_forward_purges_overlapping_labels()
    test_search_adjusted_permutation_accounts_for_candidate_search()
    test_fdr_isolated_by_explicit_testing_family()
    test_parameter_manifest_rejects_missing_learned_quantity()
    test_outer_test_outcome_does_not_change_selection()
    print("PHASE 4.10 METHODOLOGICAL HARDENING TEST: PASS")


if __name__ == "__main__":
    main()
