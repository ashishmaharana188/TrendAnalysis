from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

from .hardening_4_10 import (
    audit_parameter_manifest,
    search_adjusted_permutation_p_values,
    TestingFamily,
    benjamini_hochberg,
)

from .ranking import RelationshipRanking, rank_relationships
from .relationship import (
    HistoricalRelationshipObservation,
    RelationshipDiscoveryEngine,
)


@dataclass(frozen=True)
class MethodValidationSummary:
    method: str
    evaluated_folds: int
    directional_predictions: int
    directional_hit_rate_pct: float
    coverage_pct: float
    mean_signed_return_pct: float
    mean_realized_return_pct: float
    selected_relationships: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class WalkForwardFold:
    prediction_date: date
    training_observations: int
    actual_return_pct: float
    method_a: dict | None
    method_b: dict | None
    combined: dict | None

    def as_dict(self) -> dict:
        return {
            "prediction_date": self.prediction_date,
            "training_observations": self.training_observations,
            "actual_return_pct": self.actual_return_pct,
            "method_a": self.method_a,
            "method_b": self.method_b,
            "combined": self.combined,
        }


@dataclass(frozen=True)
class WalkForwardValidationResult:
    folds: tuple[WalkForwardFold, ...]
    method_a: MethodValidationSummary
    method_b: MethodValidationSummary
    combined: MethodValidationSummary
    leakage_violations: int
    skipped_insufficient_history: int
    skipped_no_relationship: int
    purged_training_observations: int = 0
    unknown_overlap_observations: int = 0
    selection_candidate_evaluations: int = 0
    selection_validated_predictions: int = 0
    multiple_testing_controlled_folds: int = 0

    def as_dict(self) -> dict:
        return {
            "folds": [fold.as_dict() for fold in self.folds],
            "method_a": self.method_a.as_dict(),
            "method_b": self.method_b.as_dict(),
            "combined": self.combined.as_dict(),
            "leakage_violations": self.leakage_violations,
            "skipped_insufficient_history": self.skipped_insufficient_history,
            "skipped_no_relationship": self.skipped_no_relationship,
            "purged_training_observations": self.purged_training_observations,
            "unknown_overlap_observations": self.unknown_overlap_observations,
            "selection_candidate_evaluations": self.selection_candidate_evaluations,
            "selection_validated_predictions": self.selection_validated_predictions,
            "multiple_testing_controlled_folds": self.multiple_testing_controlled_folds,
        }


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _direction(value: float) -> str:
    if value > 0:
        return "POSITIVE"
    if value < 0:
        return "NEGATIVE"
    return "NEUTRAL"


def _predictive_direction(ranking: RelationshipRanking | None) -> str | None:
    if ranking is None:
        return None
    if ranking.direction not in {"POSITIVE", "NEGATIVE"}:
        return None
    return ranking.direction


def _fold_result(
    ranking: RelationshipRanking | None,
    actual_return_pct: float,
) -> dict | None:
    predicted = _predictive_direction(ranking)
    if predicted is None:
        return None

    actual = _direction(actual_return_pct)
    if actual == "NEUTRAL":
        signed = 0.0
        hit = False
    elif predicted == "POSITIVE":
        signed = actual_return_pct
        hit = actual == "POSITIVE"
    else:
        signed = -actual_return_pct
        hit = actual == "NEGATIVE"

    return {
        "variables": list(ranking.variables),
        "condition": list(ranking.condition),
        "direction": predicted,
        "score": ranking.best_score,
        "reliability": ranking.reliability,
        "sample_count": ranking.sample_count,
        "method_agreement": ranking.method_agreement,
        "stable_method_count": ranking.stable_method_count,
        "actual_direction": actual,
        "hit": hit,
        "signed_return_pct": signed,
        "realized_return_pct": actual_return_pct,
    }


def _summary(
    method: str,
    rows: list[dict],
    evaluated_folds: int,
) -> MethodValidationSummary:
    directional_predictions = len(rows)
    hits = sum(1 for row in rows if row["hit"])

    coverage = (
        directional_predictions / evaluated_folds * 100.0
        if evaluated_folds
        else 0.0
    )
    hit_rate = (
        hits / directional_predictions * 100.0
        if directional_predictions
        else 0.0
    )
    mean_signed = (
        sum(row["signed_return_pct"] for row in rows)
        / directional_predictions
        if directional_predictions
        else 0.0
    )
    mean_realized = (
        sum(row["realized_return_pct"] for row in rows)
        / directional_predictions
        if directional_predictions
        else 0.0
    )

    return MethodValidationSummary(
        method=method,
        evaluated_folds=evaluated_folds,
        directional_predictions=directional_predictions,
        directional_hit_rate_pct=hit_rate,
        coverage_pct=coverage,
        mean_signed_return_pct=mean_signed,
        mean_realized_return_pct=mean_realized,
        selected_relationships=directional_predictions,
    )


def _top_ranking(
    results: Iterable,
) -> RelationshipRanking | None:
    rankings = rank_relationships(results)
    if not rankings:
        return None
    return rankings[0]


def validate_walk_forward_relationships(
    observations: Iterable[HistoricalRelationshipObservation],
    engine: RelationshipDiscoveryEngine,
    min_training_observations: int | None = None,
) -> WalkForwardValidationResult:
    """
    Validate relationship discovery in strict chronological walk-forward order.

    For every prediction date T:
        train = observations strictly before T
        discover relationships using only train
        select the top relationship independently for Method A and Method B
        score the directional prediction against the held-out return at T

    The held-out observation is never passed into relationship discovery.
    This validates relationship direction and usefulness before later
    probability calibration creates the final UP/SIDEWAYS/DOWN labels.
    """
    rows = sorted(
        list(observations),
        key=lambda item: _as_date(item.as_of_date),
    )
    if not rows:
        empty = _summary("A", [], 0)
        return WalkForwardValidationResult(
            folds=(),
            method_a=empty,
            method_b=_summary("B", [], 0),
            combined=_summary("COMBINED", [], 0),
            leakage_violations=0,
            skipped_insufficient_history=0,
            skipped_no_relationship=0,
        )

    minimum = (
        engine.min_observations
        if min_training_observations is None
        else int(min_training_observations)
    )
    if minimum < engine.min_observations:
        raise ValueError(
            "min_training_observations cannot be below engine.min_observations"
        )

    folds: list[WalkForwardFold] = []
    method_a_rows: list[dict] = []
    method_b_rows: list[dict] = []
    combined_rows: list[dict] = []

    leakage_violations = 0
    skipped_insufficient_history = 0
    skipped_no_relationship = 0

    discovery_engine = RelationshipDiscoveryEngine(
        max_order=engine.max_order,
        min_observations=engine.min_observations,
    )

    for index, test_observation in enumerate(rows):
        prediction_date = _as_date(test_observation.as_of_date)
        training = [
            observation
            for observation in rows[:index]
            if _as_date(observation.as_of_date) < prediction_date
            and observation.stock_return_pct is not None
        ]

        if len(training) < minimum:
            skipped_insufficient_history += 1
            continue

        if any(
            _as_date(observation.as_of_date) >= prediction_date
            for observation in training
        ):
            leakage_violations += 1
            continue

        method_a = discovery_engine.method_a_similar_states(
            current_states=test_observation.states,
            history=training,
        )
        method_b = discovery_engine.method_b_conditioned_distribution(
            current_states=test_observation.states,
            history=training,
        )

        top_a = _top_ranking(method_a)
        top_b = _top_ranking(method_b)

        row_a = _fold_result(top_a, test_observation.stock_return_pct)
        row_b = _fold_result(top_b, test_observation.stock_return_pct)

        combined_ranking = _top_ranking([
            *method_a,
            *method_b,
        ])
        row_combined = _fold_result(
            combined_ranking,
            test_observation.stock_return_pct,
        )

        if row_a is None and row_b is None and row_combined is None:
            skipped_no_relationship += 1

        if row_a is not None:
            method_a_rows.append(row_a)
        if row_b is not None:
            method_b_rows.append(row_b)
        if row_combined is not None:
            combined_rows.append(row_combined)

        folds.append(
            WalkForwardFold(
                prediction_date=prediction_date,
                training_observations=len(training),
                actual_return_pct=float(test_observation.stock_return_pct),
                method_a=row_a,
                method_b=row_b,
                combined=row_combined,
            )
        )

    evaluated = len(folds)

    return WalkForwardValidationResult(
        folds=tuple(folds),
        method_a=_summary("A", method_a_rows, evaluated),
        method_b=_summary("B", method_b_rows, evaluated),
        combined=_summary("COMBINED", combined_rows, evaluated),
        leakage_violations=leakage_violations,
        skipped_insufficient_history=skipped_insufficient_history,
        skipped_no_relationship=skipped_no_relationship,
    )



def _method_results_for_candidate(
    method: str,
    candidate_variables: tuple[str, ...],
    current_states: dict[str, str],
    history: list[HistoricalRelationshipObservation],
    engine: RelationshipDiscoveryEngine,
) -> object | None:
    subset = {feature: current_states[feature] for feature in candidate_variables if feature in current_states}
    if len(subset) != len(candidate_variables):
        return None
    if method == "A":
        results = engine.method_a_similar_states(subset, history)
    else:
        results = engine.method_b_conditioned_distribution(subset, history)
    return next((item for item in results if tuple(sorted(item.variables)) == tuple(sorted(candidate_variables))), None)


def _holdout_select(
    method: str,
    discovery_results: list,
    current_states: dict[str, str],
    selection_history: list[HistoricalRelationshipObservation],
    engine: RelationshipDiscoveryEngine,
    family: TestingFamily,
    *,
    alpha: float,
) -> tuple[object | None, int, bool]:
    """
    Select a discovered relationship on a chronological inner holdout.

    Discovery candidates are fixed before the selection set is touched. Each
    candidate is independently re-evaluated on the selection set, then the
    empirical p-values are FDR-adjusted. No candidate with a failed adjustment
    is promoted to the outer test prediction.
    """
    if not discovery_results or not selection_history:
        return None, 0, False

    evaluations: list[object] = []
    baseline_mean = sum(item.stock_return_pct for item in selection_history) / len(selection_history)

    for candidate in discovery_results:
        evaluated = _method_results_for_candidate(
            method,
            tuple(candidate.variables),
            current_states,
            selection_history,
            engine,
        )
        if evaluated is None or not getattr(evaluated, "supporting_observations", None):
            continue
        evaluations.append(evaluated)

    if not evaluations:
        return None, 0, False

    permutation = search_adjusted_permutation_p_values(
        [
            (result.supporting_observations, result.supporting_weights)
            for result in evaluations
        ],
        baseline_mean,
        family_id=family.family_id,
        universe_observations=[
            (item.as_of_date, float(item.stock_return_pct))
            for item in selection_history
        ],
        permutations=199,
        seed=sum(ord(ch) for ch in family.family_id) + len(selection_history),
    )
    adjusted, accepted = benjamini_hochberg(
        permutation.max_statistic_p_values,
        alpha=alpha,
    )
    if not accepted:
        return None, len(evaluations), True

    accepted_rows = [
        (evaluations[index], adjusted[index])
        for index in accepted
    ]
    selected, _q = min(
        accepted_rows,
        key=lambda row: (
            row[1],
            -abs(float(row[0].score)),
            -float(row[0].reliability),
        ),
    )
    return selected, len(evaluations), True


def validate_walk_forward_relationships_hardened(
    observations: Iterable[HistoricalRelationshipObservation],
    engine: RelationshipDiscoveryEngine,
    min_training_observations: int | None = None,
    *,
    selection_fraction: float = 0.30,
    multiple_testing_alpha: float = 0.10,
    purge_overlapping_labels: bool = True,
) -> WalkForwardValidationResult:
    """
    Methodological-hardening validator for Phase 4.10.

    Outer test observations are never used for discovery or selection. The
    training sample is split chronologically into discovery and selection
    segments. Candidate relationships are discovered in the earlier segment,
    selected on the later segment with FDR control, then scored on the held-out
    outer observation. When outcome_end_date is known, training labels whose
    realized horizons overlap the outer prediction date are purged. Unknown
    horizons are conservatively excluded in hardened mode.
    """
    if not 0.0 < selection_fraction < 1.0:
        raise ValueError("selection_fraction must be in (0, 1)")
    if not 0.0 < multiple_testing_alpha < 1.0:
        raise ValueError("multiple_testing_alpha must be in (0, 1)")

    rows = sorted(list(observations), key=lambda item: _as_date(item.as_of_date))
    if not rows:
        return validate_walk_forward_relationships(rows, engine, min_training_observations)

    minimum = engine.min_observations if min_training_observations is None else int(min_training_observations)
    if minimum < engine.min_observations:
        raise ValueError("min_training_observations cannot be below engine.min_observations")

    folds: list[WalkForwardFold] = []
    method_a_rows: list[dict] = []
    method_b_rows: list[dict] = []
    combined_rows: list[dict] = []
    leakage_violations = 0
    skipped_insufficient_history = 0
    skipped_no_relationship = 0
    purged_training = 0
    unknown_overlap = 0
    selection_evaluations = 0
    selection_validated = 0
    controlled_folds = 0

    for index, test_observation in enumerate(rows):
        prediction_date = _as_date(test_observation.as_of_date)
        raw_training = [
            item for item in rows[:index]
            if _as_date(item.as_of_date) < prediction_date
            and item.stock_return_pct is not None
        ]
        if len(raw_training) < minimum:
            skipped_insufficient_history += 1
            continue

        training: list[HistoricalRelationshipObservation] = []
        for item in raw_training:
            end_date = item.outcome_end_date
            if purge_overlapping_labels:
                if end_date is None:
                    unknown_overlap += 1
                    continue
                if _as_date(end_date) >= prediction_date:
                    purged_training += 1
                    continue
            training.append(item)

        if len(training) < minimum:
            skipped_insufficient_history += 1
            continue
        if any(_as_date(item.as_of_date) >= prediction_date for item in training):
            leakage_violations += 1
            continue

        split_index = int(len(training) * (1.0 - selection_fraction))
        split_index = min(max(split_index, engine.min_observations), len(training) - 1)
        discovery_history = training[:split_index]
        selection_history = training[split_index:]
        if len(discovery_history) < minimum or len(selection_history) < engine.min_observations:
            skipped_insufficient_history += 1
            continue

        discovery_engine = RelationshipDiscoveryEngine(
            max_order=engine.max_order,
            min_observations=engine.min_observations,
            min_state_coverage_pct=engine.min_state_coverage_pct,
            min_family_coverage_pct=engine.min_family_coverage_pct,
            stability_sem_multiplier=engine.stability_sem_multiplier,
        )
        discovery_a = discovery_engine.method_a_similar_states(test_observation.states, discovery_history)
        discovery_b = discovery_engine.method_b_conditioned_distribution(test_observation.states, discovery_history)

        required_a = (
            "state_relevance_frequency",
            "method_a_adaptive_neighbor_count",
        )
        required_b = (
            "state_relevance_frequency",
            "method_b_similarity_scale",
        )
        audits_a = [
            audit_parameter_manifest(required_a, result.parameter_provenance, prediction_date)
            for result in discovery_a
        ]
        audits_b = [
            audit_parameter_manifest(required_b, result.parameter_provenance, prediction_date)
            for result in discovery_b
        ]
        if any(audit.temporal_violation for audit in (*audits_a, *audits_b)):
            leakage_violations += 1
            continue
        if any(not audit.complete for audit in (*audits_a, *audits_b)):
            skipped_insufficient_history += 1
            continue

        family_a = TestingFamily(
            target=test_observation.target,
            scope=test_observation.scope,
            method="A",
            prediction_date=prediction_date,
        )
        family_b = TestingFamily(
            target=test_observation.target,
            scope=test_observation.scope,
            method="B",
            prediction_date=prediction_date,
        )

        selected_a, count_a, controlled_a = _holdout_select(
            "A", discovery_a, test_observation.states, selection_history, discovery_engine, family_a, alpha=multiple_testing_alpha
        )
        selected_b, count_b, controlled_b = _holdout_select(
            "B", discovery_b, test_observation.states, selection_history, discovery_engine, family_b, alpha=multiple_testing_alpha
        )
        selection_evaluations += count_a + count_b
        if controlled_a or controlled_b:
            controlled_folds += 1
        if selected_a is not None or selected_b is not None:
            selection_validated += 1

        # The selected inner-holdout result supplies the directional prediction
        # for the outer test date. No test outcome enters this step.
        ranking_a = _top_ranking([selected_a] if selected_a is not None else [])
        ranking_b = _top_ranking([selected_b] if selected_b is not None else [])

        row_a = _fold_result(ranking_a, float(test_observation.stock_return_pct))
        row_b = _fold_result(ranking_b, float(test_observation.stock_return_pct))

        common_ranking = None
        if ranking_a is not None and ranking_b is not None and ranking_a.direction == ranking_b.direction:
            common_ranking = ranking_a
        row_combined = _fold_result(common_ranking, float(test_observation.stock_return_pct))

        if row_a is None and row_b is None and row_combined is None:
            skipped_no_relationship += 1

        if row_a is not None:
            method_a_rows.append(row_a)
        if row_b is not None:
            method_b_rows.append(row_b)
        if row_combined is not None:
            combined_rows.append(row_combined)

        folds.append(
            WalkForwardFold(
                prediction_date=prediction_date,
                training_observations=len(training),
                actual_return_pct=float(test_observation.stock_return_pct),
                method_a=row_a,
                method_b=row_b,
                combined=row_combined,
            )
        )

    evaluated = len(folds)
    return WalkForwardValidationResult(
        folds=tuple(folds),
        method_a=_summary("A", method_a_rows, evaluated),
        method_b=_summary("B", method_b_rows, evaluated),
        combined=_summary("COMBINED", combined_rows, evaluated),
        leakage_violations=leakage_violations,
        skipped_insufficient_history=skipped_insufficient_history,
        skipped_no_relationship=skipped_no_relationship,
        purged_training_observations=purged_training,
        unknown_overlap_observations=unknown_overlap,
        selection_candidate_evaluations=selection_evaluations,
        selection_validated_predictions=selection_validated,
        multiple_testing_controlled_folds=controlled_folds,
    )
