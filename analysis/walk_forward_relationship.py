from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

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

    def as_dict(self) -> dict:
        return {
            "folds": [fold.as_dict() for fold in self.folds],
            "method_a": self.method_a.as_dict(),
            "method_b": self.method_b.as_dict(),
            "combined": self.combined.as_dict(),
            "leakage_violations": self.leakage_violations,
            "skipped_insufficient_history": self.skipped_insufficient_history,
            "skipped_no_relationship": self.skipped_no_relationship,
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
