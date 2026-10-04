from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import sqrt
from typing import Any, Iterable, Literal

from .outcome_labels import OutcomeClass, OutcomeThresholds, classify_return, learn_outcome_thresholds
from .ranking import RelationshipRanking, rank_relationships
from .relationship import HistoricalRelationshipObservation, RelationshipDiscoveryEngine

PredictionTrend = Literal["UP", "SIDEWAYS", "DOWN", "NO_CLEAR_TREND"]
Conviction = Literal["STRONG", "MODERATE", "LOW", "NONE"]


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _empty_probabilities() -> dict[OutcomeClass, float]:
    return {"UP": 0.0, "SIDEWAYS": 0.0, "DOWN": 0.0}


def _normalize_probabilities(values: dict[OutcomeClass, float]) -> dict[OutcomeClass, float]:
    total = sum(max(float(value), 0.0) for value in values.values())
    if total <= 0:
        return _empty_probabilities()
    return {
        key: max(float(value), 0.0) / total * 100.0
        for key, value in values.items()
    }


def _baseline_probabilities(
    history: list[HistoricalRelationshipObservation],
    thresholds: OutcomeThresholds,
) -> dict[OutcomeClass, float]:
    counts = _empty_probabilities()
    if thresholds.limited:
        return counts
    for observation in history:
        label = classify_return(observation.stock_return_pct, thresholds)
        counts[label] += 1.0
    return _normalize_probabilities(counts)


def _support_distribution(
    ranking: RelationshipRanking,
    thresholds: OutcomeThresholds,
) -> tuple[dict[OutcomeClass, float], float]:
    values = ranking.method_results
    if not values:
        return _empty_probabilities(), 0.0

    best = max(values, key=lambda item: abs(item.score))
    if not best.supporting_observations or thresholds.limited:
        return _empty_probabilities(), 0.0

    probabilities = _empty_probabilities()
    weights = list(best.supporting_weights)
    if len(weights) != len(best.supporting_observations):
        weights = [1.0] * len(best.supporting_observations)

    total_weight = sum(max(weight, 0.0) for weight in weights)
    if total_weight <= 0:
        return _empty_probabilities(), 0.0

    for (_support_date, return_pct), weight in zip(
        best.supporting_observations,
        weights,
    ):
        if weight <= 0:
            continue
        label = classify_return(float(return_pct), thresholds)
        probabilities[label] += float(weight)

    return _normalize_probabilities(probabilities), total_weight


def _top_direction(probabilities: dict[OutcomeClass, float]) -> OutcomeClass | None:
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    if not ordered or ordered[0][1] <= 0:
        return None
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return None
    return ordered[0][0]


def _standard_error_pct(probability_pct: float, effective_n: float) -> float:
    if effective_n <= 0:
        return 100.0
    p = max(0.0, min(1.0, probability_pct / 100.0))
    return sqrt(max(p * (1.0 - p), 0.0) / effective_n) * 100.0


def _directional_signal(
    probabilities: dict[OutcomeClass, float],
    baseline: dict[OutcomeClass, float],
    effective_n: float,
) -> PredictionTrend:
    """Choose a direction only when a directional class beats its own baseline."""
    candidates = ["UP", "DOWN"]
    lifts = {label: probabilities[label] - baseline[label] for label in candidates}
    viable = [label for label in candidates if lifts[label] > 0.0]
    if not viable:
        top = _top_direction(probabilities)
        return "SIDEWAYS" if top == "SIDEWAYS" else "NO_CLEAR_TREND"

    best = max(viable, key=lambda label: probabilities[label])
    second = max(
        (probabilities[label] for label in ("SIDEWAYS", "UP", "DOWN") if label != best),
        default=0.0,
    )

    margin = probabilities[best] - second
    uncertainty = _standard_error_pct(probabilities[best], effective_n)
    if margin <= uncertainty:
        return "NO_CLEAR_TREND"
    return best  # type: ignore[return-value]


@dataclass(frozen=True)
class MethodPrediction:
    method: str
    trend: PredictionTrend
    probabilities_pct: dict[OutcomeClass, float]
    baseline_probabilities_pct: dict[OutcomeClass, float]
    expected_return_pct: float
    baseline_return_pct: float
    return_lift_pct: float
    evidence_score: float
    reliability: float
    sample_count: int
    effective_sample_size: float
    stable: bool
    variables: tuple[str, ...]
    condition: tuple[str, ...]
    limited: bool = False
    limitations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "trend": self.trend,
            "probabilities_pct": dict(self.probabilities_pct),
            "baseline_probabilities_pct": dict(self.baseline_probabilities_pct),
            "expected_return_pct": self.expected_return_pct,
            "baseline_return_pct": self.baseline_return_pct,
            "return_lift_pct": self.return_lift_pct,
            "evidence_score": self.evidence_score,
            "reliability": self.reliability,
            "sample_count": self.sample_count,
            "effective_sample_size": self.effective_sample_size,
            "stable": self.stable,
            "variables": list(self.variables),
            "condition": list(self.condition),
            "limited": self.limited,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class PredictionResult:
    target: str
    prediction_date: date
    analysis_timeframe: str | None
    holding_period_months: float | None
    benchmark: str | None
    entry_mode: str | None
    training_observations: int
    outcome_thresholds: OutcomeThresholds
    baseline_probabilities_pct: dict[OutcomeClass, float]
    method_a: MethodPrediction | None
    method_b: MethodPrediction | None
    trend: PredictionTrend
    conviction: Conviction
    probabilities_pct: dict[OutcomeClass, float]
    expected_return_pct: float | None
    method_agreement: bool
    limited: bool
    limitations: tuple[str, ...] = ()
    validated_evidence: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "prediction_date": self.prediction_date,
            "analysis_timeframe": self.analysis_timeframe,
            "holding_period_months": self.holding_period_months,
            "benchmark": self.benchmark,
            "entry_mode": self.entry_mode,
            "training_observations": self.training_observations,
            "outcome_thresholds": self.outcome_thresholds.as_dict(),
            "baseline_probabilities_pct": dict(self.baseline_probabilities_pct),
            "method_a": self.method_a.as_dict() if self.method_a else None,
            "method_b": self.method_b.as_dict() if self.method_b else None,
            "trend": self.trend,
            "conviction": self.conviction,
            "probabilities_pct": dict(self.probabilities_pct),
            "expected_return_pct": self.expected_return_pct,
            "method_agreement": self.method_agreement,
            "limited": self.limited,
            "limitations": list(self.limitations),
            "validated_evidence": self.validated_evidence,
            "probability_basis": "empirical_conditional_class_share_uncalibrated",
        }


class PredictionEngine:
    """Phase 5 prediction layer over the validated Phase 4 relationship engine."""

    def __init__(
        self,
        relationship_engine: RelationshipDiscoveryEngine | None = None,
        min_threshold_observations: int = 9,
    ) -> None:
        self.relationship_engine = relationship_engine or RelationshipDiscoveryEngine(
            max_order=3,
            min_observations=5,
        )
        self.min_threshold_observations = int(min_threshold_observations)
        if self.min_threshold_observations < 3:
            raise ValueError("min_threshold_observations must be >= 3")

    @staticmethod
    def _training_history(
        observations: Iterable[HistoricalRelationshipObservation],
        target: str,
        cutoff_date: date,
    ) -> list[HistoricalRelationshipObservation]:
        """Return only labels fully realized before the prediction cutoff.

        A historical state may predate the cutoff while its forward outcome
        still extends beyond it. Such a row cannot be used for training at
        the cutoff without looking into the future, so hardened Phase 5
        excludes it. Unknown outcome horizons are excluded as well.
        """
        history = [
            observation
            for observation in observations
            if observation.target == target
            and _as_date(observation.as_of_date) < cutoff_date
            and observation.stock_return_pct is not None
            and observation.outcome_end_date is not None
            and _as_date(observation.outcome_end_date) < cutoff_date
        ]
        return sorted(history, key=lambda item: _as_date(item.as_of_date))

    def _method_prediction(
        self,
        ranking: RelationshipRanking | None,
        history: list[HistoricalRelationshipObservation],
        thresholds: OutcomeThresholds,
        baseline_probabilities: dict[OutcomeClass, float],
    ) -> MethodPrediction | None:
        if ranking is None:
            return None

        probabilities, support_weight = _support_distribution(
            ranking,
            thresholds,
        )
        best = max(ranking.method_results, key=lambda item: abs(item.score), default=None)
        if best is None or support_weight <= 0 or thresholds.limited:
            return MethodPrediction(
                method=best.method if best else "UNKNOWN",
                trend="NO_CLEAR_TREND",
                probabilities_pct=probabilities,
                baseline_probabilities_pct=baseline_probabilities,
                expected_return_pct=best.mean_return_pct if best else 0.0,
                baseline_return_pct=best.baseline_mean_return_pct if best else 0.0,
                return_lift_pct=best.lift_pct if best else 0.0,
                evidence_score=best.score if best else 0.0,
                reliability=best.reliability if best else 0.0,
                sample_count=best.sample_count if best else 0,
                effective_sample_size=best.effective_sample_size or support_weight if best else support_weight,
                stable=best.stable if best else False,
                variables=best.variables if best else (),
                condition=best.condition if best else (),
                limited=True,
                limitations=("Prediction evidence could not be converted into three-class probabilities.",),
            )

        effective_n = best.effective_sample_size or support_weight
        trend = _directional_signal(
            probabilities,
            baseline_probabilities,
            effective_n,
        )
        return MethodPrediction(
            method=best.method,
            trend=trend,
            probabilities_pct=probabilities,
            baseline_probabilities_pct=baseline_probabilities,
            expected_return_pct=best.mean_return_pct,
            baseline_return_pct=best.baseline_mean_return_pct,
            return_lift_pct=best.lift_pct,
            evidence_score=best.score,
            reliability=best.reliability,
            sample_count=best.sample_count,
            effective_sample_size=effective_n,
            stable=best.stable,
            variables=best.variables,
            condition=best.condition,
        )

    @staticmethod
    def _method_weight(method: MethodPrediction) -> float:
        """Evidence-adaptive weight, recomputed for each prediction."""
        sample_term = sqrt(max(method.effective_sample_size, 1.0))
        return max(method.evidence_score, 0.0) * max(method.reliability, 0.0) * sample_term

    def _combine(
        self,
        method_a: MethodPrediction | None,
        method_b: MethodPrediction | None,
        baseline: dict[OutcomeClass, float],
        *,
        validated_evidence: bool = False,
    ) -> tuple[PredictionTrend, Conviction, dict[OutcomeClass, float], float | None, bool]:
        methods = [method for method in (method_a, method_b) if method is not None and not method.limited]
        if not methods:
            return "NO_CLEAR_TREND", "NONE", dict(baseline), None, False

        weights = [self._method_weight(method) for method in methods]
        total = sum(weights)
        if total <= 0:
            return "NO_CLEAR_TREND", "NONE", dict(baseline), None, False

        combined = _empty_probabilities()
        expected_return = 0.0
        for method, weight in zip(methods, weights):
            share = weight / total
            for label in combined:
                combined[label] += method.probabilities_pct[label] * share
            expected_return += method.expected_return_pct * share
        combined = _normalize_probabilities(combined)

        directional = [method.trend for method in methods if method.trend in {"UP", "DOWN"}]
        agreement = len(directional) == 2 and directional[0] == directional[1]
        conflict = len(directional) == 2 and directional[0] != directional[1]

        trend = _directional_signal(combined, baseline, max(sum(method.effective_sample_size for method in methods), 1.0))
        if conflict:
            trend = "NO_CLEAR_TREND"

        if agreement and trend in {"UP", "DOWN"}:
            # STRONG is reserved for evidence that has survived an explicit
            # out-of-sample validation gate. Method agreement + stability alone
            # is still in-sample evidence and can only produce MODERATE.
            conviction: Conviction = (
                "STRONG"
                if validated_evidence and all(method.stable for method in methods)
                else "MODERATE"
            )
        elif len(directional) == 1 and trend == directional[0]:
            conviction = "MODERATE"
        elif trend == "SIDEWAYS":
            conviction = "LOW"
        else:
            conviction = "NONE" if conflict else "LOW"

        return trend, conviction, combined, expected_return, agreement

    def predict(
        self,
        target: str,
        current_states: dict[str, str],
        observations: Iterable[HistoricalRelationshipObservation],
        prediction_date: str | date | datetime,
        analysis_timeframe: str | None = None,
        holding_period_months: float | None = None,
        benchmark: str | None = None,
        entry_mode: str | None = None,
        validated_evidence: bool = False,
    ) -> PredictionResult:
        """
        Build a pre-calibration prediction from training observations only.

        Phase 5 produces empirical class probabilities. Probability calibration,
        cross-company robustness and trading-performance validation remain Phase 6.
        """
        cutoff = _as_date(prediction_date)
        source_observations = [
            observation
            for observation in observations
            if observation.target == target
            and _as_date(observation.as_of_date) < cutoff
            and observation.stock_return_pct is not None
        ]
        history = self._training_history(source_observations, target, cutoff)

        thresholds = learn_outcome_thresholds(
            [item.stock_return_pct for item in history],
            min_observations=self.min_threshold_observations,
        )
        baseline = _baseline_probabilities(history, thresholds)

        limitations: list[str] = []
        excluded_incomplete = len(source_observations) - len(history)
        if excluded_incomplete > 0:
            limitations.append(
                f"Excluded {excluded_incomplete} historical rows whose forward outcome "
                "was not fully realized before the prediction cutoff."
            )
        if thresholds.limited:
            limitations.append(thresholds.limitation or "Outcome thresholds are limited.")

        discovered = self.relationship_engine.discover(
            current_states=current_states,
            observations=history,
            cutoff_date=cutoff,
        )

        ranked_a = rank_relationships(discovered.get("method_a", []))
        ranked_b = rank_relationships(discovered.get("method_b", []))

        method_a = self._method_prediction(
            ranked_a[0] if ranked_a else None,
            history,
            thresholds,
            baseline,
        )
        method_b = self._method_prediction(
            ranked_b[0] if ranked_b else None,
            history,
            thresholds,
            baseline,
        )

        if method_a is None:
            limitations.append("Method A produced no usable relationship for the current state.")
        if method_b is None:
            limitations.append("Method B produced no usable relationship for the current state.")

        trend, conviction, probabilities, expected_return, agreement = self._combine(
            method_a,
            method_b,
            baseline,
            validated_evidence=validated_evidence,
        )

        # An absent relationship is an evidence limitation, not automatically
        # a data-quality limitation. Overall LIMITED is reserved for an
        # actually unusable outcome threshold or a method result that could
        # not be converted into probabilities.
        limited = thresholds.limited or bool(
            (method_a and method_a.limited) or (method_b and method_b.limited)
        )

        return PredictionResult(
            target=target,
            prediction_date=cutoff,
            analysis_timeframe=analysis_timeframe,
            holding_period_months=holding_period_months,
            benchmark=benchmark,
            entry_mode=entry_mode,
            training_observations=len(history),
            outcome_thresholds=thresholds,
            baseline_probabilities_pct=baseline,
            method_a=method_a,
            method_b=method_b,
            trend=trend,
            conviction=conviction,
            probabilities_pct=probabilities,
            expected_return_pct=expected_return,
            method_agreement=agreement,
            limited=limited,
            limitations=tuple(limitations),
            validated_evidence=validated_evidence,
        )
