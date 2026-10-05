from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite, log
from typing import Any, Callable, Iterable, Sequence
import logging
import time

from .outcome_labels import OutcomeClass, classify_return
from .prediction import PredictionEngine, PredictionResult
from .ranking import rank_relationships
from .walk_forward_relationship import validate_walk_forward_relationships_hardened
from .relationship import HistoricalRelationshipObservation

try:
    from typing import TYPE_CHECKING
except ImportError:  # pragma: no cover
    TYPE_CHECKING = False

if TYPE_CHECKING:
    from .real_olap_validation import RealOLAPValidationConfig


PredictionTrend = str

_LOGGER = logging.getLogger("trendanalysis.phase5.real_olap")

def _configure_phase5_progress(enabled: bool) -> None:
    if not enabled:
        return
    if not _LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[Phase 5.8] %(message)s"))
        _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False

def _phase5_progress(enabled: bool, message: str, *args: Any) -> None:
    if enabled:
        _LOGGER.info(message, *args)



@dataclass(frozen=True)
class PredictionFoldResult:
    prediction_date: date
    actual_return_pct: float
    actual_class: OutcomeClass | None
    predicted_trend: PredictionTrend
    conviction: str
    probabilities_pct: dict[OutcomeClass, float]
    baseline_probabilities_pct: dict[OutcomeClass, float]
    expected_return_pct: float | None
    training_observations: int
    method_a_trend: PredictionTrend | None
    method_b_trend: PredictionTrend | None
    method_agreement: bool
    limited: bool
    provenance_clean: bool
    hit: bool | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "prediction_date": self.prediction_date,
            "actual_return_pct": self.actual_return_pct,
            "actual_class": self.actual_class,
            "predicted_trend": self.predicted_trend,
            "conviction": self.conviction,
            "probabilities_pct": dict(self.probabilities_pct),
            "baseline_probabilities_pct": dict(self.baseline_probabilities_pct),
            "expected_return_pct": self.expected_return_pct,
            "training_observations": self.training_observations,
            "method_a_trend": self.method_a_trend,
            "method_b_trend": self.method_b_trend,
            "method_agreement": self.method_agreement,
            "limited": self.limited,
            "provenance_clean": self.provenance_clean,
            "hit": self.hit,
        }


@dataclass(frozen=True)
class PredictionMethodSummary:
    method: str
    evaluated_predictions: int
    directional_predictions: int
    directional_hits: int
    directional_hit_rate_pct: float
    clear_trend_predictions: int
    clear_trend_hits: int
    clear_trend_accuracy_pct: float
    no_clear_trend_predictions: int
    limited_predictions: int
    coverage_pct: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class Phase5RealOLAPValidationResult:
    ticker: str
    benchmark: str
    analysis_timeframe: str
    holding_period_months: float
    entry_mode: str
    candidate_predictions: int
    evaluated_predictions: int
    skipped_threshold_limited: int
    skipped_provenance_failed: int
    skipped_invalid_actual: int
    prediction_folds: tuple[PredictionFoldResult, ...]
    method_a: PredictionMethodSummary
    method_b: PredictionMethodSummary
    combined: PredictionMethodSummary
    baseline_majority_accuracy_pct: float
    mean_combined_probabilities_pct: dict[OutcomeClass, float]
    mean_combined_expected_return_pct: float | None
    conviction_counts: dict[str, int]
    latest_validated_prediction_date: date | None
    latest_validated_prediction: dict[str, Any] | None
    latest_market_date: date | None
    phase4_surface_audit: dict[str, Any] | None
    selection_candidate_evaluations: int = 0
    selection_validated_predictions: int = 0
    multiple_testing_controlled_folds: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "benchmark": self.benchmark,
            "analysis_timeframe": self.analysis_timeframe,
            "holding_period_months": self.holding_period_months,
            "entry_mode": self.entry_mode,
            "candidate_predictions": self.candidate_predictions,
            "evaluated_predictions": self.evaluated_predictions,
            "skipped_threshold_limited": self.skipped_threshold_limited,
            "skipped_provenance_failed": self.skipped_provenance_failed,
            "skipped_invalid_actual": self.skipped_invalid_actual,
            "prediction_folds": [fold.as_dict() for fold in self.prediction_folds],
            "method_a": self.method_a.as_dict(),
            "method_b": self.method_b.as_dict(),
            "combined": self.combined.as_dict(),
            "baseline_majority_accuracy_pct": self.baseline_majority_accuracy_pct,
            "mean_combined_probabilities_pct": dict(self.mean_combined_probabilities_pct),
            "mean_combined_expected_return_pct": self.mean_combined_expected_return_pct,
            "conviction_counts": dict(self.conviction_counts),
            "latest_validated_prediction_date": self.latest_validated_prediction_date,
            "latest_validated_prediction": self.latest_validated_prediction,
            "latest_market_date": self.latest_market_date,
            "phase4_surface_audit": self.phase4_surface_audit,
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


def _percent(value: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return (value / denominator) * 100.0


def _label_direction(trend: str | None) -> bool:
    return trend in {"UP", "DOWN"}


def _directional_hit(predicted: str | None, actual: OutcomeClass | None) -> bool | None:
    if predicted not in {"UP", "DOWN"} or actual is None:
        return None
    return predicted == actual


def _summary(
    method: str,
    folds: Sequence[PredictionFoldResult],
    trend_getter: Callable[[PredictionFoldResult], str | None],
) -> PredictionMethodSummary:
    usable = [fold for fold in folds if not fold.limited and fold.provenance_clean]
    directional = [fold for fold in usable if _label_direction(trend_getter(fold))]
    directional_hits = [fold for fold in directional if _directional_hit(trend_getter(fold), fold.actual_class) is True]
    clear = [fold for fold in usable if trend_getter(fold) in {"UP", "SIDEWAYS", "DOWN"}]
    clear_hits = [fold for fold in clear if trend_getter(fold) == fold.actual_class]
    no_clear = [fold for fold in usable if trend_getter(fold) == "NO_CLEAR_TREND"]
    limited = [fold for fold in folds if fold.limited or not fold.provenance_clean]

    return PredictionMethodSummary(
        method=method,
        evaluated_predictions=len(usable),
        directional_predictions=len(directional),
        directional_hits=len(directional_hits),
        directional_hit_rate_pct=_percent(len(directional_hits), len(directional)),
        clear_trend_predictions=len(clear),
        clear_trend_hits=len(clear_hits),
        clear_trend_accuracy_pct=_percent(len(clear_hits), len(clear)),
        no_clear_trend_predictions=len(no_clear),
        limited_predictions=len(limited),
        coverage_pct=_percent(len(clear), len(folds)),
    )


def _majority_baseline_accuracy(
    folds: Sequence[PredictionFoldResult],
) -> float:
    actual = [fold.actual_class for fold in folds if fold.actual_class is not None]
    if not actual:
        return 0.0
    counts = {label: actual.count(label) for label in ("UP", "SIDEWAYS", "DOWN")}
    return _percent(max(counts.values()), len(actual))


def _mean_probabilities(folds: Sequence[PredictionFoldResult]) -> dict[OutcomeClass, float]:
    usable = [fold for fold in folds if not fold.limited and fold.provenance_clean]
    if not usable:
        return {"UP": 0.0, "SIDEWAYS": 0.0, "DOWN": 0.0}
    return {
        label: sum(fold.probabilities_pct.get(label, 0.0) for fold in usable) / len(usable)
        for label in ("UP", "SIDEWAYS", "DOWN")
    }


def _mean_expected_return(folds: Sequence[PredictionFoldResult]) -> float | None:
    values = [
        float(fold.expected_return_pct)
        for fold in folds
        if not fold.limited
        and fold.provenance_clean
        and fold.expected_return_pct is not None
        and isfinite(float(fold.expected_return_pct))
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _assert_fold_temporal_safety(
    result: PredictionResult,
    cutoff: date,
    source: Iterable[HistoricalRelationshipObservation],
) -> None:
    """Raise if the Phase 5 output contradicts its own cutoff contract."""
    audit = result.provenance_audit
    if audit is None:
        raise AssertionError("Phase 5.7 provenance audit is missing from PredictionResult.")
    if not audit.clean:
        return
    for row in source:
        if row.target != result.target:
            continue
        state_date = _as_date(row.as_of_date)
        if state_date >= cutoff:
            continue
        if row.outcome_end_date is None:
            continue
        if _as_date(row.outcome_end_date) >= cutoff:
            continue
    if audit.threshold_fit_end_date is not None and _as_date(audit.threshold_fit_end_date) >= cutoff:
        raise AssertionError("Prediction thresholds reach or exceed the prediction cutoff.")
    if audit.latest_training_outcome_end_date is not None and _as_date(audit.latest_training_outcome_end_date) >= cutoff:
        raise AssertionError("Training outcome reaches or exceeds the prediction cutoff.")


def validate_real_olap_predictions(
    config: RealOLAPValidationConfig,
    *,
    panel_builder: Callable[[RealOLAPValidationConfig], tuple[list[HistoricalRelationshipObservation], dict[str, Any], date | None]] | None = None,
    prediction_engine: PredictionEngine | None = None,
) -> Phase5RealOLAPValidationResult:
    """Run Phase 5 predictions on the real Phase 4 OLAP state/outcome panel.

    Each prediction uses exactly one historical panel state as the current
    condition and passes the *entire* panel to ``PredictionEngine``. Phase 5.7
    then removes any observation whose state or forward outcome is not known by
    that fold's prediction cutoff. The actual outcome for that fold is used
    only after the prediction has been generated.

    This is a Phase 5 diagnostic validation, not Phase 6 calibration or a
    trading-performance claim.
    """
    if panel_builder is None:
        from .real_olap_validation import build_real_olap_relationship_panel
        panel_builder = build_real_olap_relationship_panel

    observations, panel_stats, latest_market_date = panel_builder(config)
    observations = sorted(observations, key=lambda row: _as_date(row.as_of_date))
    candidate_predictions = len(observations)

    _configure_phase5_progress(getattr(config, "progress_logging", True))

    # Hardened Phase 5.8 must not select the strongest relationship directly
    # from the same history used to estimate its apparent effect. Reuse the
    # Phase 4.10 nested discovery -> inner-selection -> FDR gate, then build
    # the Phase 5 probabilities from only the relationships that survived it.
    if prediction_engine is None and bool(getattr(config, "hardened_validation", True)):
        _phase5_progress(
            getattr(config, "progress_logging", True),
            "Nested selection start | exhaustive=%s | selection_fraction=%.2f | alpha=%.2f",
            not bool(getattr(config, "adaptive_relationship_search", False)),
            float(getattr(config, "selection_fraction", 0.30)),
            float(getattr(config, "multiple_testing_alpha", 0.10)),
        )
        from .relationship import RelationshipDiscoveryEngine
        selector_engine = RelationshipDiscoveryEngine(
            max_order=3,
            min_observations=config.min_training_observations,
            adaptive_higher_order=bool(getattr(config, "adaptive_relationship_search", False)),
            progress_callback=lambda message: _phase5_progress(getattr(config, "progress_logging", True), message),
            progress_every_candidates=int(getattr(config, "relationship_progress_every_candidates", 1000)),
            candidate_batch_size=int(getattr(config, "relationship_candidate_batch_size", 512)),
            retain_supporting_data=True,
        )
        nested = validate_walk_forward_relationships_hardened(
            observations,
            selector_engine,
            min_training_observations=config.min_training_observations,
            selection_fraction=float(getattr(config, "selection_fraction", 0.30)),
            multiple_testing_alpha=float(getattr(config, "multiple_testing_alpha", 0.10)),
            purge_overlapping_labels=True,
        )
        prediction_engine = PredictionEngine(
            relationship_engine=selector_engine,
            min_threshold_observations=config.min_training_observations,
        )
        folds: list[PredictionFoldResult] = []
        skipped_threshold_limited = 0
        skipped_provenance_failed = 0
        skipped_invalid_actual = 0
        conviction_counts = {key: 0 for key in ("STRONG", "MODERATE", "LOW", "NONE")}

        nested_by_date = {fold.prediction_date: fold for fold in nested.folds}
        for fold_index, observation in enumerate(observations, start=1):
            cutoff = _as_date(observation.as_of_date)
            if fold_index == 1 or fold_index % max(1, getattr(config, "progress_every", 10)) == 0 or fold_index == candidate_predictions:
                _phase5_progress(
                    getattr(config, "progress_logging", True),
                    "Nested fold %d/%d | prediction_date=%s",
                    fold_index, candidate_predictions, cutoff,
                )
            selected_fold = nested_by_date.get(cutoff)
            if selected_fold is None:
                continue
            selected_a = getattr(selected_fold, "selected_a_result", None)
            selected_b = getattr(selected_fold, "selected_b_result", None)
            ranking_a = rank_relationships([selected_a])[0] if selected_a is not None else None
            ranking_b = rank_relationships([selected_b])[0] if selected_b is not None else None
            result = prediction_engine.predict(
                target=observation.target,
                current_states=dict(observation.states),
                observations=observations,
                prediction_date=cutoff,
                analysis_timeframe=config.analysis_timeframe,
                holding_period_months=config.holding_period_months,
                benchmark=config.benchmark,
                entry_mode=config.entry_mode,
                validated_evidence=False,
                validation_evidence=None,
                selected_relationships={"A": ranking_a, "B": ranking_b},
            )
            thresholds = result.outcome_thresholds
            if thresholds.limited:
                skipped_threshold_limited += 1
                continue
            audit = result.provenance_audit
            provenance_clean = bool(audit and audit.clean)
            if not provenance_clean:
                skipped_provenance_failed += 1
            actual_numeric = observation.stock_return_pct
            if actual_numeric is None:
                skipped_invalid_actual += 1
                continue
            actual_numeric = float(actual_numeric)
            actual_class = classify_return(actual_numeric, thresholds)
            hit = _directional_hit(result.trend, actual_class)
            conviction_counts[result.conviction] = conviction_counts.get(result.conviction, 0) + 1
            folds.append(
                PredictionFoldResult(
                    prediction_date=cutoff,
                    actual_return_pct=actual_numeric,
                    actual_class=actual_class,
                    predicted_trend=result.trend,
                    conviction=result.conviction,
                    probabilities_pct=dict(result.probabilities_pct),
                    baseline_probabilities_pct=dict(result.baseline_probabilities_pct),
                    expected_return_pct=result.expected_return_pct,
                    training_observations=result.training_observations,
                    method_a_trend=result.method_a.trend if result.method_a else None,
                    method_b_trend=result.method_b.trend if result.method_b else None,
                    method_agreement=result.method_agreement,
                    limited=result.limited,
                    provenance_clean=provenance_clean,
                    hit=hit,
                )
            )

        method_a = _summary("A", folds, lambda fold: fold.method_a_trend)
        method_b = _summary("B", folds, lambda fold: fold.method_b_trend)
        combined = _summary("COMBINED", folds, lambda fold: fold.predicted_trend)
        latest_fold = folds[-1] if folds else None
        result = Phase5RealOLAPValidationResult(
            ticker=config.ticker, benchmark=config.benchmark,
            analysis_timeframe=config.analysis_timeframe,
            holding_period_months=config.holding_period_months,
            entry_mode=config.entry_mode,
            candidate_predictions=candidate_predictions,
            evaluated_predictions=len(folds),
            skipped_threshold_limited=skipped_threshold_limited,
            skipped_provenance_failed=skipped_provenance_failed,
            skipped_invalid_actual=skipped_invalid_actual,
            prediction_folds=tuple(folds), method_a=method_a, method_b=method_b, combined=combined,
            baseline_majority_accuracy_pct=_majority_baseline_accuracy(folds),
            mean_combined_probabilities_pct=_mean_probabilities(folds),
            mean_combined_expected_return_pct=_mean_expected_return(folds),
            conviction_counts=conviction_counts,
            latest_validated_prediction_date=latest_fold.prediction_date if latest_fold else None,
            latest_validated_prediction=latest_fold.as_dict() if latest_fold else None,
            latest_market_date=latest_market_date,
            phase4_surface_audit=panel_stats.get("state_surface_audit"),
            selection_candidate_evaluations=nested.selection_candidate_evaluations,
            selection_validated_predictions=nested.selection_validated_predictions,
            multiple_testing_controlled_folds=nested.multiple_testing_controlled_folds,
        )
        return result
    if prediction_engine is None:
        progress_every_candidates = max(1, int(getattr(config, "relationship_progress_every_candidates", 500)))

        def _relationship_progress(message: str) -> None:
            _phase5_progress(getattr(config, "progress_logging", True), message)

        from .relationship import RelationshipDiscoveryEngine
        relationship_engine = RelationshipDiscoveryEngine(
            max_order=3,
            min_observations=config.min_training_observations,
            adaptive_higher_order=bool(getattr(config, "adaptive_relationship_search", True)),
            progress_callback=_relationship_progress,
            progress_every_candidates=progress_every_candidates,
            candidate_batch_size=int(getattr(config, "relationship_candidate_batch_size", 512)),
            retain_supporting_data=False,
        )
        prediction_engine = PredictionEngine(
            relationship_engine=relationship_engine,
            min_threshold_observations=config.min_training_observations,
        )

    _phase5_progress(
        getattr(config, "progress_logging", True),
        "Prediction validation start | folds=%d | exhaustive_search=YES | batch=%d",
        candidate_predictions, int(getattr(config, "relationship_candidate_batch_size", 512)),
    )

    folds: list[PredictionFoldResult] = []
    skipped_threshold_limited = 0
    skipped_provenance_failed = 0
    skipped_invalid_actual = 0
    conviction_counts = {key: 0 for key in ("STRONG", "MODERATE", "LOW", "NONE")}

    for fold_index, observation in enumerate(observations, start=1):
        cutoff = _as_date(observation.as_of_date)
        actual_return = observation.stock_return_pct
        if actual_return is None:
            skipped_invalid_actual += 1
            continue
        try:
            actual_numeric = float(actual_return)
        except (TypeError, ValueError):
            skipped_invalid_actual += 1
            continue
        if not isfinite(actual_numeric):
            skipped_invalid_actual += 1
            continue

        fold_started = time.perf_counter()
        if fold_index == 1 or fold_index % max(1, getattr(config, "progress_every", 10)) == 0 or fold_index == candidate_predictions:
            _phase5_progress(
                getattr(config, "progress_logging", True),
                "Fold start %d/%d | prediction_date=%s | prior_completed=%d",
                fold_index, candidate_predictions, cutoff, len(folds),
            )

        result = prediction_engine.predict(
            target=observation.target,
            current_states=dict(observation.states),
            observations=observations,
            prediction_date=cutoff,
            analysis_timeframe=config.analysis_timeframe,
            holding_period_months=config.holding_period_months,
            benchmark=config.benchmark,
            entry_mode=config.entry_mode,
            validated_evidence=False,
            validation_evidence=None,
        )

        _assert_fold_temporal_safety(result, cutoff, observations)
        thresholds = result.outcome_thresholds
        if thresholds.limited:
            skipped_threshold_limited += 1
            continue

        audit = result.provenance_audit
        provenance_clean = bool(audit and audit.clean)
        if not provenance_clean:
            skipped_provenance_failed += 1

        actual_class = classify_return(actual_numeric, thresholds)
        hit = _directional_hit(result.trend, actual_class)
        conviction_counts[result.conviction] = conviction_counts.get(result.conviction, 0) + 1

        folds.append(
            PredictionFoldResult(
                prediction_date=cutoff,
                actual_return_pct=actual_numeric,
                actual_class=actual_class,
                predicted_trend=result.trend,
                conviction=result.conviction,
                probabilities_pct=dict(result.probabilities_pct),
                baseline_probabilities_pct=dict(result.baseline_probabilities_pct),
                expected_return_pct=result.expected_return_pct,
                training_observations=result.training_observations,
                method_a_trend=result.method_a.trend if result.method_a else None,
                method_b_trend=result.method_b.trend if result.method_b else None,
                method_agreement=result.method_agreement,
                limited=result.limited,
                provenance_clean=provenance_clean,
                hit=hit,
            )
        )


        if fold_index == 1 or fold_index % max(1, getattr(config, "progress_every", 10)) == 0 or fold_index == candidate_predictions:
            _phase5_progress(
                getattr(config, "progress_logging", True),
                "Fold complete %d/%d | elapsed=%.2fs | predicted=%s | A=%s B=%s | training=%d | candidates=%d",
                fold_index, candidate_predictions, time.perf_counter() - fold_started, result.trend,
                result.method_a.trend if result.method_a else None,
                result.method_b.trend if result.method_b else None,
                result.training_observations,
                (result.provenance_audit.relationship_candidate_count if result.provenance_audit else 0),
            )

    method_a = _summary("A", folds, lambda fold: fold.method_a_trend)
    method_b = _summary("B", folds, lambda fold: fold.method_b_trend)
    combined = _summary("COMBINED", folds, lambda fold: fold.predicted_trend)

    latest_fold = folds[-1] if folds else None
    return Phase5RealOLAPValidationResult(
        ticker=config.ticker,
        benchmark=config.benchmark,
        analysis_timeframe=config.analysis_timeframe,
        holding_period_months=config.holding_period_months,
        entry_mode=config.entry_mode,
        candidate_predictions=candidate_predictions,
        evaluated_predictions=len(folds),
        skipped_threshold_limited=skipped_threshold_limited,
        skipped_provenance_failed=skipped_provenance_failed,
        skipped_invalid_actual=skipped_invalid_actual,
        prediction_folds=tuple(folds),
        method_a=method_a,
        method_b=method_b,
        combined=combined,
        baseline_majority_accuracy_pct=_majority_baseline_accuracy(folds),
        mean_combined_probabilities_pct=_mean_probabilities(folds),
        mean_combined_expected_return_pct=_mean_expected_return(folds),
        conviction_counts=conviction_counts,
        latest_validated_prediction_date=latest_fold.prediction_date if latest_fold else None,
        latest_validated_prediction=latest_fold.as_dict() if latest_fold else None,
        latest_market_date=latest_market_date,
        phase4_surface_audit=panel_stats.get("state_surface_audit"),
    )


def print_real_olap_prediction_report(result: Phase5RealOLAPValidationResult) -> None:
    print("PHASE 5.8 REAL OLAP PREDICTION VALIDATION")
    print("Ticker:", result.ticker)
    print("Benchmark:", result.benchmark)
    print("Analysis timeframe:", result.analysis_timeframe)
    print("Holding period (months):", result.holding_period_months)
    print("Entry mode:", result.entry_mode)
    print("Candidate predictions:", result.candidate_predictions)
    print("Evaluated predictions:", result.evaluated_predictions)
    print("Skipped threshold-limited:", result.skipped_threshold_limited)
    print("Skipped provenance-failed:", result.skipped_provenance_failed)
    print("Skipped invalid actual:", result.skipped_invalid_actual)
    print()
    for summary in (result.method_a, result.method_b, result.combined):
        print(f"Method {summary.method}:")
        print("  directional predictions:", summary.directional_predictions)
        print("  directional hits:", summary.directional_hits)
        print("  directional hit rate:", round(summary.directional_hit_rate_pct, 2))
        print("  clear-trend predictions:", summary.clear_trend_predictions)
        print("  clear-trend accuracy:", round(summary.clear_trend_accuracy_pct, 2))
        print("  NO_CLEAR_TREND:", summary.no_clear_trend_predictions)
        print("  limited:", summary.limited_predictions)
        print("  coverage:", round(summary.coverage_pct, 2))
    print()
    print("Baseline majority accuracy:", round(result.baseline_majority_accuracy_pct, 2))
    print("Mean combined probabilities:", {k: round(v, 2) for k, v in result.mean_combined_probabilities_pct.items()})
    print("Mean combined expected return:", None if result.mean_combined_expected_return_pct is None else round(result.mean_combined_expected_return_pct, 4))
    print("Conviction counts:", result.conviction_counts)
    print("Selection candidate evaluations:", result.selection_candidate_evaluations)
    print("Selection-validated predictions:", result.selection_validated_predictions)
    print("Multiple-testing-controlled folds:", result.multiple_testing_controlled_folds)
    print("Latest validated prediction date:", result.latest_validated_prediction_date)
    print("Latest market date:", result.latest_market_date)
    print("NOTE: Phase 5.8 is diagnostic. Probability calibration and full OOS performance remain Phase 6.")
