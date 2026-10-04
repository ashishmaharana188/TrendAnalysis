from datetime import date, timedelta

from analysis.outcome_labels import OutcomeThresholds
from analysis.prediction import PredictionResult
from analysis.prediction_hardening import PredictionProvenanceAudit
from analysis.real_prediction_validation import validate_real_olap_predictions
from analysis.relationship import HistoricalRelationshipObservation
from analysis.decision import DecisionResult


def _row(day, ret, end_day, states=None):
    return HistoricalRelationshipObservation(
        as_of_date=day,
        target="TEST",
        scope="company",
        states=states or {"company.market.price": "Rising"},
        stock_return_pct=ret,
        benchmark_return_pct=0.0,
        relative_return_pct=ret,
        outcome_end_date=end_day,
    )


class FakeEngine:
    def __init__(self, future_threshold=False):
        self.future_threshold = future_threshold

    def predict(self, *, target, current_states, observations, prediction_date, **kwargs):
        cutoff = prediction_date
        threshold_end = cutoff + timedelta(days=1) if self.future_threshold else cutoff - timedelta(days=1)
        thresholds = OutcomeThresholds(
            lower_pct=-1.0,
            upper_pct=1.0,
            sample_count=3,
            limited=False,
        )
        audit = PredictionProvenanceAudit(
            target=target,
            cutoff_date=cutoff,
            source_count=len(list(observations)),
            training_count=3,
            latest_training_state_date=cutoff - timedelta(days=3),
            latest_training_outcome_end_date=threshold_end,
            excluded_target=0,
            excluded_future_state=0,
            excluded_incomplete_outcome=0,
            excluded_unknown_outcome_end=0,
            excluded_missing_return=0,
            excluded_non_finite_return=0,
            threshold_sample_count=3,
            threshold_fit_end_date=threshold_end,
            threshold_temporal_violation=self.future_threshold,
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
        limited = self.future_threshold
        trend = "NO_CLEAR_TREND" if limited else ("UP" if current_states.get("company.market.price") == "Rising" else "DOWN")
        return PredictionResult(
            target=target,
            prediction_date=cutoff,
            analysis_timeframe="1M",
            holding_period_months=1.0,
            benchmark="Nifty_50",
            entry_mode="next_trading_day",
            training_observations=3,
            outcome_thresholds=thresholds,
            baseline_probabilities_pct={"UP": 33.33, "SIDEWAYS": 33.34, "DOWN": 33.33},
            method_a=None,
            method_b=None,
            trend=trend,
            conviction="LOW" if not limited else "NONE",
            probabilities_pct={"UP": 60.0, "SIDEWAYS": 20.0, "DOWN": 20.0},
            expected_return_pct=2.0,
            method_agreement=False,
            limited=limited,
            limitations=(),
            validated_evidence=False,
            decision=DecisionResult(
                trend=trend,
                selected_class=trend if trend in {"UP", "SIDEWAYS", "DOWN"} else None,
                probability_pct=60.0,
                baseline_probability_pct=33.33,
                lift_pct=26.67,
                margin_pct=40.0,
                uncertainty_pct=5.0,
                effective_sample_size=10.0,
                reason="test",
                limited=limited,
            ),
            combination=None,
            conviction_result=None,
            provenance_audit=audit,
        )


def test_real_olap_phase5_runs_predictions_without_using_fold_outcome_as_training():
    rows = [
        _row(date(2025, 1, 1) + timedelta(days=i * 40), 2.0 if i % 2 == 0 else -2.0,
             date(2025, 1, 2) + timedelta(days=i * 40))
        for i in range(6)
    ]

    class Config:
        ticker = "TEST"
        benchmark = "Nifty_50"
        analysis_timeframe = "1M"
        holding_period_months = 1.0
        entry_mode = "next_trading_day"
        min_training_observations = 3

    result = validate_real_olap_predictions(
        Config(),
        panel_builder=lambda cfg: (rows, {}, rows[-1].as_of_date),
        prediction_engine=FakeEngine(),
    )

    assert result.candidate_predictions == 6
    assert result.evaluated_predictions == 6
    assert result.combined.directional_predictions == 6
    assert result.combined.directional_hits == 3
    assert result.latest_validated_prediction_date == rows[-1].as_of_date


def test_real_olap_prediction_validation_fails_closed_on_future_threshold_provenance():
    rows = [
        _row(date(2025, 1, 1) + timedelta(days=i * 40), 2.0, date(2025, 1, 2) + timedelta(days=i * 40))
        for i in range(4)
    ]

    class Config:
        ticker = "TEST"
        benchmark = "Nifty_50"
        analysis_timeframe = "1M"
        holding_period_months = 1.0
        entry_mode = "next_trading_day"
        min_training_observations = 3

    result = validate_real_olap_predictions(
        Config(),
        panel_builder=lambda cfg: (rows, {}, rows[-1].as_of_date),
        prediction_engine=FakeEngine(future_threshold=True),
    )

    assert result.skipped_provenance_failed == 4
    assert all(fold.limited for fold in result.prediction_folds)
    assert result.combined.directional_predictions == 0


def test_majority_baseline_is_reported_separately_from_model_accuracy():
    rows = [
        _row(date(2025, 1, 1) + timedelta(days=i * 40), 0.0, date(2025, 1, 2) + timedelta(days=i * 40))
        for i in range(3)
    ]

    class Config:
        ticker = "TEST"
        benchmark = "Nifty_50"
        analysis_timeframe = "1M"
        holding_period_months = 1.0
        entry_mode = "next_trading_day"
        min_training_observations = 3

    result = validate_real_olap_predictions(
        Config(),
        panel_builder=lambda cfg: (rows, {}, rows[-1].as_of_date),
        prediction_engine=FakeEngine(),
    )

    assert 0.0 <= result.baseline_majority_accuracy_pct <= 100.0
