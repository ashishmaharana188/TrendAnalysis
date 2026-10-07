from dataclasses import dataclass
from datetime import date

from analysis.trend_evidence import build_trend_evidence_report


@dataclass
class Fold:
    prediction_date: date
    actual_class: str
    predicted_trend: str
    method_a_trend: str | None
    method_b_trend: str | None
    limited: bool = False
    provenance_clean: bool = True
    probabilities_pct: dict[str, float] | None = None
    conviction: str = "NONE"



def test_trend_evidence_uses_matched_clear_subset_baseline() -> None:
    folds = (
        Fold(date(2026, 1, 1), "UP", "UP", "UP", "UP"),
        Fold(date(2026, 1, 2), "UP", "DOWN", "DOWN", "DOWN"),
        Fold(date(2026, 1, 3), "SIDEWAYS", "UP", "UP", "SIDEWAYS"),
        Fold(date(2026, 1, 4), "DOWN", "NO_CLEAR_TREND", "NO_CLEAR_TREND", None),
    )

    class Result:
        prediction_folds = folds

    report = build_trend_evidence_report(Result(), recent_window=3)

    assert report.combined.clear_trend_folds == 3
    assert abs(report.combined.clear_trend_accuracy_pct - (100.0 / 3.0)) < 1e-9
    # Among the same three clear-trend folds, UP is the majority actual class.
    assert abs(report.combined.matched_majority_baseline_pct - (200.0 / 3.0)) < 1e-9
    assert report.combined.accuracy_lift_vs_matched_baseline_pct < 0.0


def test_trend_evidence_reports_method_agreement() -> None:
    folds = (
        Fold(date(2026, 1, 1), "UP", "UP", "UP", "UP"),
        Fold(date(2026, 1, 2), "DOWN", "NO_CLEAR_TREND", "UP", "DOWN"),
        Fold(date(2026, 1, 3), "UP", "UP", "UP", "UP"),
    )

    class Result:
        prediction_folds = folds

    report = build_trend_evidence_report(Result(), recent_window=20)

    assert report.both_methods_clear_folds == 3
    assert report.method_agreement_folds == 2
    assert report.method_conflict_folds == 1
    assert abs(report.method_agreement_rate_pct - (200.0 / 3.0)) < 1e-9
