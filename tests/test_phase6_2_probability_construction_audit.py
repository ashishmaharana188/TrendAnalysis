from __future__ import annotations

import pytest

from analysis.phase6_2_probability_construction import audit_probability_construction


def _row(
    *,
    actual: str = "UP",
    combined: dict[str, float] | None = None,
    method_a: dict[str, float] | None = None,
    method_b: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> dict:
    a = method_a if method_a is not None else {"UP": 100.0, "SIDEWAYS": 0.0, "DOWN": 0.0}
    b = method_b if method_b is not None else {"UP": 0.0, "SIDEWAYS": 50.0, "DOWN": 50.0}
    weights = weights if weights is not None else {"A": 2.0, "B": 1.0}
    rebuilt = {
        label: (a[label] * weights.get("A", 0.0) + b[label] * weights.get("B", 0.0))
        / max(weights.get("A", 0.0) + weights.get("B", 0.0), 1e-12)
        for label in ("UP", "SIDEWAYS", "DOWN")
    }
    return {
        "prediction_date": "2026-01-01",
        "actual_class": actual,
        "probabilities_pct": combined if combined is not None else rebuilt,
        "method_a_probabilities_pct": dict(a),
        "method_b_probabilities_pct": dict(b),
        "method_a_output": {
            "probabilities_pct": dict(a),
            "sample_count": 12,
            "effective_sample_size": 12.0,
            "limited": False,
        },
        "method_b_output": {
            "probabilities_pct": dict(b),
            "sample_count": 30,
            "effective_sample_size": 14.0,
            "limited": False,
        },
        "method_a_selection_metadata": {
            "selection_status": "SELECTED_AFTER_ADJUSTED_P_THRESHOLD",
            "selected_relationship": {"supporting_observation_count": 12},
        },
        "method_b_selection_metadata": {
            "selection_status": "SELECTED_AFTER_ADJUSTED_P_THRESHOLD",
            "selected_relationship": {"supporting_observation_count": 30},
        },
        "prediction_result": {
            "probability_basis": "empirical_conditional_class_share_uncalibrated",
            "method_a": {"probabilities_pct": dict(a)},
            "method_b": {"probabilities_pct": dict(b)},
            "combination": {"method_weights": dict(weights)},
        },
    }


def test_probability_audit_counts_true_class_zeros_without_fabricating_counts() -> None:
    row = _row(actual="SIDEWAYS")
    report = audit_probability_construction([row])
    method_a = report["methods"]["Method A"]

    assert method_a["availability_status"] == "COMPLETE"
    assert method_a["zero_audit"]["true_class_exact_zero_rows"] == 1
    assert method_a["source_count_trace_status"] == "MISSING_FROM_CURRENT_FROZEN_ARTIFACT"
    assert method_a["class_counts_trace_rows"] == 0
    assert "weighted_class_counts" in report["source_contract"]["required_method_builder_fields"]


def test_auditor_recomputes_combined_probability_from_recorded_weights() -> None:
    report = audit_probability_construction([_row()])
    comparison = report["combined_reconstruction"]

    assert comparison["status"] == "CHECKED"
    assert comparison["folds_checked"] == 1
    assert comparison["folds_mismatched"] == 0
    assert comparison["max_abs_difference_pct_points"] == pytest.approx(0.0)


def test_combined_probability_reconstruction_detects_mismatch() -> None:
    row = _row(combined={"UP": 10.0, "SIDEWAYS": 45.0, "DOWN": 45.0})
    report = audit_probability_construction([row])

    assert report["combined_reconstruction"]["status"] == "MISMATCHES_FOUND"
    assert report["combined_reconstruction"]["folds_mismatched"] == 1
    assert report["combined_reconstruction"]["mismatch_examples"]


def test_unavailable_method_remains_partial_and_is_not_inferred_from_trend() -> None:
    row = _row()
    row["method_b_probabilities_pct"] = {}
    row["method_b_output"] = None
    row["prediction_result"]["method_b"] = None
    row["method_b_selection_metadata"] = {"selection_status": "NO_RELATIONSHIP_PASSED_SELECTION_GATE"}

    report = audit_probability_construction([row])
    method_b = report["methods"]["Method B"]

    assert method_b["availability_status"] == "UNAVAILABLE"
    assert method_b["zero_audit"]["available_rows"] == 0
    assert report["combined_reconstruction"]["folds_checked"] == 0


def test_builder_count_metadata_is_checked_when_present() -> None:
    row = _row(actual="SIDEWAYS")
    row["method_a_output"]["class_counts"] = {"UP": 12, "SIDEWAYS": 0, "DOWN": 0}
    row["method_a_output"]["weighted_class_counts"] = {"UP": 12.0, "SIDEWAYS": 0.0, "DOWN": 0.0}
    row["method_a_output"]["probability_basis"] = "method_a_empirical_conditional_class_share"

    report = audit_probability_construction([row])
    method_a = report["methods"]["Method A"]
    assert method_a["class_counts_trace_rows"] == 1
    assert method_a["weighted_class_counts_trace_rows"] == 1
    assert method_a["probability_basis_trace_rows"] == 1
    assert method_a["source_count_trace_status"] == "RECORDED"
