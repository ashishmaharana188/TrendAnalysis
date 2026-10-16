from __future__ import annotations

import copy

import pytest

from analysis.phase5_snapshot import validate_fold_recording_contract


OUTCOMES = ("UP", "SIDEWAYS", "DOWN")


def _usable_v2_row() -> dict:
    method_a = {
        "probabilities_pct": {"UP": 50.0, "SIDEWAYS": 25.0, "DOWN": 25.0},
        "sample_count": 4,
        "effective_sample_size": 4.0,
        "class_counts": {"UP": 2, "SIDEWAYS": 1, "DOWN": 1},
        "weighted_class_counts": {"UP": 2.0, "SIDEWAYS": 1.0, "DOWN": 1.0},
        "probability_basis": "method_a_empirical_conditional_class_share",
        "limited": False,
    }
    method_b = {
        "probabilities_pct": {"UP": 20.0, "SIDEWAYS": 30.0, "DOWN": 50.0},
        "sample_count": 4,
        "effective_sample_size": 3.5,
        "class_counts": {"UP": 1, "SIDEWAYS": 1, "DOWN": 2},
        "weighted_class_counts": {"UP": 2.0, "SIDEWAYS": 3.0, "DOWN": 5.0},
        "probability_basis": "method_b_weighted_empirical_conditional_class_share",
        "exact_condition_count": 2,
        "weight_concentration": 0.5,
        "stability_score": 0.75,
        "limited": False,
    }
    return {
        "method_a_probabilities_pct": dict(method_a["probabilities_pct"]),
        "method_b_probabilities_pct": dict(method_b["probabilities_pct"]),
        "method_a_output": method_a,
        "method_b_output": method_b,
        "method_a_selection_metadata": {"selection_status": "SELECTED"},
        "method_b_selection_metadata": {"selection_status": "SELECTED"},
        "selection_metadata": {"recording_contract_version": 2},
        "outcome_thresholds": {"limited": False},
        "prediction_provenance": {"clean": True},
        "prediction_result": {"trend": "UP"},
        "current_states": {"company.market.price": "Rising"},
        "benchmark_return_pct": 0.5,
        "relative_return_pct": 0.8,
        "outcome_end_date": "2026-02-05",
        "fold_recording_contract_version": 2,
    }


def test_v2_contract_accepts_complete_method_b_diagnostics() -> None:
    row = _usable_v2_row()
    report = validate_fold_recording_contract([row])
    assert report["validated"] is True
    assert report["version"] == 2


@pytest.mark.parametrize(
    "field",
    ("exact_condition_count", "weight_concentration", "stability_score"),
)
def test_v2_contract_rejects_missing_method_b_diagnostics(field: str) -> None:
    row = copy.deepcopy(_usable_v2_row())
    del row["method_b_output"][field]
    with pytest.raises(ValueError, match="missing required diagnostics"):
        validate_fold_recording_contract([row])


def test_v2_contract_rejects_missing_exact_condition_count_for_usable_method_b() -> None:
    row = _usable_v2_row()
    row["method_b_output"]["exact_condition_count"] = None
    with pytest.raises(ValueError, match="requires exact_condition_count"):
        validate_fold_recording_contract([row])


def test_v1_contract_remains_compatible_without_method_b_diagnostics() -> None:
    row = _usable_v2_row()
    row["fold_recording_contract_version"] = 1
    row["selection_metadata"]["recording_contract_version"] = 1
    for field in ("exact_condition_count", "weight_concentration", "stability_score"):
        row["method_b_output"].pop(field)
    report = validate_fold_recording_contract([row], contract_version=1)
    assert report["validated"] is True
    assert report["version"] == 1


def test_limited_method_b_may_have_null_exact_condition_count_but_remains_limited() -> None:
    row = _usable_v2_row()
    limited_b = row["method_b_output"]
    limited_b.update({
        "probabilities_pct": {"UP": 0.0, "SIDEWAYS": 0.0, "DOWN": 0.0},
        "sample_count": 0,
        "effective_sample_size": 0.0,
        "class_counts": {"UP": 0, "SIDEWAYS": 0, "DOWN": 0},
        "weighted_class_counts": {"UP": 0.0, "SIDEWAYS": 0.0, "DOWN": 0.0},
        "exact_condition_count": None,
        "weight_concentration": 0.0,
        "stability_score": 0.0,
        "limited": True,
    })
    row["method_b_probabilities_pct"] = dict(limited_b["probabilities_pct"])
    assert validate_fold_recording_contract([row])["validated"] is True
