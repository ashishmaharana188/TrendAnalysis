from __future__ import annotations

from datetime import date

import pytest

from analysis.phase5_snapshot import (
    FOLD_RECORDING_CONTRACT_VERSION,
    validate_fold_recording_contract,
)
from analysis.walk_forward_relationship import WalkForwardFold


def complete_row() -> dict:
    method_a = {
        "probabilities_pct": {"UP": 55.0, "SIDEWAYS": 25.0, "DOWN": 20.0},
        "evidence_score": 0.75,
        "effective_sample_size": 18.0,
        "sample_count": 22,
        "variables": ["company.market.price"],
        "condition": ["company.market.price=Rising"],
        "limited": False,
    }
    method_b = {
        "probabilities_pct": {"UP": 20.0, "SIDEWAYS": 25.0, "DOWN": 55.0},
        "evidence_score": -0.5,
        "effective_sample_size": 12.0,
        "sample_count": 15,
        "variables": ["industry.market"],
        "condition": ["industry.market=Falling"],
        "limited": False,
    }
    selection_a = {"selection_status": "SELECTED_AFTER_ADJUSTED_P_THRESHOLD"}
    selection_b = {"selection_status": "REJECTED_NO_CANDIDATE_PASSED_ADJUSTED_P_THRESHOLD"}
    return {
        "method_a_probabilities_pct": dict(method_a["probabilities_pct"]),
        "method_b_probabilities_pct": dict(method_b["probabilities_pct"]),
        "method_a_output": method_a,
        "method_b_output": method_b,
        "method_a_selection_metadata": selection_a,
        "method_b_selection_metadata": selection_b,
        "selection_metadata": {"recording_contract_version": FOLD_RECORDING_CONTRACT_VERSION},
        "outcome_thresholds": {"limited": False},
        "prediction_provenance": {"clean": True},
        "prediction_result": {"trend": "UP"},
        "current_states": {"company.market.price": "Rising / High"},
        "benchmark_return_pct": 0.4,
        "relative_return_pct": 1.1,
        "outcome_end_date": "2026-09-24",
        "fold_recording_contract_version": FOLD_RECORDING_CONTRACT_VERSION,
    }


def test_contract_counts_exact_method_outputs_not_combined_probabilities() -> None:
    record = validate_fold_recording_contract([complete_row()])
    assert record["validated"] is True
    assert record["method_a_outputs_recorded"] == 1
    assert record["method_b_outputs_recorded"] == 1
    assert "never inferred from combined probabilities" in record["probability_source"]


def test_contract_rejects_missing_individual_probability_vector() -> None:
    row = complete_row()
    del row["method_b_probabilities_pct"]
    with pytest.raises(ValueError, match="missing fields.*method_b_probabilities_pct"):
        validate_fold_recording_contract([row])


def test_contract_rejects_vector_that_differs_from_method_output() -> None:
    row = complete_row()
    row["method_a_probabilities_pct"] = {"UP": 1.0, "SIDEWAYS": 2.0, "DOWN": 97.0}
    with pytest.raises(ValueError, match="differs from the exact method output vector"):
        validate_fold_recording_contract([row])


def test_contract_rejects_non_normalized_method_vector() -> None:
    row = complete_row()
    row["method_a_probabilities_pct"] = {"UP": 10.0, "SIDEWAYS": 10.0, "DOWN": 10.0}
    row["method_a_output"]["probabilities_pct"] = dict(row["method_a_probabilities_pct"])
    with pytest.raises(ValueError, match="sums to 30.000000"):
        validate_fold_recording_contract([row])


def test_walk_forward_fold_exports_gate_metadata_per_method() -> None:
    gate_a = {
        "method": "A",
        "selection_status": "SELECTED_AFTER_ADJUSTED_P_THRESHOLD",
        "candidate_evaluation_count": 120,
        "multiple_testing": {
            "procedure": "search_adjusted_max_statistic_permutation",
            "permutations_run": 199,
            "alpha": 0.1,
            "accepted_candidate_count": 2,
            "selected_adjusted_p_value": 0.04,
        },
    }
    gate_b = {
        "method": "B",
        "selection_status": "REJECTED_NO_CANDIDATE_PASSED_ADJUSTED_P_THRESHOLD",
        "candidate_evaluation_count": 96,
        "multiple_testing": {
            "procedure": "search_adjusted_max_statistic_permutation",
            "permutations_run": 199,
            "alpha": 0.1,
            "accepted_candidate_count": 0,
        },
    }
    fold = WalkForwardFold(
        prediction_date=date(2026, 8, 24),
        training_observations=150,
        actual_return_pct=1.5,
        method_a={"direction": "POSITIVE", "hit": True},
        method_b=None,
        combined=None,
        method_a_selection_metadata=gate_a,
        method_b_selection_metadata=gate_b,
    ).as_dict()
    assert fold["method_a_selection_metadata"]["multiple_testing"]["selected_adjusted_p_value"] == 0.04
    assert fold["method_b_selection_metadata"]["multiple_testing"]["accepted_candidate_count"] == 0
