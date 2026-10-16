from __future__ import annotations

from phase6_3_snapshot_tools import _clean_for_comparison


def test_comparison_ignores_trace_and_contract_version_metadata_only() -> None:
    legacy = {
        "fold_recording_contract_version": 1,
        "selection_metadata": {"recording_contract_version": 1, "selection_mode": "nested"},
        "method_a_output": {
            "probabilities_pct": {"UP": 50.0, "SIDEWAYS": 30.0, "DOWN": 20.0},
            "class_counts": {"UP": 5, "SIDEWAYS": 3, "DOWN": 2},
            "weighted_class_counts": {"UP": 5.0, "SIDEWAYS": 3.0, "DOWN": 2.0},
        },
    }
    v2 = {
        "fold_recording_contract_version": 2,
        "selection_metadata": {"recording_contract_version": 2, "selection_mode": "nested"},
        "method_a_output": {
            "probabilities_pct": {"UP": 50.0, "SIDEWAYS": 30.0, "DOWN": 20.0},
            "class_counts": {"UP": 5, "SIDEWAYS": 3, "DOWN": 2},
            "weighted_class_counts": {"UP": 5.0, "SIDEWAYS": 3.0, "DOWN": 2.0},
            "probability_basis": "empirical", "exact_condition_count": 2,
            "weight_concentration": 0.5, "stability_score": 0.8,
        },
    }
    assert _clean_for_comparison(legacy) == _clean_for_comparison(v2)


def test_comparison_still_retains_probability_behavior_changes() -> None:
    before = {"method_a_output": {"probabilities_pct": {"UP": 50.0}, "class_counts": {"UP": 5}}}
    after = {"method_a_output": {"probabilities_pct": {"UP": 51.0}, "class_counts": {"UP": 6}}}
    assert _clean_for_comparison(before) != _clean_for_comparison(after)
