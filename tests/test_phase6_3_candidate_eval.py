from __future__ import annotations

import pytest

from phase6_3_candidate_eval import _mature_prior, _smooth, _source_effective_counts


def test_zero_smoothing_reconstructs_legacy_empirical_distribution() -> None:
    method_a_output = {
        "sample_count": 12,
        "effective_sample_size": 12.0,
        "class_counts": {"UP": 4, "SIDEWAYS": 5, "DOWN": 3},
        "weighted_class_counts": {"UP": 4.0, "SIDEWAYS": 5.0, "DOWN": 3.0},
    }
    source = _source_effective_counts(method_a_output)
    assert source is not None
    counts, total = source
    probabilities = _smooth(counts, total, 0.0, None)
    assert probabilities == {
        "UP": 4 / 12,
        "SIDEWAYS": 5 / 12,
        "DOWN": 3 / 12,
    }
    assert abs(sum(probabilities.values()) - 1.0) < 1e-12


def test_positive_smoothing_removes_exact_zero_and_normalizes() -> None:
    counts = {"UP": 0.0, "SIDEWAYS": 6.0, "DOWN": 4.0}
    probabilities = _smooth(counts, 10.0, 0.5, None)
    assert probabilities["UP"] > 0.0
    assert abs(sum(probabilities.values()) - 1.0) < 1e-12


def test_mature_prior_excludes_immature_and_same_day_outcomes() -> None:
    previous = [
        {"actual_class": "UP", "outcome_end_date": "2026-01-09"},
        {"actual_class": "DOWN", "outcome_end_date": "2026-01-10"},
        {"actual_class": "SIDEWAYS", "outcome_end_date": "2026-01-20"},
    ]
    prior = _mature_prior(previous, "2026-01-10")
    assert prior == {"UP": 0.5, "SIDEWAYS": 0.25, "DOWN": 0.25}
    assert abs(sum(prior.values()) - 1.0) < 1e-12


def test_weighted_method_b_counts_are_scaled_to_effective_sample_size() -> None:
    method_b_output = {
        "sample_count": 14,
        "effective_sample_size": 11.0,
        "class_counts": {"UP": 2, "SIDEWAYS": 4, "DOWN": 8},
        "weighted_class_counts": {"UP": 14.0, "SIDEWAYS": 26.0, "DOWN": 60.0},
    }
    source = _source_effective_counts(method_b_output)
    assert source is not None
    counts, total = source
    assert abs(total - 11.0) < 1e-12
    probs = _smooth(counts, total, 0.0, None)
    assert probs == pytest.approx({"UP": 0.14, "SIDEWAYS": 0.26, "DOWN": 0.60})
