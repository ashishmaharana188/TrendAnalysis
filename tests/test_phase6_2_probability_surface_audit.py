from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from phase6_2_probability_surface_audit import _distribution, _group_name, SnapshotFold


def test_distribution_normalizes():
    result = _distribution({"UP": 20, "SIDEWAYS": 30, "DOWN": 50})
    assert result == pytest.approx({"UP": 20, "SIDEWAYS": 30, "DOWN": 50})


def test_distribution_handles_missing_class():
    result = _distribution({"UP": 50, "SIDEWAYS": 50})
    assert result == pytest.approx({"UP": 50, "SIDEWAYS": 50, "DOWN": 0})


def test_distribution_rejects_zero_mass():
    assert _distribution({"UP": 0, "SIDEWAYS": 0, "DOWN": 0}) is None


def test_group_classifies_method_agreement():
    row = SnapshotFold(
        prediction_date=__import__("datetime").date(2026, 1, 1),
        probabilities_pct={"UP": 50.0, "SIDEWAYS": 25.0, "DOWN": 25.0},
        actual_class="UP",
        predicted_trend="UP",
        method_a_trend="UP",
        method_b_trend="UP",
        baseline_probabilities_pct={"UP": 33.33, "SIDEWAYS": 33.33, "DOWN": 33.34},
    )
    assert _group_name(row) == "BOTH_AGREE"


def test_group_classifies_method_conflict():
    row = SnapshotFold(
        prediction_date=__import__("datetime").date(2026, 1, 1),
        probabilities_pct={"UP": 50.0, "SIDEWAYS": 25.0, "DOWN": 25.0},
        actual_class="UP",
        predicted_trend="NO_CLEAR_TREND",
        method_a_trend="UP",
        method_b_trend="DOWN",
        baseline_probabilities_pct={"UP": 33.33, "SIDEWAYS": 33.33, "DOWN": 33.34},
    )
    assert _group_name(row) == "BOTH_CONFLICT"
