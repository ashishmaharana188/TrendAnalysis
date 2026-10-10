from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from phase6_2_probability_surface_audit import (
    SnapshotFold,
    _distribution,
    _group_name,
    _surface_status,
)


def test_distribution_normalizes():
    result = _distribution({"UP": 20, "SIDEWAYS": 30, "DOWN": 50})
    assert result == pytest.approx({"UP": 20, "SIDEWAYS": 30, "DOWN": 50})


def test_distribution_handles_missing_class():
    result = _distribution({"UP": 50, "SIDEWAYS": 50})
    assert result == pytest.approx({"UP": 50, "SIDEWAYS": 50, "DOWN": 0})


def test_distribution_rejects_zero_or_invalid_mass():
    assert _distribution({"UP": 0, "SIDEWAYS": 0, "DOWN": 0}) is None
    assert _distribution({"UP": -10, "SIDEWAYS": 60, "DOWN": 50}) is None


def _row(a: str | None, b: str | None) -> SnapshotFold:
    return SnapshotFold(
        prediction_date=date(2026, 1, 1),
        probabilities_pct={"UP": 50.0, "SIDEWAYS": 25.0, "DOWN": 25.0},
        actual_class="UP",
        predicted_trend="UP",
        method_a_trend=a,
        method_b_trend=b,
        baseline_probabilities_pct={"UP": 33.33, "SIDEWAYS": 33.33, "DOWN": 33.34},
    )


def test_group_classifies_method_agreement():
    assert _group_name(_row("UP", "UP")) == "BOTH_AGREE"


def test_group_classifies_method_conflict():
    assert _group_name(_row("UP", "DOWN")) == "BOTH_CONFLICT"


def test_surface_status_distinguishes_partial_from_unavailable():
    assert _surface_status(208, 208) == "COMPLETE"
    assert _surface_status(187, 208) == "PARTIAL"
    assert _surface_status(0, 208) == "UNAVAILABLE"
