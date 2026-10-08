from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from phase6_2_probability_surface_audit import _distribution


def test_distribution_normalizes():
    result = _distribution({"UP": 20, "SIDEWAYS": 30, "DOWN": 50})
    assert result == pytest.approx({"UP": 20, "SIDEWAYS": 30, "DOWN": 50})


def test_distribution_handles_missing_class():
    result = _distribution({"UP": 50, "SIDEWAYS": 50})
    assert result == pytest.approx({"UP": 50, "SIDEWAYS": 50, "DOWN": 0})


def test_distribution_rejects_zero_mass():
    assert _distribution({"UP": 0, "SIDEWAYS": 0, "DOWN": 0}) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
