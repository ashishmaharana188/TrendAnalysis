from __future__ import annotations

import pytest

from phase5_8_decision_layer_audit import _normalise


def test_probability_normalisation():
    result = _normalise({"UP": 20, "SIDEWAYS": 30, "DOWN": 50})
    assert sum(result.values()) == pytest.approx(100.0)
    assert max(result, key=result.get) == "DOWN"


def test_missing_probability_keys_are_zero():
    result = _normalise({"UP": 70, "SIDEWAYS": 30})
    assert result["DOWN"] == pytest.approx(0.0)
    assert result["UP"] == pytest.approx(70.0)


def test_zero_mass_rejected():
    with pytest.raises(ValueError):
        _normalise({"UP": 0, "SIDEWAYS": 0, "DOWN": 0})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
