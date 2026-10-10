from __future__ import annotations

from types import SimpleNamespace

from phase6_2_joint_calibration_selection import (
    _available_surface_rows,
    _grid_size_for_cap,
)


def test_grid_density_is_matched_in_log_temperature_space() -> None:
    assert _grid_size_for_cap(0.25, 4.0, 4.0, 161) == 161
    assert _grid_size_for_cap(0.25, 8.0, 4.0, 161) == 201
    assert _grid_size_for_cap(0.25, 16.0, 4.0, 161) == 241


def test_larger_temperature_cap_does_not_reduce_log_grid_points() -> None:
    sizes = [_grid_size_for_cap(0.25, cap, 4.0, 161) for cap in (4.0, 8.0, 16.0)]
    assert sizes == sorted(sizes)


def test_surface_availability_gate_handles_partial_and_invalid_vectors() -> None:
    rows = [
        SimpleNamespace(a={"UP": 50.0, "SIDEWAYS": 30.0, "DOWN": 20.0}),
        SimpleNamespace(a={"UP": 0.0, "SIDEWAYS": 0.0, "DOWN": 0.0}),
        SimpleNamespace(a=None),
    ]

    available = _available_surface_rows(rows, "a")

    assert len(available) == 1
    assert available[0] is rows[0]
