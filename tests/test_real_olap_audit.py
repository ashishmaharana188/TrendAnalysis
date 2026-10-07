from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from analysis.real_olap_audit import (
    _month_exit_date,
    _return_pct,
)


def test_return_formula_matches_phase4_contract() -> None:
    assert abs(_return_pct(100.0, 110.0) - 10.0) < 1e-12
    assert abs(_return_pct(125.0, 100.0) + 20.0) < 1e-12


def test_month_exit_date_matches_one_month_horizon() -> None:
    assert _month_exit_date(date(2026, 1, 5), 1.0) == date(2026, 2, 5)
