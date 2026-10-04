from __future__ import annotations

from datetime import date

from analysis.history_panel import (
    HistoricalStateSnapshot,
    build_historical_state_outcome_panel,
    validate_state_outcome_pair,
)
from dataclasses import dataclass


@dataclass(frozen=True)
class SimpleOutcome:
    target: str
    prediction_date: date
    data_cutoff_date: date
    entry_mode: str
    holding_period_months: float
    entry_date: date | None
    exit_date: date | None
    entry_price: float | None
    exit_price: float | None
    stock_return_pct: float | None
    benchmark_return_pct: float | None
    relative_return_pct: float | None
    benchmark: str | None
    valid: bool
    limitation: str | None


def _outcome(
    target: str,
    prediction_date: date,
    cutoff: date,
    entry_date: date,
    exit_date: date,
    entry_mode: str = "next_trading_day",
    stock_return: float = 5.0,
) -> OutcomeObservation:
    return SimpleOutcome(
        target=target,
        prediction_date=prediction_date,
        data_cutoff_date=cutoff,
        entry_mode=entry_mode,
        holding_period_months=1.0,
        entry_date=entry_date,
        exit_date=exit_date,
        entry_price=100.0,
        exit_price=100.0 + stock_return,
        stock_return_pct=stock_return,
        benchmark_return_pct=1.0,
        relative_return_pct=stock_return - 1.0,
        benchmark="Nifty_50",
        valid=True,
        limitation=None,
    )


def main() -> None:
    prediction_date = date(2025, 1, 10)
    cutoff = date(2025, 1, 9)

    snapshots = [
        HistoricalStateSnapshot(
            as_of_date=prediction_date,
            data_cutoff_date=cutoff,
            target="RELIANCE",
            scope="company",
            states={
                "company.market.price": "Rising / High",
                "company.financials.TotalRevenue": "Rising / High",
            },
        ),
        HistoricalStateSnapshot(
            as_of_date=date(2025, 1, 11),
            data_cutoff_date=date(2025, 1, 10),
            target="RELIANCE",
            scope="company",
            states={
                "company.market.price": "Falling / Low",
            },
        ),
    ]

    outcomes = [
        _outcome(
            "RELIANCE",
            prediction_date,
            cutoff,
            entry_date=date(2025, 1, 13),
            exit_date=date(2025, 2, 13),
            stock_return=6.0,
        ),
        _outcome(
            "RELIANCE",
            date(2025, 1, 11),
            date(2025, 1, 10),
            entry_date=date(2025, 1, 13),
            exit_date=date(2025, 2, 13),
            stock_return=-2.0,
        ),
    ]

    panel = build_historical_state_outcome_panel(
        snapshots,
        outcomes,
        cutoff_date=date(2025, 1, 11),
        strict=True,
    )

    assert panel.stats.valid_pairs == 1
    assert panel.stats.skipped_cutoff_violation == 1
    assert len(panel.observations) == 1

    row = panel.observations[0]
    assert row.stock_return_pct == 6.0
    assert row.benchmark_return_pct == 1.0
    assert row.relative_return_pct == 5.0
    assert row.states["company.market.price"] == "Rising / High"

    # Latest-known-data entry must be on or before the state cutoff.
    valid_latest = _outcome(
        "RELIANCE",
        prediction_date,
        cutoff,
        entry_date=cutoff,
        exit_date=date(2025, 2, 10),
        entry_mode="latest_known_data",
    )
    assert not validate_state_outcome_pair(
        snapshots[0],
        valid_latest,
    )

    invalid_latest = _outcome(
        "RELIANCE",
        prediction_date,
        cutoff,
        entry_date=date(2025, 1, 10),
        exit_date=date(2025, 2, 10),
        entry_mode="latest_known_data",
    )
    assert validate_state_outcome_pair(
        snapshots[0],
        invalid_latest,
    )

    # Missing outcome is a limitation, not an invented zero-return row.
    missing_panel = build_historical_state_outcome_panel(
        [snapshots[0]],
        [],
        strict=False,
    )
    assert missing_panel.stats.valid_pairs == 0
    assert missing_panel.stats.skipped_missing_outcome == 1
    assert missing_panel.observations == ()
    assert missing_panel.limitations

    print("PHASE 4.2 HISTORICAL STATE/OUTCOME PANEL TEST: PASS")
    print("Valid pairs:", panel.stats.valid_pairs)
    print("Skipped cutoff rows:", panel.stats.skipped_cutoff_violation)
    print("Missing outcome rows:", missing_panel.stats.skipped_missing_outcome)


if __name__ == "__main__":
    main()
