from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from typing import Any

import pyarrow as pa
from dateutil.relativedelta import relativedelta

from .history_panel import (
    HistoricalStateSnapshot,
    build_historical_state_outcome_panel,
)
from .relationship import HistoricalRelationshipObservation, RelationshipDiscoveryEngine
from .walk_forward_relationship import validate_walk_forward_relationships


TIMEFRAME_MONTHS: dict[str, float] = {
    "1W": 7 / 30.4375,
    "2W": 14 / 30.4375,
    "1M": 1.0,
    "3M": 3.0,
    "6M": 6.0,
    "9M": 9.0,
    "1Y": 12.0,
    "18M": 18.0,
    "2Y": 24.0,
    "3Y": 36.0,
    "5Y": 60.0,
}


@dataclass(frozen=True)
class RealOLAPValidationConfig:
    ticker: str = "RELIANCE"
    benchmark: str = "Nifty_50"
    analysis_timeframe: str = "6M"
    holding_period_months: float = 1.0
    entry_mode: str = "next_trading_day"
    max_folds: int = 60
    step_trading_days: int = 20
    min_training_observations: int = 12


@dataclass(frozen=True)
class RealOLAPValidationResult:
    ticker: str
    benchmark: str
    analysis_timeframe: str
    holding_period_months: float
    entry_mode: str
    panel_rows: int
    skipped_no_outcome: int
    skipped_no_state: int
    walk_forward_folds: int
    method_a_hit_rate_pct: float
    method_b_hit_rate_pct: float
    combined_hit_rate_pct: float
    leakage_violations: int
    state_future_violations: int
    outcome_temporal_violations: int
    latest_market_date: date | None
    first_prediction_date: date | None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _table_rows(table: pa.Table) -> list[dict[str, Any]]:
    rows = table.to_pylist()
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        report_date = row.get("report_date")
        close = row.get("close")
        if report_date is None or close is None:
            continue
        try:
            numeric_close = float(close)
        except (TypeError, ValueError):
            continue
        if not isfinite(numeric_close) or numeric_close <= 0:
            continue
        normalized = dict(row)
        normalized["report_date"] = _as_date(report_date)
        normalized["close"] = numeric_close
        cleaned.append(normalized)
    cleaned.sort(key=lambda row: row["report_date"])
    return cleaned


def _subset_table(
    rows: list[dict[str, Any]],
    start_date: date,
    end_date: date,
) -> pa.Table:
    selected = [
        row
        for row in rows
        if start_date <= row["report_date"] <= end_date
    ]
    if not selected:
        return pa.table({
            "report_date": pa.array([], type=pa.date32()),
            "close": pa.array([], type=pa.float64()),
            "volume": pa.array([], type=pa.float64()),
        })
    return pa.Table.from_pylist(selected)


def _close_on_or_before(
    rows: list[dict[str, Any]],
    target_date: date,
) -> tuple[date, float] | None:
    for row in reversed(rows):
        if row["report_date"] <= target_date:
            return row["report_date"], float(row["close"])
    return None


def _close_after(
    rows: list[dict[str, Any]],
    target_date: date,
) -> tuple[date, float] | None:
    for row in rows:
        if row["report_date"] > target_date:
            return row["report_date"], float(row["close"])
    return None


def _state_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        label = value.get("state")
        return label if isinstance(label, str) else None
    label = getattr(value, "state", None)
    return label if isinstance(label, str) else None


def _build_state_atoms(
    ticker: str,
    selected_table: pa.Table,
    historical_table: pa.Table,
) -> dict[str, str]:
    """
    Invoke the existing Phase 3.5 market-state engine.

    Phase 4 does not recreate state calculations. This adapter only maps the
    existing Phase 3 state objects into the Phase 4 categorical state surface.
    """
    from analysis.market_state import build_company_market_states

    states = build_company_market_states(
        ticker=ticker,
        selected_table=selected_table,
        historical_table=historical_table,
    )

    result: dict[str, str] = {}
    for metric, state_object in states.items():
        label = _state_label(state_object)
        if label is None or label in {"Unknown", ""}:
            continue
        result[f"company.market.{metric}"] = label
    return result


def _resolve_benchmark_history(
    benchmark: str,
    start_date: date,
    end_date: date,
) -> pa.Table:
    """Load the benchmark through Phase 2 repositories, never raw SQL."""
    try:
        from data_access.macro_global import get_macro_date_range, get_macro_history

        if get_macro_date_range(benchmark) is not None:
            return get_macro_history(
                benchmark,
                start_date=start_date,
                end_date=end_date,
            )
    except ImportError:
        pass

    from data_access.market import get_daily_history

    return get_daily_history(
        benchmark,
        start_date=start_date,
        end_date=end_date,
    )


def _build_table_outcome(
    company_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    target: str,
    prediction_date: date,
    holding_period_months: float,
    entry_mode: str,
    benchmark: str,
):
    """Apply the Phase 4.2 outcome contract against preloaded real OLAP rows."""
    from .outcome import OutcomeObservation

    if entry_mode == "next_trading_day":
        entry = _close_after(company_rows, prediction_date)
    else:
        entry = _close_on_or_before(company_rows, prediction_date)

    if entry is None:
        return OutcomeObservation(
            target=target,
            prediction_date=prediction_date,
            data_cutoff_date=prediction_date,
            entry_mode=entry_mode,
            holding_period_months=holding_period_months,
            entry_date=None,
            exit_date=None,
            entry_price=None,
            exit_price=None,
            stock_return_pct=None,
            benchmark_return_pct=None,
            relative_return_pct=None,
            benchmark=benchmark,
            valid=False,
            limitation="No real-OLAP entry observation exists.",
        )

    entry_date, entry_price = entry
    whole_months = int(holding_period_months)
    fractional_months = holding_period_months - whole_months

    target_exit_date = entry_date + relativedelta(months=whole_months)
    if fractional_months > 0:
        target_exit_date += relativedelta(
            days=round(fractional_months * 30.4375)
        )

    exit_row = _close_on_or_before(company_rows, target_exit_date)
    if exit_row is None or exit_row[0] < entry_date:
        return OutcomeObservation(
            target=target,
            prediction_date=prediction_date,
            data_cutoff_date=prediction_date,
            entry_mode=entry_mode,
            holding_period_months=holding_period_months,
            entry_date=entry_date,
            exit_date=None,
            entry_price=entry_price,
            exit_price=None,
            stock_return_pct=None,
            benchmark_return_pct=None,
            relative_return_pct=None,
            benchmark=benchmark,
            valid=False,
            limitation="No real-OLAP exit observation exists.",
        )

    exit_date, exit_price = exit_row
    stock_return = ((exit_price - entry_price) / abs(entry_price)) * 100.0

    benchmark_return = None
    benchmark_entry = _close_on_or_before(benchmark_rows, entry_date)
    benchmark_exit = _close_on_or_before(benchmark_rows, exit_date)
    if benchmark_entry is not None and benchmark_exit is not None:
        benchmark_return = (
            (benchmark_exit[1] - benchmark_entry[1])
            / abs(benchmark_entry[1])
        ) * 100.0

    relative_return = (
        stock_return - benchmark_return
        if benchmark_return is not None
        else None
    )

    return OutcomeObservation(
        target="REAL_OLAP",
        prediction_date=prediction_date,
        data_cutoff_date=prediction_date,
        entry_mode=entry_mode,
        holding_period_months=holding_period_months,
        entry_date=entry_date,
        exit_date=exit_date,
        entry_price=entry_price,
        exit_price=exit_price,
        stock_return_pct=stock_return,
        benchmark_return_pct=benchmark_return,
        relative_return_pct=relative_return,
        benchmark=benchmark,
        valid=isfinite(stock_return),
        limitation=None if isfinite(stock_return) else "Non-finite return.",
    )


def _prediction_dates(
    rows: list[dict[str, Any]],
    analysis_timeframe: str,
    holding_period_months: float,
    step_trading_days: int,
    max_folds: int,
) -> list[date]:
    if analysis_timeframe not in TIMEFRAME_MONTHS:
        raise ValueError(
            f"Unsupported analysis timeframe: {analysis_timeframe}"
        )

    timeframe_months = TIMEFRAME_MONTHS[analysis_timeframe]
    min_start = rows[0]["report_date"] + relativedelta(
        days=max(1, round(timeframe_months * 30.4375))
    )
    latest_usable = rows[-1]["report_date"] - relativedelta(
        days=max(1, round(holding_period_months * 30.4375))
    )

    candidates = [
        row["report_date"]
        for index, row in enumerate(rows)
        if row["report_date"] >= min_start
        and row["report_date"] <= latest_usable
        and index >= step_trading_days
    ]

    if not candidates:
        return []

    # Use evenly spaced trading-day observations rather than a fixed
    # calendar-window training period. The last folds are retained so the
    # test reaches the current data horizon.
    stepped = candidates[::max(1, step_trading_days)]
    return stepped[-max_folds:]


def build_real_olap_relationship_panel(
    config: RealOLAPValidationConfig,
) -> tuple[list[HistoricalRelationshipObservation], dict[str, int], date | None]:
    """
    Build a real historical Phase 3-state / Phase 4-outcome panel.

    The only database interaction occurs through the existing Phase 2
    repositories. State calculation is delegated to the existing Phase 3
    market-state engine.
    """
    from data_access.market import get_daily_history

    company_table = get_daily_history(config.ticker)
    company_rows = _table_rows(company_table)
    if not company_rows:
        raise RuntimeError(
            f"No real OLAP market history found for {config.ticker!r}."
        )

    latest_market_date = company_rows[-1]["report_date"]
    earliest_date = company_rows[0]["report_date"]

    benchmark_table = _resolve_benchmark_history(
        config.benchmark,
        earliest_date,
        latest_market_date,
    )
    benchmark_rows = _table_rows(benchmark_table)

    if not benchmark_rows:
        raise RuntimeError(
            f"No real OLAP benchmark history found for {config.benchmark!r}."
        )

    dates = _prediction_dates(
        company_rows,
        config.analysis_timeframe,
        config.holding_period_months,
        config.step_trading_days,
        config.max_folds,
    )

    snapshots: list[HistoricalStateSnapshot] = []
    outcomes: list[Any] = []
    skipped_no_state = 0
    skipped_no_outcome = 0
    state_future_violations = 0
    outcome_temporal_violations = 0

    timeframe_months = TIMEFRAME_MONTHS[config.analysis_timeframe]

    for prediction_date in dates:
        selected_start = prediction_date + relativedelta(
            days=-max(1, round(timeframe_months * 30.4375))
        )

        selected_table = _subset_table(
            company_rows,
            selected_start,
            prediction_date,
        )
        historical_table = _subset_table(
            company_rows,
            company_rows[0]["report_date"],
            prediction_date,
        )

        selected_dates = [
            row["report_date"]
            for row in selected_table.to_pylist()
            if row.get("report_date") is not None
        ]
        historical_dates = [
            row["report_date"]
            for row in historical_table.to_pylist()
            if row.get("report_date") is not None
        ]
        if selected_dates and max(selected_dates) > prediction_date:
            state_future_violations += 1
        if historical_dates and max(historical_dates) > prediction_date:
            state_future_violations += 1

        states = _build_state_atoms(
            config.ticker,
            selected_table,
            historical_table,
        )
        if not states:
            skipped_no_state += 1
            continue

        outcome = _build_table_outcome(
            company_rows,
            benchmark_rows,
            config.ticker,
            prediction_date,
            config.holding_period_months,
            config.entry_mode,
            config.benchmark,
        )
        if (
            outcome.entry_date is not None
            and config.entry_mode == "next_trading_day"
            and outcome.entry_date <= prediction_date
        ):
            outcome_temporal_violations += 1
        if (
            outcome.entry_date is not None
            and outcome.exit_date is not None
            and outcome.exit_date < outcome.entry_date
        ):
            outcome_temporal_violations += 1

        if not outcome.valid or outcome.stock_return_pct is None:
            skipped_no_outcome += 1
            continue

        snapshots.append(
            HistoricalStateSnapshot(
                as_of_date=prediction_date,
                data_cutoff_date=prediction_date,
                target=config.ticker,
                scope="company",
                states=states,
                limitations=(),
            )
        )
        outcomes.append(outcome)

    if not snapshots:
        raise RuntimeError(
            "Real OLAP validation produced no valid state/outcome pairs."
        )

    panel = build_historical_state_outcome_panel(
        snapshots=snapshots,
        outcomes=outcomes,
        cutoff_date=max(snapshot.as_of_date for snapshot in snapshots),
        strict=True,
    )

    stats = {
        "valid_pairs": panel.stats.valid_pairs,
        "skipped_no_outcome": skipped_no_outcome,
        "skipped_no_state": skipped_no_state,
        "skipped_cutoff_violation": panel.stats.skipped_cutoff_violation,
        "skipped_identity_mismatch": panel.stats.skipped_identity_mismatch,
        "state_future_violations": state_future_violations,
        "outcome_temporal_violations": outcome_temporal_violations,
    }
    return list(panel.observations), stats, latest_market_date


def validate_real_olap_relationships(
    config: RealOLAPValidationConfig,
) -> RealOLAPValidationResult:
    observations, stats, latest_market_date = build_real_olap_relationship_panel(config)

    # Walk-forward validation re-discovers relationships inside every fold.
    engine = RelationshipDiscoveryEngine(
        max_order=3,
        min_observations=config.min_training_observations,
    )
    result = validate_walk_forward_relationships(
        observations=observations,
        engine=engine,
        min_training_observations=config.min_training_observations,
    )

    # All state/outcome temporal checks are performed while constructing
    # the panel from real repository observations.
    state_future_violations = stats.get("state_future_violations", 0)
    outcome_temporal_violations = stats.get("outcome_temporal_violations", 0)

    first_prediction_date = min(
        (observation.as_of_date for observation in observations),
        default=None,
    )

    return RealOLAPValidationResult(
        ticker=config.ticker,
        benchmark=config.benchmark,
        analysis_timeframe=config.analysis_timeframe,
        holding_period_months=config.holding_period_months,
        entry_mode=config.entry_mode,
        panel_rows=len(observations),
        skipped_no_outcome=stats["skipped_no_outcome"],
        skipped_no_state=stats["skipped_no_state"],
        walk_forward_folds=len(result.folds),
        method_a_hit_rate_pct=result.method_a.directional_hit_rate_pct,
        method_b_hit_rate_pct=result.method_b.directional_hit_rate_pct,
        combined_hit_rate_pct=result.combined.directional_hit_rate_pct,
        leakage_violations=result.leakage_violations,
        state_future_violations=state_future_violations,
        outcome_temporal_violations=outcome_temporal_violations,
        latest_market_date=latest_market_date,
        first_prediction_date=first_prediction_date,
    )
