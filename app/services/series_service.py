from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from data_access.derivatives import get_futures_basis_history, get_options_history, get_unified_market_matrix
from data_access.institutional import get_institutional_flow_history, get_institutional_history
from data_access.macro_global import get_macro_history
from data_access.market import get_daily_history

from .catalog import (
    SOURCE_BASIS, SOURCE_INSTITUTIONAL, SOURCE_MACRO, SOURCE_MARKET,
    SOURCE_MATRIX, SOURCE_OPTIONS, FINANCIAL_METRICS, SeriesRequest, get_financial_loader,
)


def _to_frame(table: Any) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table.copy()
    if hasattr(table, "to_pandas"):
        return table.to_pandas()
    return pd.DataFrame(table)


def _empty_series() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "value": pd.Series(dtype="float64")})


def _col(df: pd.DataFrame, requested: str) -> str | None:
    target = requested.casefold()
    return next((column for column in df.columns if str(column).casefold() == target), None)


def _series_frame(df: pd.DataFrame, date_col: str, value_col: str, label: str) -> pd.DataFrame:
    date_name = _col(df, date_col)
    value_name = _col(df, value_col)
    if df.empty or date_name is None or value_name is None:
        return _empty_series()
    result = df[[date_name, value_name]].rename(columns={date_name: "date", value_name: "value"}).copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "value"])
    result = result.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    result.attrs["label"] = label
    return result


def load_series(
    request: SeriesRequest,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
) -> pd.DataFrame:
    """Load one numeric series through existing read-only data_access functions.

    `end_date` is optional by design. The app often loads observations after
    the chosen event-analysis window so realized forward responses are available.
    """
    source = request.source
    identifier = request.identifier
    metric = request.metric

    if source == SOURCE_MARKET:
        frame = _to_frame(get_daily_history(identifier, start_date=start_date, end_date=end_date))
        result = _series_frame(frame, "report_date", metric, request.label)
    elif source == SOURCE_MACRO:
        frame = _to_frame(get_macro_history(identifier, start_date=start_date, end_date=end_date))
        result = _series_frame(frame, "report_date", metric, request.label)
    elif source == SOURCE_MATRIX:
        frame = _to_frame(get_unified_market_matrix(identifier, start_date=start_date, end_date=end_date))
        result = _series_frame(frame, "date", metric, request.label)
    elif source == SOURCE_OPTIONS:
        frame = _to_frame(get_options_history(identifier, start_date=start_date, end_date=end_date))
        if not frame.empty:
            date_col = _col(frame, "report_date")
            puts_oi, calls_oi = _col(frame, "total_put_oi"), _col(frame, "total_call_oi")
            puts_vol, calls_vol = _col(frame, "total_put_volume"), _col(frame, "total_call_volume")
            aggregations: dict[str, str] = {}
            for column in (puts_oi, calls_oi, puts_vol, calls_vol):
                if column is not None:
                    aggregations[column] = "sum"
            if date_col is not None and aggregations:
                frame = frame.groupby(date_col, as_index=False).agg(aggregations)
                if puts_oi and calls_oi and puts_oi in frame and calls_oi in frame:
                    frame["oi_pcr"] = np.where(frame[calls_oi] != 0, frame[puts_oi] / frame[calls_oi], np.nan)
                if puts_vol and calls_vol and puts_vol in frame and calls_vol in frame:
                    frame["volume_pcr"] = np.where(frame[calls_vol] != 0, frame[puts_vol] / frame[calls_vol], np.nan)
        result = _series_frame(frame, "report_date", metric, request.label)
    elif source == SOURCE_BASIS:
        frame = _to_frame(get_futures_basis_history(identifier, start_date=start_date, end_date=end_date))
        if not frame.empty and "report_date" in frame.columns:
            # Select the highest-open-interest expiry per date to avoid duplicate dates.
            if "open_interest" in frame.columns:
                frame = frame.sort_values(["report_date", "open_interest"], ascending=[True, False], na_position="last")
            frame = frame.drop_duplicates("report_date", keep="first")
        result = _series_frame(frame, "report_date", metric, request.label)
    elif source == SOURCE_INSTITUTIONAL:
        # Raw ledger measures and the pre-derived net-position measures live in
        # different existing access functions. Route each metric to its source.
        raw_ledger_metrics = {
            "cash_buy_value", "cash_sell_value", "cash_net_value", "nifty_close",
            "future_index_long", "future_index_short", "future_stock_long", "future_stock_short",
            "option_index_call_long", "option_index_put_long", "option_index_call_short",
            "option_index_put_short", "option_stock_call_long", "option_stock_put_long",
            "option_stock_call_short", "option_stock_put_short", "total_long_contracts",
            "total_short_contracts",
        }
        if metric in raw_ledger_metrics:
            frame = _to_frame(get_institutional_history(
                start_date=start_date, end_date=end_date, client_type=identifier
            ))
        else:
            frame = _to_frame(get_institutional_flow_history(
                start_date=start_date, end_date=end_date, client_type=identifier
            ))
        result = _series_frame(frame, "report_date", metric, request.label)
    elif source in FINANCIAL_METRICS:
        loader = get_financial_loader(source)
        if loader is None:
            raise ValueError(f"No financial data loader is registered for {source!r}")
        frame = _to_frame(loader(identifier, start_date=start_date, end_date=end_date))
        result = _series_frame(frame, "ReportDate", metric, request.label)
    else:
        raise ValueError(f"Unsupported data source: {source}")

    if result.empty:
        result.attrs["label"] = request.label
        result.attrs["warning"] = "No numeric observations were returned for this series and date range."
    return result
