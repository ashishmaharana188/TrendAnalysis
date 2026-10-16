from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


TRANSFORMS = {
    "Percentage change": "pct",
    "Absolute change": "diff",
}


def normalize_series(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or not {"date", "value"}.issubset(frame.columns):
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "value": pd.Series(dtype="float64")})
    out = frame[["date", "value"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "value"])
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def change_series(frame: pd.DataFrame, window: int = 1, mode: str = "pct") -> pd.DataFrame:
    if window < 1:
        raise ValueError("window must be >= 1")
    out = normalize_series(frame)
    if out.empty:
        out["change"] = pd.Series(dtype="float64")
        return out[["date", "change"]]
    if mode == "pct":
        prior = out["value"].shift(window)
        out["change"] = (out["value"] / prior - 1.0) * 100.0
        out.loc[prior == 0, "change"] = np.nan
    elif mode == "diff":
        out["change"] = out["value"].diff(window)
    else:
        raise ValueError("mode must be 'pct' or 'diff'")
    return out[["date", "change"]].replace([np.inf, -np.inf], np.nan)


def _date_filter(frame: pd.DataFrame, start_date=None, end_date=None) -> pd.DataFrame:
    out = frame.copy()
    if start_date is not None:
        out = out[out["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        # The event itself must be in the requested date range. Its response
        # may occur after end_date, and is calculated before this filter.
        out = out[out["date"] <= pd.Timestamp(end_date)]
    return out


def pair_statistics(
    primary: pd.DataFrame,
    comparison: pd.DataFrame,
    window: int = 1,
    mode: str = "pct",
    start_date=None,
    end_date=None,
) -> dict[str, float | int | None]:
    p = change_series(primary, window, mode).rename(columns={"change": "primary_change"})
    c = change_series(comparison, window, mode).rename(columns={"change": "comparison_change"})
    joined = p.merge(c, on="date", how="inner").dropna(subset=["primary_change", "comparison_change"])
    joined = _date_filter(joined, start_date, end_date)
    n = int(len(joined))
    if n < 2:
        return {"observations": n, "pearson": None, "spearman": None, "slope": None}
    x = joined["primary_change"].to_numpy(dtype=float)
    y = joined["comparison_change"].to_numpy(dtype=float)
    pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None
    spearman = joined["primary_change"].corr(joined["comparison_change"], method="spearman")
    slope = float(np.polyfit(x, y, 1)[0]) if np.std(x) > 0 else None
    return {
        "observations": n,
        "pearson": pearson if pearson is None or np.isfinite(pearson) else None,
        "spearman": float(spearman) if pd.notna(spearman) else None,
        "slope": slope if slope is None or np.isfinite(slope) else None,
    }


def lead_lag_statistics(
    primary: pd.DataFrame,
    comparison: pd.DataFrame,
    window: int = 1,
    mode: str = "pct",
    max_lag: int = 10,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    if max_lag < 0:
        raise ValueError("max_lag must be >= 0")
    p = change_series(primary, window, mode).rename(columns={"change": "primary_change"})
    c = change_series(comparison, window, mode).rename(columns={"change": "comparison_change"})
    joined = p.merge(c, on="date", how="inner").sort_values("date").reset_index(drop=True)
    joined = _date_filter(joined, start_date, end_date).reset_index(drop=True)
    records = []
    for lag in range(-max_lag, max_lag + 1):
        # Positive lag means the comparison-series change is later than the primary change.
        shifted = joined["comparison_change"].shift(-lag)
        valid = pd.DataFrame({"x": joined["primary_change"], "y": shifted}).dropna()
        n = len(valid)
        corr = None
        if n >= 2 and valid["x"].std() > 0 and valid["y"].std() > 0:
            candidate = valid["x"].corr(valid["y"])
            corr = float(candidate) if pd.notna(candidate) else None
        records.append({"lag_observations": lag, "correlation": corr, "observations": int(n)})
    return pd.DataFrame(records)


def _future_response(frame: pd.DataFrame, horizon: int, mode: str) -> pd.DataFrame:
    out = normalize_series(frame)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    future = out["value"].shift(-horizon)
    if mode == "pct":
        response = (future / out["value"] - 1.0) * 100.0
        response = response.mask(out["value"] == 0)
    elif mode == "diff":
        response = future - out["value"]
    else:
        raise ValueError("mode must be 'pct' or 'diff'")
    return pd.DataFrame({"date": out["date"], "response": response}).replace([np.inf, -np.inf], np.nan)


def event_study(
    primary: pd.DataFrame,
    target: pd.DataFrame,
    measurement_window: int = 5,
    threshold: float = 2.0,
    primary_mode: str = "pct",
    response_mode: str = "pct",
    horizons: Iterable[int] = (1, 3, 5, 20),
    start_date=None,
    end_date=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe subsequent target responses after thresholded primary changes.

    Horizons count observations in the target series. For daily market prices these
    normally correspond to trading sessions; for lower-frequency datasets they do not.
    Only events within start_date/end_date are included, but later target data is
    deliberately retained to measure realized responses when available.
    """
    primary_changes = change_series(primary, measurement_window, primary_mode).rename(columns={"change": "primary_change"})
    primary_changes = _date_filter(primary_changes, start_date, end_date)
    target_values = normalize_series(target)
    valid_primary = primary_changes.dropna(subset=["primary_change"])
    joined = valid_primary.merge(target_values, on="date", how="inner")
    joined = joined[(joined["primary_change"] >= threshold) | (joined["primary_change"] <= -threshold)].copy()
    joined["direction"] = np.where(joined["primary_change"] >= threshold, "Increase", "Decrease")

    observations: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for horizon in sorted(set(int(h) for h in horizons if int(h) > 0)):
        response = _future_response(target, horizon, response_mode)
        event_rows = joined.merge(response, on="date", how="left")
        event_rows["horizon"] = horizon
        event_rows["event_value_change"] = event_rows["primary_change"]
        observations.append(event_rows[["date", "direction", "primary_change", "horizon", "value", "response"]].copy())
        for direction in ("Increase", "Decrease"):
            sample = event_rows.loc[event_rows["direction"] == direction, "response"].dropna().astype(float)
            n = int(sample.size)
            mean = float(sample.mean()) if n else None
            std = float(sample.std(ddof=1)) if n > 1 else None
            ci_low = mean - 1.96 * std / np.sqrt(n) if n > 1 else None
            ci_high = mean + 1.96 * std / np.sqrt(n) if n > 1 else None
            summaries.append({
                "horizon_observations": horizon,
                "primary_direction": direction,
                "events_detected": int((event_rows["direction"] == direction).sum()),
                "realized_responses": n,
                "mean_response": mean,
                "median_response": float(sample.median()) if n else None,
                "positive_response_pct": float((sample > 0).mean() * 100) if n else None,
                "negative_response_pct": float((sample < 0).mean() * 100) if n else None,
                "response_std": std,
                "approx_mean_ci_low": float(ci_low) if ci_low is not None else None,
                "approx_mean_ci_high": float(ci_high) if ci_high is not None else None,
            })
    observation_frame = pd.concat(observations, ignore_index=True) if observations else pd.DataFrame(
        columns=["date", "direction", "primary_change", "horizon", "value", "response"]
    )
    return pd.DataFrame(summaries), observation_frame


def variable_summary(frame: pd.DataFrame, start_date=None, end_date=None) -> dict[str, float | int | str | None]:
    out = normalize_series(frame)
    if not out.empty:
        out = _date_filter(out, start_date, end_date)
    if out.empty:
        return {
            "observations": 0, "latest_date": None, "latest_value": None,
            "change": None, "percent_change": None, "historical_percentile": None,
        }
    latest = out.iloc[-1]
    prior = out["value"].iloc[0]
    change = float(latest["value"] - prior)
    pct = float((latest["value"] / prior - 1) * 100) if prior != 0 else None
    pct_rank = float((out["value"] <= latest["value"]).mean() * 100)
    return {
        "observations": int(len(out)),
        "latest_date": pd.Timestamp(latest["date"]).date().isoformat(),
        "latest_value": float(latest["value"]),
        "change": change,
        "percent_change": pct,
        "historical_percentile": pct_rank,
    }
