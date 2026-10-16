from __future__ import annotations

from math import atanh, tanh, sqrt
from typing import Iterable, Mapping

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
        out.loc[prior <= 0, "change"] = np.nan
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
        out = out[out["date"] <= pd.Timestamp(end_date)]
    return out


def paired_change_data(
    primary: pd.DataFrame,
    comparison: pd.DataFrame,
    window: int = 1,
    mode: str = "pct",
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """Matched-date changes for scatter/relationship plots; exact dates only."""
    p = change_series(primary, window, mode).rename(columns={"change": "primary_change"})
    c = change_series(comparison, window, mode).rename(columns={"change": "comparison_change"})
    joined = p.merge(c, on="date", how="inner")
    joined = joined.dropna(subset=["primary_change", "comparison_change"])
    return _date_filter(joined, start_date, end_date).reset_index(drop=True)


def _fisher_correlation_interval(r: float | None, n: int) -> tuple[float | None, float | None]:
    """Approximate Fisher-z 95% interval; assumes independent paired observations."""
    if r is None or n < 4 or not np.isfinite(r):
        return None, None
    # Exact +/-1 correlations in short windows make Fisher z infinite. Return a
    # conservative full-range interval rather than implying perfect certainty.
    if abs(r) >= 1.0:
        return -1.0, 1.0
    clipped = float(np.clip(r, -0.999999, 0.999999))
    z = atanh(clipped)
    margin = 1.96 / sqrt(n - 3)
    return tanh(z - margin), tanh(z + margin)


def pair_statistics(
    primary: pd.DataFrame,
    comparison: pd.DataFrame,
    window: int = 1,
    mode: str = "pct",
    start_date=None,
    end_date=None,
) -> dict[str, float | int | None]:
    joined = paired_change_data(primary, comparison, window, mode, start_date, end_date)
    n = int(len(joined))
    if n < 2:
        return {"observations": n, "pearson": None, "spearman": None, "slope": None, "pearson_ci_low": None, "pearson_ci_high": None}
    x = joined["primary_change"].to_numpy(dtype=float)
    y = joined["comparison_change"].to_numpy(dtype=float)
    pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None
    spearman = joined["primary_change"].corr(joined["comparison_change"], method="spearman")
    slope = float(np.polyfit(x, y, 1)[0]) if np.std(x) > 0 else None
    ci_low, ci_high = _fisher_correlation_interval(pearson, n)
    return {
        "observations": n,
        "pearson": pearson if pearson is None or np.isfinite(pearson) else None,
        "spearman": float(spearman) if pd.notna(spearman) else None,
        "slope": slope if slope is None or np.isfinite(slope) else None,
        "pearson_ci_low": ci_low,
        "pearson_ci_high": ci_high,
    }


def correlation_matrix(
    series: Mapping[str, pd.DataFrame],
    window: int = 1,
    mode: str = "pct",
    method: str = "pearson",
    start_date=None,
    end_date=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return correlation matrix and a matched-observation-count matrix."""
    changes: dict[str, pd.Series] = {}
    for label, frame in series.items():
        transformed = change_series(frame, window, mode).rename(columns={"change": label})
        if start_date is not None:
            transformed = transformed[transformed["date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            transformed = transformed[transformed["date"] <= pd.Timestamp(end_date)]
        changes[label] = transformed.set_index("date")[label]
    if not changes:
        return pd.DataFrame(), pd.DataFrame()
    aligned = pd.concat(changes.values(), axis=1, join="outer").sort_index()
    corr = aligned.corr(method=method, min_periods=3)
    counts = pd.DataFrame(index=aligned.columns, columns=aligned.columns, dtype=float)
    for left in aligned.columns:
        for right in aligned.columns:
            counts.loc[left, right] = int(aligned[[left, right]].dropna().shape[0])
    return corr, counts.astype("Int64")


def rolling_pair_correlation(
    primary: pd.DataFrame,
    comparison: pd.DataFrame,
    window: int = 1,
    mode: str = "pct",
    rolling_observations: int = 30,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """Rolling Pearson correlation with an approximate per-window Fisher interval.

    The interval assumes independent paired observations within each window.
    Rolling windows overlap, so the bands are pointwise/approximate, not a
    simultaneous confidence band and not robust to serial dependence.
    """
    if rolling_observations < 4:
        raise ValueError("rolling_observations must be >= 4")
    joined = paired_change_data(primary, comparison, window, mode, start_date, end_date)
    joined["rolling_correlation"] = joined["primary_change"].rolling(rolling_observations, min_periods=rolling_observations).corr(joined["comparison_change"]).clip(-1.0, 1.0)
    intervals = joined["rolling_correlation"].map(
        lambda value: _fisher_correlation_interval(float(value), rolling_observations)
        if pd.notna(value) else (None, None)
    )
    joined["rolling_ci_low"] = [pair[0] for pair in intervals]
    joined["rolling_ci_high"] = [pair[1] for pair in intervals]
    joined["rolling_observations"] = rolling_observations
    return joined[["date", "rolling_correlation", "rolling_ci_low", "rolling_ci_high", "rolling_observations"]]


def lead_lag_statistics(
    primary: pd.DataFrame,
    comparison: pd.DataFrame,
    window: int = 1,
    mode: str = "pct",
    max_lag: int = 10,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """Correlation for every tested alignment, with approximate Fisher intervals."""
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
        if n >= 3 and valid["x"].std() > 0 and valid["y"].std() > 0:
            candidate = valid["x"].corr(valid["y"])
            corr = float(candidate) if pd.notna(candidate) else None
        ci_low, ci_high = _fisher_correlation_interval(corr, n)
        records.append({
            "lag_observations": lag,
            "correlation": corr,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "observations": int(n),
        })
    return pd.DataFrame(records)


def _future_response(frame: pd.DataFrame, horizon: int, mode: str) -> pd.DataFrame:
    out = normalize_series(frame)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    future = out["value"].shift(-horizon)
    if mode == "pct":
        response = (future / out["value"] - 1.0) * 100.0
        response = response.mask(out["value"] <= 0)
    elif mode == "diff":
        response = future - out["value"]
    else:
        raise ValueError("mode must be 'pct' or 'diff'")
    return pd.DataFrame({"date": out["date"], "response": response}).replace([np.inf, -np.inf], np.nan)


def _sample_summary(sample: pd.Series, prefix: str = "") -> dict:
    values = pd.to_numeric(sample, errors="coerce").dropna().astype(float)
    n = int(values.size)
    mean = float(values.mean()) if n else None
    std = float(values.std(ddof=1)) if n > 1 else None
    # Conventional approximation only. Event returns overlap and may be dependent.
    half_width = 1.96 * std / sqrt(n) if n > 1 and std is not None else None
    return {
        f"{prefix}observations": n,
        f"{prefix}mean": mean,
        f"{prefix}median": float(values.median()) if n else None,
        f"{prefix}positive_pct": float((values > 0).mean() * 100) if n else None,
        f"{prefix}negative_pct": float((values < 0).mean() * 100) if n else None,
        f"{prefix}std": std,
        f"{prefix}ci_low": mean - half_width if mean is not None and half_width is not None else None,
        f"{prefix}ci_high": mean + half_width if mean is not None and half_width is not None else None,
    }


def _block_bootstrap_event_comparison(
    aligned: pd.DataFrame,
    event_direction: str,
    horizon: int,
    replicates: int = 400,
) -> dict[str, float | int | None]:
    """Moving-block bootstrap intervals for event mean, baseline mean and difference.

    Resamples contiguous blocks from the chronological matched-observation sequence.
    This approximately preserves some local dependence and overlapping-horizon effects;
    it is not a substitute for a pre-specified inferential design. Returns no interval
    for samples too small to support a minimally useful block bootstrap.
    """
    clean = aligned[["direction", "response"]].copy().reset_index(drop=True)
    clean["response"] = pd.to_numeric(clean["response"], errors="coerce")
    n = len(clean)
    if n < 20:
        return {
            "block_bootstrap_event_ci_low": None,
            "block_bootstrap_event_ci_high": None,
            "block_bootstrap_baseline_ci_low": None,
            "block_bootstrap_baseline_ci_high": None,
            "block_bootstrap_difference_ci_low": None,
            "block_bootstrap_difference_ci_high": None,
            "bootstrap_valid_replicates": 0,
            "bootstrap_block_length": None,
        }

    # Forward-horizon returns overlap for up to `horizon` observations. Keep blocks
    # at least horizon-long when possible, but retain at least three blocks overall.
    block_length = max(2, min(int(horizon), max(2, n // 3)))
    directions = clean["direction"].to_numpy(dtype=object)
    responses = clean["response"].to_numpy(dtype=float)
    rng = np.random.default_rng(20261016 + int(horizon) + (1 if event_direction == "Increase" else 2))
    event_means: list[float] = []
    baseline_means: list[float] = []
    differences: list[float] = []
    blocks_needed = int(np.ceil(n / block_length))

    for _ in range(int(replicates)):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = np.concatenate([
            (np.arange(start, start + block_length) % n) for start in starts
        ])[:n]
        sampled_directions = directions[indices]
        sampled_responses = responses[indices]
        event_mask = (sampled_directions == event_direction) & np.isfinite(sampled_responses)
        baseline_mask = (sampled_directions == "No event") & np.isfinite(sampled_responses)
        if event_mask.sum() < 2 or baseline_mask.sum() < 2:
            continue
        event_mean = float(sampled_responses[event_mask].mean())
        baseline_mean = float(sampled_responses[baseline_mask].mean())
        event_means.append(event_mean)
        baseline_means.append(baseline_mean)
        differences.append(event_mean - baseline_mean)

    if len(differences) < max(50, int(replicates * 0.25)):
        return {
            "block_bootstrap_event_ci_low": None,
            "block_bootstrap_event_ci_high": None,
            "block_bootstrap_baseline_ci_low": None,
            "block_bootstrap_baseline_ci_high": None,
            "block_bootstrap_difference_ci_low": None,
            "block_bootstrap_difference_ci_high": None,
            "bootstrap_valid_replicates": len(differences),
            "bootstrap_block_length": block_length,
        }

    q = lambda values, prob: float(np.quantile(np.asarray(values, dtype=float), prob))
    return {
        "block_bootstrap_event_ci_low": q(event_means, 0.025),
        "block_bootstrap_event_ci_high": q(event_means, 0.975),
        "block_bootstrap_baseline_ci_low": q(baseline_means, 0.025),
        "block_bootstrap_baseline_ci_high": q(baseline_means, 0.975),
        "block_bootstrap_difference_ci_low": q(differences, 0.025),
        "block_bootstrap_difference_ci_high": q(differences, 0.975),
        "bootstrap_valid_replicates": len(differences),
        "bootstrap_block_length": block_length,
    }


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
    """Compare subsequent target responses after events with a matched non-event baseline.

    Event dates are in the requested start/end window. Target forward responses may
    extend past end_date so long as those future observations are already available.
    Event-vs-baseline uncertainty is estimated with a circular moving-block bootstrap
    over chronologically matched observations to preserve some local dependence.
    The interval is exploratory, can be unavailable for small/sparse samples, and does
    not correct for selecting thresholds/horizons after looking at results.
    """
    primary_changes = change_series(primary, measurement_window, primary_mode).rename(columns={"change": "primary_change"})
    primary_changes = _date_filter(primary_changes, start_date, end_date)
    target_values = normalize_series(target)
    matched = primary_changes.dropna(subset=["primary_change"]).merge(target_values, on="date", how="inner").sort_values("date").reset_index(drop=True)
    matched["is_event"] = (matched["primary_change"] >= threshold) | (matched["primary_change"] <= -threshold)
    matched["direction"] = np.where(
        matched["primary_change"] >= threshold, "Increase",
        np.where(matched["primary_change"] <= -threshold, "Decrease", "No event"),
    )

    observations: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for horizon in sorted(set(int(h) for h in horizons if int(h) > 0)):
        response = _future_response(target, horizon, response_mode)
        aligned = matched.merge(response, on="date", how="left").sort_values("date").reset_index(drop=True)
        aligned["horizon"] = horizon
        aligned["event_value_change"] = aligned["primary_change"]
        event_horizon = aligned[aligned["is_event"]].copy()
        observations.append(event_horizon[["date", "direction", "primary_change", "horizon", "value", "response"]].copy())
        baseline_sample = aligned.loc[aligned["direction"] == "No event", "response"].dropna().astype(float)
        baseline_summary = _sample_summary(baseline_sample, prefix="baseline_")
        for direction in ("Increase", "Decrease"):
            event_sample = aligned.loc[aligned["direction"] == direction, "response"].dropna().astype(float)
            stats = _sample_summary(event_sample)
            n_events = int((aligned["direction"] == direction).sum())
            boot = _block_bootstrap_event_comparison(aligned, direction, horizon)
            row = {
                "horizon_observations": horizon,
                "primary_direction": direction,
                "events_detected": n_events,
                "realized_responses": stats["observations"],
                "mean_response": stats["mean"],
                "median_response": stats["median"],
                "positive_response_pct": stats["positive_pct"],
                "negative_response_pct": stats["negative_pct"],
                "response_std": stats["std"],
                # Retain legacy fields for compatibility, but populate them with
                # block-bootstrap intervals rather than independent-observation CIs.
                "approx_mean_ci_low": boot["block_bootstrap_event_ci_low"],
                "approx_mean_ci_high": boot["block_bootstrap_event_ci_high"],
                "block_bootstrap_event_ci_low": boot["block_bootstrap_event_ci_low"],
                "block_bootstrap_event_ci_high": boot["block_bootstrap_event_ci_high"],
                "block_bootstrap_baseline_ci_low": boot["block_bootstrap_baseline_ci_low"],
                "block_bootstrap_baseline_ci_high": boot["block_bootstrap_baseline_ci_high"],
                "block_bootstrap_difference_ci_low": boot["block_bootstrap_difference_ci_low"],
                "block_bootstrap_difference_ci_high": boot["block_bootstrap_difference_ci_high"],
                "bootstrap_valid_replicates": boot["bootstrap_valid_replicates"],
                "bootstrap_block_length": boot["bootstrap_block_length"],
                **baseline_summary,
            }
            baseline_mean = baseline_summary["baseline_mean"]
            row["mean_difference_vs_baseline"] = (
                stats["mean"] - baseline_mean
                if stats["mean"] is not None and baseline_mean is not None else None
            )
            summaries.append(row)
    observation_frame = pd.concat(observations, ignore_index=True) if observations else pd.DataFrame(
        columns=["date", "direction", "primary_change", "horizon", "value", "response"]
    )
    return pd.DataFrame(summaries), observation_frame


def variable_summary(frame: pd.DataFrame, start_date=None, end_date=None) -> dict[str, float | int | str | None]:
    """Summarize first-to-last change inside requested dates, not sum of daily changes."""
    out = normalize_series(frame)
    if not out.empty:
        out = _date_filter(out, start_date, end_date)
    if out.empty:
        return {"observations": 0, "first_date": None, "first_value": None, "latest_date": None, "latest_value": None, "change": None, "percent_change": None, "percent_change_available": False, "historical_percentile": None}
    first, latest = out.iloc[0], out.iloc[-1]
    start_value, end_value = float(first["value"]), float(latest["value"])
    change = end_value - start_value
    pct = ((end_value / start_value) - 1.0) * 100.0 if start_value > 0 else None
    pct_rank = float((out["value"] <= end_value).mean() * 100)
    return {
        "observations": int(len(out)), "first_date": pd.Timestamp(first["date"]).date().isoformat(),
        "first_value": start_value, "latest_date": pd.Timestamp(latest["date"]).date().isoformat(),
        "latest_value": end_value, "change": float(change),
        "percent_change": float(pct) if pct is not None else None,
        "percent_change_available": pct is not None, "historical_percentile": pct_rank,
    }


def distribution_frame(frame: pd.DataFrame, mode: str = "levels", start_date=None, end_date=None) -> pd.DataFrame:
    """Return selected distribution observations; changes use prior loaded observation."""
    normalized = normalize_series(frame)
    if mode == "levels":
        out = normalized.copy()
    elif mode in ("absolute_change", "percentage_change"):
        change_mode = "diff" if mode == "absolute_change" else "pct"
        out = change_series(normalized, window=1, mode=change_mode).rename(columns={"change": "value"})
    else:
        raise ValueError("mode must be 'levels', 'absolute_change', or 'percentage_change'")
    if not out.empty:
        out = _date_filter(out, start_date, end_date)
    return out.reset_index(drop=True)


def distribution_summary(frame: pd.DataFrame, start_date=None, end_date=None) -> dict[str, float | int | None]:
    """Descriptive distribution statistics for observations in the specified period."""
    out = normalize_series(frame)
    if not out.empty:
        out = _date_filter(out, start_date, end_date)
    values = out["value"].astype(float) if not out.empty else pd.Series(dtype=float)
    if values.empty:
        return {"observations": 0, "mean": None, "median": None, "std_dev": None, "min": None, "p05": None, "p25": None, "p75": None, "p95": None, "max": None, "skewness": None, "excess_kurtosis": None}
    skew = values.skew() if values.size >= 3 else np.nan
    kurt = values.kurt() if values.size >= 4 else np.nan
    return {
        "observations": int(values.size), "mean": float(values.mean()), "median": float(values.median()),
        "std_dev": float(values.std(ddof=1)) if values.size > 1 else None,
        "min": float(values.min()), "p05": float(values.quantile(0.05)),
        "p25": float(values.quantile(0.25)), "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)), "max": float(values.max()),
        "skewness": float(skew) if pd.notna(skew) else None,
        "excess_kurtosis": float(kurt) if pd.notna(kurt) else None,
    }


def distribution_interpretation(summary: dict, mode: str) -> str:
    """Generate cautious plain-language descriptions from descriptive summaries."""
    n = int(summary.get("observations") or 0)
    if n == 0:
        return "No valid observations are available in this period."
    if n < 20:
        caution = f"Only {n} observations are available, so the shape and tail estimates are unstable. "
    else:
        caution = f"The summary uses {n} observations in the selected period. "
    mean, median = summary.get("mean"), summary.get("median")
    skew = summary.get("skewness")
    p05, p95 = summary.get("p05"), summary.get("p95")
    spread = f"The central 90% observed range is approximately {p05:,.5g} to {p95:,.5g}. " if p05 is not None and p95 is not None else ""
    if mean is not None and median is not None:
        if abs(mean - median) <= max(abs(summary.get("std_dev") or 0) * 0.1, 1e-12):
            centre = "The mean and median are close, so the centre is not strongly shifted by a difference between these two summaries. "
        elif mean > median:
            centre = "The mean is above the median; relatively large upper observations may be pulling the average upward. "
        else:
            centre = "The mean is below the median; relatively low observations may be pulling the average downward. "
    else:
        centre = ""
    if skew is not None and skew >= 0.5:
        shape = "The distribution is right-skewed: the upper tail extends farther or is more influential. "
    elif skew is not None and skew <= -0.5:
        shape = "The distribution is left-skewed: the lower tail extends farther or is more influential. "
    elif skew is not None:
        shape = "Skewness is not pronounced by the simple ±0.5 descriptive rule used here. "
    else:
        shape = "There are too few observations to describe skewness reliably. "
    if mode == "levels":
        purpose = "This describes observed levels; for a trending series, the histogram can mostly reflect the trend and is not a return distribution. "
    elif mode == "absolute_change":
        purpose = "This describes absolute step-to-step changes, including their sign and size. "
    else:
        purpose = "This describes step-to-step percentage changes, using only observations with a positive prior value. "
    return caution + purpose + centre + spread + shape + "These are descriptive patterns, not a forecast or a test of whether the series is good or bad."
