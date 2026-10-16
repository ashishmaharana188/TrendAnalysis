from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from app.components.charts import synchronized_series_chart, terminal_chart_html
from app.services.catalog import (
    FINANCIAL_METRICS,
    SOURCE_BASIS,
    SOURCE_INSTITUTIONAL,
    SOURCE_MACRO,
    SOURCE_MARKET,
    SOURCE_MATRIX,
    SOURCE_OPTIONS,
    SeriesRequest,
    list_identifiers,
    metric_options,
)
from app.services.series_service import load_series, series_construction_note
from app.services.statistics_service import (
    correlation_matrix,
    distribution_frame,
    distribution_interpretation,
    distribution_summary,
    event_study,
    lead_lag_statistics,
    pair_statistics,
    paired_change_data,
    rolling_pair_correlation,
    variable_summary,
)
from app.services.indicators import calculate_indicators
from app.services.time_ranges import normalize_date_range, subtract_period
from data_access.db import get_db_path

st.set_page_config(page_title="TrendAnalysis", page_icon="📈", layout="wide")

# Friendly categories are deliberately fewer than the underlying database tables.
DOMAIN_SOURCES = {
    "Market prices": [SOURCE_MARKET],
    "Macro & global": [SOURCE_MACRO],
    "Trading activity": [SOURCE_MATRIX],
    "Options": [SOURCE_OPTIONS],
    "Futures": [SOURCE_BASIS],
    "Institutional activity": [SOURCE_INSTITUTIONAL],
    "Company financials": list(FINANCIAL_METRICS),
}
FINANCIAL_SOURCE_LABELS = {
    "Quarterly income statement": "Quarterly income statement",
    "Yearly income statement": "Annual income statement",
    "Quarterly balance sheet": "Quarterly balance sheet",
    "Yearly balance sheet": "Annual balance sheet",
    "Quarterly cash flow": "Quarterly cash flow",
    "Yearly cash flow": "Annual cash flow",
    "Yearly indirect cash flow": "Annual cash flow details",
}


@st.cache_data(ttl=60, show_spinner=False)
def cached_identifiers(source: str) -> list[dict[str, str]]:
    return list_identifiers(source)


@st.cache_data(ttl=60, show_spinner=False)
def cached_series(source: str, identifier: str, metric: str, start_date: date) -> pd.DataFrame:
    return load_series(SeriesRequest(source, identifier, metric), start_date=start_date, end_date=None)


def default_metric(source: str, metrics) -> str:
    preferred = {
        SOURCE_MARKET: "close",
        SOURCE_MACRO: "close",
        SOURCE_MATRIX: "close",
        SOURCE_OPTIONS: "oi_pcr",
        SOURCE_BASIS: "basis_percentage",
        SOURCE_INSTITUTIONAL: "future_index_net",
    }
    if source in preferred and any(m.key == preferred[source] for m in metrics):
        return preferred[source]
    if source.endswith("income statement"):
        return "TotalRevenue" if any(m.key == "TotalRevenue" for m in metrics) else metrics[0].key
    if "balance sheet" in source:
        return "TotalAssets" if any(m.key == "TotalAssets" for m in metrics) else metrics[0].key
    if "cash flow" in source:
        candidates = ("CashFromOperations", "TotalOperatingCashFlow", "EndingCashBalance", "EndingCash")
        return next((key for key in candidates if any(m.key == key for m in metrics)), metrics[0].key)
    return metrics[0].key


def metric_help(source: str, metric: str) -> str:
    if source in (SOURCE_MARKET, SOURCE_MACRO):
        return "The default plots the series' closing/value observation. Choose another measure only when useful."
    if source == SOURCE_MATRIX:
        return "Trading activity includes fields such as volume, delivery, VWAP deviation and short activity."
    if source == SOURCE_OPTIONS:
        return "Put/call ratio compares aggregated put and call activity; it does not by itself prove market direction."
    if source == SOURCE_BASIS:
        return "Futures basis describes the futures price relative to its spot price."
    if source == SOURCE_INSTITUTIONAL:
        return "Institutional figures are reported by client category and must be interpreted in context."
    return "Financial statement dates are accounting period ends, not necessarily publication dates."


def _format_value(value, suffix: str = "") -> str:
    try:
        if value is None or pd.isna(value):
            return "Not available"
        return f"{float(value):,.3f}{suffix}"
    except (TypeError, ValueError):
        return "Not available"


def _heatmap_tick_label(label: str) -> str:
    """Short, readable two-line heatmap tick label; full name stays in hover."""
    parts = [part.strip() for part in str(label).split(" · ") if part.strip()]
    if len(parts) >= 3:
        identifier = parts[-2].replace("_", " ")
        measure = parts[-1].replace("_", " ")
    elif len(parts) == 2:
        identifier, measure = parts[0].replace("_", " "), parts[1].replace("_", " ")
    else:
        identifier, measure = (parts[0].replace("_", " ") if parts else "Series"), ""
    if len(identifier) > 22:
        identifier = identifier[:19].rstrip() + "…"
    if len(measure) > 20:
        measure = measure[:17].rstrip() + "…"
    return f"{identifier}<br>{measure}" if measure else identifier


def _request_from_entry(entry: dict[str, str]) -> SeriesRequest:
    return SeriesRequest(entry["source"], entry["identifier"], entry["metric"])


def _history_buffer_days(request: SeriesRequest) -> int:
    """Load enough lookback for the largest supported observation window."""
    if request.source in FINANCIAL_METRICS:
        if "yearly" in request.source:
            return 365 * 80
        if "quarterly" in request.source:
            return 365 * 25
    # Daily series: one year is ample for a 60-observation change window
    # while also covering market holidays and short gaps.
    return 365


def _friendly_series_label(request: SeriesRequest) -> str:
    metric_labels = {item.key: item.label for item in metric_options(request.source)}
    domain_short = {
        SOURCE_MARKET: "Price",
        SOURCE_MACRO: "Macro",
        SOURCE_MATRIX: "Trading activity",
        SOURCE_OPTIONS: "Options",
        SOURCE_BASIS: "Futures",
        SOURCE_INSTITUTIONAL: "Institutional",
    }
    domain = domain_short.get(request.source, FINANCIAL_SOURCE_LABELS.get(request.source, "Financials"))
    return f"{domain} · {request.identifier} · {metric_labels.get(request.metric, request.metric)}"


def _time_range_control(label: str, key: str, default_amount: int = 12, default_unit: str = "months") -> tuple[date, date]:
    """A plain-language date control: recent N units or manually selected dates."""
    today = date.today()
    with st.container(border=True):
        st.markdown(f"**{label}**")
        mode = st.radio(
            "Choose how to set the dates",
            ["Recent period", "Specific dates"],
            horizontal=True,
            key=f"{key}_mode",
            label_visibility="collapsed",
        )
        if mode == "Recent period":
            amount_col, unit_col = st.columns([1, 1.4])
            with amount_col:
                amount = int(st.number_input(
                    "Show last", min_value=1, max_value=3650, value=default_amount,
                    step=1, key=f"{key}_amount",
                ))
            with unit_col:
                unit = st.selectbox(
                    "Time unit", ["days", "weeks", "months", "years"],
                    index=["days", "weeks", "months", "years"].index(default_unit),
                    key=f"{key}_unit",
                    format_func=lambda item: item,
                )
            start = subtract_period(today, amount, unit)
            end = today
            st.caption(f"{start.isoformat()} to {end.isoformat()}")
        else:
            default_start = subtract_period(today, default_amount, default_unit)
            value = st.date_input(
                "From and to",
                value=(default_start, today),
                key=f"{key}_custom_dates",
            )
            start, end = normalize_date_range(value, default_start, today)
            if start > end:
                st.error("The start date must be on or before the end date.")
    return start, end


def _add_series_controls() -> None:
    entries: list[dict[str, str]] = st.session_state["workspace_series"]
    with st.expander("Add a variable to the chart", expanded=not entries):
        st.caption("Choose a data type and a series. You can add as many series as you need.")
        domain_col, detail_col = st.columns([1, 1.5])
        with domain_col:
            domain = st.selectbox("Data type", list(DOMAIN_SOURCES), key="add_domain")
        sources = DOMAIN_SOURCES[domain]
        if domain == "Company financials":
            with detail_col:
                source = st.selectbox(
                    "Financial data",
                    sources,
                    format_func=lambda item: FINANCIAL_SOURCE_LABELS.get(item, item),
                    key="add_financial_source",
                )
        else:
            source = sources[0]
            with detail_col:
                st.info({
                    "Market prices": "Historical price and volume",
                    "Macro & global": "Crude, currencies, yields, indices and other indicators",
                    "Trading activity": "Delivery, VWAP deviation, short activity and block trades",
                    "Options": "Put/call positioning and volume",
                    "Futures": "Spot/futures prices, basis and open interest",
                    "Institutional activity": "Institutional positioning and flow",
                }.get(domain, domain))

        try:
            identifiers = cached_identifiers(source)
        except Exception as exc:
            identifiers = []
            st.warning(f"Could not load this catalogue: {exc}")
        if not identifiers:
            st.info("No series are available in this data type. Check the database connection or choose another type.")
            return

        id_map = {item["value"]: item["label"] for item in identifiers}
        identifier = st.selectbox(
            "Series",
            options=list(id_map),
            format_func=lambda item: id_map.get(item, item),
            key=f"add_identifier_{source}",
            help="Choose a particular indicator, instrument, company or client category.",
        )
        metrics = metric_options(source)
        if not metrics:
            st.warning("No numeric measures are configured for this data type.")
            return
        default_key = default_metric(source, metrics)
        metric = default_key
        change_measure = st.checkbox("Choose a different measure (optional)", value=False, key="change_measure")
        if change_measure:
            metric_map = {item.key: item.label for item in metrics}
            metric = st.selectbox(
                "What value should the chart show?",
                options=list(metric_map),
                index=list(metric_map).index(default_key),
                format_func=lambda item: metric_map[item],
                key=f"add_metric_{source}",
                help=metric_help(source, default_key),
            )
        st.caption(f"Default measure: {next((item.label for item in metrics if item.key == metric), metric)}")
        if st.button("Add to chart", type="primary", use_container_width=True, key="add_series_button"):
            new_entry = {"source": source, "identifier": identifier, "metric": metric}
            if any((e["source"], e["identifier"], e["metric"]) == (source, identifier, metric) for e in entries):
                st.warning("That exact series is already in the chart.")
            else:
                entries.append(new_entry)
                st.session_state["workspace_series"] = entries
                st.rerun()


def _series_controls() -> list[SeriesRequest]:
    entries: list[dict[str, str]] = st.session_state["workspace_series"]
    if not entries:
        return []
    st.markdown("**In this chart**")
    st.caption("The first series is the main variable. All following series are comparisons. Remove or reorder them as your question changes.")
    for index, entry in enumerate(list(entries)):
        request = _request_from_entry(entry)
        label = _friendly_series_label(request)
        left, right = st.columns([6, 1])
        with left:
            st.write(("Main candidate · " if index == 0 else "") + label)
        with right:
            if st.button("Remove", key=f"remove_series_{index}"):
                current = st.session_state["workspace_series"]
                del current[index]
                st.session_state["workspace_series"] = current
                st.rerun()
    return [_request_from_entry(entry) for entry in entries]


def main() -> None:
    st.title("TrendAnalysis")
    st.caption("Explore market variables together, then test what their historical movements have been associated with.")

    if "workspace_series" not in st.session_state:
        st.session_state["workspace_series"] = []

    with st.sidebar:
        with st.expander("Database connection", expanded=False):
            try:
                db_path = get_db_path()
                st.success("Database path found")
                st.caption(str(db_path))
                st.caption("Read-only connection")
            except Exception as exc:
                st.error("Database is not configured or available.")
                st.code(str(exc))
                st.info("Set OLAP_DB_PATH to the existing ETL DuckDB file.")
                st.stop()
        if st.button("Refresh available data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption("No automatic buy/sell recommendations are generated.")

    _add_series_controls()
    requests = _series_controls()
    if not requests:
        st.info("Start by adding a variable. For example, choose a crude or currency series, then add Nifty, a company or another indicator for comparison.")
        st.stop()

    friendly_labels = [_friendly_series_label(request) for request in requests]
    if st.session_state.get("main_variable") not in friendly_labels:
        st.session_state["main_variable"] = friendly_labels[0]
    if len(friendly_labels) > 1:
        st.selectbox(
            "Main variable for statistical comparisons",
            friendly_labels,
            key="main_variable",
            help="Other selected series are compared with this variable. This does not change which series are drawn.",
        )
    else:
        st.caption(f"Main variable: {friendly_labels[0]}. Add another series to compare its movements or subsequent responses.")

    st.divider()
    chart_col, analysis_col = st.columns([1, 1])
    with chart_col:
        chart_start, chart_end = _time_range_control("Chart history", "chart_range", default_amount=1, default_unit="years")
    with analysis_col:
        same_analysis_range = st.checkbox(
            "Use chart dates for statistical analysis", value=True, key="same_analysis_range",
            help="Usually the easiest starting point. Turn this off to choose a separate period for the numerical tests.",
        )
        if same_analysis_range:
            analysis_start, analysis_end = chart_start, chart_end
            st.caption(f"Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}")
        else:
            analysis_start, analysis_end = _time_range_control("Statistical study period", "analysis_range", default_amount=1, default_unit="years")

    if chart_start > chart_end or analysis_start > analysis_end:
        st.error("The start date must be on or before the end date.")
        st.stop()

    earliest_requested_date = min(chart_start, analysis_start)
    data: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    with st.spinner("Loading the selected series…"):
        for request in requests:
            label = _friendly_series_label(request)
            try:
                load_start = earliest_requested_date - timedelta(days=_history_buffer_days(request))
                frame = cached_series(request.source, request.identifier, request.metric, load_start)
                data[label] = frame
                if frame.empty:
                    failures[label] = "No numeric observations were returned."
            except Exception as exc:
                failures[label] = str(exc)
    if failures:
        with st.expander("Data availability messages", expanded=False):
            for label, message in failures.items():
                st.warning(f"{label}: {message}")

    available = {label: frame for label, frame in data.items() if frame is not None and not frame.empty}
    if not available:
        st.error("None of the selected series returned numeric observations. Remove an unavailable series or check its source.")
        st.stop()

    # Keep the first selected series as main. If it failed, use the first available series.
    all_labels = list(data)
    requested_main = st.session_state.get("main_variable", _friendly_series_label(requests[0]))
    primary_label = requested_main if requested_main in available else next(iter(available))
    if primary_label != requested_main:
        st.warning("The selected main variable has no observations in the loaded data. Using the first available series for this analysis.")
    # Expose actual source coverage so a stale alias or lagging source is not
    # mistaken for an intentionally selected historical chart window.
    request_by_label = {_friendly_series_label(request): request for request in requests}
    coverage_rows = []
    daily_sources = {SOURCE_MARKET, SOURCE_MACRO, SOURCE_MATRIX, SOURCE_OPTIONS, SOURCE_BASIS, SOURCE_INSTITUTIONAL}
    stale_sources = []
    today_ts = pd.Timestamp(date.today())
    for label, frame in available.items():
        dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
        latest = dates.max() if not dates.empty else pd.NaT
        first = dates.min() if not dates.empty else pd.NaT
        visible_count = int(((frame["date"] >= pd.Timestamp(chart_start)) & (frame["date"] <= pd.Timestamp(chart_end))).sum())
        request = request_by_label.get(label)
        coverage_rows.append({
            "Series": label,
            "First loaded observation": first.date().isoformat() if pd.notna(first) else "Not available",
            "Latest loaded observation": latest.date().isoformat() if pd.notna(latest) else "Not available",
            "Rows loaded": len(frame),
            "Rows in chart period": visible_count,
        })
        if request and request.source in daily_sources and pd.notna(latest):
            age_days = (today_ts.normalize() - latest.normalize()).days
            if age_days > 5:
                stale_sources.append((label, latest.date().isoformat(), age_days))

    with st.expander("Data coverage and how each series is built", expanded=bool(stale_sources)):
        st.dataframe(pd.DataFrame(coverage_rows), use_container_width=True, hide_index=True)
        for label in available:
            request = request_by_label.get(label)
            if request is not None:
                st.caption(f"{label}: {series_construction_note(request)}")
    if stale_sources:
        stale_lines = "; ".join(f"{label} (latest {latest}, {age} days ago)" for label, latest, age in stale_sources)
        st.warning(
            "Some daily sources have no observations in the most recent five calendar days. "
            "This may be a source-coverage gap or a stale ticker alias; check Data coverage before relying on the latest value. " + stale_lines
        )

    primary = available[primary_label]
    primary_stats = variable_summary(primary, chart_start, chart_end)
    stat_cols = st.columns(4)
    stat_cols[0].metric("Latest observation", str(primary_stats.get("latest_date") or "Not available"))
    stat_cols[1].metric("Latest value", _format_value(primary_stats.get("latest_value")))
    stat_cols[2].metric("First-to-last change", _format_value(primary_stats.get("change")))
    pct_label = _format_value(primary_stats.get("percent_change"), "%") if primary_stats.get("percent_change_available") else "N/A"
    stat_cols[3].metric("First-to-last change (%)", pct_label)
    if primary_stats.get("observations", 0):
        st.caption(
            "Change is the last valid observed value minus the first valid observed value inside the selected chart dates, "
            f"not the sum of daily changes. Start: {primary_stats['first_date']} ({_format_value(primary_stats['first_value'])}); "
            f"end: {primary_stats['latest_date']} ({_format_value(primary_stats['latest_value'])}). "
            + ("Percentage change is omitted because the starting value is zero or negative." if not primary_stats.get("percent_change_available") else "Percentage change = (end value ÷ start value − 1) × 100.")
        )

    chart_tab, distribution_tab, analysis_tab, data_tab = st.tabs(["Charts", "Distribution & variability", "Statistical analysis", "Raw data"])
    with chart_tab:
        title_col, layout_col, sync_col = st.columns([2, 1, 1])
        with title_col:
            st.subheader("Market charts")
        with layout_col:
            columns_per_row = st.radio(
                "Charts per row",
                options=[2, 3],
                index=0,
                horizontal=True,
                key="charts_per_row",
                help="Choose two or three panels per row. Every panel remains on the same date timeline.",
            )
        with sync_col:
            sync_crosshair = st.checkbox(
                "Sync crosshair",
                value=True,
                key="sync_crosshair",
                help="Show vertical and horizontal cursor guides on every chart when you hover any one chart.",
            )
        st.caption(
            "Hover to inspect the date and OHLC readout in the top-right. Click a chart to lock the plus crosshair; "
            "drag it to move the date and value guides. Use Full-screen terminal for all charts or Expand on an individual chart."
        )
        indicator_overlays: dict[str, dict[str, pd.DataFrame]] = {}
        with st.expander("Chart indicators", expanded=False):
            st.markdown("**Moving averages and volatility bands**")
            st.caption("These are visual overlays, calculated independently from each chart's own observations. They do not alter source data or imply a trading signal.")
            indicator_cols = st.columns(3)
            with indicator_cols[0]:
                ema_9 = st.checkbox("EMA 9", value=False, key="indicator_ema_9", help="Exponential moving average over 9 observations; recent values receive more weight.")
                ema_20 = st.checkbox("EMA 20", value=False, key="indicator_ema_20", help="Exponential moving average over 20 observations.")
            with indicator_cols[1]:
                sma_20 = st.checkbox("SMA 20", value=False, key="indicator_sma_20", help="Arithmetic mean of the last 20 observations.")
                sma_50 = st.checkbox("SMA 50", value=False, key="indicator_sma_50", help="Arithmetic mean of the last 50 observations.")
            with indicator_cols[2]:
                bollinger = st.checkbox("Bollinger bands 20", value=False, key="indicator_bollinger_20", help="20-observation SMA with bands two rolling standard deviations above and below it.")
            selected_ema = tuple(period for period, enabled in ((9, ema_9), (20, ema_20)) if enabled)
            selected_sma = tuple(period for period, enabled in ((20, sma_20), (50, sma_50)) if enabled)
            if selected_ema or selected_sma or bollinger:
                for label, frame in available.items():
                    indicator_overlays[label] = calculate_indicators(
                        frame,
                        ema_periods=selected_ema,
                        sma_periods=selected_sma,
                        bollinger_period=20 if bollinger else None,
                    )
            st.caption("Windows count available observations, not calendar days. The same indicator setting can produce different values on different variables because each is calculated from its own series.")
        figure = synchronized_series_chart(
            available,
            chart_start,
            chart_end,
            columns_per_row=columns_per_row,
            sync_crosshair=sync_crosshair,
            indicator_overlays=indicator_overlays,
        )
        if figure is None:
            st.info("No selected observations fall inside the chart history range.")
        else:
            chart_html = terminal_chart_html(figure, sync_crosshair=sync_crosshair)
            # Keep the terminal at a stable height. Its header stays fixed while
            # the chart grid scrolls internally, even when many rows are present.
            components.html(
                chart_html,
                height=800,
                scrolling=False,
            )

    with distribution_tab:
        st.subheader("Understand the shape and variability of selected data")
        st.markdown(
            "A distribution describes **what values have occurred**, how tightly they cluster, and how often extreme observations appear. "
            "It does not forecast direction. Choose all series to compare them, or select a subset to keep the plots readable."
        )
        distribution_labels = st.multiselect(
            "Series to include",
            options=list(available),
            default=list(available),
            key="distribution_series_selection",
            help="Every selected series gets its own panel and numerical summary. Different units are not mixed on one axis.",
        )
        distribution_mode_label = st.selectbox(
            "What should be distributed?",
            ["Observed values", "Absolute change from previous observation", "Percentage change from previous observation"],
            index=2,
            key="distribution_mode_v11",
            help="For market-price research, start with percentage changes. Use absolute changes for signed flows or series that cross zero.",
        )
        distribution_mode_map = {
            "Observed values": "levels",
            "Absolute change from previous observation": "absolute_change",
            "Percentage change from previous observation": "percentage_change",
        }
        dist_mode = distribution_mode_map[distribution_mode_label]
        st.caption(f"Study period: {analysis_start.isoformat()} to {analysis_end.isoformat()}. Change values use the preceding available observation where available, even if it falls just before the displayed study start.")
        distribution_views: dict[str, pd.DataFrame] = {}
        distribution_summaries: dict[str, dict] = {}
        for label in distribution_labels:
            frame = distribution_frame(available[label], dist_mode, analysis_start, analysis_end)
            frame = frame.dropna(subset=["value"])
            distribution_views[label] = frame
            distribution_summaries[label] = distribution_summary(frame)

        if not distribution_labels:
            st.info("Select at least one series to inspect its distribution.")
        else:
            summary_records = []
            for label in distribution_labels:
                stats = distribution_summaries[label]
                summary_records.append({
                    "Series": label,
                    "Observations": stats["observations"],
                    "Mean": stats["mean"],
                    "Median": stats["median"],
                    "Std. deviation": stats["std_dev"],
                    "5th percentile": stats["p05"],
                    "25th percentile": stats["p25"],
                    "75th percentile": stats["p75"],
                    "95th percentile": stats["p95"],
                    "Skewness": stats["skewness"],
                })
            st.markdown("**Numerical summary**")
            st.dataframe(pd.DataFrame(summary_records), use_container_width=True, hide_index=True)
            st.caption(
                "Mean = arithmetic average; median = middle observation; standard deviation = spread around the mean; "
                "5th–95th percentiles = central 90% range; skewness describes asymmetry. Compare variables within their own units. "
                "Percentile and skewness estimates become unreliable with small samples."
            )

            plot_view = st.radio(
                "View distribution as",
                ["Histogram", "Box plot", "Cumulative percentile"],
                horizontal=True,
                key="distribution_plot_view_v11",
                help="Histogram shows counts in value ranges; box plot compares median and spread; cumulative percentile shows the share of observations at or below each value.",
            )
            selected_n = len(distribution_labels)
            grid_cols = 2 if selected_n > 1 else 1
            grid_rows = int(np.ceil(selected_n / grid_cols))
            subplot_titles = [label for label in distribution_labels]
            fig = make_subplots(
                rows=grid_rows, cols=grid_cols,
                subplot_titles=subplot_titles,
                vertical_spacing=min(0.12, 0.28 / max(grid_rows, 1)),
                horizontal_spacing=0.12 if grid_cols == 2 else 0.04,
            )
            for index, label in enumerate(distribution_labels):
                frame = distribution_views[label]
                if frame.empty:
                    continue
                row = index // grid_cols + 1
                col = index % grid_cols + 1
                values = frame["value"].to_numpy(dtype=float)
                if plot_view == "Histogram":
                    fig.add_trace(go.Histogram(
                        x=values,
                        nbinsx=min(40, max(8, int(np.sqrt(len(values))))),
                        name=label,
                        marker={"color": "#38BDF8", "line": {"color": "#0E1117", "width": 0.5}},
                        showlegend=False,
                        hovertemplate="Range: %{x}<br>Observations: %{y}<extra></extra>",
                    ), row=row, col=col)
                    fig.update_yaxes(title_text="Observation count", rangemode="tozero", row=row, col=col)
                elif plot_view == "Box plot":
                    fig.add_trace(go.Box(
                        y=values, name=label, boxpoints="outliers",
                        marker={"color": "#38BDF8", "size": 4},
                        line={"color": "#38BDF8"}, showlegend=False,
                        hovertemplate="Value: %{y:,.5g}<extra></extra>",
                    ), row=row, col=col)
                    value_axis = {
                        "levels": "Observed value (series units)",
                        "absolute_change": "Absolute change (series units)",
                        "percentage_change": "Percentage change (%)",
                    }[dist_mode]
                    fig.update_yaxes(title_text=value_axis, row=row, col=col)
                else:
                    sorted_values = np.sort(values)
                    cumulative_pct = np.arange(1, len(sorted_values) + 1) / len(sorted_values) * 100.0
                    fig.add_trace(go.Scatter(
                        x=sorted_values, y=cumulative_pct, mode="lines",
                        line={"color": "#38BDF8", "width": 2}, name=label,
                        showlegend=False,
                        hovertemplate="Value: %{x:,.5g}<br>Observations at or below: %{y:.1f}%<extra></extra>",
                    ), row=row, col=col)
                    fig.update_yaxes(title_text="Observations at or below (%)", range=[0, 100], row=row, col=col)
                if plot_view != "Box plot":
                    fig.update_xaxes(title_text={
                        "levels": "Observed value (series units)",
                        "absolute_change": "Absolute change (series units)",
                        "percentage_change": "Percentage change (%)",
                    }[dist_mode], row=row, col=col)
            figure_height = max(340, grid_rows * 300)
            fig.update_layout(
                height=figure_height,
                margin={"l": 45, "r": 20, "t": 55, "b": 40},
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                font={"color": "#D9E2F0", "size": 11},
                title={"text": f"{plot_view}: {distribution_mode_label}", "font": {"size": 16}},
                bargap=0.08,
            )
            fig.update_xaxes(gridcolor="rgba(154, 167, 186, 0.16)", zeroline=True, zerolinecolor="#5A6573")
            fig.update_yaxes(gridcolor="rgba(154, 167, 186, 0.16)")
            st.plotly_chart(fig, use_container_width=True, theme=None, config={"displaylogo": False})

            selected_interpretation = st.selectbox(
                "Read the interpretation for",
                distribution_labels,
                index=distribution_labels.index(primary_label) if primary_label in distribution_labels else 0,
                key="distribution_interpretation_series_v11",
            )
            selected_frame = distribution_views[selected_interpretation]
            selected_summary = distribution_summaries[selected_interpretation]
            st.markdown("**What the selected distribution says**")
            st.write(distribution_interpretation(selected_summary, dist_mode))
            if int(selected_summary.get("observations") or 0) < 30:
                st.warning("Fewer than 30 observations: treat shape, tail and skewness descriptions as tentative. The 30-observation guide is a caution, not a universal statistical cutoff.")
            st.caption("A histogram's height is a count, not a probability of what happens next. The cumulative percentile plot is empirical: it describes this selected historical sample, not a calibrated forecast.")

    with analysis_tab:
        st.subheader("Statistical research")
        guide_path = Path(__file__).with_name("STATISTICAL_GUIDE.md")
        with st.expander("Open statistical definitions and interpretation guide", expanded=False):
            if guide_path.exists():
                st.markdown(guide_path.read_text(encoding="utf-8"))
            else:
                st.info("The full statistical guide is not present in this app folder.")
        st.markdown(
            "This section measures historical relationships. It does not label a coefficient as universally good or bad: "
            "interpretation depends on effect size, matched sample, uncertainty, stability and the question being tested."
        )
        st.caption(f"Main variable: {primary_label} · Study dates: {analysis_start.isoformat()} to {analysis_end.isoformat()}")
        mode = st.radio(
            "Research question",
            ["Relationships", "Response after events", "Lead/lag timing"],
            horizontal=True,
            label_visibility="collapsed",
            key="analysis_mode_v11",
        )
        comparison_labels = [label for label in available if label != primary_label]
        if mode == "Relationships":
            with st.expander("How to read this study", expanded=True):
                st.markdown(
                    "**Correlation** describes how two sets of changes co-move. It ranges from −1 to +1: values near −1 indicate opposite linear movement; "
                    "near 0 indicates little linear association; near +1 indicates same-direction linear movement. This is not a probability or a causal effect. "
                    "**Spearman correlation** uses ranks and captures monotonic rather than specifically linear association."
                )
            setting_cols = st.columns(3)
            with setting_cols[0]:
                window = st.select_slider("Measure each change over", options=[1, 3, 5, 10, 20, 60], value=5, format_func=lambda n: f"{n} observation{'s' if n != 1 else ''}", key="pair_window_v11")
            with setting_cols[1]:
                transform_label = st.selectbox("Change measure", ["Percentage change", "Absolute change"], index=0, key="pair_transform_v11")
            with setting_cols[2]:
                rolling_options = [n for n in [5, 10, 20, 30, 60] if n >= 5]
                rolling_window = st.selectbox("Rolling correlation window", rolling_options, index=min(2, len(rolling_options)-1), key="rolling_window_v11", help="Number of matched observations per rolling estimate; small windows are more erratic.")
            transform = "pct" if transform_label == "Percentage change" else "diff"

            if len(available) >= 2:
                corr_matrix, count_matrix = correlation_matrix(
                    available, int(window), transform, "pearson", analysis_start, analysis_end
                )
                if not corr_matrix.empty:
                    st.markdown("**Across-variable relationship map**")
                    st.caption(f"Each cell is Pearson correlation between {window}-observation {'percentage changes' if transform == 'pct' else 'absolute changes'}. Each pair uses exact matching dates; hover a cell to see its matched count. Multi-observation changes overlap through time, so use this heatmap descriptively.")
                    full_labels = list(corr_matrix.columns)
                    short_labels = [_heatmap_tick_label(label) for label in full_labels]
                    heat = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=full_labels, y=list(corr_matrix.index),
                        zmin=-1, zmax=1, zmid=0, colorscale="RdBu",
                        text=np.round(corr_matrix.values, 2),
                        texttemplate="%{text:.2f}",
                        customdata=count_matrix.to_numpy(dtype=int),
                        hovertemplate="<b>%{y}</b><br>Compared with: %{x}<br>Pearson r: %{z:.3f}<br>Matched observations: %{customdata}<extra></extra>",
                        colorbar={"title": "Pearson r", "len": 0.85},
                    ))
                    heat.update_layout(
                        height=max(430, 82 * len(corr_matrix)),
                        margin={"l": 170, "r": 70, "t": 45, "b": 135},
                        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                        font={"color": "#D9E2F0", "size": 11},
                    )
                    heat.update_xaxes(tickmode="array", tickvals=full_labels, ticktext=short_labels, tickangle=0, tickfont={"size": 10}, automargin=True)
                    heat.update_yaxes(tickmode="array", tickvals=list(corr_matrix.index), ticktext=short_labels, tickfont={"size": 10}, automargin=True)
                    st.plotly_chart(heat, use_container_width=True, theme=None, config={"displaylogo": False})
                st.caption("A correlation magnitude has no universal 'good/bad' cutoff. Large values can be unstable, driven by a few outliers, or fail to persist in another period. Correlation here is descriptive; autocorrelation and overlapping windows can make conventional independent-observation intervals optimistic.")

            if len(available) >= 2:
                pair_select_cols = st.columns(2)
                with pair_select_cols[0]:
                    pair_x_label = st.selectbox("First variable (X-axis)", list(available), index=list(available).index(primary_label), key="pair_x_label_v11")
                pair_y_options = [label for label in available if label != pair_x_label]
                with pair_select_cols[1]:
                    default_y = pair_y_options[0]
                    pair_y_label = st.selectbox("Second variable (Y-axis)", pair_y_options, index=0, key="pair_y_label_v11")
                result = pair_statistics(available[pair_x_label], available[pair_y_label], int(window), transform, analysis_start, analysis_end)
                paired = paired_change_data(available[pair_x_label], available[pair_y_label], int(window), transform, analysis_start, analysis_end)
                mcols = st.columns(5)
                mcols[0].metric("Matched observations", str(result["observations"]))
                mcols[1].metric("Effective observations*", _format_value(result.get("effective_observations")))
                mcols[2].metric("Pearson r", _format_value(result["pearson"]))
                mcols[3].metric("Spearman ρ", _format_value(result["spearman"]))
                ci_text = "Unavailable"
                if result.get("pearson_ci_low") is not None and result.get("pearson_ci_high") is not None:
                    ci_text = f"{result['pearson_ci_low']:.2f} to {result['pearson_ci_high']:.2f}"
                mcols[4].metric("Approx. 95% interval for r", ci_text)
                st.caption("*Effective observations are a Bartlett-style approximation that discounts some serial dependence. They are not an exact sample size and cannot by themselves correct for non-stationarity, outliers, overlapping changes or trying many variable pairs.")
                if len(paired) >= 2:
                    x, y = paired["primary_change"], paired["comparison_change"]
                    scatter = go.Figure()
                    scatter.add_trace(go.Scatter(
                        x=x, y=y, mode="markers", name="Matched observations",
                        marker={"size": 6, "opacity": 0.65, "color": "#38BDF8"},
                        customdata=paired["date"].astype(str),
                        hovertemplate=f"Date: %{{customdata}}<br>{pair_x_label}: %{{x:,.5g}}<br>{pair_y_label}: %{{y:,.5g}}<extra></extra>",
                    ))
                    if np.std(x) > 0:
                        slope = result.get("slope")
                        intercept = float(np.mean(y) - slope * np.mean(x)) if slope is not None else None
                        if slope is not None and intercept is not None:
                            xline = np.linspace(float(np.min(x)), float(np.max(x)), 80)
                            scatter.add_trace(go.Scatter(x=xline, y=intercept + slope * xline, mode="lines", name="Linear fit", line={"color": "#F59E0B", "width": 2}))
                    change_unit = "percentage change (%)" if transform == "pct" else "absolute change"
                    scatter.update_layout(
                        title=f"Same-period changes · {window}-observation window · one point per matched date",
                        height=420, margin={"l": 65, "r": 25, "t": 60, "b": 75},
                        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                        font={"color": "#D9E2F0", "size": 11},
                        xaxis_title=f"{pair_x_label} · {window}-observation {change_unit}",
                        yaxis_title=f"{pair_y_label} · {window}-observation {change_unit}",
                        legend={"orientation": "h", "y": 1.08},
                    )
                    scatter.update_xaxes(gridcolor="rgba(154, 167, 186, 0.16)", zeroline=True)
                    scatter.update_yaxes(gridcolor="rgba(154, 167, 186, 0.16)", zeroline=True)
                    st.plotly_chart(scatter, use_container_width=True, theme=None, config={"displaylogo": False})
                    with st.expander("Inspect unusual matched observations", expanded=False):
                        outlier_view = paired[["date", "primary_change", "comparison_change"]].copy()
                        def _robust_z(series: pd.Series) -> pd.Series:
                            median = float(series.median())
                            mad = float((series - median).abs().median())
                            scale = 1.4826 * mad
                            if not np.isfinite(scale) or scale == 0:
                                scale = float(series.std(ddof=1)) if len(series) > 1 else 0.0
                            return (series - median) / scale if scale > 0 else pd.Series(0.0, index=series.index)
                        outlier_view["X robust deviation"] = _robust_z(outlier_view["primary_change"])
                        outlier_view["Y robust deviation"] = _robust_z(outlier_view["comparison_change"])
                        outlier_view["Combined deviation"] = np.sqrt(outlier_view["X robust deviation"] ** 2 + outlier_view["Y robust deviation"] ** 2)
                        outlier_view = outlier_view.sort_values("Combined deviation", ascending=False).head(10)
                        outlier_view["date"] = pd.to_datetime(outlier_view["date"]).dt.date.astype(str)
                        outlier_view = outlier_view.rename(columns={"date": "Date", "primary_change": f"{pair_x_label} change", "comparison_change": f"{pair_y_label} change"})
                        st.dataframe(outlier_view, use_container_width=True, hide_index=True)
                        st.caption("These are the 10 most unusual paired points by a median/MAD-based deviation score. They are flagged for inspection, not automatically removed or classified as data errors. Review the dates and raw observations before deciding how they should be handled.")

                    if result["pearson"] is not None:
                        magnitude = abs(float(result["pearson"]))
                        if magnitude < 0.20:
                            magnitude_text = "very weak linear association by a rough magnitude convention"
                        elif magnitude < 0.40:
                            magnitude_text = "weak-to-modest linear association by a rough magnitude convention"
                        elif magnitude < 0.60:
                            magnitude_text = "moderate linear association by a rough magnitude convention"
                        elif magnitude < 0.80:
                            magnitude_text = "strong linear association by a rough magnitude convention"
                        else:
                            magnitude_text = "very strong linear association by a rough magnitude convention"
                        sign_text = "positive" if result["pearson"] > 0 else "negative" if result["pearson"] < 0 else "zero"
                        st.markdown(f"**Descriptive reading:** the observed Pearson association is {sign_text} and {magnitude_text}. These magnitude labels are only rules of thumb; they are not a universal pass/fail threshold and do not say whether the relationship is economically useful.")
                    if result["pearson"] is None:
                        st.info("Pearson correlation is unavailable because one series has no variation or too few matched values.")
                    elif result["observations"] < 30:
                        st.warning(f"Only {result['observations']} matched observations: treat the estimated relationship as tentative.")
                    elif result.get("pearson_ci_low") is not None and result["pearson_ci_low"] <= 0 <= result["pearson_ci_high"]:
                        st.info("The approximate autocorrelation-adjusted interval for Pearson correlation includes zero. The estimate does not clearly distinguish this sample's linear association from zero; this does not prove the relationship is absent.")
                    elif result.get("pearson_ci_low") is not None:
                        st.info("The approximate autocorrelation-adjusted interval does not include zero in this sample. This is still exploratory: the adjustment is approximate and does not correct for outliers, non-stationarity, or selection across many variable pairs.")
                    else:
                        st.warning("A correlation interval is not shown because the estimated effective sample size is too small for a useful Fisher interval after discounting serial dependence. The raw correlation remains descriptive only.")
                    st.caption("The approximate 95% interval uses Fisher's transformation with a Bartlett-style effective sample size based on autocorrelation across several lags, including more lags for overlapping multi-observation changes where sample support permits. This is a rough dependence adjustment, not a complete time-series inferential procedure; the interval is not adjusted for selecting many variable pairs.")

                    rolling = rolling_pair_correlation(
                        available[pair_x_label], available[pair_y_label], int(window), transform,
                        int(rolling_window), analysis_start, analysis_end,
                    )
                    rolling = rolling.dropna(subset=["rolling_correlation"])
                    if not rolling.empty:
                        roll_fig = go.Figure()
                        ci_low = pd.to_numeric(rolling["rolling_ci_low"], errors="coerce")
                        ci_high = pd.to_numeric(rolling["rolling_ci_high"], errors="coerce")
                        roll_fig.add_trace(go.Scatter(
                            x=rolling["date"], y=ci_low, mode="lines", line={"width": 0},
                            showlegend=False, hoverinfo="skip", name="Approx. 95% lower bound",
                        ))
                        roll_fig.add_trace(go.Scatter(
                            x=rolling["date"], y=ci_high, mode="lines", line={"width": 0},
                            fill="tonexty", fillcolor="rgba(167, 139, 250, 0.16)",
                            showlegend=True, name="Approx. 95% pointwise interval", hoverinfo="skip",
                        ))
                        roll_fig.add_trace(go.Scatter(
                            x=rolling["date"], y=rolling["rolling_correlation"], mode="lines",
                            name="Rolling Pearson r", line={"color": "#A78BFA", "width": 2},
                            customdata=np.column_stack([ci_low, ci_high, rolling["rolling_observations"], rolling["effective_observations"]]),
                            hovertemplate="Date: %{x|%Y-%m-%d}<br>Rolling r: %{y:.3f}<br>Approx. interval: %{customdata[0]:.3f} to %{customdata[1]:.3f}<br>Pairs in window: %{customdata[2]}<br>Approx. effective observations: %{customdata[3]:.1f}<extra></extra>",
                        ))
                        roll_fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
                        roll_fig.update_layout(title=f"Relationship stability · {rolling_window}-observation rolling correlation", height=350, margin={"l": 55, "r": 20, "t": 55, "b": 45}, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font={"color": "#D9E2F0", "size": 11}, yaxis={"range": [-1, 1], "title": "Pearson r"}, xaxis_title="Date", legend={"orientation": "h", "y": 1.12})
                        roll_fig.update_xaxes(gridcolor="rgba(154, 167, 186, 0.16)")
                        roll_fig.update_yaxes(gridcolor="rgba(154, 167, 186, 0.16)")
                        st.plotly_chart(roll_fig, use_container_width=True, theme=None, config={"displaylogo": False})
                        st.caption("The shaded band is a pointwise approximate 95% Fisher interval using a Bartlett-style effective sample size that discounts some serial dependence inside each window. It is not a simultaneous band, does not fully correct dependence or non-stationarity, and windows overlap heavily. Sign flips may reflect changing relationships, sampling noise, or both.")
                    else:
                        st.info("Not enough matched observations to draw the selected rolling-correlation window.")
                else:
                    st.info("There are not enough exact matched dates to draw the relationship plot.")
            else:
                st.info("Add at least one comparison series to inspect a pairwise relationship.")

        elif mode == "Response after events":
            if not comparison_labels:
                st.info("Add a target series such as an index or company to measure its response after a primary-variable event.")
            else:
                st.markdown("**Question:** after a defined increase or decrease in the main variable, did the target's later behaviour differ from its ordinary non-event response?")
                st.caption("Formation window = how many observations are used to measure the main-variable move ending on the event date. Response horizon = how many target observations forward from that event date are measured. They are separate settings.")
                target_label = st.selectbox("Target whose response is measured", comparison_labels, key="event_target_v11")
                ecols = st.columns(4)
                with ecols[0]:
                    measurement_window = st.select_slider("Measure main-variable change over", options=[1, 3, 5, 10, 20, 60], value=5, format_func=lambda n: f"{n} observations", key="event_measurement_window_v11")
                with ecols[1]:
                    primary_change_label = st.selectbox("Primary-variable change", ["Percentage change", "Absolute change"], index=0, key="event_primary_mode_v11")
                with ecols[2]:
                    threshold = st.number_input("Event threshold", min_value=0.0, value=2.0, step=0.5, key="event_threshold_v11", help="For percentage change, 2 means at least +2% or at most −2% over the selected measurement window.")
                with ecols[3]:
                    response_label = st.selectbox("Target response units", ["Percentage change", "Absolute change"], index=0, key="event_response_mode_v11")
                horizons = st.multiselect("Measure response after", [1, 2, 3, 5, 10, 20, 30, 60], default=[1, 3, 5, 20], format_func=lambda n: f"{n} target observations", key="event_horizons_v11")
                st.warning("Predefine your event threshold and response horizons where possible. The current intervals do not correct for trying many thresholds, horizons or event definitions and then highlighting the most attractive result.")
                if not horizons:
                    st.info("Select at least one response horizon.")
                else:
                    summary, event_rows = event_study(
                        primary, available[target_label], int(measurement_window), float(threshold),
                        "pct" if primary_change_label == "Percentage change" else "diff",
                        "pct" if response_label == "Percentage change" else "diff",
                        horizons, analysis_start, analysis_end,
                    )
                    if summary.empty or not (summary["events_detected"] > 0).any():
                        st.info("No historical events met this threshold with matching target observations. Lower the threshold or widen the study period if appropriate.")
                    else:
                        st.markdown("**Subsequent target responses versus non-event baseline**")
                        rows_display = summary[["horizon_observations", "primary_direction", "events_detected", "realized_responses", "mean_response", "median_response", "positive_response_pct", "block_bootstrap_event_ci_low", "block_bootstrap_event_ci_high", "baseline_observations", "baseline_mean", "block_bootstrap_baseline_ci_low", "block_bootstrap_baseline_ci_high", "mean_difference_vs_baseline", "block_bootstrap_difference_ci_low", "block_bootstrap_difference_ci_high", "bootstrap_valid_replicates", "bootstrap_block_length"]].rename(columns={
                            "horizon_observations": "Target observations ahead", "primary_direction": "Primary event", "events_detected": "Events found", "realized_responses": "Event responses available", "mean_response": "Event mean response", "median_response": "Event median response", "positive_response_pct": "Event responses positive (%)", "block_bootstrap_event_ci_low": "Event mean CI low", "block_bootstrap_event_ci_high": "Event mean CI high", "baseline_observations": "Baseline responses available", "baseline_mean": "Non-event baseline mean", "block_bootstrap_baseline_ci_low": "Baseline mean CI low", "block_bootstrap_baseline_ci_high": "Baseline mean CI high", "mean_difference_vs_baseline": "Event minus baseline", "block_bootstrap_difference_ci_low": "Difference CI low", "block_bootstrap_difference_ci_high": "Difference CI high", "bootstrap_valid_replicates": "Valid bootstrap samples", "bootstrap_block_length": "Bootstrap block length",
                        })
                        st.dataframe(rows_display, use_container_width=True, hide_index=True)
                        response_fig = go.Figure()
                        increase = summary[summary["primary_direction"] == "Increase"].sort_values("horizon_observations")
                        decrease = summary[summary["primary_direction"] == "Decrease"].sort_values("horizon_observations")
                        baseline = summary.drop_duplicates("horizon_observations").sort_values("horizon_observations")
                        for group, name, color in ((increase, "After primary increase", "#F59E0B"), (decrease, "After primary decrease", "#38BDF8")):
                            if not group.empty:
                                y = pd.to_numeric(group["mean_response"], errors="coerce")
                                low = pd.to_numeric(group["approx_mean_ci_low"], errors="coerce")
                                high = pd.to_numeric(group["approx_mean_ci_high"], errors="coerce")
                                response_fig.add_trace(go.Scatter(
                                    x=group["horizon_observations"], y=y, mode="lines+markers", name=name,
                                    line={"color": color, "width": 2}, marker={"size": 7},
                                    error_y={"type": "data", "symmetric": False, "array": (high-y).clip(lower=0).fillna(0), "arrayminus": (y-low).clip(lower=0).fillna(0), "visible": True},
                                    hovertemplate="Horizon: %{x} observations<br>Mean target response: %{y:.4g}<extra></extra>",
                                ))
                        if not baseline.empty:
                            base_y = pd.to_numeric(baseline["baseline_mean"], errors="coerce")
                            base_low = pd.to_numeric(baseline["block_bootstrap_baseline_ci_low"], errors="coerce")
                            base_high = pd.to_numeric(baseline["block_bootstrap_baseline_ci_high"], errors="coerce")
                            response_fig.add_trace(go.Scatter(
                                x=baseline["horizon_observations"], y=base_y, mode="lines+markers", name="Non-event baseline mean",
                                line={"color": "#A78BFA", "width": 2, "dash": "dash"}, marker={"size": 6},
                                error_y={"type": "data", "symmetric": False, "array": (base_high-base_y).clip(lower=0).fillna(0), "arrayminus": (base_y-base_low).clip(lower=0).fillna(0), "visible": True},
                                hovertemplate="Horizon: %{x} observations<br>Baseline mean: %{y:.4g}<extra></extra>",
                            ))
                        response_fig.add_hline(y=0, line_dash="dot", line_color="#94A3B8")
                        response_fig.update_layout(
                            title="Mean target response by horizon (event groups vs baseline)",
                            xaxis_title="Target observations after event", yaxis_title="Target response" + (" (%)" if response_label == "Percentage change" else " (absolute units)"),
                            height=430, margin={"l": 55, "r": 20, "t": 55, "b": 50},
                            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font={"color": "#D9E2F0", "size": 11},
                            legend={"orientation": "h", "y": 1.1},
                        )
                        response_fig.update_xaxes(gridcolor="rgba(154, 167, 186, 0.16)")
                        response_fig.update_yaxes(gridcolor="rgba(154, 167, 186, 0.16)")
                        st.plotly_chart(response_fig, use_container_width=True, theme=None, config={"displaylogo": False})
                        st.markdown("**How much did the event response differ from ordinary behaviour?**")
                        diff_fig = go.Figure()
                        for direction_name, color in (("Increase", "#F59E0B"), ("Decrease", "#38BDF8")):
                            group = summary[summary["primary_direction"] == direction_name].sort_values("horizon_observations")
                            if group.empty:
                                continue
                            ydiff = pd.to_numeric(group["mean_difference_vs_baseline"], errors="coerce")
                            diff_low = pd.to_numeric(group["block_bootstrap_difference_ci_low"], errors="coerce")
                            diff_high = pd.to_numeric(group["block_bootstrap_difference_ci_high"], errors="coerce")
                            diff_fig.add_trace(go.Scatter(
                                x=group["horizon_observations"], y=ydiff, mode="lines+markers", name=f"After primary {direction_name.lower()}",
                                line={"color": color, "width": 2}, marker={"size": 7},
                                error_y={"type": "data", "symmetric": False, "array": (diff_high-ydiff).clip(lower=0).fillna(0), "arrayminus": (ydiff-diff_low).clip(lower=0).fillna(0), "visible": True},
                                customdata=np.column_stack([group["events_detected"], group["realized_responses"], group["bootstrap_valid_replicates"]]),
                                hovertemplate="Horizon: %{x} target observations<br>Event mean − baseline mean: %{y:.4g}<br>Events: %{customdata[0]}<br>Realised event responses: %{customdata[1]}<br>Valid block-bootstrap samples: %{customdata[2]}<extra></extra>",
                            ))
                        diff_fig.add_hline(y=0, line_dash="dash", line_color="#F8FAFC")
                        diff_fig.update_layout(
                            title="Event response minus non-event baseline",
                            xaxis_title="Target observations after event",
                            yaxis_title="Difference in mean target response" + (" (percentage points)" if response_label == "Percentage change" else " (target units)"),
                            height=370, margin={"l": 55, "r": 20, "t": 55, "b": 45},
                            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font={"color": "#D9E2F0", "size": 11},
                            legend={"orientation": "h", "y": 1.1},
                        )
                        diff_fig.update_xaxes(gridcolor="rgba(154, 167, 186, 0.16)")
                        diff_fig.update_yaxes(gridcolor="rgba(154, 167, 186, 0.16)")
                        st.plotly_chart(diff_fig, use_container_width=True, theme=None, config={"displaylogo": False})

                        interpret_cols = st.columns(2)
                        with interpret_cols[0]:
                            interpret_horizon = st.selectbox("Explain this horizon", sorted(summary["horizon_observations"].unique().tolist()), key="event_interpret_horizon_v13", format_func=lambda n: f"{n} target observations")
                        with interpret_cols[1]:
                            interpret_direction = st.selectbox("After which event?", ["Increase", "Decrease"], key="event_interpret_direction_v13")
                        chosen_rows = summary[(summary["horizon_observations"] == interpret_horizon) & (summary["primary_direction"] == interpret_direction)]
                        if not chosen_rows.empty:
                            chosen = chosen_rows.iloc[0]
                            difference = chosen["mean_difference_vs_baseline"]
                            unit_text = "percentage points" if response_label == "Percentage change" else "target units"
                            diff_low = chosen["block_bootstrap_difference_ci_low"]
                            diff_high = chosen["block_bootstrap_difference_ci_high"]
                            base_n = int(chosen["baseline_observations"])
                            event_n = int(chosen["realized_responses"])
                            st.markdown(
                                f"**Observed:** {event_n} realised event responses from {int(chosen['events_detected'])} detected events; "
                                f"the matched non-event baseline has {base_n} realised responses. The event mean was "
                                f"{_format_value(chosen['mean_response'])}, versus baseline {_format_value(chosen['baseline_mean'])}. "
                                f"Difference: {_format_value(difference)} {unit_text}."
                            )
                            if diff_low is None or diff_high is None or pd.isna(diff_low) or pd.isna(diff_high):
                                st.warning("A dependence-aware interval could not be estimated from this sample. The graph's difference is descriptive only.")
                            elif diff_low <= 0 <= diff_high:
                                st.info(f"The approximate 95% moving-block-bootstrap interval for the difference is {_format_value(diff_low)} to {_format_value(diff_high)} {unit_text}; it includes zero. This sample does not clearly distinguish the event response from the baseline. The interval is exploratory and not adjusted for trying multiple horizons/thresholds.")
                            else:
                                direction_text = "above" if diff_low > 0 else "below"
                                st.info(f"The approximate 95% moving-block-bootstrap interval for the difference is {_format_value(diff_low)} to {_format_value(diff_high)} {unit_text}; it lies {direction_text} zero in this sample. This is a candidate association, not proof of causation, and is not adjusted for trying multiple horizons/thresholds.")
                            if event_n < 20:
                                st.warning("Fewer than 20 realised event responses: this event estimate may be unstable. This is a practical caution, not a universal validity cutoff.")
                        with st.expander("Show individual historical events and their realised responses"):
                            st.dataframe(event_rows, use_container_width=True, hide_index=True)
                        st.caption("The dashed baseline is the mean target response on matched dates when the primary-variable threshold was not crossed. Error bars on the difference graph are 95% circular moving-block-bootstrap intervals that preserve contiguous matched-date blocks, approximately accounting for local dependence and overlapping horizons. They remain exploratory; threshold/horizon selection and multiple testing are not corrected.")

        else:
            if not comparison_labels:
                st.info("Add a comparison series to test lead/lag timing.")
            else:
                compared = st.selectbox("Compare timing with", comparison_labels, key="lag_target_v14")
                lcols = st.columns(3)
                with lcols[0]:
                    window = st.select_slider("Change-formation window", options=[1, 3, 5, 10, 20, 60], value=5, format_func=lambda n: f"{n} observations", key="lag_window_v14", help="Each variable's change is calculated over this many of its observations before matching dates.")
                with lcols[1]:
                    transform_label = st.selectbox("Change measure", ["Percentage change", "Absolute change"], index=0, key="lag_transform_v14")
                with lcols[2]:
                    max_lag = st.select_slider("Test offsets up to", options=[1, 2, 3, 5, 10], value=5, format_func=lambda n: f"±{n} windows ({n * window} observations)", key="lag_max_v14", help="Lags are spaced by a full change window. This avoids testing adjacent overlapping multi-observation return windows. For a 5-observation window, offsets are 0, ±5, ±10, and so on.")
                st.caption(f"Interpretation: each series' change is measured over {window} observations. The comparison series is then aligned {window} observations earlier/later at each step, so the price intervals are adjacent rather than overlapping. Offsets count matched observations, not guaranteed calendar days.")
                result = lead_lag_statistics(
                    primary, available[compared], int(window),
                    "pct" if transform_label == "Percentage change" else "diff",
                    int(max_lag), analysis_start, analysis_end,
                )
                lag_fig = go.Figure()
                lag_low = pd.to_numeric(result["ci_low"], errors="coerce")
                lag_high = pd.to_numeric(result["ci_high"], errors="coerce")
                lag_fig.add_trace(go.Scatter(x=result["lag_observations"], y=lag_low, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip", name="Block bootstrap lower bound"))
                lag_fig.add_trace(go.Scatter(x=result["lag_observations"], y=lag_high, mode="lines", line={"width": 0}, fill="tonexty", fillcolor="rgba(56, 189, 248, 0.15)", showlegend=True, name="95% moving-block-bootstrap interval", hoverinfo="skip"))
                lag_fig.add_trace(go.Scatter(
                    x=result["lag_observations"], y=result["correlation"], mode="lines+markers",
                    line={"color": "#38BDF8", "width": 2}, marker={"size": 7}, name="Correlation by lag",
                    customdata=np.column_stack([result["observations"], lag_low, lag_high, result["familywise_adjusted_p_value"]]),
                    hovertemplate="Offset: %{x} matched observations<br>Correlation: %{y:.3f}<br>Block-bootstrap interval: %{customdata[1]:.3f} to %{customdata[2]:.3f}<br>Matched pairs: %{customdata[0]}<br>Family-wise adjusted p: %{customdata[3]:.3f}<extra></extra>",
                ))
                lag_fig.add_vline(x=0, line_dash="dash", line_color="#F59E0B")
                lag_fig.add_hline(y=0, line_dash="dot", line_color="#94A3B8")
                valid_corr = pd.to_numeric(result["correlation"], errors="coerce").dropna()
                valid_bounds = pd.concat([lag_low, lag_high, valid_corr], ignore_index=True).dropna()
                full_lag_scale = st.checkbox("Use full correlation scale (−1 to +1)", value=False, key="lag_full_scale_v14")
                if full_lag_scale or valid_bounds.empty:
                    y_range = [-1, 1]
                else:
                    low_edge, high_edge = float(valid_bounds.min()), float(valid_bounds.max())
                    pad = max((high_edge - low_edge) * 0.20, 0.03)
                    y_range = [max(-1, low_edge - pad), min(1, high_edge + pad)]
                lag_fig.update_layout(
                    title=f"Lead/lag association · {window}-observation change windows · non-overlapping tested offsets",
                    xaxis_title="Comparison-series offset in matched observations (negative = earlier, positive = later)",
                    yaxis_title="Pearson correlation", yaxis={"range": y_range}, height=430,
                    margin={"l": 60, "r": 25, "t": 65, "b": 80},
                    paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font={"color": "#D9E2F0", "size": 11},
                    legend={"orientation": "h", "y": 1.12},
                )
                lag_fig.update_xaxes(gridcolor="rgba(154, 167, 186, 0.16)", dtick=window)
                lag_fig.update_yaxes(gridcolor="rgba(154, 167, 186, 0.16)")
                st.plotly_chart(lag_fig, use_container_width=True, theme=None, config={"displaylogo": False})
                lag_table = result[["lag_observations", "correlation", "ci_low", "ci_high", "familywise_adjusted_p_value", "observations"]].rename(columns={
                    "lag_observations": "Offset (matched observations)", "correlation": "Pearson r",
                    "ci_low": "Block-bootstrap 2.5%", "ci_high": "Block-bootstrap 97.5%",
                    "familywise_adjusted_p_value": "Family-wise adjusted p", "observations": "Matched pairs",
                })
                st.dataframe(lag_table, use_container_width=True, hide_index=True)
                usable = result.dropna(subset=["correlation"])
                if not usable.empty:
                    peak = usable.loc[usable["correlation"].abs().idxmax()]
                    lag = int(peak["lag_observations"])
                    direction = "earlier" if lag < 0 else "later" if lag > 0 else "at the same matched observation"
                    peak_value = float(peak["correlation"])
                    ci_low, ci_high = peak.get("ci_low"), peak.get("ci_high")
                    adjusted_p = peak.get("familywise_adjusted_p_value")
                    if adjusted_p is None or pd.isna(adjusted_p):
                        st.warning(f"The largest absolute correlation was {peak_value:.3f} at offset {lag} ({direction}), but there were not enough valid circular shifts to assess the tested-lag family. Treat the peak as descriptive only.")
                    elif float(adjusted_p) >= 0.05:
                        st.info(f"The largest absolute correlation was {peak_value:.3f} at offset {lag} ({direction}). Its family-wise circular-shift adjusted p-value is {float(adjusted_p):.3f}, which does not meet the conventional 0.05 threshold after accounting for the search across {int(peak['tested_lags_count'])} offsets. This is not evidence of a reliable lead/lag effect.")
                    else:
                        st.warning(f"The largest absolute correlation was {peak_value:.3f} at offset {lag} ({direction}); its family-wise circular-shift adjusted p-value is {float(adjusted_p):.3f}. This is a candidate association under the circular-shift/stationarity assumptions, not proof of prediction or causation. Confirm it on a pre-specified untouched period.")
                    if ci_low is not None and ci_high is not None and not pd.isna(ci_low) and not pd.isna(ci_high):
                        st.caption(f"Moving-block-bootstrap 95% interval at this selected peak: {float(ci_low):.3f} to {float(ci_high):.3f}. Because the peak was selected from the tested offsets, use the family-wise adjusted p-value for the search-level assessment, not the pointwise interval alone.")
                if int(result["null_shift_replicates"].max()) == 0:
                    st.warning("The study period is too short relative to the requested lag range for the circular-shift test. Reduce the number of offsets or widen the study period.")
                if result["ci_low"].isna().all():
                    st.warning("The moving-block-bootstrap interval is unavailable: the sample cannot support at least three blocks long enough to cover the largest tested lag and formation window. Reduce the tested lag span or formation window, or widen the study period.")
                st.caption("Intervals use a moving-block bootstrap of paired changes to approximately preserve local serial dependence. Family-wise adjusted p-values compare each lag against the maximum absolute correlation under circular shifts, controlling the search over the tested offsets if the change process is sufficiently stationary. This is not a multiple-variable correction and not a causal test. Choose lags before inspecting results wherever possible.")

    with data_tab:
        chosen = st.selectbox("Show observations for", list(available), key="raw_series_selection")
        raw = available[chosen].copy()
        raw = raw[(raw["date"] >= pd.Timestamp(chart_start)) & (raw["date"] <= pd.Timestamp(chart_end))]
        st.dataframe(raw, use_container_width=True, hide_index=True)
        st.download_button("Download these observations (CSV)", data=raw.to_csv(index=False).encode("utf-8"), file_name="trendanalysis_observations.csv", mime="text/csv")

    st.divider()
    st.caption("Data is read using the existing TrendAnalysis data_access package and read-only OLAP connection. Financial reporting dates are not assumed to be publication dates.")


if __name__ == "__main__":
    main()
