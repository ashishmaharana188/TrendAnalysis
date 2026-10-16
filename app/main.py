from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from app.components.charts import synchronized_series_chart
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
    event_study,
    lead_lag_statistics,
    pair_statistics,
    variable_summary,
)
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
    stat_cols[2].metric("Change over chart period", _format_value(primary_stats.get("change")))
    stat_cols[3].metric("Change over chart period (%)", _format_value(primary_stats.get("percent_change"), "%"))

    chart_tab, analysis_tab, data_tab = st.tabs(["Charts", "Statistical analysis", "Raw data"])
    with chart_tab:
        st.subheader("Synchronized charts")
        st.caption("Each variable has its own panel and units. The panels share a timeline; add as many series as you need.")
        figure = synchronized_series_chart(available, chart_start, chart_end)
        if figure is None:
            st.info("No selected observations fall inside the chart history range.")
        else:
            st.plotly_chart(figure, use_container_width=True)

    with analysis_tab:
        st.subheader("What would you like to check?")
        mode = st.radio(
            "Analysis",
            ["How do variables move together?", "What happens after a rise or fall?", "Does one tend to move first?"],
            horizontal=True,
            label_visibility="collapsed",
            key="analysis_mode",
        )
        comparison_labels = [label for label in available if label != primary_label]
        if not comparison_labels:
            st.info("Add at least one comparison series to run relationship or response analysis.")
        elif mode == "How do variables move together?":
            st.markdown(f"Changes in **{primary_label}** compared with each selected variable over the same dates.")
            with st.expander("Analysis settings", expanded=False):
                window = st.select_slider("Measure change over", options=[1, 3, 5, 10, 20, 60], value=5, format_func=lambda n: f"{n} observation{'s' if n != 1 else ''}")
                transform_label = st.selectbox("Measure changes as", ["Percentage change", "Absolute change"], index=0)
            transform = "pct" if transform_label == "Percentage change" else "diff"
            rows = []
            for label in comparison_labels:
                result = pair_statistics(primary, available[label], int(window), transform, analysis_start, analysis_end)
                rows.append({"Compared with": label, "Matched observations": result["observations"], "Pearson correlation": result["pearson"], "Spearman correlation": result["spearman"], "OLS slope": result["slope"]})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("Correlations use changes, not raw price levels. They describe association, not proof of cause. Small samples and many comparisons can produce misleading patterns.")
        elif mode == "What happens after a rise or fall?":
            target_label = st.selectbox("Measure the response of", comparison_labels, key="event_target")
            with st.expander("Event settings", expanded=False):
                measurement_window = st.select_slider("Measure the main variable over", options=[1, 3, 5, 10, 20, 60], value=5, format_func=lambda n: f"{n} observation{'s' if n != 1 else ''}")
                primary_change_label = st.selectbox("Main-variable change", ["Percentage change", "Absolute change"], index=0)
                threshold = st.number_input("Count a rise/fall from", min_value=0.0, value=2.0, step=0.5, help="For percentage change, 2 means a rise of at least 2% or a fall of at least 2% across the selected window.")
                response_label = st.selectbox("Measure the target response as", ["Percentage change", "Absolute change"], index=0)
                horizons = st.multiselect("Measure the response after", [1, 2, 3, 5, 10, 20, 30, 60], default=[1, 3, 5, 20], format_func=lambda n: f"{n} observation{'s' if n != 1 else ''}")
            if not horizons:
                st.info("Select at least one response horizon.")
            else:
                summary, event_rows = event_study(
                    primary,
                    available[target_label],
                    measurement_window=int(measurement_window),
                    threshold=float(threshold),
                    primary_mode="pct" if primary_change_label == "Percentage change" else "diff",
                    response_mode="pct" if response_label == "Percentage change" else "diff",
                    horizons=horizons,
                    start_date=analysis_start,
                    end_date=analysis_end,
                )
                if summary.empty:
                    st.info("No historical events matched that rise/fall threshold in the selected study period.")
                else:
                    st.dataframe(summary, use_container_width=True, hide_index=True)
                    with st.expander("Show individual historical events"):
                        st.dataframe(event_rows, use_container_width=True, hide_index=True)
                st.caption("Responses are shown only where subsequent observations exist. Horizons count observations in the target series. Overlapping events can make results dependent; the estimates are descriptive, not causal proof.")
        else:
            compared = st.selectbox("Check timing against", comparison_labels, key="lag_target")
            with st.expander("Timing settings", expanded=False):
                window = st.select_slider("Measure change over", options=[1, 3, 5, 10, 20, 60], value=5, format_func=lambda n: f"{n} observation{'s' if n != 1 else ''}")
                transform_label = st.selectbox("Measure changes as", ["Percentage change", "Absolute change"], index=0)
                max_lag = st.select_slider("Check up to", options=[1, 3, 5, 10, 20], value=5, format_func=lambda n: f"±{n} observations")
            result = lead_lag_statistics(
                primary, available[compared], int(window),
                "pct" if transform_label == "Percentage change" else "diff",
                int(max_lag), analysis_start, analysis_end,
            )
            st.dataframe(result.rename(columns={"lag_observations": "Compared-series timing (observations)", "correlation": "Correlation", "observations": "Matched observations"}), use_container_width=True, hide_index=True)
            st.caption("Negative timing means the comparison series tends to move earlier in the tested alignment; positive timing means later. This is exploratory and not proof that one variable causes the other.")

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
