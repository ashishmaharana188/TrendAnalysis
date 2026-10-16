# TrendAnalysis `/app` — simplified research workspace

This folder adds a Streamlit/Plotly research UI over the existing repository. It does not replace `data_access/`, refactor the existing `analysis/` package, alter the upstream ETL project, or run the Phase 5.8/6.3 prediction workflow.

## User experience

- **Add a variable:** select a plain-language data type, choose an available series, and accept its sensible default measure. The measure selector is optional.
- **Compare as many series as needed:** add any number of variables. Charts are arranged side-by-side, with either two or three panels per row. Each panel has its own value axis, while the date axes are linked.
- **Choose a chart period simply:** choose a recent number of days, weeks, months or years, or enter specific dates manually.
- **Keep statistics simple by default:** statistical analysis uses the chart period unless the user chooses a separate study period.
- **Terminal-style chart interaction:** a fixed top-right readout shows the active series, date, OHLC when available, and cursor value. Hover moves transient linked crosshair guides across all panels; clicking locks a plus-shaped crosshair that can be dragged. Escape or the header button unlocks it.
- **Choose one research question at a time:** compare same-period movements, inspect responses after a rise/fall, or explore lead/lag timing. Less-common settings are hidden under small settings expanders.
- **Inspect and export observations:** view the selected series' raw observations and download them as CSV.

## Run from the repository root

Use your existing project virtual environment:

```bash
python -m pip install -r app/requirements.txt
python -m streamlit run app/main.py
```

If the database is not at the path already configured in `data_access/db.py`, set `OLAP_DB_PATH` to the ETL DuckDB file.

Git Bash:

```bash
export OLAP_DB_PATH="/absolute/path/to/market_data.duckdb"
python -m streamlit run app/main.py
```

PowerShell:

```powershell
$env:OLAP_DB_PATH = "C:\absolute\path\to\market_data.duckdb"
python -m streamlit run app/main.py
```

## Tests

```bash
python -m unittest app.tests.test_statistics_service app.tests.test_time_ranges app.tests.test_charts app.tests.test_indicators -v
```

These tests do not require a live OLAP connection. The full terminal still needs a local test against the actual DuckDB file to validate every source and series.

## Distribution and statistical research (v11)

- **Distribution & variability** now defaults to all selected variables and draws a separate panel per variable. Choose observed levels, absolute changes or percentage changes. Use Histogram, Box plot or Cumulative percentile; each includes defined axes and a selected-series interpretation.
- The **Relationships** study includes a multi-series correlation heatmap with pairwise observation counts, a selectable X/Y scatter plot with a linear fit, an approximate Pearson interval and a rolling-correlation view. Pairwise calculations use exact matched dates.
- **Response after events** compares the target response after thresholded increases/decreases with a matched non-event baseline and shows response differences over the selected horizons.
- **Lead/lag timing** is presented as a graph of correlation versus tested lag, with sign convention and search limitations described.
- See [`STATISTICAL_GUIDE.md`](STATISTICAL_GUIDE.md) for definitions, interpretation notes and methodological limitations.

## Interpretation and data handling

- Correlations and lead/lag results describe observed associations, not proof of causation.
- Event-study summaries report only realized forward outcomes that exist in the loaded data.
- Observation-count horizons mean observations in the target series; they correspond to trading sessions for daily market series, but not necessarily for quarterly/annual financial statements.
- Financial statement `ReportDate` values are accounting period ends, not verified publication dates.
- The upstream OLAP connection remains read-only.

## Updating from the earlier app overlay

This update replaces files inside `app/` only. Before extracting it over an existing `app/` folder, keep a backup or inspect `git status --short`. No files outside `app/` are included in this overlay.

## Data freshness and plotted-series definitions

- Market-price loading in the app checks the canonical ticker and its `.NS` alias, then chooses the alias with the latest non-null closing observation. This fix is scoped to `/app`; it does not change `data_access/market.py`, which the existing backtest/prediction code also uses.
- Series cache lifetime is 60 seconds. Use **Refresh available data** to clear cached catalogues and series immediately.
- The workspace's **Data coverage and how each series is built** panel shows the first/latest loaded observations, row counts in the chart range, and the construction rule for each series. Recent daily series with no observations for more than five calendar days are flagged for inspection.
- Options PCR is calculated per date as the sum of put OI divided by the sum of call OI across expiry rows returned for that ticker/date. Volume PCR uses summed put and call volume.
- Futures-basis panels represent the expiry with the highest recorded open interest on each date. The represented expiry may change from date to date.
- Trading-activity fields come from the ETL-produced `mv_unified_market_matrix` and are plotted as stored values.
- Institutional selection is by client category. Raw ledger measures (cash, long/short positions and contract counts) come from `institutional_ledger`; derived net/change measures come from `mv_institutional_flow`.
- Options, futures and trading-activity pickers list instruments from their own source views rather than reusing the general market-price catalogue.

## Terminal-style linked chart grid

The Charts tab renders all selected series inside one Plotly figure arranged as two or three columns per row. A browser-side pointer handler updates one vertical date guide across every panel and horizontal value guides for each series. The hover readout is in a fixed header, not a floating tooltip. Price sources that provide open/high/low/close retain those fields as trace `customdata`; series without candles show the selected value instead. `app/assets/plotly.min.js` is bundled locally and corresponds to `plotly==6.5.2`; do not update the Python Plotly version without refreshing that asset and rerunning the chart tests.

## Chart layout and contrast fix

- A single selected series spans the full available chart width; it does not reserve an empty second column.
- When the final chart row is incomplete, its remaining chart panels expand across that row instead of leaving a blank subplot area.
- Series use explicit high-contrast line colors against the dark terminal background. Chart dimensions are responsive within the Streamlit component iframe.
- Chart tests cover one-series full width and expanded incomplete rows.


## Trading-terminal chart navigation

Chart interaction is configured in the chart iframe:

- **Hover inside a plot:** moves a transient crosshair and updates the top-right date/price readout.
- **Click inside a plot:** locks a plus-shaped crosshair. While locked, drag inside any chart to move the linked date line and the active panel's horizontal value guide.
- **Escape / Unlock crosshair:** returns to hover mode so plot dragging pans the chart again.
- **Drag inside an unlocked plot:** pans the shared date range and the active panel's value range.
- **Drag inside a plot:** pan the chart. Horizontal movement shifts the shared date range; vertical movement shifts the active panel's value range.
- **Mouse wheel inside a plot:** zooms the shared date range around the date under the pointer.
- **Shift + mouse wheel inside a plot:** zooms the active panel's value range around the value under the pointer.
- **Mouse wheel over the bottom time axis:** zooms the shared date range. The cursor position anchors the zoom.
- **Mouse wheel over the left value axis:** expands/compresses only that panel's value range.
- **Drag the time axis:** drag horizontally to shift the timeline; drag vertically to zoom the timeline.
- **Drag the value axis:** drag vertically to zoom the value range; drag horizontally to shift the visible value range. Neither action changes the timeline.

All navigation is handled in the browser iframe. Date-range changes are applied to every linked x-axis while each panel retains its own y-axis range. When crosshair lock is active, dragging moves the crosshair rather than the chart; unlock it to resume drag-to-pan navigation. The top-right header shows OHLC for sources that provide candle fields and a value-only readout for PCR, flow, financial and other non-OHLC data.

## Terminal layout, indicators and first-to-last change

- The chart terminal runs in a fixed-height embedded area. The OHLC/date header stays at the top while the chart grid scrolls independently inside the terminal.
- **Full-screen terminal** opens all chart panels as a full-screen workspace. Each panel also has an **Expand** action to open that one chart in a full-screen view.
- Chart titles are explicit annotations per real panel, including an incomplete final row.
- **Moving averages and bands apply to every selected chart series.** Each panel calculates its own EMA/SMA/Bollinger values from that panel's underlying series. Windows count observations, not calendar days. Bollinger bands use a 20-observation mean and ±2 rolling standard deviations.
- **First-to-last change** is the final valid observation minus the first valid observation within the selected chart date range. It is not the sum of daily changes. Percentage change is `(last / first - 1) * 100` only when the starting value is positive. Percentage-change observations also omit transitions whose prior value is zero or negative.
- The current **Distribution & variability** and **Statistical research** workflows are described in the v11 section above and in `STATISTICAL_GUIDE.md`.

## Statistical interpretation improvements (v13)

- Relationship heatmap axis labels are shortened into readable two-line labels; hovering still exposes the full series names and matched-observation count.
- Pairwise and rolling-correlation intervals use a Bartlett-style effective sample size to discount some autocorrelation. These intervals are approximations, not complete corrections for non-stationarity, outliers, or multiple comparisons.
- Event studies include a separate event-minus-baseline graph. Its uncertainty intervals use a circular moving-block bootstrap over the chronological matched-observation sequence to preserve some local dependence and overlapping forward-horizon effects. The intervals are still exploratory and are not corrected for selecting multiple thresholds/horizons.
- Event interpretation is based on a user-selected horizon and event direction, rather than automatically highlighting whichever tested combination has the largest difference. This helps avoid presenting the maximum selected result as if it had been specified in advance.
- Lead/lag offsets are spaced by a full change-formation window, reducing direct overlap between adjacent tested change intervals. Confidence intervals use a moving-block bootstrap of paired changes; family-wise adjusted p-values use a circular-shift maximum-statistic test across the tested offsets. These results assume a sufficiently stationary change process and do not correct searches across multiple variables, windows or study periods.

## Full app test command

```bash
python -m unittest app.tests.test_statistics_service app.tests.test_time_ranges app.tests.test_charts app.tests.test_indicators -v
```

The statistical interval methods are deliberately labelled approximate. The lead/lag procedure controls selection over the tested offsets under a circular-shift/stationarity assumption, but does not correct for trying multiple series, windows or periods. These methods do not establish causation or replace pre-specified validation on untouched chronological data.


## Statistical validity updates (v14)

- Lead/lag shifts comparison returns only at multiples of the selected change window. For example, a 5-observation formation window tests offsets 0, ±5, ±10, etc., rather than every overlapping daily shift. Offsets are counts of exact matched observations, not guaranteed calendar days.
- Lead/lag confidence intervals use a moving-block bootstrap of the paired change sequence. Family-wise adjusted p-values compare each observed correlation with the maximum absolute correlation across all tested offsets in circular-shift null samples. If the study is too short relative to the offset range, the app reports that the inferential test is unavailable.
- Pairwise and rolling correlations now use a Bartlett-style effective sample size estimated from autocorrelation products across short lags, extending the range to reflect overlapping return windows when enough observations exist. This is an approximate adjustment, not an exact correction.
- Event analysis distinguishes the primary-variable formation window from the target response horizon and warns that trying multiple thresholds/horizons and then selecting a favourable result is a multiple-testing problem.
- The scatter view provides a table of unusual matched observations based on robust median/MAD deviations. These points are flagged for inspection, never silently removed.
