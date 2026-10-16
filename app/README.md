# TrendAnalysis `/app` — simplified research workspace

This folder adds a Streamlit/Plotly research UI over the existing repository. It does not replace `data_access/`, refactor the existing `analysis/` package, alter the upstream ETL project, or run the Phase 5.8/6.3 prediction workflow.

## User experience

- **Add a variable:** select a plain-language data type, choose an available series, and accept its sensible default measure. The measure selector is optional.
- **Compare as many series as needed:** add any number of variables. Each receives its own chart panel, and all panels share a historical timeline. The first series is the main variable; use `Make main` to change it.
- **Choose a chart period simply:** choose a recent number of days, weeks, months or years, or enter specific dates manually.
- **Keep statistics simple by default:** statistical analysis uses the chart period unless the user chooses a separate study period.
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
python -m unittest app.tests.test_statistics_service app.tests.test_time_ranges -v
```

These tests do not require a live OLAP connection. The full terminal still needs a local test against the actual DuckDB file to validate every source and series.

## Interpretation and data handling

- Correlations and lead/lag results describe observed associations, not proof of causation.
- Event-study summaries report only realized forward outcomes that exist in the loaded data.
- Observation-count horizons mean observations in the target series; they correspond to trading sessions for daily market series, but not necessarily for quarterly/annual financial statements.
- Financial statement `ReportDate` values are accounting period ends, not verified publication dates.
- The upstream OLAP connection remains read-only.

## Updating from the earlier app overlay

This update replaces files inside `app/` only. Before extracting it over an existing `app/` folder, keep a backup or inspect `git status --short`. No files outside `app/` are included in this overlay.
