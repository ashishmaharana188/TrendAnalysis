# TrendAnalysis — Phase 1 Context / Hydration

## Purpose

This document records the completed **Phase 1: OLAP/Data Audit** for the TrendAnalysis project.

Use this context in later phases so requirements and data decisions are not re-interviewed or re-derived.

---

# 1. Project Boundary

TrendAnalysis is a separate project that reads the OLAP database produced by:

`company_financial_data_ETL`

TrendAnalysis must:

- read OLAP data
- not refactor or relocate the ETL pipeline
- not become a duplicate ETL/scraper
- reuse finalized OLAP calculations/views where appropriate
- remain separate from `NewsAnalysis`

`ScreenerFundamentalScrapper` remains responsible for candidate/company-universe discovery.

---

# 2. OLAP Connection

The existing ETL uses:

`market_data.duckdb`

TrendAnalysis should connect to the same database in **read-only mode**.

The source database is not to be copied into TrendAnalysis merely for analysis.

---

# 3. Current Company Universe

`market_metadata` currently contains:

- 101 equity companies
- 9 sectors
- 47 industries

There are:

- 99 companies with current cash/equity market history
- 2 metadata companies without current cash/equity market history

The two without current cash/equity history in the audited coverage are:

- `SNDK`
- `MU`

`market_metadata` therefore remains the authoritative company → industry → sector mapping for the current universe.

Two companies currently have missing Sector/Industry classification and require handling as data-quality limitations where group analysis depends on that mapping.

---

# 4. Primary Market Data Source

For the TrendAnalysis company universe, the primary Indian market source is:

`unified_market_master`

For regular equity/cash market analysis, the relevant combination is:

- `InstrumentType = 'CASH'`
- `Exchange_Series = 'EQ'`

Current audited coverage:

- First date: `2019-06-27`
- Latest date: `2026-10-01`

The table contains approximately 143.6M total rows across instruments.

The market master contains multiple instrument types including:

- CASH
- STO
- STF
- FUTSTK
- FUTIDX
- OPTSTK
- OPTIDX
- FUTCOM
- OPTCOM
- other instrument categories

Do not treat the entire market master as one homogeneous stock dataset.

---

# 5. Secondary Global Asset Dataset

`global_assets_daily` is a separate data source.

It has:

- long historical coverage
- OHLCV-style asset data
- historical observations for various equity/global assets
- current audited coverage only through `2026-07-07`

Example observed company history can extend much further back than the cash/equity series in `unified_market_master`.

Decision:

**Do not blindly splice `global_assets_daily` and `unified_market_master` together.**

Treat them as separate sources with different semantics/coverage until a later technical design explicitly establishes when and how either should be used.

---

# 6. Macro Data

`macro_daily_ledger` currently contains 11 identifiable macro variables:

- `Brent_Crude`
- `Broad_Commodity`
- `India_10Y_Yield`
- `India_CPI`
- `India_VIX`
- `Nifty_50`
- `USD_INR`
- `US_10Y_Yield`
- `US_Dollar_Index`
- `US_VIX`
- `Yield_Spread`

Coverage differs by variable.

Examples:

- Brent Crude: `2007-07-30 → 2026-07-07`
- Nifty 50: `2007-09-17 → 2026-07-07`
- USD/INR: `2003-12-01 → 2026-07-07`
- US 10Y: `1962-01-02 → 2026-07-07`
- US Dollar Index: `1971-01-04 → 2026-07-07`
- US VIX: `1990-01-02 → 2026-07-07`

Important:

The current database snapshot has macro/global daily data ending earlier than the Indian market feed.

This is a **coverage/freshness fact**, not automatically a project defect. Later technical design must account for differing data availability.

---

# 7. Institutional and Trade Data

`institutional_ledger`

- Coverage: `2015-01-01 → 2026-10-01`
- Approximately 14,495 rows

`trade_events_ledger`

- Coverage: `2015-01-01 → 2026-10-01`
- Approximately 164,452 rows

These are candidate TrendAnalysis domains where applicable.

They should be used through their natural semantics rather than forced into ordinary equity OHLC features.

---

# 8. Materialized Views

Existing analytical materialized views include:

- `mv_unified_market_matrix`
- `mv_options_aggregates`
- `mv_spot_futures_basis`
- `mv_institutional_flow`

The current audit snapshot showed these views lagging behind the base tables.

This is **not considered a project data defect**.

The user confirmed that the materialized tables have simply not been refreshed yet and will later contain data through October.

Therefore:

> In TrendAnalysis design, assume the materialized views are intended to be refreshed through the current ETL/data horizon.

Do not redesign the project around the stale snapshot.

---

# 9. Market Feature Availability

The audited cash/equity subset of `unified_market_master` showed approximately:

- OHLC present for ~88.7% of rows
- Volume present for ~88.7%
- Delivery quantity present for ~88.7%
- Delivery percentage present for ~88.7%

The following were not populated on these cash/equity rows:

- Open Interest
- Change in Open Interest
- Short Volume

This does **not** mean these features are globally unavailable.

They belong to the appropriate derivatives/instrument/materialized-view context.

Implementation rule:

> Do not expect every variable to contain every possible market field.

Each variable/domain must use only the attributes that actually exist for that source.

---

# 10. Financial Data

Available financial tables include:

### Quarterly

- `quarterly_income_statement`
- `quarterly_balance_sheet`
- `quarterly_cash_flow`

### Annual

- `yearly_income_statement`
- `yearly_balance_sheet`
- `yearly_cash_flow`
- `yearly_indirect_cash_flow`

Current overall date ranges include:

- Quarterly income: `2006-02-28 → 2026-06-30`
- Yearly income: `2006-03-31 → 2026-03-31`
- Yearly balance sheet: `2001-06-30 → 2026-03-31`
- Yearly cash flow: `2001-06-30 → 2026-03-31`
- Yearly indirect cash flow: `2001-06-30 → 2026-03-31`

Financial coverage is highly uneven by company.

Examples:

- many companies have about 13 recent quarterly income observations
- `COLPAL` has no current quarterly income series and only old annual observations
- `HONAUT` has no income-statement history in the audited result
- `TIMKEN` has only 3 quarterly income observations
- some companies have historical annual data but weak/current quarterly coverage
- some newer companies naturally have shorter history

Therefore:

> Financial history must be treated as variable/target-specific and cannot be assumed complete.

---

# 11. Financial Source

The quarterly income statement is dominated by:

`DataSource = 'screener'`

Other sources include:

- `vantage`
- `yfinance`

Observed source-level coverage:

- Screener: 1,245 rows
- Vantage: 167 rows
- yfinance: 11 rows

The system should use the finalized OLAP financial data and not build another financial scraper unless a specific data gap is demonstrated.

---

# 12. Critical Financial Date Rule

This is a permanent requirement.

`ReportDate` in the financial tables is:

> **the accounting period-end date**

It is **not** a publication/availability timestamp.

Example:

`ReportDate = 2026-03-31`

does not mean:

`the market knew the March 31 financial information on March 31`.

Therefore, historical backtesting must not automatically treat accounting period-end as information availability.

TrendAnalysis must preserve the distinction between:

- accounting/reporting date
- information-available-as-of date

If an actual publication/availability date is not available, the technical design must explicitly handle that limitation instead of inventing one.

This is a core anti-look-ahead requirement.

---

# 13. No-Look-Ahead Rule

For every historical prediction date `T`:

```text
Only information legitimately available before T
                    ↓
Build state
                    ↓
Generate prediction
                    ↓
Observe future outcome
```

For ordinary daily market data:

If checking October 3:

- data through October 2 is usable
- October 3+ information is not usable

This rule applies to:

- features
- financials
- relationship discovery
- feature selection
- similarity learning
- model fitting
- probability calibration

during historical evaluation.

---

# 14. Company Market History

The audited company-level cash/equity coverage in `unified_market_master` is mostly strong.

Most of the 101-company universe has approximately the full available market window beginning around:

`2019-06-27`

Some companies have later starts.

Examples of shorter histories:

- `TATATECH`
- `ZAGGLE`
- `WAAREEENER`
- `MOSCHIP`
- `HBLENGINE`

One audited result also showed `SNDK` and `MU` with no matching cash/equity history in the selected universe.

This reinforces:

> The engine must determine viable historical windows per target rather than require one universal start date.

---

# 15. Group Structure

TrendAnalysis preserves:

```text
Company
├── Financials
├── Market behaviour
└── Industry
    ├── Financials
    ├── Market behaviour
    └── Sector
        ├── Financials
        ├── Market behaviour
        └── Macro / Global
```

Industry/Sector analysis may use:

- constituent-company aggregates
- group breadth
- group financial state
- group market state

No complete-market prediction is required.

---

# 16. Phase 1 Data Decisions

The following are now considered settled:

### Primary sources

```text
Indian company market
    → unified_market_master

Macro
    → macro_daily_ledger

Global
    → global_assets_daily

Institutional
    → institutional_ledger / materialized analytical views

Trade events
    → trade_events_ledger

Financials
    → existing OLAP financial tables
```

### Source handling

Do not blindly combine different sources with different semantics.

### Missingness

If data is genuinely not applicable:

- do not flag it

If the target is eligible but required data is missing:

- flag limitation
- mark analysis `LIMITED`
- continue with available evidence

### Current-vs-historical freshness

Different domains can have different latest dates.

The model must respect each source's actual available-as-of date.

---

# 17. Implications for Later Phases

Phase 2 and later must inherit these constraints.

### Data Access Layer

Must provide domain-specific retrieval without hiding source semantics.

### Temporal Engine

Must enforce information availability and source-specific cutoff handling.

### State Engine

Must work with variable-specific available fields.

### Group Engine

Must use the company → industry → sector hierarchy.

### Outcome Engine

Must use market dates and future windows without financial look-ahead.

### Relationship Engine

Must discover usefulness empirically rather than encode manual economic relationships.

### Backtesting Engine

Must adapt to target-specific history and data availability.

---

# 18. What Phase 1 Did NOT Decide

Do not treat these as requirements already chosen:

- specific ML algorithm
- exact similarity metric
- exact normalization technique
- exact feature-selection algorithm
- exact interaction-search algorithm
- exact probability-calibration technique
- exact conviction thresholds
- exact dashboard/UI implementation
- exact backtest scoring/report format

These are **technical/model-design decisions** to be made after the data architecture is established and validated.

Do not re-ask the user to manually specify these unless actual data reveals a genuine requirements conflict.

---

# 19. Phase 1 Status

**STATUS: COMPLETE**

The next phase is:

```text
Phase 2
Data Access Layer
```

Its purpose is to create a clean, read-only abstraction over the OLAP sources so all later TrendAnalysis components consume data consistently.

Do not start model development before the data-access and temporal boundaries are established.
