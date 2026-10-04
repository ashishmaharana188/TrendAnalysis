# TrendAnalysis — Phase 3 Context / Hydration

## Status

Phase 3 is the temporal and descriptive state-construction layer of TrendAnalysis.

Completed and validated:
- Phase 3.1 — Analysis Calendar / Parameter Contract
- Phase 3.2 — Observation Window Builder
- Phase 3.3 — Generic Variable State Engine
- Phase 3.4 — Financial State Engine
- Phase 3.5 — Company Market State Engine

Phase 3.6 — Industry / Sector State Engine has been designed/implemented, but its test validation is still pending as of this handoff.

Next major phase after 3.6 validation: Phase 4 — Relationship Discovery Engine.

---

# 1. Purpose of Phase 3

Phase 3 converts validated OLAP repository outputs into a time-consistent descriptive representation of the current company, industry, sector, and macro/global environment.

Phase 3 does NOT:
- discover predictive relationships
- calculate future probabilities
- fit prediction models
- calibrate UP/SIDEWAYS/DOWN outcomes
- perform final backtesting
- assign permanent indicator weights
- assign manual economic meanings to variables

Those belong to later phases.

Core pipeline:

```text
User parameters
    ↓
Temporal contract
    ↓
Observation window
    ↓
Variable states
    ↓
Financial states
    ↓
Company market state
    ↓
Industry / Sector state
    ↓
Phase 4 Relationship Discovery
```

---

# 2. Locked User Parameters

The analysis configuration is driven by:

- analysis_date
- analysis_timeframe
- holding_period_months
- benchmark
- entry_mode

Holding period is user-defined from 0.1 to 12.0 calendar months.

Analysis timeframe is independent of holding period. The analysis timeframe controls how much historical/current state information is examined. The holding period controls the future outcome horizon used later for prediction and backtesting.

Typical analysis timeframes span roughly 1 week to 5 years, with useful standard intervals such as:

```text
1W, 2W, 1M, 3M, 6M, 9M, 1Y, 18M, 2Y, 3Y, 5Y
```

The user chooses the benchmark, such as Nifty 50 or a relevant sector index.

Absolute stock return/trend is the primary target later. Relative performance versus the selected benchmark is an additional comparison.

---

# 3. Temporal Rules

For analysis date T, ordinary daily market data must not use future information.

Example:

```text
Analysis date = 2026-10-03
Usable ordinary market data = through the latest completed observation before the cutoff, normally 2026-10-02
```

Two backtest entry modes are supported:

1. next-trading-day entry
   - prediction state is built from data before the analysis date/cutoff
   - next available trading day's close is the entry/reference price

2. latest-known-data entry
   - latest close available at the cutoff is the entry/reference price

These are later used by the outcome/backtesting layer.

Financial data has an additional temporal limitation:

```text
financial ReportDate = accounting period end
```

It is NOT guaranteed to be the publication/availability date. TrendAnalysis must not assume that a quarter-end value was known to the market on the quarter-end date.

Current OLAP schema does not store verified financial publication/availability timestamps. This limitation must remain explicit in historical backtesting until solved or conservatively handled.

---

# 4. Observation Window — Phase 3.2

`analysis/window.py` builds a reusable `ObservationWindow` containing:

- config
- company metadata
- selected company market history
- full company market history through cutoff
- benchmark history and source
- macro histories
- timeframe-driven quarterly financial histories
- full-history yearly financial baselines
- limitations

Current financial rule:

```text
1M  → latest available quarter as basis
6M  → latest 2 quarters / half-year
1Y  → roughly 4 quarters / annual
3Y  → roughly 3 years
5Y  → roughly 5 years
```

The engine must not invent monthly financial statements.

Yearly financial data is the mandatory baseline for:
- income statement
- balance sheet
- cash flow
- indirect cash flow

Quarterly income and balance-sheet history are optional and may produce a limitation when eligible but unavailable.

Quarterly cash flow is explicitly optional and is not itself a missing-data limitation.

The window layer retains limitations rather than silently filling missing data.

---

# 5. Important Data-Source Decision in Current Phase 3 Implementation

The current Phase 3 implementation uses:

```text
global_assets_daily
```

for regular company daily market history used by `market.py` and the company market state engine.

The repository result is aliased to:

```text
ticker
report_date
asset_class
open
high
low
close
volume
```

Do NOT reintroduce delivery percentage into the Phase 3 company market-state contract when using this source. Delivery percentage belongs to the richer `unified_market_master` context, while the current company market-state implementation uses the OHLCV fields from `global_assets_daily`.

This is an implementation-level Phase 3 decision and supersedes any earlier ambiguity between the audited source alternatives for this specific state engine.

---

# 6. Ticker Lookup Architecture

External ticker formats must be tolerated without mutating the stored OLAP identifier.

Current helper:

```python
normalize_ticker(ticker: str) -> list[str]
```

Examples:

```text
RELIANCE      → ["RELIANCE", "RELIANCE.NS"]
RELIANCE.NS   → ["RELIANCE", "RELIANCE.NS"]
```

The canonical OLAP/Screener identifier remains `RELIANCE` where applicable.

The fallback is performed at data-access boundaries:

```text
input ticker
    ↓
normalize_ticker()
    ↓
try canonical ticker
    ↓ if not found
try .NS ticker
```

Do not add a second ticker helper or normalize a candidate list again.

`window.py` passes the original ticker string. Repository/data-access functions perform candidate resolution.

---

# 7. Phase 3.3 — Generic Variable State Engine

Core file:

```text
analysis/state.py
```

The generic state engine describes a variable relative to its own history.

A variable state contains concepts such as:

- current value
- selected-timeframe starting value
- absolute change
- relative change percentage
- historical percentile
- historical minimum
- historical maximum
- historical median
- recent direction
- direction strength
- state label
- selected observation count
- historical observation count
- limitation status

Current descriptive state form is broadly:

```text
Direction / Position
```

where position is based on the variable's own historical distribution, e.g.:

```text
Low / Mid / High
```

and direction can be:

```text
Rising / Falling / Stable / Unknown
```

The current implementation uses data-derived direction strength rather than manually assigned economic meaning.

Batch daily variable-state construction is supported.

Example concept:

```text
Crude
Current: current value
6M change: observed change
Historical position: High
Direction: Rising
State: Rising / High
```

Important separation:

```text
VARIABLE STATE
= what the variable is doing relative to its own history

RELATIONSHIP EFFECT
= what that state historically meant for a company / industry / sector outcome
```

Phase 3 only establishes the first. Phase 4 discovers the second.

---

# 8. Phase 3.4 — Financial State Engine

Core file:

```text
analysis/financial_state.py
```

Financial states operate at the natural reporting cadence. They are not converted into artificial daily observations.

The financial engine combines:

```text
recent/timeframe-driven quarterly lens
        +
full-history yearly baseline
```

Financial metric categories currently include:

### Income / flow
- TotalRevenue
- GrossProfit
- OperatingIncome
- NetIncome

### Balance / stock
- TotalAssets
- CurrentAssets
- CurrentLiabilities
- LongTermDebtAndCapitalLeaseObligation
- StockholdersEquity

### Cash flow
- CashFromOperations
- FixedAssetPurchases
- NetBorrowing
- EndingCashBalance
- TotalOperatingCashFlow
- TotalInvestingCashFlow
- TotalFinancingCashFlow
- NetChangeInCash
- TreasuryOpacityRatio

Flow metrics are aggregated over the relevant recent period where appropriate.

Stock metrics use the latest relevant observation rather than summing balances.

Financial trend should consider:

1. selected-period performance
2. previous equivalent-period comparison
3. position relative to full company history

Quarterly data can be absent. Yearly baseline remains mandatory.

The financial engine preserves missingness and limitations.

---

# 9. Phase 3.5 — Company Market State

Core file:

```text
analysis/market_state.py
```

Validated successfully on 2026-10-04.

The company market state consumes repository tables supplied by the observation window. It does not query DuckDB directly.

Current market-state contract for `global_assets_daily` is exactly:

```text
price
volume
daily_range
volatility_20d
```

### Price

Uses close values.

### Volume

Uses daily volume values.

### Daily range

Derived as:

```text
(high - low) / abs(close) × 100
```

### 20-day volatility

Derived from daily percentage returns using a rolling 20-observation population standard deviation.

Each metric receives descriptive state information analogous to the generic variable-state structure:

```text
current value
selected timeframe change
historical position
historical range
recent direction
state
observation counts
limitations
```

The engine validates that the market table corresponds to the target ticker, accepting the `.NS` alias when the OLAP result uses the canonical ticker.

The previous test expectation of `delivery_percentage` was stale and was intentionally removed because it does not belong to the current OHLCV source contract.

---

# 10. Phase 3.6 — Industry / Sector State

Core planned file:

```text
analysis/group_state.py
```

Test:

```text
tests/test_group_state.py
```

Purpose:

Build first-class Industry and Sector descriptive states from constituent companies using the current `market_metadata` company → Industry → Sector hierarchy.

Current `market_metadata` remains the authoritative current-universe mapping.

### Group hierarchy

```text
Company
  ↓
Industry
  ↓
Sector
```

### Group market construction

The group state aggregates constituent company market histories into a group-level daily series.

Current implementation uses an equal-weighted average of constituent daily percentage returns.

From that series it constructs:

- group index
- daily return state
- 20-day volatility state

### Breadth

The group also exposes constituent period breadth:

```text
UP %
SIDEWAYS %
DOWN %
```

The current Phase 3.6 implementation classifies period-return signs into those three categories for breadth reporting. This is descriptive group participation information; it is not yet the final predictive outcome-labeling system.

Breadth should sum to approximately 100% over the classified constituent universe.

### Benchmark comparison

For the selected analysis timeframe, the group computes:

```text
group return
benchmark return
group relative return = group return - benchmark return
```

This retains absolute group behavior and relative benchmark behavior separately.

### Financial aggregation

The group engine has a hook for aggregating already-built company financial states. It must not recalculate the underlying financial metrics independently.

### Missing constituents

Constituents without usable market history are tracked explicitly.

If the group has insufficient valid constituents, the group state carries limitations but does not fabricate values.

### Historical membership limitation

Current `market_metadata` gives current constituent membership, not a verified historical membership timeline.

Therefore Phase 3.6 historical group series should be understood as:

```text
historical behavior of the current constituent universe
```

not a reconstructed historical index with historically accurate constituent membership.

This limitation must remain visible until historical membership data exists.

---

# 11. Missing-Data Rule Across Phase 3

Permanent rule:

```text
Not applicable
    → do not flag

Eligible but missing
    → flag limitation
    → mark analysis LIMITED
    → continue with available evidence
```

Do not:
- convert missing values to zero
- forward-fill without an explicit future design
- fabricate observations
- suppress missingness
- interpret absence as a directional signal

Different domains may naturally have different historical coverage.

---

# 12. Structural Evidence Hierarchy Established in Phase 3

Phase 3 preserves the following hierarchy:

```text
Raw OLAP fields
      ↓
Variable state / trend
      ↓
Domain or group state
      ↓
Cross-domain interaction
      ↓
Prediction
```

At the domain/group level, the project preserves:

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

Financials and market behavior remain equally legitimate evidence sources. Phase 3 does not assign permanent weights.

---

# 13. Examples of Valid Later Relationships

The Phase 3 output is designed to support structurally valid relationships such as:

```text
company financials × company market
company financials × industry market
industry financials × industry market
industry market × sector market
industry market × macro/global variables
sector market × benchmark
industry + sector + macro/global
```

Do NOT hard-code assumptions such as:

```text
Crude → Energy
DXY → IT
Copper → Metals
```

Phase 4 must discover whether a relationship is useful from historical evidence.

---

# 14. Phase 3 Output Is Descriptive, Not Predictive

A state such as:

```text
Crude = Rising / High
```

means only that the crude variable is currently rising and historically high according to the descriptive state engine.

It does NOT mean:

```text
Energy stock = UP
```

Likewise:

```text
Industry = Rising / High
```

does not itself generate a probability.

Prediction happens only after historical conditional relationships and outcomes are learned in later phases.

---

# 15. Testing Status

### Phase 3.5

Validated with:

```text
RELIANCE.NS
analysis_date = 2026-10-03
analysis_timeframe = 6M
holding_period = 1.0 month
benchmark = Nifty_50
entry_mode = next_trading_day
```

Ticker lookup required an alias mechanism because external/Yahoo-style tickers can contain `.NS` while OLAP/Screener may use the base ticker.

The final Phase 3.5 required states are:

```text
price
volume
daily_range
volatility_20d
```

### Phase 3.6

Implementation supplied:
- Industry group state
- Sector group state
- equal-weighted constituent market return aggregation
- group index
- daily return state
- 20-day volatility state
- constituent breadth
- benchmark-relative group return
- financial-state aggregation hook
- missing-constituent limitations
- current-membership historical limitation

Validation test has not yet been reported as passing.

---

# 16. Files / Components Relevant to Future Hydration

Current architecture includes:

```text
data_access/
    ticker.py
    metadata.py
    market.py
    financials.py
    macro_global.py
    query.py
    db.py

analysis/
    config.py
    window.py
    state.py
    financial_state.py
    market_state.py
    group_state.py   # Phase 3.6

tests/
    test_market_state.py
    test_group_state.py  # Phase 3.6 validation
```

Important current ticker helper:

```text
data_access/ticker.py
```

Important company market repository:

```text
data_access/market.py
```

Important temporal window:

```text
analysis/window.py
```

---

# 17. Phase 4 Handoff

Once Phase 3.6 validation passes, move to:

```text
PHASE 4 — RELATIONSHIP DISCOVERY ENGINE
```

Phase 4 must consume Phase 3 outputs rather than raw SQL.

Its job is to discover historically useful relationships between states and subsequent outcomes.

Two later methods are required:

### Method A — Similar Historical States

```text
Current state
    ↓
historically comparable states
    ↓
subsequent outcomes
    ↓
empirical UP / SIDEWAYS / DOWN distribution
```

Similarity must be adaptive, not based on arbitrary fixed tolerances.

### Method B — Historical Distribution + Current-Condition Weighting

```text
all relevant historical outcomes
    ↓
historical baseline distribution
    ↓
current conditions
    ↓
adaptive relevance weighting
    ↓
weighted outcome distribution
```

Both methods must remain distinct.

Historical learning windows must be adaptive based on available history, observations, stability, regime changes, selected timeframe, holding period, and usefulness.

Combination discovery should proceed:

```text
individual variables
    ↓
pairs
    ↓
3-variable combinations
    ↓
higher-order only when justified
```

No manually assigned permanent weights.

---

# 18. Non-Negotiable Future Constraints

Do not violate these when implementing Phase 4+:

1. No look-ahead bias.
2. No forced UP/SIDEWAYS/DOWN prediction when evidence is unclear.
3. No arbitrary economic mappings.
4. No fixed indicator weighting.
5. No universal training window hard-coded by the user.
6. Do not use legacy `prediction_ledger` as model input.
7. Do not build duplicate ETL/scrapers.
8. Reuse finalized OLAP calculations where available.
9. Preserve raw stock return, benchmark return, and relative difference separately.
10. Company, Industry, and Sector are all first-class analytical levels.
11. Financials and market behavior are both first-class evidence.
12. Missing eligible data must be visible as a limitation.
13. Financial accounting period-end must not be mistaken for information availability.
14. Historical membership must not be fabricated.

---

# 19. Overall Phase 3 Completion Meaning

Phase 3 is successful when TrendAnalysis can take:

```text
Target company
Analysis date
Analysis timeframe
Holding period
Benchmark
Entry mode
```

and produce a consistent, time-bounded descriptive state containing:

```text
Company metadata
Company market state
Company financial state
Industry state
Sector state
Macro/global variable states
Benchmark context
Data limitations
```

That state is the formal input surface for Phase 4 relationship discovery.
