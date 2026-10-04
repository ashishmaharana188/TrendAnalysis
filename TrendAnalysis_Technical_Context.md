# TrendAnalysis Project — Technical Requirements & Context

## 1. Purpose

TrendAnalysis is a new, independent project for historical/current market-trend analysis and approximate directional conviction.

It is intended for retail-investor-style analysis, not exact price prediction or institutional-grade forecasting.

The project must discover useful relationships from historical data rather than depend on manually hard-coded economic rules or indicator weights.

---

## 2. Project Boundaries

### Upstream

`company_financial_data_ETL`

- Remains the data factory.
- Continues through the OLAP database.
- TrendAnalysis reads from the OLAP database.
- TrendAnalysis must not move/refactor the ETL pipeline.
- TrendAnalysis should reuse finalized OLAP data instead of becoming a second ETL/scraping system.

### Related project

`ScreenerFundamentalScrapper`

- Provides company-universe/candidate selection.
- Selected companies can be inserted into `market_metadata` through the ETL.
- TrendAnalysis can analyze any company represented in `market_metadata` or a selected subset.

### Separate project

`NewsAnalysis`

- Exists separately.
- News/event/qualitative analysis is outside the initial TrendAnalysis scope.

---

## 3. Core Objective

Given:

- analysis target: company, industry, or sector
- analysis timeframe
- holding period
- user-selected benchmark

TrendAnalysis should determine the historical/current state of the relevant variables and estimate:

- `UP`
- `SIDEWAYS`
- `DOWN`

with empirical probabilities and a derived conviction level.

The system must be allowed to conclude that there is no clear trend rather than forcing a directional result.

---

## 4. User Inputs

### 4.1 Analysis Target

Supported scopes:

- Company
- Industry
- Sector

Complete-market prediction is not required.

For Industry/Sector analysis, constituent-company behavior can be aggregated to represent the group.

Possible group-level outputs include:

- aggregate constituent returns
- breadth: percentage of constituents classified UP/SIDEWAYS/DOWN
- group financial state
- group market state

### 4.2 Analysis Timeframe

User-selectable.

Approximate useful range:

- 1 week
- 1 month
- 3 months
- 6 months
- 1 year
- 3 years
- 5 years
- other useful standard market intervals

The analysis timeframe determines how historical/current state is examined.

### 4.3 Holding Period

Independent from analysis timeframe.

Range:

`0.1 to 12.0 calendar months`

The holding period determines the future outcome window used for prediction and backtesting.

### 4.4 Benchmark

User-selected benchmark.

Examples may include:

- Nifty
- sector index
- another appropriate benchmark available in the data

Relative performance is supplemental. The primary prediction is absolute stock/group direction.

---

## 5. Data Scope

TrendAnalysis operates on the data actually available in the OLAP database.

Major data domains include:

1. Company financials
2. Company market behaviour
3. Industry financials
4. Industry market behaviour
5. Sector financials
6. Sector market behaviour
7. Macro variables
8. Global variables
9. Institutional/derivative data where available

The system must not assume that every variable contains the same fields.

For example:

- macro/global assets may have OHLCV
- equity market data may additionally have delivery, OI, derivatives and related fields
- financial statements have quarterly/annual fields

A variable's state must be derived from the fields actually present for that variable.

---

## 6. Current OLAP Structure

Relevant tables currently identified include:

### Market metadata

`market_metadata`

Contains:

- Ticker
- IndicatorName
- TargetTable
- Sector
- Industry
- AssetClass
- Exchange
- IsActive
- valid_data_since
- Description

### Market data

`unified_market_master`

Contains fields including:

- Ticker
- ReportDate
- InstrumentType
- ExpiryDate
- StrikePrice
- OptionType
- Exchange_Series
- OHLC
- Volume
- Turnover
- No_Of_Trades
- Delivery_Qty
- Delivery_Percentage
- Short_Volume
- Open_Interest
- Change_In_OI
- Settlement_Price
- Underlying_Price

### Macro/global data

`macro_daily_ledger`

Daily OHLCV macro/market series.

`global_assets_daily`

Daily OHLCV global asset series.

Intraday tables also exist, but intraday stock data is excluded from the initial TrendAnalysis scope.

### Institutional data

`institutional_ledger`

Contains institutional cash and derivatives positioning/flow information, including:

- cash buy/sell/net
- Nifty close
- index futures long/short
- stock futures long/short
- index/stock option positioning
- total long/short contracts

### Trade events

`trade_events_ledger`

Contains structured trade-event data such as:

- block/bulk-type events
- ticker
- client
- transaction
- quantity
- price
- remarks

### Financial statements

Quarterly and annual:

- `quarterly_income_statement`
- `yearly_income_statement`
- quarterly/yearly balance-sheet tables
- quarterly/yearly cash-flow tables
- yearly indirect cash-flow data

### Materialized analytical views

Existing views include:

- `mv_options_aggregates`
- `mv_spot_futures_basis`
- `mv_institutional_flow`
- `mv_unified_market_matrix`

These should be reused where they already provide finalized analytical fields needed by TrendAnalysis.

### Legacy prediction data

`prediction_ledger` exists in the OLAP environment.

TrendAnalysis must not use legacy prediction outputs as model inputs.

A separate prediction/output ledger may be introduced for TrendAnalysis if required.

---

## 7. Financial Data Handling

Financials are not converted into artificial daily financial data.

Financial information follows its natural reporting cadence.

Examples:

- 1-month analysis → latest available quarter as financial basis
- 6-month analysis → latest two quarters / half-year
- 1-year analysis → approximately four quarters / annual data
- 3-year analysis → approximately three years of history
- 5-year analysis → approximately five years of history

Financial trend should consider:

1. performance over the selected analysis timeframe
2. comparison with the previous equivalent period
3. position relative to the company's full available history

The system must not invent monthly financial statements where they do not exist.

Existing ETL-derived financial ratios/valuation fields should be reused when available.

TrendAnalysis should not recreate existing calculations unnecessarily.

---

## 8. Data Availability and Missing Data

If a data category is not applicable to a target, it should not be treated as a missing-data problem.

If a target is eligible for a data category but required data is missing:

- flag the limitation
- mark the analysis as `LIMITED`
- continue analysis using available evidence

The user decides whether the limitation is material.

The system must not manufacture values to fill missing information.

---

## 9. No Look-Ahead Bias

Historical prediction must only use information available before the prediction date.

Example:

If the analysis date is October 3:

- data available through October 2 may be used
- information that became available on/after October 3 must not be used

The same rule applies to every historical backtest date.

A critical implementation requirement remains unresolved:

> Determine whether financial-table `ReportDate` represents information availability/publication date or only the accounting period-end date.

This must be verified against the actual OLAP data before implementation.

---

## 10. Variable State

The system should first describe what each variable is doing relative to its own history.

A variable state may contain:

- current value
- change over selected analysis timeframe
- historical position/range/distribution
- recent direction
- derived state/trend

Example:

```text
Crude
Current: 82
6M change: +14%
Historical position: High
Direction: Rising
State: Rising / High
```

The exact state representation must be selected based on the actual data and validated empirically.

### Important separation

Do not conflate:

```text
Variable state
```

with:

```text
Relationship effect
```

Example:

`Crude is rising`

does not inherently mean:

`Crude rising is bullish for this company`.

The first is descriptive. The second must be discovered from historical evidence.

---

## 11. Hierarchical Structure

The model should preserve the natural structural relationships in the data:

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

Valid relationships can include, for example:

- company financials × company market
- company financials × industry market
- industry financials × industry market
- industry market × sector market
- industry market × macro variables
- sector market × benchmark
- industry + sector + macro/global

The model should discover useful relationships within this structural space.

It should not generate arbitrary combinations merely because mathematical correlation is possible.

---

## 12. Relationship Discovery

No manually hard-coded mappings such as:

```text
Crude → Energy
Copper → Metals
DXY → IT
```

should determine the model.

Instead, historical data should determine:

- which variables matter
- which variable pairs matter
- which higher-order combinations matter
- how consistently they matter
- whether their usefulness persists across periods/regimes

Relationship discovery should primarily operate at Industry/Sector level for external variables.

Company-level analysis remains important, but company-level macro relationships do not need to become separate hard-coded models for every company.

---

## 13. Interaction Between Financials and Market Data

Financials and market behaviour are both first-class evidence.

Neither receives a permanently hard-coded priority.

Historical evidence may show that:

- financials strengthen market conviction
- financials weaken market conviction
- financials reverse a market signal
- market behaviour strengthens financial evidence
- market behaviour weakens financial evidence
- neither materially changes the other

These effects must be learned from historical outcomes.

Raw variable states remain independently visible even when their interaction affects the prediction.

---

## 14. Group-Level Analysis

The model should expose both:

### Variable level

Examples:

```text
Crude       → Rising / High
DXY         → Falling / Low
Volume      → Increasing
Delivery    → Increasing
```

### Domain/group level

Examples:

```text
Macro       → Positive
Market      → Strong Positive
Financials  → Neutral
```

The final prediction is derived after considering:

```text
Raw fields
  ↓
Variable state
  ↓
Domain/group state
  ↓
Structural relationships/interactions
  ↓
Prediction
```

---

## 15. Two Prediction Methods

The system must implement two conceptually distinct approaches using the same input data and outcome definitions.

### Method A — Similar Historical States

Concept:

```text
Current state
    ↓
Find historically comparable states
    ↓
Observe subsequent outcomes
    ↓
Empirical UP/SIDEWAYS/DOWN distribution
```

Similarity must be data-adaptive.

No fixed rules such as:

- variable must be within ±5%
- RSI must be within X points
- crude must be within Y dollars

Instead, the system should determine useful similarity from historical data.

Similarity should respect the hierarchical structure rather than reducing everything to one giant flat state vector.

The historical window used for learning should also be adaptive.

### Method B — Historical Distribution + Current-Condition Weighting

Concept:

```text
All relevant historical outcomes
          ↓
Historical baseline distribution
          ↓
Current conditions
          ↓
Adaptive relevance weighting
          ↓
Weighted UP/SIDEWAYS/DOWN distribution
```

Method B must learn:

- individual-variable effects
- interactions
- higher-order combinations where justified

No manually assigned indicator weights.

Method A and Method B should remain genuinely different methods rather than two names for the same algorithm.

---

## 16. Adaptive Historical Learning

Training/history selection should be adaptive.

The system should determine usable historical history based on factors such as:

- available history
- number of valid observations
- data completeness
- relationship stability
- regime changes
- predictive usefulness
- selected analysis timeframe
- selected holding period

The user should not need to manually configure a training-window size.

---

## 17. Combination Discovery

There is no fixed maximum combination size.

The system should:

1. evaluate individual variables
2. evaluate meaningful pairs
3. expand to 3-variable combinations where evidence supports expansion
4. continue to higher-order combinations only when justified by data

The search must avoid uncontrolled combinatorial explosion.

Structural hierarchy should constrain the search space.

A combination must demonstrate historical usefulness before influencing the final prediction.

---

## 18. Relationship Usefulness

A relationship is not considered useful merely because it has high correlation.

Usefulness should be evaluated using out-of-sample historical evidence, including factors such as:

- predictive improvement over baseline
- consistency
- observation count
- stability across periods
- stability across regimes
- ability to distinguish UP/SIDEWAYS/DOWN
- incremental contribution when other relevant variables are present

Relationships should be ranked rather than simply discarded into a binary useful/not-useful bucket.

Illustrative ranking:

```text
Relationship                  Usefulness
-----------------------------------------
Industry Market + Crude       High
Company Financials + Market   High
Sector + DXY                  Medium
Weak isolated variable        Low
```

These are illustrative labels, not predefined classifications.

---

## 19. Outcome Definition

Primary prediction:

```text
Absolute stock/group direction
```

Absolute outcome is the return over the selected holding period.

Relative performance is a secondary measurement:

```text
Relative return =
Stock/group return - Benchmark return
```

Raw values must be retained:

- stock/group return
- benchmark return
- relative difference

Relative performance is not a separate primary prediction target.

---

## 20. Entry/Measurement Modes

Backtesting should support both:

### Mode 1 — Next Trading Day Entry

Prediction is made using information available before the analysis date.

Entry/reference:

- next trading day's close

Then measure the selected holding-period outcome.

### Mode 2 — Latest Known Data Entry

Use the latest available close at the cutoff date as the entry/reference.

Then measure the selected holding-period outcome.

Both modes should be retained for validation.

For relative performance, use the same entry/exit dates as the absolute measurement.

---

## 21. UP / SIDEWAYS / DOWN Classification

No universal fixed return thresholds.

Classification should be adaptive to:

- company/group
- selected holding period
- historical return behaviour

The system should derive meaningful UP/SIDEWAYS/DOWN regions from the relevant historical future-return distribution.

A high-volatility stock should not be classified using the same fixed thresholds as a low-volatility stock.

The same classification methodology must be applied consistently throughout historical backtesting.

---

## 22. Probability Semantics

Displayed probabilities are empirical probabilities, not subjective confidence scores.

Example:

```text
UP = 65%
```

means approximately:

> Among historically comparable situations, approximately 65% subsequently produced an UP outcome over the selected holding period.

Probabilities must not be presented as exact future certainty.

---

## 23. Probability Calibration

Raw model probabilities must be calibrated using walk-forward historical evidence.

Example:

```text
Raw output:
UP = 68%

Historical calibration:
Predictions around 65–70%
actually produced UP ≈ 57%

Calibrated output:
UP ≈ 57%
```

Calibration must obey the same no-look-ahead rule.

At historical prediction date `T`, calibration cannot use outcomes occurring after the information available at `T`.

---

## 24. Method A vs Method B

Both methods must expose their independent results.

Example:

```text
Method A
UP: 72%
SIDEWAYS: 18%
DOWN: 10%

Method B
UP: 48%
SIDEWAYS: 31%
DOWN: 21%
```

The system should not blindly average them.

The eventual combined result should be determined from historical evidence regarding each method's usefulness under the relevant scope/timeframe/holding-period conditions.

If methods strongly disagree and neither has a demonstrated advantage, conviction should be reduced rather than hiding the disagreement.

The disagreement itself is useful diagnostic information.

---

## 25. Conviction

Final output includes:

- Trend: UP / SIDEWAYS / DOWN
- Empirical probabilities
- Conviction level

Conviction should be derived from empirical historical reliability and evidence quality rather than a manually assigned indicator score.

The exact calibration/labeling mechanism is an implementation/model-design decision to be validated during backtesting.

---

## 26. Backtesting Philosophy

Backtesting is walk-forward and timeframe-driven.

For a selected historical prediction date:

```text
Available data before T
        ↓
Build state at T
        ↓
Generate prediction
        ↓
Wait selected holding period
        ↓
Observe actual outcome
        ↓
Move forward
        ↓
Repeat
```

The system should automatically determine viable historical windows subject to data availability and the selected timeframe/holding period.

No future information may enter:

- state construction
- relationship discovery
- model fitting
- similarity learning
- probability calibration
- feature selection

when evaluating an earlier historical prediction date.

---

## 27. Backtest Scopes

Supported:

- Company
- Industry
- Sector

Complete-market backtesting is not required.

For Industry/Sector:

- constituent-company outcomes may be aggregated
- breadth may be measured
- group financial and market states may be constructed

Historical membership availability must be respected. Current `market_metadata` provides Sector/Industry fields, but historical membership semantics need verification before implementation if membership changed over time.

---

## 28. Required Final Evidence Chain

The final analysis should make the reasoning traceable.

At minimum:

```text
Target
Analysis timeframe
Holding period
Benchmark

        ↓

Company state
Industry state
Sector state
Macro/global state

        ↓

Variable trends

        ↓

Ranked historical relationships

        ↓

Method A result
Method B result

        ↓

Calibrated UP/SIDEWAYS/DOWN probabilities

        ↓

Final trend
Conviction
Data-quality/limitation status
```

The user should be able to see why the system reached its result without exposing internal implementation details that do not aid interpretation.

---

## 29. Data Quality

Analysis should expose when important evidence is unavailable.

States:

- normal/full analysis
- limited analysis

Missing data that is not applicable should not trigger warnings.

Eligible-but-missing data should trigger a limitation.

The system should continue using available evidence rather than automatically refusing to analyze.

---

## 30. Existing ETL Responsibilities

`company_financial_data_ETL` remains responsible for:

- ingestion
- transformation
- financial extraction
- market data processing
- OLAP construction
- existing ratio/valuation calculations
- existing materialized analytical views

TrendAnalysis should consume those outputs.

If a required calculation already exists in OLAP, reuse it.

Do not build duplicate data pipelines without a demonstrated gap.

---

## 31. Existing Screener Responsibilities

`ScreenerFundamentalScrapper` remains responsible for fundamental candidate discovery.

TrendAnalysis does not replace the screener.

Conceptually:

```text
Screener
   ↓
Candidate companies
   ↓
market_metadata / OLAP
   ↓
TrendAnalysis
   ↓
Historical/current trend analysis
```

---

## 32. Explicit Non-Goals

Initial TrendAnalysis does not include:

- intraday stock prediction
- exact price targets
- news sentiment
- legal-event interpretation
- qualitative fraud/governance classification
- independent SEC/event-tag analysis
- institutional-grade forecasting
- manually maintained economic mappings
- manually assigned indicator weights
- duplicate scraping/ETL
- forced directional predictions
- use of legacy prediction outputs as model features

If an external event affects financial statements or market data already present in OLAP, its effect may naturally appear through those data.

---

## 33. Current Known OLAP Analytical Views

Existing materialized views identified during requirements work:

### `mv_options_aggregates`

Provides option aggregation such as:

- put OI
- call OI
- OI PCR
- put/call volume
- volume PCR

### `mv_spot_futures_basis`

Provides:

- spot
- futures
- OI
- absolute basis
- basis %

### `mv_institutional_flow`

Provides institutional positioning and changes.

### `mv_unified_market_matrix`

Provides combined market fields including:

- close
- volume
- delivery %
- daily high-low spread
- VWAP deviation
- OI PCR
- delta OI PCR
- futures basis
- F&O eligibility
- short volume/percentage
- block volume
- block-deal flag
- average block premium

These are candidate inputs. Actual use must be determined by data availability and historical validation.

---

## 34. Legacy Quantitative Pipeline

A previous quantitative pipeline existed with engines covering areas such as:

- OLS/EOD prediction
- global regime
- macro/commodity relationships
- valuation elasticity
- smart-money tracking
- bankruptcy/margin shock
- derivatives
- volatility
- time-lag synchronization

This legacy architecture is **context only**.

TrendAnalysis is a fresh design.

Do not automatically reproduce its engines or assumptions.

---

## 35. Critical Implementation Principle

The project should be **data-driven but structurally constrained**.

That means:

```text
NOT:
Hard-coded economic assumptions
        +
Hard-coded indicator weights
        +
Huge flat feature matrix
```

Instead:

```text
Actual OLAP data
      ↓
Natural hierarchy
      ↓
Adaptive state representation
      ↓
Historical relationship discovery
      ↓
Out-of-sample validation
      ↓
Adaptive prediction
      ↓
Calibrated empirical probabilities
```

The model chooses what historically matters.

The project architecture constrains what relationships are economically/data-structurally meaningful.

---

## 36. Remaining Pre-Implementation Checks

Requirements are sufficiently defined to move into technical design.

Before implementation, only a small number of factual/data checks remain:

### 36.1 Financial information availability

Verify what `ReportDate` means in the financial statement tables:

- accounting period end
- publication/availability date
- or another timestamp

This is required to enforce no-look-ahead correctly.

### 36.2 Actual OLAP data audit

Inspect the actual database for:

- date ranges
- ticker coverage
- market/financial coverage
- missingness
- variable availability
- sector/industry coverage
- historical membership availability
- sufficient history for backtesting

### 36.3 Output interface

Determine the first delivery interface:

- Python API/CLI
- notebook
- dashboard
- other

This is an implementation decision, not a modeling requirement.

### 36.4 Backtest reporting

Define the final report/dashboard metrics after inspecting actual data and implementing the validation framework.

---

## 37. Decision Status

The core requirements are considered **locked**.

Do not restart broad requirements interviews unless actual data inspection reveals a genuine requirements gap.

The next phase should be:

```text
Actual OLAP audit
        ↓
Technical architecture
        ↓
Model/algorithm selection
        ↓
Backtest framework
        ↓
Implementation
        ↓
Validation
```

Algorithmic choices such as similarity metric, adaptive combination discovery, calibration method, feature selection and statistical/modeling techniques should be selected by engineering/model design based on the actual data and validated through walk-forward testing rather than pushed back to the user as dozens of configuration questions.
