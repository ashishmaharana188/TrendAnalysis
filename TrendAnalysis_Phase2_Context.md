# TrendAnalysis — Phase 2 Context Hydration

## Purpose

Phase 2 built the **read-only data-access and normalization foundation** for the new `TrendAnalysis` project.

`TrendAnalysis` reads directly from the existing OLAP DuckDB database produced by `company_financial_data_ETL`.
The ETL project remains the data factory. No prediction logic, feature engineering, relationship discovery, or model training belongs in Phase 2.

---

# 1. Phase 2 Architecture

```text
TrendAnalysis
│
├── data_access/
│   ├── db.py
│   ├── query.py
│   ├── catalog.py
│   ├── metadata.py
│   ├── market.py
│   ├── financials.py
│   ├── macro_global.py
│   ├── institutional.py
│   ├── derivatives.py
│   ├── trade_events.py
│   └── contracts.py
│
└── tests/
    ├── test_data_access.py
    ├── test_catalog.py
    ├── test_metadata.py
    ├── test_market.py
    ├── test_financials.py
    ├── test_macro_global.py
    ├── test_market_flows.py
    └── test_contracts.py
```

Core separation:

```text
OLAP
  ↓
repositories / data_access
  ↓
validated repository results
  ↓
Phase 3 feature + variable-state engine
```

---

# 2. Database Connection Foundation

## `data_access/db.py`

Responsibilities:

- Resolve OLAP database path.
- Support `OLAP_DB_PATH` environment override.
- Default to the existing `market_data.duckdb` location relative to the workspace/repository layout.
- Open the database **read-only**.
- Do not import or depend on ETL implementation code.

Important principle:

```text
TrendAnalysis → reads OLAP
TrendAnalysis → does NOT write OLAP in Phase 2
```

The actual ETL database code uses:

```python
DB_PATH = "market_data.duckdb"
```

and DuckDB read-only connections. This was verified against the uploaded ETL schema.

---

# 3. Query Execution Layer

## `data_access/query.py`

Implemented helpers for:

- Arrow table execution.
- Python-row execution for lightweight/scalar queries.
- Streaming large result sets with Arrow record batches.

Important because `unified_market_master` is very large (over 100M rows in the audited database).

The analysis/model layer should not contain ad-hoc DuckDB connection logic.

---

# 4. Catalog Layer

## `data_access/catalog.py`

Provides:

- List database tables.
- List columns for a table.

Purpose: make the OLAP schema inspectable from TrendAnalysis without embedding catalog SQL throughout the system.

---

# 5. Metadata Repository

## `data_access/metadata.py`

Primary source:

```text
market_metadata
```

Actual schema verified from the ETL:

```text
Ticker
IndicatorName
TargetTable
Sector
Industry
AssetClass
Exchange
IsActive
valid_data_since
Description
```

Implemented access patterns:

- `get_active_companies()`
- `get_company(ticker)`
- `get_companies_by_sector(sector)`
- `get_companies_by_industry(industry)`
- `get_sectors()`
- `get_industries(sector=None)`

Important design rule:

- `market_metadata` is the current company → industry → sector mapping.
- Do not manufacture historical membership in Phase 2.
- Current audited metadata includes 101 equity companies, 9 sectors, and 47 industries.
- Some metadata records can have missing sector/industry values. Do not invent replacements.

---

# 6. Market Repository

## `data_access/market.py`

### Corrected source mapping

Company daily market data is read from:

```text
global_assets_daily
```

NOT from `unified_market_master` for the primary company-history repository.

The actual `global_assets_daily` schema is:

```text
Ticker
ReportDate
AssetClass
Open
High
Low
Close
Volume
```

This was explicitly corrected during Phase 2 after the initial implementation used `unified_market_master`.

### Implemented methods

- `get_daily_history()`
- `get_latest_daily_row()`
- `get_next_trading_day()`
- `get_price_on_or_before()`
- `get_market_date_range()`

### Temporal rule

For an analysis cutoff date `T`:

```text
latest company observation must satisfy ReportDate <= T
```

For next-trading-day entry:

```text
entry date > T
and use first usable trading row after T
```

This supports the two required backtest/measurement modes:

1. Next-trading-day entry.
2. Latest-known-data entry.

---

# 7. Financial Repository

## `data_access/financials.py`

Financial source tables exposed:

```text
quarterly_income_statement
yearly_income_statement
quarterly_balance_sheet
yearly_balance_sheet
quarterly_cash_flow
yearly_cash_flow
yearly_indirect_cash_flow
```

### Exact schemas verified

#### Quarterly income

```text
DataSource
Ticker
ReportDate
Currency
TotalRevenue
CostOfRevenue
GrossProfit
OperatingExpense
OperatingIncome
NetInterestIncome
TaxProvision
NetIncome
```

#### Yearly income

Same as above plus:

```text
IsValid
```

#### Quarterly balance sheet

```text
DataSource
Ticker
ReportDate
Currency
CashCashEquivalentsAndShortTermInvestments
Receivables
Inventory
CurrentAssets
TotalNonCurrentAssets
GrossPPE
AccumulatedDepreciation
NetPPE
TotalAssets
PayablesAndAccruedExpenses
CurrentDebtAndCapitalLeaseObligation
TotalTaxPayable
CurrentLiabilities
LongTermDebtAndCapitalLeaseObligation
TotalLiabilitiesNetMinorityInterest
CapitalStock
RetainedEarnings
StockholdersEquity
```

#### Yearly balance sheet

Same columns plus:

```text
IsValid
```

#### Quarterly cash flow

```text
DataSource
Ticker
ReportDate
Currency
BeginningCashBalance
CashReceipts
CashDisbursements
CashFromOperations
FixedAssetPurchases
NetBorrowing
IncomeTaxPaid
SaleOfStock
EndingCashBalance
```

#### Yearly cash flow

Same columns plus:

```text
IsValid
```

#### Yearly indirect cash flow

```text
DataSource
Ticker
ReportDate
Currency
IsValid
IsSectionValid
IsRollforwardValid
TreasuryOpacityRatio
NetIncome
DepreciationAndAmortization
OtherNonCashAdjustments
ChangeInAccountsReceivable
ChangeInInventory
ChangeInAccountsPayable
OtherWorkingCapitalChanges
IncomeTaxPaid
TotalOperatingCashFlow
Unmapped_Operating
CapExPurchaseOfPPE
PurchaseSaleOfInvestments
OtherInvestingActivities
TotalInvestingCashFlow
Unmapped_Investing
NetDebtIssuedRepaid
NetStockIssuedRepurchased
DividendsPaid
OtherFinancingActivities
TotalFinancingCashFlow
Unmapped_Financing
EffectOfExchangeRates
NetChangeInCash
BeginningCash
EndingCash
Unmapped_Rollforward
```

### Temporal warning

Critical distinction:

```text
financial ReportDate = accounting period end
```

It is **NOT** a reliable publication/availability date.

Therefore Phase 2 does NOT claim that a financial statement was known to the market on its accounting period-end date.

This must be handled explicitly in later temporal/backtesting design.

### Implemented access patterns

- Historical quarterly income.
- Historical yearly income.
- Latest quarterly/yearly periods.
- Historical quarterly/yearly balance sheets.
- Historical quarterly/yearly cash flow.
- Historical yearly indirect cash flow.
- Financial coverage by table.

---

# 8. Macro + Global/Company Repository

## `data_access/macro_global.py`

The source mapping was explicitly corrected during Phase 2.

### Macro source

```text
macro_daily_ledger
```

This contains the macro/commodity/global-indicator series such as:

```text
Brent_Crude
Broad_Commodity
India_10Y_Yield
India_CPI
India_VIX
Nifty_50
USD_INR
US_10Y_Yield
US_Dollar_Index
US_VIX
Yield_Spread
```

Actual macro daily schema:

```text
IndicatorName
ReportDate
Open
High
Low
Close_Value
Volume
```

### Macro intraday schema

```text
IndicatorName
ReportDate
Timeframe
Open
High
Low
Close_Value
Volume
```

Intraday is available but TrendAnalysis initially focuses on daily data.

### Company/global asset source

`global_assets_daily` contains company daily history and is exposed here as generic global-asset access where needed.

### Implemented macro methods

- `get_macro_history()`
- `get_latest_macro()`
- `get_macro_indicators()`
- `get_macro_date_range()`

### Implemented global/company methods

- `get_global_asset_history()`
- `get_latest_global_asset()`
- `get_global_assets()`
- `get_global_asset_date_range()`

### Important rule

Do NOT hard-code relationships such as:

```text
Brent_Crude → Energy
DXY → IT
Copper → Metals
```

The data-access layer only retrieves the variables.

Historical relationship discovery belongs to later phases and must be learned from data.

---

# 9. Institutional Repository

## `data_access/institutional.py`

Raw source:

```text
institutional_ledger
```

Exact source columns:

```text
ReportDate
ClientType
Cash_Buy_Value
Cash_Sell_Value
Cash_Net_Value
Nifty_Close
Future_Index_Long
Future_Index_Short
Future_Stock_Long
Future_Stock_Short
Option_Index_Call_Long
Option_Index_Put_Long
Option_Index_Call_Short
Option_Index_Put_Short
Option_Stock_Call_Long
Option_Stock_Put_Long
Option_Stock_Call_Short
Option_Stock_Put_Short
Total_Long_Contracts
Total_Short_Contracts
```

Derived source exposed:

```text
mv_institutional_flow
```

The ETL derives net positioning and changes using window functions.

Implemented access patterns:

- Raw institutional history.
- Derived institutional flow history.
- Latest derived institutional flow as-of cutoff.

No bullish/bearish interpretation is assigned here.

---

# 10. Derivatives Repository

## `data_access/derivatives.py`

### Options

Source:

```text
mv_options_aggregates
```

Exact columns:

```text
Ticker
ReportDate
ExpiryDate
Total_Put_OI
Total_Call_OI
OI_PCR
Total_Put_Volume
Total_Call_Volume
Volume_PCR
```

Implemented:

- `get_options_history()`
- `get_latest_options()`

### Futures basis

Source:

```text
mv_spot_futures_basis
```

Exact columns:

```text
Ticker
ReportDate
ExpiryDate
Spot_Price
Futures_Price
Open_Interest
Absolute_Basis
Basis_Percentage
```

Implemented:

- `get_futures_basis_history()`
- `get_latest_futures_basis()`

### Unified analytical market matrix

Source:

```text
mv_unified_market_matrix
```

The ETL-created matrix combines:

- cash market behavior
- options PCR
- futures basis
- trade-event/block-bulk information

Current matrix output fields in the repository:

```text
ticker
date
close
volume
delivery_percentage
daily_hl_spread
daily_vwap_dev
oi_pcr
delta_oi_pcr
futures_basis
is_fo_eligible
short_volume
short_percentage
net_block_volume
has_block_deal
avg_block_premium
```

Implemented:

- `get_unified_market_matrix()`
- `get_latest_unified_market_matrix()`

TrendAnalysis should consume this derived ETL product rather than recreate those same calculations.

---

# 11. Trade Events Repository

## `data_access/trade_events.py`

Source:

```text
trade_events_ledger
```

Exact columns:

```text
EventID
ReportDate
Ticker
EventType
SecurityName
ClientName
TransactionType
Quantity
TradePrice
Remarks
```

Implemented:

- `get_trade_events()`
- `get_latest_trade_events()`

Again, this layer only exposes the observations. It does not decide whether a buy/sell event is bullish or bearish.

---

# 12. Phase 2.10: Contracts + Normalization

## `data_access/contracts.py`

There was an important correction here.

### Problem discovered

The first version mixed:

```text
raw OLAP column names
```

with:

```text
repository result aliases
```

For example, raw `global_assets_daily` has:

```text
Ticker
ReportDate
Close
```

while `get_daily_history()` intentionally returns:

```text
ticker
report_date
close
```

The contract layer was rewritten to validate the **repository result schema**, not the underlying raw schema.

### Final principle

Two layers are conceptually distinct:

```text
RAW OLAP SOURCE SCHEMA
        ↓
repository SQL / aliases
        ↓
REPOSITORY RESULT SCHEMA
        ↓
contract validation
```

The final `contracts.py` therefore uses explicit `*_RESULT_COLUMNS` sets matching what the repositories actually return.

### Current normalization helpers

- `normalize_ticker()`
- `normalize_indicator()`
- `normalize_date()`
- `normalize_optional_date()`

### Current validation helpers

- `validate_columns()`
- `validate_not_empty()`
- `validate_market_result()`
- `validate_macro_result()`
- `validate_macro_intraday_result()`
- financial result validators
- `validate_institutional_result()`
- `validate_institutional_flow_result()`
- `validate_options_result()`
- `validate_basis_result()`
- `validate_unified_matrix_result()`
- `validate_trade_events_result()`

### Missing-data rule

Phase 2 does NOT:

- convert NULL to zero
- forward-fill missing values
- fabricate observations
- assign economic meaning
- suppress missingness

Later project rule remains:

```text
not applicable → do not flag
eligible but missing → flag + analysis LIMITED
still analyze using available evidence
```

---

# 13. Verified/Important Source Schema Facts

The uploaded ETL database schema confirms:

### `market_metadata`

```text
Ticker
IndicatorName
TargetTable
Sector
Industry
AssetClass
Exchange
IsActive
valid_data_since
Description
```

### `macro_daily_ledger`

```text
IndicatorName
ReportDate
Open
High
Low
Close_Value
Volume
```

### `macro_intraday_ledger`

```text
IndicatorName
ReportDate
Timeframe
Open
High
Low
Close_Value
Volume
```

### `global_assets_daily`

```text
Ticker
ReportDate
AssetClass
Open
High
Low
Close
Volume
```

### `global_assets_intraday`

```text
Ticker
ReportDate
Timeframe
Open
High
Low
Close
Volume
```

### `unified_market_master`

```text
Ticker
ReportDate
InstrumentType
ExpiryDate
StrikePrice
OptionType
Exchange_Series
Open
High
Low
Close
Volume
Turnover
No_Of_Trades
Delivery_Qty
Delivery_Percentage
Short_Volume
Open_Interest
Change_In_OI
Settlement_Price
Underlying_Price
```

### `institutional_ledger`

As documented in Section 9 above.

### financial tables

As documented in Section 7 above.

---

# 14. What Phase 2 Does NOT Do

Phase 2 does not:

```text
create features
calculate EMA/MACD/RSI
classify UP/SIDEWAYS/DOWN
learn historical relationships
select predictive variables
weight signals
calculate probabilities
calibrate probabilities
predict returns
backtest models
build company/industry/sector state
choose adaptive lookback windows
choose holding-period outcomes
```

These are later analytical phases.

---

# 15. Locked Project Decisions Relevant to Phase 2

These decisions must not be reopened during future hydration unless actual implementation evidence forces a change.

### Data ownership

```text
company_financial_data_ETL
    = data factory + OLAP

TrendAnalysis
    = read-only analysis/prediction project
```

### Primary company daily market source

```text
global_assets_daily
```

### Macro daily source

```text
macro_daily_ledger
```

### Intraday

Available, but excluded initially because TrendAnalysis starts with daily data.

### User analysis date cutoff

For analysis on date `T`, use only data available through the relevant cutoff, normally the latest completed observation before/at the cutoff. Do not use future rows.

### Financial cutoff caveat

Financial `ReportDate` is accounting-period end and must not be treated as publication date.

### External relationships

Learn them from history at industry/sector level rather than hard-coding economic maps.

### Primary prediction target

Absolute stock trend is primary. Relative performance to the user-selected benchmark is an additional comparison.

---

# 16. Phase 2 Completion State

Phase 2 is considered complete at the data-access level when the following all work:

```text
2.1 DB connection
2.2 Query execution
2.3 Large-query streaming
2.4 Catalog
2.5 Metadata repository
2.6 Market repository
2.7 Financial repository
2.8 Macro + global/company repository
2.9 Institutional + derivatives + trade-event repository
2.10 Contracts + normalization
```

The remaining known cleanup item is architectural rather than a missing capability:

`global_assets_daily` is exposed from both `market.py` and `macro_global.py` for historical compatibility with the repository split. This can be consolidated later without changing the underlying data source.

---

# 17. Handoff to Phase 3

Phase 3 should begin from **validated repository outputs**, not raw DuckDB SQL.

Expected flow:

```text
User parameters
    │
    ├── analysis_date
    ├── analysis_timeframe
    ├── holding_period
    └── benchmark
            ↓
      Phase 3 temporal layer
            ↓
    select historical observations
            ↓
    construct variable states
            ↓
    domain/group states
            ↓
    historical relationship discovery
            ↓
    predictive methods A + B
```

Phase 3 must preserve the critical conceptual separation:

```text
VARIABLE STATE
= what a variable is doing relative to its own history

RELATIONSHIP EFFECT
= what that state historically meant for the relevant
  company / industry / sector outcome
```

Do not collapse those two ideas into one feature.

---

# 18. Immediate Next Phase

## Phase 3.1 — Analysis Calendar / Temporal Frame

Define the reusable temporal objects for:

- analysis date
- analysis timeframe
- holding period (0.1–12.0 calendar months)
- historical observation window
- next-trading-day entry
- latest-known-data entry
- future outcome endpoint
- user-selected benchmark alignment

The temporal layer must be designed before feature/state generation because every later feature must obey the same information cutoff and holding-period rules.

