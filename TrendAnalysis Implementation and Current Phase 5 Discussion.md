# TrendAnalysis: Implementation and Current Discussion

## 1. Project Objective

TrendAnalysis is a separate analytical/prediction project that reads the OLAP DuckDB database from `company_financial_data_ETL` in read-only mode.

The project is intended to take:

- Target: Company / Industry / Sector
- Analysis timeframe: approximately 1 week to 5 years
- Holding period: `0.1–12.0` calendar months
- Benchmark: user-selected
- Entry mode:
  - `next_trading_day`
  - `latest_known_data`

and produce:

- `UP / SIDEWAYS / DOWN`
- empirical approximate probabilities
- Conviction
- `NO_CLEAR_TREND` / limited evidence when evidence is insufficient
- absolute target return as primary outcome
- benchmark-relative return as secondary evidence

No forced directional prediction is allowed.

---

# 2. Project Separation

### `company_financial_data_ETL`

Remains responsible for:

- data ingestion
- transformations
- financial extraction
- market data processing
- OLAP construction
- materialized analytical views
- existing ratios/valuation calculations

### TrendAnalysis

Responsible for:

- descriptive state construction
- historical relationship discovery
- prediction
- walk-forward validation
- probability estimation
- conviction
- Company / Industry / Sector aggregation

TrendAnalysis must not duplicate ETL calculations or use legacy `prediction_ledger` as model input.

---

# 3. Phase 3: Descriptive State

Phase 3 is about:

> What is each variable doing relative to its own historical behavior?

It is **descriptive**, not predictive.

The chain is:

```text
Raw OLAP data
    ↓
Variable state
    ↓
Domain / group state
```

Current state families:

```text
company.market
company.financials
industry.market
industry.financials
sector.market
sector.financials
macro
global
benchmark
institutional
derivatives
microstructure
trade_events
```

13 families are represented.

Financial families remain timing-limited because `ReportDate` is accounting period end, not a verified publication/availability timestamp.

---

# 4. Phase 4: Historical Relationship Discovery

Phase 4 tests whether historical state configurations were associated with subsequent outcomes.

The important distinction is:

> A variable combination is a candidate hypothesis. The empirical observation is the actual historical outcome that followed similar states.

Example:

```text
FII flow
+
OI PCR
+
RELIANCE volatility
```

is only a candidate relationship.

Historical observations might look like:

```text
Historical state occurrence → subsequent 1M return

2020-03-23 → -18.2%
2020-09-24 →  -6.1%
2021-04-19 →  -4.7%
2022-01-24 →  -8.3%
2022-05-09 → -11.4%
```

Those subsequent returns are the empirical observations.

The system therefore evaluates:

```text
P(Future Return | Current/Historical State Configuration)
```

rather than assuming an economic rule such as:

```text
FII bearish → stock must fall
```

---

# 5. Phase 4 Relationship Levels

Relationship search includes:

### Level 1

```text
Variable A → future outcome
```

### Level 2

```text
Variable A + Variable B → future outcome
```

### Level 3

```text
Variable A + Variable B + Variable C → future outcome
```

Higher-order search can exist where supported.

For example:

```text
Institutional flow
+
Derivatives OI PCR
+
Company volatility
```

can be tested empirically.

---

# 6. Structural Relationship Rules

The relationship engine does not blindly combine every variable with every other variable.

There is a structural family graph that defines which combinations are legitimate candidates to test.

Examples of allowed conceptual relationships include:

```text
company financials × company market
company financials × industry market

industry financials × industry market
industry market × sector market

sector market × benchmark
industry market × macro
industry market × global

institutional × company market
derivatives × company market
microstructure × company market
trade_events × company market

derivatives × institutional
derivatives × macro/global
institutional × macro/global
```

These are **candidate eligibility rules only**.

They do not imply that the relationship is predictive.

Important future audit:

```text
Allowed and tested
Blocked by structural rule
Ineligible due to missing/insufficient data
```

This distinguishes a relationship rejected by empirical evidence from a relationship that was never tested.

---

# 7. Phase 4 Methods

## Method A: Adaptive Historical Similarity

Concept:

```text
Current state
    ↓
historical comparable states
    ↓
subsequent outcomes
    ↓
prediction
```

Safeguards implemented:

- missing variables reduce comparability
- missingness is not treated as similarity
- adaptive similarity rather than crude fixed tolerances
- magnitude consistency
- dispersion
- historical support
- can return `NO_CLEAR_TREND`

Method A is considered conceptually correct and functioning as designed.

Its weak RELIANCE accuracy is a predictive result, not automatically a software defect.

---

## Method B: Adaptive Historical Distribution

Concept:

```text
Historical outcomes
    ↓
state relevance weighting
    ↓
weighted historical distribution
    ↓
prediction
```

The estimator keeps:

- weighted mean
- weighted median
- weighted positive rate
- effective sample size
- reliability
- conditional distribution

The weighted statistics are designed to come from a coherent weighted population.

---

# 8. Phase 4 Hardening

Phase 4.10 introduced safeguards for major statistical failure modes.

Implemented safeguards include:

1. Missing-state comparability
2. Continuous stability assessment
3. Search-wide permutation adjustment
4. Benjamini-Hochberg FDR
5. FDR family isolation
6. Overlapping forward-outcome purging
7. Parameter provenance manifests
8. Future-parameter detection
9. Outer-test isolation
10. State-surface audit

Important principle:

> Temporal cleanliness alone does not guarantee absence of selection bias or overfitting.

The relationship discovery process therefore distinguishes:

```text
Discovery
    ↓
Selection holdout
    ↓
FDR/search adjustment
    ↓
Outer test
```

---

# 9. Institutional + Derivatives Expansion

Phase 4.11 / 4.12 expanded the state surface to include:

### Institutional

From:

`institutional_ledger`

and:

`mv_institutional_flow`

including FII/DII-style cash and derivatives positioning fields.

### Derivatives

From:

`mv_options_aggregates`

including:

- OI PCR
- volume PCR
- call/put OI
- call/put volume

and:

`mv_spot_futures_basis`

including:

- futures price
- spot price
- open interest
- absolute basis
- basis percentage

### Microstructure

From:

`mv_unified_market_matrix`

including:

- delivery percentage
- daily high-low spread
- VWAP deviation
- OI PCR
- delta OI PCR
- futures basis
- short volume
- short percentage
- block volume
- block deal indicators
- block premium

### Trade Events

From:

`trade_events_ledger`

including:

- event type
- client
- transaction type
- quantity
- trade price
- remarks

The new data is consumed through the ETL-derived analytical products rather than recalculated inside TrendAnalysis.

---

# 10. Phase 4/5 Wiring

A major integration issue was found and fixed.

Nested variable paths such as:

```text
institutional.FII_FPI.cash_net_value
derivatives.options.oi_pcr
```

were initially being interpreted as separate relationship families.

They are now mapped to canonical families such as:

```text
institutional
derivatives
```

A canonical family registry is intended to be shared by:

- relationship candidate generation
- combination expansion
- provenance

This ensures institutional and derivative variables can participate in Level 1/2/3 combinations.

---

# 11. Exhaustive Search vs Adaptive Pruning

A previous adaptive-pruning version did:

```text
L1/L2 candidates
    ↓
select seed pairs
    ↓
generate only L3 combinations from those seeds
```

Example:

```text
5926 L1/L2 candidates
→ 77 seed pairs
→ ~4600 L3 candidates
```

This is faster, but it changes the statistical experiment.

A potentially useful triple whose pairwise parents are weak would never be tested.

Therefore:

> Adaptive pruning is NOT used for the definitive real OLAP validation.

Current requirement:

```text
Exhaustive valid L1
Exhaustive valid L2
Exhaustive valid L3
```

Performance must be improved by implementation optimization, not by reducing the candidate universe.

---

# 12. Performance Optimization

The original exhaustive implementation was too slow.

The accepted optimization strategy is:

- vectorized NumPy calculations
- batch candidate evaluation
- precomputed state representations
- reusable outcome arrays
- reusable candidate definitions
- reuse sorted outcome ordering
- avoid storing full historical support for every candidate
- reconstruct detailed support only when needed
- reuse state/group history across folds
- avoid repeated data access

A parallel-fold experiment was attempted but was slower because of NumPy/BLAS contention.

Therefore parallel execution is not the default.

The key performance bug found was:

> The supposedly vectorized Method B engine was sometimes being called one candidate at a time by the nested validator.

That defeated the batching optimization.

This was fixed so candidate batches are passed into the vectorized engine.

---

# 13. Method B Selection Problem

The first expanded Phase 5.8 result was:

```text
Method B:
directional predictions = 233
directional hits = 78
directional hit rate = 33.48%
NO_CLEAR_TREND = 0
coverage = 100%
```

Baseline:

```text
36.05%
```

Combined:

```text
33.50%
```

This did not demonstrate predictive edge.

The major methodological flaw was:

```text
151,898 relationships
        ↓
in-sample ranking
        ↓
strongest relationship
        ↓
prediction
```

This creates severe selection bias.

Method B appeared to make a clear prediction almost every time because one apparently strong relationship was always available.

---

# 14. Correct Phase 5.8 Selection

The updated nested selection path is intended to use:

```text
Historical discovery
        ↓
chronological selection holdout
        ↓
search-wide multiple-testing correction
        ↓
BH-FDR
        ↓
selected relationships
        ↓
outer-test prediction
```

The selected relationships should feed the prediction engine directly.

They should NOT be:

```text
selected again by in-sample ranking
```

inside the prediction step.

This is the central correction to Method B.

---

# 15. Method 5.1–5.8 Treatment of New Data

The new institutional/derivatives information does **not** require hard-coded indicator logic inside Phase 5.

The intended architecture is generic:

```text
Phase 4
Institutional
Derivatives
Microstructure
Trade Events
        ↓
state variables
        ↓
relationships
        ↓
Phase 5
generic prediction machinery
```

Therefore:

### 5.1
Outcome classification remains generic.

### 5.2+
Prediction machinery consumes discovered relationships/states.

The new data enters through the Phase 4 state surface rather than through special cases such as:

```text
if OI_PCR > X → UP
if FII < X → DOWN
```

Those relationships must be empirically discovered.

---

# 16. Empirical Validation Logic

For each fold:

```text
Past information
    ↓
discover candidate relationships
    ↓
select relationships using only permitted historical information
    ↓
produce prediction
    ↓
future period occurs
    ↓
observe actual return
    ↓
compare prediction vs actual
```

Example:

```text
Prediction = UP
Actual     = UP
→ HIT

Prediction = UP
Actual     = DOWN
→ MISS
```

The resulting historical prediction record is the true out-of-sample empirical observation for model validation.

---

# 17. Current Understanding of Hit Rate

A number such as:

```text
78 / 233 = 33.48%
```

is only the **overall three-class accuracy**.

It does NOT mean:

```text
UP hit rate = 33.48%
```

It means:

```text
all Method B predictions
    ↓
78 correct
    ↓
233 total
    ↓
33.48% overall accuracy
```

The validation therefore needs class-specific results.

---

# 18. Required Class-Level Validation

For:

```text
UP
SIDEWAYS
DOWN
```

report:

- predicted count
- actual count
- correct count
- precision
- recall
- class hit rate

Also report a confusion matrix:

```text
                 ACTUAL
              UP  SIDE  DOWN

PRED UP       ...
PRED SIDE     ...
PRED DOWN     ...
```

This shows whether the model is systematically confusing one class with another.

---

# 19. Required Relationship-Level Validation

For each selected relationship, retain an auditable record containing at minimum:

```text
prediction_date
method
relationship_id
variables
variable_families
order
historical_support
historical_UP_count
historical_SIDEWAYS_count
historical_DOWN_count
historical_class_distribution
predicted_class
predicted_prob_UP
predicted_prob_SIDEWAYS
predicted_prob_DOWN
expected_return
actual_class
actual_return
hit_miss
```

This lets us trace:

```text
Prediction
    ↓
Selected relationship
    ↓
Variables involved
    ↓
Historical comparable observations
    ↓
Historical outcome distribution
    ↓
Predicted probability/class
    ↓
Actual future outcome
    ↓
HIT / MISS
```

This is required because aggregate accuracy alone hides what the model is actually learning.

---

# 20. Latest Phase 5.8 Result

The latest run after the hit-rate/reporting implementation produced:

```text
PHASE 5.8 REAL OLAP PREDICTION VALIDATION

Ticker: RELIANCE
Benchmark: Nifty_50
Analysis timeframe: 6M
Holding period: 1.0
Entry mode: next_trading_day

Candidate predictions: 249
Evaluated predictions: 208
Skipped threshold-limited: 0
Skipped provenance-failed: 0
Skipped invalid actual: 0
```

Method A:

```text
directional predictions: 0
directional hits: 0
directional hit rate: 0.0
clear-trend predictions: 0
clear-trend accuracy: 0.0
NO_CLEAR_TREND: 0
limited: 0
coverage: 0.0
```

Method B:

```text
directional predictions: 0
directional hits: 0
directional hit rate: 0.0
clear-trend predictions: 0
clear-trend accuracy: 0.0
NO_CLEAR_TREND: 0
limited: 0
coverage: 0.0
```

Combined:

```text
directional predictions: 0
directional hits: 0
clear-trend predictions: 0
clear-trend accuracy: 0.0
NO_CLEAR_TREND: 208
coverage: 0.0
```

Other results:

```text
Baseline majority accuracy: 35.58

Mean combined probabilities:
UP: 33.33
SIDEWAYS: 33.33
DOWN: 33.33

Mean combined expected return:
None

Conviction:
STRONG: 0
MODERATE: 0
LOW: 0
NONE: 208

Selection candidate evaluations: 0
Selection-validated predictions: 0
Multiple-testing-controlled folds: 0

Latest validated prediction date: 2026-08-24
Latest market date: 2026-10-01
```

---

# 21. Current Critical Issue

The latest result is **not a valid statement that the model has no predictions**.

It means the nested selection process accepted:

```text
0 relationships
```

for every evaluated fold.

Therefore:

```text
Selection candidate evaluations = 0
Selection-validated predictions = 0
FDR-controlled folds = 0
```

This needs to be investigated before rerunning a complete 249-fold test.

The likely problem is in the new selection/reporting integration, not necessarily in the underlying relationship discovery.

Possible places to inspect:

```text
Discovery results
        ↓
selection input
        ↓
FDR/search-adjustment
        ↓
accepted relationships
        ↓
prediction engine
```

We must verify exactly where the chain becomes empty.

Do NOT immediately interpret the zero-selection result as evidence that no relationships exist.

---

# 22. What Should Happen Next

Before another full OLAP run, inspect the zero-acceptance path.

For at least one fold, log:

```text
discovery candidate count
discovery result count
selection candidate count
raw selection statistics
search-wide permutation result
FDR threshold
number passing raw threshold
number passing FDR
number finally accepted
```

Then determine whether:

```text
0 accepted
```

is:

1. statistically expected, or
2. caused by an implementation/wiring error.

Only after confirming that should the 249-fold run be repeated.

---

# 23. Current Methodological Rules

Do NOT:

- restore adaptive L3 pruning for speed
- force relationships through FDR
- force Method B to predict
- interpret model probabilities as calibrated probabilities
- equate overall accuracy with class-specific accuracy
- claim predictive edge from the current 33.5% result
- assume institutional/derivative variables are predictive merely because they are included
- silently block variable combinations without being able to audit the block

Do:

- exhaustively evaluate valid L1/L2/L3 candidates
- preserve chronological integrity
- preserve selection holdouts
- use search-wide multiple-testing control
- report class-level performance
- report relationship-level evidence
- retain actual outcomes and hit/miss
- distinguish relationship discovery from prediction performance
- distinguish structural eligibility from empirical evidence

---

# 24. Current Status

```text
Phase 2     COMPLETE
Phase 3     COMPLETE
Phase 4     FOUNDATION + HARDENING COMPLETE
Phase 4.11  INSTITUTIONAL / DERIVATIVES / MICROSTRUCTURE / TRADE EVENTS ADDED
Phase 4.12  FLOW RELATIONSHIP WIRING COMPLETE
Phase 5.1   IMPLEMENTED
Phase 5.2+  IMPLEMENTED
Phase 5.8    NESTED WALK-FORWARD PREDICTION IMPLEMENTED
Phase 5.8    PERFORMANCE OPTIMIZED
Phase 5.8    CLASS/RELATIONSHIP VALIDATION REPORTING ADDED
Phase 5.8    ZERO-SELECTION ISSUE CURRENTLY UNDER INVESTIGATION

Phase 6     NOT YET COMPLETE
Phase 7     NOT STARTED
Phase 8     NOT STARTED
```

## Current authoritative principle

The system should ultimately answer:

> **Which state combinations repeatedly occurred historically, what outcomes followed them, which relationships survived statistically valid out-of-sample selection, and how accurately did those relationships predict unseen future outcomes?**

The final model should be judged from the actual walk-forward predictions and outcomes, not merely from impressive-looking relationships found somewhere inside 151,000 candidates.