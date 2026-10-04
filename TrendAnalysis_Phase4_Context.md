# TrendAnalysis — Phase 4 Context / Hydration

## Status

**PHASE 4 — RELATIONSHIP DISCOVERY ENGINE: COMPLETE**

Phase 4 converts descriptive Phase 3 states into historically testable relationships with subsequent forward outcomes.

The Phase 4 design preserves the separation:

```text
Phase 3
Variable / domain state
        ↓
Phase 4
Historical relationship effect
        ↓
Later phases
Prediction / probabilities / final trend
```

Phase 4 does **not** produce the final user-facing UP / SIDEWAYS / DOWN prediction, calibrated probabilities, or conviction. Those belong to Phase 5 and Phase 6.

---

# 1. Phase 4 Objective

The purpose of Phase 4 is to answer:

> Given a state observed before a prediction date, what has historically happened afterward under comparable or relevant conditions?

The engine must discover relationships empirically rather than use manually hard-coded mappings or permanent indicator weights.

Examples of structurally permitted relationships include:

```text
company financials × company market
company financials × industry market
industry financials × industry market
industry market × sector market
industry market × macro/global variables
sector market × benchmark
industry + sector + macro/global
```

External relationships are primarily discovered at Industry/Sector level. Company-level relationships remain important for the target prediction.

---

# 2. Phase 4 Input Contract

Phase 4 consumes Phase 3 descriptive states and raw forward outcomes.

It must not introduce a parallel raw-SQL feature pipeline.

The historical learning object is effectively:

```text
prediction_date
    ↓
information cutoff
    ↓
Phase 3 state snapshot
    ↓
forward outcome
```

The state remains descriptive. The outcome remains raw until later probability-labeling/calibration logic.

---

# 3. Phase 4.2 — Historical State / Outcome Panel

Core components:

```text
analysis/history_panel.py
analysis/outcome.py
```

Purpose:

Pair a Phase 3 historical state snapshot with the realized forward market outcome for the same prediction date.

Key guarantees:

- matching on target + prediction date
- state cutoff validation
- outcome temporal validation
- correct handling of both entry modes
- no conversion of invalid/missing outcomes into zero
- preservation of raw stock return
- preservation of raw benchmark return
- preservation of raw relative return
- preservation of state limitations
- no UP/SIDEWAYS/DOWN labels at this layer

Test result:

```text
PHASE 4.2 HISTORICAL STATE/OUTCOME PANEL TEST: PASS
Valid pairs: 1
Skipped cutoff rows: 1
Missing outcome rows: 1
```

The outcome definition supports:

```text
next_trading_day
latest_known_data
```

and holding periods from 0.1 to 12.0 calendar months.

Relative return remains:

```text
relative_return = stock_return - benchmark_return
```

---

# 4. Phase 4.3 — Method A: Adaptive Historical Similarity

Method A asks:

> Which historical states looked most like the current state, and what happened afterward?

Conceptually:

```text
Current state
    ↓
adaptive similarity
    ↓
comparable historical states
    ↓
subsequent outcomes
```

Important implementation principles:

- no arbitrary fixed tolerances
- no fixed user-defined training window
- similarity is adaptive
- structural groups are preserved rather than flattening everything without hierarchy
- missing state fields do not automatically become negative evidence
- similarity determines relevance, not economic direction

Validation result:

```text
PHASE 4.3 METHOD A ADAPTIVE SIMILARITY TEST: PASS
Selected comparable states: 5
Mean matched return: 3.0
Similarity-adjusted score: 0.244
Stable: True
```

---

# 5. Phase 4.4 — Method B: Adaptive Historical Distribution

Method B is intentionally different from Method A.

It asks:

> What does the full historical outcome distribution look like, and how relevant is each historical observation to the current condition?

Conceptually:

```text
All historical outcomes
        ↓
historical baseline distribution
        ↓
current-condition relevance
        ↓
adaptive observation weights
        ↓
weighted outcome distribution
```

Method B tracks evidence such as:

- weighted mean return
- weighted positive rate
- effective sample size
- weight concentration
- weighted lift/effect strength
- adaptive reliability

Validation result:

```text
PHASE 4.4 METHOD B ADAPTIVE DISTRIBUTION TEST: PASS
Conditioned relationship samples: 24
Weighted mean return: 2.1628
Weighted positive rate: 76.2937
Effective sample size: 48.0426
Weight concentration: 0.0269
```

The effective sample size demonstrates that Method B is using a weighted historical distribution rather than simply becoming a nearest-neighbour selector.

---

# 6. Phase 4.5 — Relationship Evidence Ranking

Phase 4.5 ranks discovered relationships rather than treating them as binary useful/not-useful objects.

Evidence dimensions include concepts such as:

- Method A / Method B agreement
- relationship direction
- observed strength
- sample support
- stability
- reliability
- incremental usefulness

Validation result:

```text
PHASE 4.5 RELATIONSHIP EVIDENCE RANKING TEST: PASS
Ranked relationships: 2
Top method agreement: True
Top direction: POSITIVE
Top score: 0.8
```

Important limitation:

A high Phase 4.5 evidence score is still not equivalent to proven out-of-sample predictive validity. Phase 4.8 and 4.9 address that distinction.

---

# 7. Phase 4.6 — Adaptive Combination Discovery

Combination discovery proceeds hierarchically:

```text
individual variables
        ↓
pairs
        ↓
3-variable combinations
        ↓
higher order only when justified
```

Only evidence-supported parent relationships are allowed to seed expansion.

Structural compatibility constrains the search space.

Validation result:

```text
PHASE 4.6 ADAPTIVE COMBINATION DISCOVERY TEST: PASS
Parent pair relationships: 2
Selected seed pairs: 2
3-variable candidates: 1
Evaluated method results: 2
Top direction: POSITIVE
Top score: 0.4996
```

The system must not generate arbitrary high-order combinations merely because they are mathematically possible.

---

# 8. Phase 4.7 — Adaptive Higher-Order Stopping

Higher-order expansion requires incremental improvement over the parent relationship.

Validation result:

```text
PHASE 4.7 ADAPTIVE HIGHER-ORDER STOPPING TEST: PASS
Highest order: 3
Expansion steps: 1

Order 2→3:
candidates = 2
evaluated  = 2
retained   = 0
best_gain  = 0.0000
stopped    = True

Stopped reason:
no_child_improves_over_adaptive_gain
```

This establishes a stopping mechanism against combinatorial explosion and prevents unnecessarily complex relationships from influencing later prediction.

---

# 9. Phase 4.8 — Walk-Forward Relationship Validation

Phase 4.8 validates the relationship discovery process itself using chronological out-of-sample folds.

For each historical prediction date:

```text
older observations only
        ↓
relationship discovery
        ↓
hold out prediction date
        ↓
compare to realized outcome
```

No future observation may enter relationship discovery for that fold.

Synthetic validation result:

```text
PHASE 4.8 WALK-FORWARD RELATIONSHIP VALIDATION TEST: PASS
Evaluated folds: 55
Method A hit rate: 100.0
Method B hit rate: 100.0
Combined hit rate: 100.0
Leakage violations: 0
```

The 100% synthetic result proves the walk-forward machinery can recover the intentionally persistent synthetic relationship. It must **not** be interpreted as real-market performance.

---

# 10. Phase 4.9 — Real OLAP Relationship Validation

Phase 4.9 runs the relationship-validation path against the actual TrendAnalysis OLAP database rather than synthetic data.

Validation configuration:

```text
Ticker: RELIANCE
Benchmark: Nifty_50
```

Observed result:

```text
Panel rows: 59
Walk-forward folds: 47
Method A hit rate: 38.64
Method B hit rate: 52.78
Combined hit rate: 43.18
Leakage violations: 0
State future violations: 0
Outcome temporal violations: 0
Latest market date: 2026-10-01
```

## Interpretation

The most important outcome is that the real-OLAP pipeline is temporally clean:

```text
Leakage violations = 0
State future violations = 0
Outcome temporal violations = 0
```

This means the test did not detect future information entering the relationship-validation process.

The predictive results are preliminary but informative:

```text
Method B  = 52.78%
Method A  = 38.64%
Combined  = 43.18%
```

Therefore:

1. Method B currently appears materially stronger than Method A on this real RELIANCE configuration.
2. A blind 50/50 combination is not justified by these results because the combined result is below Method B.
3. Method A should not be discarded merely from this one target; it remains a distinct method and may behave differently across companies, timeframes, holding periods, and regimes.
4. The result is **not yet sufficient to claim a robust trading edge**. It is one company/configuration with 47 real walk-forward folds.
5. If the reported hit rate is interpreted as three-class directional accuracy, 52.78% is above the 33.3% equal-class naive baseline, but this must still be evaluated across a broader real historical test matrix before drawing strong conclusions.

Phase 5 must therefore keep Method A and Method B as separate prediction channels and allow later validation/calibration to determine their relative usefulness instead of forcing a permanent equal weight.

---

# 11. Phase 4 Global Regression Status

All Phase 4 components validated:

```text
4.0 Relationship Discovery foundation       PASS
4.2 Historical State/Outcome Panel          PASS
4.3 Method A                               PASS
4.4 Method B                               PASS
4.5 Evidence Ranking                       PASS
4.6 Combination Discovery                  PASS
4.7 Higher-Order Stopping                  PASS
4.8 Walk-Forward Validation                PASS
4.9 Real OLAP Validation                   PASS
```

Phase 4 therefore meets its implementation objective.

---

# 12. Locked Phase 4 Principles

These must remain unchanged unless actual evidence forces architectural revision:

1. No look-ahead bias.
2. Phase 4 consumes Phase 3 outputs, not a duplicate raw-SQL feature system.
3. Variable state and relationship effect remain separate concepts.
4. No hard-coded economic mappings.
5. No permanent manually assigned indicator weights.
6. No universal user-defined training window.
7. Historical learning windows remain adaptive.
8. Relationships are ranked by empirical evidence.
9. Combination growth is structurally constrained.
10. Higher-order relationships require incremental usefulness.
11. Method A and Method B remain genuinely distinct.
12. Raw stock, benchmark, and relative returns remain separate.
13. Company, Industry, and Sector remain first-class analytical levels.
14. Financials and market behavior remain first-class evidence.
15. Missing eligible data remains visible as a limitation.
16. Historical financial accounting dates must not automatically be treated as information availability dates.
17. Historical membership must not be fabricated.
18. Final UP / SIDEWAYS / DOWN probabilities are not produced by Phase 4.

---

# 13. Handoff to Phase 5

Phase 5 is the **Prediction Engine**.

It should consume:

```text
Current Phase 3 state
        ↓
Phase 4 ranked relationships
        ↓
Method A evidence
Method B evidence
Combination evidence
        ↓
Prediction
```

Phase 5 responsibilities:

- generate empirical UP / SIDEWAYS / DOWN probabilities
- produce a directional prediction when evidence supports one
- permit no-clear-trend / weak-evidence situations
- maintain Method A and Method B as distinguishable prediction paths
- derive an initial conviction measure from evidence rather than fixed indicator scores
- preserve raw supporting evidence for later calibration/backtesting

Probability thresholds for UP / SIDEWAYS / DOWN must not be permanently hard-coded from arbitrary return percentages. The later calibration layer must learn the outcome labeling/distribution for each holding period from historical data.

---

# 14. Important Phase 5/6 Warning

The real 4.9 result must not be treated as the final model score.

The next major validation question is:

> Across many companies, analysis timeframes, holding periods, and historical prediction dates, does the prediction engine remain useful out of sample and are its probabilities calibrated?

That belongs to Phase 6.

The 4.9 result establishes that the existing relationship-discovery machinery can reach the real OLAP data without detected temporal leakage and that Method B is currently the stronger empirical candidate in this single real test.
