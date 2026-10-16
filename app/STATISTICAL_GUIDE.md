# Reading TrendAnalysis statistical panels

The terminal is a research aid. No single coefficient is universally good or bad, and no plot here is a buy/sell signal. Read each result alongside its units, selected dates, valid/matched observation count, uncertainty, missing-data notes, and stability over time.

## Distribution & variability

Choose all selected series or a subset. Every selected variable receives its own chart and summary; unrelated raw units are not placed on one shared numerical axis.

- **Observed values:** original recorded levels. Useful for ranges and clustering, but trending levels can make the distribution mainly reflect the trend.
- **Absolute change:** current observation minus the previous available observation. Useful for size and direction in the variable's own units, including signed values that cross zero.
- **Percentage change:** `(current / previous - 1) × 100`, only when the previous value is positive. Useful for ordinary price returns, but inappropriate for negative/zero bases and not automatically comparable across different observation frequencies.

Visuals:

- **Histogram:** each bar counts observations inside a numerical range. Bar height is not the probability of a future move; shape depends on the bin width.
- **Box plot:** median, middle 50% of observations, and points flagged as potential outliers under the plotted convention. An outlier is not automatically a data error or trading opportunity.
- **Cumulative percentile:** at each x-value, the y-value is the share of observations at or below that value. It describes the chosen historical sample, not a calibrated forecast.

Numbers:

- **Mean:** arithmetic average; sensitive to extreme values.
- **Median:** middle observation; less sensitive to extremes.
- **Standard deviation:** spread around the mean in the metric's units.
- **25th–75th percentiles:** middle half of observed values.
- **5th–95th percentiles:** central 90% observed range.
- **Skewness:** distribution asymmetry; positive values often reflect a longer/more influential upper tail, negative values a lower tail.
- **Excess kurtosis:** tail/peakedness descriptor relative to a normal-distribution reference. Sensitive to extremes and requires a reasonable sample.

Plain-language descriptions are rule-based and descriptive. Fewer than 30 observations triggers a caution, not a universal validity threshold. Large samples can still be biased, serially dependent or unrepresentative.

## Relationships

Changes are calculated first, then paired on exact matching observation dates. Matched counts can differ by pair.

- **Pearson r:** linear association between two sets of changes, from −1 to +1. Negative means opposite-direction linear association; positive means same-direction linear association; near zero means little linear association. It is not a probability and does not establish cause.
- **Spearman rho:** association between ranks; useful for monotonic relationships that are not strictly linear.
- **OLS slope:** estimated average change in the Y-variable's measured change for a one-unit change in X's measured change. Its magnitude depends on the units and transformation.
- **Scatter plot:** each point is a matched date; the line is a simple linear fit. Inspect outliers and curved patterns.
- **Correlation heatmap:** Pearson correlations for selected pairs. Short two-line axis labels are used for space; hover a cell to see its full series name and matched count.
- **Rolling correlation:** repeats the correlation over a moving window. Its interval uses a Fisher transformation with a Bartlett-style effective sample size that discounts some autocorrelation between the two change series. This is an approximation, not a complete dependence correction. Adjacent rolling windows overlap heavily, so pointwise bands are not simultaneous bands; non-stationarity and small windows can still mislead. Sign changes can reflect true instability or sampling noise.

A rough magnitude convention may describe absolute Pearson values below 0.2 as very weak, 0.2–0.4 weak-to-modest, 0.4–0.6 moderate, 0.6–0.8 strong, and above 0.8 very strong. These are **not** universal thresholds, significance cutoffs or usefulness scores. Economic relevance and reliability require separate judgement and validation.

The full-sample Pearson interval uses Fisher's transformation with a Bartlett-style effective sample size estimated from autocorrelation at several short lags. This discounts some serial dependence, but it is only approximate: it does not fully handle non-stationarity, overlapping multi-observation changes, outliers, all forms of heteroskedasticity, or selecting among many variable pairs. Treat intervals as exploratory rather than final significance tests.

## Event response

The app marks dates when the main variable's change over the selected window reaches the positive or negative threshold, then measures target response after 1 or more target observations. Forward target data after the study end date is used only to measure realized responses; late events can lack complete outcomes.

- **Event mean/median:** average/middle target response after dates meeting the threshold.
- **Non-event baseline:** response on matched dates where neither event threshold was crossed.
- **Event minus baseline:** difference between the event mean and baseline mean, expressed in the target response units. When response is a percentage return, the difference is measured in percentage points.
- **Events found vs responses available:** these counts differ when future target observations are not yet available or are missing.

The added difference graph focuses on the quantity of interest: event response minus ordinary baseline. Its intervals use a **circular moving-block bootstrap** over chronologically matched observations. Blocks try to preserve local dependence and some dependence from overlapping forward horizons. Block length is limited by sample size; the interval is unavailable for small/sparse samples or too few valid replicates. This is more appropriate than treating every event as independent, but it remains exploratory: block length is approximate and thresholds/horizons tried after viewing results are not corrected for multiple testing. An interval crossing zero means this sample does not clearly distinguish the effect estimate from zero; an interval excluding zero is a candidate association, not proof of causation.

## Lead/lag

The graph aligns measured changes at multiple offsets. Negative lag means the comparison series is aligned earlier than the main variable under this app's convention; positive means later. Offsets count exact matched observations, not necessarily calendar days or a guaranteed number of trading sessions.

For a selected change window of W observations, the app tests offsets spaced by W observations (for example, W=5 tests −10, −5, 0, +5, +10 when testing two windows in either direction). This reduces direct overlap between adjacent change windows, but it cannot eliminate serial dependence, regime changes, or timing irregularities in data with different frequencies or missing dates. The displayed offset and window length must be read together.

Pointwise confidence intervals use a moving-block bootstrap of paired changes. This resamples contiguous blocks and approximately preserves local dependence; the block length is chosen from the data length and change window. Family-wise adjusted p-values come from a circular-shift maximum-statistic test: the comparison series is shifted relative to the primary series, shifts close to the tested offsets are excluded, and each observed lag is compared with the maximum absolute correlation across all tested lags under each shifted alignment. This helps account for choosing the strongest of the tested lags while preserving each series' broad serial structure approximately.

The circular-shift test assumes the aligned change process is sufficiently stationary for shifts to be meaningful. It does not correct for testing many different variable pairs, windows, thresholds or study periods. The p-value can be unavailable for short samples or wide lag ranges; the app should say so instead of manufacturing a result. A peak that survives this test remains an association, not proof of prediction or cause. Confirm candidates on a pre-specified untouched chronological period.

## How to judge a result

Do not label a result good or bad from one coefficient. Inspect:

1. **Magnitude:** is the effect large enough to matter in the relevant units?
2. **Uncertainty:** how wide is the interval, and what assumptions support it?
3. **Support:** how many valid observations and events are present?
4. **Stability:** does the relationship persist across dates and regimes?
5. **Incremental value:** does it add information over an appropriate ordinary-response baseline?

The current UI is exploratory. The lead/lag plot now uses non-overlapping tested offsets, moving-block-bootstrap intervals, and a circular-shift maximum-statistic p-value adjusted across the tested lags. Pair/rolling intervals use an approximate autocorrelation-based effective sample size; event-response differences use moving-block bootstrap intervals. None of these methods corrects all forms of dependence or all the other analyst choices. Cross-variable searches, repeated threshold/horizon tuning and chart interpretation still require pre-specified hypotheses, explicit multiple-testing control and confirmation on untouched chronological data before strong claims of statistical significance, causality or trading usefulness are justified.


## Multiple variables in event and lead/lag analyses

The event-defining variable/reference variable is a role in a particular research question, not a restriction to the workspace's main series. Event Study allows one selected event definition to be compared against multiple response targets. Lead/Lag allows one selected reference series to be compared against multiple series in one run. Each target/comparison remains a distinct result; it is not collapsed into a single pooled outcome. Lead/lag p-values are adjusted across offsets within each pair and then conservatively adjusted across the selected comparison set. Event-study intervals are not automatically corrected for selecting many event targets, thresholds or horizons, so treat multi-target results as exploratory and pre-specify the key comparisons when possible.
