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
- **Rolling correlation:** repeats the correlation over a moving window. The shaded interval is an approximate pointwise 95% Fisher-transformation interval for each window, assuming independent paired changes. Adjacent windows overlap, so this is not a simultaneous confidence band; serial dependence can make it too narrow. Sign changes can reflect true instability or sampling noise.

A rough magnitude convention may describe absolute Pearson values below 0.2 as very weak, 0.2–0.4 weak-to-modest, 0.4–0.6 moderate, 0.6–0.8 strong, and above 0.8 very strong. These are **not** universal thresholds, significance cutoffs or usefulness scores. Economic relevance and reliability require separate judgement and validation.

The full-sample Pearson interval uses Fisher's transformation and assumes independent paired observations. Exact +/-1 correlations in very short samples are given the conservative full-range interval. Serial correlation, overlapping multi-observation changes, outliers and selecting among many pairs can make conventional intervals too optimistic.

## Event response

The app marks dates when the main variable's change over the selected window reaches the positive or negative threshold, then measures target response after 1 or more target observations. Forward target data after the study end date is used only to measure realized responses; late events can lack complete outcomes.

- **Event mean/median:** average/middle target response after dates meeting the threshold.
- **Non-event baseline:** response on matched dates where neither event threshold was crossed.
- **Event minus baseline:** difference between the event mean and baseline mean, expressed in the target response units. When response is a percentage return, the difference is measured in percentage points.
- **Events found vs responses available:** these counts differ when future target observations are not yet available or are missing.

The added difference graph focuses on the quantity of interest: event response minus ordinary baseline. Its intervals use a **circular moving-block bootstrap** over chronologically matched observations. Blocks try to preserve local dependence and some dependence from overlapping forward horizons. Block length is limited by sample size; the interval is unavailable for small/sparse samples or too few valid replicates. This is more appropriate than treating every event as independent, but it remains exploratory: block length is approximate and thresholds/horizons tried after viewing results are not corrected for multiple testing. An interval crossing zero means this sample does not clearly distinguish the effect estimate from zero; an interval excluding zero is a candidate association, not proof of causation.

## Lead/lag

The graph shows correlation at each tested alignment. Negative lag means the comparison series is aligned earlier than the main variable under this app's convention; positive means later. Lag counts matched observations, not necessarily calendar days. The graph includes approximate pointwise Fisher intervals under an independent-pair assumption. They are **not** adjusted for serial dependence or the number of tested lags.

The y-axis zooms to the observed correlations and intervals by default to make small values readable. Enable the full-scale option for the complete −1 to +1 range. The strongest absolute correlation is selected from the lags tested, so it is biased upward by the search. A peak whose interval includes zero is not clearly distinguished from zero under the simplifying assumptions. Even an interval excluding zero is only a candidate when it is the maximum of many tested lags; confirm it on pre-specified periods or use a multiple-testing/dependence-aware procedure. Lead/lag correlation does not prove prediction or cause.

## How to judge a result

Do not label a result good or bad from one coefficient. Inspect:

1. **Magnitude:** is the effect large enough to matter in the relevant units?
2. **Uncertainty:** how wide is the interval, and what assumptions support it?
3. **Support:** how many valid observations and events are present?
4. **Stability:** does the relationship persist across dates and regimes?
5. **Incremental value:** does it add information over an appropriate ordinary-response baseline?

The current UI is exploratory. Formal claims of significance, causality or out-of-sample usefulness require stronger pre-specified validation than these descriptive plots alone.
