from __future__ import annotations

import unittest

import pandas as pd

from app.services.statistics_service import (
    change_series,
    correlation_matrix,
    distribution_frame,
    distribution_interpretation,
    distribution_summary,
    event_study,
    lead_lag_statistics,
    pair_statistics,
    paired_change_data,
    rolling_pair_correlation,
    variable_summary,
)


def series(values):
    return pd.DataFrame({"date": pd.date_range("2025-01-01", periods=len(values), freq="D"), "value": values})


class StatisticsServiceTests(unittest.TestCase):
    def test_percentage_change_window(self):
        frame = series([100, 110, 121])
        result = change_series(frame, window=2, mode="pct")
        self.assertAlmostEqual(result["change"].iloc[-1], 21.0)

    def test_pair_statistics_uses_changes(self):
        a = series([100, 101, 103, 104, 108, 110])
        b = series([50, 50.5, 51.5, 52, 54, 55])
        result = pair_statistics(a, b, window=1, mode="pct")
        self.assertEqual(result["observations"], 5)
        self.assertGreater(result["pearson"], 0.99)

    def test_lead_lag_contains_zero_and_requested_edges(self):
        a = series([100, 101, 103, 102, 105, 107, 106])
        b = series([10, 10.1, 10.3, 10.2, 10.5, 10.7, 10.6])
        result = lead_lag_statistics(a, b, window=1, mode="pct", max_lag=2)
        self.assertEqual(result["lag_observations"].tolist(), [-2, -1, 0, 1, 2])
        self.assertEqual(int(result.loc[result["lag_observations"] == 0, "observations"].iloc[0]), 6)

    def test_variable_summary_respects_selected_chart_period(self):
        frame = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=5, freq="D"),
            "value": [10, 100, 110, 120, 200],
        })
        result = variable_summary(frame, "2025-01-02", "2025-01-04")
        self.assertEqual(result["observations"], 3)
        self.assertEqual(result["latest_value"], 120.0)
        self.assertAlmostEqual(result["percent_change"], 20.0)

    def test_event_study_reports_realized_forward_responses(self):
        primary = series([100, 103, 106, 102, 100, 105, 111, 106, 100, 104])
        target = series([100, 101, 102, 99, 98, 100, 103, 101, 97, 99])
        summary, events = event_study(
            primary, target, measurement_window=1, threshold=2.0,
            primary_mode="pct", response_mode="pct", horizons=[1, 2],
        )
        self.assertFalse(summary.empty)
        self.assertIn("realized_responses", summary.columns)
        self.assertIn("response", events.columns)
        self.assertTrue((summary["realized_responses"] <= summary["events_detected"]).all())

    def test_pair_statistics_adds_approximate_correlation_interval(self):
        a = series([100, 102, 103, 101, 105, 107, 106, 110, 112, 111, 115, 116])
        b = series([200, 204, 203, 205, 210, 211, 212, 220, 221, 225, 227, 229])
        result = pair_statistics(a, b, window=1, mode="pct")
        self.assertIsNotNone(result["pearson_ci_low"])
        self.assertIsNotNone(result["pearson_ci_high"])
        self.assertLessEqual(result["pearson_ci_low"], result["pearson"])
        self.assertGreaterEqual(result["pearson_ci_high"], result["pearson"])

    def test_correlation_matrix_reports_pairwise_counts(self):
        a = series([100, 101, 103, 102, 105, 107, 106])
        b = series([50, 50.5, 51.5, 51, 52.5, 53.5, 53])
        matrix, counts = correlation_matrix({"A": a, "B": b}, window=1, mode="pct")
        self.assertEqual(list(matrix.columns), ["A", "B"])
        self.assertGreater(matrix.loc["A", "B"], 0.99)
        self.assertEqual(int(counts.loc["A", "B"]), 6)

    def test_rolling_pair_correlation_is_defined_after_window(self):
        a = series([100, 101, 103, 102, 105, 107, 106, 109, 110, 112, 111, 114])
        b = series([50, 50.5, 51.5, 51, 52.5, 53.5, 53, 54.5, 55, 56, 55.5, 57])
        result = rolling_pair_correlation(a, b, window=1, rolling_observations=5)
        self.assertIn("rolling_correlation", result.columns)
        self.assertGreater(result["rolling_correlation"].notna().sum(), 0)

    def test_rolling_correlation_includes_pointwise_interval(self):
        a = series([100, 101, 103, 102, 105, 107, 106, 109, 110, 112, 111, 114, 116, 115, 118])
        b = series([50, 50.5, 51.5, 51, 52.5, 53.5, 53, 54.5, 55, 56, 55.5, 57, 58, 57.5, 59])
        result = rolling_pair_correlation(a, b, window=1, rolling_observations=6)
        valid = result.dropna(subset=["rolling_correlation"])
        self.assertGreater(len(valid), 0)
        self.assertIn("rolling_ci_low", result.columns)
        self.assertIn("rolling_ci_high", result.columns)
        self.assertTrue((valid["rolling_ci_low"] <= valid["rolling_correlation"]).all())
        self.assertTrue((valid["rolling_ci_high"] >= valid["rolling_correlation"]).all())

    def test_lead_lag_includes_pointwise_intervals(self):
        a = series([100, 101, 103, 102, 105, 107, 106, 109, 110, 112, 111, 114, 116, 115, 118])
        b = series([50, 50.5, 51.5, 51, 52.5, 53.5, 53, 54.5, 55, 56, 55.5, 57, 58, 57.5, 59])
        result = lead_lag_statistics(a, b, window=1, max_lag=2)
        self.assertIn("ci_low", result.columns)
        self.assertIn("ci_high", result.columns)
        for _, row in result.dropna(subset=["correlation", "ci_low", "ci_high"]).iterrows():
            self.assertLessEqual(row["ci_low"], row["correlation"])
            self.assertGreaterEqual(row["ci_high"], row["correlation"])

    def test_event_study_adds_block_bootstrap_difference_interval(self):
        import numpy as np
        n = 160
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        primary_values = [100.0]
        target_values = [100.0]
        rng = np.random.default_rng(44)
        for idx in range(1, n):
            primary_step = 3.2 if idx % 12 == 0 else (-3.1 if idx % 17 == 0 else float(rng.normal(0, 0.3)))
            target_step = float(rng.normal(0.05, 0.7))
            primary_values.append(primary_values[-1] * (1 + primary_step / 100.0))
            target_values.append(target_values[-1] * (1 + target_step / 100.0))
        primary = pd.DataFrame({"date": dates, "value": primary_values})
        target = pd.DataFrame({"date": dates, "value": target_values})
        summary, _ = event_study(primary, target, measurement_window=1, threshold=2.0, horizons=[1, 3, 5])
        self.assertIn("block_bootstrap_difference_ci_low", summary.columns)
        self.assertIn("block_bootstrap_difference_ci_high", summary.columns)
        self.assertIn("bootstrap_valid_replicates", summary.columns)
        populated = summary.loc[summary["bootstrap_valid_replicates"] >= 50]
        self.assertGreater(len(populated), 0)
        self.assertTrue((populated["block_bootstrap_difference_ci_low"] <= populated["mean_difference_vs_baseline"]).all())
        self.assertTrue((populated["block_bootstrap_difference_ci_high"] >= populated["mean_difference_vs_baseline"]).all())

    def test_distribution_modes_are_independent_of_series_units(self):
        prices = series([100, 105, 103, 110, 111])
        flow = series([-10, 2, -4, 8, 0])
        changes = distribution_frame(prices, mode="percentage_change")
        flow_changes = distribution_frame(flow, mode="absolute_change")
        self.assertAlmostEqual(changes["value"].iloc[2], (103 / 105 - 1) * 100)
        self.assertEqual(flow_changes["value"].iloc[-1], -8)

    def test_distribution_interpretation_mentions_sample_and_mode(self):
        frame = series([100, 101, 99, 102, 98, 103, 97, 104, 96, 105, 95, 106, 94, 107, 93, 108, 92, 109, 91, 110, 90])
        summary = distribution_summary(distribution_frame(frame, "percentage_change"))
        text = distribution_interpretation(summary, "percentage_change")
        self.assertIn("observations", text.lower())
        self.assertIn("percentage", text.lower())
        self.assertIn("not a forecast", text.lower())

    def test_event_study_compares_events_with_non_event_baseline(self):
        primary = series([100, 103, 106, 103, 102, 101, 106, 112, 110, 109, 108, 112, 117, 115, 114, 116])
        target = series([100, 101, 102, 104, 105, 104, 106, 105, 107, 108, 106, 109, 111, 110, 112, 113])
        summary, events = event_study(primary, target, measurement_window=1, threshold=2, horizons=[1, 3])
        self.assertIn("baseline_mean", summary.columns)
        self.assertIn("mean_difference_vs_baseline", summary.columns)
        self.assertIn("baseline_observations", summary.columns)
        self.assertTrue((summary["baseline_observations"] >= 0).all())
        self.assertIn("response", events.columns)



if __name__ == "__main__":
    unittest.main()
