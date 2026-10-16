from __future__ import annotations

import unittest

import pandas as pd

from app.services.statistics_service import (
    change_series,
    event_study,
    lead_lag_statistics,
    pair_statistics,
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


if __name__ == "__main__":
    unittest.main()
