from __future__ import annotations

from analysis.real_olap_validation import (
    RealOLAPValidationConfig,
    validate_real_olap_relationships,
)


def main() -> None:
    config = RealOLAPValidationConfig(
        ticker="RELIANCE",
        benchmark="Nifty_50",
        analysis_timeframe="6M",
        holding_period_months=1.0,
        entry_mode="next_trading_day",
        max_folds=250,
        step_trading_days=5,
        min_training_observations=12,
        hardened_validation=True,
    )

    result = validate_real_olap_relationships(config)

    assert result.panel_rows >= config.min_training_observations
    assert result.walk_forward_folds > 0
    assert result.leakage_violations == 0
    assert result.state_future_violations == 0
    assert result.outcome_temporal_violations == 0
    assert result.latest_market_date is not None
    assert result.first_prediction_date is not None

    print("PHASE 4.10 HARDENED REAL OLAP INTEGRITY VALIDATION: PASS")
    print("Ticker:", result.ticker)
    print("Benchmark:", result.benchmark)
    print("Panel rows:", result.panel_rows)
    print("Candidate prediction observations:", result.candidate_observations)
    print("Valid state observations:", result.valid_state_observations)
    print("Walk-forward folds:", result.walk_forward_folds)
    print("Method A hit rate:", round(result.method_a_hit_rate_pct, 2))
    print("Method B hit rate:", round(result.method_b_hit_rate_pct, 2))
    print("Combined hit rate:", round(result.combined_hit_rate_pct, 2))
    print("Method A directional predictions:", result.method_a_directional_predictions)
    print("Method B directional predictions:", result.method_b_directional_predictions)
    print("Combined directional predictions:", result.combined_directional_predictions)
    print("Selection candidate evaluations:", result.selection_candidate_evaluations)
    print("Selection-validated predictions:", result.selection_validated_predictions)
    print("FDR-controlled folds:", result.multiple_testing_controlled_folds)
    print("Purged training observations:", result.purged_training_observations)
    print("Unknown overlap observations:", result.unknown_overlap_observations)
    print("Industry constituents:", result.industry_constituents)
    print("Sector constituents:", result.sector_constituents)
    print("Macro series:", result.macro_series)
    print("Global series:", result.global_series)
    print("Leakage violations:", result.leakage_violations)
    print("State future violations:", result.state_future_violations)
    print("Outcome temporal violations:", result.outcome_temporal_violations)
    print("Latest market date:", result.latest_market_date)
    print("State surface coverage:", result.state_surface_coverage_pct)
    print("Missing state families:", list(result.missing_state_families))
    print("Timing-limited state families:", list(result.timing_limited_state_families))
    print("Financial timing-limited observations:", result.financial_timing_limited_observations)
    print("Broad relationship surface validated:", result.broad_relationship_surface_validated)


if __name__ == "__main__":
    main()
