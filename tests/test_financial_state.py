from datetime import date

from analysis.config import AnalysisConfig
from analysis.window import build_observation_window
from analysis.financial_state import (
    build_financial_states,
)


def main() -> None:

    print("=" * 70)
    print("PHASE 3.4 - FINANCIAL STATE ENGINE")
    print("=" * 70)

    config = AnalysisConfig(
        analysis_date=date(2026, 10, 3),
        analysis_timeframe="6M",
        holding_period_months=1.0,
        benchmark="Nifty_50",
        entry_mode="next_trading_day",
    )

    print("\n1. Building observation window")

    window = build_observation_window(
        ticker="RELIANCE.NS",
        config=config,
    )

    print(
        f"   Company: {window.ticker}"
    )

    print(
        f"   Timeframe: "
        f"{config.analysis_timeframe}"
    )

    # --------------------------------------------------------
    # Build states
    # --------------------------------------------------------

    print("\n2. Building financial states")

    states = build_financial_states(
        window
    )

    print(
        f"   Metrics built: {len(states)}"
    )

    # --------------------------------------------------------
    # Show selected metrics
    # --------------------------------------------------------

    metrics_to_show = [
        "TotalRevenue",
        "OperatingIncome",
        "NetIncome",
        "TotalAssets",
        "StockholdersEquity",
        "CashFromOperations",
        "NetBorrowing",
        "TotalOperatingCashFlow",
    ]

    print("\n3. Financial metric states")

    for metric in metrics_to_show:

        state = states[metric]

        print("\n" + "-" * 60)
        print(
            f"   {state.metric}"
        )

        print(
            f"   Current value      : "
            f"{state.current_value}"
        )

        print(
            f"   Recent start       : "
            f"{state.recent_start_value}"
        )

        print(
            f"   Recent change      : "
            f"{state.recent_change}"
        )

        print(
            f"   Recent change %    : "
            f"{state.recent_change_pct}"
        )

        print(
            f"   Previous comparable: "
            f"{state.previous_comparable_value}"
        )

        print(
            f"   Latest comparable  : "
            f"{state.latest_comparable_value}"
        )

        print(
            f"   Comparable change  : "
            f"{state.comparable_change}"
        )

        print(
            f"   Comparable change %: "
            f"{state.comparable_change_pct}"
        )

        print(
            f"   Historical position: "
            f"{state.historical_percentile}"
        )

        print(
            f"   Direction          : "
            f"{state.recent_direction}"
        )

        print(
            f"   Direction strength : "
            f"{state.direction_strength}"
        )

        print(
            f"   State              : "
            f"{state.state}"
        )

        print(
            f"   Source frequency   : "
            f"{state.source_frequency}"
        )

        print(
            f"   Limited            : "
            f"{state.limited}"
        )

        # Basic sanity checks
        assert state.metric == metric
        assert state.historical_periods >= 0

    # --------------------------------------------------------
    # Mandatory yearly baseline
    # --------------------------------------------------------

    print("\n4. Mandatory yearly baseline")

    assert (
        window.yearly_income.num_rows > 0
    )

    assert (
        window.yearly_balance_sheet.num_rows > 0
    )

    assert (
        window.yearly_cash_flow.num_rows > 0
    )

    assert (
        window.yearly_indirect_cash_flow.num_rows > 0
    )

    print(
        "   Yearly income       : PASS"
    )

    print(
        "   Yearly balance sheet: PASS"
    )

    print(
        "   Yearly cash flow    : PASS"
    )

    print(
        "   Yearly indirect CF  : PASS"
    )

    # --------------------------------------------------------
    # Quarterly optionality
    # --------------------------------------------------------

    print("\n5. Quarterly optionality")

    if window.quarterly_cash_flow.num_rows == 0:

        print(
            "   Quarterly cash flow: unavailable"
        )

        cash_flow_state = states[
            "CashFromOperations"
        ]

        assert (
            cash_flow_state.source_frequency
            == "yearly_only"
        )

        print(
            "   Yearly fallback: PASS"
        )

    else:

        print(
            "   Quarterly cash flow: available"
        )

        cash_flow_state = states[
            "CashFromOperations"
        ]

        assert (
            cash_flow_state.source_frequency
            == "quarterly_recent_yearly_baseline"
        )

        print(
            "   Quarterly preference: PASS"
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 3.4 TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()