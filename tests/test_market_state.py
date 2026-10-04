from datetime import date

from analysis.config import AnalysisConfig
from analysis.window import build_observation_window
from analysis.market_state import (
    build_company_market_states,
)


def main() -> None:

    print("=" * 70)
    print("PHASE 3.5 - COMPANY MARKET STATE")
    print("=" * 70)

    config = AnalysisConfig(
        analysis_date=date(2026, 10, 3),
        analysis_timeframe="6M",
        holding_period_months=1.0,
        benchmark="Nifty_50",
        entry_mode="next_trading_day",
    )

    # --------------------------------------------------------
    # Observation window
    # --------------------------------------------------------

    print("\n1. Building observation window")

    window = build_observation_window(
        ticker="RELIANCE.NS",
        config=config,
    )

    print(
        f"   Selected rows   : "
        f"{window.company_market.num_rows}"
    )

    print(
        f"   Historical rows : "
        f"{window.company_market_history.num_rows}"
    )

    print(
        f"   Selected columns: "
        f"{window.company_market.column_names}"
    )

    print(
        f"   Historical columns: "
        f"{window.company_market_history.column_names}"
    )

    # --------------------------------------------------------
    # Required checks before state engine
    # --------------------------------------------------------

    assert window.company_market.num_rows > 0, (
        "Selected company market history is empty."
    )

    assert window.company_market_history.num_rows > 0, (
        "Historical company market history is empty."
    )

    assert "close" in window.company_market_history.column_names

    # --------------------------------------------------------
    # State engine
    # --------------------------------------------------------

    print("\n2. Building market states")

    states = build_company_market_states(
        ticker=window.ticker,
        selected_table=window.company_market,
        historical_table=window.company_market_history,
    )

    print(
        f"   Metrics: {len(states)}"
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n3. Market states")

    for metric, state in states.items():

        print("\n" + "-" * 60)

        print(f"   {metric}")
        print(
            f"   Current value        : "
            f"{state.current_value}"
        )
        print(
            f"   Selected start       : "
            f"{state.selected_start_value}"
        )
        print(
            f"   Change               : "
            f"{state.absolute_change}"
        )
        print(
            f"   Change %             : "
            f"{state.relative_change_pct}"
        )
        print(
            f"   Historical percentile: "
            f"{state.historical_percentile}"
        )
        print(
            f"   Direction            : "
            f"{state.recent_direction}"
        )
        print(
            f"   Direction strength   : "
            f"{state.direction_strength}"
        )
        print(
            f"   State                : "
            f"{state.state}"
        )
        print(
            f"   Selected observations: "
            f"{state.selected_observations}"
        )
        print(
            f"   Historical observations: "
            f"{state.historical_observations}"
        )
        print(
            f"   Limited              : "
            f"{state.limited}"
        )

    # --------------------------------------------------------
    # Required metrics
    # --------------------------------------------------------

    print("\n4. Required market states")

    required = {
        "price",
        "volume",
        "daily_range",
        "volatility_20d",
    }

    assert required.issubset(states.keys()), (
    f"Missing market states: "
    f"{sorted(required - set(states.keys()))}"
)

    price_state = states["price"]

    assert price_state.historical_observations > 0
    assert price_state.current_value is not None

    print("   Price history: PASS")
    print("   Required metrics: PASS")

    # --------------------------------------------------------
    # Cutoff
    # --------------------------------------------------------

    print("\n5. Cutoff validation")

    print(
        f"   Analysis date: "
        f"{config.analysis_date}"
    )

    print(
        f"   Data cutoff: "
        f"{config.data_cutoff_date}"
    )

    historical_dates = (
        window.company_market_history[
            "report_date"
        ].to_pylist()
    )

    latest_date = max(historical_dates)

    print(
        f"   Latest historical observation: "
        f"{latest_date}"
    )

    assert latest_date <= config.data_cutoff_date

    print("   PASS")

    print("\n" + "=" * 70)
    print("PHASE 3.5 TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()