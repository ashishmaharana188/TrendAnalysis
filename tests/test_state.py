from datetime import date

from analysis.config import AnalysisConfig
from analysis.state import build_variable_state

from data_access.market import get_daily_history
from data_access.macro_global import (
    get_macro_history,
)


def main() -> None:

    print("=" * 70)
    print("PHASE 3.3 - VARIABLE STATE ENGINE")
    print("=" * 70)

    config = AnalysisConfig(
        analysis_date=date(2026, 10, 3),
        analysis_timeframe="6M",
        holding_period_months=1.0,
        benchmark="Nifty_50",
        entry_mode="next_trading_day",
    )

    # ========================================================
    # BRENT CRUDE
    # ========================================================

    print("\n1. Brent Crude")

    crude_selected = get_macro_history(
        "Brent_Crude",
        start_date=config.analysis_start_date,
        end_date=config.analysis_end_date,
    )

    crude_history = get_macro_history(
        "Brent_Crude",
        end_date=config.analysis_end_date,
    )

    crude_state = build_variable_state(
        variable="Brent_Crude",
        selected_table=crude_selected,
        historical_table=crude_history,
        value_column="close",
    )

    for key, value in crude_state.as_dict().items():
        print(f"   {key}: {value}")

    assert crude_state.historical_observations > 0

    assert (
        crude_state.current_value is not None
    )

    print("   PASS")

    # ========================================================
    # RELIANCE
    # ========================================================

    print("\n2. RELIANCE")

    reliance_selected = get_daily_history(
        "RELIANCE",
        start_date=config.analysis_start_date,
        end_date=config.analysis_end_date,
    )

    reliance_history = get_daily_history(
        "RELIANCE",
        end_date=config.analysis_end_date,
    )

    reliance_state = build_variable_state(
        variable="RELIANCE",
        selected_table=reliance_selected,
        historical_table=reliance_history,
        value_column="close",
    )

    for key, value in reliance_state.as_dict().items():
        print(f"   {key}: {value}")

    assert reliance_state.historical_observations > 0

    assert (
        reliance_state.current_value is not None
    )

    print("   PASS")

    # ========================================================
    # NIFTY
    # ========================================================

    print("\n3. Nifty 50")

    nifty_selected = get_macro_history(
        "Nifty_50",
        start_date=config.analysis_start_date,
        end_date=config.analysis_end_date,
    )

    nifty_history = get_macro_history(
        "Nifty_50",
        end_date=config.analysis_end_date,
    )

    nifty_state = build_variable_state(
        variable="Nifty_50",
        selected_table=nifty_selected,
        historical_table=nifty_history,
        value_column="close",
    )

    for key, value in nifty_state.as_dict().items():
        print(f"   {key}: {value}")

    assert nifty_state.historical_observations > 0

    print("   PASS")

    # ========================================================
    # CUTOFF CHECK
    # ========================================================

    print("\n4. Cutoff validation")

    print(
        f"   Analysis date : "
        f"{config.analysis_date}"
    )

    print(
        f"   Data cutoff   : "
        f"{config.data_cutoff_date}"
    )

    assert (
        crude_history["report_date"][-1].as_py()
        <= config.data_cutoff_date
    )

    print("   PASS")

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("PHASE 3.3 TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()