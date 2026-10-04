from datetime import date

from analysis.config import (
    AnalysisConfig,
    add_holding_period,
    subtract_timeframe,
    validate_holding_period,
)


def main() -> None:
    print("=" * 70)
    print("PHASE 3.1 - ANALYSIS CALENDAR / PARAMETER CONTRACT")
    print("=" * 70)

    # ========================================================
    # 1. Main configuration
    # ========================================================

    config = AnalysisConfig(
        analysis_date=date(2026, 10, 3),
        analysis_timeframe="6M",
        holding_period_months=1.0,
        benchmark="Nifty_50",
        entry_mode="next_trading_day",
    )

    print("\n1. Configuration")

    for key, value in config.summary().items():
        print(f"   {key}: {value}")

    assert config.analysis_date == date(
        2026, 10, 3
    )

    assert config.data_cutoff_date == date(
        2026, 10, 2
    )

    assert config.analysis_start_date == date(
        2026, 4, 3
    )

    assert config.analysis_end_date == date(
        2026, 10, 2
    )

    print("   PASS")

    # ========================================================
    # 2. Timeframe calculations
    # ========================================================

    print("\n2. Timeframe calculations")

    start = subtract_timeframe(
        date(2026, 10, 3),
        "1Y",
    )

    print(f"   1Y start: {start}")

    assert start == date(
        2025, 10, 3
    )

    start = subtract_timeframe(
        date(2026, 10, 3),
        "5Y",
    )

    print(f"   5Y start: {start}")

    assert start == date(
        2021, 10, 3
    )

    print("   PASS")

    # ========================================================
    # 3. Holding period
    # ========================================================

    print("\n3. Holding period calculations")

    entry_date = date(
        2026,
        10,
        5,
    )

    target = add_holding_period(
        entry_date,
        1.0,
    )

    print(
        f"   Entry: {entry_date}"
    )

    print(
        f"   1M target: {target}"
    )

    assert target == date(
        2026,
        11,
        5,
    )

    target = add_holding_period(
        entry_date,
        3.0,
    )

    print(
        f"   3M target: {target}"
    )

    assert target == date(
        2027,
        1,
        5,
    )

    target = add_holding_period(
        entry_date,
        0.1,
    )

    print(
        f"   0.1M target: {target}"
    )

    assert target == date(
        2026,
        10,
        8,
    )

    print("   PASS")

    # ========================================================
    # 4. Boundary validation
    # ========================================================

    print("\n4. Holding-period validation")

    validate_holding_period(0.1)
    validate_holding_period(12.0)

    try:
        validate_holding_period(0.09)
    except ValueError:
        print("   0.09M correctly rejected")
    else:
        raise AssertionError(
            "0.09M should have been rejected"
        )

    try:
        validate_holding_period(12.1)
    except ValueError:
        print("   12.1M correctly rejected")
    else:
        raise AssertionError(
            "12.1M should have been rejected"
        )

    print("   PASS")

    # ========================================================
    # 5. No-look-ahead rule
    # ========================================================

    print("\n5. No-look-ahead rule")

    assert (
        config.data_cutoff_date
        < config.analysis_date
    )

    print(
        "   Analysis date :",
        config.analysis_date,
    )

    print(
        "   Allowed daily data through:",
        config.data_cutoff_date,
    )

    print("   PASS")

    # ========================================================
    # Complete
    # ========================================================

    print("\n" + "=" * 70)
    print("PHASE 3.1 TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()