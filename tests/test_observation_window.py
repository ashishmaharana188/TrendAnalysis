from datetime import date

from analysis.config import AnalysisConfig
from analysis.window import build_observation_window


def main() -> None:
    print("=" * 70)
    print("PHASE 3.2 - OBSERVATION WINDOW")
    print("=" * 70)

    config = AnalysisConfig(
        analysis_date=date(2026, 10, 3),
        analysis_timeframe="6M",
        holding_period_months=1.0,
        benchmark="Nifty_50",
        entry_mode="next_trading_day",
    )

    window = build_observation_window(
        ticker="RELIANCE.NS",
        config=config,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n1. Window summary")

    for key, value in window.summary().items():
        print(f"   {key}: {value}")

    # --------------------------------------------------------
    # Company
    # --------------------------------------------------------

    print("\n2. Company")

    print(
        f"   Ticker  : {window.ticker}"
    )

    print(
        f"   Sector  : {window.sector}"
    )

    print(
        f"   Industry: {window.industry}"
    )

    # --------------------------------------------------------
    # Market
    # --------------------------------------------------------

    print("\n3. Company market")

    print(
        f"   Rows: {window.company_market.num_rows}"
    )

    print(
        f"   Columns: "
        f"{window.company_market.column_names}"
    )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print("\n4. Benchmark")

    print(
        f"   Identifier: "
        f"{window.benchmark['identifier']}"
    )

    print(
        f"   Source: "
        f"{window.benchmark['source']}"
    )

    print(
        f"   Rows: "
        f"{window.benchmark['history'].num_rows}"
    )

    # --------------------------------------------------------
    # Macro
    # --------------------------------------------------------

    print("\n5. Macro")

    print(
        f"   Series: "
        f"{len(window.macro)}"
    )

    for indicator, table in list(
        window.macro.items()
    )[:10]:
        print(
            f"   {indicator}: {table.num_rows} rows"
        )

    # --------------------------------------------------------
    # Financials
    # --------------------------------------------------------

    print("\n6. Financials")

    print(
        "   Quarterly income:",
        window.quarterly_income.num_rows,
    )

    print(
        "   Yearly income:",
        window.yearly_income.num_rows,
    )

    print(
        "   Quarterly balance:",
        window.quarterly_balance_sheet.num_rows,
    )

    print(
        "   Yearly balance:",
        window.yearly_balance_sheet.num_rows,
    )

    print(
        "   Quarterly cash flow:",
        window.quarterly_cash_flow.num_rows,
    )

    print(
        "   Yearly cash flow:",
        window.yearly_cash_flow.num_rows,
    )

    print(
        "   Yearly indirect cash flow:",
        window.yearly_indirect_cash_flow.num_rows,
    )

    # --------------------------------------------------------
    # Cutoff validation
    # --------------------------------------------------------

    print("\n7. Temporal cutoff")

    print(
        "   Analysis date:",
        config.analysis_date,
    )

    print(
        "   Data cutoff:",
        config.data_cutoff_date,
    )

    assert (
        config.data_cutoff_date
        < config.analysis_date
    )

    print("   PASS")

    # --------------------------------------------------------
    # Limitations
    # --------------------------------------------------------

    print("\n8. Limitations")

    for limitation in window.limitations:
        print(
            f"   - {limitation}"
        )

    print("\n" + "=" * 70)
    print("PHASE 3.2 TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()