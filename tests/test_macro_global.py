from data_access.macro_global import (
    get_macro_history,
    get_latest_macro,
    get_macro_indicators,
    get_macro_date_range,
    get_global_asset_history,
    get_latest_global_asset,
    get_global_assets,
    get_global_asset_date_range,
)


def print_table_sample(title: str, table, rows: int = 5) -> None:
    print(f"\n{title}")
    print(f"Rows: {table.num_rows}")
    print(table.slice(0, min(rows, table.num_rows)))


def main() -> None:
    print("=" * 70)
    print("PHASE 2.8 - MACRO + GLOBAL ASSET DATA ACCESS TEST")
    print("=" * 70)

    # ========================================================
    # MACRO DATA
    # Source: macro_daily_ledger
    # ========================================================

    print("\n" + "=" * 70)
    print("MACRO DATA")
    print("=" * 70)

    indicators = get_macro_indicators()

    print(f"\n1. Available macro indicators")
    print(f"Count: {len(indicators)}")
    print(indicators)

    # --------------------------------------------------------
    # Test one known macro indicator
    # --------------------------------------------------------

    indicator = "US_Dollar_Index"

    print(f"\n2. {indicator} coverage")
    print(get_macro_date_range(indicator))

    print(f"\n3. Latest {indicator}")
    print(get_latest_macro(indicator))

    print(f"\n4. Latest {indicator} as of 2026-07-08")
    print(
        get_latest_macro(
            indicator,
            as_of_date="2026-07-08",
        )
    )

    print(f"\n5. {indicator} recent history")
    macro_history = get_macro_history(
        indicator,
        start_date="2026-06-01",
        end_date="2026-07-08",
    )

    print_table_sample(
        f"{indicator} history",
        macro_history,
    )

    # --------------------------------------------------------
    # Test another known macro/commodity variable
    # --------------------------------------------------------

    crude_indicator = "Brent_Crude"

    print(f"\n6. {crude_indicator} coverage")
    print(get_macro_date_range(crude_indicator))

    print(f"\n7. Latest {crude_indicator}")
    print(get_latest_macro(crude_indicator))

    print(f"\n8. {crude_indicator} recent history")
    crude_history = get_macro_history(
        crude_indicator,
        start_date="2026-06-01",
        end_date="2026-07-08",
    )

    print_table_sample(
        f"{crude_indicator} history",
        crude_history,
    )

    # ========================================================
    # COMPANY DATA
    # Source: global_assets_daily
    # ========================================================

    print("\n" + "=" * 70)
    print("COMPANY / GLOBAL ASSET DATA")
    print("=" * 70)

    ticker = "RELIANCE"

    print(f"\n9. {ticker} coverage")
    print(get_global_asset_date_range(ticker))

    print(f"\n10. Latest {ticker}")
    print(get_latest_global_asset(ticker))

    print(f"\n11. Latest {ticker} as of 2026-07-08")
    print(
        get_latest_global_asset(
            ticker,
            as_of_date="2026-07-08",
        )
    )

    print(f"\n12. {ticker} recent history")
    company_history = get_global_asset_history(
        ticker,
        start_date="2026-06-01",
        end_date="2026-07-08",
    )

    print_table_sample(
        f"{ticker} history",
        company_history,
    )

    # --------------------------------------------------------
    # Available companies / global assets
    # --------------------------------------------------------

    print("\n13. Available global assets / companies")

    assets = get_global_assets()

    print(f"Count: {len(assets)}")

    for asset in assets[:20]:
        print(asset)

    print("\n" + "=" * 70)
    print("PHASE 2.8 TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()