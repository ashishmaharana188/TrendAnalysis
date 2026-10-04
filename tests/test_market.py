from data_access.market import (
    get_daily_history,
    get_latest_daily_row,
    get_next_trading_day,
    get_price_on_or_before,
    get_market_date_range,
)


def main() -> None:
    ticker = "RELIANCE"

    print("=" * 60)
    print("MARKET DATA TEST")
    print("=" * 60)

    print("\n1. Date range")
    print(get_market_date_range(ticker))

    print("\n2. Latest row")
    print(get_latest_daily_row(ticker))

    print("\n3. Latest row as of 2026-10-02")
    print(get_latest_daily_row(ticker, "2026-10-02"))

    print("\n4. Next trading day after 2026-10-02")
    print(get_next_trading_day(ticker, "2026-10-02"))

    print("\n5. Price on/before 2026-09-30")
    print(get_price_on_or_before(ticker, "2026-09-30"))

    print("\n6. Historical data sample")
    table = get_daily_history(
        ticker,
        start_date="2026-09-01",
        end_date="2026-10-02",
    )

    print(f"Rows: {table.num_rows}")
    print(table)


if __name__ == "__main__":
    main()