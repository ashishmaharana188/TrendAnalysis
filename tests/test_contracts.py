from data_access.contracts import (
    normalize_ticker,
    normalize_indicator,
    normalize_date,
    validate_market_result,
    validate_macro_result,
    validate_options_result,
    validate_basis_result,
)

from data_access.market import get_daily_history
from data_access.macro_global import get_macro_history
from data_access.derivatives import (
    get_options_history,
    get_futures_basis_history,
)


def main() -> None:
    print("=" * 70)
    print("PHASE 2.10 - DATA CONTRACTS / NORMALIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Identifier normalization
    # --------------------------------------------------------

    print("\n1. Ticker normalization")

    ticker = normalize_ticker(" reliance ")

    print(f"   Input : ' reliance '")
    print(f"   Output: {ticker}")

    assert ticker == "RELIANCE"

    print("   PASS")

    print("\n2. Indicator normalization")

    indicator = normalize_indicator(" Brent_Crude ")

    print("   Input : ' Brent_Crude '")
    print(f"   Output: {indicator}")

    assert indicator == "Brent_Crude"

    print("   PASS")

    # --------------------------------------------------------
    # Date normalization
    # --------------------------------------------------------

    print("\n3. Date normalization")

    normalized_date = normalize_date("2026-10-02")

    print("   Input : '2026-10-02'")
    print(f"   Output: {normalized_date}")

    assert normalized_date.isoformat() == "2026-10-02"

    print("   PASS")

    # --------------------------------------------------------
    # Market
    # --------------------------------------------------------

    print("\n4. Market repository result")

    market = get_daily_history(
        "RELIANCE",
        start_date="2026-06-01",
        end_date="2026-07-08",
    )

    print(f"   Rows: {market.num_rows}")
    print(f"   Columns: {market.column_names}")

    validate_market_result(market)

    print("   PASS")

    # --------------------------------------------------------
    # Macro
    # --------------------------------------------------------

    print("\n5. Macro repository result")

    macro = get_macro_history(
        "Brent_Crude",
        start_date="2026-06-01",
        end_date="2026-07-08",
    )

    print(f"   Rows: {macro.num_rows}")
    print(f"   Columns: {macro.column_names}")

    validate_macro_result(macro)

    print("   PASS")

    # --------------------------------------------------------
    # Options
    # --------------------------------------------------------

    print("\n6. Options repository result")

    options = get_options_history(
        "RELIANCE",
        start_date="2026-06-01",
        end_date="2026-07-08",
    )

    print(f"   Rows: {options.num_rows}")
    print(f"   Columns: {options.column_names}")

    validate_options_result(options)

    print("   PASS")

    # --------------------------------------------------------
    # Futures basis
    # --------------------------------------------------------

    print("\n7. Futures basis repository result")

    basis = get_futures_basis_history(
        "RELIANCE",
        start_date="2026-06-01",
        end_date="2026-07-08",
    )

    print(f"   Rows: {basis.num_rows}")
    print(f"   Columns: {basis.column_names}")

    validate_basis_result(basis)

    print("   PASS")

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 2.10 TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()