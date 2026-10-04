from data_access.institutional import (
    get_institutional_history,
    get_institutional_flow_history,
    get_latest_institutional_flow,
)

from data_access.derivatives import (
    get_options_history,
    get_latest_options,
    get_futures_basis_history,
    get_latest_futures_basis,
    get_unified_market_matrix,
    get_latest_unified_market_matrix,
)

from data_access.trade_events import (
    get_trade_events,
    get_latest_trade_events,
)


def main() -> None:
    ticker = "RELIANCE"

    print("=" * 70)
    print("PHASE 2.9 - INSTITUTIONAL / DERIVATIVES / TRADE EVENTS")
    print("=" * 70)

    # ========================================================
    # INSTITUTIONAL
    # ========================================================

    print("\n1. Raw institutional history")

    institutional = get_institutional_history(
        start_date="2026-06-01",
        end_date="2026-07-08",
    )

    print(f"Rows: {institutional.num_rows}")
    print(institutional.slice(0, min(5, institutional.num_rows)))

    print("\n2. Derived institutional flow")

    flow = get_institutional_flow_history(
        start_date="2026-06-01",
        end_date="2026-07-08",
    )

    print(f"Rows: {flow.num_rows}")
    print(flow.slice(0, min(5, flow.num_rows)))

    print("\n3. Latest institutional flow as of 2026-07-08")

    print(
        get_latest_institutional_flow(
            as_of_date="2026-07-08"
        )
    )

    # ========================================================
    # OPTIONS
    # ========================================================

    print(f"\n4. {ticker} options history")

    options = get_options_history(
        ticker,
        start_date="2026-06-01",
        end_date="2026-07-08",
    )

    print(f"Rows: {options.num_rows}")
    print(options.slice(0, min(5, options.num_rows)))

    print(f"\n5. Latest {ticker} options as of 2026-07-08")

    latest_options = get_latest_options(
        ticker,
        as_of_date="2026-07-08",
    )

    print(f"Rows: {latest_options.num_rows}")
    print(latest_options.slice(0, min(5, latest_options.num_rows)))

    # ========================================================
    # FUTURES BASIS
    # ========================================================

    print(f"\n6. {ticker} futures basis history")

    basis = get_futures_basis_history(
        ticker,
        start_date="2026-06-01",
        end_date="2026-07-08",
    )

    print(f"Rows: {basis.num_rows}")
    print(basis.slice(0, min(5, basis.num_rows)))

    print(f"\n7. Latest {ticker} futures basis")

    print(
        get_latest_futures_basis(
            ticker,
            as_of_date="2026-07-08",
        )
    )

    # ========================================================
    # UNIFIED MATRIX
    # ========================================================

    print(f"\n8. {ticker} unified market matrix")

    matrix = get_unified_market_matrix(
        ticker,
        start_date="2026-06-01",
        end_date="2026-07-08",
    )

    print(f"Rows: {matrix.num_rows}")
    print(matrix.slice(0, min(5, matrix.num_rows)))

    print(f"\n9. Latest {ticker} unified matrix")

    print(
        get_latest_unified_market_matrix(
            ticker,
            as_of_date="2026-07-08",
        )
    )

    # ========================================================
    # TRADE EVENTS
    # ========================================================

    print(f"\n10. {ticker} trade events")

    events = get_trade_events(
        ticker=ticker,
        start_date="2026-06-01",
        end_date="2026-07-08",
    )

    print(f"Rows: {events.num_rows}")
    print(events.slice(0, min(5, events.num_rows)))

    print(f"\n11. Latest {ticker} trade events")

    latest_events = get_latest_trade_events(
        ticker,
        as_of_date="2026-07-08",
    )

    print(f"Rows: {latest_events.num_rows}")
    print(latest_events.slice(0, min(5, latest_events.num_rows)))

    print("\n" + "=" * 70)
    print("PHASE 2.9 TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()