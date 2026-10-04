from data_access.financials import (
    get_latest_quarterly_income,
    get_latest_quarterly_balance_sheet,
    get_latest_quarterly_cash_flow,

    get_latest_yearly_income,
    get_latest_yearly_balance_sheet,
    get_latest_yearly_cash_flow,
    get_latest_yearly_indirect_cash_flow,

    get_financial_coverage,
)


def main() -> None:

    ticker = "RELIANCE"

    print("=" * 70)
    print("PHASE 2.7 - FINANCIAL DATA REPOSITORY TEST")
    print("=" * 70)

    # ========================================================
    # 1. COVERAGE
    # ========================================================

    print("\n1. Financial coverage")

    coverage = get_financial_coverage(
        ticker
    )

    for table, info in coverage.items():

        print(
            f"   {table}: "
            f"{info['row_count']} rows | "
            f"{info['first_report_date']} -> "
            f"{info['last_report_date']}"
        )

    # ========================================================
    # 2. YEARLY INCOME - MANDATORY
    # ========================================================

    print("\n2. Latest yearly income")

    yearly_income = get_latest_yearly_income(
        ticker,
        periods=5,
    )

    print(
        f"   Rows: {yearly_income.num_rows}"
    )

    print(
        f"   Columns: {yearly_income.column_names}"
    )

    assert yearly_income.num_rows > 0, (
        "Mandatory yearly income data is missing."
    )

    # ========================================================
    # 3. YEARLY BALANCE SHEET - MANDATORY
    # ========================================================

    print("\n3. Latest yearly balance sheet")

    yearly_balance = (
        get_latest_yearly_balance_sheet(
            ticker,
            periods=5,
        )
    )

    print(
        f"   Rows: {yearly_balance.num_rows}"
    )

    print(
        f"   Columns: {yearly_balance.column_names}"
    )

    assert yearly_balance.num_rows > 0, (
        "Mandatory yearly balance-sheet data is missing."
    )

    # ========================================================
    # 4. YEARLY CASH FLOW - MANDATORY
    # ========================================================

    print("\n4. Latest yearly cash flow")

    yearly_cash_flow = (
        get_latest_yearly_cash_flow(
            ticker,
            periods=5,
        )
    )

    print(
        f"   Rows: {yearly_cash_flow.num_rows}"
    )

    print(
        f"   Columns: {yearly_cash_flow.column_names}"
    )

    assert yearly_cash_flow.num_rows > 0, (
        "Mandatory yearly cash-flow data is missing."
    )

    # ========================================================
    # 5. YEARLY INDIRECT CASH FLOW - MANDATORY
    # ========================================================

    print("\n5. Latest yearly indirect cash flow")

    yearly_indirect = (
        get_latest_yearly_indirect_cash_flow(
            ticker,
            periods=5,
        )
    )

    print(
        f"   Rows: {yearly_indirect.num_rows}"
    )

    print(
        f"   Columns: {yearly_indirect.column_names}"
    )

    assert yearly_indirect.num_rows > 0, (
        "Mandatory yearly indirect cash-flow "
        "data is missing."
    )

    # ========================================================
    # 6. QUARTERLY INCOME - OPTIONAL
    # ========================================================

    print("\n6. Latest quarterly income")

    quarterly_income = (
        get_latest_quarterly_income(
            ticker,
            periods=5,
        )
    )

    print(
        f"   Rows: {quarterly_income.num_rows}"
    )

    if quarterly_income.num_rows == 0:
        print(
            "   Quarterly income unavailable "
            "(allowed)."
        )
    else:
        print(
            f"   Columns: "
            f"{quarterly_income.column_names}"
        )

    # ========================================================
    # 7. QUARTERLY BALANCE - OPTIONAL
    # ========================================================

    print("\n7. Latest quarterly balance sheet")

    quarterly_balance = (
        get_latest_quarterly_balance_sheet(
            ticker,
            periods=5,
        )
    )

    print(
        f"   Rows: {quarterly_balance.num_rows}"
    )

    if quarterly_balance.num_rows == 0:
        print(
            "   Quarterly balance sheet unavailable "
            "(allowed)."
        )
    else:
        print(
            f"   Columns: "
            f"{quarterly_balance.column_names}"
        )

    # ========================================================
    # 8. QUARTERLY CASH FLOW - OPTIONAL
    # ========================================================

    print("\n8. Latest quarterly cash flow")

    quarterly_cash_flow = (
        get_latest_quarterly_cash_flow(
            ticker,
            periods=5,
        )
    )

    print(
        f"   Rows: {quarterly_cash_flow.num_rows}"
    )

    if quarterly_cash_flow.num_rows == 0:
        print(
            "   Quarterly cash flow unavailable "
            "(allowed)."
        )
    else:
        print(
            f"   Columns: "
            f"{quarterly_cash_flow.column_names}"
        )

    # ========================================================
    # 9. FINAL RULE CHECK
    # ========================================================

    print("\n9. Financial availability rules")

    print(
        "   Yearly income           : REQUIRED"
    )

    print(
        "   Yearly balance sheet    : REQUIRED"
    )

    print(
        "   Yearly cash flow        : REQUIRED"
    )

    print(
        "   Yearly indirect CF      : REQUIRED"
    )

    print(
        "   Quarterly income        : OPTIONAL"
    )

    print(
        "   Quarterly balance sheet: OPTIONAL"
    )

    print(
        "   Quarterly cash flow     : OPTIONAL"
    )

    print("   PASS")

    print("\n" + "=" * 70)
    print("PHASE 2.7 FINANCIAL TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()