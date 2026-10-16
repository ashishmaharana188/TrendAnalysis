from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_access.financials import (
    get_quarterly_balance_sheet_history,
    get_quarterly_cash_flow_history,
    get_quarterly_income_history,
    get_yearly_balance_sheet_history,
    get_yearly_cash_flow_history,
    get_yearly_income_history,
    get_yearly_indirect_cash_flow_history,
)
from data_access.institutional import get_institutional_flow_history
from data_access.macro_global import get_global_assets, get_macro_indicators
from data_access.metadata import get_active_companies
from data_access.query import execute_rows


SOURCE_MARKET = "Market asset / company price"
SOURCE_MACRO = "Macro / global indicator"
SOURCE_MATRIX = "Derivatives & microstructure matrix"
SOURCE_OPTIONS = "Options aggregates"
SOURCE_BASIS = "Futures basis"
SOURCE_INSTITUTIONAL = "Institutional flows"
SOURCE_Q_INCOME = "Quarterly income statement"
SOURCE_Y_INCOME = "Yearly income statement"
SOURCE_Q_BALANCE = "Quarterly balance sheet"
SOURCE_Y_BALANCE = "Yearly balance sheet"
SOURCE_Q_CASH = "Quarterly cash flow"
SOURCE_Y_CASH = "Yearly cash flow"
SOURCE_Y_INDIRECT = "Yearly indirect cash flow"

SOURCES = [
    SOURCE_MARKET,
    SOURCE_MACRO,
    SOURCE_MATRIX,
    SOURCE_OPTIONS,
    SOURCE_BASIS,
    SOURCE_INSTITUTIONAL,
    SOURCE_Q_INCOME,
    SOURCE_Y_INCOME,
    SOURCE_Q_BALANCE,
    SOURCE_Y_BALANCE,
    SOURCE_Q_CASH,
    SOURCE_Y_CASH,
    SOURCE_Y_INDIRECT,
]

@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    kind: str = "numeric"

@dataclass(frozen=True)
class SeriesRequest:
    source: str
    identifier: str
    metric: str

    @property
    def label(self) -> str:
        metric_label = metric_label_for(self.source, self.metric)
        return f"{self.source} · {self.identifier} · {metric_label}"


PRICE_METRICS = [
    Metric("close", "Close"),
    Metric("open", "Open"),
    Metric("high", "High"),
    Metric("low", "Low"),
    Metric("volume", "Volume"),
]

MATRIX_METRICS = [
    Metric("close", "Close"),
    Metric("volume", "Volume"),
    Metric("delivery_percentage", "Delivery percentage"),
    Metric("daily_hl_spread", "Daily high-low spread"),
    Metric("daily_vwap_dev", "Daily VWAP deviation"),
    Metric("oi_pcr", "Open-interest put/call ratio"),
    Metric("delta_oi_pcr", "Change in OI put/call ratio"),
    Metric("futures_basis", "Futures basis"),
    Metric("short_volume", "Short volume"),
    Metric("short_percentage", "Short percentage"),
    Metric("net_block_volume", "Net block volume"),
    Metric("avg_block_premium", "Average block premium"),
]

OPTIONS_METRICS = [
    Metric("total_put_oi", "Total put open interest"),
    Metric("total_call_oi", "Total call open interest"),
    Metric("oi_pcr", "Open-interest put/call ratio"),
    Metric("total_put_volume", "Total put volume"),
    Metric("total_call_volume", "Total call volume"),
    Metric("volume_pcr", "Volume put/call ratio"),
]

BASIS_METRICS = [
    Metric("spot_price", "Spot price"),
    Metric("futures_price", "Futures price"),
    Metric("open_interest", "Open interest"),
    Metric("absolute_basis", "Absolute basis"),
    Metric("basis_percentage", "Basis percentage"),
]

INSTITUTIONAL_METRICS = [
    # Raw daily ledger fields.
    Metric("cash_buy_value", "Cash buy value"),
    Metric("cash_sell_value", "Cash sell value"),
    Metric("cash_net_value", "Net cash value"),
    Metric("nifty_close", "Nifty close recorded in ledger"),
    Metric("future_index_long", "Index futures long contracts"),
    Metric("future_index_short", "Index futures short contracts"),
    Metric("future_stock_long", "Stock futures long contracts"),
    Metric("future_stock_short", "Stock futures short contracts"),
    Metric("option_index_call_long", "Index call long contracts"),
    Metric("option_index_put_long", "Index put long contracts"),
    Metric("option_index_call_short", "Index call short contracts"),
    Metric("option_index_put_short", "Index put short contracts"),
    Metric("option_stock_call_long", "Stock call long contracts"),
    Metric("option_stock_put_long", "Stock put long contracts"),
    Metric("option_stock_call_short", "Stock call short contracts"),
    Metric("option_stock_put_short", "Stock put short contracts"),
    Metric("total_long_contracts", "Total long contracts"),
    Metric("total_short_contracts", "Total short contracts"),
    # Pre-derived net-position fields.
    Metric("future_index_net", "Index futures net"),
    Metric("future_index_net_change", "Index futures net change"),
    Metric("option_index_call_net", "Index call net"),
    Metric("option_index_call_net_change", "Index call net change"),
    Metric("option_index_put_net", "Index put net"),
    Metric("option_index_put_net_change", "Index put net change"),
    Metric("future_stock_net", "Stock futures net"),
    Metric("future_stock_net_change", "Stock futures net change"),
    Metric("option_stock_call_net", "Stock call net"),
    Metric("option_stock_call_net_change", "Stock call net change"),
    Metric("option_stock_put_net", "Stock put net"),
    Metric("option_stock_put_net_change", "Stock put net change"),
]

FINANCIAL_METRICS: dict[str, list[Metric]] = {
    SOURCE_Q_INCOME: [
        Metric("TotalRevenue", "Total revenue"), Metric("CostOfRevenue", "Cost of revenue"),
        Metric("GrossProfit", "Gross profit"), Metric("OperatingExpense", "Operating expense"),
        Metric("OperatingIncome", "Operating income"), Metric("NetInterestIncome", "Net interest income"),
        Metric("TaxProvision", "Tax provision"), Metric("NetIncome", "Net income"),
    ],
    SOURCE_Y_INCOME: [
        Metric("TotalRevenue", "Total revenue"), Metric("CostOfRevenue", "Cost of revenue"),
        Metric("GrossProfit", "Gross profit"), Metric("OperatingExpense", "Operating expense"),
        Metric("OperatingIncome", "Operating income"), Metric("NetInterestIncome", "Net interest income"),
        Metric("TaxProvision", "Tax provision"), Metric("NetIncome", "Net income"),
    ],
    SOURCE_Q_BALANCE: [
        Metric("CashCashEquivalentsAndShortTermInvestments", "Cash and equivalents"),
        Metric("Inventory", "Inventory"), Metric("CurrentAssets", "Current assets"),
        Metric("TotalAssets", "Total assets"), Metric("CurrentLiabilities", "Current liabilities"),
        Metric("LongTermDebtAndCapitalLeaseObligation", "Long-term debt and lease obligations"),
        Metric("TotalLiabilitiesNetMinorityInterest", "Total liabilities"),
        Metric("StockholdersEquity", "Shareholders' equity"),
    ],
    SOURCE_Y_BALANCE: [
        Metric("CashCashEquivalentsAndShortTermInvestments", "Cash and equivalents"),
        Metric("Inventory", "Inventory"), Metric("CurrentAssets", "Current assets"),
        Metric("TotalAssets", "Total assets"), Metric("CurrentLiabilities", "Current liabilities"),
        Metric("LongTermDebtAndCapitalLeaseObligation", "Long-term debt and lease obligations"),
        Metric("TotalLiabilitiesNetMinorityInterest", "Total liabilities"),
        Metric("StockholdersEquity", "Shareholders' equity"),
    ],
    SOURCE_Q_CASH: [
        Metric("CashReceipts", "Cash receipts"), Metric("CashDisbursements", "Cash disbursements"),
        Metric("CashFromOperations", "Cash from operations"), Metric("FixedAssetPurchases", "Fixed-asset purchases"),
        Metric("NetBorrowing", "Net borrowing"), Metric("IncomeTaxPaid", "Income tax paid"),
        Metric("EndingCashBalance", "Ending cash balance"),
    ],
    SOURCE_Y_CASH: [
        Metric("CashReceipts", "Cash receipts"), Metric("CashDisbursements", "Cash disbursements"),
        Metric("CashFromOperations", "Cash from operations"), Metric("FixedAssetPurchases", "Fixed-asset purchases"),
        Metric("NetBorrowing", "Net borrowing"), Metric("IncomeTaxPaid", "Income tax paid"),
        Metric("EndingCashBalance", "Ending cash balance"),
    ],
    SOURCE_Y_INDIRECT: [
        Metric("TreasuryOpacityRatio", "Treasury opacity ratio"),
        Metric("NetIncome", "Net income"), Metric("TotalOperatingCashFlow", "Operating cash flow"),
        Metric("TotalInvestingCashFlow", "Investing cash flow"), Metric("TotalFinancingCashFlow", "Financing cash flow"),
        Metric("NetDebtIssuedRepaid", "Net debt issued / repaid"), Metric("NetChangeInCash", "Net change in cash"),
        Metric("EndingCash", "Ending cash"),
    ],
}


def metric_options(source: str) -> list[Metric]:
    if source in (SOURCE_MARKET, SOURCE_MACRO):
        return PRICE_METRICS
    if source == SOURCE_MATRIX:
        return MATRIX_METRICS
    if source == SOURCE_OPTIONS:
        return OPTIONS_METRICS
    if source == SOURCE_BASIS:
        return BASIS_METRICS
    if source == SOURCE_INSTITUTIONAL:
        return INSTITUTIONAL_METRICS
    return FINANCIAL_METRICS.get(source, [])


def metric_label_for(source: str, metric: str) -> str:
    for option in metric_options(source):
        if option.key == metric:
            return option.label
    return metric.replace("_", " ").replace("Camel", "")


def list_identifiers(source: str) -> list[dict[str, str]]:
    """List selectable identifiers using existing read-only data-access APIs."""
    if source == SOURCE_MACRO:
        return [{"value": item, "label": item} for item in get_macro_indicators()]

    if source == SOURCE_MARKET:
        # This source legitimately uses the general market asset catalogue.
        assets = get_global_assets()
        latest_by_ticker: dict[str, str] = {}
        for row in assets:
            ticker = str(row.get("ticker") or "").strip()
            if not ticker:
                continue
            latest = row.get("last_date")
            latest_text = latest.isoformat() if hasattr(latest, "isoformat") else str(latest or "")
            if ticker not in latest_by_ticker or latest_text > latest_by_ticker[ticker]:
                latest_by_ticker[ticker] = latest_text
        return [
            {
                "value": ticker,
                "label": f"{ticker} · latest {latest}" if latest else ticker,
            }
            for ticker, latest in sorted(latest_by_ticker.items(), key=lambda item: item[0].casefold())
        ]

    # These selectors must be sourced from the same views that supply their
    # plotted observations. Listing all global assets caused invalid choices
    # for options, futures and the precomputed derivatives/microstructure view.
    catalogue_queries = {
        SOURCE_MATRIX: ("mv_unified_market_matrix", "date"),
        SOURCE_OPTIONS: ("mv_options_aggregates", "ReportDate"),
        SOURCE_BASIS: ("mv_spot_futures_basis", "ReportDate"),
    }
    if source in catalogue_queries:
        table_name, date_column = catalogue_queries[source]
        sql = (
            f'SELECT ticker, MAX("{date_column}") AS latest_date '
            f'FROM "{table_name}" WHERE ticker IS NOT NULL '
            'GROUP BY ticker ORDER BY ticker'
        )
        rows = execute_rows(sql)
        choices = []
        for ticker, latest in rows:
            ticker_text = str(ticker or "").strip()
            if not ticker_text:
                continue
            latest_text = latest.isoformat() if hasattr(latest, "isoformat") else str(latest or "")
            label = f"{ticker_text} · latest {latest_text}" if latest_text else ticker_text
            choices.append({"value": ticker_text, "label": label})
        return choices

    if source in FINANCIAL_METRICS:
        companies = get_active_companies()
        seen_tickers: set[str] = set()
        choices = []
        for row in companies:
            ticker = str(row.get("Ticker") or "").strip()
            if ticker and ticker not in seen_tickers:
                seen_tickers.add(ticker)
                choices.append({"value": ticker, "label": ticker})
        return sorted(choices, key=lambda row: row["label"].casefold())

    if source == SOURCE_INSTITUTIONAL:
        table = get_institutional_flow_history()
        df = table.to_pandas()
        if df.empty or "client_type" not in df.columns:
            return []
        values = sorted({str(v) for v in df["client_type"].dropna().tolist()})
        return [{"value": item, "label": item} for item in values]

    return []


def get_financial_loader(source: str):
    mapping = {
        SOURCE_Q_INCOME: get_quarterly_income_history,
        SOURCE_Y_INCOME: get_yearly_income_history,
        SOURCE_Q_BALANCE: get_quarterly_balance_sheet_history,
        SOURCE_Y_BALANCE: get_yearly_balance_sheet_history,
        SOURCE_Q_CASH: get_quarterly_cash_flow_history,
        SOURCE_Y_CASH: get_yearly_cash_flow_history,
        SOURCE_Y_INDIRECT: get_yearly_indirect_cash_flow_history,
    }
    return mapping.get(source)
