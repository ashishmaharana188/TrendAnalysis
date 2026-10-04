from __future__ import annotations

from analysis.config import AnalysisConfig
from analysis.window import build_observation_window
from analysis.group_state import build_group_state

from data_access.metadata import (
    get_companies_by_industry,
    get_companies_by_sector,
)


def main() -> None:

    print("=" * 70)
    print("PHASE 3.6 - INDUSTRY / SECTOR STATE")
    print("=" * 70)

    config = AnalysisConfig(
        analysis_date="2026-10-03",
        analysis_timeframe="6M",
        holding_period_months=1.0,
        benchmark="Nifty_50",
        entry_mode="next_trading_day",
    )

    # --------------------------------------------------------
    # Target company
    # --------------------------------------------------------

    print("\n1. Building target observation window")

    window = build_observation_window(
        ticker="RELIANCE.NS",
        config=config,
    )

    print(
        f"Company : {window.ticker}"
    )

    print(
        f"Sector  : {window.sector}"
    )

    print(
        f"Industry: {window.industry}"
    )

    assert window.sector is not None
    assert window.industry is not None

    # --------------------------------------------------------
    # Industry
    # --------------------------------------------------------

    print("\n2. Building industry state")

    industry_companies = (
        get_companies_by_industry(
            window.industry
        )
    )

    industry_tickers = [
        company["Ticker"]
        for company in industry_companies
        if company.get("Ticker")
    ]

    print(
        f"Industry: {window.industry}"
    )

    print(
        f"Constituents: "
        f"{len(industry_tickers)}"
    )

    industry_state = build_group_state(
        group_type="industry",
        group_name=window.industry,
        constituents=industry_tickers,
        analysis_start_date=config.analysis_start_date,
        analysis_end_date=config.analysis_end_date,
        benchmark_history=window.benchmark["history"],
    )

    print(
        f"Valid constituents: "
        f"{len(industry_state.valid_constituents)}"
    )

    print(
        f"Breadth: "
        f"{industry_state.breadth}"
    )

    print(
        f"Relative return: "
        f"{industry_state.relative_return_pct}"
    )

    assert industry_state.group_type == "industry"

    assert (
        industry_state.group_name
        == window.industry
    )

    assert len(
        industry_state.valid_constituents
    ) > 0

    assert "group_index" in (
        industry_state.market_states.keys()
    )

    assert "daily_return" in (
        industry_state.market_states.keys()
    )

    assert "volatility_20d" in (
        industry_state.market_states.keys()
    )

    assert set(
        industry_state.breadth.keys()
    ) == {
        "UP",
        "SIDEWAYS",
        "DOWN",
    }

    breadth_total = sum(
        industry_state.breadth.values()
    )

    assert abs(
        breadth_total - 100.0
    ) < 0.0001

    # --------------------------------------------------------
    # Sector
    # --------------------------------------------------------

    print("\n3. Building sector state")

    sector_companies = (
        get_companies_by_sector(
            window.sector
        )
    )

    sector_tickers = [
        company["Ticker"]
        for company in sector_companies
        if company.get("Ticker")
    ]

    print(
        f"Sector: {window.sector}"
    )

    print(
        f"Constituents: "
        f"{len(sector_tickers)}"
    )

    sector_state = build_group_state(
        group_type="sector",
        group_name=window.sector,
        constituents=sector_tickers,
        analysis_start_date=config.analysis_start_date,
        analysis_end_date=config.analysis_end_date,
        benchmark_history=window.benchmark["history"],
    )

    print(
        f"Valid constituents: "
        f"{len(sector_state.valid_constituents)}"
    )

    print(
        f"Breadth: "
        f"{sector_state.breadth}"
    )

    print(
        f"Relative return: "
        f"{sector_state.relative_return_pct}"
    )

    assert sector_state.group_type == "sector"

    assert (
        sector_state.group_name
        == window.sector
    )

    assert len(
        sector_state.valid_constituents
    ) > 0

    assert "group_index" in (
        sector_state.market_states.keys()
    )

    assert "daily_return" in (
        sector_state.market_states.keys()
    )

    assert "volatility_20d" in (
        sector_state.market_states.keys()
    )

    breadth_total = sum(
        sector_state.breadth.values()
    )

    assert abs(
        breadth_total - 100.0
    ) < 0.0001

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n4. Phase 3.6 validation")

    print(
        "Industry state: PASS"
    )

    print(
        "Sector state:   PASS"
    )

    print(
        "\nPHASE 3.6 SUCCESS"
    )


if __name__ == "__main__":
    main()