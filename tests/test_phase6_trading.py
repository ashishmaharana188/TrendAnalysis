from datetime import date, timedelta

from analysis.phase6_trading import (
    DailyBar,
    TradePolicyConfig,
    TradeSignal,
    simulate_trades,
)


def main() -> None:
    start = date(2026, 1, 1)
    prices = [100.0, 102.0, 104.0, 103.0, 101.0, 99.0, 98.0]

    bars = [
        DailyBar(
            date=start + timedelta(days=i),
            open=p,
            high=p + 1.0,
            low=p - 1.0,
            close=p,
        )
        for i, p in enumerate(prices)
    ]

    signals = [
        TradeSignal(
            date=start,
            direction="UP",
            probability_pct=65.0,
            baseline_probability_pct=33.0,
            eligible=True,
            reason="DIRECTIONAL_AGREEMENT",
        )
    ]

    result = simulate_trades(
        signals,
        bars,
        TradePolicyConfig(
            stop_loss_pct=5.0,
            take_profit_pct=None,
            max_holding_days=5,
            exit_on_model_reversal=False,
        ),
    )

    assert result.total_trades == 1
    trade = result.trades[0]
    assert trade.entry_date == start + timedelta(days=1)
    assert trade.exit_date == start + timedelta(days=5)
    assert trade.exit_reason == "TIME_STOP"
    assert trade.holding_days == 5

    print("PHASE 6 TRADING SIMULATOR TEST: PASS")
    print(f"Trades: {result.total_trades}")
    print(f"Return: {result.total_directional_return_pct:.4f}%")


if __name__ == "__main__":
    main()

