from __future__ import annotations

from datetime import date, timedelta

import pyarrow as pa

from analysis.performance import DailyHistoryCache


def main() -> None:
    calls: list[str] = []
    base = date(2026, 1, 1)
    table = pa.table(
        {
            "ticker": ["ABC"] * 5,
            "report_date": [base + timedelta(days=i) for i in range(5)],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
        }
    )

    def loader(ticker: str) -> pa.Table:
        calls.append(ticker)
        return table

    cache = DailyHistoryCache(loader)
    first = cache.get_daily_history("ABC", end_date=base + timedelta(days=2))
    second = cache.get_daily_history("ABC", end_date=base + timedelta(days=4))

    assert first.num_rows == 3
    assert second.num_rows == 5
    assert calls == ["ABC"]
    assert cache.stats.requests == 2
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert cache.unique_tickers == 1

    print("PHASE 4 PERFORMANCE CACHE TEST: PASS")
    print("Loader calls:", len(calls))
    print("Cache hits:", cache.stats.hits)
    print("Cache misses:", cache.stats.misses)


if __name__ == "__main__":
    main()
