from __future__ import annotations

from analysis.real_olap_validation import (
    EXPECTED_PHASE4_STATE_FAMILIES,
    FINANCIAL_TIMING_LIMITED_FAMILIES,
    TIME_SAFE_PHASE4_STATE_FAMILIES,
    UPSTREAM_GLOBAL_CONTEXT_SERIES,
)


def main() -> None:
    assert set(TIME_SAFE_PHASE4_STATE_FAMILIES).isdisjoint(
        FINANCIAL_TIMING_LIMITED_FAMILIES
    )
    assert set(TIME_SAFE_PHASE4_STATE_FAMILIES) | set(FINANCIAL_TIMING_LIMITED_FAMILIES) == set(
        EXPECTED_PHASE4_STATE_FAMILIES
    )
    assert {
        "Brent_Crude",
        "US_Dollar_Index",
        "US_10Y_Yield",
        "US_VIX",
        "USD_INR",
        "Broad_Commodity",
    } <= set(UPSTREAM_GLOBAL_CONTEXT_SERIES)

    print("PHASE 4 STATE-SURFACE REGISTRY TEST: PASS")
    print("Expected families:", len(EXPECTED_PHASE4_STATE_FAMILIES))
    print("Time-safe families:", len(TIME_SAFE_PHASE4_STATE_FAMILIES))
    print("Timing-limited financial families:", len(FINANCIAL_TIMING_LIMITED_FAMILIES))
    print("Upstream global context series:", len(UPSTREAM_GLOBAL_CONTEXT_SERIES))


if __name__ == "__main__":
    main()
