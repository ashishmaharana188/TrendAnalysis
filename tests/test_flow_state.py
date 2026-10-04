from datetime import date

from analysis.flow_state import (
    build_derivatives_state_atoms,
    build_institutional_state_atoms,
    build_microstructure_state_atoms,
    build_trade_event_state_atoms,
)


def _fake_builder(*, variable, selected_table, historical_table, value_column):
    assert value_column == "value"
    values = historical_table.column("value").to_pylist()
    if not values:
        return {"state": "Unknown"}
    return {"state": "High" if values[-1] > values[0] else "Stable"}


def test_institutional_exposes_fii_dii_and_net_position_states():
    rows = [
        {"report_date": date(2026, 1, 1), "client_type": "FII/FPI", "cash_net_value": -10, "future_index_long": 100, "future_index_short": 80, "total_long_contracts": 150, "total_short_contracts": 120},
        {"report_date": date(2026, 2, 1), "client_type": "FII/FPI", "cash_net_value": 20, "future_index_long": 120, "future_index_short": 70, "total_long_contracts": 180, "total_short_contracts": 100},
        {"report_date": date(2026, 1, 1), "client_type": "DII", "cash_net_value": 15, "future_index_long": 60, "future_index_short": 50, "total_long_contracts": 80, "total_short_contracts": 70},
        {"report_date": date(2026, 2, 1), "client_type": "DII", "cash_net_value": 30, "future_index_long": 70, "future_index_short": 40, "total_long_contracts": 100, "total_short_contracts": 60},
    ]
    states = build_institutional_state_atoms(rows, date(2026, 1, 1), date(2026, 2, 1), _fake_builder)
    assert any(key.startswith("institutional.FII_FPI.") for key in states)
    assert any(key.startswith("institutional.DII.") for key in states)
    assert "institutional.FII_FPI.future_index_net" in states
    assert "institutional.DII.total_net_contracts" in states


def test_options_pcr_is_aggregated_from_oi_before_state_creation():
    rows = [
        {"report_date": date(2026, 2, 1), "expiry_date": date(2026, 2, 5), "total_put_oi": 100, "total_call_oi": 50, "total_put_volume": 10, "total_call_volume": 5},
        {"report_date": date(2026, 2, 1), "expiry_date": date(2026, 2, 26), "total_put_oi": 50, "total_call_oi": 100, "total_put_volume": 5, "total_call_volume": 20},
        {"report_date": date(2026, 2, 2), "expiry_date": date(2026, 2, 5), "total_put_oi": 250, "total_call_oi": 50, "total_put_volume": 20, "total_call_volume": 4},
    ]
    states = build_derivatives_state_atoms(rows, [], date(2026, 2, 1), date(2026, 2, 2), _fake_builder)
    assert "derivatives.options.oi_pcr" in states
    assert "derivatives.options.volume_pcr" in states


def test_futures_basis_uses_highest_oi_row_per_date():
    rows = [
        {"report_date": date(2026, 2, 1), "open_interest": 10, "spot_price": 100, "futures_price": 101, "absolute_basis": 1, "basis_percentage": 1},
        {"report_date": date(2026, 2, 1), "open_interest": 50, "spot_price": 100, "futures_price": 103, "absolute_basis": 3, "basis_percentage": 3},
        {"report_date": date(2026, 2, 2), "open_interest": 60, "spot_price": 100, "futures_price": 102, "absolute_basis": 2, "basis_percentage": 2},
    ]
    states = build_derivatives_state_atoms([], rows, date(2026, 2, 1), date(2026, 2, 2), _fake_builder)
    assert "derivatives.futures.basis_percentage" in states


def test_trade_events_become_daily_flow_states():
    rows = [
        {"report_date": date(2026, 2, 1), "transaction_type": "BUY", "quantity": 100, "trade_price": 100},
        {"report_date": date(2026, 2, 1), "transaction_type": "SELL", "quantity": 40, "trade_price": 101},
        {"report_date": date(2026, 2, 2), "transaction_type": "BUY", "quantity": 180, "trade_price": 105},
    ]
    states = build_trade_event_state_atoms(rows, date(2026, 2, 1), date(2026, 2, 2), _fake_builder)
    assert "trade_events.net_event_volume" in states
    assert "trade_events.event_count" in states


def test_microstructure_exposes_non_duplicate_matrix_fields():
    rows = [
        {"report_date": date(2026, 2, 1), "delivery_percentage": 40, "daily_hl_spread": 2, "daily_vwap_dev": -1, "short_percentage": 10, "net_block_volume": -100, "avg_block_premium": -2, "oi_pcr": 1.1, "delta_oi_pcr": 0.1, "futures_basis": 0.5, "has_block_deal": False, "is_fo_eligible": True},
        {"report_date": date(2026, 2, 2), "delivery_percentage": 60, "daily_hl_spread": 3, "daily_vwap_dev": 1, "short_percentage": 15, "net_block_volume": 150, "avg_block_premium": 1, "oi_pcr": 1.4, "delta_oi_pcr": 0.3, "futures_basis": 1.0, "has_block_deal": True, "is_fo_eligible": True},
    ]
    states = build_microstructure_state_atoms(rows, date(2026, 2, 1), date(2026, 2, 2), _fake_builder)
    assert "microstructure.delivery_percentage" in states
    assert "microstructure.net_block_volume" in states
    assert "microstructure.has_block_deal" in states


def test_future_flow_rows_are_excluded_at_cutoff():
    rows = [
        {"report_date": date(2026, 2, 1), "client_type": "FII/FPI", "cash_net_value": -10},
        {"report_date": date(2026, 2, 2), "client_type": "FII/FPI", "cash_net_value": 1000},
    ]
    states = build_institutional_state_atoms(rows, date(2026, 2, 1), date(2026, 2, 1), _fake_builder)
    assert "institutional.FII_FPI.cash_net_value" in states


def test_unknown_trade_event_type_is_not_classified_as_sell():
    from analysis.flow_state import _aggregate_trade_events

    rows = [
        {
            "report_date": date(2026, 2, 1),
            "transaction_type": "UNKNOWN",
            "quantity": 100,
            "trade_price": 100,
        },
    ]
    aggregated = _aggregate_trade_events(rows)
    assert aggregated[0]["buy_volume"] == 0.0
    assert aggregated[0]["sell_volume"] == 0.0
    assert aggregated[0]["gross_event_volume"] == 0.0
