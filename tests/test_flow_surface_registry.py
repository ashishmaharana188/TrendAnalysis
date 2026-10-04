from analysis.relationship import ALLOWED_RELATIONSHIP_FAMILIES


def test_flow_families_have_structural_market_links():
    expected_pairs = {
        frozenset({"institutional", "company.market"}),
        frozenset({"institutional", "industry.market"}),
        frozenset({"institutional", "sector.market"}),
        frozenset({"institutional", "derivatives"}),
        frozenset({"institutional", "microstructure"}),
        frozenset({"institutional", "trade_events"}),
        frozenset({"derivatives", "company.market"}),
        frozenset({"derivatives", "industry.market"}),
        frozenset({"derivatives", "sector.market"}),
        frozenset({"derivatives", "microstructure"}),
        frozenset({"derivatives", "trade_events"}),
        frozenset({"microstructure", "company.market"}),
        frozenset({"microstructure", "industry.market"}),
        frozenset({"microstructure", "sector.market"}),
        frozenset({"microstructure", "trade_events"}),
        frozenset({"trade_events", "company.market"}),
        frozenset({"trade_events", "industry.market"}),
        frozenset({"trade_events", "sector.market"}),
    }
    assert expected_pairs <= ALLOWED_RELATIONSHIP_FAMILIES
