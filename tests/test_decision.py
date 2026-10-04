from __future__ import annotations

from analysis.decision import decide_baseline_relative

BASELINE = {"UP": 33.0, "SIDEWAYS": 34.0, "DOWN": 33.0}


def test_direction_requires_baseline_lift_and_margin() -> None:
    result = decide_baseline_relative(
        {"UP": 60.0, "SIDEWAYS": 20.0, "DOWN": 20.0},
        BASELINE,
        100.0,
    )
    assert result.trend == "UP"
    assert result.selected_class == "UP"
    assert result.lift_pct == 27.0
    assert result.margin_pct == 40.0
    assert result.uncertainty_pct > 0.0


def test_direction_fails_when_leading_direction_is_not_above_baseline() -> None:
    result = decide_baseline_relative(
        {"UP": 20.0, "SIDEWAYS": 45.0, "DOWN": 35.0},
        {"UP": 25.0, "SIDEWAYS": 20.0, "DOWN": 55.0},
        100.0,
    )
    assert result.trend == "SIDEWAYS"
    assert result.selected_class == "SIDEWAYS"


def test_sideways_requires_margin_and_baseline_lift() -> None:
    result = decide_baseline_relative(
        {"UP": 33.0, "SIDEWAYS": 34.0, "DOWN": 33.0},
        BASELINE,
        100.0,
    )
    assert result.trend == "NO_CLEAR_TREND"
    assert "baseline" in result.reason.lower()


def test_small_margin_fails_closed() -> None:
    result = decide_baseline_relative(
        {"UP": 40.0, "SIDEWAYS": 35.0, "DOWN": 25.0},
        BASELINE,
        4.0,
    )
    assert result.trend == "NO_CLEAR_TREND"
    assert result.selected_class == "UP"
    assert result.margin_pct == 5.0
    assert result.uncertainty_pct > result.margin_pct


def test_tied_leading_class_fails_closed() -> None:
    result = decide_baseline_relative(
        {"UP": 40.0, "SIDEWAYS": 40.0, "DOWN": 20.0},
        BASELINE,
        100.0,
    )
    assert result.trend == "NO_CLEAR_TREND"
    assert "tied" in result.reason.lower()


def test_malformed_distribution_is_limited() -> None:
    result = decide_baseline_relative(
        {"UP": 60.0, "SIDEWAYS": 20.0, "DOWN": float("nan")},
        BASELINE,
        100.0,
    )
    assert result.trend == "NO_CLEAR_TREND"
    assert result.limited


def test_positive_directional_candidate_can_win_over_non_directional_top() -> None:
    # DOWN is the raw probability leader, but it is below its own baseline.
    # UP is above baseline and therefore becomes the only directional candidate.
    result = decide_baseline_relative(
        {"UP": 48.0, "SIDEWAYS": 20.0, "DOWN": 32.0},
        {"UP": 25.0, "SIDEWAYS": 15.0, "DOWN": 60.0},
        100.0,
    )
    assert result.selected_class == "UP"
    assert result.trend == "UP"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("PHASE 5.4 BASELINE-RELATIVE DECISION TEST: PASS")
