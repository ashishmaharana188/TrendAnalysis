from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

OutcomeClass = Literal["UP", "SIDEWAYS", "DOWN"]


@dataclass(frozen=True)
class OutcomeThresholds:
    """Empirically learned return boundaries for one holding-period distribution."""

    lower_pct: float
    upper_pct: float
    sample_count: int
    lower_quantile: float = 1.0 / 3.0
    upper_quantile: float = 2.0 / 3.0
    method: str = "empirical_tertiles"
    limited: bool = False
    limitation: str | None = None

    def as_dict(self) -> dict:
        return {
            "lower_pct": self.lower_pct,
            "upper_pct": self.upper_pct,
            "sample_count": self.sample_count,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "method": self.method,
            "limited": self.limited,
            "limitation": self.limitation,
        }


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a quantile from an empty series.")
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * q
    left = int(position)
    right = min(left + 1, len(sorted_values) - 1)
    fraction = position - left
    return sorted_values[left] + (
        sorted_values[right] - sorted_values[left]
    ) * fraction


def learn_outcome_thresholds(
    returns_pct: Iterable[float],
    min_observations: int = 9,
) -> OutcomeThresholds:
    """
    Learn UP/SIDEWAYS/DOWN boundaries from the historical return distribution.

    The boundaries are quantiles of the supplied holding-period outcome
    distribution. No fixed percentage return threshold is imposed.
    """
    values = sorted(float(value) for value in returns_pct)
    if len(values) < min_observations:
        return OutcomeThresholds(
            lower_pct=0.0,
            upper_pct=0.0,
            sample_count=len(values),
            limited=True,
            limitation=(
                "Insufficient historical forward returns to learn "
                "three-class outcome thresholds."
            ),
        )

    lower = _quantile(values, 1.0 / 3.0)
    upper = _quantile(values, 2.0 / 3.0)

    if lower >= upper:
        return OutcomeThresholds(
            lower_pct=lower,
            upper_pct=upper,
            sample_count=len(values),
            limited=True,
            limitation=(
                "Historical forward-return distribution does not provide "
                "distinct three-class boundaries."
            ),
        )

    return OutcomeThresholds(
        lower_pct=lower,
        upper_pct=upper,
        sample_count=len(values),
    )


def classify_return(
    return_pct: float,
    thresholds: OutcomeThresholds,
) -> OutcomeClass:
    if thresholds.limited:
        raise ValueError(
            "Cannot classify returns with limited outcome thresholds."
        )

    if return_pct < thresholds.lower_pct:
        return "DOWN"
    if return_pct > thresholds.upper_pct:
        return "UP"
    return "SIDEWAYS"


def classify_returns(
    returns_pct: Iterable[float],
    thresholds: OutcomeThresholds,
) -> list[OutcomeClass]:
    return [classify_return(float(value), thresholds) for value in returns_pct]
