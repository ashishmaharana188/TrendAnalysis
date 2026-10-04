from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .relationship import RelationshipResult


@dataclass(frozen=True)
class RelationshipRanking:
    """Evidence-oriented ranking for a discovered relationship."""

    variables: tuple[str, ...]
    condition: tuple[str, ...]
    method_results: tuple[RelationshipResult, ...]

    method_count: int
    method_agreement: bool
    direction: str

    best_score: float
    reliability: float
    sample_count: int
    stable_method_count: int

    rank_key: tuple

    @property
    def validated_usefulness(self) -> bool:
        """False by design: validation belongs to walk-forward testing."""
        return False

    def as_dict(self) -> dict:
        return {
            "variables": list(self.variables),
            "condition": list(self.condition),
            "method_count": self.method_count,
            "method_agreement": self.method_agreement,
            "direction": self.direction,
            "best_score": self.best_score,
            "reliability": self.reliability,
            "sample_count": self.sample_count,
            "stable_method_count": self.stable_method_count,
            "validated_usefulness": self.validated_usefulness,
            "methods": [item.as_dict() for item in self.method_results],
        }


def _condition_state_map(condition: tuple[str, ...]) -> dict[str, str]:
    """Normalize '=' and '~=' condition syntax for cross-method comparison."""
    result: dict[str, str] = {}
    for item in condition:
        if "~=" in item:
            feature, state = item.split("~=", 1)
        elif "=" in item:
            feature, state = item.split("=", 1)
        else:
            continue
        result[feature] = state
    return result


def _relationship_key(result: RelationshipResult) -> tuple:
    """Identify the same logical relationship across Method A and Method B."""
    state_map = _condition_state_map(result.condition)
    normalized_condition = tuple(
        f"{feature}={state_map[feature]}"
        for feature in sorted(state_map)
    )
    return tuple(result.variables), normalized_condition


def _direction(result: RelationshipResult) -> str:
    if result.mean_return_pct > 0:
        return "POSITIVE"
    if result.mean_return_pct < 0:
        return "NEGATIVE"
    return "NEUTRAL"


def rank_relationships(
    results: Iterable[RelationshipResult],
) -> list[RelationshipRanking]:
    """
    Rank relationships using evidence hierarchy, without fixed indicator weights.

    Ordering logic:
      1. Cross-method agreement.
      2. Directional stability across methods.
      3. Absolute empirical score.
      4. Reliability.
      5. Sample support.
      6. Stability count.

    This is an evidence ranking only. It does not claim out-of-sample
    predictive validity or create UP/SIDEWAYS/DOWN probabilities.
    """
    grouped: dict[tuple, list[RelationshipResult]] = {}

    for result in results:
        grouped.setdefault(_relationship_key(result), []).append(result)

    rankings: list[RelationshipRanking] = []

    for key, method_results in grouped.items():
        ordered = tuple(
            sorted(
                method_results,
                key=lambda item: item.method,
            )
        )

        directions = {_direction(item) for item in ordered}
        non_neutral = directions - {"NEUTRAL"}
        method_agreement = (
            len(ordered) >= 2
            and len(non_neutral) <= 1
        )

        if not non_neutral:
            direction = "NEUTRAL"
        elif len(non_neutral) == 1:
            direction = next(iter(non_neutral))
        else:
            direction = "MIXED"
            method_agreement = False

        best = max(ordered, key=lambda item: abs(item.score))
        reliability = max(item.reliability for item in ordered)
        sample_count = max(item.sample_count for item in ordered)
        stable_count = sum(1 for item in ordered if item.stable)

        # Lexicographic evidence hierarchy avoids a manually weighted
        # composite score. Method agreement is stronger evidence than merely
        # having a numerically large in-sample score.
        rank_key = (
            int(method_agreement),
            int(len(ordered) > 1),
            int(direction != "NEUTRAL"),
            abs(best.score),
            reliability,
            sample_count,
            stable_count,
        )

        rankings.append(
            RelationshipRanking(
                variables=key[0],
                condition=key[1],
                method_results=ordered,
                method_count=len(ordered),
                method_agreement=method_agreement,
                direction=direction,
                best_score=best.score,
                reliability=reliability,
                sample_count=sample_count,
                stable_method_count=stable_count,
                rank_key=rank_key,
            )
        )

    rankings.sort(key=lambda item: item.rank_key, reverse=True)
    return rankings
