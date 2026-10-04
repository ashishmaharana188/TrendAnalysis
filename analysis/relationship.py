from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from itertools import combinations
from math import exp, log1p, sqrt
from statistics import median
from typing import Any, Iterable


# ============================================================
# STRUCTURAL RELATIONSHIP RULES
# ============================================================

# These are structural/domain relationships, not economic claims.
# Example: industry.market may interact with macro.Brent_Crude;
# whether it is actually useful is discovered from history.

ALLOWED_RELATIONSHIP_FAMILIES = {
    frozenset({"company.financials", "company.market"}),
    frozenset({"company.financials", "industry.market"}),
    frozenset({"industry.financials", "industry.market"}),
    frozenset({"industry.market", "sector.market"}),
    frozenset({"industry.market", "macro"}),
    frozenset({"industry.market", "global"}),
    frozenset({"sector.market", "benchmark"}),
    frozenset({"sector.market", "macro"}),
    frozenset({"sector.market", "global"}),
}


@dataclass(frozen=True)
class HistoricalRelationshipObservation:
    """One historical state snapshot paired with a realized outcome."""

    as_of_date: date
    target: str
    scope: str

    states: dict[str, str]

    stock_return_pct: float
    benchmark_return_pct: float | None
    relative_return_pct: float | None


@dataclass(frozen=True)
class RelationshipResult:
    """Empirical usefulness of one state or state combination."""

    method: str
    variables: tuple[str, ...]
    condition: tuple[str, ...]

    sample_count: int

    mean_return_pct: float
    median_return_pct: float

    baseline_mean_return_pct: float
    lift_pct: float

    positive_rate_pct: float

    effect_strength: float
    reliability: float

    score: float

    stable: bool

    weighted_mean_return_pct: float | None = None
    weighted_positive_rate_pct: float | None = None
    effective_sample_size: float | None = None
    weight_concentration: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "variables": list(self.variables),
            "condition": list(self.condition),
            "sample_count": self.sample_count,
            "mean_return_pct": self.mean_return_pct,
            "median_return_pct": self.median_return_pct,
            "baseline_mean_return_pct": self.baseline_mean_return_pct,
            "lift_pct": self.lift_pct,
            "positive_rate_pct": self.positive_rate_pct,
            "effect_strength": self.effect_strength,
            "reliability": self.reliability,
            "score": self.score,
            "stable": self.stable,
            "weighted_mean_return_pct": self.weighted_mean_return_pct,
            "weighted_positive_rate_pct": self.weighted_positive_rate_pct,
            "effective_sample_size": self.effective_sample_size,
            "weight_concentration": self.weight_concentration,
        }


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    return date.fromisoformat(str(value))


def _family(feature: str) -> str:
    """Map a state path to a structural family."""

    parts = feature.split(".")

    if len(parts) < 2:
        return parts[0]

    return f"{parts[0]}.{parts[1]}"


def _compatible_pair(
    left: str,
    right: str,
) -> bool:
    left_family = _family(left)
    right_family = _family(right)

    if left_family == right_family:
        return False

    families = frozenset({
        left_family,
        right_family,
    })

    return families in ALLOWED_RELATIONSHIP_FAMILIES


def _flatten_states(
    value: Any,
    prefix: str = "",
) -> dict[str, str]:
    """
    Flatten Phase 3-style nested state dictionaries.

    Only descriptive `state` leaves are emitted. Numeric values,
    ranges, breadth, and bookkeeping fields remain outside the
    Phase 4 categorical relationship surface for now.
    """

    result: dict[str, str] = {}

    if not isinstance(value, dict):
        return result

    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)

        if key == "state" and isinstance(child, str):
            result[prefix] = child
            continue

        if isinstance(child, dict):
            result.update(
                _flatten_states(
                    child,
                    path,
                )
            )

    return result


def flatten_phase3_states(
    states: dict[str, Any],
) -> dict[str, str]:
    """
    Convert nested Phase 3 outputs into categorical state atoms.

    Already-flat `{feature: state}` dictionaries are also accepted.
    """

    flat: dict[str, str] = {}

    for key, value in states.items():
        if isinstance(value, str):
            flat[key] = value
            continue

        if isinstance(value, dict):
            nested = _flatten_states(
                value,
                str(key),
            )
            flat.update(nested)

    return flat


def _match_condition(
    observation: HistoricalRelationshipObservation,
    condition: tuple[str, ...],
) -> bool:
    return all(
        "=" in item
        and observation.states.get(item.split("=", 1)[0])
        == item.split("=", 1)[1]
        for item in condition
    )


def _condition_for_feature(
    observation: HistoricalRelationshipObservation,
    feature: str,
) -> tuple[str, ...]:
    return (
        f"{feature}={observation.states[feature]}",
    )


def _condition_key(
    states: dict[str, str],
    features: tuple[str, ...],
) -> tuple[str, ...] | None:
    values: list[str] = []

    for feature in features:
        state = states.get(feature)
        if state is None:
            return None

        values.append(
            f"{feature}={state}"
        )

    return tuple(values)


def _adaptive_relevance_weights(
    observations: list[HistoricalRelationshipObservation],
) -> dict[tuple[str, str], float]:
    """
    Give more weight to informative/rarer state occurrences.

    No fixed tolerance or manually assigned feature weight is used.
    """

    total = len(observations)
    frequencies: dict[tuple[str, str], int] = {}

    for observation in observations:
        for feature, state in observation.states.items():
            key = (feature, state)
            frequencies[key] = (
                frequencies.get(key, 0) + 1
            )

    weights: dict[tuple[str, str], float] = {}

    for key, frequency in frequencies.items():
        weights[key] = log1p(
            total / max(frequency, 1)
        )

    return weights


def _condition_matches_current(
    condition: tuple[str, ...],
    current_states: dict[str, str],
) -> bool:
    for item in condition:
        feature, state = item.split("=", 1)

        if current_states.get(feature) != state:
            return False

    return True


def _summary_result(
    method: str,
    variables: tuple[str, ...],
    condition: tuple[str, ...],
    matching: list[HistoricalRelationshipObservation],
    baseline: list[HistoricalRelationshipObservation],
) -> RelationshipResult | None:
    if not matching or not baseline:
        return None

    outcomes = [
        observation.stock_return_pct
        for observation in matching
    ]

    baseline_outcomes = [
        observation.stock_return_pct
        for observation in baseline
    ]

    mean_return = sum(outcomes) / len(outcomes)
    baseline_mean = (
        sum(baseline_outcomes)
        / len(baseline_outcomes)
    )

    lift = mean_return - baseline_mean

    median_return = median(outcomes)

    positive_rate = (
        sum(value > 0 for value in outcomes)
        / len(outcomes)
    ) * 100.0

    baseline_dispersion = sqrt(
        sum(
            (value - baseline_mean) ** 2
            for value in baseline_outcomes
        )
        / max(len(baseline_outcomes), 1)
    )

    effect_strength = (
        abs(lift) / baseline_dispersion
        if baseline_dispersion > 0
        else 0.0
    )

    # Data-adaptive reliability. As sample size approaches the
    # historical universe, reliability approaches 1 without using
    # a user-chosen fixed sample multiplier.
    reliability = min(
        1.0,
        sqrt(len(matching) / max(len(baseline), 1)),
    )

    score = effect_strength * reliability

    return RelationshipResult(
        method=method,
        variables=variables,
        condition=condition,
        sample_count=len(matching),
        mean_return_pct=mean_return,
        median_return_pct=median_return,
        baseline_mean_return_pct=baseline_mean,
        lift_pct=lift,
        positive_rate_pct=positive_rate,
        effect_strength=effect_strength,
        reliability=reliability,
        score=score,
        stable=False,
    )


def _structurally_connected(features: tuple[str, ...]) -> bool:
    """Return True when the selected families form a connected graph."""

    families = {_family(feature) for feature in features}

    if len(families) != len(features):
        # Do not duplicate the same structural family in one combination.
        return False

    if len(families) <= 1:
        return False

    remaining = set(families)
    visited = {next(iter(remaining))}

    changed = True
    while changed:
        changed = False
        for left in tuple(visited):
            for right in tuple(remaining - visited):
                if frozenset({left, right}) in ALLOWED_RELATIONSHIP_FAMILIES:
                    visited.add(right)
                    changed = True

    return visited == families


def _candidate_feature_sets(
    features: Iterable[str],
    max_order: int,
) -> list[tuple[str, ...]]:
    features = sorted(set(features))

    candidates: list[tuple[str, ...]] = []

    # Individual variables are always allowed.
    for feature in features:
        candidates.append((feature,))

    # Pairs remain explicitly constrained by the structural graph.
    if max_order >= 2:
        for left, right in combinations(features, 2):
            if _compatible_pair(left, right):
                candidates.append((left, right))

    # Higher-order combinations are generated only when the family graph
    # is connected and every member belongs to a distinct family. This keeps
    # the search structurally bounded rather than allowing arbitrary feature
    # cartesian products.
    for order in range(3, max_order + 1):
        for candidate in combinations(features, order):
            if _structurally_connected(candidate):
                candidates.append(candidate)

    return candidates


class RelationshipDiscoveryEngine:
    """
    Phase 4 relationship discovery foundation.

    Method A:
        adaptive similarity against historically comparable states.

    Method B:
        empirical conditional effects with adaptive relevance
        weighting for the current conditions.

    This class does not produce final calibrated probabilities.
    """

    def __init__(
        self,
        max_order: int = 2,
        min_observations: int = 5,
    ) -> None:
        if max_order < 1:
            raise ValueError("max_order must be >= 1")

        if min_observations < 1:
            raise ValueError(
                "min_observations must be >= 1"
            )

        self.max_order = max_order
        self.min_observations = min_observations

    def prepare_history(
        self,
        observations: Iterable[
            HistoricalRelationshipObservation
        ],
        cutoff_date: str | date | datetime,
    ) -> list[HistoricalRelationshipObservation]:
        """Return only observations strictly before the cutoff."""

        cutoff = _as_date(cutoff_date)

        return sorted(
            [
                observation
                for observation in observations
                if (
                    _as_date(observation.as_of_date)
                    < cutoff
                )
                and (
                    observation.stock_return_pct
                    is not None
                )
            ],
            key=lambda observation: observation.as_of_date,
        )

    # --------------------------------------------------------
    # Method A
    # --------------------------------------------------------

    def method_a_similar_states(
        self,
        current_states: dict[str, str],
        history: list[HistoricalRelationshipObservation],
    ) -> list[RelationshipResult]:
        """
        Compare the complete current descriptive state with historical
        observations using adaptive categorical similarity.

        Similarity is built from:

        * current-state agreement at the variable level
        * information content of the observed state
        * structural family coverage
        * an adaptive nearest-neighbour count based on available history

        No fixed numeric tolerance is used because Phase 3 exposes
        categorical states rather than raw numeric indicator values.
        """
        if not current_states or not history:
            return []

        baseline = history

        # ----------------------------------------------------
        # 1. Build adaptive feature weights. Rare informative
        #    states receive more relevance, common states less.
        # ----------------------------------------------------
        state_weights = _adaptive_relevance_weights(history)

        current_features = tuple(sorted(current_states))

        feature_weights: dict[str, float] = {}
        for feature in current_features:
            state = current_states[feature]
            feature_weights[feature] = state_weights.get(
                (feature, state),
                1.0,
            )

        # ----------------------------------------------------
        # 2. Family coverage is used only to keep one family from
        #    dominating because it happens to contain many fields.
        #    This is structural normalization, not an economic weight.
        # ----------------------------------------------------
        families: dict[str, list[str]] = {}
        for feature in current_features:
            families.setdefault(
                _family(feature),
                [],
            ).append(feature)

        def similarity(
            observation: HistoricalRelationshipObservation,
        ) -> float:
            family_scores: list[float] = []

            for family_features in families.values():
                numerator = 0.0
                denominator = 0.0

                for feature in family_features:
                    weight = feature_weights[feature]
                    observed_state = observation.states.get(feature)

                    # Missing historical state means the variable is not
                    # comparable for this observation. It is not a mismatch.
                    if observed_state is None:
                        continue

                    denominator += weight

                    if observed_state == current_states[feature]:
                        numerator += weight

                if denominator > 0.0:
                    family_scores.append(
                        numerator / denominator
                    )

            if not family_scores:
                return 0.0

            return sum(family_scores) / len(family_scores)

        scored = [
            (observation, similarity(observation))
            for observation in history
        ]

        # Only positive-similarity observations are genuinely comparable.
        scored = [
            item
            for item in scored
            if item[1] > 0.0
        ]

        if len(scored) < self.min_observations:
            return []

        # ----------------------------------------------------
        # 3. Adaptive neighbourhood size. sqrt(N) grows slowly with
        #    history and therefore avoids a fixed training-window rule.
        # ----------------------------------------------------
        neighbour_count = max(
            self.min_observations,
            int(
                max(
                    1,
                    round(sqrt(len(scored))),
                )
            ),
        )
        neighbour_count = min(
            neighbour_count,
            len(scored),
        )

        scored.sort(
            key=lambda item: (
                item[1],
                item[0].as_of_date,
            ),
            reverse=True,
        )

        neighbours = [
            item[0]
            for item in scored[:neighbour_count]
        ]

        neighbour_similarities = [
            item[1]
            for item in scored[:neighbour_count]
        ]

        result = _summary_result(
            method="A",
            variables=current_features,
            condition=tuple(
                f"{feature}~={current_states[feature]}"
                for feature in current_features
            ),
            matching=neighbours,
            baseline=baseline,
        )

        if result is None:
            return []

        # ----------------------------------------------------
        # 4. Chronological stability check. A relation is stable
        #    when the first/second halves of the selected neighbours
        #    point in the same return direction.
        # ----------------------------------------------------
        ordered_neighbours = sorted(
            neighbours,
            key=lambda observation: observation.as_of_date,
        )

        stable = False
        if len(ordered_neighbours) >= 4:
            midpoint = len(ordered_neighbours) // 2
            first_half = ordered_neighbours[:midpoint]
            second_half = ordered_neighbours[midpoint:]

            first_mean = sum(
                item.stock_return_pct
                for item in first_half
            ) / len(first_half)

            second_mean = sum(
                item.stock_return_pct
                for item in second_half
            ) / len(second_half)

            stable = (
                (first_mean == 0.0 and second_mean == 0.0)
                or (first_mean > 0.0 and second_mean > 0.0)
                or (first_mean < 0.0 and second_mean < 0.0)
            )

        # Similarity is an evidence-quality multiplier. It is deliberately
        # normalized to [0, 1] and does not assign bullish/bearish meaning.
        mean_similarity = sum(neighbour_similarities) / len(
            neighbour_similarities
        )

        adjusted_score = result.score * mean_similarity

        return [
            RelationshipResult(
                **{
                    **result.__dict__,
                    "score": adjusted_score,
                    "stable": stable,
                }
            )
        ]

    # --------------------------------------------------------
    # Method B
    # --------------------------------------------------------

    def method_b_conditioned_distribution(
        self,
        current_states: dict[str, str],
        history: list[HistoricalRelationshipObservation],
    ) -> list[RelationshipResult]:
        """
        Method B: full historical outcome distribution with adaptive
        current-condition weighting.

        Unlike Method A, Method B does not select a nearest-neighbour
        subset. Every valid historical observation remains in the
        distribution. Observations receive higher/lower relevance based
        on how strongly their candidate feature states agree with the
        current condition.

        The weighting is learned from the observed state frequencies and
        structural family coverage. It does not assign a permanent
        bullish/bearish weight to any variable.
        """
        if not current_states or not history:
            return []

        baseline = history
        features = list(current_states.keys())
        candidates = _candidate_feature_sets(
            features,
            self.max_order,
        )
        relevance = _adaptive_relevance_weights(history)

        results: list[RelationshipResult] = []

        for feature_set in candidates:
            # Structural-family normalization prevents a family with more
            # fields from mechanically receiving more influence.
            family_groups: dict[str, list[str]] = {}
            for feature in feature_set:
                family_groups.setdefault(
                    _family(feature),
                    [],
                ).append(feature)

            def observation_similarity(
                observation: HistoricalRelationshipObservation,
            ) -> float:
                family_scores: list[float] = []

                for family_features in family_groups.values():
                    numerator = 0.0
                    denominator = 0.0

                    for feature in family_features:
                        observed_state = observation.states.get(feature)
                        current_state = current_states.get(feature)
                        if observed_state is None or current_state is None:
                            continue

                        info_weight = relevance.get(
                            (feature, current_state),
                            1.0,
                        )
                        denominator += info_weight

                        if observed_state == current_state:
                            numerator += info_weight

                    if denominator > 0.0:
                        family_scores.append(
                            numerator / denominator
                        )

                if not family_scores:
                    return 0.0

                return sum(family_scores) / len(family_scores)

            scored = [
                (observation, observation_similarity(observation))
                for observation in history
            ]

            comparable_scores = [score for _, score in scored if score > 0.0]
            if len(comparable_scores) < self.min_observations:
                continue

            # Full-distribution weighting. The exponential transform keeps
            # all rows in the distribution while making stronger matches
            # more relevant. The scale is data-derived from the candidate
            # similarity distribution rather than a hard-coded tolerance.
            similarity_values = [score for _, score in scored]
            positive_similarity = [
                score for score in similarity_values
                if score > 0.0
            ]
            similarity_scale = (
                sum(positive_similarity) / len(positive_similarity)
                if positive_similarity
                else 1.0
            )
            similarity_scale = max(similarity_scale, 1e-9)

            weighted_rows: list[tuple[HistoricalRelationshipObservation, float]] = []
            for observation, similarity in scored:
                # baseline weight keeps every valid outcome in the sample;
                # relevance then tilts the contribution toward the current
                # state without dropping dissimilar observations entirely.
                weight = exp(
                    similarity / similarity_scale
                )
                weighted_rows.append((observation, weight))

            total_weight = sum(weight for _, weight in weighted_rows)
            if total_weight <= 0.0:
                continue

            weighted_mean = sum(
                observation.stock_return_pct * weight
                for observation, weight in weighted_rows
            ) / total_weight

            weighted_positive_rate = (
                sum(
                    weight
                    for observation, weight in weighted_rows
                    if observation.stock_return_pct > 0.0
                )
                / total_weight
            ) * 100.0

            weight_square_sum = sum(
                weight * weight
                for _, weight in weighted_rows
            )
            effective_sample_size = (
                (total_weight * total_weight) / weight_square_sum
                if weight_square_sum > 0.0
                else 0.0
            )

            weight_concentration = (
                max(weight for _, weight in weighted_rows)
                / total_weight
            )

            condition = _condition_key(
                current_states,
                feature_set,
            )
            if condition is None:
                continue

            # For Method B the matched subset is still used to report the
            # observed conditional effect, while the weighted distribution
            # is the primary method-specific signal.
            matching = [
                observation
                for observation in history
                if _match_condition(observation, condition)
            ]
            if len(matching) < self.min_observations:
                continue

            result = _summary_result(
                method="B",
                variables=feature_set,
                condition=condition,
                matching=matching,
                baseline=baseline,
            )
            if result is None:
                continue

            weighted_lift = weighted_mean - result.baseline_mean_return_pct
            baseline_dispersion = sqrt(
                sum(
                    (observation.stock_return_pct - result.baseline_mean_return_pct) ** 2
                    for observation in baseline
                )
                / max(len(baseline), 1)
            )
            weighted_effect_strength = (
                abs(weighted_lift) / baseline_dispersion
                if baseline_dispersion > 0.0
                else 0.0
            )

            weighted_reliability = min(
                1.0,
                sqrt(
                    effective_sample_size / max(len(baseline), 1)
                ),
            )

            weighted_score = (
                weighted_effect_strength
                * weighted_reliability
            )

            results.append(
                RelationshipResult(
                    **{
                        **result.__dict__,
                        "mean_return_pct": weighted_mean,
                        "lift_pct": weighted_lift,
                        "positive_rate_pct": weighted_positive_rate,
                        "effect_strength": weighted_effect_strength,
                        "reliability": weighted_reliability,
                        "score": weighted_score,
                        "weighted_mean_return_pct": weighted_mean,
                        "weighted_positive_rate_pct": weighted_positive_rate,
                        "effective_sample_size": effective_sample_size,
                        "weight_concentration": weight_concentration,
                    }
                )
            )

        return sorted(
            results,
            key=lambda result: abs(result.score),
            reverse=True,
        )

    # --------------------------------------------------------
    # Combined discovery
    # --------------------------------------------------------

    def discover(
        self,
        current_states: dict[str, str],
        observations: Iterable[
            HistoricalRelationshipObservation
        ],
        cutoff_date: str | date | datetime,
    ) -> dict[str, list[RelationshipResult]]:
        """
        Run both relationship methods using only pre-cutoff data.
        """

        history = self.prepare_history(
            observations,
            cutoff_date,
        )

        method_a = self.method_a_similar_states(
            current_states,
            history,
        )

        method_b = self.method_b_conditioned_distribution(
            current_states,
            history,
        )

        return {
            "method_a": method_a,
            "method_b": method_b,
        }
