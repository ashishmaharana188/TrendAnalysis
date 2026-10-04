from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from itertools import combinations
from math import exp, log1p, sqrt
from statistics import median
from typing import Any, Iterable

from .hardening_4_10 import (
    categorical_similarity,
    split_stability,
    weighted_median,
)


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
    frozenset({"sector.financials", "sector.market"}),
    frozenset({"sector.market", "macro"}),
    frozenset({"sector.market", "global"}),
    # Institutional / derivatives / flow branches are structural inputs.
    # Their directional meaning is still learned empirically in Phase 4.
    frozenset({"institutional", "company.market"}),
    frozenset({"institutional", "industry.market"}),
    frozenset({"institutional", "sector.market"}),
    frozenset({"institutional", "macro"}),
    frozenset({"institutional", "global"}),
    frozenset({"institutional", "benchmark"}),
    frozenset({"institutional", "derivatives"}),
    frozenset({"institutional", "microstructure"}),
    frozenset({"institutional", "trade_events"}),
    frozenset({"derivatives", "company.market"}),
    frozenset({"derivatives", "industry.market"}),
    frozenset({"derivatives", "sector.market"}),
    frozenset({"derivatives", "macro"}),
    frozenset({"derivatives", "global"}),
    frozenset({"derivatives", "benchmark"}),
    frozenset({"derivatives", "microstructure"}),
    frozenset({"derivatives", "trade_events"}),
    frozenset({"microstructure", "company.market"}),
    frozenset({"microstructure", "industry.market"}),
    frozenset({"microstructure", "sector.market"}),
    frozenset({"microstructure", "macro"}),
    frozenset({"microstructure", "global"}),
    frozenset({"microstructure", "benchmark"}),
    frozenset({"microstructure", "trade_events"}),
    frozenset({"trade_events", "company.market"}),
    frozenset({"trade_events", "industry.market"}),
    frozenset({"trade_events", "sector.market"}),
    frozenset({"trade_events", "macro"}),
    frozenset({"trade_events", "global"}),
    frozenset({"trade_events", "benchmark"}),
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
    outcome_end_date: date | None = None


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

    # Raw support retained for Phase 5 so the prediction layer can convert
    # the exact Method A/B evidence into three-class probabilities without
    # duplicating the Phase 4 algorithms.
    supporting_observations: tuple[tuple[date, float], ...] = ()
    supporting_weights: tuple[float, ...] = ()

    # Phase 4.10 methodological diagnostics.
    state_coverage_pct: float = 100.0
    family_coverage_pct: float = 100.0
    stability_score: float = 0.0
    exact_condition_count: int | None = None
    parameter_provenance: tuple[tuple[str, date | None, date | None], ...] = ()

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
            "supporting_observation_count": len(self.supporting_observations),
            "supporting_weight_count": len(self.supporting_weights),
            "state_coverage_pct": self.state_coverage_pct,
            "family_coverage_pct": self.family_coverage_pct,
            "stability_score": self.stability_score,
            "exact_condition_count": self.exact_condition_count,
            "parameter_provenance": [
                {
                    "name": name,
                    "fit_start_date": fit_start,
                    "fit_end_date": fit_end,
                }
                for name, fit_start, fit_end in self.parameter_provenance
            ],
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
        min_state_coverage_pct: float = 75.0,
        min_family_coverage_pct: float = 75.0,
        stability_sem_multiplier: float = 2.0,
    ) -> None:
        if max_order < 1:
            raise ValueError("max_order must be >= 1")

        if min_observations < 1:
            raise ValueError(
                "min_observations must be >= 1"
            )

        if not 0.0 < min_state_coverage_pct <= 100.0:
            raise ValueError("min_state_coverage_pct must be in (0, 100]")
        if not 0.0 < min_family_coverage_pct <= 100.0:
            raise ValueError("min_family_coverage_pct must be in (0, 100]")
        if stability_sem_multiplier <= 0.0:
            raise ValueError("stability_sem_multiplier must be > 0")

        self.max_order = max_order
        self.min_observations = min_observations
        self.min_state_coverage_pct = float(min_state_coverage_pct)
        self.min_family_coverage_pct = float(min_family_coverage_pct)
        self.stability_sem_multiplier = float(stability_sem_multiplier)

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
        ) -> tuple[float, float, float]:
            diagnostics = categorical_similarity(
                current_states=current_states,
                observed_states=observation.states,
                feature_weights=feature_weights,
                family_lookup=_family,
            )
            return (
                diagnostics.score,
                diagnostics.state_coverage_pct,
                diagnostics.family_coverage_pct,
            )

        scored = [
            (observation, *similarity(observation))
            for observation in history
        ]

        # A historical row must be sufficiently comparable before it may
        # influence nearest-neighbour selection. Missingness now lowers
        # similarity and is prevented from becoming a similarity advantage.
        scored = [
            item
            for item in scored
            if item[1] > 0.0
            and item[2] >= self.min_state_coverage_pct
            and item[3] >= self.min_family_coverage_pct
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
        neighbour_coverages = [
            item[2]
            for item in scored[:neighbour_count]
        ]
        neighbour_family_coverages = [
            item[3]
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

        stability = split_stability(
            [item.stock_return_pct for item in ordered_neighbours],
            sem_multiplier=self.stability_sem_multiplier,
        )
        stable = stability.stable

        # Similarity is an evidence-quality multiplier. It is deliberately
        # normalized to [0, 1] and does not assign bullish/bearish meaning.
        mean_similarity = sum(neighbour_similarities) / len(
            neighbour_similarities
        )

        adjusted_score = result.score * mean_similarity
        fit_dates = [item.as_of_date for item in history]
        fit_start = min(fit_dates) if fit_dates else None
        fit_end = max(fit_dates) if fit_dates else None
        parameter_provenance = (
            ("state_relevance_frequency", fit_start, fit_end),
            ("method_a_adaptive_neighbor_count", fit_start, fit_end),
        )

        return [
            RelationshipResult(
                **{
                    **result.__dict__,
                    "score": adjusted_score,
                    "stable": stable,
                    "supporting_observations": tuple(
                        (item.as_of_date, float(item.stock_return_pct))
                        for item in neighbours
                    ),
                    "supporting_weights": tuple(
                        1.0 for _ in neighbours
                    ),
                    "state_coverage_pct": sum(neighbour_coverages) / len(neighbour_coverages),
                    "family_coverage_pct": sum(neighbour_family_coverages) / len(neighbour_family_coverages),
                    "stability_score": stability.stability_score,
                    "parameter_provenance": parameter_provenance,
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
        Method B: weighted historical distribution conditioned on the current
        state.

        The estimator is explicitly: ``P(Y | X ~= x)``. Every reported
        distribution statistic is calculated from the same weighted population.
        Exact condition matches are retained only as a secondary diagnostic and
        never define the primary sample or its summary statistics.
        """
        if not current_states or not history:
            return []

        baseline = history
        features = list(current_states.keys())
        candidates = _candidate_feature_sets(features, self.max_order)
        relevance = _adaptive_relevance_weights(history)

        results: list[RelationshipResult] = []

        for feature_set in candidates:
            feature_weights = {
                feature: relevance.get(
                    (feature, current_states[feature]),
                    1.0,
                )
                for feature in feature_set
            }

            scored: list[tuple[HistoricalRelationshipObservation, float, float, float]] = []
            for observation in history:
                diagnostics = categorical_similarity(
                    current_states={feature: current_states[feature] for feature in feature_set},
                    observed_states=observation.states,
                    feature_weights=feature_weights,
                    family_lookup=_family,
                )
                if (
                    diagnostics.state_coverage_pct >= self.min_state_coverage_pct
                    and diagnostics.family_coverage_pct >= self.min_family_coverage_pct
                ):
                    scored.append(
                        (
                            observation,
                            diagnostics.score,
                            diagnostics.state_coverage_pct,
                            diagnostics.family_coverage_pct,
                        )
                    )

            if len(scored) < self.min_observations:
                continue

            similarity_values = [item[1] for item in scored]
            positive_similarity = [value for value in similarity_values if value > 0.0]
            similarity_scale = (
                median(positive_similarity)
                if positive_similarity
                else 1.0
            )
            similarity_scale = max(similarity_scale, 1e-9)
            max_similarity = max(similarity_values)

            # Stable softmax-style weighting avoids exponential overflow while
            # preserving the intended relative relevance ordering. All valid
            # comparable rows remain in the distribution.
            weighted_rows: list[tuple[HistoricalRelationshipObservation, float]] = []
            for observation, similarity, _coverage, _family_coverage in scored:
                weight = exp(
                    (similarity - max_similarity) / similarity_scale
                )
                weighted_rows.append((observation, weight))

            total_weight = sum(weight for _, weight in weighted_rows)
            if total_weight <= 0.0:
                continue

            outcomes = [observation.stock_return_pct for observation, _ in weighted_rows]
            weights = [weight for _, weight in weighted_rows]
            weighted_mean = sum(
                outcome * weight
                for outcome, weight in zip(outcomes, weights)
            ) / total_weight
            weighted_median_value = weighted_median(outcomes, weights)
            weighted_positive_rate = (
                sum(
                    weight
                    for outcome, weight in zip(outcomes, weights)
                    if outcome > 0.0
                )
                / total_weight
            ) * 100.0

            weight_square_sum = sum(weight * weight for weight in weights)
            effective_sample_size = (
                (total_weight * total_weight) / weight_square_sum
                if weight_square_sum > 0.0
                else 0.0
            )
            weight_concentration = max(weights) / total_weight

            baseline_mean = (
                sum(observation.stock_return_pct for observation in baseline)
                / len(baseline)
            )
            baseline_dispersion = sqrt(
                sum(
                    (observation.stock_return_pct - baseline_mean) ** 2
                    for observation in baseline
                )
                / max(len(baseline), 1)
            )
            weighted_lift = weighted_mean - baseline_mean
            weighted_effect_strength = (
                abs(weighted_lift) / baseline_dispersion
                if baseline_dispersion > 0.0
                else 0.0
            )
            weighted_reliability = min(
                1.0,
                sqrt(effective_sample_size / max(len(baseline), 1)),
            )
            weighted_score = weighted_effect_strength * weighted_reliability

            exact_condition = _condition_key(current_states, feature_set)
            exact_condition_count = (
                sum(
                    1
                    for observation, _weight in weighted_rows
                    if exact_condition is not None
                    and _match_condition(observation, exact_condition)
                )
                if exact_condition is not None
                else 0
            )

            stability = split_stability(
                outcomes,
                sem_multiplier=self.stability_sem_multiplier,
            )
            fit_dates = [item.as_of_date for item in history]
            fit_start = min(fit_dates) if fit_dates else None
            fit_end = max(fit_dates) if fit_dates else None
            parameter_provenance = (
                ("state_relevance_frequency", fit_start, fit_end),
                ("method_b_similarity_scale", fit_start, fit_end),
            )
            mean_coverage = sum(item[2] for item in scored) / len(scored)
            mean_family_coverage = sum(item[3] for item in scored) / len(scored)

            results.append(
                RelationshipResult(
                    method="B",
                    variables=feature_set,
                    condition=exact_condition or (),
                    sample_count=len(weighted_rows),
                    mean_return_pct=weighted_mean,
                    median_return_pct=weighted_median_value,
                    baseline_mean_return_pct=baseline_mean,
                    lift_pct=weighted_lift,
                    positive_rate_pct=weighted_positive_rate,
                    effect_strength=weighted_effect_strength,
                    reliability=weighted_reliability,
                    score=weighted_score,
                    stable=stability.stable,
                    weighted_mean_return_pct=weighted_mean,
                    weighted_positive_rate_pct=weighted_positive_rate,
                    effective_sample_size=effective_sample_size,
                    weight_concentration=weight_concentration,
                    supporting_observations=tuple(
                        (item.as_of_date, float(item.stock_return_pct))
                        for item, _weight in weighted_rows
                    ),
                    supporting_weights=tuple(
                        float(weight) for _item, weight in weighted_rows
                    ),
                    state_coverage_pct=mean_coverage,
                    family_coverage_pct=mean_family_coverage,
                    stability_score=stability.stability_score,
                    exact_condition_count=exact_condition_count,
                    parameter_provenance=parameter_provenance,
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
