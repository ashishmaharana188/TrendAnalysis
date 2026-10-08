from __future__ import annotations

"""Fast, fail-fast validation for the Phase 5.8 optimization overlay.

Run from the TrendAnalysis repository root before any expensive OLAP validation.
It validates interfaces, numerical equivalence on a synthetic panel, permutation
-equivalence, and gives a small timing comparison. It never touches the OLAP DB.
"""

import inspect
import os
import sys
from datetime import date, timedelta
from time import perf_counter


def _fail(message: str) -> None:
    print(f"PREFLIGHT FAIL: {message}", file=sys.stderr)
    raise SystemExit(2)


def _assert_close(a, b, name: str, tol: float = 1e-12) -> None:
    try:
        diff = abs(float(a) - float(b))
    except (TypeError, ValueError):
        if a != b:
            _fail(f"{name}: {a!r} != {b!r}")
        return
    if diff > tol:
        _fail(f"{name}: difference {diff} exceeds tolerance {tol}")


def _make_history(n: int = 32):
    from analysis.relationship import HistoricalRelationshipObservation

    base = date(2020, 1, 1)
    features = {
        "company.market.price": ("Rising / High", "Falling / Low"),
        "industry.market.price": ("Rising / Mid", "Stable / Mid"),
        "sector.market.price": ("Rising / High", "Falling / Mid"),
        "macro.rate": ("High", "Low"),
        "global.US_VIX": ("Low", "High"),
        "benchmark.NIFTY": ("Rising / High", "Falling / Low"),
    }
    rows = []
    for i in range(n):
        states = {}
        for j, (feature, values) in enumerate(features.items()):
            states[feature] = values[(i + j) % 2]
        rows.append(
            HistoricalRelationshipObservation(
                as_of_date=base + timedelta(days=i),
                target="T",
                scope="company",
                states=states,
                stock_return_pct=float(((i * 7) % 19) - 9) / 3.0,
                benchmark_return_pct=None,
                relative_return_pct=None,
                outcome_end_date=base + timedelta(days=i + 1),
            )
        )
    return rows


def main() -> None:
    root = os.getcwd()
    required = [
        os.path.join(root, "analysis", "relationship_fastpath.py"),
        os.path.join(root, "analysis", "fast_permutation.py"),
        os.path.join(root, "analysis", "walk_forward_relationship.py"),
        os.path.join(root, "analysis", "real_prediction_validation.py"),
        os.path.join(root, "analysis", "real_olap_validation.py"),
        os.path.join(root, "data_access", "metadata.py"),
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        _fail("required files missing: " + ", ".join(missing))

    from analysis.relationship import RelationshipDiscoveryEngine
    from analysis.relationship_fastpath import (
        method_a_prepared,
        method_b_prepared,
        prepare_relationship_context,
    )
    from analysis.relationship_graph import candidate_feature_sets
    from analysis.hardening_4_10 import search_adjusted_permutation_p_values
    from analysis.fast_permutation import search_adjusted_permutation_p_values_fast

    sig_a = inspect.signature(RelationshipDiscoveryEngine.method_a_similar_states)
    sig_b = inspect.signature(RelationshipDiscoveryEngine.method_b_conditioned_distribution)
    if "candidate_sets" not in sig_a.parameters or "candidate_sets" not in sig_b.parameters:
        _fail("current RelationshipDiscoveryEngine does not expose candidate_sets")

    engine = RelationshipDiscoveryEngine(
        max_order=3,
        min_observations=5,
        candidate_batch_size=64,
        retain_supporting_data=True,
    )
    if not hasattr(engine, "candidate_batch_size") or not hasattr(engine, "retain_supporting_data"):
        _fail("current engine is missing candidate_batch_size/retain_supporting_data")

    walk_text = open(required[2], encoding="utf-8").read()
    if "method_a_prepared(" not in walk_text or "method_b_prepared(" not in walk_text:
        _fail("walk-forward module is not wired to the prepared execution path")
    if "compact_support=True" not in walk_text:
        _fail("inner selection is not using compact support mode")
    if "search_adjusted_permutation_p_values_matrix(" not in walk_text:
        _fail("walk-forward module is not wired to the compact max-statistic gate")
    if "materialize_selected_support(" not in walk_text:
        _fail("selected support is not materialized after the search gate")

    prediction_text = open(required[3], encoding="utf-8").read()
    gate_markers = (
        "observed_trend: PredictionTrend",
        "trade_eligible: bool",
        "trade_reason: str",
    )
    missing_gate = [marker for marker in gate_markers if marker not in prediction_text]
    if missing_gate:
        _fail(
            "trade-gate interface is still missing. Run the Phase 5.8 trade-gate/membership repair first: "
            + ", ".join(missing_gate)
        )

    olap_text = open(required[4], encoding="utf-8").read()
    metadata_text = open(required[5], encoding="utf-8").read()
    if "as_of_date: date | None = None" not in olap_text:
        _fail("historical point-in-time membership is not installed in real_olap_validation.py")
    if "as_of_date: str | date | datetime | None = None" not in metadata_text:
        _fail("historical point-in-time membership is not installed in data_access/metadata.py")

    history = _make_history()
    current = dict(history[-1].states)
    candidates = candidate_feature_sets(current.keys(), 3)
    sample_candidates = candidates[: min(80, len(candidates))]
    if not sample_candidates:
        _fail("synthetic candidate universe is empty")

    # Numerical equivalence against the current public/vectorized engine.
    reference_a = engine.method_a_similar_states(current, history, candidate_sets=sample_candidates)
    reference_b = engine.method_b_conditioned_distribution(current, history, candidate_sets=sample_candidates)
    context = prepare_relationship_context(current, history)
    fast_a = method_a_prepared(engine, current, history, sample_candidates, context)
    fast_b = method_b_prepared(engine, current, history, sample_candidates, context)

    def compare(left, right, label):
        lmap = {tuple(x.variables): x for x in left}
        rmap = {tuple(x.variables): x for x in right}
        if set(lmap) != set(rmap):
            _fail(f"{label}: candidate result sets differ")
        numeric_fields = (
            "mean_return_pct", "median_return_pct", "baseline_mean_return_pct",
            "lift_pct", "positive_rate_pct", "effect_strength", "reliability",
            "score", "state_coverage_pct", "family_coverage_pct", "stability_score",
        )
        for key in lmap:
            a, b = lmap[key], rmap[key]
            for field in numeric_fields:
                _assert_close(getattr(a, field), getattr(b, field), f"{label} {key} {field}")
            if a.sample_count != b.sample_count or a.stable != b.stable or a.exact_condition_count != b.exact_condition_count:
                _fail(f"{label} {key}: structural diagnostics differ")
            if a.supporting_observations != b.supporting_observations:
                _fail(f"{label} {key}: supporting observations differ")
            for x, y in zip(a.supporting_weights, b.supporting_weights):
                _assert_close(x, y, f"{label} {key} supporting weight")

    compare(reference_a, fast_a, "Method A")
    compare(reference_b, fast_b, "Method B")

    # Exact permutation-gate equivalence on a smaller synthetic family.
    gate_candidates = reference_b[: min(12, len(reference_b))]
    universe = [(row.as_of_date, float(row.stock_return_pct)) for row in history]
    old = search_adjusted_permutation_p_values(
        [(r.supporting_observations, r.supporting_weights) for r in gate_candidates],
        context.baseline_mean,
        family_id="PREFLIGHT",
        universe_observations=universe,
        permutations=13,
        seed=41,
        compute_raw_p_values=False,
    ).max_statistic_p_values
    fast = search_adjusted_permutation_p_values_fast(
        gate_candidates,
        context.baseline_mean,
        family_id="PREFLIGHT",
        universe_observations=universe,
        permutations=13,
        seed=41,
    )
    if old != fast:
        _fail("fast permutation gate is not exactly equivalent on the synthetic test")

    # Small timing benchmark, informational only. It never causes failure.
    bench_candidates = candidates[: min(1500, len(candidates))]
    t0 = perf_counter()
    engine.method_b_conditioned_distribution(current, history, candidate_sets=bench_candidates)
    old_seconds = perf_counter() - t0
    context = prepare_relationship_context(current, history)
    t0 = perf_counter()
    method_b_prepared(engine, current, history, bench_candidates, context)
    fast_seconds = perf_counter() - t0
    ratio = old_seconds / fast_seconds if fast_seconds else float("inf")

    print("PHASE 5.8 OPTIMIZATION PREFLIGHT: PASS")
    print(f"Synthetic candidates checked: {len(sample_candidates)}")
    print(f"Permutation equivalence: exact ({13} permutations)")
    print(f"Prepared Method B timing: {old_seconds:.4f}s -> {fast_seconds:.4f}s ({ratio:.2f}x)")
    print("Trade-gate interface: present")
    print("Point-in-time membership interface: present")
    print("No OLAP query or 250-fold validation was executed.")


if __name__ == "__main__":
    main()
