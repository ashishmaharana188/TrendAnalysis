from datetime import date, timedelta
from random import Random

from analysis.fast_permutation import search_adjusted_permutation_p_values_fast
from analysis.relationship import RelationshipResult


def _result(support, weights):
    return RelationshipResult(
        method="B",
        variables=("f",),
        condition=("f=A",),
        sample_count=len(support),
        mean_return_pct=0.0,
        median_return_pct=0.0,
        baseline_mean_return_pct=0.0,
        lift_pct=0.0,
        positive_rate_pct=0.0,
        effect_strength=0.0,
        reliability=1.0,
        score=0.0,
        stable=True,
        supporting_observations=tuple(support),
        supporting_weights=tuple(weights),
    )


def _reference(candidates, baseline, universe, permutations, seed):
    dates = sorted(day for day, _ in universe)
    values = dict(universe)
    maps = []
    observed = []
    for result in candidates:
        m = {day: (float(value), max(float(weight), 0.0)) for (day, value), weight in zip(result.supporting_observations, result.supporting_weights)}
        maps.append(m)
        total = sum(w for _, w in m.values())
        observed.append(abs(sum(v*w for v,w in m.values())/total - baseline))
    exceed = [1] * len(maps)
    rng = Random(seed)
    vals = [values[d] for d in dates]
    for _ in range(permutations):
        shuffled = list(vals)
        rng.shuffle(shuffled)
        perm = dict(zip(dates, shuffled))
        max_stat = 0.0
        for m in maps:
            total = sum(w for _,w in m.values())
            pm = sum(perm[d] * w for d, (_,w) in m.items()) / total
            max_stat = max(max_stat, abs(pm - baseline))
        for i, stat in enumerate(observed):
            if max_stat >= stat:
                exceed[i] += 1
    return tuple(min(1.0, x/(permutations+1)) for x in exceed)


def test_fast_gate_matches_reference():
    base=date(2021,1,1)
    universe=[(base+timedelta(days=i), float(i-3)/10.0) for i in range(12)]
    a=universe[:8]
    b=universe[2:10]
    candidates=[
        _result(a, [1.0,0.5,2.0,1.5,1.0,0.75,1.25,0.25]),
        _result(b, [0.5,1.0,1.0,0.5,2.0,1.0,0.75,1.25]),
    ]
    expected=_reference(candidates, 0.0, universe, 37, 19)
    actual=search_adjusted_permutation_p_values_fast(
        candidates, 0.0, family_id="f", universe_observations=universe,
        permutations=37, seed=19,
    )
    assert actual == expected


def test_matrix_gate_matches_scalar_reference():
    import numpy as np
    from analysis.fast_permutation import search_adjusted_permutation_p_values_matrix

    base = date(2022, 1, 1)
    universe = [(base + timedelta(days=i), float(i - 5) / 7.0) for i in range(15)]
    values = np.asarray([value for _day, value in universe], dtype=np.float64)
    support = np.zeros((3, len(values)), dtype=np.float64)
    support[0, [0, 2, 5, 7]] = [1.0, 0.5, 2.0, 1.5]
    support[1, [1, 3, 8, 10, 12]] = [0.25, 1.0, 0.75, 2.0, 1.25]
    support[2, [2, 4, 6, 9, 11, 14]] = 1.0

    baseline = float(np.mean(values))
    permutations = 37
    seed = 23
    actual = search_adjusted_permutation_p_values_matrix(
        support, values, baseline, permutations=permutations, seed=seed
    )

    from random import Random
    observed = np.abs((support @ values) / support.sum(axis=1) - baseline)
    exceed = np.ones(3, dtype=np.int64)
    rng = Random(seed)
    for _ in range(permutations):
        shuffled = list(values)
        rng.shuffle(shuffled)
        permuted = np.asarray(shuffled, dtype=np.float64)
        max_stat = float(np.max(np.abs((support @ permuted) / support.sum(axis=1) - baseline)))
        exceed += max_stat >= observed
    expected = tuple(np.minimum(1.0, exceed / (permutations + 1)).tolist())
    assert actual == expected
