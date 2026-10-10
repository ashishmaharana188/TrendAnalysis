from __future__ import annotations

"""Read-only probability-zero diagnostics and prequential smoothing sensitivity.

The script loads the frozen Phase 5.8 snapshot only. For each model surface it:
- audits exact/near-zero class probabilities and probabilities assigned to outcomes
  that actually occurred;
- compares predeclared additive probability-smoothing strengths;
- calibrates each candidate prequentially, using only prior outcomes that matured
  strictly before each prediction date;
- selects a smoothing alpha on an early chronological development period; and
- evaluates that selected alpha on a later holdout that is not used to select alpha.

Temperature settings are fixed during an individual run so that this experiment
isolates smoothing. The temperature calibrator may update prequentially during
the holdout only after previous outcomes mature; the smoothing alpha remains fixed.
"""

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analysis.outcome_labels import OUTCOME_CLASSES
from analysis.phase6_probability_metrics import (
    calibrate_with_mature_labels,
    compare_probabilities_paired,
    normalize_distribution,
    print_paired_comparison,
    print_score_line,
    score_probabilities,
)
from phase6_2_method_probability_audit import load_folds

_CLASSES = tuple(OUTCOME_CLASSES)
_DEFAULT_ALPHAS = (0.0, 0.0001, 0.001, 0.005, 0.01, 0.02)
_METHODS = (
    ("Combined model", "combined_probabilities_pct", None),
    ("Method A", "method_a_probabilities_pct", "method_a_selection_status"),
    ("Method B", "method_b_probabilities_pct", "method_b_selection_status"),
)
_DEFAULT_NEAR_ZERO_THRESHOLDS = (0.0, 1e-12, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2)


@dataclass(frozen=True)
class ProbabilityAuditCase:
    prediction_date: str
    actual_class: str
    predicted_class: str
    actual_class_probability: float
    actual_class_probability_pct: float
    log_loss_contribution: float
    probability_vector_pct: dict[str, float]
    selection_status: str | None = None


def smooth_distribution(
    probabilities_pct: dict[str, float] | None,
    alpha: float,
) -> dict[str, float] | None:
    """Add a symmetric per-class pseudo-mass alpha to a probability vector.

    Input and output are percentages, but alpha is expressed on the normalized
    probability scale [0, 1]. For example, alpha=0.001 means adding 0.1% mass to
    each class before renormalization; an exact zero becomes approximately
    0.0997% for a three-class distribution. alpha=0 preserves the vector.
    """
    if not math.isfinite(float(alpha)) or alpha < 0.0:
        raise ValueError("alpha must be a finite non-negative number.")
    normalized = normalize_distribution(probabilities_pct)
    if normalized is None:
        return None
    values = np.asarray([normalized[label] / 100.0 for label in _CLASSES], dtype=np.float64)
    if alpha == 0.0:
        return {label: float(values[i] * 100.0) for i, label in enumerate(_CLASSES)}
    smoothed = (values + float(alpha)) / (1.0 + len(_CLASSES) * float(alpha))
    # Guard against tiny floating drift; this does not add information or labels.
    smoothed = smoothed / smoothed.sum()
    return {label: float(smoothed[i] * 100.0) for i, label in enumerate(_CLASSES)}


def probability_zero_diagnostics(
    rows: Sequence[Any],
    probability_field: str,
    *,
    near_zero_thresholds: Sequence[float] = _DEFAULT_NEAR_ZERO_THRESHOLDS,
    top_cases: int = 10,
    selection_status_field: str | None = None,
) -> dict[str, Any]:
    """Audit exact/near-zero entries and probability assigned to the true class.

    Threshold values are normalized probabilities, not percentages. The raw
    scores elsewhere use log-probability clipping at 1e-15, so the audit reports
    true-class probability before clipping and does not disguise zeros.
    """
    if top_cases < 0:
        raise ValueError("top_cases must be non-negative.")
    thresholds = sorted({float(value) for value in near_zero_thresholds})
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("near_zero_thresholds must be finite probabilities in [0, 1].")

    available: list[tuple[Any, dict[str, float], np.ndarray, float, float]] = []
    all_entries: list[float] = []
    exact_zero_by_class = {label: 0 for label in _CLASSES}
    status_counts: Counter[str] = Counter()

    for row in rows:
        if selection_status_field:
            status_counts[str(getattr(row, selection_status_field, "MISSING"))] += 1
        distribution = normalize_distribution(getattr(row, probability_field))
        if distribution is None:
            continue
        values = np.asarray([distribution[label] / 100.0 for label in _CLASSES], dtype=np.float64)
        actual = getattr(row, "actual_class")
        if actual not in _CLASSES:
            continue
        actual_probability = float(values[_CLASSES.index(actual)])
        predicted_index = int(np.argmax(values))
        log_loss_contribution = -math.log(max(actual_probability, 1e-15))
        available.append((row, distribution, values, actual_probability, log_loss_contribution))
        all_entries.extend(float(value) for value in values)
        for index, label in enumerate(_CLASSES):
            if values[index] == 0.0:
                exact_zero_by_class[label] += 1

    if not available:
        return {
            "status": "UNAVAILABLE",
            "available_observations": 0,
            "total_rows": len(rows),
            "exact_zero_entries_total": 0,
            "exact_zero_entries_by_class": exact_zero_by_class,
            "true_class_probability": None,
            "near_zero_counts": [],
            "lowest_probability_cases": [],
            "selection_status_counts": dict(status_counts),
        }

    true_probabilities = np.asarray([item[3] for item in available], dtype=np.float64)
    quantiles = {
        f"p{int(q * 100):02d}": float(np.quantile(true_probabilities, q))
        for q in (0.0, 0.01, 0.05, 0.25, 0.5, 0.9, 1.0)
    }
    near_zero_counts = []
    for threshold in thresholds:
        near_zero_counts.append({
            "threshold_probability": threshold,
            "threshold_pct": threshold * 100.0,
            "true_class_probability_at_or_below_count": int(np.sum(true_probabilities <= threshold)),
            "true_class_probability_at_or_below_pct": float(np.mean(true_probabilities <= threshold) * 100.0),
            "class_entries_at_or_below_count": int(np.sum(np.asarray(all_entries) <= threshold)),
            "class_entries_at_or_below_pct": float(np.mean(np.asarray(all_entries) <= threshold) * 100.0),
        })

    worst = sorted(available, key=lambda item: (item[3], getattr(item[0], "prediction_date")))[:top_cases]
    cases = []
    for row, distribution, values, actual_probability, contribution in worst:
        predicted = _CLASSES[int(np.argmax(values))]
        cases.append(asdict(ProbabilityAuditCase(
            prediction_date=str(getattr(row, "prediction_date")),
            actual_class=str(getattr(row, "actual_class")),
            predicted_class=predicted,
            actual_class_probability=actual_probability,
            actual_class_probability_pct=actual_probability * 100.0,
            log_loss_contribution=contribution,
            probability_vector_pct=distribution,
            selection_status=(
                str(getattr(row, selection_status_field, "MISSING"))
                if selection_status_field else None
            ),
        )))

    exact_zeros = int(sum(exact_zero_by_class.values()))
    return {
        "status": "COMPLETE" if len(available) == len(rows) else "PARTIAL",
        "available_observations": len(available),
        "total_rows": len(rows),
        "availability_pct": len(available) / len(rows) * 100.0 if rows else 0.0,
        "probability_entries": len(all_entries),
        "exact_zero_entries_total": exact_zeros,
        "exact_zero_entries_by_class": exact_zero_by_class,
        "true_class_probability": {
            "min": float(true_probabilities.min()),
            "max": float(true_probabilities.max()),
            "mean": float(true_probabilities.mean()),
            "median": float(np.median(true_probabilities)),
            "quantiles_probability_scale": quantiles,
            "exact_zero_true_class_count": int(np.sum(true_probabilities == 0.0)),
            "exact_zero_true_class_pct": float(np.mean(true_probabilities == 0.0) * 100.0),
            "mean_log_loss_contribution_with_1e-15_floor": float(np.mean([item[4] for item in available])),
        },
        "near_zero_counts": near_zero_counts,
        "lowest_probability_cases": cases,
        "selection_status_counts": dict(status_counts),
        "note": "Near-zero thresholds use normalized probability units in [0,1]; raw values are reported before log-loss clipping.",
    }


def chronological_split_indices(total_rows: int, development_fraction: float = 0.70) -> tuple[list[int], list[int]]:
    """Split by global chronological fold index to keep dates aligned across methods."""
    if total_rows < 2:
        raise ValueError("At least two chronological folds are required.")
    if not 0.5 <= development_fraction < 1.0:
        raise ValueError("development_fraction must be in [0.5, 1.0).")
    split = min(max(int(math.floor(total_rows * development_fraction)), 1), total_rows - 1)
    return list(range(split)), list(range(split, total_rows))


def _available_indices(
    indices: Sequence[int],
    probability_rows: Sequence[dict[str, float] | None],
) -> list[int]:
    return [index for index in indices if probability_rows[index] is not None]


def _score_indices(
    rows: Sequence[Any],
    indices: Sequence[int],
    probability_rows: Sequence[dict[str, float] | None],
) -> dict[str, Any] | None:
    if not indices:
        return None
    forecasts = [probability_rows[index] for index in indices]
    if any(item is None for item in forecasts):
        raise ValueError("Internal error: score indices include unavailable forecasts.")
    return score_probabilities(
        [rows[index].actual_class for index in indices],
        [item for item in forecasts if item is not None],
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _print_near_zero_audit(name: str, audit: dict[str, Any]) -> None:
    print(f"  {name}: available={audit['available_observations']}/{audit['total_rows']} status={audit['status']}")
    if audit["available_observations"] == 0:
        return
    true_prob = audit["true_class_probability"]
    print(
        f"    exact zero entries={audit['exact_zero_entries_total']}/"
        f"{audit['probability_entries']} class probabilities; "
        f"true-class exact zeros={true_prob['exact_zero_true_class_count']}/"
        f"{audit['available_observations']} ({true_prob['exact_zero_true_class_pct']:.2f}%)"
    )
    print(
        f"    true-class probability: min={true_prob['min']:.3g}; "
        f"p01={true_prob['quantiles_probability_scale']['p01']:.3g}; "
        f"median={true_prob['median']:.3g}; mean={true_prob['mean']:.3g}"
    )
    for row in audit["near_zero_counts"]:
        print(
            f"    p(actual) <= {row['threshold_probability']:.3g}: "
            f"{row['true_class_probability_at_or_below_count']}/"
            f"{audit['available_observations']} "
            f"({row['true_class_probability_at_or_below_pct']:.2f}%)"
        )
    print("    Lowest true-class probability cases:")
    for case in audit["lowest_probability_cases"][:5]:
        print(
            f"      {case['prediction_date']} actual={case['actual_class']} "
            f"predicted={case['predicted_class']} p(actual)="
            f"{case['actual_class_probability']:.3g} "
            f"log-loss contribution={case['log_loss_contribution']:.3f}"
        )


def _candidate_probabilities(
    rows: Sequence[Any],
    field: str,
    alpha: float,
) -> list[dict[str, float] | None]:
    return [smooth_distribution(getattr(row, field), alpha) for row in rows]


def _analyze_method(
    rows: Sequence[Any],
    method_name: str,
    probability_field: str,
    selection_status_field: str | None,
    *,
    alphas: Sequence[float],
    development_indices: Sequence[int],
    holdout_indices: Sequence[int],
    min_calibration_observations: int,
    min_development_fitted_observations: int,
    top_cases: int,
    temperature_min: float,
    temperature_max: float,
    temperature_grid_size: int,
    block_length: int,
    bootstrap_replicates: int,
    seed_base: int,
) -> dict[str, Any]:
    dates = [row.prediction_date for row in rows]
    ends = [row.outcome_end_date for row in rows]
    actuals = [row.actual_class for row in rows]
    raw = [normalize_distribution(getattr(row, probability_field)) for row in rows]
    baseline = [normalize_distribution(row.baseline_probabilities_pct) for row in rows]
    available = [index for index, probs in enumerate(raw) if probs is not None]
    dev_available = [index for index in development_indices if raw[index] is not None]
    test_available = [index for index in holdout_indices if raw[index] is not None]
    holdout_start_date = rows[holdout_indices[0]].prediction_date
    # Hyperparameter selection at the holdout boundary may only use development
    # labels that had actually matured before that boundary. A development fold
    # whose forward outcome ended on/after the boundary is excluded from alpha
    # selection, even though its eventual label exists in this retrospective file.
    dev_selection_matured = [
        index for index in dev_available
        if rows[index].outcome_end_date is not None
        and rows[index].outcome_end_date < holdout_start_date
    ]

    print(f"\n{method_name}")
    print(f"  probability vectors: {len(available)}/{len(rows)}; development={len(dev_available)}; holdout={len(test_available)}")
    if selection_status_field:
        statuses = Counter(getattr(row, selection_status_field) for row in rows)
        print(f"  stored selection statuses: {dict(statuses)}")
    _print_near_zero_audit(
        "raw probability-zero audit",
        probability_zero_diagnostics(
            rows,
            probability_field,
            top_cases=top_cases,
            selection_status_field=selection_status_field,
        ),
    )
    if not available:
        print("  STATUS: no available forecasts; smoothing sensitivity not estimated.")
        return {
            "status": "UNAVAILABLE",
            "available_observations": 0,
            "raw_probability_audit": probability_zero_diagnostics(rows, probability_field, top_cases=top_cases, selection_status_field=selection_status_field),
            "development_candidates": [],
            "selected_alpha": None,
            "holdout": None,
        }

    candidates: dict[float, dict[str, Any]] = {}
    development_scores: list[dict[str, Any]] = []
    for candidate_index, alpha in enumerate(alphas):
        smoothed = _candidate_probabilities(rows, probability_field, alpha)
        calibration = calibrate_with_mature_labels(
            dates,
            ends,
            actuals,
            smoothed,
            min_calibration_observations=min_calibration_observations,
            temperature_min=temperature_min,
            temperature_max=temperature_max,
            temperature_grid_size=temperature_grid_size,
        )
        calibrated = list(calibration.probabilities_pct)
        dev_fitted = [
            index for index in dev_selection_matured
            if smoothed[index] is not None
            and calibration.statuses[index] == "FITTED"
            and calibrated[index] is not None
        ]
        dev_calibrated_fitted = _score_indices(rows, dev_fitted, calibrated) if dev_fitted else None
        dev_raw = _score_indices(rows, dev_available, smoothed)
        dev_calibrated_all = _score_indices(
            rows,
            [index for index in dev_available if calibrated[index] is not None],
            calibrated,
        )
        eligible_for_selection = (
            dev_calibrated_fitted is not None
            and dev_calibrated_fitted["observations"] >= min_development_fitted_observations
        )
        item = {
            "alpha": float(alpha),
            "interpretation": "per-class pseudo-mass on normalized probability scale",
            "development_available_observations": len(dev_available),
            "development_outcomes_matured_before_holdout_observations": len(dev_selection_matured),
            "development_fitted_observations": len(dev_fitted),
            "development_raw_smoothed_metrics": dev_raw,
            "development_prequential_calibrated_all_metrics": dev_calibrated_all,
            "development_prequential_calibrated_fitted_only_metrics": dev_calibrated_fitted,
            "selection_eligible": eligible_for_selection,
            "calibration_status_counts": dict(Counter(
                calibration.statuses[index] for index in dev_available
            )),
            "calibration_statuses": list(calibration.statuses),
            "temperatures": list(calibration.temperatures),
            "smoothed_probabilities": smoothed,
            "calibrated_probabilities": calibrated,
            "mature_history_counts": list(calibration.mature_training_observations),
        }
        candidates[float(alpha)] = item
        development_scores.append({
            "alpha": float(alpha),
            "selection_eligible": eligible_for_selection,
            "fitted_observations": len(dev_fitted),
            "fitted_only_log_loss": (
                dev_calibrated_fitted["log_loss"] if dev_calibrated_fitted is not None else None
            ),
            "fitted_only_brier_score": (
                dev_calibrated_fitted["brier_score"] if dev_calibrated_fitted is not None else None
            ),
        })

    eligible = [row for row in development_scores if row["selection_eligible"]]
    if not eligible:
        print(
            f"  STATUS: insufficient development sample; no alpha selected "
            f"(minimum fitted observations={min_development_fitted_observations})."
        )
        return {
            "status": "INSUFFICIENT_DEVELOPMENT_SAMPLE",
            "available_observations": len(available),
            "raw_probability_audit": probability_zero_diagnostics(rows, probability_field, top_cases=top_cases, selection_status_field=selection_status_field),
            "development_candidates": development_scores,
            "selected_alpha": None,
            "holdout": None,
        }

    # Selection uses only the chronological development period and only rows
    # whose calibrator had the configured minimum mature-history sample.
    chosen_summary = min(
        eligible,
        key=lambda item: (float(item["fitted_only_log_loss"]), float(item["alpha"])),
    )
    chosen_alpha = float(chosen_summary["alpha"])
    chosen = candidates[chosen_alpha]
    no_smoothing = candidates.get(0.0)
    print("  Development-only alpha selection (minimize fitted-only calibrated log loss):")
    for result in sorted(development_scores, key=lambda item: item["alpha"]):
        metric = result["fitted_only_log_loss"]
        print(
            f"    alpha={result['alpha']:.6g}: n={result['fitted_observations']}; "
            f"fitted-only calibrated log loss={metric if metric is not None else float('nan'):.5f}; "
            f"eligible={result['selection_eligible']}"
        )
    print(f"  selected alpha from development only: {chosen_alpha:g}")
    print(
        "  holdout protocol: alpha stays fixed; temperature is updated only "
        "prequentially from prior labels matured before each holdout prediction."
    )

    test_avail = [index for index in holdout_indices if raw[index] is not None]
    chosen_raw_test = _score_indices(rows, test_avail, chosen["smoothed_probabilities"])
    chosen_cal_test = _score_indices(
        rows,
        [index for index in test_avail if chosen["calibrated_probabilities"][index] is not None],
        chosen["calibrated_probabilities"],
    )
    baseline_test_indices = [index for index in test_avail if baseline[index] is not None]
    baseline_test = _score_indices(rows, baseline_test_indices, baseline)

    reference = no_smoothing if no_smoothing is not None else chosen
    reference_raw_test = _score_indices(rows, test_avail, reference["smoothed_probabilities"])
    reference_cal_test = _score_indices(
        rows,
        [index for index in test_avail if reference["calibrated_probabilities"][index] is not None],
        reference["calibrated_probabilities"],
    )

    comparisons: dict[str, Any] = {}
    if test_avail:
        actual_test = [actuals[index] for index in test_avail]
        comparisons["selected_smoothed_raw_vs_unsmoothed_raw"] = compare_probabilities_paired(
            actual_test,
            [chosen["smoothed_probabilities"][index] for index in test_avail],
            [reference["smoothed_probabilities"][index] for index in test_avail],
            block_length=block_length,
            replicates=bootstrap_replicates,
            seed=seed_base + 11,
        )
        comparisons["selected_calibrated_vs_unsmoothed_calibrated"] = compare_probabilities_paired(
            actual_test,
            [chosen["calibrated_probabilities"][index] for index in test_avail],
            [reference["calibrated_probabilities"][index] for index in test_avail],
            block_length=block_length,
            replicates=bootstrap_replicates,
            seed=seed_base + 12,
        )
    if baseline_test_indices:
        comparisons["selected_calibrated_vs_stored_baseline"] = compare_probabilities_paired(
            [actuals[index] for index in baseline_test_indices],
            [chosen["calibrated_probabilities"][index] for index in baseline_test_indices],
            [baseline[index] for index in baseline_test_indices],
            block_length=block_length,
            replicates=bootstrap_replicates,
            seed=seed_base + 13,
        )

    print("  Holdout results (smoothing alpha selected on development only):")
    print_score_line("selected alpha raw-smoothed", chosen_raw_test) if chosen_raw_test else print("    selected alpha raw-smoothed: n=0")
    print_score_line("selected alpha prequential calibrated", chosen_cal_test) if chosen_cal_test else print("    selected alpha prequential calibrated: n=0")
    print_score_line("unsmoothed raw reference", reference_raw_test) if reference_raw_test else print("    unsmoothed raw reference: n=0")
    print_score_line("unsmoothed prequential calibrated reference", reference_cal_test) if reference_cal_test else print("    unsmoothed calibrated reference: n=0")
    print_score_line("stored per-fold baseline, paired holdout", baseline_test) if baseline_test else print("    stored per-fold baseline: n=0")
    for title, value in comparisons.items():
        print_paired_comparison(title.replace("_", " "), value)

    holdout_status = "EVALUATED" if len(test_avail) >= 10 else "TOO_FEW_HOLDOUT_OBSERVATIONS_FOR_ROBUST_INFERENCE"
    return {
        "status": holdout_status,
        "available_observations": len(available),
        "raw_probability_audit": probability_zero_diagnostics(
            rows,
            probability_field,
            selection_status_field=selection_status_field,
        ),
        "selection": {
            "selected_alpha": chosen_alpha,
            "selection_metric": "minimum_prequential_calibrated_log_loss_on_development_fitted_only_rows_whose_outcomes_matured_before_holdout_start",
            "development_start_date": str(rows[development_indices[0]].prediction_date),
            "development_end_date": str(rows[development_indices[-1]].prediction_date),
            "development_folds_global": len(development_indices),
            "development_available_observations": len(dev_available),
            "development_outcomes_matured_before_holdout_observations": len(dev_selection_matured),
            "development_selection_last_prediction_date": str(rows[dev_selection_matured[-1]].prediction_date) if dev_selection_matured else None,
            "selection_outcome_maturity_cutoff_exclusive": str(holdout_start_date),
            "development_fitted_observations": int(chosen_summary["fitted_observations"]),
            "holdout_start_date": str(rows[holdout_indices[0]].prediction_date),
            "holdout_end_date": str(rows[holdout_indices[-1]].prediction_date),
            "holdout_folds_global": len(holdout_indices),
            "holdout_available_observations": len(test_avail),
            "warning": "Alpha was selected only from development predictions whose outcomes matured strictly before holdout start. This is not an independent holdout for the whole Phase 5.8 modeling pipeline; it is reserved for this post-hoc smoothing sensitivity experiment.",
        },
        "development_candidates": development_scores,
        "holdout": {
            "selected_alpha_raw_smoothed": chosen_raw_test,
            "selected_alpha_prequential_calibrated": chosen_cal_test,
            "unsmoothed_raw_reference": reference_raw_test,
            "unsmoothed_prequential_calibrated_reference": reference_cal_test,
            "stored_baseline_paired": baseline_test,
            "paired_comparisons": comparisons,
            "selected_alpha_calibration_status_counts": dict(Counter(
                chosen["calibration_statuses"][index] for index in test_avail
            )),
            "selected_alpha_mature_history_min": min(
                chosen["mature_history_counts"][index] for index in test_avail
            ) if test_avail else None,
            "selected_alpha_mature_history_median": float(np.median([
                chosen["mature_history_counts"][index] for index in test_avail
            ])) if test_avail else None,
            "selected_alpha_mature_history_max": max(
                chosen["mature_history_counts"][index] for index in test_avail
            ) if test_avail else None,
        },
    }


def run_sensitivity(
    rows: Sequence[Any],
    *,
    alphas: Sequence[float] = _DEFAULT_ALPHAS,
    development_fraction: float = 0.70,
    min_calibration_observations: int = 30,
    min_development_fitted_observations: int = 20,
    temperature_min: float = 0.25,
    temperature_max: float = 4.0,
    temperature_grid_size: int = 161,
    near_zero_thresholds: Sequence[float] = _DEFAULT_NEAR_ZERO_THRESHOLDS,
    block_length: int = 5,
    bootstrap_replicates: int = 1000,
    seed: int = 20261010,
    top_cases: int = 10,
) -> dict[str, Any]:
    """Run zero-probability audit and maturity-aware development-selected smoothing test."""
    if not rows:
        raise ValueError("No usable snapshot folds were supplied.")
    clean_alphas = tuple(sorted({float(alpha) for alpha in alphas}))
    if not clean_alphas:
        raise ValueError("At least one smoothing alpha is required.")
    if any(not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0 for alpha in clean_alphas):
        raise ValueError("Smoothing alphas must be finite and in [0,1].")
    if 0.0 not in clean_alphas:
        raise ValueError("Alpha 0.0 must be included as the unsmoothed reference.")
    if not 0.0 < temperature_min < temperature_max:
        raise ValueError("Require 0 < temperature_min < temperature_max.")
    if temperature_grid_size < 3:
        raise ValueError("temperature_grid_size must be >= 3.")
    if min_calibration_observations < 1 or min_development_fitted_observations < 1:
        raise ValueError("Calibration and development minimum sample sizes must be >= 1.")
    if block_length < 1 or bootstrap_replicates < 100:
        raise ValueError("block_length must be >=1 and bootstrap_replicates must be >=100.")
    if top_cases < 0:
        raise ValueError("top_cases must be >= 0.")

    development_indices, holdout_indices = chronological_split_indices(len(rows), development_fraction)
    date_start, date_end = str(rows[0].prediction_date), str(rows[-1].prediction_date)
    print("PHASE 6.2 PROBABILITY-ZERO AUDIT AND SMOOTHING SENSITIVITY")
    print(f"Snapshot folds: {len(rows)}")
    print(f"Dates: {date_start} through {date_end}")
    print("Protocol: FROZEN_FOLDS; NO_OLAP_RECOMPUTATION")
    print("Calibration: prequential; prior outcomes used only if they matured strictly before a prediction date")
    print(f"Temperature range fixed for smoothing comparison: [{temperature_min:g}, {temperature_max:g}], grid={temperature_grid_size}")
    print(f"Predeclared alphas: {list(clean_alphas)} (per-class pseudo-mass in probability units)")
    print(f"Chronological split: development={len(development_indices)} folds through {rows[development_indices[-1]].prediction_date}; holdout={len(holdout_indices)} folds from {rows[holdout_indices[0]].prediction_date}")
    print(f"Alpha-selection label cutoff: only development outcomes matured strictly before {rows[holdout_indices[0]].prediction_date} are eligible.")
    print(f"Block bootstrap: block_length={block_length}, replicates={bootstrap_replicates}")
    print("WARNING: this is a post-hoc smoothing sensitivity study. Alpha selection uses development only; the later period is reserved for this experiment, not a fresh evaluation of the original model-development process.")

    report: dict[str, Any] = {
        "protocol": "PROBABILITY_ZERO_AUDIT_AND_DEVELOPMENT_SELECTED_PREQUENTIAL_SMOOTHING",
        "snapshot_status": "FROZEN",
        "snapshot_folds": len(rows),
        "date_start": date_start,
        "date_end": date_end,
        "outcome_maturity_gate": "STRICTLY_BEFORE_CURRENT_PREDICTION_DATE",
        "temperature_min": temperature_min,
        "temperature_max": temperature_max,
        "temperature_grid_size": temperature_grid_size,
        "alphas": list(clean_alphas),
        "development_fraction": development_fraction,
        "development_index_split": len(development_indices),
        "development_start_date": str(rows[development_indices[0]].prediction_date),
        "development_end_date": str(rows[development_indices[-1]].prediction_date),
        "selection_cutoff_exclusive": str(rows[holdout_indices[0]].prediction_date),
        "development_outcomes_matured_before_selection_cutoff": sum(
            1 for index in development_indices
            if rows[index].outcome_end_date is not None
            and rows[index].outcome_end_date < rows[holdout_indices[0]].prediction_date
        ),
        "holdout_start_date": str(rows[holdout_indices[0]].prediction_date),
        "holdout_end_date": str(rows[holdout_indices[-1]].prediction_date),
        "minimum_calibration_observations": min_calibration_observations,
        "minimum_development_fitted_observations": min_development_fitted_observations,
        "near_zero_thresholds_probability_scale": list(near_zero_thresholds),
        "block_length": block_length,
        "bootstrap_replicates": bootstrap_replicates,
        "methods": {},
    }

    for index, (name, field, status_field) in enumerate(_METHODS):
        report["methods"][name] = _analyze_method(
            rows,
            name,
            field,
            status_field,
            alphas=clean_alphas,
            development_indices=development_indices,
            holdout_indices=holdout_indices,
            min_calibration_observations=min_calibration_observations,
            min_development_fitted_observations=min_development_fitted_observations,
            top_cases=top_cases,
            temperature_min=temperature_min,
            temperature_max=temperature_max,
            temperature_grid_size=temperature_grid_size,
            block_length=block_length,
            bootstrap_replicates=bootstrap_replicates,
            seed_base=seed + index * 10000,
        )

    return _json_safe(report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit near-zero frozen probabilities and test predeclared smoothing strengths without OLAP recomputation."
    )
    parser.add_argument("snapshot", help="Frozen Phase 5.8 snapshot directory.")
    parser.add_argument("--alphas", nargs="+", type=float, default=list(_DEFAULT_ALPHAS),
                        help="Predeclared per-class pseudo-mass values on probability scale; must include 0.0.")
    parser.add_argument("--development-fraction", type=float, default=0.70,
                        help="Chronological development fraction; later folds are reserved for holdout evaluation.")
    parser.add_argument("--min-calibration-observations", type=int, default=30)
    parser.add_argument("--min-development-fitted-observations", type=int, default=20)
    parser.add_argument("--temperature-min", type=float, default=0.25)
    parser.add_argument("--temperature-max", type=float, default=4.0,
                        help="Fixed during this experiment to isolate smoothing from temperature-range sensitivity.")
    parser.add_argument("--temperature-grid-size", type=int, default=161)
    parser.add_argument("--near-zero-thresholds", nargs="+", type=float, default=list(_DEFAULT_NEAR_ZERO_THRESHOLDS),
                        help="True-class probability thresholds in normalized [0,1] units.")
    parser.add_argument("--top-cases", type=int, default=10)
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--json-out", help="Optional output path for machine-readable report.")
    args = parser.parse_args()

    # top-cases is parsed for compatibility with the diagnostic helper; keep the
    # requested value on the module call by setting a CLI override below.
    alphas = tuple(sorted(set(args.alphas)))
    if 0.0 not in alphas:
        parser.error("--alphas must include 0.0 as the unsmoothed reference.")
    if args.top_cases < 0:
        parser.error("--top-cases must be >= 0.")

    rows = load_folds(args.snapshot)
    report = run_sensitivity(
        rows,
        alphas=alphas,
        development_fraction=args.development_fraction,
        min_calibration_observations=args.min_calibration_observations,
        min_development_fitted_observations=args.min_development_fitted_observations,
        temperature_min=args.temperature_min,
        temperature_max=args.temperature_max,
        temperature_grid_size=args.temperature_grid_size,
        near_zero_thresholds=args.near_zero_thresholds,
        block_length=args.block_length,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        top_cases=args.top_cases,
    )
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nJSON report written: {output}")

    print("\nInterpretation guardrails:")
    print("  - Exact zero counts are reported before the scoring code's 1e-15 log-loss floor.")
    print("  - Alpha is selected using only fitted prequential calibration scores in the development period.")
    print("  - Holdout metrics do not influence alpha selection; the temperature calibrator can update only from matured prior labels.")
    print("  - Additive smoothing changes probabilities, not the original stored forecast or outcome labels.")
    print("  - No Phase 5.8 relationship search, snapshot mutation or OLAP query is performed.")


if __name__ == "__main__":
    main()
