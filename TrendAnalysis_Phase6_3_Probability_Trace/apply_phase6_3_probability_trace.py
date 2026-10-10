from __future__ import annotations

"""Apply Phase 6.3 source-probability trace propagation and recording contract v2.

Run from the TrendAnalysis repository root:
    py Phase6_3_Probability_Trace/apply_phase6_3_probability_trace.py

The script builds every edit in memory, checks exact source anchors, compiles the
edited Python, and only then writes files. It does not run Phase 5.8 or alter any
existing snapshot/artifact.
"""

from datetime import datetime
from pathlib import Path
import shutil
import sys

ROOT = Path.cwd()
PREDICTION = ROOT / "analysis" / "prediction.py"
VALIDATION = ROOT / "analysis" / "real_prediction_validation.py"
SNAPSHOT = ROOT / "analysis" / "phase5_snapshot.py"
CONTRACT_TEST = ROOT / "tests" / "test_phase5_8_fold_recording_contract.py"
PREFLIGHT_TEST = ROOT / "tests" / "test_phase5_8_snapshot_export_preflight.py"
AUDIT_TEST = ROOT / "tests" / "test_phase6_2_method_probability_audit.py"
TRACE_TEST = ROOT / "tests" / "test_phase6_3_probability_trace.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Preflight failed for {label}: expected one anchor, found {count}. "
            "No source files have been written. Check the source version."
        )
    return text.replace(old, new, 1)


def replace_n(text: str, old: str, new: str, expected: int, label: str) -> str:
    count = text.count(old)
    if count != expected:
        raise RuntimeError(
            f"Preflight failed for {label}: expected {expected} anchors, found {count}. "
            "No source files have been written. Check the source version."
        )
    return text.replace(old, new)


def patch_function_block(
    text: str,
    start_marker: str,
    end_marker: str,
    old: str,
    new: str,
    expected: int,
    label: str,
) -> str:
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"Preflight failed for {label}: start marker not found.")
    end = text.find(end_marker, start + len(start_marker))
    if end < 0:
        raise RuntimeError(f"Preflight failed for {label}: end marker not found.")
    block = text[start:end]
    block = replace_n(block, old, new, expected, label)
    return text[:start] + block + text[end:]


def patch_prediction(text: str) -> str:
    if "probability_basis: str | None = None" in text and '"weighted_class_counts": dict(self.weighted_class_counts)' in text:
        raise RuntimeError(
            "analysis/prediction.py already appears to contain Phase 6.3 trace fields. "
            "Do not apply this overlay twice."
        )

    text = replace_once(
        text,
        "from dataclasses import dataclass\n",
        "from dataclasses import dataclass, field\n",
        "dataclass field import",
    )
    text = replace_once(
        text,
        "    conviction_result: ConvictionResult | None = None\n\n    def as_dict(self) -> dict[str, Any]:\n",
        "    conviction_result: ConvictionResult | None = None\n"
        "    # Source probability-builder diagnostics. These fields are audit-only;\n"
        "    # they do not change probability values or decision behavior.\n"
        "    class_counts: dict[OutcomeClass, int] = field(default_factory=dict)\n"
        "    weighted_class_counts: dict[OutcomeClass, float] = field(default_factory=dict)\n"
        "    probability_basis: str | None = None\n"
        "    exact_condition_count: int | None = None\n"
        "    weight_concentration: float | None = None\n"
        "    stability_score: float | None = None\n\n"
        "    def as_dict(self) -> dict[str, Any]:\n",
        "MethodPrediction trace fields",
    )
    text = replace_once(
        text,
        '            "decision": self.decision.as_dict() if self.decision else None,\n        }\n',
        '            "decision": self.decision.as_dict() if self.decision else None,\n'
        '            "class_counts": dict(self.class_counts),\n'
        '            "weighted_class_counts": dict(self.weighted_class_counts),\n'
        '            "probability_basis": self.probability_basis,\n'
        '            "exact_condition_count": self.exact_condition_count,\n'
        '            "weight_concentration": self.weight_concentration,\n'
        '            "stability_score": self.stability_score,\n'
        '        }\n',
        "MethodPrediction serialized trace fields",
    )

    trace_args = (
        "            condition=probability.condition,\n"
        "            class_counts=dict(probability.class_counts),\n"
        "            weighted_class_counts=dict(probability.weighted_class_counts),\n"
        "            probability_basis=probability.probability_basis,\n"
        "            exact_condition_count=getattr(probability, \"exact_condition_count\", None),\n"
        "            weight_concentration=getattr(probability, \"weight_concentration\", None),\n"
        "            stability_score=getattr(probability, \"stability_score\", None),\n"
    )
    text = patch_function_block(
        text,
        "    def _method_a_prediction(\n",
        "    def _method_prediction(\n",
        "            condition=probability.condition,\n",
        trace_args,
        2,
        "Method A trace propagation in both limited and usable paths",
    )
    text = patch_function_block(
        text,
        "    def _method_prediction(\n",
        "    @staticmethod\n    def _method_weight(",
        "            condition=probability.condition,\n",
        trace_args,
        2,
        "Method B trace propagation in both limited and usable paths",
    )
    compile(text, str(PREDICTION), "exec")
    return text


def patch_validation(text: str) -> str:
    text = replace_once(
        text,
        '        "recording_contract_version": 1,\n',
        '        "recording_contract_version": 2,\n',
        "selection metadata recording-contract version",
    )
    text = replace_once(
        text,
        '            "fold_recording_contract_version": 1,\n',
        '            "fold_recording_contract_version": 2,\n',
        "fold recording-contract version",
    )
    compile(text, str(VALIDATION), "exec")
    return text


def patch_snapshot(text: str) -> str:
    text = replace_once(
        text,
        "FOLD_RECORDING_CONTRACT_VERSION = 1\n",
        "FOLD_RECORDING_CONTRACT_VERSION = 2\n"
        "SUPPORTED_FOLD_RECORDING_CONTRACT_VERSIONS = frozenset({1, FOLD_RECORDING_CONTRACT_VERSION})\n",
        "recording-contract version and compatibility set",
    )
    text = replace_once(
        text,
        "def validate_fold_recording_contract(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:\n",
        "def validate_fold_recording_contract(\n"
        "    rows: Iterable[dict[str, Any]],\n"
        "    *,\n"
        "    contract_version: int | None = None,\n"
        ") -> dict[str, Any]:\n",
        "recording-contract validator signature",
    )
    text = replace_once(
        text,
        '    rows = list(rows)\n    method_present = {"A": 0, "B": 0}\n',
        '    expected_version = (\n'
        '        FOLD_RECORDING_CONTRACT_VERSION if contract_version is None else int(contract_version)\n'
        '    )\n'
        '    if expected_version not in SUPPORTED_FOLD_RECORDING_CONTRACT_VERSIONS:\n'
        '        raise ValueError(f"Unsupported fold recording-contract version: {expected_version!r}")\n'
        '    rows = list(rows)\n    method_present = {"A": 0, "B": 0}\n',
        "recording-contract version selection and compatibility guard",
    )
    text = replace_once(
        text,
        '                f"Fold {index} violates recording contract v{FOLD_RECORDING_CONTRACT_VERSION}; "\n',
        '                f"Fold {index} violates recording contract v{expected_version}; "\n',
        "version-aware recording-contract error",
    )
    text = replace_once(
        text,
        '        if row.get("fold_recording_contract_version") != FOLD_RECORDING_CONTRACT_VERSION:\n',
        '        if row.get("fold_recording_contract_version") != expected_version:\n',
        "fold contract version validation",
    )
    text = replace_once(
        text,
        '        if row["selection_metadata"].get("recording_contract_version") != FOLD_RECORDING_CONTRACT_VERSION:\n',
        '        if row["selection_metadata"].get("recording_contract_version") != expected_version:\n',
        "selection metadata contract version validation",
    )
    text = replace_once(
        text,
        '                if vector_total == 0.0 and not bool(output.get("limited")):\n'
        '                    raise ValueError(f"Fold {index} {vector_key} is all zero but the method is not marked limited.")\n'
        '                method_present[label] += 1\n',
        '                if vector_total == 0.0 and not bool(output.get("limited")):\n'
        '                    raise ValueError(f"Fold {index} {vector_key} is all zero but the method is not marked limited.")\n'
        '                if expected_version >= 2:\n'
        '                    _validate_probability_source_trace(index, label, output, numeric_values)\n'
        '                method_present[label] += 1\n',
        "v2 trace audit call",
    )
    text = replace_once(
        text,
        '        "version": FOLD_RECORDING_CONTRACT_VERSION,\n',
        '        "version": expected_version,\n',
        "recording-contract returned version",
    )
    helper = '''\ndef _validate_probability_source_trace(\n    fold_index: int,\n    method_label: str,\n    output: dict[str, Any],\n    stored_probability_values: list[float],\n) -> None:\n    """Validate the builder metadata needed to explain empirical probabilities."""\n    from .outcome_labels import OUTCOME_CLASSES\n\n    vector_key = f"method_{method_label.lower()}_probabilities_pct"\n    for key in ("class_counts", "weighted_class_counts"):\n        mapping = output.get(key)\n        if not isinstance(mapping, dict) or any(label not in mapping for label in OUTCOME_CLASSES):\n            raise ValueError(\n                f"Fold {fold_index} method {method_label} output is missing valid {key} source trace."\n            )\n\n    counts: dict[str, float] = {}\n    weights: dict[str, float] = {}\n    for label in OUTCOME_CLASSES:\n        try:\n            count = float(output["class_counts"][label])\n            weight = float(output["weighted_class_counts"][label])\n        except (TypeError, ValueError, OverflowError) as exc:\n            raise ValueError(\n                f"Fold {fold_index} method {method_label} source counts are non-numeric."\n            ) from exc\n        if not isfinite(count) or count < 0.0 or not count.is_integer():\n            raise ValueError(\n                f"Fold {fold_index} method {method_label} class_counts must be finite non-negative integers."\n            )\n        if not isfinite(weight) or weight < 0.0:\n            raise ValueError(\n                f"Fold {fold_index} method {method_label} weighted_class_counts must be finite and non-negative."\n            )\n        counts[label] = count\n        weights[label] = weight\n\n    basis = output.get("probability_basis")\n    if not isinstance(basis, str) or not basis.strip():\n        raise ValueError(\n            f"Fold {fold_index} method {method_label} output is missing probability_basis."\n        )\n\n    # Limited outputs intentionally carry an all-zero vector and empty counts.\n    # Usable outputs must reconstruct exactly from the weighted class shares.\n    if bool(output.get("limited")):\n        return\n\n    try:\n        sample_count = float(output.get("sample_count"))\n    except (TypeError, ValueError, OverflowError) as exc:\n        raise ValueError(\n            f"Fold {fold_index} method {method_label} sample_count is missing or invalid."\n        ) from exc\n    if not isfinite(sample_count) or not sample_count.is_integer() or sample_count < 1:\n        raise ValueError(\n            f"Fold {fold_index} method {method_label} sample_count must be a positive integer."\n        )\n    if sum(counts.values()) != sample_count:\n        raise ValueError(\n            f"Fold {fold_index} method {method_label} class_counts sum to {sum(counts.values()):g}, "\n            f"but sample_count is {sample_count:g}."\n        )\n\n    total_weight = sum(weights.values())\n    if not isfinite(total_weight) or total_weight <= 0.0:\n        raise ValueError(\n            f"Fold {fold_index} method {method_label} usable output has no positive weighted class mass."\n        )\n    for label, stored_value in zip(OUTCOME_CLASSES, stored_probability_values):\n        reconstructed_pct = weights[label] / total_weight * 100.0\n        if abs(reconstructed_pct - stored_value) > 1e-4:\n            raise ValueError(\n                f"Fold {fold_index} {vector_key} does not match weighted_class_counts "\n                f"for {label}: stored={stored_value:.8f}, reconstructed={reconstructed_pct:.8f}."\n            )\n\n\n'''
    text = replace_once(
        text,
        "def validate_fold_recording_contract(\n",
        helper + "def validate_fold_recording_contract(\n",
        "source trace validator helper insertion",
    )
    text = replace_once(
        text,
        '        contract = validate_fold_recording_contract(folds)\n',
        '        contract = validate_fold_recording_contract(\n'
        '            folds, contract_version=int(manifest["fold_recording_contract_version"])\n'
        '        )\n',
        "version-aware snapshot load for older contracts",
    )
    compile(text, str(SNAPSHOT), "exec")
    return text


def patch_contract_test(text: str) -> str:
    if "weighted_class_counts" in text and '"probability_basis"' in text:
        raise RuntimeError(
            "The recording-contract test fixture already contains source-trace fields. "
            "Review it manually rather than applying this overlay twice."
        )

    text = replace_once(
        text,
        '        "effective_sample_size": 18.0,\n        "sample_count": 22,\n',
        '        "effective_sample_size": 20.0,\n        "sample_count": 20,\n',
        "Method A consistent support size",
    )
    text = replace_once(
        text,
        '        "effective_sample_size": 12.0,\n        "sample_count": 15,\n',
        '        "effective_sample_size": 20.0,\n        "sample_count": 20,\n',
        "Method B consistent support size",
    )
    # Both examples need realistic source traces. Method A has unit weights;
    # Method B fixture uses a constant weight of 5 per support item.
    text = replace_once(
        text,
        '        "limited": False,\n    }\n    method_b = {',
        '        "limited": False,\n'
        '        "class_counts": {"UP": 11, "SIDEWAYS": 5, "DOWN": 4},\n'
        '        "weighted_class_counts": {"UP": 11.0, "SIDEWAYS": 5.0, "DOWN": 4.0},\n'
        '        "probability_basis": "method_a_empirical_conditional_class_share",\n'
        '    }\n    method_b = {',
        "Method A fixture source trace",
    )
    text = replace_once(
        text,
        '        "limited": False,\n    }\n    selection_a = {',
        '        "limited": False,\n'
        '        "class_counts": {"UP": 4, "SIDEWAYS": 5, "DOWN": 11},\n'
        '        "weighted_class_counts": {"UP": 20.0, "SIDEWAYS": 25.0, "DOWN": 55.0},\n'
        '        "probability_basis": "method_b_weighted_empirical_conditional_class_share",\n'
        '    }\n    selection_a = {',
        "Method B fixture source trace",
    )
    text += '''\n\ndef test_contract_v2_rejects_missing_builder_source_trace() -> None:\n    row = complete_row()\n    del row["method_a_output"]["class_counts"]\n    with pytest.raises(ValueError, match="class_counts source trace"):\n        validate_fold_recording_contract([row])\n\n\ndef test_contract_v2_rejects_weighted_counts_that_do_not_reconstruct_vector() -> None:\n    row = complete_row()\n    row["method_b_output"]["weighted_class_counts"]["UP"] += 5.0\n    with pytest.raises(ValueError, match="does not match weighted_class_counts"):\n        validate_fold_recording_contract([row])\n\n\ndef test_legacy_contract_v1_remains_validatable() -> None:\n    row = complete_row()\n    for method in ("a", "b"):\n        output = row[f"method_{method}_output"]\n        output.pop("class_counts")\n        output.pop("weighted_class_counts")\n        output.pop("probability_basis")\n    row["fold_recording_contract_version"] = 1\n    row["selection_metadata"]["recording_contract_version"] = 1\n    report = validate_fold_recording_contract([row], contract_version=1)\n    assert report["version"] == 1\n'''
    compile(text, str(CONTRACT_TEST), "exec")
    return text



def patch_preflight_test(text: str) -> str:
    if '"probability_basis": "method_a_empirical_conditional_class_share"' in text:
        raise RuntimeError(
            "The snapshot-export preflight test already has Phase 6.3 source traces. "
            "Review it manually rather than applying this overlay twice."
        )

    text = replace_once(
        text,
        '            "effective_sample_size": 18.0,\n            "sample_count": 20,\n',
        '            "effective_sample_size": 100.0,\n            "sample_count": 100,\n',
        "preflight Method A unit-weight support fixture",
    )
    text = replace_once(
        text,
        '            "condition": ["company.market.price=Rising", "macro.Brent_Crude=High"],\n            "limited": False,\n',
        '            "condition": ["company.market.price=Rising", "macro.Brent_Crude=High"],\n            "class_counts": {"UP": 67, "SIDEWAYS": 21, "DOWN": 12},\n            "weighted_class_counts": {"UP": 67.0, "SIDEWAYS": 21.0, "DOWN": 12.0},\n            "probability_basis": "method_a_empirical_conditional_class_share",\n            "limited": False,\n',
        "preflight Method A source trace",
    )
    text = replace_once(
        text,
        '            "condition": ["industry.market=Stable", "macro.Brent_Crude=High"],\n            "limited": False,\n',
        '            "condition": ["industry.market=Stable", "macro.Brent_Crude=High"],\n            "class_counts": {"UP": 2, "SIDEWAYS": 4, "DOWN": 8},\n            "weighted_class_counts": {"UP": 14.0, "SIDEWAYS": 26.0, "DOWN": 60.0},\n            "probability_basis": "method_b_weighted_empirical_conditional_class_share",\n            "exact_condition_count": 2,\n            "weight_concentration": 0.12,\n            "stability_score": 0.7,\n            "limited": False,\n',
        "preflight Method B source trace",
    )
    text = replace_once(
        text,
        '"selection_metadata": {"recording_contract_version": 1,',
        '"selection_metadata": {"recording_contract_version": 2,',
        "preflight selection contract version",
    )
    text = replace_once(
        text,
        '"fold_recording_contract_version": 1,',
        '"fold_recording_contract_version": 2,',
        "preflight fold contract version",
    )
    compile(text, str(PREFLIGHT_TEST), "exec")
    return text


def patch_audit_test(text: str) -> str:
    if '"probability_basis": "method_a_empirical_conditional_class_share"' in text:
        raise RuntimeError(
            "The Phase 6.2 method-probability audit test already has Phase 6.3 source traces. "
            "Review it manually rather than applying this overlay twice."
        )

    text = replace_once(
        text,
        '                "effective_sample_size": 18.0,\n                "sample_count": 20,\n',
        '                "effective_sample_size": 100.0,\n                "sample_count": 100,\n',
        "audit test Method A unit-weight support fixture",
    )
    text = replace_once(
        text,
        '                "condition": ["company.market.price=Rising", "macro.Brent_Crude=High"],\n                "limited": False,\n',
        '                "condition": ["company.market.price=Rising", "macro.Brent_Crude=High"],\n'
        '                "class_counts": {"UP": 67, "SIDEWAYS": 21, "DOWN": 12},\n'
        '                "weighted_class_counts": {"UP": 67.0, "SIDEWAYS": 21.0, "DOWN": 12.0},\n'
        '                "probability_basis": "method_a_empirical_conditional_class_share",\n'
        '                "limited": False,\n',
        "audit test Method A source trace",
    )
    text = replace_once(
        text,
        '                "condition": ["industry.market=Stable", "macro.Brent_Crude=High"],\n                "limited": False,\n',
        '                "condition": ["industry.market=Stable", "macro.Brent_Crude=High"],\n'
        '                "class_counts": {"UP": 2, "SIDEWAYS": 4, "DOWN": 8},\n'
        '                "weighted_class_counts": {"UP": 14.0, "SIDEWAYS": 26.0, "DOWN": 60.0},\n'
        '                "probability_basis": "method_b_weighted_empirical_conditional_class_share",\n'
        '                "exact_condition_count": 2,\n'
        '                "weight_concentration": 0.12,\n'
        '                "stability_score": 0.7,\n'
        '                "limited": False,\n',
        "audit test Method B source trace",
    )
    text = replace_once(
        text,
        '            "recording_contract_version": 1,\n',
        '            "recording_contract_version": 2,\n',
        "audit test fold metadata contract version",
    )
    text = replace_once(
        text,
        'assert row["selection_metadata"]["recording_contract_version"] == 1',
        'assert row["selection_metadata"]["recording_contract_version"] == 2',
        "audit test serialized metadata version expectation",
    )
    text = replace_once(
        text,
        'assert snapshot.manifest["fold_recording_contract_version"] == 1',
        'assert snapshot.manifest["fold_recording_contract_version"] == 2',
        "audit test snapshot contract version expectation",
    )
    compile(text, str(AUDIT_TEST), "exec")
    return text

def trace_test_source() -> str:
    return r'''from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from analysis.decision import DecisionResult
from analysis.prediction import PredictionEngine


BASELINE = {"UP": 33.0, "SIDEWAYS": 34.0, "DOWN": 33.0}


def _fake_probability(*, method_b: bool):
    values = {
        "probabilities_pct": {"UP": 50.0, "SIDEWAYS": 25.0, "DOWN": 25.0},
        "expected_return_pct": 1.2,
        "baseline_return_pct": 0.2,
        "return_lift_pct": 1.0,
        "evidence_score": 0.8,
        "reliability": 0.9,
        "sample_count": 4,
        "effective_sample_size": 4.0,
        "stable": True,
        "variables": ("company.market.price",),
        "condition": ("company.market.price=Rising",),
        "class_counts": {"UP": 2, "SIDEWAYS": 1, "DOWN": 1},
        "weighted_class_counts": {"UP": 2.0, "SIDEWAYS": 1.0, "DOWN": 1.0},
        "probability_basis": "synthetic_weighted_class_share_test",
        "limited": False,
        "limitations": (),
    }
    if method_b:
        values.update({
            "exact_condition_count": 2,
            "weight_concentration": 0.25,
            "stability_score": 0.8,
        })
    return SimpleNamespace(**values)


def _decision():
    return DecisionResult(
        trend="UP",
        selected_class="UP",
        probability_pct=50.0,
        baseline_probability_pct=33.0,
        lift_pct=17.0,
        margin_pct=25.0,
        uncertainty_pct=50.0,
        effective_sample_size=4.0,
        reason="synthetic test decision",
        limited=False,
    )


def _assert_trace(output: dict, *, method_b: bool) -> None:
    assert output["class_counts"] == {"UP": 2, "SIDEWAYS": 1, "DOWN": 1}
    assert output["weighted_class_counts"] == {"UP": 2.0, "SIDEWAYS": 1.0, "DOWN": 1.0}
    assert output["probability_basis"] == "synthetic_weighted_class_share_test"
    if method_b:
        assert output["exact_condition_count"] == 2
        assert output["weight_concentration"] == 0.25
        assert output["stability_score"] == 0.8
    else:
        assert output["exact_condition_count"] is None
        assert output["weight_concentration"] is None
        assert output["stability_score"] is None


def test_method_a_builder_trace_survives_method_prediction_wrapper():
    ranking = SimpleNamespace(method_results=[SimpleNamespace(score=1.0)])
    with patch("analysis.prediction.build_method_a_probability", return_value=_fake_probability(method_b=False)), patch(
        "analysis.prediction.decide_baseline_relative", return_value=_decision()
    ):
        method = PredictionEngine()._method_a_prediction(ranking, None, BASELINE)
    assert method is not None
    _assert_trace(method.as_dict(), method_b=False)


def test_method_b_builder_trace_survives_method_prediction_wrapper():
    best = SimpleNamespace(score=1.0, supporting_observations=[("2020-01-01", 1.0)])
    ranking = SimpleNamespace(method_results=[best])
    with patch("analysis.prediction.build_method_b_probability", return_value=_fake_probability(method_b=True)), patch(
        "analysis.prediction.decide_baseline_relative", return_value=_decision()
    ):
        method = PredictionEngine()._method_prediction(ranking, [], None, BASELINE)
    assert method is not None
    _assert_trace(method.as_dict(), method_b=True)
'''


def main() -> int:
    required = (PREDICTION, VALIDATION, SNAPSHOT, CONTRACT_TEST, PREFLIGHT_TEST, AUDIT_TEST)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        print("ERROR: Run this script from the TrendAnalysis repository root; missing:")
        print("\n".join(f"  - {item}" for item in missing))
        return 2
    if TRACE_TEST.exists():
        print(f"ERROR: {TRACE_TEST.relative_to(ROOT)} already exists; inspect it before retrying.")
        return 3

    originals = {
        PREDICTION: PREDICTION.read_text(encoding="utf-8"),
        VALIDATION: VALIDATION.read_text(encoding="utf-8"),
        SNAPSHOT: SNAPSHOT.read_text(encoding="utf-8"),
        CONTRACT_TEST: CONTRACT_TEST.read_text(encoding="utf-8"),
        PREFLIGHT_TEST: PREFLIGHT_TEST.read_text(encoding="utf-8"),
        AUDIT_TEST: AUDIT_TEST.read_text(encoding="utf-8"),
    }
    try:
        updated = {
            PREDICTION: patch_prediction(originals[PREDICTION]),
            VALIDATION: patch_validation(originals[VALIDATION]),
            SNAPSHOT: patch_snapshot(originals[SNAPSHOT]),
            CONTRACT_TEST: patch_contract_test(originals[CONTRACT_TEST]),
            PREFLIGHT_TEST: patch_preflight_test(originals[PREFLIGHT_TEST]),
            AUDIT_TEST: patch_audit_test(originals[AUDIT_TEST]),
            TRACE_TEST: trace_test_source(),
        }
        for path, content in updated.items():
            compile(content, str(path), "exec") if path.suffix == ".py" else None
    except (RuntimeError, SyntaxError, ValueError) as exc:
        print(f"ERROR: {exc}")
        print("Preflight failed. No source files have been written.")
        return 4

    backup_dir = ROOT / f".phase6_3_probability_trace_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir()
    for path, original in originals.items():
        backup = backup_dir / path.relative_to(ROOT)
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_text(original, encoding="utf-8", newline="\n")

    for path, content in updated.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        print(f"UPDATED: {path.relative_to(ROOT)}")
    print(f"Backup: {backup_dir.relative_to(ROOT)}")
    print("Probability values, combination weights, and decisions are unchanged.")
    print("No Phase 5.8 computation or snapshot was run or modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
