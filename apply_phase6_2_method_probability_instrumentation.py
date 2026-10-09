from __future__ import annotations

"""Apply the Phase 6.2 Method A/B probability recording patch.

Run from the TrendAnalysis repository root:
    py apply_phase6_2_method_probability_instrumentation.py

The script makes timestamp-free .phase6_2.bak copies before modifying source
files. It refuses to proceed when expected source anchors are not found.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path.cwd()
VALIDATION = ROOT / "analysis" / "real_prediction_validation.py"
SNAPSHOT = ROOT / "analysis" / "phase5_snapshot.py"


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one source anchor for {description}; found {count}. "
            "No source files have been modified. Check that this patch matches "
            "your current repository version."
        )
    return text.replace(old, new, 1)


def backup_and_write(path: Path, new_text: str) -> None:
    backup = path.with_name(path.name + ".phase6_2.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(new_text, encoding="utf-8", newline="\n")


def patch_validation(text: str) -> str:
    if "method_a_probabilities_pct: dict[OutcomeClass, float] = field(default_factory=dict)" in text and "method_b_probabilities_pct=(" in text:
        print("Prediction-fold vector recording already appears to be installed.")
        return text

    text = replace_once(
        text,
        "from dataclasses import dataclass\n",
        "from dataclasses import dataclass, field\n",
        "dataclass field import",
    )
    text = replace_once(
        text,
        "    trade_reason: str\n\n    def as_dict(self) -> dict[str, Any]:\n",
        "    trade_reason: str\n"
        "    # Instrumentation only: these vectors are copied from the method results;\n"
        "    # they do not affect Phase 5.8 selection, probabilities, or decisions.\n"
        "    method_a_probabilities_pct: dict[OutcomeClass, float] = field(default_factory=dict)\n"
        "    method_b_probabilities_pct: dict[OutcomeClass, float] = field(default_factory=dict)\n\n"
        "    def as_dict(self) -> dict[str, Any]:\n",
        "prediction fold vector fields",
    )
    text = replace_once(
        text,
        '            "trade_reason": self.trade_reason,\n        }\n',
        '            "trade_reason": self.trade_reason,\n'
        '            "method_a_probabilities_pct": dict(self.method_a_probabilities_pct),\n'
        '            "method_b_probabilities_pct": dict(self.method_b_probabilities_pct),\n'
        '        }\n',
        "prediction fold JSON serialization",
    )
    text = replace_once(
        text,
        "                    method_b_trend=result.method_b.trend if result.method_b else None,\n",
        "                    method_b_trend=result.method_b.trend if result.method_b else None,\n"
        "                    method_a_probabilities_pct=(\n"
        "                        dict(result.method_a.probabilities_pct) if result.method_a else {}\n"
        "                    ),\n"
        "                    method_b_probabilities_pct=(\n"
        "                        dict(result.method_b.probabilities_pct) if result.method_b else {}\n"
        "                    ),\n",
        "prediction fold method result recording",
    )
    return text


def patch_snapshot(text: str) -> str:
    # Keep this patch idempotent and backwards compatible with existing v1 snapshots.
    if 'SNAPSHOT_SCHEMA_VERSION = "2"' not in text:
        text = replace_once(
            text,
            'SNAPSHOT_SCHEMA_VERSION = "1"\n',
            'SNAPSHOT_SCHEMA_VERSION = "2"\n'
            'SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS = frozenset({"1", SNAPSHOT_SCHEMA_VERSION})\n',
            "snapshot schema constant",
        )
    elif "SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS" not in text:
        text = replace_once(
            text,
            'SNAPSHOT_SCHEMA_VERSION = "2"\n',
            'SNAPSHOT_SCHEMA_VERSION = "2"\n'
            'SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS = frozenset({"1", SNAPSHOT_SCHEMA_VERSION})\n',
            "supported snapshot schema set",
        )

    old_guard = '    if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:\n'
    new_guard = '    if manifest.get("schema_version") not in SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS:\n'
    if old_guard in text:
        text = replace_once(text, old_guard, new_guard, "backwards-compatible schema guard")
    elif new_guard not in text:
        raise RuntimeError("Could not locate the snapshot schema validation guard.")

    old_default = '    version: str = "v1",\n'
    new_default = '    version: str = "v2",\n'
    if old_default in text:
        text = replace_once(text, old_default, new_default, "default snapshot version")
    elif new_default not in text:
        raise RuntimeError("Could not locate default_snapshot_path version parameter.")
    return text


def main() -> int:
    missing = [str(path.relative_to(ROOT)) for path in (VALIDATION, SNAPSHOT) if not path.is_file()]
    if missing:
        print("ERROR: Run this script from the TrendAnalysis repository root; missing:")
        print("\n".join(f"  - {item}" for item in missing))
        return 2

    original_validation = VALIDATION.read_text(encoding="utf-8")
    original_snapshot = SNAPSHOT.read_text(encoding="utf-8")
    try:
        patched_validation = patch_validation(original_validation)
        patched_snapshot = patch_snapshot(original_snapshot)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 3

    # Write only after all anchors have validated, so a mismatch cannot leave a half-patched tree.
    if patched_validation != original_validation:
        backup_and_write(VALIDATION, patched_validation)
        print(f"PATCHED: {VALIDATION.relative_to(ROOT)}")
    else:
        print(f"UNCHANGED: {VALIDATION.relative_to(ROOT)}")
    if patched_snapshot != original_snapshot:
        backup_and_write(SNAPSHOT, patched_snapshot)
        print(f"PATCHED: {SNAPSHOT.relative_to(ROOT)}")
    else:
        print(f"UNCHANGED: {SNAPSHOT.relative_to(ROOT)}")
    print("Next: run tests before any Phase 5.8 OLAP validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
