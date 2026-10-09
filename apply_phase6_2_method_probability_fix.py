from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path


def require_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Preflight failed for {label}: expected 1 match, found {count}. No files were changed.")
    return text.replace(old, new, 1)


def main() -> None:
    root = Path.cwd()
    required = [
        root / "analysis" / "real_prediction_validation.py",
        root / "analysis" / "phase5_snapshot.py",
        root / "tests" / "test_phase5_snapshot.py",
        root / "tests" / "test_phase5_8_snapshot_export_preflight.py",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            "Run this script from the TrendAnalysis repository root. Missing: "
            + ", ".join(missing)
        )

    validation_path, snapshot_path, snapshot_test_path, snapshot_preflight_test_path = required
    validation = validation_path.read_text(encoding="utf-8")
    snapshot = snapshot_path.read_text(encoding="utf-8")
    snapshot_test = snapshot_test_path.read_text(encoding="utf-8")
    snapshot_preflight_test = snapshot_preflight_test_path.read_text(encoding="utf-8")

    # Prepare all edits in memory first; do not write a partial patch.
    validation = require_once(
        validation,
        "from dataclasses import dataclass\n",
        "from dataclasses import dataclass, field\n",
        "dataclass field import",
    )
    validation = require_once(
        validation,
        "    trade_eligible: bool\n"
        "    trade_reason: str\n\n"
        "    def as_dict(self) -> dict[str, Any]:",
        "    trade_eligible: bool\n"
        "    trade_reason: str\n"
        "    method_a_probabilities_pct: dict[OutcomeClass, float] = field(default_factory=dict)\n"
        "    method_b_probabilities_pct: dict[OutcomeClass, float] = field(default_factory=dict)\n\n"
        "    def as_dict(self) -> dict[str, Any]:",
        "fold dataclass fields",
    )
    validation = require_once(
        validation,
        '            "method_b_trend": self.method_b_trend,\n'
        '            "method_agreement": self.method_agreement,',
        '            "method_b_trend": self.method_b_trend,\n'
        '            "method_a_probabilities_pct": dict(self.method_a_probabilities_pct),\n'
        '            "method_b_probabilities_pct": dict(self.method_b_probabilities_pct),\n'
        '            "method_agreement": self.method_agreement,',
        "fold serialization fields",
    )

    constructor_pattern = re.compile(
        r"(?m)^(?P<indent>[ ]*)"
        r"method_b_trend=result\.method_b\.trend if result\.method_b else None,\n"
        r"(?P=indent)method_agreement="
    )
    constructor_count = len(constructor_pattern.findall(validation))
    if constructor_count != 2:
        raise RuntimeError(
            f"Preflight failed for method vector capture: expected 2 fold constructors, "
            f"found {constructor_count}. No files were changed."
        )

    def add_method_vectors(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}method_b_trend=result.method_b.trend if result.method_b else None,\n"
            f"{indent}method_a_probabilities_pct=(\n"
            f"{indent}    dict(result.method_a.probabilities_pct) if result.method_a else {{}}\n"
            f"{indent}),\n"
            f"{indent}method_b_probabilities_pct=(\n"
            f"{indent}    dict(result.method_b.probabilities_pct) if result.method_b else {{}}\n"
            f"{indent}),\n"
            f"{indent}method_agreement="
        )

    validation = constructor_pattern.sub(add_method_vectors, validation)

    snapshot = require_once(
        snapshot,
        'SNAPSHOT_SCHEMA_VERSION = "1"\n',
        'SNAPSHOT_SCHEMA_VERSION = "2"\n'
        'SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS = frozenset({"1", SNAPSHOT_SCHEMA_VERSION})\n',
        "snapshot schema version",
    )
    snapshot = require_once(
        snapshot,
        'if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:',
        'if manifest.get("schema_version") not in SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS:',
        "backward-compatible schema validation",
    )
    snapshot = require_once(
        snapshot,
        '    version: str = "v1",',
        '    version: str = "v2",',
        "default snapshot version",
    )
    snapshot_test = require_once(
        snapshot_test,
        'assert path.name == "RELIANCE_Nifty_50_6M_1M_v1"',
        'assert path.name == "RELIANCE_Nifty_50_6M_1M_v2"',
        "snapshot test default version",
    )

    snapshot_preflight_test = require_once(
        snapshot_preflight_test,
        '"RELIANCE_Nifty_50_6M_1M_v1"\n        )',
        '"RELIANCE_Nifty_50_6M_1M_v2"\n        )',
        "snapshot export preflight default version",
    )

    # Syntax-check the edited Python before writing anything.
    compile(validation, str(validation_path), "exec")
    compile(snapshot, str(snapshot_path), "exec")
    compile(snapshot_test, str(snapshot_test_path), "exec")
    compile(snapshot_preflight_test, str(snapshot_preflight_test_path), "exec")

    backup_dir = root / f".phase6_2_fix_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir()
    for path in required:
        shutil.copy2(path, backup_dir / path.relative_to(root))

    validation_path.write_text(validation, encoding="utf-8")
    snapshot_path.write_text(snapshot, encoding="utf-8")
    snapshot_test_path.write_text(snapshot_test, encoding="utf-8")
    snapshot_preflight_test_path.write_text(snapshot_preflight_test, encoding="utf-8")

    print("Applied Phase 6.2 method-probability serialization fix.")
    print(f"Backup files: {backup_dir}")
    print("New snapshots default to v2; schema-v1 snapshots remain loadable.")
    print("Method A/B probability vectors are captured from the actual PredictionResult; unavailable methods serialize as {}.")
    print()
    print("Run:")
    print("  py -m pytest -q tests/test_phase6_2_method_probability_audit.py tests/test_phase5_snapshot.py tests/test_real_prediction_validation.py")


if __name__ == "__main__":
    main()
