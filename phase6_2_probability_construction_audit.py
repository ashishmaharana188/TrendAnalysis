from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analysis.phase5_snapshot import load_phase5_snapshot
from analysis.phase6_2_probability_construction import (
    audit_probability_construction,
    format_probability_construction_report,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Trace stored Phase 5.8 combined and Method A/B probability vectors, zero probabilities, "
            "method output consistency and available source-count diagnostics. Read-only; no OLAP."
        )
    )
    parser.add_argument("snapshot", help="Frozen Phase 5.8 artifact directory")
    parser.add_argument("--top-cases", type=int, default=10)
    parser.add_argument("--json-out", help="Optional path for machine-readable audit JSON")
    args = parser.parse_args()
    if args.top_cases < 0:
        parser.error("--top-cases must be >= 0")

    snapshot = load_phase5_snapshot(args.snapshot)
    rows = list(snapshot.prediction_folds)
    report = audit_probability_construction(rows, top_cases=args.top_cases)
    print(format_probability_construction_report(report, snapshot=args.snapshot, top_cases=args.top_cases))
    print(f"Snapshot status: {snapshot.manifest.get('status', 'MISSING')}")
    print(f"Snapshot schema version: {snapshot.manifest.get('schema_version', 'MISSING')}")
    print(f"Recording contract validated: {bool((snapshot.manifest.get('fold_recording_contract') or {}).get('validated', False))}")

    if args.json_out:
        destination = Path(args.json_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8")
        print(f"JSON report written: {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
