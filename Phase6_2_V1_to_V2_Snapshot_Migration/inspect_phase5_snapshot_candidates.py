from __future__ import annotations

"""Inventory frozen Phase 5.8 snapshots to locate Method A/B sidecars.

This is read-only. It does not change snapshots, run OLAP, or infer method
probabilities from trend labels or the combined probability vector.
"""

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

LABELS = ("UP", "SIDEWAYS", "DOWN")
VECTOR_FIELDS = ("method_a_probabilities_pct", "method_b_probabilities_pct")
METADATA_MARKERS = (
    "method_a_evidence_score",
    "method_b_evidence_score",
    "method_a_effective_sample_size",
    "method_b_effective_sample_size",
    "method_a_sample_count",
    "method_b_sample_count",
    "method_a_variables",
    "method_b_variables",
    "method_a_condition",
    "method_b_condition",
)


def is_vector(value: Any) -> bool:
    if not isinstance(value, dict) or any(k not in value for k in LABELS):
        return False
    try:
        probs = [float(value[k]) for k in LABELS]
    except (TypeError, ValueError):
        return False
    return all(v >= 0 and v < float("inf") for v in probs) and abs(sum(probs) - 100.0) <= 0.25


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only inventory of Phase 5.8 snapshots, including candidates for Method A/B sidecars."
    )
    parser.add_argument("root", nargs="?", default="artifacts", help="Directory to search (default: artifacts)")
    args = parser.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        parser.error(f"Directory does not exist: {root}")

    manifests = sorted(root.rglob("manifest.json"))
    found = 0
    print(f"Snapshot inventory root: {root.resolve()}")
    print(f"Manifest files found: {len(manifests)}")
    print("Counts: folds | combined vectors | Method A vectors | Method B vectors | folds with method-selection metadata")

    for manifest_path in manifests:
        snap = manifest_path.parent
        if any(part.endswith(".tmp") or part.startswith(".") and part.endswith("tmp") for part in snap.parts):
            continue
        folds_path = snap / "prediction_folds.jsonl.gz"
        if not folds_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rows = load_rows(folds_path)
        except (OSError, json.JSONDecodeError, EOFError, gzip.BadGzipFile) as exc:
            print(f"\nINVALID {snap}: {exc}")
            continue

        a_count = sum(is_vector(row.get(VECTOR_FIELDS[0])) for row in rows)
        b_count = sum(is_vector(row.get(VECTOR_FIELDS[1])) for row in rows)
        combined_count = sum(is_vector(row.get("probabilities_pct")) for row in rows)
        metadata_count = sum(any(row.get(key) not in (None, "", [], {}) for key in METADATA_MARKERS) for row in rows)
        name = str(snap).lower()
        a_name = any(token in name for token in ("method_a", "method-a", "methoda", "a_only", "a-only"))
        b_name = any(token in name for token in ("method_b", "method-b", "methodb", "b_only", "b-only"))
        if a_name and b_name:
            role = "PATH_NAME_AMBIGUOUS_A_AND_B"
        elif a_name:
            role = "POSSIBLE_METHOD_A_SIDECAR"
        elif b_name:
            role = "POSSIBLE_METHOD_B_SIDECAR"
        elif a_count and b_count:
            role = "HAS_EXPLICIT_A_AND_B_VECTORS"
        elif a_count or b_count:
            role = "HAS_ONE_EXPLICIT_METHOD_VECTOR"
        else:
            role = "CHECK_MANIFEST_AND_PATH_FOR_METHOD_ROLE"

        found += 1
        print(f"\n[{role}] {snap}")
        print(
            f"  schema={manifest.get('schema_version')} status={manifest.get('status')} "
            f"folds={len(rows)} combined={combined_count} A={a_count} B={b_count} selection_metadata={metadata_count}"
        )
        print(
            f"  ticker={manifest.get('ticker')} benchmark={manifest.get('benchmark')} "
            f"timeframe={manifest.get('analysis_timeframe')} holding={manifest.get('holding_period_months')}"
        )
        migration = manifest.get("migration")
        if isinstance(migration, dict):
            print(
                "  migration="
                + json.dumps(
                    {k: migration.get(k) for k in ("method_probability_source", "method_a_vectors_available", "method_b_vectors_available", "prediction_recalculated")},
                    ensure_ascii=False,
                )
            )
        if rows:
            keys = sorted({key for row in rows[:3] for key in row})
            print(f"  sample_fold_keys={keys}")

    print(f"\nUsable snapshots inventoried: {found}")
    print("Important: probabilities_pct is usable as a sidecar only when that snapshot was explicitly generated for that individual method.")
    print("Do not designate a combined-probability snapshot as Method A or Method B merely because its field is named probabilities_pct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
