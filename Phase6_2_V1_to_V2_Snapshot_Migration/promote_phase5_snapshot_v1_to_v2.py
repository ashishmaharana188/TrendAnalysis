from __future__ import annotations

"""Create a schema-v2 Phase 5.8 snapshot from an existing frozen snapshot.

This does NOT rerun Phase 5.8 or modify the source snapshot.

If the source folds already include method_a_probabilities_pct and
method_b_probabilities_pct, those exact stored vectors are retained.
If vectors are absent, pass method-only Phase 5.8 snapshot directories with
--method-a-snapshot and --method-b-snapshot. In those inputs, each fold's
probabilities_pct must genuinely represent that individual method.
"""

import argparse
import gzip
import hashlib
import json
import math
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LABELS = ("UP", "SIDEWAYS", "DOWN")
REQUIRED_FILES = (
    "manifest.json",
    "prediction_folds.jsonl.gz",
    "panel_observations.jsonl.gz",
    "market_daily.parquet",
)
VECTOR_FIELDS = ("method_a_probabilities_pct", "method_b_probabilities_pct")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        if path.name == "snapshot.sha256":
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON manifest: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object in {path} line {line_number}")
                rows.append(row)
    except (OSError, EOFError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read compressed JSONL: {path}: {exc}") from exc
    return rows


def write_jsonl_gzip(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def check_distribution(raw: Any, *, label: str) -> dict[str, float] | None:
    """Validate a stored percentage distribution without renormalizing it."""
    if raw is None or raw == {}:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{label}: expected a probability object or {{}} for unavailable method")
    missing = [name for name in LABELS if name not in raw]
    if missing:
        raise ValueError(f"{label}: probability vector missing classes: {missing}")
    try:
        values = {name: float(raw[name]) for name in LABELS}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: probability values must be numeric") from exc
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError(f"{label}: probability values must be finite and non-negative")
    if abs(sum(values.values()) - 100.0) > 0.25:
        raise ValueError(
            f"{label}: percentage values sum to {sum(values.values()):.8f}, not 100; "
            "refusing to silently renormalize stored probabilities"
        )
    return values


def validate_snapshot(root: Path, *, verify_directory_hash: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Snapshot directory not found: {root}")
    for name in REQUIRED_FILES:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Required snapshot artifact missing: {root / name}")

    manifest = read_json(root / "manifest.json")
    if str(manifest.get("schema_version")) not in {"1", "2"}:
        raise ValueError(f"Unsupported source schema version: {manifest.get('schema_version')!r}")
    if manifest.get("status") != "FROZEN":
        raise ValueError(f"Snapshot is not marked FROZEN: {root}")
    if manifest.get("immutability", {}).get("overwrite_allowed", True):
        raise ValueError(f"Snapshot immutability metadata is invalid: {root}")

    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"Manifest has no artifact_sha256 map: {root}")
    for filename, expected in hashes.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Manifest artifact is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Artifact checksum mismatch in {root.name}: {filename}")

    marker = root / "snapshot.sha256"
    if verify_directory_hash and marker.exists():
        expected_digest = marker.read_text(encoding="ascii").strip()
        actual_digest = directory_digest(root)
        if expected_digest != actual_digest:
            raise ValueError(f"Snapshot directory checksum mismatch: {root}")

    folds = read_jsonl_gzip(root / "prediction_folds.jsonl.gz")
    expected_count = int(manifest.get("prediction_fold_count", -1))
    if expected_count != len(folds):
        raise ValueError(
            f"Fold count mismatch in {root}: manifest={expected_count}, file={len(folds)}"
        )
    return manifest, folds


def row_date(row: dict[str, Any], *, source_name: str) -> str:
    value = row.get("prediction_date")
    if not value:
        raise ValueError(f"A fold in {source_name} has no prediction_date")
    return str(value)[:10]


def sidecar_index(snapshot_root: Path, *, method_name: str) -> dict[str, dict[str, Any]]:
    """Load an individual-method snapshot whose probabilities_pct is method-specific."""
    _manifest, folds = validate_snapshot(snapshot_root)
    result: dict[str, dict[str, Any]] = {}
    for row in folds:
        key = row_date(row, source_name=str(snapshot_root))
        if key in result:
            raise ValueError(f"Duplicate prediction_date {key} in {method_name} snapshot")
        raw = row.get("probabilities_pct")
        dist = check_distribution(raw, label=f"{method_name} snapshot {key} probabilities_pct")
        if dist is None:
            raise ValueError(
                f"{method_name} snapshot has no actual method probability vector for {key}; "
                "this does not look like a method-only snapshot"
            )
        result[key] = row
    return result


def resolve_vectors(
    rows: list[dict[str, Any]],
    *,
    source_root: Path,
    method_a_snapshot: Path | None,
    method_b_snapshot: Path | None,
) -> tuple[list[dict[str, Any]], str, int, int]:
    a_sidecar = sidecar_index(method_a_snapshot, method_name="Method A") if method_a_snapshot else None
    b_sidecar = sidecar_index(method_b_snapshot, method_name="Method B") if method_b_snapshot else None

    migrated = [dict(row) for row in rows]
    a_count = b_count = 0
    vector_keys_present = all(all(field in row for field in VECTOR_FIELDS) for row in migrated)
    source_is_vectorized = vector_keys_present

    for index, row in enumerate(migrated):
        key = row_date(row, source_name=str(source_root))
        for field, sidecar, method_label in (
            ("method_a_probabilities_pct", a_sidecar, "Method A"),
            ("method_b_probabilities_pct", b_sidecar, "Method B"),
        ):
            raw = row.get(field, {})
            validated = check_distribution(raw, label=f"source fold {key} {field}")
            if sidecar is not None:
                if key not in sidecar:
                    raise ValueError(
                        f"{method_label} snapshot lacks prediction_date {key}; "
                        "refusing to attach probabilities to a different fold"
                    )
                method_row = sidecar[key]
                source_actual = row.get("actual_class")
                sidecar_actual = method_row.get("actual_class")
                if source_actual is not None and sidecar_actual is not None and source_actual != sidecar_actual:
                    raise ValueError(
                        f"Actual-class mismatch for {method_label} on {key}: "
                        f"source={source_actual!r}, method snapshot={sidecar_actual!r}"
                    )
                # Sidecar input is explicitly declared to be a single-method run, so its
                # probabilities_pct are the individual method vector, not combined output.
                sidecar_dist = check_distribution(
                    method_row.get("probabilities_pct"),
                    label=f"{method_label} snapshot {key}",
                )
                if sidecar_dist is None:
                    raise ValueError(f"No usable {method_label} probability vector on {key}")
                row[field] = sidecar_dist
                validated = sidecar_dist
            elif validated is None:
                row[field] = {}
            if validated is not None:
                if field == "method_a_probabilities_pct":
                    a_count += 1
                else:
                    b_count += 1

    method_source = "existing_fold_vectors" if source_is_vectorized and not (a_sidecar or b_sidecar) else "merged_method_only_snapshots"
    return migrated, method_source, a_count, b_count


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def default_destination(source: Path) -> Path:
    name = source.name
    if name.endswith("_v1"):
        name = name[:-3] + "_v2"
    elif name.endswith("_v2"):
        raise ValueError("Source directory already ends in _v2; pass an explicit --destination for a distinct copy")
    else:
        name += "_v2"
    return source.parent / name


def promote_snapshot(
    source: Path,
    destination: Path | None = None,
    *,
    method_a_snapshot: Path | None = None,
    method_b_snapshot: Path | None = None,
) -> Path:
    source = source.resolve()
    source_manifest, source_folds = validate_snapshot(source)
    destination = (destination or default_destination(source)).resolve()
    if destination == source:
        raise ValueError("Destination must differ from source; v1 is never overwritten")
    if destination.exists():
        raise FileExistsError(f"Destination already exists; refusing to overwrite: {destination}")

    migrated_rows, vector_source, a_count, b_count = resolve_vectors(
        source_folds,
        source_root=source,
        method_a_snapshot=method_a_snapshot,
        method_b_snapshot=method_b_snapshot,
    )
    if a_count == 0 or b_count == 0:
        missing = []
        if a_count == 0:
            missing.append("Method A")
        if b_count == 0:
            missing.append("Method B")
        raise ValueError(
            f"Cannot create an auditable v2 snapshot: no valid exact probability vectors found for {', '.join(missing)}. "
            "This v1 snapshot alone does not contain the required probabilities. Provide the corresponding "
            "individual-method snapshots with --method-a-snapshot and/or --method-b-snapshot. "
            "Trend labels cannot be converted back into probability distributions. No files were written."
        )

    if method_a_snapshot is not None:
        _m, a_rows = validate_snapshot(method_a_snapshot.resolve())
        a_dates = {row_date(row, source_name=str(method_a_snapshot)) for row in a_rows}
        missing_dates = sorted({row_date(row, source_name=str(source)) for row in source_folds} - a_dates)
        if missing_dates:
            raise ValueError(f"Method A snapshot does not cover all source folds; missing: {missing_dates[:10]}")
    if method_b_snapshot is not None:
        _m, b_rows = validate_snapshot(method_b_snapshot.resolve())
        b_dates = {row_date(row, source_name=str(method_b_snapshot)) for row in b_rows}
        missing_dates = sorted({row_date(row, source_name=str(source)) for row in source_folds} - b_dates)
        if missing_dates:
            raise ValueError(f"Method B snapshot does not cover all source folds; missing: {missing_dates[:10]}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = destination.parent
    temp_root: Path | None = None
    try:
        temp_root = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=temp_parent))
        # mkdtemp made an empty directory; copy the source contents into it while preserving v1.
        for item in source.iterdir():
            target = temp_root / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

        write_jsonl_gzip(temp_root / "prediction_folds.jsonl.gz", migrated_rows)
        manifest = dict(source_manifest)
        manifest["schema_version"] = "2"
        manifest["migration"] = {
            "migrated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "source_snapshot": str(source),
            "source_schema_version": str(source_manifest.get("schema_version")),
            "method_probability_source": vector_source,
            "method_a_vectors_available": a_count,
            "method_b_vectors_available": b_count,
            "fold_count": len(migrated_rows),
            "prediction_recalculated": False,
        }
        hashes = dict(manifest.get("artifact_sha256", {}))
        hashes["prediction_folds.jsonl.gz"] = sha256_file(temp_root / "prediction_folds.jsonl.gz")
        # Refresh all existing artifact hashes in case an older manifest tracked extra artifact files.
        for filename in list(hashes):
            artifact_path = temp_root / filename
            if not artifact_path.is_file():
                raise FileNotFoundError(f"Manifest references missing artifact during migration: {artifact_path}")
            hashes[filename] = sha256_file(artifact_path)
        manifest["artifact_sha256"] = hashes
        write_manifest(temp_root / "manifest.json", manifest)
        (temp_root / "snapshot.sha256").write_text(directory_digest(temp_root) + "\n", encoding="ascii")

        # Verify the new snapshot fully before publishing it.
        check_manifest, check_rows = validate_snapshot(temp_root)
        if check_manifest.get("schema_version") != "2":
            raise ValueError("Internal validation failed: migrated snapshot is not schema v2")
        for row in check_rows:
            for field in VECTOR_FIELDS:
                if field not in row:
                    raise ValueError(f"Internal validation failed: missing {field} in a fold")
                check_distribution(row[field], label=f"output fold {row.get('prediction_date')} {field}")
        temp_root.replace(destination)
        temp_root = None
        return destination
    finally:
        if temp_root is not None and temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a schema-v2 Phase 5.8 snapshot from an existing frozen snapshot without rerunning the 250 folds."
    )
    parser.add_argument("source", help="Existing frozen snapshot directory, typically ending in _v1")
    parser.add_argument("--destination", help="New v2 directory; default replaces the _v1 suffix with _v2")
    parser.add_argument(
        "--method-a-snapshot",
        help="Optional frozen method-only snapshot whose prediction_folds probabilities_pct are Method A probabilities",
    )
    parser.add_argument(
        "--method-b-snapshot",
        help="Optional frozen method-only snapshot whose prediction_folds probabilities_pct are Method B probabilities",
    )
    args = parser.parse_args()
    try:
        destination = promote_snapshot(
            Path(args.source),
            Path(args.destination) if args.destination else None,
            method_a_snapshot=Path(args.method_a_snapshot) if args.method_a_snapshot else None,
            method_b_snapshot=Path(args.method_b_snapshot) if args.method_b_snapshot else None,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    manifest, rows = validate_snapshot(destination)
    print("PHASE 5.8 SNAPSHOT V1 -> V2 MIGRATION COMPLETE")
    print(f"Source retained unchanged: {args.source}")
    print(f"New snapshot: {destination}")
    print(f"Schema version: {manifest['schema_version']}")
    print(f"Folds: {len(rows)}")
    print(f"Method A vectors: {manifest['migration']['method_a_vectors_available']}/{len(rows)}")
    print(f"Method B vectors: {manifest['migration']['method_b_vectors_available']}/{len(rows)}")
    print("Prediction recalculated: NO")
    print("Checksums: VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
