from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from .outcome_labels import OUTCOME_CLASSES


SNAPSHOT_SCHEMA_VERSION = "2"
SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS = frozenset({"1", SNAPSHOT_SCHEMA_VERSION})
PHASE5_BASELINE_VERSION = "5.8"
REQUIRED_PHASE5_SOURCES = (
    "analysis/relationship.py",
    "analysis/relationship_graph.py",
    "analysis/hardening_4_10.py",
    "analysis/walk_forward_relationship.py",
    "analysis/real_olap_validation.py",
    "analysis/real_prediction_validation.py",
    "analysis/prediction.py",
    "analysis/method_a_prediction.py",
    "analysis/method_b_prediction.py",
    "analysis/method_combination.py",
    "analysis/decision.py",
    "analysis/outcome_labels.py",
    "analysis/ranking.py",
    "data_access/metadata.py",
    "data_access/market.py",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "as_dict"):
        return value.as_dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            default=_json_default,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl_gzip(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    default=_json_default,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _row_from_observation(observation: Any) -> dict[str, Any]:
    return {
        "as_of_date": observation.as_of_date.isoformat(),
        "target": observation.target,
        "scope": observation.scope,
        "states": dict(observation.states),
        "stock_return_pct": observation.stock_return_pct,
        "benchmark_return_pct": observation.benchmark_return_pct,
        "relative_return_pct": observation.relative_return_pct,
        "outcome_end_date": (
            observation.outcome_end_date.isoformat()
            if observation.outcome_end_date is not None
            else None
        ),
    }


def _rows_from_prediction_folds(folds: Iterable[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        if hasattr(fold, "as_dict"):
            rows.append(fold.as_dict())
        else:
            rows.append(dict(fold))
    return rows


def _normalise_market_table(table: pa.Table, instrument: str) -> pa.Table:
    """Normalize market history to one stable schema, including empty tables."""

    names = set(table.column_names)
    required = ("report_date", "close")
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(
            f"Market history for {instrument!r} is missing columns: {missing}"
        )

    row_count = table.num_rows
    if row_count == 0:
        # Explicit Arrow types are required. pa.array([]) otherwise becomes
        # null-typed and cannot be concatenated with a string column.
        return pa.table(
            {
                "instrument": pa.array([], type=pa.string()),
                "date": pa.array([], type=pa.date32()),
                "open": pa.array([], type=pa.float64()),
                "high": pa.array([], type=pa.float64()),
                "low": pa.array([], type=pa.float64()),
                "close": pa.array([], type=pa.float64()),
                "volume": pa.array([], type=pa.float64()),
            }
        )

    dates = table["report_date"]
    if pa.types.is_timestamp(dates.type):
        dates = pa.compute.cast(dates, pa.date32())
    elif not pa.types.is_date(dates.type):
        dates = pa.compute.cast(dates, pa.date32())

    def numeric_column(name: str) -> pa.Array:
        if name in names:
            return pa.compute.cast(table[name], pa.float64())
        return pa.nulls(row_count, type=pa.float64())

    return pa.table(
        {
            "instrument": pa.array([str(instrument)] * row_count, type=pa.string()),
            "date": dates,
            "open": numeric_column("open"),
            "high": numeric_column("high"),
            "low": numeric_column("low"),
            "close": numeric_column("close"),
            "volume": numeric_column("volume"),
        }
    )


def _load_benchmark_snapshot_history(
    benchmark: str,
    *,
    start_date: date,
    end_date: date,
) -> pa.Table:
    """Resolve the benchmark through the same source used by Phase 5.8."""
    try:
        from data_access.macro_global import get_macro_date_range, get_macro_history

        if get_macro_date_range(benchmark) is not None:
            return get_macro_history(
                benchmark,
                start_date=start_date,
                end_date=end_date,
            )
    except ImportError:
        pass

    from data_access.market import get_daily_history

    return get_daily_history(
        benchmark,
        start_date=start_date,
        end_date=end_date,
    )


@dataclass(frozen=True)
class Phase5Snapshot:
    root: Path
    manifest: dict[str, Any]
    prediction_folds: tuple[dict[str, Any], ...]
    panel_observations: tuple[dict[str, Any], ...]
    market_daily: pa.Table

    @property
    def prediction_count(self) -> int:
        return len(self.prediction_folds)

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "manifest": dict(self.manifest),
            "prediction_count": self.prediction_count,
            "panel_observation_count": len(self.panel_observations),
            "market_daily_rows": self.market_daily.num_rows,
        }


def export_phase5_snapshot(
    result: Any,
    config: Any,
    destination: str | Path,
    *,
    repository_root: str | Path | None = None,
    benchmark_market_loader: Any | None = None,
    ticker_market_loader: Any | None = None,
) -> Path:
    """Write an immutable Phase 5.8 development snapshot.

    The destination must not already exist. The exporter uses a temporary
    directory and atomically renames it into place only after every artifact
    and its content hash have been produced.

    This function does not alter Phase 5 calculations.
    """

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(
            f"Snapshot already exists: {destination}. "
            "Frozen Phase 5 snapshots are immutable; create a new version instead."
        )

    repo = Path(repository_root) if repository_root else Path.cwd()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if ticker_market_loader is None:
        from data_access.market import get_daily_history

        ticker_market_loader = get_daily_history

    temp_parent = destination.parent
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=temp_parent,
        )
    )

    try:
        folds_path = temp_dir / "prediction_folds.jsonl.gz"
        panel_path = temp_dir / "panel_observations.jsonl.gz"
        summary_path = temp_dir / "phase5_result_summary.json"
        config_path = temp_dir / "phase5_config.json"
        market_path = temp_dir / "market_daily.parquet"

        _write_jsonl_gzip(
            folds_path,
            _rows_from_prediction_folds(result.prediction_folds),
        )
        _write_jsonl_gzip(
            panel_path,
            (
                _row_from_observation(observation)
                for observation in result.panel_observations
            ),
        )

        _write_json(summary_path, result.as_dict())
        _write_json(
            config_path,
            {
                key: value
                for key, value in vars(config).items()
                if not key.startswith("_")
            },
        )

        ticker_table = ticker_market_loader(config.ticker)
        ticker_market = _normalise_market_table(ticker_table, config.ticker)

        cutoff = result.latest_market_date
        if ticker_market.num_rows:
            start_date = ticker_market["date"][0].as_py()
        else:
            start_date = (
                getattr(result, "first_prediction_date", None)
                or cutoff
            )

        if benchmark_market_loader is not None:
            benchmark_table = benchmark_market_loader(config.benchmark)
        else:
            if start_date is None or cutoff is None:
                raise ValueError(
                    "Cannot resolve benchmark history without a market date range."
                )
            benchmark_table = _load_benchmark_snapshot_history(
                config.benchmark,
                start_date=start_date,
                end_date=cutoff,
            )

        benchmark_market = _normalise_market_table(
            benchmark_table,
            config.benchmark,
        )
        if cutoff is not None:
            cutoff_date = cutoff if isinstance(cutoff, date) else date.fromisoformat(str(cutoff))
            ticker_market = ticker_market.filter(
                pa.compute.less_equal(
                    ticker_market["date"],
                    pa.scalar(cutoff_date, type=pa.date32()),
                )
            )
            benchmark_market = benchmark_market.filter(
                pa.compute.less_equal(
                    benchmark_market["date"],
                    pa.scalar(cutoff_date, type=pa.date32()),
                )
            )

        market = pa.concat_tables([ticker_market, benchmark_market])
        market = market.sort_by(
            [("instrument", "ascending"), ("date", "ascending")]
        )
        pq.write_table(market, market_path, compression="zstd")

        source_hashes: dict[str, str] = {}
        source_root = repo
        for relative in REQUIRED_PHASE5_SOURCES:
            source_path = source_root / relative
            if source_path.exists():
                source_hashes[relative] = _sha256_file(source_path)
            else:
                source_hashes[relative] = "MISSING"

        artifact_files = [
            folds_path,
            panel_path,
            summary_path,
            config_path,
            market_path,
        ]
        file_hashes = {
            path.name: _sha256_file(path)
            for path in artifact_files
        }

        manifest = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "phase5_baseline_version": PHASE5_BASELINE_VERSION,
            "status": "FROZEN",
            "created_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "ticker": config.ticker,
            "benchmark": config.benchmark,
            "analysis_timeframe": config.analysis_timeframe,
            "holding_period_months": float(config.holding_period_months),
            "entry_mode": config.entry_mode,
            "candidate_predictions": int(result.candidate_predictions),
            "evaluated_predictions": int(result.evaluated_predictions),
            "selection_candidate_evaluations": int(result.selection_candidate_evaluations),
            "selection_validated_predictions": int(result.selection_validated_predictions),
            "multiple_testing_controlled_folds": int(result.multiple_testing_controlled_folds),
            "latest_validated_prediction_date": result.latest_validated_prediction_date,
            "latest_market_date": result.latest_market_date,
            "prediction_fold_count": len(result.prediction_folds),
            "panel_observation_count": len(result.panel_observations),
            "outcome_classes": list(OUTCOME_CLASSES),
            "source_sha256": source_hashes,
            "artifact_sha256": file_hashes,
            "immutability": {
                "overwrite_allowed": False,
                "purpose": "PHASE5_DEVELOPMENT_BASELINE",
            },
        }
        _write_json(temp_dir / "manifest.json", manifest)

        # Include manifest itself in a final directory-level hash file.
        all_files = sorted(
            path for path in temp_dir.rglob("*")
            if path.is_file()
        )
        directory_hash = hashlib.sha256()
        for path in all_files:
            if path.name == "snapshot.sha256":
                continue
            relative = path.relative_to(temp_dir).as_posix()
            directory_hash.update(relative.encode("utf-8"))
            directory_hash.update(b"\0")
            directory_hash.update(_sha256_file(path).encode("ascii"))
            directory_hash.update(b"\0")

        (temp_dir / "snapshot.sha256").write_text(
            directory_hash.hexdigest() + "\n",
            encoding="utf-8",
        )

        temp_dir.replace(destination)
        return destination

    except Exception:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def load_phase5_snapshot(
    root: str | Path,
    *,
    expected_ticker: str | None = None,
    expected_benchmark: str | None = None,
    expected_analysis_timeframe: str | None = None,
    expected_holding_period_months: float | None = None,
) -> Phase5Snapshot:
    """Load and validate an immutable Phase 5.8 snapshot."""

    root = Path(root)
    manifest_path = root / "manifest.json"
    folds_path = root / "prediction_folds.jsonl.gz"
    panel_path = root / "panel_observations.jsonl.gz"
    market_path = root / "market_daily.parquet"

    for path in (manifest_path, folds_path, panel_path, market_path):
        if not path.exists():
            raise FileNotFoundError(f"Required snapshot artifact missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("schema_version") not in SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported snapshot schema: {manifest.get('schema_version')!r}"
        )
    if manifest.get("phase5_baseline_version") != PHASE5_BASELINE_VERSION:
        raise ValueError("Snapshot is not a Phase 5.8 baseline.")
    if manifest.get("status") != "FROZEN":
        raise ValueError("Snapshot is not marked FROZEN.")
    if manifest.get("immutability", {}).get("overwrite_allowed", True):
        raise ValueError("Snapshot immutability contract is invalid.")

    if expected_ticker is not None and manifest.get("ticker") != expected_ticker:
        raise ValueError("Snapshot ticker does not match requested ticker.")
    if expected_benchmark is not None and manifest.get("benchmark") != expected_benchmark:
        raise ValueError("Snapshot benchmark does not match requested benchmark.")
    if (
        expected_analysis_timeframe is not None
        and manifest.get("analysis_timeframe") != expected_analysis_timeframe
    ):
        raise ValueError("Snapshot analysis timeframe does not match.")
    if (
        expected_holding_period_months is not None
        and abs(
            float(manifest.get("holding_period_months", 0.0))
            - float(expected_holding_period_months)
        ) > 1e-12
    ):
        raise ValueError("Snapshot holding period does not match.")

    for filename, expected_hash in manifest.get("artifact_sha256", {}).items():
        actual_hash = _sha256_file(root / filename)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Snapshot artifact hash mismatch: {filename}"
            )

    folds = _read_jsonl_gzip(folds_path)
    panel = _read_jsonl_gzip(panel_path)
    market = pq.read_table(market_path)

    expected_folds = int(manifest.get("prediction_fold_count", -1))
    expected_panel = int(manifest.get("panel_observation_count", -1))
    if len(folds) != expected_folds:
        raise ValueError("Prediction-fold count does not match manifest.")
    if len(panel) != expected_panel:
        raise ValueError("Panel-observation count does not match manifest.")

    if market.num_rows == 0:
        raise ValueError("Snapshot contains no market_daily rows.")

    return Phase5Snapshot(
        root=root,
        manifest=manifest,
        prediction_folds=tuple(folds),
        panel_observations=tuple(panel),
        market_daily=market,
    )


def default_snapshot_path(
    config: Any,
    *,
    root: str | Path = "artifacts/phase5_8",
    version: str = "v2",
) -> Path:
    safe_ticker = str(config.ticker).replace("/", "_")
    safe_benchmark = str(config.benchmark).replace("/", "_")
    name = (
        f"{safe_ticker}_{safe_benchmark}_"
        f"{config.analysis_timeframe}_{config.holding_period_months:g}M_{version}"
    )
    return Path(root) / name
