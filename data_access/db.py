# data_access/db.py

from __future__ import annotations

import os
from pathlib import Path

import duckdb


DEFAULT_DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "TrendAnalysis"
    / "market_data.duckdb"
)


def get_db_path() -> Path:
    path = Path(
        os.getenv(
            "OLAP_DB_PATH",
            str(DEFAULT_DB_PATH),
        )
    ).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"OLAP database not found: {path}"
        )

    if path.suffix.lower() != ".duckdb":
        raise ValueError(
            f"Expected a DuckDB database file, got: {path}"
        )

    return path


def get_connection() -> duckdb.DuckDBPyConnection:
    """
    Open a read-only connection to the ETL OLAP database.
    TrendAnalysis must never write to the upstream OLAP.
    """
    return duckdb.connect(
        database=str(get_db_path()),
        read_only=True,
    )