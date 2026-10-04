# tests/test_data_access.py

from data_access.db import get_db_path, get_connection


def main() -> None:
    print(f"Database: {get_db_path()}")

    con = get_connection()

    try:
        result = con.execute(
            """
            SELECT
                current_database() AS database_name,
                version() AS duckdb_version
            """
        ).fetchone()

        print(f"Database name: {result[0]}")
        print(f"DuckDB version: {result[1]}")

    finally:
        con.close()


if __name__ == "__main__":
    main()