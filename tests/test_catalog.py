# tests/test_catalog.py

from data_access.catalog import list_tables


def main() -> None:
    tables = list_tables()

    print(tables.to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()