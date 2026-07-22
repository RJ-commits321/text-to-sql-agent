import sqlite3

import pytest

from text2sql.guardrails import execute
from text2sql.schema import column_catalog, get_schema


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE artists (id INTEGER PRIMARY KEY, name TEXT, country TEXT)")
    conn.execute("CREATE TABLE empty_table (id INTEGER)")
    conn.executemany(
        "INSERT INTO artists VALUES (?, ?, ?)",
        [
            (1, "AC/DC", "Australia"),
            (2, "Accept", "Germany"),
            (3, "A very long artist name " * 5, "Germany"),
        ],
    )
    conn.commit()
    conn.close()
    return str(path)


def test_schema_contains_create_statements(db_path):
    schema = get_schema(db_path)
    assert "CREATE TABLE artists" in schema
    assert "CREATE TABLE empty_table" in schema


def test_schema_contains_sample_rows(db_path):
    schema = get_schema(db_path)
    assert "AC/DC" in schema


def test_long_values_truncated(db_path):
    schema = get_schema(db_path)
    assert "A very long artist name " * 5 not in schema


def test_value_hints_for_low_cardinality_text(db_path):
    schema = get_schema(db_path)
    assert "artists.country: 'Australia', 'Germany'" in schema
    # 'name' contains a value longer than the hint cap -> whole column skipped
    assert "artists.name:" not in schema


def test_column_catalog(db_path):
    catalog = column_catalog(db_path)
    assert "artists(id, name, country)" in catalog
    assert "empty_table(id)" in catalog


def test_execute_is_read_only(db_path):
    with pytest.raises(sqlite3.OperationalError):
        execute(db_path, "INSERT INTO artists VALUES (99, 'nope')")


def test_execute_returns_rows(db_path):
    columns, rows = execute(db_path, "SELECT name FROM artists ORDER BY id LIMIT 2")
    assert columns == ["name"]
    assert rows == [("AC/DC",), ("Accept",)]
