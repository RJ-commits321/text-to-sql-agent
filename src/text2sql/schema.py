"""Extract a compact, prompt-ready description of a SQLite database:
CREATE TABLE statements, a few sample rows per table, and — for text columns
with few distinct values — the exact stored values, so the model copies real
casing ('usa', not 'USA') instead of guessing."""

import sqlite3

_MAX_DISTINCT_VALUES = 12
_MAX_HINT_VALUE_LEN = 30

_TABLE_NAMES_SQL = (
    "SELECT name FROM sqlite_master "
    "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
)


def _connect(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _table_names(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute(_TABLE_NAMES_SQL)]


def get_schema(db_path: str, sample_rows: int = 3, max_value_len: int = 40) -> str:
    conn = _connect(db_path)
    try:
        tables = _table_names(conn)
        parts = []
        for table in tables:
            create = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()[0]
            parts.append(create.strip() + ";")

            cur = conn.execute(f'SELECT * FROM "{table}" LIMIT ?', (sample_rows,))
            columns = [d[0] for d in cur.description]
            rows = cur.fetchall()
            if rows:
                lines = [" | ".join(columns)]
                lines += [" | ".join(_fmt(v, max_value_len) for v in row) for row in rows]
                parts.append("/* sample rows:\n" + "\n".join(lines) + "\n*/")

            hints = _value_hints(conn, table)
            if hints:
                parts.append("/* exact stored values:\n" + "\n".join(hints) + "\n*/")

        joins = _foreign_keys(conn, tables)
        if joins:
            parts.append(
                "/* foreign keys (use these to JOIN):\n" + "\n".join(joins) + "\n*/"
            )
        return "\n\n".join(parts)
    finally:
        conn.close()


def _foreign_keys(conn: sqlite3.Connection, tables: list[str]) -> list[str]:
    """Explicit join hints from each table's foreign keys: 'child.col -> parent.col'."""
    joins = []
    for table in tables:
        try:
            fks = conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
        except sqlite3.Error:
            continue
        for fk in fks:
            # PRAGMA columns: id, seq, table, from, to, on_update, on_delete, match
            parent, from_col, to_col = fk[2], fk[3], fk[4]
            joins.append(f"{table}.{from_col} -> {parent}.{to_col}")
    return joins


def _value_hints(conn: sqlite3.Connection, table: str) -> list[str]:
    """For each low-cardinality text column, list every stored value verbatim."""
    hints = []
    for _, name, col_type, *_ in conn.execute(f'PRAGMA table_info("{table}")'):
        if not ("CHAR" in (col_type or "").upper() or "TEXT" in (col_type or "").upper()):
            continue
        try:
            values = [
                r[0]
                for r in conn.execute(
                    f'SELECT DISTINCT "{name}" FROM "{table}" '
                    f'WHERE "{name}" IS NOT NULL LIMIT {_MAX_DISTINCT_VALUES + 1}'
                )
            ]
        except sqlite3.Error:
            continue
        if not values or len(values) > _MAX_DISTINCT_VALUES:
            continue
        if any(len(str(v)) > _MAX_HINT_VALUE_LEN for v in values):
            continue
        rendered = ", ".join(f"'{v}'" for v in values)
        hints.append(f"{table}.{name}: {rendered}")
    return hints


def table_overview(db_path: str) -> list[dict]:
    """Human-friendly summary for the UI: one row per table with size and columns."""
    conn = _connect(db_path)
    try:
        overview = []
        for table in _table_names(conn):
            columns = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
            count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            overview.append({"Table": table, "Rows": count, "Columns": ", ".join(columns)})
        return overview
    finally:
        conn.close()


def column_catalog(db_path: str) -> str:
    """One line per table: 'table(col1, col2, ...)' — compact enough for retry feedback."""
    conn = _connect(db_path)
    try:
        lines = []
        for table in _table_names(conn):
            cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
            lines.append(f"{table}({', '.join(cols)})")
        return "\n".join(lines)
    finally:
        conn.close()


def list_tables(db_path: str) -> list[str]:
    conn = _connect(db_path)
    try:
        return _table_names(conn)
    finally:
        conn.close()


def _fmt(value, max_len: int) -> str:
    s = "NULL" if value is None else str(value)
    return s[:max_len] + "…" if len(s) > max_len else s
