"""Safety layers between model output and the database:

1. read-only SQLite connection (mode=ro) — writes are physically impossible
2. AST validation via sqlglot — single statement, SELECT-family only
3. LIMIT injection — a runaway cross-join can't flood the UI
4. execution timeout — long-running queries are interrupted
"""

import re
import sqlite3
import time

import sqlglot
from sqlglot import exp


class GuardrailError(Exception):
    pass


_ALLOWED_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)


def extract_sql(text: str) -> str:
    """Pull SQL out of the model reply (handles markdown fences despite the prompt rules)."""
    match = re.search(r"```(?:sql)?\s*(.+?)```", text, re.S | re.I)
    sql = (match.group(1) if match else text).strip()
    return sql.rstrip(";").strip()


def validate(sql: str) -> exp.Expression:
    """Parse and reject anything that is not exactly one SELECT-family statement."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except sqlglot.errors.SqlglotError as e:
        # SqlglotError covers ParseError AND TokenError (e.g. unbalanced quotes,
        # which small models produce; found the hard way when it crashed an eval run)
        raise GuardrailError(f"SQL failed to parse: {e}") from e
    if len(statements) != 1:
        raise GuardrailError("Exactly one SQL statement is allowed.")
    tree = statements[0]
    if not isinstance(tree, _ALLOWED_ROOTS):
        raise GuardrailError(f"Only SELECT queries are allowed, got: {tree.key.upper()}")
    return tree


def enforce_limit(tree: exp.Expression, row_limit: int) -> str:
    if tree.args.get("limit") is None:
        tree = tree.limit(row_limit)
    return tree.sql(dialect="sqlite")


def execute(
    db_path: str, sql: str, timeout_s: float = 10.0, max_rows: int = 500
) -> tuple[list[str], list[tuple]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: time.monotonic() > deadline, 100_000)
    try:
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(max_rows)
        return columns, rows
    finally:
        conn.close()
