"""DuckDB connection manager and guarded read-only SQL executor.

Two layers of protection sit between an LLM-written query and the warehouse:

1. The connection itself is opened ``read_only`` with external file/network
   access disabled and the configuration locked, so even a query that slipped
   past validation cannot write, attach other databases or read local files.
2. ``validate_sql`` parses the text with DuckDB's own parser and only accepts a
   single SELECT statement (CTEs included) that touches known tables.

Results are capped at ``MAX_ROWS`` and the query runs under a timeout so a
runaway cross join cannot freeze the app.
"""
from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import duckdb
import pandas as pd

from src import DATA_DIR

DEFAULT_DB_PATH = DATA_DIR / "cpg_warehouse.duckdb"
MAX_ROWS = 500
QUERY_TIMEOUT_S = 15

# Belt-and-braces deny list on top of the parser check; matched on word
# boundaries against the query with string literals removed.
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|merge|drop|create|alter|attach|detach|copy|export|import|install|load|"
    r"pragma|set|reset|call|checkpoint|vacuum|grant|revoke|truncate|replace\s+into)\b",
    re.IGNORECASE,
)
_FILE_FUNCS = re.compile(
    r"\b(read_csv\w*|read_parquet|read_json\w*|read_text|read_blob|parquet_scan|csv_scan|glob|"
    r"sniff_csv|query_table|query)\s*\(",
    re.IGNORECASE,
)
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")


class UnsafeQueryError(ValueError):
    """Raised when a query is rejected by the read-only guard."""


@dataclass
class QueryResult:
    sql: str
    df: pd.DataFrame
    truncated: bool = False
    row_count: int = 0
    tables: list[str] = field(default_factory=list)

    def to_markdown(self, max_rows: int = 50) -> str:
        if self.df.empty:
            return "_Query returned no rows._"
        shown = self.df.head(max_rows)
        md = shown.to_markdown(index=False, floatfmt=",.2f")
        if len(self.df) > max_rows or self.truncated:
            md += f"\n\n_Showing {len(shown)} of {self.row_count}{'+' if self.truncated else ''} rows._"
        return md


def db_path() -> Path:
    return Path(os.getenv("CPG_DB_PATH", DEFAULT_DB_PATH))


_lock = threading.Lock()


@lru_cache(maxsize=1)
def _base_connection(path: str) -> duckdb.DuckDBPyConnection:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Warehouse not found at {path}. Run `python data/generate_cpg_data.py` first."
        )
    con = duckdb.connect(
        path,
        read_only=True,
        config={"enable_external_access": False, "autoinstall_known_extensions": False,
                "autoload_known_extensions": False},
    )
    con.execute("SET lock_configuration = true")
    return con


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return a per-call cursor on the shared read-only connection (thread safe)."""
    with _lock:
        return _base_connection(str(db_path())).cursor()


def reset_connection() -> None:
    """Drop the cached connection (e.g. after regenerating the warehouse)."""
    with _lock:
        try:
            _base_connection(str(db_path())).close()
        except Exception:
            pass
        _base_connection.cache_clear()


@lru_cache(maxsize=1)
def _known_tables(path: str) -> frozenset[str]:
    rows = get_connection().execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    return frozenset(r[0].lower() for r in rows)


def known_tables() -> frozenset[str]:
    return _known_tables(str(db_path()))


def validate_sql(sql: str) -> str:
    """Return the cleaned query or raise ``UnsafeQueryError``."""
    cleaned = sql.strip().strip("`").strip()
    cleaned = re.sub(r"^sql\s*\n", "", cleaned, flags=re.IGNORECASE).strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeQueryError("Empty query.")

    con = get_connection()
    try:
        statements = con.extract_statements(cleaned)
    except duckdb.Error as exc:
        raise UnsafeQueryError(f"SQL parse error: {exc}") from exc
    if len(statements) != 1:
        raise UnsafeQueryError("Exactly one statement is allowed.")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise UnsafeQueryError(f"Only SELECT queries are allowed (got {statements[0].type.name}).")

    no_strings = _STRING_LITERAL.sub("''", cleaned)
    if m := _FORBIDDEN.search(no_strings):
        raise UnsafeQueryError(f"Keyword '{m.group(1).upper()}' is not allowed in read-only mode.")
    if m := _FILE_FUNCS.search(no_strings):
        raise UnsafeQueryError(f"Table function '{m.group(1)}' is not allowed.")
    return cleaned


def referenced_tables(sql: str) -> list[str]:
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", _STRING_LITERAL.sub("''", sql).lower()))
    return sorted(tokens & known_tables())


def run_query(sql: str, params: list | dict | None = None, max_rows: int = MAX_ROWS,
              validate: bool = True) -> QueryResult:
    """Execute a read-only query and return at most ``max_rows`` rows.

    ``validate=False`` is reserved for trusted, parameterised queries written
    in this code base; anything originating from an LLM or a user must go
    through validation.
    """
    sql = validate_sql(sql) if validate else sql
    con = get_connection()
    timer = threading.Timer(QUERY_TIMEOUT_S, con.interrupt)
    timer.start()
    try:
        rel = con.execute(f"SELECT * FROM ({sql}) AS _q LIMIT {max_rows + 1}", params) if validate \
            else con.execute(sql, params)
        df = rel.fetchdf()
    except duckdb.InterruptException as exc:
        raise TimeoutError(f"Query exceeded {QUERY_TIMEOUT_S}s and was cancelled.") from exc
    finally:
        timer.cancel()
    truncated = len(df) > max_rows
    if truncated:
        df = df.head(max_rows)
    return QueryResult(sql=sql, df=df, truncated=truncated, row_count=len(df),
                       tables=referenced_tables(sql) if validate else [])


def query_df(sql: str, params: list | dict | None = None) -> pd.DataFrame:
    """Trusted internal query helper (no row cap, no validation)."""
    return get_connection().execute(sql, params).fetchdf()


def as_of_date() -> pd.Timestamp:
    val = get_connection().execute(
        "SELECT value FROM meta_dataset_info WHERE key = 'as_of_date'").fetchone()[0]
    return pd.Timestamp(val)


@lru_cache(maxsize=1)
def _schema_description(path: str) -> str:
    con = get_connection()
    tables = con.execute(
        "SELECT table_name, comment FROM duckdb_tables() WHERE schema_name = 'main' "
        "AND table_name NOT LIKE 'meta_injected%' ORDER BY table_name").fetchall()
    cols = con.execute(
        "SELECT table_name, column_name, data_type, comment FROM duckdb_columns() "
        "WHERE schema_name = 'main' ORDER BY table_name, column_index").fetchdf()
    lines = []
    for name, comment in tables:
        lines.append(f"TABLE {name} -- {comment or ''}".rstrip(" -"))
        for r in cols[cols.table_name == name].itertuples():
            note = f"  -- {r.comment}" if r.comment else ""
            lines.append(f"  {r.column_name} {r.data_type}{note}")
    return "\n".join(lines)


def schema_description() -> str:
    """Human/LLM readable schema built from the COMMENTs stored in the warehouse."""
    return _schema_description(str(db_path()))
