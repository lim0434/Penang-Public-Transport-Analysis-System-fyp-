from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

from database import SCHEMA, connect


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
TABLE_EXTENSIONS = {".csv"}
ASSET_EXTENSIONS = {".geojson", ".tif", ".tiff", ".graphml", ".txt"}
SKIP_EXTENSIONS = {".html", ".pyc", ".bak"}
IDENTIFIER = re.compile(r"(^id$|_id$|^id_|_code$|^kod_|nearest_node$)", re.I)


def quote(value):
    return '"' + str(value).replace('"', '""') + '"'


def table_name(key):
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", key.lower()).strip("_")
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"file_{stem[:45]}_{digest}"


def sql_type(series):
    if pd.api.types.is_bool_dtype(series.dtype):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(series.dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series.dtype):
        return "DOUBLE PRECISION"
    return "TEXT"


def database_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def is_generated_result(conn, source_path):
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT sha256 FROM {SCHEMA}.file_registry WHERE source_path = %s",
        (source_path,),
    )
    row = cursor.fetchone()
    cursor.close()
    return bool(row and row[0] == "generated")


def ensure_schema(conn):
    cursor = conn.cursor()
    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    cursor.execute(
        f"CREATE TABLE IF NOT EXISTS {SCHEMA}.file_registry ("
        "source_path TEXT PRIMARY KEY, data_kind TEXT NOT NULL, table_name TEXT, "
        "file_size BIGINT NOT NULL, sha256 TEXT NOT NULL, "
        "updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    cursor.execute(
        f"CREATE TABLE IF NOT EXISTS {SCHEMA}.file_assets ("
        "source_path TEXT PRIMARY KEY REFERENCES "
        f"{SCHEMA}.file_registry(source_path) ON DELETE CASCADE, "
        "content BYTEA NOT NULL)"
    )
    conn.commit()
    cursor.close()


def import_table(conn, path, key, checksum):
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    dtype = {column: str for column in header if IDENTIFIER.search(str(column))}
    frame = pd.read_csv(path, dtype=dtype or None, low_memory=False, encoding="utf-8-sig")
    name = table_name(key)
    columns = list(frame.columns)
    definitions = ", ".join(
        f"{quote(str(column))} {sql_type(frame[column])}" for column in columns
    )
    column_sql = ", ".join(quote(str(column)) for column in columns)
    placeholders = ", ".join(["%s"] * len(columns))
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {SCHEMA}.{quote(name)}")
    cursor.execute(f"CREATE TABLE {SCHEMA}.{quote(name)} ({definitions})")
    insert_sql = f"INSERT INTO {SCHEMA}.{quote(name)} ({column_sql}) VALUES ({placeholders})"
    batch = []
    for row in frame.itertuples(index=False, name=None):
        batch.append(tuple(database_value(value) for value in row))
        if len(batch) >= 2000:
            cursor.executemany(insert_sql, batch)
            batch.clear()
    if batch:
        cursor.executemany(insert_sql, batch)
    cursor.execute(
        f"INSERT INTO {SCHEMA}.file_registry "
        "(source_path, data_kind, table_name, file_size, sha256, updated_at) "
        "VALUES (%s, 'table', %s, %s, %s, CURRENT_TIMESTAMP) "
        "ON CONFLICT (source_path) DO UPDATE SET data_kind='table', "
        "table_name=EXCLUDED.table_name, file_size=EXCLUDED.file_size, "
        "sha256=EXCLUDED.sha256, updated_at=CURRENT_TIMESTAMP",
        (key, name, path.stat().st_size, checksum),
    )
    conn.commit()
    cursor.close()
    return len(frame)


def import_asset(conn, path, key, checksum, data):
    cursor = conn.cursor()
    cursor.execute(
        f"INSERT INTO {SCHEMA}.file_registry "
        "(source_path, data_kind, table_name, file_size, sha256, updated_at) "
        "VALUES (%s, 'asset', NULL, %s, %s, CURRENT_TIMESTAMP) "
        "ON CONFLICT (source_path) DO UPDATE SET data_kind='asset', table_name=NULL, "
        "file_size=EXCLUDED.file_size, sha256=EXCLUDED.sha256, "
        "updated_at=CURRENT_TIMESTAMP",
        (key, len(data), checksum),
    )
    cursor.execute(
        f"INSERT INTO {SCHEMA}.file_assets (source_path, content) VALUES (%s, %s) "
        "ON CONFLICT (source_path) DO UPDATE SET content=EXCLUDED.content",
        (key, data),
    )
    conn.commit()
    cursor.close()
    return len(data)


def main():
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Project data folder not found: {DATA_ROOT}")
    conn = connect()
    try:
        ensure_schema(conn)
        imported_tables = 0
        imported_assets = 0
        for path in sorted(DATA_ROOT.rglob("*")):
            if not path.is_file() or path.suffix.lower() in SKIP_EXTENSIONS:
                continue
            suffix = path.suffix.lower()
            if suffix == ".graphml" and path.name != "walking_network_candidate.graphml":
                continue
            if suffix not in TABLE_EXTENSIONS | ASSET_EXTENSIONS:
                continue
            key = path.relative_to(ROOT).as_posix()
            if is_generated_result(conn, key):
                print(f"Kept database-generated result: {key}")
                continue
            raw = path.read_bytes()
            checksum = hashlib.sha256(raw).hexdigest()
            if suffix in TABLE_EXTENSIONS:
                rows = import_table(conn, path, key, checksum)
                imported_tables += 1
                print(f"Imported table {key}: {rows:,} rows")
            else:
                size = import_asset(conn, path, key, checksum, raw)
                imported_assets += 1
                print(f"Stored asset {key}: {size:,} bytes")
        print(
            f"Completed: {imported_tables} data tables and {imported_assets} file assets. "
            "Original files were kept."
        )
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Import stopped: {exc}", file=sys.stderr)
        raise
