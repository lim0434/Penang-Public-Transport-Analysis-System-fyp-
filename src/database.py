from __future__ import annotations

import hashlib
import io
import json
import os
import re
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
SCHEMA = "dss_data"


def _read_env_file():
    env_file = ROOT / ".env"
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def connect():
    _read_env_file()
    required = ["PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Set the PostgreSQL connection details in the project .env file: "
            + ", ".join(missing)
        )
    try:
        import pg8000.dbapi

        return pg8000.dbapi.connect(
            host=os.environ["PGHOST"],
            port=int(os.getenv("PGPORT", "5432")),
            database=os.environ["PGDATABASE"],
            user=os.environ["PGUSER"],
            password=os.environ["PGPASSWORD"],
            timeout=10,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not connect to PostgreSQL. Check the host, port, database, "
            "user and password in .env, and make sure the PostgreSQL server is running."
        ) from exc


def source_key(path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"Data file is outside the project: {path}") from exc


def _quote_identifier(value):
    return '"' + str(value).replace('"', '""') + '"'


def _database_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _registry_row(conn, path):
    key = source_key(path)
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT data_kind, table_name, updated_at FROM {SCHEMA}.file_registry "
        "WHERE source_path = %s",
        (key,),
    )
    result = cursor.fetchone()
    cursor.close()
    if result is None:
        raise FileNotFoundError(
            f"This project dataset has not been imported into PostgreSQL yet: {key}. "
            "Run src/migrate_to_postgres.py first."
        )
    return key, result


def dataset_exists(path):
    try:
        with connect() as conn:
            _registry_row(conn, path)
        return True
    except FileNotFoundError:
        return False


def dataset_version(path):
    with connect() as conn:
        _, (_, _, updated_at) = _registry_row(conn, path)
        return str(updated_at)


def read_dataframe(path, dtype=None):
    with connect() as conn:
        key, (kind, table_name, _) = _registry_row(conn, path)
        if kind != "table":
            raise TypeError(f"{key} is stored as a file asset, not a data table.")
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT * FROM {SCHEMA}.{_quote_identifier(table_name)}"
        )
        rows = cursor.fetchall()
        columns = [item[0] for item in cursor.description]
        cursor.close()
    frame = pd.DataFrame(rows, columns=columns)
    if dtype:
        for column, target in dtype.items():
            if column in frame.columns:
                frame[column] = frame[column].astype(target)
    return frame


def read_asset_bytes(path):
    with connect() as conn:
        key, (kind, _, _) = _registry_row(conn, path)
        if kind != "asset":
            raise TypeError(f"{key} is stored as a data table, not a file asset.")
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT content FROM {SCHEMA}.file_assets WHERE source_path = %s",
            (key,),
        )
        row = cursor.fetchone()
        cursor.close()
    if not row:
        raise FileNotFoundError(f"Database asset is missing: {key}")
    return bytes(row[0])


def read_asset_dataframe(path, **kwargs):
    return pd.read_csv(io.BytesIO(read_asset_bytes(path)), **kwargs)


def read_geojson(path):
    document = json.loads(read_asset_bytes(path).decode("utf-8"))
    crs = document.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    return gpd.GeoDataFrame.from_features(document["features"], crs=crs)


def read_graphml(path):
    return nx.read_graphml(io.BytesIO(read_asset_bytes(path)))


def _sql_type(series):
    if pd.api.types.is_bool_dtype(series.dtype):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(series.dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series.dtype):
        return "DOUBLE PRECISION"
    return "TEXT"


def write_dataframe(path, frame):
    key = source_key(path)
    table_name = _table_name(key)
    columns = list(frame.columns)
    definitions = ", ".join(
        f"{_quote_identifier(str(column))} {_sql_type(frame[column])}"
        for column in columns
    )
    col_sql = ", ".join(_quote_identifier(str(column)) for column in columns)
    placeholders = ", ".join(["%s"] * len(columns))
    rows = [
        tuple(_database_value(value) for value in row)
        for row in frame.itertuples(index=False, name=None)
    ]
    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {SCHEMA}.{_quote_identifier(table_name)}")
        cursor.execute(
            f"CREATE TABLE {SCHEMA}.{_quote_identifier(table_name)} ({definitions})"
        )
        if rows:
            cursor.executemany(
                f"INSERT INTO {SCHEMA}.{_quote_identifier(table_name)} ({col_sql}) "
                f"VALUES ({placeholders})",
                rows,
            )
        cursor.execute(
            f"INSERT INTO {SCHEMA}.file_registry "
            "(source_path, data_kind, table_name, file_size, sha256, updated_at) "
            "VALUES (%s, 'table', %s, %s, %s, CURRENT_TIMESTAMP) "
            "ON CONFLICT (source_path) DO UPDATE SET data_kind='table', "
            "table_name=EXCLUDED.table_name, file_size=EXCLUDED.file_size, "
            "sha256=EXCLUDED.sha256, updated_at=CURRENT_TIMESTAMP",
            (key, table_name, 0, "generated"),
        )
        conn.commit()
        cursor.close()


def _table_name(key):
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", key.lower()).strip("_")
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"file_{stem[:45]}_{digest}"
