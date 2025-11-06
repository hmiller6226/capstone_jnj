#!/usr/bin/env python3
"""
upload_postgres.py

Load the merged table (combined_master.*) into PostgreSQL.

Usage examples:
  python upload_postgres.py \
    --input data/clean/combined_master.csv \
    --table combined_master \
    --mode replace

  python upload_postgres.py \
    --input data/clean/combined_master.parquet \
    --table combined_master \
    --mode upsert

Connection via environment variables (standard libpq):
  PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD, (optional) PGSSLMODE

Notes:
- Primary key is assumed to be cve_id (text).
- Datetime columns are loaded as TIMESTAMPTZ.
- Booleans -> BOOLEAN, ints -> BIGINT, floats -> DOUBLE PRECISION, others -> TEXT.
"""

from __future__ import annotations

import argparse
import os
import sys
import io
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

try:
    import psycopg2
    from psycopg2 import sql
except ImportError as e:
    print("psycopg2 is required. Install with: pip install psycopg2-binary", file=sys.stderr)
    raise

# ---------- Type inference ----------

def infer_sql_type(dtype: pd.api.types.CategoricalDtype | pd.Series | pd.Series.dtype) -> str:
    # Dtype or pandas dtype string to SQL type
    try:
        if pd.api.types.is_bool_dtype(dtype):
            return "BOOLEAN"
        if pd.api.types.is_integer_dtype(dtype):
            return "BIGINT"
        if pd.api.types.is_float_dtype(dtype):
            return "DOUBLE PRECISION"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "TIMESTAMPTZ"
    except Exception:
        pass
    return "TEXT"

def build_schema_from_df(df: pd.DataFrame) -> Dict[str, str]:
    types: Dict[str, str] = {}
    for c in df.columns:
        types[c] = infer_sql_type(df[c].dtype)
    return types

def normalize_df_for_copy(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure datetimes are isoformat with timezone (UTC) and booleans are 0/1 strings
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            # Convert to UTC ISO 8601; empty -> \N (COPY null)
            out[c] = s.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%S%z").fillna("")
        elif pd.api.types.is_bool_dtype(s):
            out[c] = s.map({True: "t", False: "f"}).fillna("")
        else:
            out[c] = s.astype(str).where(~s.isna(), "")
    return out

# ---------- SQL helpers ----------

def quote_ident(name: str) -> sql.Identifier:
    return sql.Identifier(name)

def create_table_if_needed(cur, table: str, schema: Dict[str, str], pk: str | None):
    cols = []
    for col, typ in schema.items():
        cols.append(sql.SQL("{} {}").format(quote_ident(col), sql.SQL(typ)))
    if pk and pk in schema:
        cols.append(sql.SQL("PRIMARY KEY ({})").format(quote_ident(pk)))
    q = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({});").format(
        quote_ident(table), sql.SQL(", ").join(cols)
    )
    cur.execute(q)

def truncate_table(cur, table: str):
    cur.execute(sql.SQL("TRUNCATE TABLE {}").format(quote_ident(table)))

def create_temp_table(cur, temp_table: str, schema: Dict[str, str]):
    cols = [sql.SQL("{} {}").format(quote_ident(c), sql.SQL(t)) for c, t in schema.items()]
    q = sql.SQL("CREATE TEMP TABLE {} ({});").format(quote_ident(temp_table), sql.SQL(", ").join(cols))
    cur.execute(q)

def copy_dataframe(cur, df: pd.DataFrame, table: str):
    # Use CSV COPY FROM STDIN for speed
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    cols = sql.SQL(", ").join([quote_ident(c) for c in df.columns])
    q = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT CSV)").format(quote_ident(table), cols)
    cur.copy_expert(q.as_string(cur), buf)

def upsert_from_temp(cur, temp_table: str, dest_table: str, columns: List[str], pk: str):
    cols_sql = sql.SQL(", ").join([quote_ident(c) for c in columns])
    excluded_updates = sql.SQL(", ").join(
        [sql.SQL("{} = EXCLUDED.{}").format(quote_ident(c), quote_ident(c)) for c in columns if c != pk]
    )
    q = sql.SQL("""
        INSERT INTO {dest} ({cols})
        SELECT {cols} FROM {tmp}
        ON CONFLICT ({pk})
        DO UPDATE SET {updates};
    """).format(
        dest=quote_ident(dest_table),
        tmp=quote_ident(temp_table),
        cols=cols_sql,
        pk=quote_ident(pk),
        updates=excluded_updates
    )
    cur.execute(q)

# ---------- Main loader ----------

def load_to_postgres(
    input_path: str,
    table: str,
    mode: str = "replace",
    pk: str = "cve_id",
) -> Tuple[int, str]:
    """
    input_path: combined_master.csv or .parquet
    table: destination table name (e.g., combined_master)
    mode: "replace" (truncate+copy) or "upsert" (staging+merge by pk)
    pk: primary key column (default cve_id)
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    # Read data
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if df.empty:
        return 0, "No rows to load."

    # Ensure pk column exists as text
    if pk not in df.columns:
        raise ValueError(f"Primary key column '{pk}' not in input.")
    df[pk] = df[pk].astype(str)

    # Infer schema & normalize for COPY
    schema = build_schema_from_df(df)
    df_norm = normalize_df_for_copy(df)

    # Connection
    conn = psycopg2.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT", "5432"),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        sslmode=os.getenv("PGSSLMODE", "prefer"),
    )
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            create_table_if_needed(cur, table, schema, pk=pk)

            if mode == "replace":
                truncate_table(cur, table)
                copy_dataframe(cur, df_norm, table)
                conn.commit()
                return len(df_norm), f"Loaded (replace) {len(df_norm):,} rows into {table}"

            elif mode == "upsert":
                temp_table = f"_{table}_staging_{os.getpid()}"
                create_temp_table(cur, temp_table, schema)
                copy_dataframe(cur, df_norm, temp_table)
                upsert_from_temp(cur, temp_table, table, list(df_norm.columns), pk)
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(quote_ident(temp_table)))
                conn.commit()
                return len(df_norm), f"Upserted {len(df_norm):,} rows into {table} (pk={pk})"

            else:
                raise ValueError("mode must be 'replace' or 'upsert'")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Upload combined_master.* to PostgreSQL")
    ap.add_argument("--input", required=True, help="Path to combined_master.csv or .parquet")
    ap.add_argument("--table", default="combined_master", help="Destination table name")
    ap.add_argument("--mode", default="replace", choices=["replace", "upsert"], help="Load mode")
    ap.add_argument("--pk", default="cve_id", help="Primary key column for upsert mode")
    args = ap.parse_args()

    rows, msg = load_to_postgres(args.input, args.table, args.mode, args.pk)
    print(msg)

if __name__ == "__main__":
    main()
