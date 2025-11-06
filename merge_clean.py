#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Optional: adopt future pandas behavior now (avoids silent downcasting surprises)
pd.set_option("future.no_silent_downcasting", True)

# ---------------------------------------------------------------------
# Paths (match your existing pipeline layout)
# ---------------------------------------------------------------------
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
CLEAN_DIR = DATA_DIR / "clean"

NVD_MASTER_CSV = CLEAN_DIR / "nvd_master.csv"
NVD_MASTER_PARQUET = CLEAN_DIR / "nvd_master.parquet"

KEV_DIR = DATA_DIR / "kev"
KEV_CSV = KEV_DIR / "known_exploited_vulnerabilities.csv"

JVN_DIR = DATA_DIR / "jvn"
JVN_CSV = JVN_DIR / "jvndb_hnd_incremental.csv"

EUVD_DIR = DATA_DIR / "euvd"
EUVD_CSV = EUVD_DIR / "EU_vulnerability_details_incremental.csv"

COMBINED_PARQUET = CLEAN_DIR / "combined_master.parquet"
COMBINED_CSV = CLEAN_DIR / "combined_master.csv"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    def norm(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"[^\w]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s
    out = df.copy()
    out.columns = [norm(c) for c in df.columns]
    return out

def _safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        logging.info(f"[read] missing: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        logging.warning(f"[read] failed {path}: {e}")
        return pd.DataFrame()

def _dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)

def _unique_nonnull(seq: List[Optional[str]]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            continue
        s = str(x).strip()
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out

def _explode_cves_from_string_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns or df.empty:
        return pd.DataFrame(columns=["cve_id"])
    s = df[col].fillna("").astype(str)
    parts = (
        s.str.replace("\n", " ", regex=False)
         .str.replace("\t", " ", regex=False)
         .str.split(r"[,\;\|\s]+", regex=True)
    )
    exploded = parts.explode().dropna()
    cves = exploded[exploded.str.match(CVE_RE, na=False)].str.upper().unique()
    return pd.DataFrame({"cve_id": cves})

def _extract_cves_from_text(series: pd.Series) -> pd.DataFrame:
    if series is None or series.empty:
        return pd.DataFrame(columns=["cve_id"])
    hits = series.dropna().astype(str).str.upper().str.findall(CVE_RE)
    vals: List[str] = []
    for lst in hits:
        vals.extend([x.upper() for x in lst])
    return pd.DataFrame({"cve_id": sorted(set(vals))}) if vals else pd.DataFrame(columns=["cve_id"])

def _to_bool(series: Optional[pd.Series], index) -> pd.Series:
    """Coerce common representations (1/0, True/False, yes/no, t/f) to pure bool; NaN -> False."""
    if series is None:
        return pd.Series(False, index=index, dtype=bool)
    s = series
    if s.dtype == bool:
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)
    truthy = {"1", "true", "yes", "y", "t"}
    s_str = s.astype(str).str.strip().str.lower()
    return s_str.isin(truthy).astype(bool)

def _rowwise_min_datetime(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Coerce each column to datetime individually, then take row-wise min."""
    series_list = []
    for c in cols:
        if c in df.columns:
            series_list.append(pd.to_datetime(df[c], errors="coerce", utc=True))
    if not series_list:
        return pd.Series(pd.NaT, index=df.index)
    stacked = pd.concat(series_list, axis=1)
    return stacked.min(axis=1)

def _rowwise_max_datetime(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    series_list = []
    for c in cols:
        if c in df.columns:
            series_list.append(pd.to_datetime(df[c], errors="coerce", utc=True))
    if not series_list:
        return pd.Series(pd.NaT, index=df.index)
    stacked = pd.concat(series_list, axis=1)
    return stacked.max(axis=1)

# ---------------------------------------------------------------------
# Load minimal shapes from each dataset keyed by cve_id
# ---------------------------------------------------------------------
def load_nvd() -> pd.DataFrame:
    if NVD_MASTER_PARQUET.exists():
        df = pd.read_parquet(NVD_MASTER_PARQUET)
    else:
        df = _safe_read_csv(NVD_MASTER_CSV)
        for col in ["published","lastmodified","cisa_exploit_add","cisa_action_due"]:
            if col in df.columns:
                df[col] = _dt(df[col])
    if df.empty:
        return pd.DataFrame(columns=["cve_id"])

    df = _norm_cols(df)
    if "id" not in df.columns:
        return pd.DataFrame(columns=["cve_id"])

    df = df.rename(columns={"id": "cve_id"}).copy()
    df["cve_id"] = df["cve_id"].astype(str).str.upper()

    keep = [c for c in [
        "cve_id",
        "published","lastmodified",
        "cvss_basescore","cvss_baseseverity","cvss_vectorstring","cvss_attackvector",
        "cvss_version","cvss_exploitability","cvss_impact",
        "cwes","vendors","products","cwe_list",   # comma fixed
        "is_known_exploited",
    ] if c in df.columns]
    return df[keep].drop_duplicates("cve_id", keep="last")

def load_kev() -> pd.DataFrame:
    df = _safe_read_csv(KEV_CSV)
    if df.empty:
        return pd.DataFrame(columns=["cve_id"])
    df = _norm_cols(df)

    cve_col = None
    for cand in ["cve_id", "cveid", "cve", "cve_id_number"]:
        if cand in df.columns:
            cve_col = cand; break
    if cve_col is None:
        row_text = df.apply(lambda r: " ".join(map(str, r.values)), axis=1)
        df = _extract_cves_from_text(row_text)
        if df.empty:
            return df
        df["kev_listed"] = True
        return df

    df = df.rename(columns={cve_col: "cve_id"})
    df["cve_id"] = df["cve_id"].astype(str).str.upper()

    for dcol in ["dateadded","duedate"]:
        if dcol in df.columns:
            df[dcol] = _dt(df[dcol])
    if "knownransomwarecampaignuse" not in df.columns:
        df["knownransomwarecampaignuse"] = None

    df["kev_listed"] = True
    keep = [c for c in ["cve_id","dateadded","duedate","knownransomwarecampaignuse","kev_listed"] if c in df.columns]
    return df[keep].drop_duplicates("cve_id", keep="last")

def load_euvd() -> pd.DataFrame:
    df = _safe_read_csv(EUVD_CSV)
    if df.empty:
        return pd.DataFrame(columns=["cve_id"])
    df = _norm_cols(df)

    cve_df = pd.DataFrame(columns=["cve_id"])
    if "aliases" in df.columns:
        cve_df = _explode_cves_from_string_col(df, "aliases")
    if cve_df.empty and "description" in df.columns:
        cve_df = _extract_cves_from_text(df["description"])
    if cve_df.empty:
        return pd.DataFrame(columns=["cve_id"])

    for dcol in ["datepublished","dateupdated","exploitedsince"]:
        if dcol in df.columns:
            df[dcol] = _dt(df[dcol])
        else:
            df[dcol] = None

    proj_cols = {
        "euvd_datepublished": "datepublished",
        "euvd_dateupdated": "dateupdated",
        "euvd_exploitedsince": "exploitedsince",
        "euvd_basescore": "basescore",
        "euvd_basevector": "basescorevector",
        "euvd_baseversion": "basescoreversion",
        "euvd_epss": "epss",
        "euvd_assigner": "assigner",
    }
    present = {k: v for k, v in proj_cols.items() if v in df.columns}
    if present:
        df_proj = df[list(present.values())].copy()
        df_proj.columns = list(present.keys())
        df_proj_first = df_proj.head(1).assign(_tmp_key=1)
        cve_df["_tmp_key"] = 1
        out = cve_df.merge(df_proj_first, on="_tmp_key", how="left").drop(columns=["_tmp_key"])
    else:
        out = cve_df.copy()
    out["cve_id"] = out["cve_id"].str.upper()
    return out.drop_duplicates("cve_id", keep="first")

def load_jvn() -> pd.DataFrame:
    df = _safe_read_csv(JVN_CSV)
    if df.empty:
        return pd.DataFrame(columns=["cve_id"])
    df = _norm_cols(df)

    cve_df = pd.DataFrame(columns=["cve_id"])
    if "cve_ids" in df.columns:
        cve_df = _explode_cves_from_string_col(df, "cve_ids")
    if cve_df.empty:
        return pd.DataFrame(columns=["cve_id"])

    if "published_date" in df.columns:
        df["published_date"] = _dt(df["published_date"])

    proj_cols = {
        "jvn_published_date": "published_date",
        "jvn_cvss_score": "cvss_score",
        "jvn_cvss_severity": "cvss_severity",
    }
    present = {k: v for k, v in proj_cols.items() if v in df.columns}
    if present:
        df_proj = df[list(present.values())].copy()
        df_proj.columns = list(present.keys())
        df_proj_first = df_proj.head(1).assign(_tmp_key=1)
        cve_df["_tmp_key"] = 1
        out = cve_df.merge(df_proj_first, on="_tmp_key", how="left").drop(columns=["_tmp_key"])
    else:
        out = cve_df.copy()

    out["cve_id"] = out["cve_id"].str.upper()
    return out.drop_duplicates("cve_id", keep="first")

# ---------------------------------------------------------------------
# Consolidation entrypoint (what you’ll call from your scheduler)
# ---------------------------------------------------------------------
def consolidate() -> pd.DataFrame:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    nvd = load_nvd()
    kev = load_kev()
    euvd = load_euvd()
    jvn = load_jvn()

    logging.info(f"[merge] nvd={len(nvd)} kev={len(kev)} euvd={len(euvd)} jvn={len(jvn)}")

    # Union of CVE IDs
    all_ids = pd.Series(
        _unique_nonnull(
            list(nvd.get("cve_id", [])) +
            list(kev.get("cve_id", [])) +
            list(euvd.get("cve_id", [])) +
            list(jvn.get("cve_id", []))
        ),
        name="cve_id",
    )
    if all_ids.empty:
        logging.info("[merge] no CVEs found; writing empty outputs.")
        out = pd.DataFrame(columns=["cve_id"])
        _write_outputs(out)
        return out

    base = pd.DataFrame({"cve_id": all_ids})
    out = base.merge(nvd, on="cve_id", how="left", suffixes=("", "_nvd"))
    out = out.merge(kev, on="cve_id", how="left")
    out = out.merge(euvd, on="cve_id", how="left")
    out = out.merge(jvn, on="cve_id", how="left")

    # --- Canonical exploitation flag (robust) ---
    a = out["is_known_exploited"] if "is_known_exploited" in out.columns else pd.Series(False, index=out.index)
    b = out["kev_listed"] if "kev_listed" in out.columns else pd.Series(False, index=out.index)
    out["is_known_exploited"] = _to_bool(a, out.index) | _to_bool(b, out.index)

    # --- Canonical first_seen / last_updated (safe row-wise min/max) ---
    date_first_cols = [c for c in ["published","euvd_datepublished","jvn_published_date"] if c in out.columns]
    date_last_cols  = [c for c in ["lastmodified","euvd_dateupdated"] if c in out.columns]
    out["first_seen"]  = _rowwise_min_datetime(out, date_first_cols) if date_first_cols else pd.NaT
    out["last_updated"] = _rowwise_max_datetime(out, date_last_cols) if date_last_cols else pd.NaT

    sort_cols = [c for c in ["is_known_exploited","first_seen","last_updated","cve_id"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False, True, True, True][:len(sort_cols)])

    _write_outputs(out)
    logging.info(f"[merge] wrote rows={len(out):,}")
    return out

def _write_outputs(df: pd.DataFrame) -> None:
    tmp_pq = COMBINED_PARQUET.with_suffix(".parquet.tmp")
    tmp_csv = COMBINED_CSV.with_suffix(".csv.tmp")
    df.to_parquet(tmp_pq, index=False)
    df.to_csv(tmp_csv, index=False)
    os.replace(tmp_pq, COMBINED_PARQUET)
    os.replace(tmp_csv, COMBINED_CSV)

# --- CLI entrypoint ---
if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="Merge NVD/KEV/JVN/EUVD into combined_master.*")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        consolidate()
    except Exception:
        logging.exception("Consolidation failed")
        sys.exit(1)
