from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np

pd.set_option("future.no_silent_downcasting", True)

#paths
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

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    def norm(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"[^\w]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s
    out = df.copy()
    out.columns = [norm(c) for c in df.columns]
    return out

#loaders
def load_nvd() -> pd.DataFrame:
    if NVD_MASTER_PARQUET.exists():
        df = pd.read_parquet(NVD_MASTER_PARQUET)
    else:
        df = _safe_read_csv(NVD_MASTER_CSV)
        for col in ["published", "lastmodified", "cisa_exploit_add", "cisa_action_due"]:
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
        "cwes","vendors","products","cwe_list",
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
            cve_col = cand
            break
    if cve_col is None:
        # fallback: extract CVEs from free text
        row_text = df.apply(lambda r: " ".join(map(str, r.values)), axis=1)
        hits = row_text.dropna().astype(str).str.upper().str.findall(
            re.compile(r"\bCVE-\d{4}-\d{4,7}\b")
        )
        vals: List[str] = []
        for lst in hits:
            vals.extend([x.upper() for x in lst])
        if not vals:
            return pd.DataFrame(columns=["cve_id"])
        out = pd.DataFrame({"cve_id": sorted(set(vals))})
        out["kev_listed"] = True
        return out

    df = df.rename(columns={cve_col: "cve_id"})
    df["cve_id"] = df["cve_id"].astype(str).str.upper()

    for dcol in ["dateadded", "duedate"]:
        if dcol in df.columns:
            df[dcol] = _dt(df[dcol])
    if "knownransomwarecampaignuse" not in df.columns:
        df["knownransomwarecampaignuse"] = None

    df["kev_listed"] = True
    keep = [c for c in ["cve_id","dateadded","duedate",
                        "knownransomwarecampaignuse","kev_listed"]
            if c in df.columns]
    return df[keep].drop_duplicates("cve_id", keep="last")

def load_euvd() -> pd.DataFrame:
    df = _safe_read_csv(EUVD_CSV)
    if df.empty:
        return pd.DataFrame(columns=["cve_id"])
    df = _norm_cols(df)

    rows = []
    for _, row in df.iterrows():
        aliases = str(row.get("aliases", ""))
        aliases = aliases.replace("\n", " ").replace("\t", " ")
        for token in re.split(r"[,\;\|\s]+", aliases):
            token = token.strip().upper()
            if re.match(r"\bCVE-\d{4}-\d{4,7}\b", token):
                rows.append({
                    "cve_id": token,
                    "datepublished": row.get("datepublished"),
                    "dateupdated": row.get("dateupdated"),
                    "exploitedsince": row.get("exploitedsince"),
                    "basescore": row.get("basescore"),
                    "basescorevector": row.get("basescorevector"),
                    "basescoreversion": row.get("basescoreversion"),
                    "epss": row.get("epss"),
                    "assigner": row.get("assigner"),
                })

    if not rows:
        return pd.DataFrame(columns=["cve_id"])

    out = pd.DataFrame(rows)

    # Parse dates
    for dcol in ["datepublished", "dateupdated", "exploitedsince"]:
        if dcol in out.columns:
            out[dcol] = _dt(out[dcol])

    out["cve_id"] = out["cve_id"].str.upper()

    # If a CVE appears multiple times, keep the last published entry
    if "datepublished" in out.columns:
        out = out.sort_values("datepublished").drop_duplicates("cve_id", keep="last")
    else:
        out = out.drop_duplicates("cve_id", keep="last")

    return out

def load_jvn() -> pd.DataFrame:
    df = _safe_read_csv(JVN_CSV)
    if df.empty:
        return pd.DataFrame(columns=["cve_id"])
    df = _norm_cols(df)

    rows = []
    for _, row in df.iterrows():
        ids = str(row.get("cve_ids", ""))
        ids = ids.replace("\n", " ").replace("\t", " ")
        for token in re.split(r"[,\;\|\s]+", ids):
            token = token.strip().upper()
            if re.match(r"\bCVE-\d{4}-\d{4,7}\b", token):
                rows.append({
                    "cve_id": token,
                    "jvn_published_date": row.get("published_date"),
                    "jvn_cvss_score": row.get("cvss_score"),
                    "jvn_cvss_severity": row.get("cvss_severity"),
                })

    if not rows:
        return pd.DataFrame(columns=["cve_id"])

    out = pd.DataFrame(rows)

    if "jvn_published_date" in out.columns:
        out["jvn_published_date"] = _dt(out["jvn_published_date"])

    out["cve_id"] = out["cve_id"].str.upper()

    # Keep latest JVN entry per CVE
    if "jvn_published_date" in out.columns:
        out = out.sort_values("jvn_published_date").drop_duplicates("cve_id", keep="last")
    else:
        out = out.drop_duplicates("cve_id", keep="last")

    return out

CVE_RE = re.compile(r"(CVE-\d{4}-\d+)", re.I)

def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.encode("utf-8", "ignore").str.decode("utf-8")
    df.columns = df.columns.str.strip()
    return df.loc[:, ~df.columns.duplicated()]

def _first(series_list):
    if not series_list:
        return pd.Series(index=pd.RangeIndex(0), dtype="object")
    out = series_list[0].copy()
    for s in series_list[1:]:
        out = out.where(out.notna(), s)
    return out

def _extract_cve_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.extract(CVE_RE, expand=False)

def _build_cve_id(df: pd.DataFrame, preferred_cols):
    candid = []
    cols_lc = {c.lower(): c for c in df.columns}
    for name in preferred_cols:
        if name.lower() in cols_lc:
            c = cols_lc[name.lower()]
            candid.append(_extract_cve_series(df[c]))
    if not candid or all(s.isna().all() for s in candid):
        for c in df.columns:
            s = _extract_cve_series(df[c])
            if s.notna().any():
                candid.append(s)
    cve = _first(candid)
    cve = cve.str.upper().str.strip()
    cve = cve.where(cve.str.match(r"^CVE-\d{4}-\d+$", na=False), np.nan)
    return cve

def _prep_source(df: pd.DataFrame, src: str) -> pd.DataFrame:
    df = _sanitize_columns(df)

    if src == "nvd":
        cve = _build_cve_id(df, preferred_cols=["cve_id", "id"])
        pub_col = next((c for c in ["published"] if c in df.columns), None)
        score_col = next((c for c in ["cvss_basescore"] if c in df.columns), None)
        if pub_col:
            df = df.rename(columns={pub_col: "nvd_published"})
        if score_col:
            df = df.rename(columns={score_col: "nvd_base_score"})

    elif src == "jvn":
        cve = _build_cve_id(df, preferred_cols=["cve_id", "jvndb_id"])
        pub_col = next(
            (c for c in ["jvn_published_date", "published_date", "last_updated"]
             if c in df.columns),
            None,
        )
        # Accept either jvn_cvss_score (from load_jvn) or raw cvss_score
        score_col = next(
            (c for c in ["jvn_cvss_score", "cvss_score"] if c in df.columns),
            None,
        )
        if pub_col:
            df = df.rename(columns={pub_col: "jvn_published"})
        if score_col:
            df = df.rename(columns={score_col: "jvn_base_score"})

    elif src == "eu":
        cve = _build_cve_id(df, preferred_cols=["cve_id", "aliases"])
        pub_col = next(
            (c for c in ["eu_published", "datepublished"] if c in df.columns),
            None,
        )
        # Accept basescore from load_euvd or any pre-renamed euvd_basescore
        score_col = next(
            (c for c in ["eu_base_score", "basescore", "euvd_basescore"]
             if c in df.columns),
            None,
        )
        if pub_col:
            df = df.rename(columns={pub_col: "eu_published"})
        if score_col:
            df = df.rename(columns={score_col: "eu_base_score"})

    elif src == "kev":
        cve = _build_cve_id(df, preferred_cols=["cve_id", "cveID"])
        pub_col = next(
            (c for c in ["kev_published", "dateadded", "dateAdded"]
             if c in df.columns),
            None,
        )
        if pub_col:
            df = df.rename(columns={pub_col: "kev_published"})

    else:
        raise ValueError(f"Unknown src {src}")

    df["cve_id"] = cve.astype("string")
    df[f"{src}_present"] = df["cve_id"].notna()

    if df["cve_id"].notna().sum() == 0:
        pass
    return df

def merge_vuln_sources(nvd, jvn, eu, kev, min_year: int = 2002) -> pd.DataFrame:
    nvd = _prep_source(nvd, "nvd")
    jvn = _prep_source(jvn, "jvn")
    eu  = _prep_source(eu,  "eu")
    kev = _prep_source(kev, "kev")

    for df in (nvd, jvn, eu, kev):
        df["cve_id"] = df["cve_id"].astype("string")

    merged = (
        nvd.merge(jvn, on="cve_id", how="outer", suffixes=("_nvd", "_jvn"))
           .merge(eu,  on="cve_id", how="outer", suffixes=("", "_eu"))
           .merge(kev, on="cve_id", how="outer", suffixes=("", "_kev"))
    )
    merged = merged.loc[:, ~merged.columns.duplicated()]

    for flag in ["nvd_present","jvn_present","eu_present","kev_present"]:
        if flag not in merged.columns:
            merged[flag] = False
        merged[flag] = merged[flag].fillna(False).astype(bool)

    merged["source_list"] = merged.apply(
        lambda r: [name for present, name in [
            (r["nvd_present"], "NVD"),
            (r["jvn_present"], "JVN"),
            (r["eu_present"],  "EUVD"),
            (r["kev_present"], "KEV"),
        ] if present],
        axis=1,
    )
    merged["sources"] = merged["source_list"].apply(lambda xs: ",".join(xs))
    merged["source_count"] = merged["source_list"].apply(len)

    for c in ["nvd_published","jvn_published","eu_published","kev_published"]:
        if c not in merged.columns:
            merged[c] = np.nan
        else:
            merged[c] = (
                merged[c]
                .astype(str)
                .str.strip()
                .replace({"": np.nan, "None": np.nan, "NULL": np.nan, "NaN": np.nan})
            )
    pub = (
        merged[["nvd_published","jvn_published","eu_published","kev_published"]]
        .apply(pd.to_datetime, errors="coerce", utc=True)
        .stack()
        .groupby(level=0)
        .first()
    )
    merged["published_date"] = pub.reindex(merged.index)
    merged["published"] = merged["published_date"]

    for c in ["nvd_base_score","jvn_base_score","eu_base_score"]:
        if c not in merged.columns:
            merged[c] = np.nan
    bs = (
        merged[["nvd_base_score", "eu_base_score", "jvn_base_score"]]
        .apply(pd.to_numeric, errors="coerce")
        .stack()
        .groupby(level=0)
        .first()
    )
    merged["base_score"] = bs.reindex(merged.index)

    merged["cve_year"] = (
        merged["cve_id"]
        .astype(str)
        .str.extract(r"^CVE-(\d{4})-")
        .astype(float)
    )
    merged = merged[merged["cve_year"].ge(min_year).fillna(False)].copy()

    front = [
        "cve_id","cve_year","published_date","base_score",
        "nvd_present","jvn_present","eu_present","kev_present",
        "source_list","sources","source_count",
        "cwe_list",
    ]
    front = [c for c in front if c in merged.columns]
    merged = merged.loc[:, front + [c for c in merged.columns if c not in front]]

    return merged


def consolidate(min_year: int = 2002) -> pd.DataFrame:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    nvd  = load_nvd()
    kev  = load_kev()
    euvd = load_euvd()
    jvn  = load_jvn()

    logging.info(f"[merge] nvd={len(nvd)} kev={len(kev)} euvd={len(euvd)} jvn={len(jvn)}")

    out = merge_vuln_sources(nvd=nvd, jvn=jvn, eu=euvd, kev=kev, min_year=min_year)

    # Write outputs
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

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(
        description="Merge NVD/KEV/JVN/EUVD into combined_master.*"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG","INFO","WARNING","ERROR"]
    )
    parser.add_argument(
        "--min-year", type=int, default=2002,
        help="Filter to CVEs with CVE year >= this"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        consolidate(min_year=args.min_year)
    except Exception:
        logging.exception("Consolidation failed")
        sys.exit(1)
