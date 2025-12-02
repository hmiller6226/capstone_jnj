#!/usr/bin/env python3
"""
Daily vulnerability pipeline:
- Pulls data from NVD/KEV/JVN/EUVD
- Cleans and merges into a unified master via `merge_clean.consolidate()`
- Optionally uploads the merged table to Postgres
- Schedules the above to run daily

Usage examples:
  python pipeline.py --run-once
  python pipeline.py --time 06:30
  python pipeline.py --init-nvd --start-year 2002
  python pipeline.py --from-year 2024 --upload --upload-schema vuln --upload-table combined_master
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- NEW: env + db deps
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # .env is optional; safe to continue without it
    pass

import io
import getpass
import psycopg2
from psycopg2 import sql

# --- NEW: import your merger module (must be in same project dir)
import merge_clean  # exposes consolidate()

# =============================================================================
# Timezone / Paths
# =============================================================================

try:
    from tzlocal import get_localzone_name
    LOCAL_TZ = ZoneInfo(get_localzone_name())
except Exception:
    LOCAL_TZ = ZoneInfo("America/Los_Angeles")

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "daily_refresh.log"
STATE_PATH = BASE_DIR / "state.json"

# NVD
NVD_DIR = DATA_DIR / "nvd"
NVD_ZIPS = NVD_DIR / "zips"
NVD_JSON = NVD_DIR / "json"
CLEAN_DIR = DATA_DIR / "clean"
NVD_MASTER_CSV = CLEAN_DIR / "nvd_master.csv"
NVD_MASTER_PARQUET = CLEAN_DIR / "nvd_master.parquet"

# KEV
KEV_DIR = DATA_DIR / "kev"
KEV_CSV = KEV_DIR / "known_exploited_vulnerabilities.csv"
KEV_URL = "https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv"

# JVN
JVN_DIR = DATA_DIR / "jvn"
JVN_CSV = JVN_DIR / "jvndb_hnd_incremental.csv"
JVN_BASE_URL = "https://jvndb.jvn.jp/myjvn"
JVN_NS = {
    "rss": "http://purl.org/rss/1.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "sec": "http://jvn.jp/rss/mod_sec/3.0/",
    "status": "http://jvndb.jvn.jp/myjvn/Status",
}

# EUVD
EUVD_DIR = DATA_DIR / "euvd"
EUVD_CSV = EUVD_DIR / "EU_vulnerability_details_incremental.csv"
EUVD_BASE_URL = "https://euvdservices.enisa.europa.eu/api/search"

HTTP_HEADERS = {
    "User-Agent": "vuln-pipeline/1.0",
    "Accept": "*/*",
}

# Network tuning
NET_CONNECT_TIMEOUT = 10
NET_READ_TIMEOUT = 120
NET_TOTAL_TIMEOUT = (NET_CONNECT_TIMEOUT, NET_READ_TIMEOUT)
NET_RETRIES = 5
NET_BACKOFF = 1.5
EUVD_PAGE_DELAY = 0.3

# =============================================================================
# Logging / Utilities
# =============================================================================

def setup_logging() -> None:
    """Configure file + console logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger("").addHandler(sh)

def try_step(label: str, fn, *args, **kwargs):
    """
    Run a step and catch all exceptions.

    Parameters
    ----------
    label : str
        Human-friendly step name for logging.
    fn : callable
        Function to execute.
    *args, **kwargs :
        Arguments forwarded to the function.
    """
    try:
        fn(*args, **kwargs)
    except Exception as e:
        logging.exception(f"[{label}] failed; continuing: {e}")

# =============================================================================
# Generic HTTP helpers (NVD/KEV/EUVD)
# =============================================================================

def make_session() -> requests.Session:
    """
    Create a configured requests Session with retry/backoff.

    Returns
    -------
    requests.Session
        Session with retry adapters and common headers.
    """
    s = requests.Session()
    s.headers.update({
        "User-Agent": HTTP_HEADERS.get("User-Agent", "vuln-pipeline/1.0"),
        "Accept": "application/json, text/xml, */*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    })
    retry = Retry(
        total=NET_RETRIES, connect=NET_RETRIES, read=NET_RETRIES,
        backoff_factor=NET_BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def get_with_retry(session: requests.Session, url: str, **kwargs) -> requests.Response:
    """
    GET with session-level retry and sensible default timeout.

    Parameters
    ----------
    session : requests.Session
        Session returned from make_session().
    url : str
        Target URL.
    **kwargs :
        Requests arguments (timeout, params, headers, stream, ...)

    Returns
    -------
    requests.Response
    """
    if "timeout" not in kwargs:
        kwargs["timeout"] = NET_TOTAL_TIMEOUT
    return session.get(url, **kwargs)

def http_get(url: str, *, timeout: int | Tuple[int, int] = NET_TOTAL_TIMEOUT,
             retries: int = NET_RETRIES, backoff: float = NET_BACKOFF,
             stream: bool = True) -> requests.Response:
    """
    Convenience wrapper to perform a GET with a fresh retrying Session.

    Notes
    -----
    Use this when you don't need to reuse the Session object.
    """
    s = make_session()
    return get_with_retry(s, url, **{"timeout": timeout, "stream": stream})

# =============================================================================
# State
# =============================================================================

def load_state() -> Dict[str, Any]:
    """
    Load pipeline state (last success time, per-source watermarks).

    Returns
    -------
    dict
        State dictionary with defaults if file is missing/corrupt.
    """
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    now = datetime.now(timezone.utc)
    return {
        "last_success_iso": (now - timedelta(days=1)).isoformat(),
        "jvn": {"since_iso": (now - timedelta(days=7)).isoformat()},
        "euvd": {"since_iso": (now - timedelta(days=7)).isoformat()},
    }

def save_state(state: Dict[str, Any]) -> None:
    """
    Persist pipeline state to disk.

    Parameters
    ----------
    state : dict
        State dictionary.
    """
    STATE_PATH.write_text(json.dumps(state, indent=2))

def month_windows(start_dt: datetime, end_dt: datetime):
    """
    Yield (month_start, month_end) windows from start_dt to end_dt inclusive.

    Parameters
    ----------
    start_dt, end_dt : datetime (UTC)

    Yields
    ------
    Tuple[datetime, datetime]
        Start/end bounds within the same month.
    """
    cur = datetime(start_dt.year, start_dt.month, 1, tzinfo=timezone.utc)
    end_guard = datetime(end_dt.year, end_dt.month, 1, tzinfo=timezone.utc)
    while cur <= end_guard:
        nxt = datetime(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1, tzinfo=timezone.utc)
        yield cur, min(nxt - timedelta(seconds=1), end_dt)
        cur = nxt

# =============================================================================
# NVD (2.0)
# =============================================================================

NVD_BASE = "https://nvd.nist.gov/feeds/json/cve/2.0"
NVD_YEAR_FEED = "nvdcve-2.0-{year}.json.zip"
NVD_RECENT = "nvdcve-2.0-recent.json.zip"
NVD_MODIFIED = "nvdcve-2.0-modified.json.zip"

def nvd_years(start: int = 2002, end: Optional[int] = None) -> List[int]:
    """
    Build a list of NVD years from start to end (inclusive).

    Returns
    -------
    list[int]
    """
    end = end or datetime.utcnow().year
    return list(range(start, end + 1))

def download_zip_file(url: str, dest: Path, overwrite: bool = False) -> Path:
    """
    Download a URL to a local zip file (atomically).

    If the file exists and overwrite=False, it's skipped.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        return dest
    r = http_get(url)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            if chunk:
                f.write(chunk)
    os.replace(tmp, dest)
    return dest

def unzip_json_files(src: Path, out_dir: Path, overwrite: bool = False) -> List[Path]:
    """
    Extract JSON files from a zip into out_dir (atomically).
    Returns list of written/kept JSON paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    with zipfile.ZipFile(src, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".json"):
                continue
            dst = out_dir / Path(name).name
            if dst.exists() and not overwrite:
                out.append(dst)
                continue
            with zf.open(name) as zsrc, open(dst.with_suffix(".partial"), "wb") as f:
                f.write(zsrc.read())
            os.replace(dst.with_suffix(".partial"), dst)
            out.append(dst)
    return out

def iter_cves_from_file(json_path: Path):
    """
    Yield CVE dicts from an NVD JSON file.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for w in data.get("vulnerabilities", []):
        cve = w.get("cve")
        if cve:
            yield cve

JSON_SEPARATORS: Tuple[str, str] = (",", ":")
CWE_RE = re.compile(r"CWE-\d{1,5}", re.IGNORECASE)
CPE_RE = re.compile(r"^cpe:2\.3:[aho]:([^:]+):([^:]+):([^:]*):")

def to_json_compact(obj: Any) -> str:
    """
    Compact JSON dump with stable separators; preserves non-ASCII.
    """
    return json.dumps(obj, ensure_ascii=False, separators=JSON_SEPARATORS)

def normalize_name(name: str) -> str:
    """
    Normalize arbitrary strings to safe snake_case tokens (for columns).
    """
    name = name.strip()
    name = name.replace(".", "_").replace("/", "_").replace(" ", "_")
    name = name.replace("[", "_").replace("]", "_")
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name.lower()

def flatten_to_columns(obj: Any, prefix: str = "all") -> Dict[str, Any]:
    """
    Flatten nested dict/list object into a flat dict of scalar columns.

    Lists are serialized as compact JSON strings to keep 1-cell values.
    """
    out: Dict[str, Any] = {}
    def walk(cur: Any, kp: str):
        if isinstance(cur, dict):
            for k, v in cur.items():
                walk(v, f"{kp}.{k}" if kp else k)
        elif isinstance(cur, (list, tuple)):
            out[kp] = to_json_compact(cur)
        else:
            out[kp] = cur
    walk(obj, prefix)
    return {normalize_name(k): v for k, v in out.items()}

def join_values(vals) -> Optional[str]:
    """
    Join non-empty stringy values by ' | ' with de-duplication.
    """
    vals = [v for v in vals if v]
    return " | ".join(sorted(set(vals))) if vals else None

def desc_en(cve: dict) -> Optional[str]:
    """
    Extract the English description text from an NVD CVE object.
    """
    arr = cve.get("descriptions") or cve.get("description") or []
    en = [d.get("value") for d in arr if d.get("value") and str(d.get("lang","")).lower()=="en"]
    return " ".join(en) if en else None

def cvss_fields(cve: dict) -> Dict[str, Any]:
    """
    Pull a single CVSS block (preferring 3.1 -> 3.0 -> 2.0) into flat fields.
    """
    out = {
        "cvss_basescore": None, "cvss_baseseverity": None, "cvss_vectorstring": None,
        "cvss_attackvector": None, "cvss_version": None,
        "cvss_exploitability": None, "cvss_impact": None,
    }
    metrics = cve.get("metrics") or {}
    for key, ver in (("cvssMetricV31","3.1"), ("cvssMetricV30","3.0"), ("cvssMetricV2","2.0")):
        arr = metrics.get(key) or []
        if arr:
            m = arr[0]
            data = m.get("cvssData", {})
            out.update({
                "cvss_basescore": data.get("baseScore"),
                "cvss_baseseverity": data.get("baseSeverity"),
                "cvss_vectorstring": data.get("vectorString"),
                "cvss_attackvector": data.get("attackVector"),
                "cvss_version": ver,
            })
            out["cvss_exploitability"] = m.get("exploitabilityScore")
            out["cvss_impact"] = m.get("impactScore")
            break
    return out

def cwe_list(cve: dict) -> List[str]:
    """
    Extract all referenced CWE identifiers (normalized) from a CVE object.
    """
    ids = set()
    for w in cve.get("weaknesses") or []:
        for d in (w.get("description") or w.get("descriptions") or []):
            val = (d.get("value") or d.get("description") or "") if isinstance(d, dict) else ""
            for m in CWE_RE.findall(val or ""):
                ids.add(m.upper())
            v = (val or "").upper()
            if "NVD-CWE-NOINFO" in v: ids.add("NVD-CWE-noinfo")
            if "NVD-CWE-OTHER" in v: ids.add("NVD-CWE-Other")
    return sorted(ids)

def cpe_vendors_products(cve: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse configuration nodes to collect vendor and product tokens from CPEs.

    Returns
    -------
    (vendors_str, products_str) joined by ' | ', or None if empty.
    """
    vendors, products = set(), set()
    cfg = cve.get("configurations")
    nodes = cfg.get("nodes") if isinstance(cfg, dict) else (cfg if isinstance(cfg, list) else [])

    def use(cm: dict):
        crit = cm.get("criteria") or cm.get("cpe23Uri") or cm.get("cpe23URI")
        if not isinstance(crit, str):
            return
        m = CPE_RE.match(crit)
        if not m:
            return
        v, p, _ = m.groups()
        if v: vendors.add(v)
        if p: products.add(p)

    def walk(n):
        if isinstance(n, dict):
            matches = n.get("cpeMatch") or n.get("cpeMatches") or []
            for cm in matches:
                if isinstance(cm, dict):
                    use(cm)
            for ch in (n.get("children") or []):
                walk(ch)
        elif isinstance(n, list):
            for x in n: walk(x)

    walk(nodes)
    fmt = lambda s: " | ".join(sorted(s)) if s else None
    return fmt(vendors), fmt(products)

def extract_reference_urls(cve: dict) -> Optional[str]:
    """
    Join all reference URLs in a CVE into a single ' | '-separated string.
    """
    urls = [r.get("url") for r in (cve.get("references") or []) if r.get("url")]
    return join_values(urls)

def jsonify_sequence_cells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert list/tuple cells in a DataFrame to compact JSON strings (row-safe).
    """
    def is_seq(x): return isinstance(x, (list, tuple))
    out = df.copy()
    for c in out.columns:
        if out[c].apply(is_seq).any():
            out[c] = out[c].apply(lambda x: to_json_compact(x) if is_seq(x) else x)
    return out

def clean_nvd_records(cves: Iterable[dict]) -> pd.DataFrame:
    """
    Transform raw NVD CVE dicts into a curated, flat DataFrame.

    - Picks English description
    - Parses CVSS fields
    - Extracts vendors/products/CWEs/references
    - Normalizes column names
    - Coerces date columns to UTC
    - Adds is_known_exploited flag based on CISA fields
    """
    rows = []
    for c in cves:
        vendors, products = cpe_vendors_products(c)
        cwe = cwe_list(c)
        row = {
            "id": c.get("id"),
            "sourceidentifier": c.get("sourceIdentifier"),
            "published": c.get("published"),
            "lastmodified": c.get("lastModified"),
            "description_en": desc_en(c),
            "cwe_list": cwe,
            "cwes": join_values(cwe),
            "references": extract_reference_urls(c),
            "vendors": vendors,
            "products": products,
            "cisa_exploit_add": c.get("cisaExploitAdd"),
            "cisa_action_due": c.get("cisaActionDue"),
            "cisa_required_action": c.get("cisaRequiredAction"),
            "cisa_vulnerability_name": c.get("cisaVulnerabilityName"),
            "raw_json": to_json_compact(c),
        }
        row.update(cvss_fields(c))
        row.update(flatten_to_columns(c, "all"))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[
            "id","sourceidentifier","published","lastmodified","description_en",
            "cwe_list","cwes","references","vendors","products","cvss_basescore",
            "cvss_baseseverity","cvss_vectorstring","cvss_attackvector","cvss_version",
            "cvss_exploitability","cvss_impact","cisa_exploit_add","cisa_action_due",
            "cisa_required_action","cisa_vulnerability_name","is_known_exploited","raw_json"
        ])

    # Normalize names and parse dates
    df.columns = [normalize_name(c) for c in df.columns]
    for col in ["published", "lastmodified", "cisa_exploit_add", "cisa_action_due"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # CISA presence implies "known exploited"
    df["is_known_exploited"] = df.get("cisa_exploit_add").notna() if "cisa_exploit_add" in df.columns else False

    # Normalize CWE representations
    if "cwe_list" in df.columns:
        def fix(v):
            if v is None or (isinstance(v, float) and pd.isna(v)): return None
            if isinstance(v, (list, tuple, set)): return [str(x) for x in v]
            if isinstance(v, str): return [p.strip() for p in (v.split("|") if "|" in v else [v]) if p.strip()]
            return None
        df["cwe_list"] = df["cwe_list"].apply(fix)

    if "cwes" in df.columns:
        def fix2(v):
            if v is None or (isinstance(v, float) and pd.isna(v)): return None
            if isinstance(v, (list, tuple, set)): return " | ".join(sorted(set(str(x) for x in v)))
            return str(v)
        df["cwes"] = df["cwes"].apply(fix2)

    df = df.where(pd.notna(df), None).dropna(subset=["id"]).copy()
    return df

def upsert_nvd_master(inc: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new NVD increments into the persisted master table, dedup by id.

    - Reads existing master from parquet/csv if available
    - Aligns columns
    - Prefers latest 'lastmodified' per id
    - Writes updated master to parquet and csv (atomic)
    """
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    if NVD_MASTER_PARQUET.exists():
        master = pd.read_parquet(NVD_MASTER_PARQUET)
    elif NVD_MASTER_CSV.exists():
        master = pd.read_csv(NVD_MASTER_CSV)
        for col in ["published","lastmodified","cisa_exploit_add","cisa_action_due"]:
            if col in master.columns:
                master[col] = pd.to_datetime(master[col], errors="coerce", utc=True)
    else:
        master = pd.DataFrame()

    if not master.empty:
        cols = sorted(set(master.columns) | set(inc.columns))
        master = master.reindex(columns=cols)
        inc = inc.reindex(columns=cols)
        combined = pd.concat([master, inc], ignore_index=True)
    else:
        combined = inc.copy()

    # Keep the row with the latest lastmodified per CVE id
    lm = combined.get("lastmodified")
    if lm is not None:
        lm = lm.fillna(pd.Timestamp(0, tz="UTC"))
        combined = combined.assign(_lm=lm).sort_values(["id","_lm"])
        combined = combined.drop_duplicates(subset=["id"], keep="last").drop(columns=["_lm"])
    else:
        combined = combined.drop_duplicates(subset=["id"], keep="last")

    # Atomic writes
    tmp_parquet = NVD_MASTER_PARQUET.with_suffix(".parquet.tmp")
    tmp_csv = NVD_MASTER_CSV.with_suffix(".csv.tmp")

    safe = jsonify_sequence_cells(combined)
    safe.to_parquet(tmp_parquet, index=False)
    safe.to_csv(tmp_csv, index=False)
    os.replace(tmp_parquet, NVD_MASTER_PARQUET)
    os.replace(tmp_csv, NVD_MASTER_CSV)
    return combined

def nvd_refresh_recent_modified() -> None:
    """
    Fetch NVD recent+modified feeds, clean, and upsert into master.
    """
    logging.info("[NVD] recent + modified")
    for d in (NVD_ZIPS, NVD_JSON, CLEAN_DIR):
        d.mkdir(parents=True, exist_ok=True)

    rec_zip = NVD_ZIPS / NVD_RECENT
    mod_zip = NVD_ZIPS / NVD_MODIFIED
    download_zip_file(f"{NVD_BASE}/{NVD_RECENT}", rec_zip, overwrite=True)
    download_zip_file(f"{NVD_BASE}/{NVD_MODIFIED}", mod_zip, overwrite=True)

    files = unzip_json_files(rec_zip, NVD_JSON, overwrite=True) + unzip_json_files(mod_zip, NVD_JSON, overwrite=True)
    cves = [c for fp in files for c in iter_cves_from_file(fp)]
    df = clean_nvd_records(cves)
    if df.empty:
        logging.info("[NVD] nothing new")
        return
    out = upsert_nvd_master(df)
    logging.info(f"[NVD] master rows: {len(out):,}")

def nvd_backfill_from(year: int) -> None:
    """
    Backfill NVD by year from `year` to present, updating master each year.
    """
    logging.info(f"[NVD] backfill {year}..present")
    NVD_ZIPS.mkdir(parents=True, exist_ok=True)
    NVD_JSON.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    for y in nvd_years(year, None):
        url = f"{NVD_BASE}/{NVD_YEAR_FEED.format(year=y)}"
        z = NVD_ZIPS / Path(url).name
        try:
            download_zip_file(url, z, overwrite=False)
            jsons = unzip_json_files(z, NVD_JSON, overwrite=False)
        except Exception as e:
            logging.warning(f"[NVD] year {y} failed: {e}")
            continue
        for jp in jsons:
            rows = list(iter_cves_from_file(jp))
            if not rows:
                continue
            df = clean_nvd_records(rows)
            if not df.empty:
                upsert_nvd_master(df)
    logging.info("[NVD] backfill done")

# =============================================================================
# KEV
# =============================================================================

def kev_pull() -> None:
    """
    Download the KEV CSV to data/kev (atomic write).
    """
    KEV_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("[KEV] download")
    r = http_get(KEV_URL, timeout=NET_TOTAL_TIMEOUT, retries=NET_RETRIES, backoff=NET_BACKOFF, stream=True)
    tmp = KEV_CSV.with_suffix(".csv.partial")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)
    os.replace(tmp, KEV_CSV)
    logging.info(f"[KEV] saved {KEV_CSV}")

# =============================================================================
# JVN — year-based HND fetch with hardened per-page retry
# =============================================================================

def make_jvn_session() -> requests.Session:
    """
    Build a JVN-tuned HTTP session with slightly different headers/timeouts.
    """
    s = requests.Session()
    s.headers.update({
        "User-Agent": "vuln-pipeline/1.0",
        "Accept": "text/xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    })
    retry = Retry(
        total=6, connect=6, read=6,
        backoff_factor=1.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def jvn_fetch_year_raw(year: int, lang: str = "ja", sleep_sec: float = 0.6) -> List[Dict[str, Any]]:
    """
    Fetch one calendar year's JVN HND feed with resilient per-page retry.

    Returns
    -------
    list[dict]
        Parsed items with title/description/ids/dates/products etc.
    """
    session = make_jvn_session()
    start_item = 1
    page_size = 50  # JVN max
    records: List[Dict[str, Any]] = []

    def make_params(si: int, size: int) -> Dict[str, str]:
        return {
            "method": "getVulnOverviewList",
            "feed": "hnd",
            "lang": lang,
            "rangeDatePublic": "n",
            "datePublicStartY": str(year),
            "datePublicStartM": "1",
            "datePublicStartD": "1",
            "datePublicEndY": str(year),
            "datePublicEndM": "12",
            "datePublicEndD": "31",
            "maxCountItem": str(size),
            "startItem": str(si),
        }

    first_page_logged = False
    while True:
        params = make_params(start_item, page_size)

        # retries for THIS page (adjust page_size if needed)
        attempts, lowered = 0, False
        while True:
            attempts += 1
            try:
                r = session.get(JVN_BASE_URL, params=params, timeout=(10, 180))
                r.raise_for_status()
                r.encoding = "utf-8"
                xml_text = r.text
                break
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                if not lowered and attempts >= 3 and page_size > 25:
                    lowered = True
                    page_size = 25
                    params = make_params(start_item, page_size)
                    logging.warning(f"[JVN] {year} startItem={start_item} timed out; lowering page_size=25 and retrying")
                elif attempts >= 6:
                    logging.warning(f"[JVN] {year} startItem={start_item} still failing after {attempts} tries: {e}. Skipping page.")
                    xml_text = None
                    break
                time.sleep(1.2)

        if not xml_text:
            start_item += page_size
            continue

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as pe:
            logging.warning(f"[JVN] XML parse error for year {year} startItem={start_item}: {pe}. Skipping page.")
            start_item += page_size
            continue

        st = root.find(".//status:Status", JVN_NS)
        if st is not None and not first_page_logged:
            first_page_logged = True
            logging.info(
                f"[JVN] {year} retCd={st.get('retCd')} totalRes={st.get('totalRes')} "
                f"totalResRet={st.get('totalResRet')} retMax={st.get('retMax')} "
                f"errCd={st.get('errCd')} errMsg={st.get('errMsg')}"
            )
        elif st is None and not first_page_logged:
            first_page_logged = True
            logging.warning("[JVN] No <status:Status> block; response may be malformed.")

        items = root.findall(".//rss:item", JVN_NS)
        if not items:
            if st is not None and st.get("errMsg"):
                logging.warning(f"[JVN] {year} startItem={start_item} err={st.get('errMsg')}")
            else:
                logging.info(f"[JVN] {year} done at startItem={start_item}")
            break

        for item in items:
            title = item.findtext("rss:title", default="", namespaces=JVN_NS)
            link = item.findtext("rss:link", default="", namespaces=JVN_NS)
            desc = item.findtext("rss:description", default="", namespaces=JVN_NS)
            pubdate = item.findtext("dc:date", default="", namespaces=JVN_NS)
            jvndb_id = item.findtext("sec:identifier", default="", namespaces=JVN_NS)

            cvss_el = item.find("sec:cvss", JVN_NS)
            cvss_score = cvss_el.get("score") if cvss_el is not None else ""
            cvss_severity = cvss_el.get("severity") if cvss_el is not None else ""

            cve_ids = ", ".join(
                ref.get("id") for ref in item.findall("sec:references[@source='CVE']", JVN_NS)
                if ref.get("id")
            )

            products = "; ".join(
                f"{cpe.get('vendor')}:{cpe.get('product')}"
                for cpe in item.findall("sec:cpe", JVN_NS)
                if cpe.get("vendor") or cpe.get("product")
            )

            records.append({
                "year": year,
                "jvndb_id": jvndb_id,
                "title": title,
                "description": desc,
                "published_date": pubdate,
                "cvss_score": cvss_score,
                "cvss_severity": cvss_severity,
                "cve_ids": cve_ids,
                "affected_products": products,
                "link": link,
            })

        logging.info(f"[JVN] {year}: received {len(items)} items (startItem={start_item}, page_size={page_size})")
        start_item += page_size
        time.sleep(sleep_sec)

    return records

def jvn_fetch_year_df(year: int, lang: str = "ja", sleep_sec: float = 0.6) -> pd.DataFrame:
    """
    Convenience wrapper to fetch JVN items for one year and return a DataFrame.
    """
    recs = jvn_fetch_year_raw(year, lang=lang, sleep_sec=sleep_sec)
    df = pd.DataFrame(recs)
    if not df.empty:
        df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce", utc=True)
        df.sort_values(["year", "published_date"], ascending=[True, True], inplace=True)
    return df

def jvn_fetch_year_range_df(start_year: int, end_year: int, lang: str = "ja", sleep_sec: float = 0.6) -> pd.DataFrame:
    """
    Fetch JVN for a range of years and return a unique, concatenated DataFrame.
    """
    all_rows: List[pd.DataFrame] = []
    for yr in range(start_year, end_year + 1):
        logging.info(f"[JVN] Fetching year {yr} (HND, lang={lang})")
        df_year = jvn_fetch_year_df(yr, lang=lang, sleep_sec=sleep_sec)
        if not df_year.empty:
            all_rows.append(df_year)
    if not all_rows:
        return pd.DataFrame(columns=["year","jvndb_id","title","description","published_date","cvss_score",
                                     "cvss_severity","cve_ids","affected_products","link"])
    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.drop_duplicates(subset=["jvndb_id"], keep="last", inplace=True)
    return df_all

def jvn_incremental(state: Dict[str, Any]) -> None:
    """
    Incrementally fetch JVN since the last watermark and append to CSV.
    """
    JVN_DIR.mkdir(parents=True, exist_ok=True)
    since_iso = state.get("jvn", {}).get("since_iso")
    since = datetime.fromisoformat(since_iso) if since_iso else (datetime.now(timezone.utc) - timedelta(days=7))
    until = datetime.now(timezone.utc)
    start_year, end_year = since.year, until.year

    logging.info(f"[JVN] incremental {since.date()}..{until.date()} (years {start_year}..{end_year})")
    df_new = jvn_fetch_year_range_df(start_year, end_year, lang="ja", sleep_sec=0.6)
    if not df_new.empty and "published_date" in df_new.columns:
        df_new = df_new[(df_new["published_date"] >= since) & (df_new["published_date"] <= until)]

    if df_new.empty:
        state.setdefault("jvn", {})["since_iso"] = until.isoformat()
        save_state(state)
        logging.info("[JVN] nothing new")
        return

    if JVN_CSV.exists():
        df_old = pd.read_csv(JVN_CSV)
        if "published_date" in df_old.columns:
            df_old["published_date"] = pd.to_datetime(df_old["published_date"], errors="coerce", utc=True)
        df_all = (pd.concat([df_old, df_new], ignore_index=True)
                    .sort_values("published_date")
                    .drop_duplicates(subset=["jvndb_id"], keep="last"))
    else:
        df_all = df_new.sort_values("published_date").drop_duplicates(subset=["jvndb_id"], keep="last")

    tmp = JVN_CSV.with_suffix(".csv.tmp")
    df_all.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, JVN_CSV)

    newest = max(df_new["published_date"].dropna().tolist(), default=until)
    state.setdefault("jvn", {})["since_iso"] = newest.isoformat()
    save_state(state)
    logging.info(f"[JVN] total {len(df_all)}")

def jvn_backfill_from(year: int) -> None:
    """
    Backfill JVN from the given year to present, maintaining a unique CSV.
    """
    logging.info(f"[JVN] backfill {year}..present (year-based)")
    JVN_DIR.mkdir(parents=True, exist_ok=True)

    end_year = datetime.now(timezone.utc).year
    df_all = jvn_fetch_year_range_df(year, end_year, lang="ja", sleep_sec=0.6)

    if JVN_CSV.exists():
        try:
            df_prev = pd.read_csv(JVN_CSV)
            if "published_date" in df_prev.columns:
                df_prev["published_date"] = pd.to_datetime(df_prev["published_date"], errors="coerce", utc=True)
            df_all = (pd.concat([df_prev, df_all], ignore_index=True)
                        .sort_values("published_date")
                        .drop_duplicates(subset=["jvndb_id"], keep="last"))
        except Exception:
            pass

    tmp = JVN_CSV.with_suffix(".csv.tmp")
    df_all.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, JVN_CSV)
    logging.info(f"[JVN] backfill total {len(df_all)}")

# =============================================================================
# EUVD
# =============================================================================

def euvd_fetch(start_dt: datetime, end_dt: datetime, page_size: int = 200) -> pd.DataFrame:
    """
    Fetch EUVD items for a date window with pagination and light backoff.

    Returns
    -------
    pd.DataFrame
        Unique items with parsed date columns.
    """
    session = make_session()
    headers = {
        "User-Agent": "EUVD-bulk/1.0",
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    params = {
        "fromDate": start_dt.date().isoformat(),
        "toDate": end_dt.date().isoformat(),
        "page": 0,
        "size": page_size,
    }

    items: List[Dict[str, Any]] = []

    r = get_with_retry(session, EUVD_BASE_URL, params=params, headers=headers, timeout=NET_TOTAL_TIMEOUT)
    if r.status_code == 403:
        params["size"] = max(50, page_size // 2)
        r = get_with_retry(session, EUVD_BASE_URL, params=params, headers=headers, timeout=NET_TOTAL_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    page_items = data.get("items", data.get("content", []))
    total = data.get("total", data.get("totalElements", len(page_items)))
    items.extend(page_items)
    pages = max(1, math.ceil(total / params["size"]))
    logging.info(f"[EUVD] {start_dt.date()}..{end_dt.date()} pages={pages} first={len(page_items)} total={total}")

    for p in range(1, pages):
        params["page"] = p
        r = get_with_retry(session, EUVD_BASE_URL, params=params, headers=headers, timeout=NET_TOTAL_TIMEOUT)
        if r.status_code == 403:
            params["size"] = max(50, params["size"] // 2)
            r = get_with_retry(session, EUVD_BASE_URL, params=params, headers=headers, timeout=NET_TOTAL_TIMEOUT)
        r.raise_for_status()
        data_p = r.json()
        items.extend(data_p.get("items", data_p.get("content", [])))
        time.sleep(EUVD_PAGE_DELAY)

    df = pd.DataFrame(items)
    if not df.empty:
        for c in ["datePublished", "dateUpdated", "exploitedSince"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
        if "id" in df.columns:
            df = df.drop_duplicates(subset=["id"], keep="last")
    return df

def euvd_incremental(state: Dict[str, Any]) -> None:
    """
    Incrementally fetch EUVD since the last watermark and append to CSV.
    """
    EUVD_DIR.mkdir(parents=True, exist_ok=True)
    since_iso = state.get("euvd", {}).get("since_iso")
    since = datetime.fromisoformat(since_iso) if since_iso else (datetime.now(timezone.utc) - timedelta(days=7))
    until = datetime.now(timezone.utc)

    logging.info(f"[EUVD] {since.date()}..{until.date()}")
    df_new = euvd_fetch(since, until, page_size=200)
    if df_new.empty:
        state.setdefault("euvd", {})["since_iso"] = until.isoformat()
        save_state(state)
        logging.info("[EUVD] nothing new")
        return

    if EUVD_CSV.exists():
        df_old = pd.read_csv(EUVD_CSV)
        for c in ["datePublished", "dateUpdated", "exploitedSince"]:
            if c in df_old.columns:
                df_old[c] = pd.to_datetime(df_old[c], errors="coerce", utc=True)
        if "id" in df_new.columns and "id" in df_old.columns:
            df_all = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=["id"], keep="last")
        else:
            df_all = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(keep="last")
    else:
        df_all = df_new

    tmp = EUVD_CSV.with_suffix(".csv.tmp")
    df_all.to_csv(tmp, index=False)
    os.replace(tmp, EUVD_CSV)

    candidates = []
    for c in ["datePublished", "dateUpdated"]:
        if c in df_new.columns:
            candidates.extend(df_new[c].dropna().tolist())
    newest = max(candidates) if candidates else until
    state.setdefault("euvd", {})["since_iso"] = newest.isoformat()
    save_state(state)
    logging.info(f"[EUVD] total {len(df_all)}")

def euvd_backfill_from(year: int) -> None:
    """
    Backfill EUVD from given year to present, maintaining a unique CSV.
    """
    logging.info(f"[EUVD] backfill {year}..present")
    EUVD_DIR.mkdir(parents=True, exist_ok=True)
    if EUVD_CSV.exists():
        df_all = pd.read_csv(EUVD_CSV)
        for c in ["datePublished", "dateUpdated", "exploitedSince"]:
            if c in df_all.columns:
                df_all[c] = pd.to_datetime(df_all[c], errors="coerce", utc=True)
    else:
        df_all = pd.DataFrame()

    total = 0
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)
    for m0, m1 in month_windows(start, end):
        df = euvd_fetch(m0, m1, page_size=200)
        if df.empty:
            continue
        total += len(df)
        if "id" in df.columns:
            df_all = pd.concat([df_all, df], ignore_index=True).drop_duplicates(subset=["id"], keep="last")
        else:
            df_all = pd.concat([df_all, df], ignore_index=True).drop_duplicates(keep="last")
        tmp = EUVD_CSV.with_suffix(".csv.tmp")
        df_all.to_csv(tmp, index=False)
        os.replace(tmp, EUVD_CSV)
    logging.info(f"[EUVD] backfill added {total}, total {len(df_all)}")

# =============================================================================
# DB UPLOAD HELPERS (NEW)
# =============================================================================

def _q_table(table: str, schema: Optional[str]):
    """
    Build a psycopg2 SQL identifier for [schema.]table.
    """
    return sql.SQL(".").join([sql.Identifier(schema), sql.Identifier(table)]) if schema else sql.Identifier(table)

def _infer_sql_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infer a minimal Postgres column type mapping from pandas dtypes.
    """
    mapping = {}
    for col, dtype in df.dtypes.items():
        if pd.api.types.is_bool_dtype(dtype):
            mapping[col] = "BOOLEAN"
        elif pd.api.types.is_integer_dtype(dtype):
            mapping[col] = "BIGINT"
        elif pd.api.types.is_float_dtype(dtype):
            mapping[col] = "DOUBLE PRECISION"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            mapping[col] = "TIMESTAMPTZ"
        else:
            mapping[col] = "TEXT"
    return mapping

def _create_table_if_needed(cur, table: str, schema_name: Optional[str], schema_map: Dict[str, str], pk: Optional[str]):
    """
    CREATE TABLE IF NOT EXISTS with optional PRIMARY KEY (for upsert mode).
    """
    cols = [sql.SQL("{} {}").format(sql.Identifier(c), sql.SQL(t)) for c, t in schema_map.items()]
    if pk and pk in schema_map:
        cols.append(sql.SQL("PRIMARY KEY ({})").format(sql.Identifier(pk)))
    q = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(_q_table(table, schema_name), sql.SQL(", ").join(cols))
    cur.execute(q)

def _truncate_table(cur, table: str, schema_name: Optional[str]):
    """
    TRUNCATE TABLE target (used by replace mode).
    """
    cur.execute(sql.SQL("TRUNCATE TABLE {}").format(_q_table(table, schema_name)))

def _copy_dataframe(cur, df: pd.DataFrame, table: str, schema_name: Optional[str]):
    """
    Bulk COPY a DataFrame into Postgres using CSV over STDIN.
    """
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    cols_sql = sql.SQL(", ").join([sql.Identifier(c) for c in df.columns])
    q = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT CSV)").format(_q_table(table, schema_name), cols_sql)
    cur.copy_expert(q.as_string(cur), buf)

def _upsert_from_temp(cur, temp_table: str, dest_table: str, columns: List[str], pk: str, schema_name: Optional[str]):
    """
    INSERT ... ON CONFLICT (pk) DO UPDATE using a session-local temp table.
    """
    cols_sql = sql.SQL(", ").join([sql.Identifier(c) for c in columns])
    updates = sql.SQL(", ").join([
        sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(c), sql.Identifier(c)) for c in columns if c != pk
    ])
    q = sql.SQL("""
        INSERT INTO {dest} ({cols})
        SELECT {cols} FROM {tmp}
        ON CONFLICT ({pk})
        DO UPDATE SET {updates}
    """).format(dest=_q_table(dest_table, schema_name), tmp=sql.Identifier(temp_table), cols=cols_sql,
               pk=sql.Identifier(pk), updates=updates)
    cur.execute(q)

def _connect_postgres(dsn: Optional[str], prompt_password: bool):
    """
    Establish a psycopg2 connection.

    Priority:
    1) Provided DSN
    2) PG* environment variables (optionally prompt for password)
    """
    password = None
    if prompt_password:
        password = getpass.getpass("Enter Postgres password: ")
    if dsn:
        return psycopg2.connect(dsn, password=password) if password else psycopg2.connect(dsn)
    # env vars (.env supported)
    missing = [v for v in ("PGHOST","PGDATABASE","PGUSER") if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)} (or provide --dsn).")
    return psycopg2.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT", "5432"),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=(password if prompt_password else os.getenv("PGPASSWORD")),
        sslmode=os.getenv("PGSSLMODE", "prefer"),
    )

def upload_combined_to_postgres(
    table: str,
    schema_name: Optional[str],
    mode: str = "upsert",
    pk: str = "cve_id",
    dsn: Optional[str] = None,
    prompt_password: bool = False,
) -> int:
    """
    Load data/clean/combined_master.{parquet|csv} to Postgres.

    Parameters
    ----------
    table : str
        Destination table name.
    schema_name : Optional[str]
        Destination schema (None -> default search_path).
    mode : {"append","replace","upsert"}
        Upload mode. For "upsert", pk must exist in data and table.
    pk : str
        Primary key column for upsert.
    dsn : Optional[str]
        psycopg2 DSN; if None, uses PG* env vars (with optional prompt).
    prompt_password : bool
        If True, prompt for password (overrides PGPASSWORD).

    Returns
    -------
    int
        Number of rows uploaded/affected.
    """
    pq = CLEAN_DIR / "combined_master.parquet"
    cs = CLEAN_DIR / "combined_master.csv"

    # Prefer parquet for preserved dtypes
    if pq.exists():
        df = pd.read_parquet(pq)
    elif cs.exists():
        df = pd.read_csv(cs, low_memory=False)
    else:
        raise FileNotFoundError("combined_master.{parquet,csv} not found. Run merge step first.")

    if df.empty:
        logging.info("[UPLOAD] nothing to upload (empty DataFrame)")
        return 0

    # Normalize schema from df dtypes
    df_norm = df.copy()
    schema_map = _infer_sql_types(df_norm)

    conn = _connect_postgres(dsn, prompt_password)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            # Ensure schema exists (if provided)
            if schema_name:
                cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name)))

            # Create table if missing (pk only added for upsert mode creation)
            _create_table_if_needed(cur, table, schema_name, schema_map, pk if mode == "upsert" else None)

            if mode == "replace":
                _truncate_table(cur, table, schema_name)
                _copy_dataframe(cur, df_norm, table, schema_name)

            elif mode == "append":
                _copy_dataframe(cur, df_norm, table, schema_name)

            elif mode == "upsert":
                if not pk or pk not in df_norm.columns:
                    raise ValueError("--pk must be a column of the data for upsert mode")
                # Create a temp table matching the data schema
                tmp = f"tmp_{table}_{int(time.time())}"
                cols = [sql.SQL("{} {}").format(sql.Identifier(c), sql.SQL(schema_map[c])) for c in df_norm.columns]
                cur.execute(sql.SQL("CREATE TEMP TABLE {} ({})").format(sql.Identifier(tmp), sql.SQL(", ").join(cols)))
                _copy_dataframe(cur, df_norm, tmp, None)
                _upsert_from_temp(cur, tmp, table, list(df_norm.columns), pk, schema_name)
            else:
                raise ValueError("mode must be one of: append, replace, upsert")

        conn.commit()
        logging.info(f"[UPLOAD] {mode} -> {schema_name+'.' if schema_name else ''}{table} rows={len(df_norm):,}")
        return len(df_norm)
    except Exception:
        conn.rollback()
        logging.exception("[UPLOAD] failed; transaction rolled back")
        raise
    finally:
        conn.close()

# =============================================================================
# Orchestration
# =============================================================================

def run_once(args) -> None:
    """
    Execute a single end-to-end refresh:
    - NVD/KEV/JVN/EUVD increments
    - Merge unified table
    - Optional Postgres upload
    - Update watermarks/state
    """
    state = load_state()
    logging.info("=== refresh start ===")

    # NVD and KEV first (independent)
    try_step("NVD", nvd_refresh_recent_modified)
    try_step("KEV", kev_pull)

    # JVN: try twice; on failure, move on to EUVD
    jvn_ok = False
    for attempt in range(2):
        try:
            logging.info(f"[JVN] attempt {attempt + 1}/2")
            jvn_incremental(state)
            jvn_ok = True
            break
        except Exception as e:
            logging.exception(f"[JVN] attempt {attempt + 1} failed: {e}")
    if not jvn_ok:
        logging.warning("[JVN] failed twice; skipping and continuing to EUVD")

    # EUVD last
    try_step("EUVD", euvd_incremental, state)

    # MERGE unified table
    merged_len = 0
    def _merge_wrapper():
        """
        Wrapper to capture merged length for logging outside of try_step.
        """
        nonlocal merged_len
        df = merge_clean.consolidate()
        merged_len = len(df)
        logging.info(f"[MERGE] combined rows: {merged_len:,}")
    try_step("MERGE", _merge_wrapper)

    # Optional DB upload after successful merge
    if args.upload:
        def _upload_wrapper():
            """
            Wrapper to call the upload with CLI-provided arguments.
            """
            rows = upload_combined_to_postgres(
                table=args.upload_table,
                schema_name=args.upload_schema,
                mode=args.upload_mode,
                pk=args.pk,
                dsn=args.dsn,
                prompt_password=args.prompt_password,
            )
            logging.info(f"[UPLOAD] completed rows={rows:,}")
        try_step("UPLOAD", _upload_wrapper)

    # Save overall watermark even on partial success
    state["last_success_iso"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    logging.info("=== refresh end (partial success possible) ===")

def main():
    """
    CLI entry point:
    - Parses arguments
    - Handles one-shot modes (--run-once, --init-nvd, --from-year)
    - Starts daily scheduler if not one-shot
    """
    setup_logging()

    parser = argparse.ArgumentParser(description="Daily pulls for NVD/KEV/JVN/EUVD + merge + optional Postgres upload.")
    parser.add_argument("--run-once", action="store_true", help="Run one refresh and exit.")
    parser.add_argument("--time", type=str, default="06:00", help="Daily HH:MM (local).")
    parser.add_argument("--init-nvd", action="store_true", help="Full NVD build by year and exit.")
    parser.add_argument("--start-year", type=int, default=2002, help="Start year for --init-nvd.")
    parser.add_argument("--end-year", type=int, default=None, help="End year for --init-nvd (default: current).")
    parser.add_argument("--from-year", type=int, default=None, help="Backfill all sources from Jan 1 of this year to now, then continue.")

    # --- NEW: upload controls
    parser.add_argument("--upload", action="store_true", help="Upload merged table to Postgres after merge.")
    parser.add_argument("--upload-table", default="combined_master", help="Destination table name.")
    parser.add_argument("--upload-schema", default=None, help="Destination schema (e.g., 'vuln').")
    parser.add_argument("--upload-mode", choices=["append","replace","upsert"], default="upsert", help="Upload mode.")
    parser.add_argument("--pk", default="cve_id", help="Primary key column for upsert mode.")
    parser.add_argument("--dsn", default=None, help="Optional Postgres DSN. If omitted, uses PG* env vars.")
    parser.add_argument("--prompt-password", action="store_true", help="Prompt for Postgres password at runtime.")

    args, _ = parser.parse_known_args()

    for d in [NVD_ZIPS, NVD_JSON, CLEAN_DIR, KEV_DIR, JVN_DIR, EUVD_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if args.init_nvd:
        # One-time full NVD build
        if args.end_year is None:
            nvd_backfill_from(args.start_year)
        else:
            logging.info(f"[NVD] full build {args.start_year}..{args.end_year}")
            for y in nvd_years(args.start_year, args.end_year):
                try:
                    url = f"{NVD_BASE}/{NVD_YEAR_FEED.format(year=y)}"
                    z = NVD_ZIPS / Path(url).name
                    download_zip_file(url, z, overwrite=False)
                    for jp in unzip_json_files(z, NVD_JSON, overwrite=False):
                        rows = list(iter_cves_from_file(jp))
                        if rows:
                            df = clean_nvd_records(rows)
                            if not df.empty:
                                upsert_nvd_master(df)
                except Exception as e:
                    logging.warning(f"[NVD] year {y} failed: {e}")

        # Merge (and optional upload) after --init-nvd
        try:
            logging.info("[MERGE] after --init-nvd")
            df = merge_clean.consolidate()
            logging.info(f"[MERGE] combined rows: {len(df):,}")
            if args.upload:
                upload_combined_to_postgres(
                    table=args.upload_table,
                    schema_name=args.upload_schema,
                    mode=args.upload_mode,
                    pk=args.pk,
                    dsn=args.dsn,
                    prompt_password=args.prompt_password,
                )
        except Exception as e:
            logging.exception(f"[MERGE/UPLOAD] failed after --init-nvd: {e}")
        return

    if args.from_year:
        # One-time full backfill for all sources
        yr = int(args.from_year)
        logging.info(f"[BACKFILL] {yr}..present (all sources)")
        try_step("NVD backfill", nvd_backfill_from, yr)
        try_step("KEV", kev_pull)
        try_step("JVN backfill", jvn_backfill_from, yr)
        try_step("EUVD backfill", euvd_backfill_from, yr)
        st = load_state()
        now = datetime.now(timezone.utc).isoformat()
        st["last_success_iso"] = now
        st.setdefault("jvn", {})["since_iso"] = now
        st.setdefault("euvd", {})["since_iso"] = now
        save_state(st)
        logging.info("[BACKFILL] done")

        # Merge + optional upload after backfill
        try:
            logging.info("[MERGE] after backfill")
            df = merge_clean.consolidate()
            logging.info(f"[MERGE] combined rows: {len(df):,}")
            if args.upload:
                upload_combined_to_postgres(
                    table=args.upload_table,
                    schema_name=args.upload_schema,
                    mode=args.upload_mode,
                    pk=args.pk,
                    dsn=args.dsn,
                    prompt_password=args.prompt_password,
                )
        except Exception as e:
            logging.exception(f"[MERGE/UPLOAD] failed after backfill: {e}")

    if args.run_once:
        # Run a single refresh cycle and exit
        run_once(args)
        return

    # Otherwise, start the scheduler
    try:
        hh, mm = args.time.split(":")
        hour, minute = int(hh), int(mm)
        assert 0 <= hour <= 23 and 0 <= minute <= 59
    except Exception:
        raise SystemExit("Bad --time. Use HH:MM, e.g. 06:00")

    sched = BackgroundScheduler(timezone=LOCAL_TZ)
    # Use a lambda to capture current args each run
    sched.add_job(lambda: run_once(args), CronTrigger(hour=hour, minute=minute, timezone=LOCAL_TZ))
    sched.start()
    logging.info(f"scheduler: daily {args.time} ({LOCAL_TZ.key}) -> {LOG_FILE}")

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        sched.shutdown()
        logging.info("stopped")

if __name__ == "__main__":
    main()
