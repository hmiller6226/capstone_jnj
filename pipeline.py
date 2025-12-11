#!/usr/bin/env python3
"""
Daily vulnerability pipeline:
- Pulls data from NVD/KEV/JVN/EUVD
- Cleans and merges into a unified master via `merge_clean.consolidate()`
- Optionally uploads the merged table to Postgres
- Optionally runs:
    - Model scoring  (models.py)
    - Feature engineering (build_features.py)
    - Hazard survival modeling (hazard_survival_noplot_cleaned.py)
    - Dashboard (dashboard_combined_final.py)
- Schedules the above to run daily

Usage examples:
  python pipeline.py --run-once
  python pipeline.py --time 06:30
  python pipeline.py --from-year 2024 --upload --upload-schema vuln --upload-table combined_master

With analysis steps:
  python pipeline.py --run-once --run-model --run-features --run-hazard --run-dashboard
"""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import io
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import psycopg2
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from psycopg2 import sql
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from zoneinfo import ZoneInfo

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import merge_clean


try:
    from tzlocal import get_localzone_name
    TZ_LOCAL = ZoneInfo(get_localzone_name())
except Exception:
    TZ_LOCAL = ZoneInfo("America/Los_Angeles")


ROOT = Path(".")
DIR_DATA = ROOT / "data"
DIR_LOGS = ROOT / "logs"
PATH_LOG = DIR_LOGS / "daily_refresh.log"
PATH_STATE = ROOT / "state.json"

DIR_NVD = DIR_DATA / "nvd"
DIR_NVD_Z = DIR_NVD / "zips"
DIR_NVD_J = DIR_NVD / "json"
DIR_CLEAN = DIR_DATA / "clean"
PATH_NVD_CSV = DIR_CLEAN / "nvd_master.csv"
PATH_NVD_PQ = DIR_CLEAN / "nvd_master.parquet"

DIR_KEV = DIR_DATA / "kev"
PATH_KEV = DIR_KEV / "known_exploited_vulnerabilities.csv"
URL_KEV = "https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv"

DIR_JVN = DIR_DATA / "jvn"
PATH_JVN = DIR_JVN / "jvndb_hnd_incremental.csv"
URL_JVN = "https://jvndb.jvn.jp/myjvn"
NS_JVN = {
    "rss": "http://purl.org/rss/1.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "sec": "http://jvn.jp/rss/mod_sec/3.0/",
    "status": "http://jvndb.jvn.jp/myjvn/Status",
}

DIR_EU = DIR_DATA / "euvd"
PATH_EU = DIR_EU / "EU_vulnerability_details_incremental.csv"
URL_EU = "https://euvdservices.enisa.europa.eu/api/search"

DIR_MODEL = DIR_DATA / "model"
DIR_HAZ = DIR_DATA / "hazard"
PATH_FEAT = DIR_CLEAN / "combined_with_features.csv"
PATH_SCORES = DIR_MODEL / "model_scores.csv"
PATH_HAZ_OUT = DIR_HAZ / "hazard_outputs.csv"

HDR = {
    "User-Agent": "vuln-pipeline/1.0",
    "Accept": "*/*",
}

T_CONN = 10
T_READ = 120
T_TOTAL = (T_CONN, T_READ)
N_RETRY = 5
B_OFF = 1.5
EU_DELAY = 0.3

NVD_BASE = "https://nvd.nist.gov/feeds/json/cve/2.0"
NVD_Y_FMT = "nvdcve-2.0-{year}.json.zip"
NVD_REC = "nvdcve-2.0-recent.json.zip"
NVD_MOD = "nvdcve-2.0-modified.json.zip"

JSON_SEP: Tuple[str, str] = (",", ":")
RE_CWE = re.compile(r"CWE-\d{1,5}", re.IGNORECASE)
RE_CPE = re.compile(r"^cpe:2\.3:[aho]:([^:]+):([^:]+):([^:]*):")


def log_setup() -> None:
    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(PATH_LOG),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger("").addHandler(h)


def step_try(lbl: str, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except Exception as e:
        logging.exception(f"[{lbl}] failed; continuing: {e}")


def sess_make() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": HDR.get("User-Agent", "vuln-pipeline/1.0"),
        "Accept": "application/json, text/xml, */*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    })
    r = Retry(
        total=N_RETRY,
        connect=N_RETRY,
        read=N_RETRY,
        backoff_factor=B_OFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    ad = HTTPAdapter(max_retries=r, pool_connections=10, pool_maxsize=10)
    s.mount("https://", ad)
    s.mount("http://", ad)
    return s


def get_r(session: requests.Session, url: str, **kw) -> requests.Response:
    if "timeout" not in kw:
        kw["timeout"] = T_TOTAL
    return session.get(url, **kw)


def get_simple(url: str, *, timeout: int | Tuple[int, int] = T_TOTAL,
               retries: int = N_RETRY, backoff: float = B_OFF,
               stream: bool = True) -> requests.Response:
    s = sess_make()
    return get_r(s, url, **{"timeout": timeout, "stream": stream})


def state_load() -> Dict[str, Any]:
    if PATH_STATE.exists():
        try:
            return json.loads(PATH_STATE.read_text())
        except Exception:
            pass
    now = datetime.now(timezone.utc)
    return {
        "last_success_iso": (now - timedelta(days=1)).isoformat(),
        "jvn": {"since_iso": (now - timedelta(days=7)).isoformat()},
        "euvd": {"since_iso": (now - timedelta(days=7)).isoformat()},
    }


def state_save(st: Dict[str, Any]) -> None:
    PATH_STATE.write_text(json.dumps(st, indent=2))


def win_months(dt0: datetime, dt1: datetime):
    cur = datetime(dt0.year, dt0.month, 1, tzinfo=timezone.utc)
    guard = datetime(dt1.year, dt1.month, 1, tzinfo=timezone.utc)
    while cur <= guard:
        nxt = datetime(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1, tzinfo=timezone.utc)
        yield cur, min(nxt - timedelta(seconds=1), dt1)
        cur = nxt


def nvd_year_list(start: int = 2002, end: Optional[int] = None) -> List[int]:
    end = end or datetime.utcnow().year
    return list(range(start, end + 1))


def zip_dl(url: str, dst: Path, overwrite: bool = False) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return dst
    r = get_simple(url)
    tmp = dst.with_suffix(dst.suffix + ".partial")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            if chunk:
                f.write(chunk)
    os.replace(tmp, dst)
    return dst


def zip_unpack(src: Path, out_dir: Path, overwrite: bool = False) -> List[Path]:
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


def cve_iter(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for w in data.get("vulnerabilities", []):
        cve = w.get("cve")
        if cve:
            yield cve


def json_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=JSON_SEP)


def name_norm(n: str) -> str:
    n = n.strip()
    n = n.replace(".", "_").replace("/", "_").replace(" ", "_")
    n = n.replace("[", "_").replace("]", "_")
    n = re.sub(r"[^A-Za-z0-9_]+", "_", n)
    n = re.sub(r"_+", "_", n).strip("_")
    return n.lower()


def flat_cols(obj: Any, prefix: str = "all") -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def walk(cur: Any, kp: str):
        if isinstance(cur, dict):
            for k, v in cur.items():
                walk(v, f"{kp}.{k}" if kp else k)
        elif isinstance(cur, (list, tuple)):
            out[kp] = json_compact(cur)
        else:
            out[kp] = cur

    walk(obj, prefix)
    return {name_norm(k): v for k, v in out.items()}


def join_vals(vs) -> Optional[str]:
    vs = [v for v in vs if v]
    return " | ".join(sorted(set(vs))) if vs else None


def desc_en_pick(cve: dict) -> Optional[str]:
    arr = cve.get("descriptions") or cve.get("description") or []
    en = [d.get("value") for d in arr if d.get("value") and str(d.get("lang", "")).lower() == "en"]
    return " ".join(en) if en else None


def cvss_pick(cve: dict) -> Dict[str, Any]:
    out = {
        "cvss_basescore": None,
        "cvss_baseseverity": None,
        "cvss_vectorstring": None,
        "cvss_attackvector": None,
        "cvss_version": None,
        "cvss_exploitability": None,
        "cvss_impact": None,
    }
    metrics = cve.get("metrics") or {}
    for key, ver in (("cvssMetricV31", "3.1"), ("cvssMetricV30", "3.0"), ("cvssMetricV2", "2.0")):
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


def cwe_extract(cve: dict) -> List[str]:
    ids = set()
    for w in cve.get("weaknesses") or []:
        for d in (w.get("description") or w.get("descriptions") or []):
            val = (d.get("value") or d.get("description") or "") if isinstance(d, dict) else ""
            for m in RE_CWE.findall(val or ""):
                ids.add(m.upper())
            v = (val or "").upper()
            if "NVD-CWE-NOINFO" in v:
                ids.add("NVD-CWE-noinfo")
            if "NVD-CWE-OTHER" in v:
                ids.add("NVD-CWE-Other")
    return sorted(ids)


def cpe_vp(cve: dict) -> Tuple[Optional[str], Optional[str]]:
    vendors, products = set(), set()
    cfg = cve.get("configurations")

    def crit_get(cm: dict) -> Optional[str]:
        crit = cm.get("criteria") or cm.get("cpe23Uri") or cm.get("cpe23URI")
        return crit if isinstance(crit, str) else None

    def use_cpe(crit: str):
        m = RE_CPE.match(crit)
        if not m:
            return
        v, p, _ = m.groups()
        if v:
            vendors.add(v)
        if p:
            products.add(p)

    def walk(node):
        if isinstance(node, dict):
            if "nodes" in node and isinstance(node["nodes"], list):
                for n in node["nodes"]:
                    walk(n)
            matches = node.get("cpeMatch") or node.get("cpeMatches") or []
            if isinstance(matches, list):
                for cm in matches:
                    if isinstance(cm, dict):
                        crit = crit_get(cm)
                        if crit:
                            use_cpe(crit)
            children = node.get("children") or []
            if isinstance(children, list):
                for ch in children:
                    walk(ch)
        elif isinstance(node, list):
            for x in node:
                walk(x)

    if isinstance(cfg, dict):
        walk(cfg)
    elif isinstance(cfg, list):
        for block in cfg:
            walk(block)

    fmt = lambda s: " | ".join(sorted(s)) if s else None
    return fmt(vendors), fmt(products)


def refs_join(cve: dict) -> Optional[str]:
    urls = [r.get("url") for r in (cve.get("references") or []) if r.get("url")]
    return join_vals(urls)


def seq_cells_json(df: pd.DataFrame) -> pd.DataFrame:
    def is_seq(x): return isinstance(x, (list, tuple))
    out = df.copy()
    for c in out.columns:
        if out[c].apply(is_seq).any():
            out[c] = out[c].apply(lambda x: json_compact(x) if is_seq(x) else x)
    return out


def nvd_clean(cves: Iterable[dict]) -> pd.DataFrame:
    rows = []
    for c in cves:
        vdr, prd = cpe_vp(c)
        cwel = cwe_extract(c)
        row = {
            "id": c.get("id"),
            "sourceidentifier": c.get("sourceIdentifier"),
            "published": c.get("published"),
            "lastmodified": c.get("lastModified"),
            "description_en": desc_en_pick(c),
            "cwe_list": cwel,
            "cwes": join_vals(cwel),
            "references": refs_join(c),
            "vendors": vdr,
            "products": prd,
            "cisa_exploit_add": c.get("cisaExploitAdd"),
            "cisa_action_due": c.get("cisaActionDue"),
            "cisa_required_action": c.get("cisaRequiredAction"),
            "cisa_vulnerability_name": c.get("cisaVulnerabilityName"),
            "raw_json": json_compact(c),
        }
        row.update(cvss_pick(c))
        row.update(flat_cols(c, "all"))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[
            "id", "sourceidentifier", "published", "lastmodified", "description_en",
            "cwe_list", "cwes", "references", "vendors", "products", "cvss_basescore",
            "cvss_baseseverity", "cvss_vectorstring", "cvss_attackvector", "cvss_version",
            "cvss_exploitability", "cvss_impact", "cisa_exploit_add", "cisa_action_due",
            "cisa_required_action", "cisa_vulnerability_name", "is_known_exploited", "raw_json"
        ])

    df.columns = [name_norm(c) for c in df.columns]
    for col in ["published", "lastmodified", "cisa_exploit_add", "cisa_action_due"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="ISO8601", errors="coerce", utc=True)

    df["is_known_exploited"] = df.get("cisa_exploit_add").notna() if "cisa_exploit_add" in df.columns else False

    if "cwe_list" in df.columns:
        def fix(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            if isinstance(v, (list, tuple, set)):
                return [str(x) for x in v]
            if isinstance(v, str):
                return [p.strip() for p in (v.split("|") if "|" in v else [v]) if p.strip()]
            return None
        df["cwe_list"] = df["cwe_list"].apply(fix)

    if "cwes" in df.columns:
        def fix2(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            if isinstance(v, (list, tuple, set)):
                return " | ".join(sorted(set(str(x) for x in v)))
            return str(v)
        df["cwes"] = df["cwes"].apply(fix2)

    df = df.where(pd.notna(df), None).dropna(subset=["id"]).copy()
    return df


def nvd_upsert(df_inc: pd.DataFrame) -> pd.DataFrame:
    DIR_CLEAN.mkdir(parents=True, exist_ok=True)

    if PATH_NVD_PQ.exists():
        df_master = pd.read_parquet(PATH_NVD_PQ)
    elif PATH_NVD_CSV.exists():
        df_master = pd.read_csv(PATH_NVD_CSV)
        for col in ["published", "lastmodified", "cisa_exploit_add", "cisa_action_due"]:
            if col in df_master.columns:
                df_master[col] = pd.to_datetime(df_master[col], format="ISO8601", errors="coerce", utc=True)
    else:
        df_master = pd.DataFrame()

    if not df_master.empty:
        cols = sorted(set(df_master.columns) | set(df_inc.columns))
        df_master = df_master.reindex(columns=cols)
        df_inc = df_inc.reindex(columns=cols)
        df_all = pd.concat([df_master, df_inc], ignore_index=True)
    else:
        df_all = df_inc.copy()

    lm = df_all.get("lastmodified")
    if lm is not None:
        lm = lm.fillna(pd.Timestamp(0, tz="UTC"))
        df_all = df_all.assign(_lm=lm).sort_values(["id", "_lm"])
        df_all = df_all.drop_duplicates(subset=["id"], keep="last").drop(columns=["_lm"])
    else:
        df_all = df_all.drop_duplicates(subset=["id"], keep="last")

    tmp_pq = PATH_NVD_PQ.with_suffix(".parquet.tmp")
    tmp_csv = PATH_NVD_CSV.with_suffix(".csv.tmp")

    safe = seq_cells_json(df_all)
    safe.to_parquet(tmp_pq, index=False)
    safe.to_csv(tmp_csv, index=False)
    os.replace(tmp_pq, PATH_NVD_PQ)
    os.replace(tmp_csv, PATH_NVD_CSV)
    return df_all


def nvd_refresh_rm() -> None:
    logging.info("[NVD] recent + modified")
    for d in (DIR_NVD_Z, DIR_NVD_J, DIR_CLEAN):
        d.mkdir(parents=True, exist_ok=True)

    p_rec = DIR_NVD_Z / NVD_REC
    p_mod = DIR_NVD_Z / NVD_MOD
    zip_dl(f"{NVD_BASE}/{NVD_REC}", p_rec, overwrite=True)
    zip_dl(f"{NVD_BASE}/{NVD_MOD}", p_mod, overwrite=True)

    files = zip_unpack(p_rec, DIR_NVD_J, overwrite=True) + zip_unpack(p_mod, DIR_NVD_J, overwrite=True)
    cves = [c for fp in files for c in cve_iter(fp)]
    df = nvd_clean(cves)
    if df.empty:
        logging.info("[NVD] nothing new")
        return
    out = nvd_upsert(df)
    logging.info(f"[NVD] master rows: {len(out):,}")


def nvd_backfill(year: int) -> None:
    logging.info(f"[NVD] backfill {year}..present")
    DIR_NVD_Z.mkdir(parents=True, exist_ok=True)
    DIR_NVD_J.mkdir(parents=True, exist_ok=True)
    DIR_CLEAN.mkdir(parents=True, exist_ok=True)

    for y in nvd_year_list(year, None):
        url = f"{NVD_BASE}/{NVD_Y_FMT.format(year=y)}"
        z = DIR_NVD_Z / Path(url).name
        try:
            zip_dl(url, z, overwrite=False)
            jsons = zip_unpack(z, DIR_NVD_J, overwrite=False)
        except Exception as e:
            logging.warning(f"[NVD] year {y} failed: {e}")
            continue
        for jp in jsons:
            rows = list(cve_iter(jp))
            if not rows:
                continue
            df = nvd_clean(rows)
            if not df.empty:
                nvd_upsert(df)
    logging.info("[NVD] backfill done")


def kev_pull() -> None:
    DIR_KEV.mkdir(parents=True, exist_ok=True)
    logging.info("[KEV] download")
    r = get_simple(URL_KEV, timeout=T_TOTAL, retries=N_RETRY, backoff=B_OFF, stream=True)
    tmp = PATH_KEV.with_suffix(".csv.partial")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)
    os.replace(tmp, PATH_KEV)
    logging.info(f"[KEV] saved {PATH_KEV}")


def jvn_sess_make() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "vuln-pipeline/1.0",
        "Accept": "text/xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    })
    r = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=1.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    ad = HTTPAdapter(max_retries=r, pool_connections=4, pool_maxsize=4)
    s.mount("https://", ad)
    s.mount("http://", ad)
    return s


def jvn_fetch_year_raw(year: int, lang: str = "ja", sleep_sec: float = 0.6) -> List[Dict[str, Any]]:
    session = jvn_sess_make()
    start_item = 1
    page_size = 50
    recs: List[Dict[str, Any]] = []

    def params_make(si: int, size: int) -> Dict[str, str]:
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

    first_logged = False
    while True:
        params = params_make(start_item, page_size)

        attempts, lowered = 0, False
        while True:
            attempts += 1
            try:
                r = session.get(URL_JVN, params=params, timeout=(10, 180))
                r.raise_for_status()
                r.encoding = "utf-8"
                xml_text = r.text
                break
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                if not lowered and attempts >= 3 and page_size > 25:
                    lowered = True
                    page_size = 25
                    params = params_make(start_item, page_size)
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

        st = root.find(".//status:Status", NS_JVN)
        if st is not None and not first_logged:
            first_logged = True
            logging.info(
                f"[JVN] {year} retCd={st.get('retCd')} totalRes={st.get('totalRes')} "
                f"totalResRet={st.get('totalResRet')} retMax={st.get('retMax')} "
                f"errCd={st.get('errCd')} errMsg={st.get('errMsg')}"
            )
        elif st is None and not first_logged:
            first_logged = True
            logging.warning("[JVN] No <status:Status> block; response may be malformed.")

        items = root.findall(".//rss:item", NS_JVN)
        if not items:
            if st is not None and st.get("errMsg"):
                logging.warning(f"[JVN] {year} startItem={start_item} err={st.get('errMsg')}")
            else:
                logging.info(f"[JVN] {year} done at startItem={start_item}")
            break

        for item in items:
            title = item.findtext("rss:title", default="", namespaces=NS_JVN)
            link = item.findtext("rss:link", default="", namespaces=NS_JVN)
            desc = item.findtext("rss:description", default="", namespaces=NS_JVN)
            pubdate = item.findtext("dc:date", default="", namespaces=NS_JVN)
            jvndb_id = item.findtext("sec:identifier", default="", namespaces=NS_JVN)

            cvss_el = item.find("sec:cvss", NS_JVN)
            score = cvss_el.get("score") if cvss_el is not None else ""
            sev = cvss_el.get("severity") if cvss_el is not None else ""

            cve_ids = ", ".join(
                ref.get("id") for ref in item.findall("sec:references[@source='CVE']", NS_JVN)
                if ref.get("id")
            )

            products = "; ".join(
                f"{cpe.get('vendor')}:{cpe.get('product')}"
                for cpe in item.findall("sec:cpe", NS_JVN)
                if cpe.get("vendor") or cpe.get("product")
            )

            recs.append({
                "year": year,
                "jvndb_id": jvndb_id,
                "title": title,
                "description": desc,
                "published_date": pubdate,
                "cvss_score": score,
                "cvss_severity": sev,
                "cve_ids": cve_ids,
                "affected_products": products,
                "link": link,
            })

        logging.info(f"[JVN] {year}: received {len(items)} items (startItem={start_item}, page_size={page_size})")
        start_item += page_size
        time.sleep(sleep_sec)

    return recs


def jvn_fetch_year_df(year: int, lang: str = "ja", sleep_sec: float = 0.6) -> pd.DataFrame:
    recs = jvn_fetch_year_raw(year, lang=lang, sleep_sec=sleep_sec)
    df = pd.DataFrame(recs)
    if not df.empty:
        df["published_date"] = pd.to_datetime(df["published_date"], format="ISO8601", errors="coerce", utc=True)
        df.sort_values(["year", "published_date"], ascending=[True, True], inplace=True)
    return df


def jvn_fetch_range_df(y0: int, y1: int, lang: str = "ja", sleep_sec: float = 0.6) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for yr in range(y0, y1 + 1):
        logging.info(f"[JVN] Fetching year {yr} (HND, lang={lang})")
        dfy = jvn_fetch_year_df(yr, lang=lang, sleep_sec=sleep_sec)
        if not dfy.empty:
            parts.append(dfy)
    if not parts:
        return pd.DataFrame(columns=[
            "year", "jvndb_id", "title", "description", "published_date",
            "cvss_score", "cvss_severity", "cve_ids", "affected_products", "link"
        ])
    df_all = pd.concat(parts, ignore_index=True)
    df_all.drop_duplicates(subset=["jvndb_id"], keep="last", inplace=True)
    return df_all


def jvn_incremental(st: Dict[str, Any]) -> None:
    DIR_JVN.mkdir(parents=True, exist_ok=True)
    since_iso = st.get("jvn", {}).get("since_iso")
    since = datetime.fromisoformat(since_iso) if since_iso else (datetime.now(timezone.utc) - timedelta(days=7))
    until = datetime.now(timezone.utc)
    y0, y1 = since.year, until.year

    logging.info(f"[JVN] incremental {since.date()}..{until.date()} (years {y0}..{y1})")
    df_new = jvn_fetch_range_df(y0, y1, lang="ja", sleep_sec=0.6)
    if not df_new.empty and "published_date" in df_new.columns:
        df_new = df_new[(df_new["published_date"] >= since) & (df_new["published_date"] <= until)]

    if df_new.empty:
        st.setdefault("jvn", {})["since_iso"] = until.isoformat()
        state_save(st)
        logging.info("[JVN] nothing new")
        return

    if PATH_JVN.exists():
        df_old = pd.read_csv(PATH_JVN)
        if "published_date" in df_old.columns:
            df_old["published_date"] = pd.to_datetime(df_old["published_date"], format="ISO8601", errors="coerce", utc=True)
        df_all = (pd.concat([df_old, df_new], ignore_index=True)
                  .sort_values("published_date")
                  .drop_duplicates(subset=["jvndb_id"], keep="last"))
    else:
        df_all = df_new.sort_values("published_date").drop_duplicates(subset=["jvndb_id"], keep="last")

    tmp = PATH_JVN.with_suffix(".csv.tmp")
    df_all.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, PATH_JVN)

    newest = max(df_new["published_date"].dropna().tolist(), default=until)
    st.setdefault("jvn", {})["since_iso"] = newest.isoformat()
    state_save(st)
    logging.info(f"[JVN] total {len(df_all)}")


def jvn_backfill(year: int) -> None:
    logging.info(f"[JVN] backfill {year}..present (year-based)")
    DIR_JVN.mkdir(parents=True, exist_ok=True)
    end_year = datetime.now(timezone.utc).year
    df_all = jvn_fetch_range_df(year, end_year, lang="ja", sleep_sec=0.6)

    if PATH_JVN.exists():
        try:
            df_prev = pd.read_csv(PATH_JVN)
            if "published_date" in df_prev.columns:
                df_prev["published_date"] = pd.to_datetime(df_prev["published_date"], format="ISO8601", errors="coerce", utc=True)
            df_all = (pd.concat([df_prev, df_all], ignore_index=True)
                      .sort_values("published_date")
                      .drop_duplicates(subset=["jvndb_id"], keep="last"))
        except Exception:
            pass

    tmp = PATH_JVN.with_suffix(".csv.tmp")
    df_all.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, PATH_JVN)
    logging.info(f"[JVN] backfill total {len(df_all)}")


def eu_fetch(dt0: datetime, dt1: datetime, page_size: int = 200) -> pd.DataFrame:
    session = sess_make()
    headers = {
        "User-Agent": "EUVD-bulk/1.0",
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    params = {
        "fromDate": dt0.date().isoformat(),
        "toDate": dt1.date().isoformat(),
        "page": 0,
        "size": page_size,
    }

    items: List[Dict[str, Any]] = []

    r = get_r(session, URL_EU, params=params, headers=headers, timeout=T_TOTAL)
    if r.status_code == 403:
        params["size"] = max(50, page_size // 2)
        r = get_r(session, URL_EU, params=params, headers=headers, timeout=T_TOTAL)
    r.raise_for_status()
    data = r.json()
    page_items = data.get("items", data.get("content", []))
    total = data.get("total", data.get("totalElements", len(page_items)))
    items.extend(page_items)
    pages = max(1, math.ceil(total / params["size"]))
    logging.info(f"[EUVD] {dt0.date()}..{dt1.date()} pages={pages} first={len(page_items)} total={total}")

    for p in range(1, pages):
        params["page"] = p
        r = get_r(session, URL_EU, params=params, headers=headers, timeout=T_TOTAL)
        if r.status_code == 403:
            params["size"] = max(50, params["size"] // 2)
            r = get_r(session, URL_EU, params=params, headers=headers, timeout=T_TOTAL)
        r.raise_for_status()
        data_p = r.json()
        items.extend(data_p.get("items", data_p.get("content", [])))
        time.sleep(EU_DELAY)

    df = pd.DataFrame(items)
    if not df.empty:
        for c in ["datePublished", "dateUpdated", "exploitedSince"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], format="ISO8601", errors="coerce", utc=True)
        if "id" in df.columns:
            df = df.drop_duplicates(subset=["id"], keep="last")
    return df


def eu_incremental(st: Dict[str, Any]) -> None:
    DIR_EU.mkdir(parents=True, exist_ok=True)
    since_iso = st.get("euvd", {}).get("since_iso")
    since = datetime.fromisoformat(since_iso) if since_iso else (datetime.now(timezone.utc) - timedelta(days=7))
    until = datetime.now(timezone.utc)

    logging.info(f"[EUVD] {since.date()}..{until.date()}")
    df_new = eu_fetch(since, until, page_size=200)
    if df_new.empty:
        st.setdefault("euvd", {})["since_iso"] = until.isoformat()
        state_save(st)
        logging.info("[EUVD] nothing new")
        return

    if PATH_EU.exists():
        df_old = pd.read_csv(PATH_EU, dtype={7: "string", 8: "string", 10: "string", 12: "string"}, low_memory=False)
        for c in ["datePublished", "dateUpdated", "exploitedSince"]:
            if c in df_old.columns:
                df_old[c] = pd.to_datetime(df_old[c], format="ISO8601", errors="coerce", utc=True)
        if "id" in df_new.columns and "id" in df_old.columns:
            df_all = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=["id"], keep="last")
        else:
            df_all = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(keep="last")
    else:
        df_all = df_new

    tmp = PATH_EU.with_suffix(".csv.tmp")
    df_all.to_csv(tmp, index=False)
    os.replace(tmp, PATH_EU)

    cand = []
    for c in ["datePublished", "dateUpdated"]:
        if c in df_new.columns:
            cand.extend(df_new[c].dropna().tolist())
    newest = max(cand) if cand else until
    st.setdefault("euvd", {})["since_iso"] = newest.isoformat()
    state_save(st)
    logging.info(f"[EUVD] total {len(df_all)}")


def eu_backfill(year: int) -> None:
    logging.info(f"[EUVD] backfill {year}..present")
    DIR_EU.mkdir(parents=True, exist_ok=True)
    if PATH_EU.exists():
        df_all = pd.read_csv(PATH_EU)
        for c in ["datePublished", "dateUpdated", "exploitedSince"]:
            if c in df_all.columns:
                df_all[c] = pd.to_datetime(df_all[c], format="ISO8601", errors="coerce", utc=True)
    else:
        df_all = pd.DataFrame()

    total = 0
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)
    for m0, m1 in win_months(start, end):
        df = eu_fetch(m0, m1, page_size=200)
        if df.empty:
            continue
        total += len(df)
        if "id" in df.columns:
            df_all = pd.concat([df_all, df], ignore_index=True).drop_duplicates(subset=["id"], keep="last")
        else:
            df_all = pd.concat([df_all, df], ignore_index=True).drop_duplicates(keep="last")
        tmp = PATH_EU.with_suffix(".csv.tmp")
        df_all.to_csv(tmp, index=False)
        os.replace(tmp, PATH_EU)
    logging.info(f"[EUVD] backfill added {total}, total {len(df_all)}")


def q_table(table: str, schema: Optional[str]):
    return sql.SQL(".").join([sql.Identifier(schema), sql.Identifier(table)]) if schema else sql.Identifier(table)


def infer_pg_types(df: pd.DataFrame) -> Dict[str, str]:
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


def create_table_if_needed(cur, table: str, schema_name: Optional[str], schema_map: Dict[str, str], pk: Optional[str]):
    cols = [sql.SQL("{} {}").format(sql.Identifier(c), sql.SQL(t)) for c, t in schema_map.items()]
    if pk and pk in schema_map:
        cols.append(sql.SQL("PRIMARY KEY ({})").format(sql.Identifier(pk)))
    q = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(q_table(table, schema_name), sql.SQL(", ").join(cols))
    cur.execute(q)


def truncate_table(cur, table: str, schema_name: Optional[str]):
    cur.execute(sql.SQL("TRUNCATE TABLE {}").format(q_table(table, schema_name)))


def copy_df(cur, df: pd.DataFrame, table: str, schema_name: Optional[str]):
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    cols_sql = sql.SQL(", ").join([sql.Identifier(c) for c in df.columns])
    q = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT CSV)").format(q_table(table, schema_name), cols_sql)
    cur.copy_expert(q.as_string(cur), buf)


def upsert_temp(cur, temp_table: str, dest_table: str, columns: List[str], pk: str, schema_name: Optional[str]):
    cols_sql = sql.SQL(", ").join([sql.Identifier(c) for c in columns])
    updates = sql.SQL(", ").join([
        sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(c), sql.Identifier(c))
        for c in columns if c != pk
    ])
    q = sql.SQL("""
        INSERT INTO {dest} ({cols})
        SELECT {cols} FROM {tmp}
        ON CONFLICT ({pk})
        DO UPDATE SET {updates}
    """).format(
        dest=q_table(dest_table, schema_name),
        tmp=sql.Identifier(temp_table),
        cols=cols_sql,
        pk=sql.Identifier(pk),
        updates=updates
    )
    cur.execute(q)


def pg_connect(dsn: Optional[str], prompt_password: bool):
    password = None
    if prompt_password:
        password = getpass.getpass("Enter Postgres password: ")
    if dsn:
        return psycopg2.connect(dsn, password=password) if password else psycopg2.connect(dsn)
    missing = [v for v in ("PGHOST", "PGDATABASE", "PGUSER") if not os.getenv(v)]
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


def upload_combined(
    table: str,
    schema_name: Optional[str],
    mode: str = "upsert",
    pk: str = "cve_id",
    dsn: Optional[str] = None,
    prompt_password: bool = False,
) -> int:
    pq = DIR_CLEAN / "combined_master.parquet"
    cs = DIR_CLEAN / "combined_master.csv"

    if pq.exists():
        df = pd.read_parquet(pq)
    elif cs.exists():
        df = pd.read_csv(cs, low_memory=False)
    else:
        raise FileNotFoundError("combined_master.{parquet,csv} not found. Run merge step first.")

    if df.empty:
        logging.info("[UPLOAD] nothing to upload (empty DataFrame)")
        return 0

    df_norm = df.copy()
    schema_map = infer_pg_types(df_norm)

    conn = pg_connect(dsn, prompt_password)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            if schema_name:
                cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name)))

            create_table_if_needed(cur, table, schema_name, schema_map, pk if mode == "upsert" else None)

            if mode == "replace":
                truncate_table(cur, table, schema_name)
                copy_df(cur, df_norm, table, schema_name)

            elif mode == "append":
                copy_df(cur, df_norm, table, schema_name)

            elif mode == "upsert":
                if not pk or pk not in df_norm.columns:
                    raise ValueError("--pk must be a column of the data for upsert mode")
                tmp = f"tmp_{table}_{int(time.time())}"
                cols = [sql.SQL("{} {}").format(sql.Identifier(c), sql.SQL(schema_map[c])) for c in df_norm.columns]
                cur.execute(sql.SQL("CREATE TEMP TABLE {} ({})").format(sql.Identifier(tmp), sql.SQL(", ").join(cols)))
                copy_df(cur, df_norm, tmp, None)
                upsert_temp(cur, tmp, table, list(df_norm.columns), pk, schema_name)
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


def run_ext(lbl: str, script_name: str, extra_args: List[str]) -> None:
    p = ROOT / script_name
    if not p.exists():
        logging.error(f"[{lbl}] script not found: {p} (skipping)")
        return
    cmd = [sys.executable, str(p)] + extra_args
    logging.info(f"[{lbl}] running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        logging.exception(f"[{lbl}] failed: {e}")


def run_once(cli) -> None:
    st = state_load()
    logging.info("=== refresh start ===")

    step_try("NVD", nvd_refresh_rm)
    step_try("KEV", kev_pull)

    jvn_ok = False
    for attempt in range(2):
        try:
            logging.info(f"[JVN] attempt {attempt + 1}/2")
            jvn_incremental(st)
            jvn_ok = True
            break
        except Exception as e:
            logging.exception(f"[JVN] attempt {attempt + 1} failed: {e}")
    if not jvn_ok:
        logging.warning("[JVN] failed twice; skipping and continuing to EUVD")

    step_try("EUVD", eu_incremental, st)

    merged_len = 0

    def merge_wrap():
        nonlocal merged_len
        df = merge_clean.consolidate()
        merged_len = len(df)
        logging.info(f"[MERGE] combined rows: {merged_len:,}")
        out_csv = DIR_CLEAN / "combined_master.csv"
        try:
            df.to_csv(out_csv, index=False)
            logging.info(f"[MERGE] wrote {out_csv}")
        except Exception as e:
            logging.warning(f"[MERGE] failed to write combined_master.csv: {e}")

    step_try("MERGE", merge_wrap)

    if cli.upload:
        def upload_wrap():
            rows = upload_combined(
                table=cli.upload_table,
                schema_name=cli.upload_schema,
                mode=cli.upload_mode,
                pk=cli.pk,
                dsn=cli.dsn,
                prompt_password=cli.prompt_password,
            )
            logging.info(f"[UPLOAD] completed rows={rows:,}")
        step_try("UPLOAD", upload_wrap)

    do_model = getattr(cli, "run_model", False)
    do_feat = getattr(cli, "run_features", False)
    do_haz = getattr(cli, "run_hazard", False)
    do_dash = getattr(cli, "run_dashboard", False)

    if do_dash:
        do_model = True
        do_feat = True
        do_haz = True
    if do_haz:
        do_feat = True

    if merged_len > 0 and (do_model or do_feat or do_haz or do_dash):
        DIR_MODEL.mkdir(parents=True, exist_ok=True)
        DIR_HAZ.mkdir(parents=True, exist_ok=True)

        combined_csv = DIR_CLEAN / "combined_master.csv"

        if do_model:
            run_ext(
                "model",
                "models.py",
                ["--input", str(combined_csv), "--output", str(PATH_SCORES)],
            )

        if do_feat:
            run_ext(
                "FEATURES",
                "build_features.py",
                ["--input", str(combined_csv), "--output", str(PATH_FEAT)],
            )

        if do_haz:
            run_ext(
                "HAZARD",
                "hazard_model_final.py",
                ["--input", str(PATH_FEAT), "--output", str(PATH_HAZ_OUT)],
            )

        if do_dash:
            run_ext("DASHBOARD", "dashboard_combined_final_wayne.py", [])
    elif (do_model or do_feat or do_haz or do_dash):
        logging.warning("[ANALYSIS] Skipping MODEL/features/hazard/dashboard because merge produced 0 rows or failed.")

    st["last_success_iso"] = datetime.now(timezone.utc).isoformat()
    state_save(st)
    logging.info("=== refresh end (partial success possible) ===")


def main():
    log_setup()

    p = argparse.ArgumentParser(
        description="Daily pulls for NVD/KEV/JVN/EUVD + merge + optional Postgres upload + optional MODEL/feature/hazard/dashboard steps."
    )
    p.add_argument("--run-once", action="store_true")
    p.add_argument("--time", type=str, default="06:00")
    p.add_argument("--init-nvd", action="store_true")
    p.add_argument("--start-year", type=int, default=2021)
    p.add_argument("--end-year", type=int, default=None)
    p.add_argument("--from-year", type=int, default=None)

    p.add_argument("--upload", action="store_true")
    p.add_argument("--upload-table", default="combined_master")
    p.add_argument("--upload-schema", default=None)
    p.add_argument("--upload-mode", choices=["append", "replace", "upsert"], default="upsert")
    p.add_argument("--pk", default="cve_id")
    p.add_argument("--dsn", default=None)
    p.add_argument("--prompt-password", action="store_true")

    p.add_argument("--run-model", action="store_true")
    p.add_argument("--run-features", action="store_true")
    p.add_argument("--run-hazard", action="store_true")
    p.add_argument("--run-dashboard", action="store_true")

    cli, _ = p.parse_known_args()

    for d in [DIR_NVD_Z, DIR_NVD_J, DIR_CLEAN, DIR_KEV, DIR_JVN, DIR_EU, DIR_MODEL, DIR_HAZ]:
        d.mkdir(parents=True, exist_ok=True)

    if cli.init_nvd:
        if cli.end_year is None:
            nvd_backfill(cli.start_year)
        else:
            logging.info(f"[NVD] full build {cli.start_year}..{cli.end_year}")
            for y in nvd_year_list(cli.start_year, cli.end_year):
                try:
                    url = f"{NVD_BASE}/{NVD_Y_FMT.format(year=y)}"
                    z = DIR_NVD_Z / Path(url).name
                    zip_dl(url, z, overwrite=False)
                    for jp in zip_unpack(z, DIR_NVD_J, overwrite=False):
                        rows = list(cve_iter(jp))
                        if rows:
                            df = nvd_clean(rows)
                            if not df.empty:
                                nvd_upsert(df)
                except Exception as e:
                    logging.warning(f"[NVD] year {y} failed: {e}")

        try:
            logging.info("[MERGE] after --init-nvd")
            df = merge_clean.consolidate()
            logging.info(f"[MERGE] combined rows: {len(df):,}")
            out_csv = DIR_CLEAN / "combined_master.csv"
            try:
                df.to_csv(out_csv, index=False)
                logging.info(f"[MERGE] wrote {out_csv}")
            except Exception as e:
                logging.warning(f"[MERGE] failed to write combined_master.csv: {e}")

            if cli.upload:
                upload_combined(
                    table=cli.upload_table,
                    schema_name=cli.upload_schema,
                    mode=cli.upload_mode,
                    pk=cli.pk,
                    dsn=cli.dsn,
                    prompt_password=cli.prompt_password,
                )
        except Exception as e:
            logging.exception(f"[MERGE/UPLOAD] failed after --init-nvd: {e}")
        return

    if cli.from_year:
        yr = int(cli.from_year)
        logging.info(f"[BACKFILL] {yr}..present (all sources)")
        step_try("NVD backfill", nvd_backfill, yr)
        step_try("KEV", kev_pull)
        step_try("JVN backfill", jvn_backfill, yr)
        step_try("EUVD backfill", eu_backfill, yr)
        st = state_load()
        now = datetime.now(timezone.utc).isoformat()
        st["last_success_iso"] = now
        st.setdefault("jvn", {})["since_iso"] = now
        st.setdefault("euvd", {})["since_iso"] = now
        state_save(st)
        logging.info("[BACKFILL] done")

        try:
            logging.info("[MERGE] after backfill")
            df = merge_clean.consolidate()
            logging.info(f"[MERGE] combined rows: {len(df):,}")
            out_csv = DIR_CLEAN / "combined_master.csv"
            try:
                df.to_csv(out_csv, index=False)
                logging.info(f"[MERGE] wrote {out_csv}")
            except Exception as e:
                logging.warning(f"[MERGE] failed to write combined_master.csv: {e}")

            if cli.upload:
                upload_combined(
                    table=cli.upload_table,
                    schema_name=cli.upload_schema,
                    mode=cli.upload_mode,
                    pk=cli.pk,
                    dsn=cli.dsn,
                    prompt_password=cli.prompt_password,
                )
        except Exception as e:
            logging.exception(f"[MERGE/UPLOAD] failed after backfill: {e}")

    if cli.run_once:
        run_once(cli)
        return

    try:
        hh, mm = cli.time.split(":")
        hour, minute = int(hh), int(mm)
        assert 0 <= hour <= 23 and 0 <= minute <= 59
    except Exception:
        raise SystemExit("Bad --time. Use HH:MM, e.g. 06:00")

    sched = BackgroundScheduler(timezone=TZ_LOCAL)
    sched.add_job(lambda: run_once(cli), CronTrigger(hour=hour, minute=minute, timezone=TZ_LOCAL))
    sched.start()
    logging.info(f"scheduler: daily {cli.time} ({TZ_LOCAL.key}) -> {PATH_LOG}")

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        sched.shutdown()
        logging.info("stopped")


if __name__ == "__main__":
    main()

