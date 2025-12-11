import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from xgboost import XGBClassifier



ABLATION_CONFIGS = [
    {"name": "full", "disable_epss": False, "disable_poc": False, "drop": []},
    {"name": "no_epss", "disable_epss": True, "disable_poc": False, "drop": []},
    {"name": "no_poc", "disable_epss": False, "disable_poc": True, "drop": []},
    {"name": "no_epss_no_poc", "disable_epss": True, "disable_poc": True, "drop": []},
    {"name": "no_cvss_ords", "disable_epss": False, "disable_poc": False,
     "drop": [
         "cvss_av_ord", "cvss_ac_ord", "cvss_pr_ord", "cvss_ui_ord", "cvss_s_ord",
         "cvss_c_ord", "cvss_i_ord", "cvss_a_ord", "cvss_v3_present"
     ]},
    {"name": "no_base_score", "disable_epss": False, "disable_poc": False,
     "drop": ["base_score"]},
    {"name": "severity_only", "disable_epss": True, "disable_poc": True,
     "drop": [
         "repo_publication_lag", "update_frequency", "time_since_first_reference",
         "cross_listing_count", "cross_listing_std_days", "cross_listing_variance",
         "related_cwe_count", "weakness_frequency", "vendor_frequency",
         "word_count", "desc_len",
         "github_poc_count", "exploitdb_present", "metasploit_present", "poc_url_count",
         "poc_any_present", "first_poc_lag_days", "poc_within_30d",
         "epss_score", "epss_percentile", "epss_present",
     ]},
    {"name": "exploit_only", "disable_epss": False, "disable_poc": False,
     "drop": [
         "base_score",
         "cvss_av_ord", "cvss_ac_ord", "cvss_pr_ord", "cvss_ui_ord", "cvss_s_ord",
         "cvss_c_ord", "cvss_i_ord", "cvss_a_ord", "cvss_v3_present",
         "repo_publication_lag", "update_frequency", "time_since_first_reference",
         "cross_listing_count", "cross_listing_std_days", "cross_listing_variance",
         "related_cwe_count", "weakness_frequency", "vendor_frequency",
         "word_count", "desc_len"
     ]},
]


def with_ablation_suffix(path: str, ablation_name: str) -> str:
    if not ablation_name:
        return path
    if path.lower().endswith(".csv"):
        stem = path[:-4]
        return f"{stem}_{ablation_name}.csv"
    return f"{path}_{ablation_name}"


def _safe_series(df: pd.DataFrame, name: str, fill=""):
    return df[name] if name in df.columns else pd.Series([fill] * len(df), index=df.index)


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _dtframe_to_days(dt_df: pd.DataFrame) -> pd.DataFrame:
    out_df = pd.DataFrame(index=dt_df.index)
    for col in dt_df.columns:
        s = pd.to_datetime(dt_df[col], errors="coerce", utc=True)
        arr = s.view("int64").astype("float64")
        arr[arr == np.iinfo("int64").min] = np.nan
        out_df[col] = arr / 86_400_000_000_000.0
    return out_df


def drop_all_nan_train_columns(
    X_tr_df: pd.DataFrame,
    *others: pd.DataFrame
) -> Tuple[pd.DataFrame, List[pd.DataFrame], List[str]]:
    all_nan_cols = [c for c in X_tr_df.columns if X_tr_df[c].isna().all()]

    if all_nan_cols:
        X_tr_df = X_tr_df.drop(columns=all_nan_cols)
        new_others = []
        for other_df in others:
            new_others.append(other_df.drop(columns=[c for c in all_nan_cols if c in other_df.columns]))
        return X_tr_df, new_others, all_nan_cols

    return X_tr_df, list(others), []


def clean_cve_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    cve_df = raw_df.copy()
    cve_df.columns = cve_df.columns.str.strip()

    if "cve_id" not in cve_df.columns:
        for col in cve_df.columns:
            cl = col.lower()
            if "cve" in cl and "id" in cl:
                cve_df["cve_id"] = cve_df[col]
                break

    cve_df["cve_id"] = (
        _safe_series(cve_df, "cve_id")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
    )

    date_cols = [
        c for c in cve_df.columns
        if any(k in c.lower() for k in ["publish", "published", "kev", "date", "added"])
    ]
    for col in date_cols:
        cve_df[col] = pd.to_datetime(cve_df[col], errors="coerce", utc=True)

    score_candidates = [
        "nvd_base_score", "jvn_base_score", "eu_base_score",
        "nvd_cvss_score",
        "base_score", "cvss_basescore", "cvss_score", "baseScore",
    ]
    present = [c for c in score_candidates if c in cve_df.columns]
    for col in present:
        cve_df[col] = pd.to_numeric(cve_df[col], errors="coerce")

    if present:
        ordered = [c for c in ["nvd_base_score", "jvn_base_score", "eu_base_score"] if c in present] + \
                  [c for c in present if c not in {"nvd_base_score", "jvn_base_score", "eu_base_score"}]
        cve_df["base_score"] = cve_df[ordered].bfill(axis=1).iloc[:, 0]
    else:
        cve_df["base_score"] = np.nan

    kev_cols = [
        c for c in [
            "is_kev", "kev_present", "is_known_exploited",
            "kev_published", "dateAdded", "dateadded", "kev_listed",
        ]
        if c in cve_df.columns
    ]

    def _row_is_kev(row):
        for col in kev_cols:
            v = row.get(col, np.nan)
            if pd.isna(v):
                continue
            if isinstance(v, (bool, np.bool_)) and v:
                return True
            if isinstance(v, (int, float, np.integer, np.floating)) and v == 1:
                return True
            if isinstance(v, pd.Timestamp):
                return True
        return False

    cve_df["is_kev"] = cve_df.apply(_row_is_kev, axis=1).astype(int) if kev_cols else 0

    cve_df["desc_len"] = _safe_series(cve_df, "description", "").astype(str).str.len()

    if "references_count" in cve_df.columns:
        cve_df["references_count"] = (
            pd.to_numeric(cve_df["references_count"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    else:
        refs_text = _safe_series(cve_df, "references", "")
        cve_df["references_count"] = (
            refs_text.astype(str).str.count(r"https?://").fillna(0).astype(int)
        )

    cve_df["cve_year"] = (
        cve_df["cve_id"]
        .astype(str)
        .str.extract(r"^CVE-(\d{4})-", expand=False)
        .astype(float)
    )

    return cve_df.drop_duplicates(subset="cve_id")


CVSS_V3_KEYS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]


def parse_cvss_v3_vector(vec: str) -> Dict[str, str]:
    out = {k: np.nan for k in CVSS_V3_KEYS}
    if not isinstance(vec, str) or not vec:
        return out

    s = vec.strip()
    if s.upper().startswith("CVSS:"):
        parts = s.split("/", 1)
        s = parts[1] if len(parts) > 1 else ""

    tokens = s.split("/")
    for tok in tokens:
        if ":" not in tok:
            continue
        k, v = tok.split(":", 1)
        k = k.strip().upper()
        v = v.strip().upper()
        if k in out:
            out[k] = v
    return out


def add_cvss_vector_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    feat_df = feat_df.copy()

    vec_col = _first_present(feat_df, [
        "cvss_vector", "nvd_vector", "vectorString",
        "cvss_v3_vector", "cvssVector"
    ])

    for k in CVSS_V3_KEYS:
        feat_df[f"cvss_{k.lower()}"] = np.nan
    feat_df["cvss_v3_present"] = 0
    for nm in ["av", "ac", "pr", "ui", "s", "c", "i", "a"]:
        feat_df[f"cvss_{nm}_ord"] = np.nan

    if not vec_col:
        return feat_df

    parsed = feat_df[vec_col].apply(parse_cvss_v3_vector).apply(pd.Series)

    for k in CVSS_V3_KEYS:
        feat_df[f"cvss_{k.lower()}"] = parsed.get(k, np.nan)

    feat_df["cvss_v3_present"] = parsed.notna().any(axis=1).astype(int)

    av_map = {"N": 3, "A": 2, "L": 1, "P": 0}
    ac_map = {"L": 1, "H": 0}
    pr_map = {"N": 2, "L": 1, "H": 0}
    ui_map = {"N": 1, "R": 0}
    s_map = {"C": 1, "U": 0}
    impact_map = {"H": 2, "L": 1, "N": 0}

    feat_df["cvss_av_ord"] = feat_df["cvss_av"].map(av_map)
    feat_df["cvss_ac_ord"] = feat_df["cvss_ac"].map(ac_map)
    feat_df["cvss_pr_ord"] = feat_df["cvss_pr"].map(pr_map)
    feat_df["cvss_ui_ord"] = feat_df["cvss_ui"].map(ui_map)
    feat_df["cvss_s_ord"] = feat_df["cvss_s"].map(s_map)
    feat_df["cvss_c_ord"] = feat_df["cvss_c"].map(impact_map)
    feat_df["cvss_i_ord"] = feat_df["cvss_i"].map(impact_map)
    feat_df["cvss_a_ord"] = feat_df["cvss_a"].map(impact_map)

    return feat_df


def add_epss_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    feat_df = feat_df.copy()

    epss_score_col = _first_present(feat_df, ["epss_score", "epss", "first_epss_score"])
    epss_pct_col = _first_present(feat_df, ["epss_percentile", "first_epss_percentile"])

    feat_df["epss_score"] = pd.to_numeric(feat_df[epss_score_col], errors="coerce") if epss_score_col else np.nan
    feat_df["epss_percentile"] = pd.to_numeric(feat_df[epss_pct_col], errors="coerce") if epss_pct_col else np.nan
    feat_df["epss_present"] = (
        pd.Series(feat_df["epss_score"]).notna() | pd.Series(feat_df["epss_percentile"]).notna()
    ).astype(int)

    return feat_df


def add_poc_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    feat_df = feat_df.copy()

    github_col = _first_present(feat_df, ["github_poc_count", "poc_github_count", "github_pocs"])
    exploitdb_col = _first_present(feat_df, ["exploitdb_present", "exploit_db_present"])
    metasploit_col = _first_present(feat_df, ["metasploit_present"])
    poc_present_col = _first_present(feat_df, ["poc_present", "exploit_present"])

    feat_df["github_poc_count"] = pd.to_numeric(feat_df[github_col], errors="coerce").fillna(0) if github_col else 0
    feat_df["exploitdb_present"] = pd.to_numeric(feat_df[exploitdb_col], errors="coerce").fillna(0).astype(int) if exploitdb_col else 0
    feat_df["metasploit_present"] = pd.to_numeric(feat_df[metasploit_col], errors="coerce").fillna(0).astype(int) if metasploit_col else 0
    feat_df["poc_present"] = pd.to_numeric(feat_df[poc_present_col], errors="coerce").fillna(0).astype(int) if poc_present_col else 0

    refs = _safe_series(feat_df, "references", "").astype(str)

    github_hits = refs.str.count(r"(?:https?://)?(?:www\.)?github\.com/")
    exploitdb_hits = refs.str.count(r"(?:https?://)?(?:www\.)?exploit-db\.com/")
    metasploit_hits = refs.str.count(r"(?:https?://)?(?:www\.)?metasploit\.com/")

    if not github_col:
        feat_df["github_poc_count"] = github_hits.fillna(0).astype(int)
    if not exploitdb_col:
        feat_df["exploitdb_present"] = (exploitdb_hits.fillna(0) > 0).astype(int)
    if not metasploit_col:
        feat_df["metasploit_present"] = (metasploit_hits.fillna(0) > 0).astype(int)

    feat_df["poc_url_count"] = (
        github_hits.fillna(0) + exploitdb_hits.fillna(0) + metasploit_hits.fillna(0)
    ).astype(int)

    feat_df["poc_any_present"] = (
        (feat_df["github_poc_count"] > 0) |
        (feat_df["exploitdb_present"] == 1) |
        (feat_df["metasploit_present"] == 1) |
        (feat_df["poc_present"] == 1)
    ).astype(int)

    fpoc_col = _first_present(feat_df, ["first_poc_date", "poc_first_date", "first_exploit_date"])
    feat_df["first_poc_date"] = _to_utc(feat_df[fpoc_col]) if fpoc_col else pd.NaT

    if "published_date" in feat_df.columns:
        first_poc = _to_utc(feat_df["first_poc_date"])
        pub = _to_utc(feat_df["published_date"])
        lag = (first_poc - pub).dt.total_seconds() / 86400.0
        feat_df["first_poc_lag_days"] = lag
        feat_df["poc_within_30d"] = ((lag >= 0) & (lag <= 30)).astype(int)
    else:
        feat_df["first_poc_lag_days"] = np.nan
        feat_df["poc_within_30d"] = 0

    return feat_df


def build_feature_set(
    cve_df: pd.DataFrame,
    ref_time: Optional[pd.Timestamp] = None,
    use_epss: bool = True,
    use_poc: bool = True,
) -> pd.DataFrame:
    feat_df = cve_df.copy()

    if ref_time is None:
        ref_time = pd.Timestamp.utcnow().tz_localize("UTC")

    if {"kev_published", "published_date"} <= set(feat_df.columns):
        feat_df["days_to_kev"] = (
            pd.to_datetime(feat_df["kev_published"], errors="coerce", utc=True)
            - pd.to_datetime(feat_df["published_date"], errors="coerce", utc=True)
        ).dt.total_seconds() / 86400.0
    else:
        feat_df["days_to_kev"] = np.nan

    pub_cols = [c for c in ["nvd_published", "jvn_published", "eu_published"] if c in feat_df.columns]
    if pub_cols:
        pub_dates = feat_df[pub_cols].apply(pd.to_datetime, errors="coerce", utc=True)
        max_dt = pub_dates.max(axis=1)
        min_dt = pub_dates.min(axis=1)
        feat_df["repo_publication_lag"] = (max_dt - min_dt).dt.total_seconds() / 86400.0
    else:
        feat_df["repo_publication_lag"] = np.nan

    feat_df["update_frequency"] = feat_df["references_count"].astype(float)

    if "first_reference_date" in feat_df.columns:
        fr = pd.to_datetime(feat_df["first_reference_date"], errors="coerce", utc=True)
        feat_df["time_since_first_reference"] = (ref_time - fr).dt.total_seconds() / 86400.0
    else:
        feat_df["time_since_first_reference"] = np.nan

    src_flags = [c for c in feat_df.columns if c.endswith("_present") and "kev" not in c.lower()]
    if src_flags:
        feat_df[src_flags] = feat_df[src_flags].astype(float).fillna(0).astype(int)
        feat_df["cross_listing_count"] = feat_df[src_flags].sum(axis=1)

        if pub_cols:
            pub_dates = feat_df[pub_cols].apply(pd.to_datetime, errors="coerce", utc=True)
            days = _dtframe_to_days(pub_dates)
            feat_df["cross_listing_std_days"] = days.std(axis=1, skipna=True)
            feat_df["cross_listing_variance"] = days.var(axis=1, skipna=True)
        else:
            feat_df["cross_listing_std_days"] = np.nan
            feat_df["cross_listing_variance"] = np.nan
    else:
        feat_df["cross_listing_count"] = 0
        feat_df["cross_listing_std_days"] = np.nan
        feat_df["cross_listing_variance"] = np.nan

    if "cwes" in feat_df.columns:
        feat_df["cwe_category"] = feat_df["cwes"].astype(str).str.extract(r"(CWE-\d+)", expand=False)
        feat_df["related_cwe_count"] = feat_df["cwes"].astype(str).str.count(r"CWE-\d+")
        cwe_counts = feat_df["cwe_category"].value_counts(dropna=True)
        feat_df["weakness_frequency"] = feat_df["cwe_category"].map(cwe_counts)

        if "is_kev" in feat_df.columns:
            kev_rate = feat_df.groupby("cwe_category")["is_kev"].mean().to_dict()
            feat_df["cwe_risk_factor"] = feat_df["cwe_category"].map(kev_rate)
        else:
            feat_df["cwe_risk_factor"] = np.nan
    else:
        feat_df["cwe_category"] = np.nan
        feat_df["related_cwe_count"] = np.nan
        feat_df["weakness_frequency"] = np.nan
        feat_df["cwe_risk_factor"] = np.nan

    if "vendor" in feat_df.columns:
        vendor_counts = feat_df["vendor"].value_counts(dropna=True)
        feat_df["vendor_frequency"] = feat_df["vendor"].map(vendor_counts)

        if "is_kev" in feat_df.columns:
            vendor_kev_rate = feat_df.groupby("vendor")["is_kev"].mean().to_dict()
            feat_df["vendor_risk_factor"] = feat_df["vendor"].map(vendor_kev_rate)
        else:
            feat_df["vendor_risk_factor"] = np.nan
    else:
        feat_df["vendor_frequency"] = np.nan
        feat_df["vendor_risk_factor"] = np.nan

    desc_series = feat_df["description"] if "description" in feat_df.columns else pd.Series([""] * len(feat_df), index=feat_df.index)
    feat_df["word_count"] = desc_series.astype(str).str.split().apply(len)

    feat_df = add_cvss_vector_features(feat_df)

    if use_epss:
        feat_df = add_epss_features(feat_df)

    if use_poc:
        feat_df = add_poc_features(feat_df)

    return feat_df


def get_model_feature_df(feat_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "cve_id",
        "repo_publication_lag",
        "update_frequency",
        "time_since_first_reference",
        "cross_listing_count",
        "cross_listing_std_days",
        "cross_listing_variance",
        "related_cwe_count",
        "weakness_frequency",
        "vendor_frequency",
        "word_count",
        "desc_len",
        "base_score",
        "cve_year",
        "epss_score",
        "epss_percentile",
        "epss_present",
        "github_poc_count",
        "exploitdb_present",
        "metasploit_present",
        "poc_url_count",
        "poc_any_present",
        "first_poc_lag_days",
        "poc_within_30d",
        "cvss_v3_present",
        "cvss_av_ord",
        "cvss_ac_ord",
        "cvss_pr_ord",
        "cvss_ui_ord",
        "cvss_s_ord",
        "cvss_c_ord",
        "cvss_i_ord",
        "cvss_a_ord",
        "is_kev",
        "cwe_category",
    ]

    cols = [c for c in cols if c in feat_df.columns]
    feat_small_df = feat_df[cols].copy()

    fill_zero = [
        "vendor_frequency", "word_count", "related_cwe_count", "weakness_frequency",
        "cross_listing_count", "github_poc_count", "poc_url_count", "poc_within_30d",
        "epss_present", "exploitdb_present", "metasploit_present", "poc_any_present",
        "cvss_v3_present",
    ]
    for col in fill_zero:
        if col in feat_small_df.columns:
            feat_small_df[col] = pd.to_numeric(feat_small_df[col], errors="coerce").fillna(0)

    return feat_small_df


def apply_leakage_guard(feat_cols: List[str]) -> List[str]:
    patterns = [
        "days_to_kev",
        "lead_days",
        "time_to_kev",
        "cwe_risk_factor",
        "vendor_risk_factor",
        "kev_published",
        "dateadded",
        "kev_listed",
        "kev_present",
        "is_kev",
        "is_known_exploited",
        "known_exploited",
    ]

    def is_leaky(col):
        cl = col.lower()
        return any(p in cl for p in patterns)

    return [c for c in feat_cols if not is_leaky(c)]


def compute_metrics(y_true, y_score) -> Dict[str, float]:
    out = {"auroc": np.nan, "auprc": np.nan}
    if len(np.unique(y_true)) > 1:
        out["auroc"] = roc_auc_score(y_true, y_score)
        out["auprc"] = average_precision_score(y_true, y_score)
    return out


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if k <= 0:
        return np.nan
    k = min(k, len(y_true))
    if k == 0:
        return np.nan
    order = np.argsort(-y_score)
    topk = y_true[order][:k]
    return float(np.sum(topk == 1) / k)


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    total_pos = int(np.sum(y_true == 1))
    if total_pos == 0 or k <= 0:
        return np.nan
    k = min(k, len(y_true))
    order = np.argsort(-y_score)
    topk = y_true[order][:k]
    return float(np.sum(topk == 1) / total_pos)


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    k = min(k, len(relevances))
    if k <= 0:
        return 0.0
    rel = relevances[:k].astype(float)
    discounts = np.log2(np.arange(2, k + 2))
    return float(np.sum(rel / discounts))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if k <= 0:
        return np.nan
    k = min(k, len(y_true))
    if k == 0:
        return np.nan
    order = np.argsort(-y_score)
    rel_sorted = y_true[order]
    dcg = dcg_at_k(rel_sorted, k)
    ideal = np.sort(y_true)[::-1]
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return np.nan
    return float(dcg / idcg)


def compute_rank_metrics(y_true: np.ndarray, y_score: np.ndarray, k_list: List[int]) -> Dict[str, float]:
    out = {}
    for k in k_list:
        out[f"p@{k}"] = precision_at_k(y_true, y_score, k)
        out[f"r@{k}"] = recall_at_k(y_true, y_score, k)
        out[f"ndcg@{k}"] = ndcg_at_k(y_true, y_score, k)
    return out


def fit_xgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    spw: float,
    params: Dict,
) -> XGBClassifier:
    p = params.copy()
    tree_method = p.pop("tree_method", "hist")
    n_jobs = p.pop("n_jobs", -1)
    random_state = p.pop("random_state", 42)

    model = XGBClassifier(
        **p,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method=tree_method,
        n_jobs=n_jobs,
        random_state=random_state,
        scale_pos_weight=spw,
        verbosity=0,
    )
    model.fit(X_tr, y_tr)
    return model


def quick_tune_params() -> List[Dict]:
    grids = []
    for md in [3, 5]:
        for lr in [0.05, 0.1]:
            for ne in [300, 600]:
                grids.append({
                    "max_depth": md,
                    "learning_rate": lr,
                    "n_estimators": ne,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 1,
                    "reg_lambda": 1.0,
                    "gamma": 0.0,
                    "tree_method": "hist",
                    "n_jobs": -1,
                    "random_state": 42,
                })
    return grids


def fit_logreg(X_tr_z: np.ndarray, y_tr: np.ndarray, seed: int = 42) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )
    model.fit(X_tr_z, y_tr)
    return model


def fit_rf(X_tr: np.ndarray, y_tr: np.ndarray, seed: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=seed,
    )
    model.fit(X_tr, y_tr)
    return model


class PULogRegBagging:
    def __init__(self, n_bags: int = 30, sample_ratio: float = 1.0, seed: int = 42):
        self.n_bags = n_bags
        self.sample_ratio = sample_ratio
        self.seed = seed
        self.models: List[LogisticRegression] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        pos_idx = np.where(y == 1)[0]
        unl_idx = np.where(y == 0)[0]

        n_pos = len(pos_idx)
        if n_pos == 0 or len(unl_idx) == 0:
            self.models = []
            return self

        take = max(1, int(n_pos * self.sample_ratio))

        self.models = []
        for b in range(self.n_bags):
            sampled_unl = rng.choice(unl_idx, size=min(take, len(unl_idx)), replace=False)

            bag_idx = np.concatenate([pos_idx, sampled_unl])
            bag_y = np.concatenate([np.ones(len(pos_idx), dtype=int), np.zeros(len(sampled_unl), dtype=int)])

            perm = rng.permutation(len(bag_idx))
            bag_idx = bag_idx[perm]
            bag_y = bag_y[perm]

            m = LogisticRegression(
                max_iter=2000,
                class_weight=None,
                solver="lbfgs",
                random_state=int(self.seed + b),
            )
            m.fit(X[bag_idx], bag_y)
            self.models.append(m)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            out = np.zeros((X.shape[0], 2), dtype=float)
            out[:, 1] = 0.0
            out[:, 0] = 1.0
            return out

        probs = np.zeros(X.shape[0], dtype=float)
        for m in self.models:
            probs += m.predict_proba(X)[:, 1]
        probs /= len(self.models)

        out = np.zeros((X.shape[0], 2), dtype=float)
        out[:, 1] = probs
        out[:, 0] = 1.0 - probs
        return out


def make_scrambled_noise(X: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Xn = X.copy()
    n, d = Xn.shape
    for j in range(d):
        perm = rng.permutation(n)
        Xn[:, j] = Xn[perm, j]
    return Xn


def fit_kde_triplet(
    X_tr_kde: np.ndarray,
    y_tr: np.ndarray,
    bandwidth: float,
    max_nonkev: int,
    noise_ratio: float = 1.0,
    seed: int = 42,
) -> Tuple[KernelDensity, KernelDensity, KernelDensity, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_tr == 1)[0]
    neg_idx = np.where(y_tr == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("KDE3 requires both positive and negative samples in TRAIN.")

    if max_nonkev is not None and max_nonkev > 0 and len(neg_idx) > max_nonkev:
        neg_idx_used = rng.choice(neg_idx, size=max_nonkev, replace=False)
    else:
        neg_idx_used = neg_idx

    X_pos = X_tr_kde[pos_idx]
    X_neg = X_tr_kde[neg_idx_used]

    X_scrambled = make_scrambled_noise(X_tr_kde, seed=seed + 13)

    if noise_ratio is None or noise_ratio <= 0:
        noise_ratio = 1.0

    target_noise_n = int(len(neg_idx_used) * noise_ratio)
    if target_noise_n <= 0:
        target_noise_n = len(neg_idx_used)

    if target_noise_n < len(X_scrambled):
        noise_idx = rng.choice(np.arange(len(X_scrambled)), size=target_noise_n, replace=False)
        X_noise = X_scrambled[noise_idx]
    else:
        X_noise = X_scrambled

    kde_p = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X_pos)
    kde_n = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X_neg)
    kde_z = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X_noise)

    return kde_p, kde_n, kde_z, neg_idx_used


def kde_risk_scores_3class(
    kde_p: KernelDensity,
    kde_n: KernelDensity,
    kde_z: KernelDensity,
    Xk: np.ndarray,
) -> np.ndarray:
    lp = kde_p.score_samples(Xk)
    ln = kde_n.score_samples(Xk)
    lz = kde_z.score_samples(Xk)

    M = np.maximum(np.maximum(lp, ln), lz)
    ep = np.exp(lp - M)
    en = np.exp(ln - M)
    ez = np.exp(lz - M)

    denom = ep + en + ez
    denom = np.where(denom == 0, 1.0, denom)

    return ep / denom


def run_pipeline(args, ablation_name: str = "") -> pd.DataFrame:
    enabled_models = set([m.lower().strip() for m in args.models])

    k_list = [k for k in args.k_values if isinstance(k, int) and k > 0]
    if not k_list:
        k_list = [100]

    if args.train_min_year > args.train_max_year:
        raise ValueError("--train-min-year cannot be greater than --train-max-year")

    raw_df = pd.read_csv(args.input, low_memory=False)

    cve_df = clean_cve_df(raw_df)

    if "cve_year" not in cve_df.columns:
        raise ValueError("cve_year not found; required for time split.")

    ref_time = pd.Timestamp(f"{args.train_max_year}-12-31", tz="UTC")

    feat_df = build_feature_set(
        cve_df,
        ref_time=ref_time,
        use_epss=not args.disable_epss,
        use_poc=not args.disable_poc,
    )

    feat_small_df = get_model_feature_df(feat_df)

    for drop_name in args.drop_features:
        if drop_name in feat_small_df.columns:
            feat_small_df = feat_small_df.drop(columns=[drop_name])

    year_mask = feat_small_df["cve_year"].notna() & (feat_small_df["cve_year"] >= args.train_min_year)
    pool_df = feat_small_df[year_mask].copy()

    if pool_df.empty:
        raise ValueError("No rows left in modeling pool after applying train-min-year filter.")

    mask_train = (pool_df["cve_year"] >= args.train_min_year) & (pool_df["cve_year"] <= args.train_max_year)
    mask_test = pool_df["cve_year"] > args.train_max_year

    if mask_test.sum() == 0:
        rng = np.random.default_rng(42)
        idx_all = pool_df.index.to_numpy()
        rng.shuffle(idx_all)
        split = int(0.8 * len(idx_all))
        idx_train = idx_all[:split]
        idx_test = idx_all[split:]
        mask_train = pool_df.index.isin(idx_train)
        mask_test = pool_df.index.isin(idx_test)

    df_train = pool_df[mask_train].copy()
    df_test = pool_df[mask_test].copy()

    if "is_kev" not in df_train.columns:
        raise ValueError("is_kev missing in feature frame.")

    y_tr = df_train["is_kev"].astype(int).values
    y_te = df_test["is_kev"].astype(int).values

    num_cols = pd.concat([df_train, df_test], axis=0).select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c != "is_kev"]
    feat_cols = apply_leakage_guard(feat_cols)

    X_tr_df = df_train[feat_cols].copy()
    X_te_df = df_test[feat_cols].copy()

    X_tr_df, [X_te_df], _ = drop_all_nan_train_columns(X_tr_df, X_te_df)

    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_tr_df)
    X_te = imp.transform(X_te_df)

    feat_cols = X_tr_df.columns.tolist()

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr_z = scaler.fit_transform(X_tr)
    X_te_z = scaler.transform(X_te)

    n_pos = max(1, int(y_tr.sum()))
    n_neg = max(1, int((y_tr == 0).sum()))
    spw = n_neg / n_pos
    pos_rate = float(y_tr.mean())

    model_results: Dict[str, Dict] = {}

    if "base_only" in enabled_models:
        if "base_score" in df_train.columns and "base_score" in df_test.columns:
            tr_base_raw = pd.to_numeric(df_train["base_score"], errors="coerce").values
            te_base_raw = pd.to_numeric(df_test["base_score"], errors="coerce").values

            tr_median = np.nanmedian(tr_base_raw) if np.isfinite(np.nanmedian(tr_base_raw)) else 0.0
            tr_base = np.where(np.isnan(tr_base_raw), tr_median, tr_base_raw)
            te_base = np.where(np.isnan(te_base_raw), tr_median, te_base_raw)

            m_tr = compute_metrics(y_tr, tr_base)
            m_te = compute_metrics(y_te, te_base)

            rank_tr = compute_rank_metrics(y_tr, tr_base, k_list)
            rank_te = compute_rank_metrics(y_te, te_base, k_list)

            model_results["base_only"] = {
                "model": None,
                "train_prob": tr_base,
                "test_prob": te_base,
                "m_tr": m_tr,
                "m_te": m_te,
                "rank_tr": rank_tr,
                "rank_te": rank_te,
                "train_base_raw": tr_base_raw,
                "test_base_raw": te_base_raw,
            }

    if "kde" in enabled_models:
        Xk_tr = X_tr_z
        Xk_te = X_te_z
        pca_kde = None

        if args.kde_use_pca:
            n_comp = max(2, int(args.kde_pca_components))
            n_comp = min(n_comp, Xk_tr.shape[1])
            pca_kde = PCA(n_components=n_comp, random_state=42)
            Xk_tr = pca_kde.fit_transform(Xk_tr)
            Xk_te = pca_kde.transform(Xk_te)

        kde_p, kde_n, kde_z, _ = fit_kde_triplet(
            Xk_tr,
            y_tr,
            bandwidth=args.kde_bandwidth,
            max_nonkev=args.kde_max_nonkev_train,
            noise_ratio=args.kde_noise_ratio,
            seed=42,
        )

        tr_prob = kde_risk_scores_3class(kde_p, kde_n, kde_z, Xk_tr)
        te_prob = kde_risk_scores_3class(kde_p, kde_n, kde_z, Xk_te)

        m_tr = compute_metrics(y_tr, tr_prob)
        m_te = compute_metrics(y_te, te_prob)

        rank_tr = compute_rank_metrics(y_tr, tr_prob, k_list)
        rank_te = compute_rank_metrics(y_te, te_prob, k_list)

        model_results["kde"] = {
            "model": (kde_p, kde_n, kde_z),
            "train_prob": tr_prob,
            "test_prob": te_prob,
            "m_tr": m_tr,
            "m_te": m_te,
            "rank_tr": rank_tr,
            "rank_te": rank_te,
            "pca": pca_kde,
        }

    if "xgb" in enabled_models:
        xgb_default = {
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 600,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
            "gamma": 0.0,
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42,
        }
        xgb_best = xgb_default

        if args.quick_tune and "cve_year" in df_train.columns and not df_train.empty:
            max_year = int(np.nanmax(df_train["cve_year"]))
            inner_tr_df = df_train[df_train["cve_year"] < max_year].copy()
            inner_va_df = df_train[df_train["cve_year"] == max_year].copy()

            if not inner_tr_df.empty and not inner_va_df.empty:
                inner_y_tr = inner_tr_df["is_kev"].astype(int).values
                inner_y_va = inner_va_df["is_kev"].astype(int).values

                inner_X_tr_df = inner_tr_df[feat_cols].copy()
                inner_X_va_df = inner_va_df[feat_cols].copy()

                inner_X_tr_df, [inner_X_va_df], _ = drop_all_nan_train_columns(
                    inner_X_tr_df, inner_X_va_df
                )

                inner_imp = SimpleImputer(strategy="median")
                inner_X_tr = inner_imp.fit_transform(inner_X_tr_df)
                inner_X_va = inner_imp.transform(inner_X_va_df)

                pos_i = max(1, int(inner_y_tr.sum()))
                neg_i = max(1, int((inner_y_tr == 0).sum()))
                spw_i = neg_i / pos_i

                best_ap = -1.0
                for p in quick_tune_params():
                    m = fit_xgb(inner_X_tr, inner_y_tr, spw_i, p)
                    va_prob = m.predict_proba(inner_X_va)[:, 1]
                    ap = average_precision_score(inner_y_va, va_prob) if len(np.unique(inner_y_va)) > 1 else np.nan
                    if np.isfinite(ap) and ap > best_ap:
                        best_ap = ap
                        xgb_best = p

        xgb_model = fit_xgb(X_tr, y_tr, spw, xgb_best)

        tr_prob = xgb_model.predict_proba(X_tr)[:, 1]
        te_prob = xgb_model.predict_proba(X_te)[:, 1]

        m_tr = compute_metrics(y_tr, tr_prob)
        m_te = compute_metrics(y_te, te_prob)

        rank_tr = compute_rank_metrics(y_tr, tr_prob, k_list)
        rank_te = compute_rank_metrics(y_te, te_prob, k_list)

        model_results["xgb"] = {
            "model": xgb_model,
            "train_prob": tr_prob,
            "test_prob": te_prob,
            "m_tr": m_tr,
            "m_te": m_te,
            "rank_tr": rank_tr,
            "rank_te": rank_te,
        }

        try:
            booster = xgb_model.get_booster()
            score = booster.get_score(importance_type="gain")
            fi_df = (
                pd.DataFrame({"feature": list(score.keys()), "gain": list(score.values())})
                .sort_values("gain", ascending=False)
            )
            fi_path = args.output.replace(".csv", "_xgb_feature_importance.csv")
            fi_df.to_csv(fi_path, index=False)
        except Exception:
            pass

    if "logreg" in enabled_models:
        lr_model = fit_logreg(X_tr_z, y_tr)

        tr_prob = lr_model.predict_proba(X_tr_z)[:, 1]
        te_prob = lr_model.predict_proba(X_te_z)[:, 1]

        m_tr = compute_metrics(y_tr, tr_prob)
        m_te = compute_metrics(y_te, te_prob)

        rank_tr = compute_rank_metrics(y_tr, tr_prob, k_list)
        rank_te = compute_rank_metrics(y_te, te_prob, k_list)

        model_results["logreg"] = {
            "model": lr_model,
            "train_prob": tr_prob,
            "test_prob": te_prob,
            "m_tr": m_tr,
            "m_te": m_te,
            "rank_tr": rank_tr,
            "rank_te": rank_te,
        }

        try:
            coef = lr_model.coef_.ravel()
            lr_coef_df = pd.DataFrame({"feature": feat_cols, "coef": coef})
            lr_coef_df["abs_coef"] = np.abs(lr_coef_df["coef"])
            lr_coef_df = lr_coef_df.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])
            lr_path = args.output.replace(".csv", "_logreg_coefficients.csv")
            lr_coef_df.to_csv(lr_path, index=False)
        except Exception:
            pass

    if "rf" in enabled_models:
        rf_model = fit_rf(X_tr, y_tr)

        tr_prob = rf_model.predict_proba(X_tr)[:, 1]
        te_prob = rf_model.predict_proba(X_te)[:, 1]

        m_tr = compute_metrics(y_tr, tr_prob)
        m_te = compute_metrics(y_te, te_prob)

        rank_tr = compute_rank_metrics(y_tr, tr_prob, k_list)
        rank_te = compute_rank_metrics(y_te, te_prob, k_list)

        model_results["rf"] = {
            "model": rf_model,
            "train_prob": tr_prob,
            "test_prob": te_prob,
            "m_tr": m_tr,
            "m_te": m_te,
            "rank_tr": rank_tr,
            "rank_te": rank_te,
        }

        try:
            impv = rf_model.feature_importances_
            rf_imp_df = pd.DataFrame({"feature": feat_cols, "importance": impv}).sort_values("importance", ascending=False)
            rf_path = args.output.replace(".csv", "_rf_feature_importance.csv")
            rf_imp_df.to_csv(rf_path, index=False)
        except Exception:
            pass

    if "pu" in enabled_models:
        pu_model = PULogRegBagging(
            n_bags=args.pu_bags,
            sample_ratio=args.pu_sample_ratio,
            seed=42
        )
        pu_model.fit(X_tr_z, y_tr)

        tr_prob = pu_model.predict_proba(X_tr_z)[:, 1]
        te_prob = pu_model.predict_proba(X_te_z)[:, 1]

        m_tr = compute_metrics(y_tr, tr_prob)
        m_te = compute_metrics(y_te, te_prob)

        rank_tr = compute_rank_metrics(y_tr, tr_prob, k_list)
        rank_te = compute_rank_metrics(y_te, te_prob, k_list)

        model_results["pu"] = {
            "model": pu_model,
            "train_prob": tr_prob,
            "test_prob": te_prob,
            "m_tr": m_tr,
            "m_te": m_te,
            "rank_tr": rank_tr,
            "rank_te": rank_te,
        }

    for key in ["xgb", "logreg", "rf", "pu", "kde", "base_only"]:
        feat_df[f"risk_score_{key}"] = np.nan

    for name, obj in model_results.items():
        if name == "base_only":
            feat_df.loc[df_train.index, "risk_score_base_only"] = obj.get("train_base_raw", np.nan)
            feat_df.loc[df_test.index, "risk_score_base_only"] = obj.get("test_base_raw", np.nan)
        else:
            feat_df.loc[df_train.index, f"risk_score_{name}"] = obj["train_prob"]
            feat_df.loc[df_test.index, f"risk_score_{name}"] = obj["test_prob"]

        feat_df[f"{name}_auroc_train"] = obj["m_tr"]["auroc"]
        feat_df[f"{name}_auprc_train"] = obj["m_tr"]["auprc"]
        feat_df[f"{name}_auroc_test"] = obj["m_te"]["auroc"]
        feat_df[f"{name}_auprc_test"] = obj["m_te"]["auprc"]

    feat_df["train_prevalence"] = pos_rate
    feat_df["train_min_year_used"] = args.train_min_year
    feat_df["train_max_year_used"] = args.train_max_year
    feat_df["ablation_name"] = ablation_name or "single"

    rows = []
    for name, obj in model_results.items():
        rows.append({
            "ablation": ablation_name or "single",
            "model": name,
            "split": "train",
            "train_min_year": args.train_min_year,
            "train_max_year": args.train_max_year,
            "auroc": obj["m_tr"].get("auroc", np.nan),
            "auprc": obj["m_tr"].get("auprc", np.nan),
            **obj.get("rank_tr", {})
        })
        rows.append({
            "ablation": ablation_name or "single",
            "model": name,
            "split": "test",
            "train_min_year": args.train_min_year,
            "train_max_year": args.train_max_year,
            "auroc": obj["m_te"].get("auroc", np.nan),
            "auprc": obj["m_te"].get("auprc", np.nan),
            **obj.get("rank_te", {})
        })

    metrics_out_df = pd.DataFrame(rows)
    metrics_path = args.output.replace(".csv", "_ranking_metrics.csv")
    metrics_out_df.to_csv(metrics_path, index=False)

    feat_df.to_csv(args.output, index=False)

    return metrics_out_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", "-i", default="data/clean/combined_master.csv")
    parser.add_argument("--output", "-o", default="data/model/model_scores.csv")

    parser.add_argument("--train-min-year", type=int, default=0)
    parser.add_argument("--train-max-year", type=int, default=2024)

    parser.add_argument("--quick-tune", action="store_true")
    parser.add_argument("--drop-features", nargs="*", default=[])

    parser.add_argument("--disable-epss", action="store_true")
    parser.add_argument("--disable-poc", action="store_true")

    parser.add_argument(
        "--models",
        nargs="*",
        default=["xgb", "logreg", "rf", "pu", "kde", "base_only"],
    )

    parser.add_argument("--pu-bags", type=int, default=30)
    parser.add_argument("--pu-sample-ratio", type=float, default=1.0)

    parser.add_argument(
        "--k-values",
        nargs="*",
        type=int,
        default=[50, 100, 200, 500, 1000],
    )

    parser.add_argument("--kde-bandwidth", type=float, default=0.5)
    parser.add_argument("--kde-max-nonkev-train", type=int, default=20000)
    parser.add_argument("--kde-use-pca", action="store_true")
    parser.add_argument("--kde-pca-components", type=int, default=20)
    parser.add_argument("--kde-noise-ratio", type=float, default=1.0)

    parser.add_argument("--ablation-suite", action="store_true")

    args = parser.parse_args()

    if not args.ablation_suite:
        run_pipeline(args, ablation_name="")
        return

    base_output = args.output
    all_metrics = []

    for cfg in ABLATION_CONFIGS:
        ab_name = cfg["name"]

        a = argparse.Namespace(**vars(args))

        a.disable_epss = bool(cfg.get("disable_epss", False))
        a.disable_poc = bool(cfg.get("disable_poc", False))

        extra_drop = cfg.get("drop", [])
        a.drop_features = list(a.drop_features) + list(extra_drop)

        a.output = with_ablation_suffix(base_output, ab_name)

        mdf = run_pipeline(a, ablation_name=ab_name)
        all_metrics.append(mdf)

    ablation_df = pd.concat(all_metrics, axis=0, ignore_index=True)
    suite_path = base_output.replace(".csv", "_ablation_suite_summary.csv")
    ablation_df.to_csv(suite_path, index=False)


if __name__ == "__main__":
    main()
