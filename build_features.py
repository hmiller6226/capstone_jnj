"""
Usage:
    python build_features.py \
        --input data/clean/combined_master.csv \
        --output data/clean/combined_with_features.csv
"""

import argparse
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_cve_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    def _col(name: str, fill=""):
        return df[name] if name in df.columns else pd.Series([fill] * len(df), index=df.index)

    if "cve_id" not in df.columns:
        for c in df.columns:
            if "cve" in c.lower() and "id" in c.lower():
                df["cve_id"] = df[c]
                break
    df["cve_id"] = (
        _col("cve_id")
        .astype(str)
        .str.upper()
        .str.extract(r"(CVE-\d{4}-\d+)", expand=False)
    )

    date_cols = [c for c in df.columns if ("date" in c.lower()) or ("published" in c.lower())]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    # Published date (best-effort)
    pub_candidates = [
        c
        for c in [
            "nvd_published",
            "jvn_published",
            "eu_published",
            "kev_published",
            "dateAdded",
            "dateadded",
            "published",
        ]
        if c in df.columns
    ]
    df["published_date"] = (
        df[pub_candidates].bfill(axis=1).iloc[:, 0] if pub_candidates else pd.NaT
    )

    score_cols = [
        c
        for c in [
            "base_score",
            "nvd_base_score",
            "jvn_base_score",
            "eu_base_score",
            "cvss_basescore",
            "cvss_score",
            "baseScore",
        ]
        if c in df.columns
    ]
    for c in score_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["base_score"] = (
        df[score_cols].bfill(axis=1).iloc[:, 0] if score_cols else np.nan
    )

    sev_cols = [c for c in df.columns if "severity" in c.lower()]
    if sev_cols:
        df["severity"] = (
            _col(sev_cols[0])
            .astype(str)
            .str.upper()
            .replace({"MODERATE": "MEDIUM"})
        )
    else:
        df["severity"] = np.nan

    kev_cols = [
        c
        for c in [
            "kev_present",
            "is_known_exploited",
            "kev_published",
            "dateAdded",
            "dateadded",
            "kev_listed",
        ]
        if c in df.columns
    ]

    def _row_is_kev(row):
        for c in kev_cols:
            v = row.get(c, np.nan)
            if pd.isna(v):
                continue
            if isinstance(v, (bool, np.bool_)) and v:
                return True
            if isinstance(v, (int, float, np.integer, np.floating)) and v == 1:
                return True
            if isinstance(v, pd.Timestamp):
                return True
            s = str(v).strip().lower()
            if s in {"true", "1", "yes"}:
                return True
            if re.match(r"20\d{2}-\d{2}-\d{2}", s):
                return True
        return False

    df["is_kev"] = df.apply(_row_is_kev, axis=1)

    df["desc_len"] = _col("description", "").astype(str).str.len()

    if "references_count" in df.columns:
        df["references_count"] = (
            pd.to_numeric(df["references_count"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    else:
        refs_text = _col("references", "")
        df["references_count"] = (
            refs_text.astype(str).str.count(r"https?://").fillna(0).astype(int)
        )

    df["cve_year"] = (
        df["cve_id"]
        .astype(str)
        .str.extract(r"^CVE-(\d{4})-")
        .astype(float)
    )

    df = df.drop_duplicates(subset="cve_id")

    return df


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    #days_to_kev
    if {"kev_published", "published_date"} <= set(df.columns):
        df["days_to_kev"] = (
            pd.to_datetime(df["kev_published"], errors="coerce", utc=True)
            - pd.to_datetime(df["published_date"], errors="coerce", utc=True)
        ).dt.total_seconds() / 86400.0
    else:
        df["days_to_kev"] = np.nan

    #repo_lag_time
    pub_cols = [c for c in ["nvd_published", "jvn_published", "eu_published"] if c in df.columns]
    if pub_cols:
        pub_dates = df[pub_cols].apply(pd.to_datetime, errors="coerce", utc=True)
        df["repo_publication_lag"] = (
            (pub_dates.max(axis=1) - pub_dates.min(axis=1))
            .dt.total_seconds()
            / 86400.0
        )
    else:
        df["repo_publication_lag"] = np.nan

    #update_frequency
    lastmod_col = None
    if "lastModified" in df.columns:
        lastmod_col = "lastModified"
    elif "lastmodified" in df.columns:
        lastmod_col = "lastmodified"

    if lastmod_col and "published_date" in df.columns:
        df["update_frequency"] = (
            pd.to_datetime(df[lastmod_col], errors="coerce", utc=True)
            - pd.to_datetime(df["published_date"], errors="coerce", utc=True)
        ).dt.total_seconds() / 86400.0
    else:
        df["update_frequency"] = np.nan

    # time_since_first_reference
    if {"references", "published_date"} <= set(df.columns):
        pub_dt = pd.to_datetime(df["published_date"], errors="coerce", utc=True)
        # Synthetic "first reference" within 0-29 days after publication
        ref_time = pub_dt + pd.to_timedelta(
            np.random.randint(0, 30, size=len(df)), unit="D"
        )
        df["time_since_first_reference"] = (ref_time - pub_dt).dt.days
    else:
        df["time_since_first_reference"] = np.nan

    # cross_listing
    src_flags = [c for c in df.columns if c.endswith("_present") and "kev" not in c.lower()]
    if src_flags:
        df["cross_listing_count"] = (
            df[src_flags].astype(float).fillna(0).astype(int).sum(axis=1)
        )

        pub_cols_var = [
            c for c in ["nvd_published", "jvn_published", "eu_published"] if c in df.columns
        ]
        if pub_cols_var:
            pub_dates = df[pub_cols_var].apply(pd.to_datetime, errors="coerce", utc=True)
            arr_ns = pub_dates.to_numpy(dtype="datetime64[ns]").astype("int64")
            # Replace NaT -> nan
            arr_ns = np.where(arr_ns == np.iinfo("int64").min, np.nan, arr_ns)
            arr_days = arr_ns / 86_400_000_000_000.0

            df["cross_listing_std_days"] = np.nanstd(arr_days, axis=1)
            df["cross_listing_variance"] = np.nanvar(arr_days, axis=1)
        else:
            df["cross_listing_std_days"] = np.nan
            df["cross_listing_variance"] = np.nan

        df["repo_coverage_vector"] = (
            df[src_flags].astype(float).fillna(0).astype(int).apply(
                lambda row: row.to_dict(), axis=1
            )
        )
    else:
        df["cross_listing_count"] = np.nan
        df["cross_listing_std_days"] = np.nan
        df["cross_listing_variance"] = np.nan
        df["repo_coverage_vector"] = [{}] * len(df)

    # CWE features
    if "cwes" in df.columns:
        df["cwe_category"] = df["cwes"].astype(str).str.extract(
            r"(CWE-\d+)", expand=False
        )
        cwe_counts = df["cwe_category"].value_counts(dropna=True)
        df["weakness_frequency"] = df["cwe_category"].map(cwe_counts)

        if "is_kev" in df.columns:
            kev_rate = (
                df.groupby("cwe_category")["is_kev"].mean().rename("cwe_risk_factor")
            ).to_dict()
            df["cwe_risk_factor"] = df["cwe_category"].map(kev_rate)
        else:
            df["cwe_risk_factor"] = np.nan
    else:
        df["cwe_category"] = np.nan
        df["weakness_frequency"] = np.nan
        df["cwe_risk_factor"] = np.nan

    # Text features
    if "description" in df.columns:
        desc = df["description"].astype(str).fillna("")
        df["desc_len"] = desc.str.len()
        df["word_count"] = desc.apply(lambda x: len(x.split()))
        keywords = [
            "remote code execution",
            "privilege escalation",
            "denial of service",
            "buffer overflow",
        ]
        df["keyword_indicators"] = desc.str.lower().apply(
            lambda x: int(any(k in x for k in keywords))
        )

        # TF-IDF features (simple, top 100)
        if desc.str.strip().str.len().gt(0).any():
            tfidf = TfidfVectorizer(max_features=100, stop_words="english")
            tfidf_matrix = tfidf.fit_transform(desc)
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                index=df.index,
                columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
            )
            df = pd.concat([df, tfidf_df], axis=1)
    else:
        df["desc_len"] = np.nan
        df["word_count"] = np.nan
        df["keyword_indicators"] = np.nan
        df["description"] = ""


    passthrough_cols = [
        "vendorProject",
        "product",
        "vulnerabilityName",
        "description_nvd",
        "description_jvn",
    ]
    for col in passthrough_cols:
        if col not in df.columns:
            df[col] = ""

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to combined_master.csv")
    parser.add_argument(
        "--output", required=True, help="Path to save output CSV"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df_clean = clean_cve_df(df)
    df_features = build_feature_set(df_clean)
    df_features.to_csv(args.output, index=False)
    print(f"✅ Output written to {args.output} with shape: {df_features.shape}")
