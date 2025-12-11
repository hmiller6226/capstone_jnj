#!/usr/bin/env python3

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def _predicted_day_quantile(model: CoxPHFitter, x: pd.DataFrame, horizon_days: int = 1691, q: float = 0.05) -> pd.Series:
    if x.empty:
        return pd.Series([], dtype="object")

    sf = model.predict_survival_function(x)
    if sf is None or sf.empty:
        return pd.Series(["Never"] * len(x), index=x.index, dtype="object")

    sf = sf[sf.index <= horizon_days]
    if sf.empty:
        return pd.Series(["Never"] * len(x), index=x.index, dtype="object")

    vals = sf.values.ravel()
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return pd.Series(["Never"] * len(x), index=x.index, dtype="object")

    thr = float(np.quantile(vals, q))

    out = {}
    for col in sf.columns:
        s = sf[col]
        below = s[s <= thr]
        out[col] = "Never" if below.empty else float(below.index[0])

    ser = pd.Series(out, dtype="object")
    ser.index = pd.Index(ser.index)
    return ser


def run_model(input_path: str, output_path: str, output_scope: str, penalizer: float) -> None:
    frame = pd.read_csv(input_path)

    required = [
        "published_date",
        "kev_published",
        "kev_present",
        "base_score",
        "repo_publication_lag",
        "cross_listing_count",
        "cross_listing_variance",
        "cwe_risk_factor",
    ]
    miss = [c for c in required if c not in frame.columns]
    if miss:
        raise ValueError(f"Missing required columns in input: {miss}")

    frame["published_date"] = pd.to_datetime(frame["published_date"], errors="coerce", utc=True)
    frame["kev_published"] = pd.to_datetime(frame["kev_published"], errors="coerce", utc=True)

    frame["event"] = frame["kev_present"].astype(int)

    delta = (frame["kev_published"] - frame["published_date"]).dt.total_seconds() / 86400.0
    frame["time_to_event"] = delta

    now_utc = pd.Timestamp.now(tz="UTC")
    non_mask = frame["event"] == 0
    frame.loc[non_mask, "time_to_event"] = (
        (now_utc - frame.loc[non_mask, "published_date"]).dt.total_seconds() / 86400.0
    )

    feats = [
        "base_score",
        "repo_publication_lag",
        "cross_listing_count",
        "cross_listing_variance",
        "cwe_risk_factor",
    ]

    base = frame[feats + ["time_to_event", "event"]].copy()
    base = base.dropna()

    base = base[base["time_to_event"].notna()].copy()
    base = base[base["time_to_event"] > 0].copy()

    base["cross_listing_variance"] = pd.to_numeric(base["cross_listing_variance"], errors="coerce")
    base.loc[base["cross_listing_variance"] < 0, "cross_listing_variance"] = np.nan

    base["repo_publication_lag_rank"] = base["repo_publication_lag"].rank(method="average")
    base["cross_listing_variance_sqrt"] = np.sqrt(base["cross_listing_variance"])

    base["cross_listing_count_cat"] = pd.cut(
        base["cross_listing_count"],
        bins=[0, 1, 2, float("inf")],
        labels=["1", "2", "3"],
        include_lowest=True,
    )

    base["cwe_risk_category"] = pd.qcut(
        base["cwe_risk_factor"],
        q=4,
        labels=["low", "medium", "high", "critical"],
        duplicates="drop",
    )

    base = base.dropna(subset=["cross_listing_count_cat", "cwe_risk_category"]).copy()

    use_cols = [
        "base_score",
        "repo_publication_lag_rank",
        "cross_listing_variance_sqrt",
        "time_to_event",
        "event",
        "cross_listing_count_cat",
        "cwe_risk_category",
    ]

    model_df = base[use_cols].copy()
    model_df.index = base.index

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(
        model_df,
        duration_col="time_to_event",
        event_col="event",
        strata=["cross_listing_count_cat", "cwe_risk_category"],
    )

    nonkev = model_df[model_df["event"] == 0].copy()
    if nonkev.empty:
        pd.DataFrame(
            columns=[
                "cve_id",
                "risk_score",
                "predicted_day_to_kev_quantile",
                "description",
                "vendorProject",
                "product",
                "vulnerabilityName",
            ]
        ).to_csv(output_path, index=False)
        return

    nonkev["risk_score"] = cph.predict_partial_hazard(nonkev)

    meta = [
        "vendorProject",
        "product",
        "vulnerabilityName",
        "cve_id",
        "description_nvd",
        "description_jvn",
        "description",
    ]
    for c in meta:
        if c not in frame.columns:
            frame[c] = ""

    scored = nonkev.merge(frame[meta], left_index=True, right_index=True, how="left")

    if "description" not in scored.columns or scored["description"].fillna("").eq("").all():
        d = scored.get("description_nvd", "").fillna("")
        m = d.eq("")
        d.loc[m] = scored.get("description_jvn", "").fillna("").loc[m]
        scored["description"] = d

    scored["stratum"] = (
        scored["cross_listing_count_cat"].astype(str)
        + "/"
        + scored["cwe_risk_category"].astype(str)
    )

    focus = ["2/low", "2/medium", "2/high", "2/critical"]

    if output_scope == "critical":
        out = scored[scored["stratum"] == "2/critical"].copy()
    elif output_scope == "focus":
        out = scored[scored["stratum"].isin(focus)].copy()
    elif output_scope == "all":
        out = scored.copy()
    else:
        raise ValueError("output_scope must be one of: critical, focus, all")

    if out.empty:
        pd.DataFrame(
            columns=[
                "cve_id",
                "risk_score",
                "predicted_day_to_kev_quantile",
                "description",
                "vendorProject",
                "product",
                "vulnerabilityName",
            ]
        ).to_csv(output_path, index=False)
        return

    qs = _predicted_day_quantile(cph, out[use_cols])
    out["predicted_day_to_kev_quantile"] = qs.reindex(out.index).fillna("Never").astype("object")

    export_cols = [
        "cve_id",
        "risk_score",
        "predicted_day_to_kev_quantile",
        "description",
        "vendorProject",
        "product",
        "vulnerabilityName",
    ]

    table = out[export_cols].sort_values("risk_score", ascending=False)
    table.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--output-scope", choices=["critical", "focus", "all"], default="focus")
    p.add_argument("--penalizer", type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run_model(a.input, a.output, a.output_scope, a.penalizer)
