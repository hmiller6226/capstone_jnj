#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

# -----------------------------------------------------------
# Date parsing
# -----------------------------------------------------------
def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = [
        "published_date", "nvd_published", "jvn_published",
        "eu_published", "kev_published", "lastmodified",
        "dateupdated", "published"
    ]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df


# -----------------------------------------------------------
# Choose effective publication date
# -----------------------------------------------------------
def choose_published(row: pd.Series):
    for c in [
        "published_date", "nvd_published", "jvn_published",
        "eu_published", "published"
    ]:
        if c in row and pd.notna(row[c]):
            return row[c]
    return pd.NaT


# -----------------------------------------------------------
# Build time-to-event + event indicator
# -----------------------------------------------------------
def build_time_to_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set:
      - event = kev_present (1 if KEV, else 0)
      - time_to_event (days) from published_effective to KEV (event)
        or to ref (censoring).
    """
    df = df.copy()
    df["published_effective"] = df.apply(choose_published, axis=1)
    now = pd.Timestamp.now(tz="UTC")

    # Reference date for censoring
    ref = df[["lastmodified", "dateupdated", "kev_published"]].max(axis=1)
    ref = ref.fillna(now)

    # Event indicator
    df["event"] = df["kev_present"].astype(int)

    # Event rows: from published_effective to kev_published
    mask_e = (
        df["event"].eq(1)
        & df["kev_published"].notna()
        & df["published_effective"].notna()
    )
    df.loc[mask_e, "time_to_event"] = (
        df.loc[mask_e, "kev_published"]
        - df.loc[mask_e, "published_effective"]
    ).dt.total_seconds() / (60 * 60 * 24)

    # Censored rows: from published_effective to ref date
    mask_c = df["event"].eq(0) & df["published_effective"].notna()
    df.loc[mask_c, "time_to_event"] = (
        ref[mask_c] - df.loc[mask_c, "published_effective"]
    ).dt.total_seconds() / (60 * 60 * 24)

    # Clean
    df = df.dropna(subset=["time_to_event"]).copy()
    df["time_to_event"] = df["time_to_event"].clip(lower=1.0, upper=3650.0)

    return df


# -----------------------------------------------------------
# Numeric features for Cox model
# -----------------------------------------------------------
def build_features(df: pd.DataFrame):
    """
    Standardize numeric covariates and build the modeling frame
    for CoxPH.
    """
    df = df.copy()
    candidates = [
        "base_score", "nvd_base_score", "jvn_base_score", "eu_base_score",
        "epss", "source_count", "is_known_exploited",
        "cvss_exploitability", "cvss_impact",
    ]

    features = []
    for c in candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())
            mean, std = df[c].mean(), df[c].std()
            if std > 0:
                df[c] = (df[c] - mean) / std
                features.append(c)

    if not features:
        raise ValueError(
            "No usable numeric features found. "
            f"Expected at least one of: {candidates}"
        )

    df_model = df[["time_to_event", "event"] + features].dropna().copy()
    return df_model, features


# -----------------------------------------------------------
# Interpolated days-to-KEV from survival model
# -----------------------------------------------------------
def interpolated_days_to_kev(
    df_model: pd.DataFrame,
    cph: CoxPHFitter,
    features: list,
    df_original: pd.DataFrame,
    low_q: float = 0.15,   # start at 15th percentile KEV time
    high_q: float = 0.95,  # end at 95th percentile KEV time
    gamma: float = 0.3,    # stronger smoothing (< 0.5 → earlier days rarer)
) -> pd.DataFrame:
    """
    1) Use CoxPH to get relative risk (partial hazards).
    2) Convert to percentile ranks (0..1; higher = riskier).
    3) Build empirical KEV timing curve from event=1 rows.
    4) Interpolate ranks into KEV timing distribution with
       smoothed top tail, and floor/ceiling the days.
    """
    # ---- 1. Cox risk scores ----
    risk = cph.predict_partial_hazard(df_model[features]).values

    # Rank → percentile (0..1); higher = riskier
    ranks = pd.Series(risk).rank(method="average", pct=True).values

    # ---- 2. Empirical KEV timing distribution ----
    kev_times = df_model.loc[df_model["event"] == 1, "time_to_event"].values
    if len(kev_times) < 5:
        raise ValueError(
            "Not enough KEV events to build interpolated timing curve "
            f"(got {len(kev_times)} events, need at least 5)."
        )

    kev_times_sorted = np.sort(kev_times)

    # Quantile positions for sorted KEV times (0..1)
    q_positions = np.linspace(0.0, 1.0, len(kev_times_sorted))

    # ---- 3. Map risk percentile into [low_q, high_q] with smoothing ----
    low_q = float(low_q)
    high_q = float(high_q)
    gamma = float(gamma)

    if not (0.0 <= low_q < high_q <= 1.0):
        raise ValueError("Require 0 <= low_q < high_q <= 1.")
    if not (0.0 < gamma < 1.0):
        raise ValueError("gamma should be in (0,1) for tail smoothing.")

    # q_raw: 1 - rank → high risk gives small q_raw
    q_raw = 1.0 - ranks

    # Strong tail smoothing: gamma < 1 makes early times rarer
    q_scaled = q_raw ** gamma

    # Map to effective quantiles
    q_eff = low_q + q_scaled * (high_q - low_q)
    q_eff = np.clip(q_eff, low_q, high_q)

    # ---- 4. Continuous interpolation ----
    mapped_days = np.interp(q_eff, q_positions, kev_times_sorted)

    # ---- 5. Floor and ceiling for realism ----
    # Floor at max(14 days, 5th percentile of KEV times)
    floor_days = max(14.0, float(np.quantile(kev_times_sorted, 0.05)))
    # Ceiling at 97th percentile of KEV times
    ceil_days = float(np.quantile(kev_times_sorted, 0.97))

    mapped_days = np.clip(mapped_days, floor_days, ceil_days)

    # ---- 6. Attach CVE IDs ----
    result = pd.DataFrame(
        {
            "cve_id": df_original.loc[df_model.index, "cve_id"].values,
            "days_to_kev": mapped_days,
        }
    )

    return result


# -----------------------------------------------------------
# Main entry
# -----------------------------------------------------------
def main(input_csv: str, output_csv: str):
    print("Loading:", input_csv)
    df = pd.read_csv(input_csv, low_memory=False)
    df = parse_dates(df)

    print("Building time-to-event...")
    df_time = build_time_to_event(df)

    print("Building feature matrix...")
    df_model, features = build_features(df_time)

    print("Training Cox model...")
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_model, duration_col="time_to_event", event_col="event")
    print("Concordance Index:", cph.concordance_index_)

    print("Interpolating days-to-KEV (sparser early bucket)...")
    out = interpolated_days_to_kev(
        df_model,
        cph,
        features,
        df_time,
        low_q=0.15,   # 15th–95th KEV quantiles
        high_q=0.95,
        gamma=0.3,    # stronger smoothing → earliest KEV days rarer
    )

    out.to_csv(output_csv, index=False)
    print("Saved:", output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to combined_master.csv")
    parser.add_argument("--output", default="days_to_kev.csv", help="Output CSV path")
    args = parser.parse_args()
    main(args.input, args.output)
