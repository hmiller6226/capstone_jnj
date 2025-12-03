"""
hazard_model.py

Survival (CoxPH) + hazard scoring for CVE KEV prediction.

- Loads df_after_feature_engineering.csv
- Fits stratified Cox Proportional Hazards model
- Computes partial hazard (risk_score) for non-KEVs
- Focuses on critical / specific strata (e.g., 2/critical)
- Produces hazard_outputs.csv with:
    cve_id, risk_score, predicted_day_to_kev_quantile,
    description, vendorProject, product, vulnerabilityName
- Generates diagnostic plots (matplotlib + Plotly)
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def main():
    # ------------------------------
    # Load and prepare data
    # ------------------------------
    df = pd.read_csv("C:/Users/jvgat/Downloads/Hazard/df_after_feature_engineering.csv")

    # Parse dates
    date_cols = ["published_date", "kev_published"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Time-to-event in days
    df["time_to_event"] = (df["kev_published"] - df["published_date"]).dt.total_seconds() / (60 * 60 * 24)

    # Event indicator: kev_present
    df["event"] = df["kev_present"]

    # For non-KEVs (event == 0), censor at "now"
    df.loc[df["event"] == 0, "time_to_event"] = (
        pd.Timestamp.now(tz="UTC") - df.loc[df["event"] == 0, "published_date"]
    ).dt.total_seconds() / (60 * 60 * 24)

    # Features used in the model
    features = [
        "base_score",
        "repo_publication_lag",
        "cross_listing_count",
        "cross_listing_variance",
        "cwe_risk_factor",
    ]

    # Model dataframe
    df_model = df[features + ["time_to_event", "event"]].dropna().copy()

    # Transformations
    df_model["repo_publication_lag_rank"] = df_model["repo_publication_lag"].rank()
    df_model["cross_listing_variance_sqrt"] = np.sqrt(df_model["cross_listing_variance"])

    # Inspect distributions (optional debug prints)
    print("Cross-listing count distribution:")
    print(df_model["cross_listing_count"].describe())
    print("\nCWE risk factor distribution:")
    print(df_model["cwe_risk_factor"].describe())

    # ------------------------------
    # Stratification variables
    # ------------------------------
    # Cross-listing categories
    df_model["cross_listing_count_cat"] = pd.cut(
        df_model["cross_listing_count"],
        bins=[0, 1, 2, float("inf")],  # 1, 2, >=3
        labels=["1", "2", "3"],
    )

    # CWE risk categories (quantile-based)
    df_model["cwe_risk_category"] = pd.qcut(
        df_model["cwe_risk_factor"],
        q=4,
        labels=["low", "medium", "high", "critical"],
        duplicates="drop",
    )

    # Check strata
    print("\nNew strata distribution:")
    strata_counts = df_model.groupby(
        ["cross_listing_count_cat", "cwe_risk_category"], observed=True
    ).size()
    print(strata_counts)
    print(f"\nNumber of valid strata: {len(strata_counts[strata_counts > 0])}")

    # ------------------------------
    # Final model data
    # ------------------------------
    transformed_features = [
        "base_score",
        "repo_publication_lag_rank",
        "cross_listing_variance_sqrt",
    ]

    model_cols = transformed_features + [
        "time_to_event",
        "event",
        "cross_listing_count_cat",
        "cwe_risk_category",
    ]

    df_final = df_model[model_cols].copy()

    # ------------------------------
    # Fit CoxPH model with strata
    # ------------------------------
    cph = CoxPHFitter()
    cph.fit(
        df_final,
        duration_col="time_to_event",
        event_col="event",
        strata=["cross_listing_count_cat", "cwe_risk_category"],
    )

    print("\n" + "=" * 50)
    print("Model Summary:")
    print("=" * 50)
    print(cph.summary)

    # Proportional hazards assumption check
    cph.check_assumptions(
        df_final,
        columns=transformed_features,
        p_value_threshold=0.05,
        show_plots=True,
    )

    print(f"\nConcordance index: {cph.concordance_index_}")

    # ------------------------------
    # Non-KEV subset and risk_score
    # ------------------------------
    df_nonkev = df_final[df_final["event"] == 0].copy()
    df_nonkev["risk_score"] = cph.predict_partial_hazard(df_nonkev)

    # Merge metadata back in (by index)
    meta_cols = [
        "vendorProject",
        "product",
        "vulnerabilityName",
        "cve_id",
        "description_nvd",
        "description_jvn",
        "description",
    ]
    df_nonkev = df_nonkev.merge(
        df[meta_cols],
        left_index=True,
        right_index=True,
        how="left",
    )

    # Stratum label
    df_nonkev["stratum"] = (
        df_nonkev["cross_listing_count_cat"].astype(str)
        + "/"
        + df_nonkev["cwe_risk_category"].astype(str)
    )

    # ------------------------------
    # Focus on specific strata (e.g., 2 / [low, medium, high, critical])
    # ------------------------------
    strata_focus = ["2/low", "2/medium", "2/high", "2/critical"]
    df_focus = df_nonkev[df_nonkev["stratum"].isin(strata_focus)].copy()

    # ------------------------------
    # Plot: average survival curves per stratum
    # ------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    for stratum in strata_focus:
        df_stratum = df_focus[df_focus["stratum"] == stratum]
        if df_stratum.empty:
            continue

        surv_funcs = cph.predict_survival_function(df_stratum)
        avg_surv = surv_funcs.mean(axis=1)

        ax.plot(avg_surv.index, avg_surv.values, label=stratum, linewidth=2)

    ax.set_title("Average Predicted Survival Functions — Medium Cross-Listing (Non-KEVs)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Average Survival Probability")
    ax.legend(title="Stratum")
    plt.tight_layout()
    plt.show()

    # Hazard score distribution by stratum
    sns.histplot(df_focus, x="risk_score", hue="stratum", kde=True, bins=30)
    plt.title("Hazard Score Distribution — Medium Cross-Listing (Non-KEVs)")
    plt.xlabel("Predicted Partial Hazard (Risk Score)")
    plt.ylabel("Count")
    plt.show()

    # Cumulative hazard by stratum
    plt.figure(figsize=(10, 6))
    for stratum in strata_focus:
        df_stratum = df_focus[df_focus["stratum"] == stratum]
        if df_stratum.empty:
            continue

        cumhaz = cph.predict_cumulative_hazard(df_stratum).mean(axis=1)
        plt.plot(cumhaz.index, cumhaz.values, label=stratum)

    plt.title("Average Cumulative Hazard — Medium Cross-Listing (Non-KEVs)")
    plt.xlabel("Time (days)")
    plt.ylabel("Cumulative Hazard")
    plt.legend(title="Stratum")
    plt.tight_layout()
    plt.show()

    # Boxplot of risk_score by stratum
    sns.boxplot(data=df_focus, x="stratum", y="risk_score")
    plt.title("Hazard Score Distribution - Medium Cross-Listing (Non-KEVs)")
    plt.xlabel("Stratum")
    plt.ylabel("Predicted Partial Hazard")
    plt.tight_layout()
    plt.show()

    # ------------------------------
    # Non-KEVs in 2/critical — survival plateau analysis
    # ------------------------------
    df_critical_nonkev_plateau = df_nonkev[df_nonkev["stratum"] == "2/critical"].copy()

    if df_critical_nonkev_plateau.empty:
        print("No non-KEVs in the '2/critical' stratum for plateau analysis.")
    else:
        surv_funcs_plateau = cph.predict_survival_function(df_critical_nonkev_plateau)
        final_time = surv_funcs_plateau.index[-1]

        df_critical_nonkev_plateau["predicted_survival_probability"] = (
            surv_funcs_plateau.loc[final_time].values
        )

        leveling_times = []
        for idx in df_critical_nonkev_plateau.index:
            surv = surv_funcs_plateau[idx]
            decreasing_times = surv[surv.diff() < 0].index
            if len(decreasing_times) > 0:
                leveling_times.append(decreasing_times[-1])
            else:
                leveling_times.append(np.nan)

        df_critical_nonkev_plateau["survival_plateau_day"] = leveling_times

        plateau_table = df_critical_nonkev_plateau[
            ["cve_id", "risk_score", "predicted_survival_probability", "survival_plateau_day"]
        ].sort_values("predicted_survival_probability")

        print("\nPredicted Survival & Leveling-Off Time — Non-KEVs in 2/critical:")
        print(plateau_table.to_string(index=False))

    # ------------------------------
    # Non-KEVs in 2/critical — predicted day-to-KEV (quantile-based)
    # ------------------------------
    df_critical_nonkev = df_focus[
        (df_focus["stratum"] == "2/critical") & (df_focus["event"] == 0)
    ].copy()

    if df_critical_nonkev.empty:
        print("No non-KEVs in the '2/critical' stratum for quantile-based prediction.")
        return

    # Risk score for this subset (already computed, but safe to recompute)
    df_critical_nonkev["risk_score"] = cph.predict_partial_hazard(df_critical_nonkev)

    # Predict survival functions and restrict horizon (example: <= 1691 days)
    surv_funcs = cph.predict_survival_function(df_critical_nonkev)
    surv_funcs = surv_funcs[surv_funcs.index <= 1691]

    # Data-driven survival threshold (e.g., 5% quantile of all survival values)
    all_surv_values = surv_funcs.values.flatten()
    quantile_threshold = np.quantile(all_surv_values, 0.05)
    print("Data-driven survival threshold (5th percentile):", quantile_threshold)

    predicted_days_quantile = []
    for idx in surv_funcs.columns:
        surv_series = surv_funcs[idx]
        below = surv_series[surv_series <= quantile_threshold]
        if not below.empty:
            predicted_days_quantile.append(below.index[0])
        else:
            predicted_days_quantile.append(np.nan)

    df_critical_nonkev["predicted_day_to_kev_quantile"] = predicted_days_quantile

    # Fill missing predicted days with "Never" for interpretability
    df_critical_nonkev["predicted_day_to_kev_quantile"] = (
        df_critical_nonkev["predicted_day_to_kev_quantile"].fillna("Never")
    )

    # ------------------------------
    # Final hazard outputs table
    # ------------------------------
    out_cols = [
        "cve_id",
        "risk_score",
        "predicted_day_to_kev_quantile",
        "description",
        "vendorProject",
        "product",
        "vulnerabilityName",
    ]
    out_cols = [c for c in out_cols if c in df_critical_nonkev.columns]

    table = df_critical_nonkev[out_cols].sort_values("risk_score", ascending=False)

    print("\nTop rows of hazard outputs (2/critical, non-KEVs):")
    print(table.head(50).to_string(index=False))

    # Write to CSV
    table.to_csv("hazard_outputs.csv", index=False)
    print("\nSaved hazard_outputs.csv")

    # ------------------------------
    # Plotly: Predicted days vs risk_score (ignore 'Never' for Y-axis)
    # ------------------------------
    df_plot = df_critical_nonkev.copy()

    # Keep only rows with numeric predictions
    df_plot_numeric = df_plot[df_plot["predicted_day_to_kev_quantile"] != "Never"].copy()
    if not df_plot_numeric.empty:
        df_plot_numeric["predicted_day_to_kev_quantile"] = (
            df_plot_numeric["predicted_day_to_kev_quantile"].astype(float)
        )

        fig = px.scatter(
            df_plot_numeric,
            x="risk_score",
            y="predicted_day_to_kev_quantile",
            color="risk_score",
            color_continuous_scale="Reds",
            hover_data={
                "cve_id": True,
                "risk_score": True,
                "predicted_day_to_kev_quantile": True,
            },
            labels={
                "risk_score": "Predicted Risk Score",
                "predicted_day_to_kev_quantile": "Predicted Days to KEV (Quantile Threshold)",
            },
            title="Predicted Days to KEV vs Risk Score — Critical Stratum Non-KEVs",
            opacity=0.7,
        )

        # Optional: quantile lines for risk_score
        quantiles = [0.5, 0.75, 0.9]
        for q in quantiles:
            q_value = df_plot_numeric["risk_score"].quantile(q)
            fig.add_vline(
                x=q_value,
                line=dict(color="gray", dash="dash"),
                annotation_text=f"{int(q * 100)}%",
                annotation_position="top right",
            )

        # Optional: highlight high-risk region (>90th percentile)
        high_risk_threshold = df_plot_numeric["risk_score"].quantile(0.9)
        fig.add_vrect(
            x0=high_risk_threshold,
            x1=df_plot_numeric["risk_score"].max(),
            fillcolor="yellow",
            opacity=0.2,
            layer="below",
            line_width=0,
            annotation_text="High-risk region (>90th percentile)",
            annotation_position="top left",
        )

        fig.update_layout(coloraxis_colorbar=dict(title="Risk Score"))
        fig.show()
    else:
        print("No numeric predicted_day_to_kev_quantile values to plot.")

    # ------------------------------
    # Notes / interpretation (formerly markdown in notebook)
    # ------------------------------
    # predicted_day_to_kev_quantile:
    #   The day (time in days) at which the survival curve for a CVE
    #   first falls below a data-driven threshold based on the 5th
    #   percentile of all predicted survival probabilities.
    #
    #   "Never" indicates that, within the modeled time horizon,
    #   the survival curve never crosses this threshold.


if __name__ == "__main__":
    main()
