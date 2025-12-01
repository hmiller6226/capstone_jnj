# dashboard.py
"""
CVE Risk & Coverage Dashboard with CVSS Vectorstring attack-vector analysis.

Features added:
- Parses cvss_vectorstring to extract Attack Vector (AV) and other metrics.
- Attack Vector distribution chart (counts).
- P(KEV | AttackVector) chart (probabilities).
- Top cvss_vectorstring entries by P(KEV | vectorstring) (requires minimum support).
- Defensive handling for missing columns.
"""

import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objs as go
import numpy as np

# ==============================
# CONFIG
# ==============================
CSV_PATH = "kde_scores.csv"
RISK_SCORE_COL = "risk_score"

# ==============================
# HELPERS: parse CVSS vectorstring
# ==============================
def parse_cvss_vector(vecstr):
    """
    Parse a CVSS vector string like:
      "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    Returns dict of components, e.g. {'AV':'N','AC':'L', ...}
    If vecstr is NaN / invalid, returns empty dict.
    """
    if pd.isna(vecstr):
        return {}
    if not isinstance(vecstr, str):
        return {}
    parts = vecstr.strip().split("/")
    d = {}
    for p in parts:
        if ":" in p:
            k, v = p.split(":", 1)
            d[k] = v
    return d

# Human-readable mapping for AV
AV_MAP = {
    "N": "NETWORK",
    "A": "ADJACENT",
    "L": "LOCAL",
    "P": "PHYSICAL"
}

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(CSV_PATH, low_memory=False)

# Basic typing / cleanup (defensive)
if "cve_year" in df.columns:
    df["cve_year"] = pd.to_numeric(df["cve_year"], errors="coerce")

# Ensure is_kev (numeric 0/1) exists; prefer 'is_kev' then 'kev_present'
if "is_kev" in df.columns:
    df["is_kev"] = pd.to_numeric(df["is_kev"], errors="coerce").fillna(0).astype(int)
elif "kev_present" in df.columns:
    # kev_present might be boolean
    df["is_kev"] = df["kev_present"].astype(int)
else:
    df["is_kev"] = 0

# base_score numeric
if "base_score" in df.columns:
    df["base_score"] = pd.to_numeric(df["base_score"], errors="coerce")
else:
    df["base_score"] = np.nan

# risk_score numeric fallback
if RISK_SCORE_COL in df.columns:
    df[RISK_SCORE_COL] = pd.to_numeric(df[RISK_SCORE_COL], errors="coerce")
else:
    df[RISK_SCORE_COL] = df["base_score"]

# severity label
if "severity" not in df.columns:
    def map_sev(s):
        if pd.isna(s):
            return None
        s = float(s)
        if s <= 3.9:
            return "LOW"
        elif s <= 6.9:
            return "MEDIUM"
        elif s <= 8.9:
            return "HIGH"
        else:
            return "CRITICAL"
    df["severity"] = df["base_score"].apply(map_sev)

# ==============================
# PARSE cvss_vectorstring into columns
# ==============================
VECTOR_COL = "cvss_vectorstring"  # user specified

# create parsed columns only if present
if VECTOR_COL in df.columns:
    # Parse each row to a dict, then create separate columns for components
    parsed = df[VECTOR_COL].apply(parse_cvss_vector)
    # Determine all keys found
    all_keys = set()
    for d in parsed.dropna().tolist():
        if isinstance(d, dict):
            all_keys.update(d.keys())
    # Create columns for each key (AV, AC, PR, UI, S, C, I, A, etc.)
    for k in sorted(all_keys):
        df[f"vec_{k}"] = parsed.apply(lambda x: x.get(k) if isinstance(x, dict) else np.nan)
    # Also create a human-readable attack vector column
    if "AV" in all_keys:
        df["attack_vector_code"] = df["vec_AV"]
        df["attack_vector"] = df["vec_AV"].map(lambda x: AV_MAP.get(x, x) if pd.notna(x) else np.nan)
    else:
        df["attack_vector_code"] = np.nan
        df["attack_vector"] = np.nan
else:
    # ensure columns exist to avoid KeyErrors later
    df["attack_vector_code"] = np.nan
    df["attack_vector"] = np.nan

# ==============================
# DERIVED STATS: P(KEV | attack_vector) and P(KEV | full vectorstring)
# ==============================
def compute_vector_probabilities(dfin, min_support_vectorstring=30):
    out = {}
    # By attack_vector (AV human readable)
    if "attack_vector" in dfin.columns:
        av_counts = (
            dfin[["attack_vector", "is_kev"]]
            .dropna(subset=["attack_vector"])
            .groupby("attack_vector")
            .agg(total=("is_kev", "count"), kev_count=("is_kev", "sum"))
            .reset_index()
        )
        av_counts["p_kev_given_av"] = av_counts["kev_count"] / av_counts["total"]
        av_counts = av_counts.sort_values("total", ascending=False)
    else:
        av_counts = pd.DataFrame(columns=["attack_vector", "total", "kev_count", "p_kev_given_av"])

    # By full cvss_vectorstring (exact string)
    if VECTOR_COL in dfin.columns:
        vs_counts = (
            dfin[[VECTOR_COL, "is_kev"]]
            .dropna(subset=[VECTOR_COL])
            .groupby(VECTOR_COL)
            .agg(total=("is_kev", "count"), kev_count=("is_kev", "sum"))
            .reset_index()
        )
        vs_counts["p_kev_given_vector"] = vs_counts["kev_count"] / vs_counts["total"]
        # filter by min support to avoid tiny-sample noise
        vs_top = vs_counts[vs_counts["total"] >= min_support_vectorstring].sort_values(
            "p_kev_given_vector", ascending=False
        )
    else:
        vs_top = pd.DataFrame(columns=[VECTOR_COL, "total", "kev_count", "p_kev_given_vector"])

    out["av_counts"] = av_counts
    out["vectorstring_counts"] = vs_counts if 'vs_counts' in locals() else pd.DataFrame()
    out["vectorstring_top"] = vs_top
    return out

vec_probs = compute_vector_probabilities(df, min_support_vectorstring=25)

# ==============================
# Dash App
# ==============================
app = Dash(__name__)
app.title = "CVE Risk & Coverage Dashboard (with CVSS Vector Analysis)"

# Some existing selectors
if "cve_year" in df.columns and df["cve_year"].notna().any():
    year_min = int(df["cve_year"].min())
    year_max = int(df["cve_year"].max())
else:
    year_min, year_max = 2000, 2025

severity_options = sorted([s for s in df["severity"].dropna().unique()]) if "severity" in df.columns else []

app.layout = html.Div(
    style={"margin": "20px"},
    children=[
        html.H1("CVE Risk & Coverage Dashboard — CVSS Vector Analysis"),

        # Filters
        html.Div(
            style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={"width": "420px"},
                    children=[
                        html.Label("CVE Year Range"),
                        dcc.RangeSlider(
                            id="year-slider",
                            min=year_min,
                            max=year_max,
                            step=1,
                            value=[year_min, year_max],
                            marks={
                                y: str(y)
                                for y in range(
                                    year_min,
                                    year_max + 1,
                                    max(1, (year_max - year_min) // 6 or 1),
                                )
                            },
                            allowCross=False,
                        ),
                    ],
                ),
                html.Div(
                    style={"width": "260px"},
                    children=[
                        html.Label("Severity"),
                        dcc.Dropdown(
                            id="severity-dropdown",
                            options=[{"label": s, "value": s} for s in severity_options],
                            value=[],
                            multi=True,
                            placeholder="All severities",
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        html.Label("KEV Filter"),
                        dcc.RadioItems(
                            id="kev-filter",
                            options=[
                                {"label": "All", "value": "all"},
                                {"label": "KEV only", "value": "kev"},
                                {"label": "Non-KEV only", "value": "nonkev"},
                            ],
                            value="all",
                            inline=True,
                        ),
                    ]
                ),
                html.Div(
                    style={"width": "230px"},
                    children=[
                        html.Label("Score Pair (Risk-Score Correlation)"),
                        dcc.Dropdown(
                            id="score-pair",
                            options=[
                                {"label": "NVD vs JVN", "value": "nvd_jvn"},
                                {"label": "NVD vs EUVD", "value": "nvd_euvd"},
                                {"label": "JVN vs EUVD", "value": "jvn_euvd"},
                            ],
                            value="nvd_euvd",
                            clearable=False,
                        ),
                    ],
                ),
            ],
        ),

        html.Hr(),

        # Main charts grid - extended with Attack Vector charts
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "24px",
            },
            children=[
                dcc.Graph(id="cross-listing-chart"),
                dcc.Graph(id="registration-lag-chart"),
                dcc.Graph(id="risk-correlation-chart"),
                dcc.Graph(id="kev-leadtime-chart"),
                # New attack vector charts
                dcc.Graph(id="attack-vector-distribution"),
                dcc.Graph(id="attack-vector-kev-prob"),
                dcc.Graph(id="vectorstring-top-kevprob"),
            ],
        ),

        html.Hr(),

        # Top CVEs table (unchanged)
        html.Div(
            style={"display": "flex", "flexDirection": "column", "gap": "10px"},
            children=[
                html.H2("Top CVEs by Risk Score (Searchable)"),
                html.Div(
                    style={"display": "flex", "gap": "10px", "alignItems": "center"},
                    children=[
                        html.Label("Search CVE ID:"),
                        dcc.Input(
                            id="cve-search",
                            type="text",
                            placeholder="e.g., CVE-2023-12345",
                            style={"width": "260px"},
                        ),
                        html.Div(
                            f"Empty search = show top 5 by {RISK_SCORE_COL}",
                            style={"fontSize": "12px", "color": "#555"},
                        ),
                    ],
                ),
                dash_table.DataTable(
                    id="top-risk-table",
                    columns=[
                        {"name": "CVE ID", "id": "cve_id"},
                        {"name": "Risk Score", "id": RISK_SCORE_COL},
                        {"name": "Base Score", "id": "base_score"},
                        {"name": "Severity", "id": "severity"},
                        {"name": "Is KEV", "id": "is_kev"},
                        {"name": "Days to KEV", "id": "days_to_kev"} if "days_to_kev" in df.columns else {},
                    ],
                    page_size=10,
                    sort_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "padding": "5px"},
                    style_header={
                        "backgroundColor": "#f0f0f0",
                        "fontWeight": "bold",
                    },
                ),
            ],
        ),
    ],
)

# Helper: apply filters
def filter_df(df_in, year_range, severities, kev_mode):
    dff = df_in.copy()
    if "cve_year" in dff.columns:
        dff = dff[(dff["cve_year"] >= year_range[0]) & (dff["cve_year"] <= year_range[1])]
    if severities and "severity" in dff.columns:
        dff = dff[dff["severity"].isin(severities)]
    if kev_mode == "kev" and "is_kev" in dff.columns:
        dff = dff[dff["is_kev"] == 1]
    elif kev_mode == "nonkev" and "is_kev" in dff.columns:
        dff = dff[dff["is_kev"] == 0]
    return dff

# Main charts callback: Extended outputs for the attack-vector charts
@app.callback(
    Output("cross-listing-chart", "figure"),
    Output("registration-lag-chart", "figure"),
    Output("risk-correlation-chart", "figure"),
    Output("kev-leadtime-chart", "figure"),
    Output("attack-vector-distribution", "figure"),
    Output("attack-vector-kev-prob", "figure"),
    Output("vectorstring-top-kevprob", "figure"),
    Input("year-slider", "value"),
    Input("severity-dropdown", "value"),
    Input("kev-filter", "value"),
    Input("score-pair", "value"),
)
def update_charts(year_range, severities, kev_mode, score_pair):
    dff = filter_df(df, year_range, severities, kev_mode)

    # 1) Cross-listing
    if "cross_listing_count" in dff.columns and "cve_id" in dff.columns:
        tmp = dff.copy()
        tmp["Cross-Listed?"] = tmp["cross_listing_count"].ge(2)
        summary = (
            tmp.groupby("Cross-Listed?")
            .agg(CVE_Count=("cve_id", "nunique"))
            .reset_index()
        )
        summary["Cross-Listed?"] = summary["Cross-Listed?"].map({True: "≥ 2 repos", False: "< 2 repos"})
        fig_cross = px.bar(summary, x="Cross-Listed?", y="CVE_Count", text="CVE_Count",
                           title="CVE IDs Present in ≥ 2 Repositories")
        fig_cross.update_layout(xaxis_title="Cross-Listing Category", yaxis_title="Distinct CVE Count")
    else:
        fig_cross = go.Figure().add_annotation(text="Need 'cross_listing_count' and 'cve_id' columns", x=0.5, y=0.5, showarrow=False)

    # 2) Registration lag
    if "repo_publication_lag_clean" in dff.columns:
        lag_df = dff.copy()
        if "cross_listing_count" in lag_df.columns:
            lag_df = lag_df[lag_df["cross_listing_count"] >= 2]
        lag_df = lag_df.dropna(subset=["repo_publication_lag_clean"])
        lag_df = lag_df[lag_df["repo_publication_lag_clean"] > 0]
        if not lag_df.empty:
            fig_lag = px.histogram(lag_df, x="repo_publication_lag_clean", nbins=40,
                                   title="Registration Lag (Δ Days) for Cross-Listed CVEs")
            fig_lag.update_layout(xaxis_title="Repo Publication Lag (days)", yaxis_title="Number of CVEs")
        else:
            fig_lag = go.Figure().add_annotation(text="No valid registration lag data for cross-listed CVEs under current filters", x=0.5, y=0.5, showarrow=False)
    else:
        fig_lag = go.Figure().add_annotation(text="Need 'repo_publication_lag' column", x=0.5, y=0.5, showarrow=False)

    # 3) Risk-score correlation
    pair_map = {
        "nvd_jvn": ("nvd_base_score", "jvn_cvss_score", "NVD Base Score", "JVN CVSS Score"),
        "nvd_euvd": ("nvd_base_score", "euvd_basescore", "NVD Base Score", "EUVD Base Score"),
        "jvn_euvd": ("jvn_cvss_score", "euvd_basescore", "JVN CVSS Score", "EUVD Base Score"),
    }
    x_col, y_col, x_label, y_label = pair_map.get(score_pair, ("nvd_base_score", "euvd_basescore", "NVD Base Score", "EUVD Base Score"))
    if x_col in dff.columns and y_col in dff.columns:
        corr_df = dff.dropna(subset=[x_col, y_col])
        if not corr_df.empty:
            fig_corr = px.scatter(corr_df, x=x_col, y=y_col, color="severity" if "severity" in corr_df.columns else None,
                                  hover_data=["cve_id"] if "cve_id" in corr_df.columns else None,
                                  title=f"Risk-Score Correlation: {x_label} vs {y_label}")
            fig_corr.update_layout(xaxis_title=x_label, yaxis_title=y_label)
        else:
            fig_corr = go.Figure().add_annotation(text=f"No overlapping non-null {x_label} and {y_label} values for current filters", x=0.5, y=0.5, showarrow=False)
    else:
        missing = [c for c in (x_col, y_col) if c not in dff.columns]
        fig_corr = go.Figure().add_annotation(text=f"Missing columns for correlation: {', '.join(missing)}", x=0.5, y=0.5, showarrow=False)

    # 4) KEV Lead Time
    if "days_to_kev" in dff.columns and "is_kev" in dff.columns:
        if "is_high_risk" not in dff.columns:
            dff["is_high_risk"] = dff["base_score"].astype(float) >= 7.0
        kev_hr = dff[(dff["is_kev"] == 1) & (dff["is_high_risk"] == True)].dropna(subset=["days_to_kev"])
        if not kev_hr.empty:
            fig_lead = px.box(kev_hr, x="severity" if "severity" in kev_hr.columns else None, y="days_to_kev",
                              points="outliers", title="KEV Lead Time (Days) — High-Risk KEV CVEs")
            fig_lead.update_layout(xaxis_title="Severity" if "severity" in kev_hr.columns else "", yaxis_title="Days from First Publication to KEV")
        else:
            fig_lead = go.Figure().add_annotation(text="No high-risk KEV records for current filters", x=0.5, y=0.5, showarrow=False)
    else:
        fig_lead = go.Figure().add_annotation(text="Need 'days_to_kev' and 'is_kev' columns", x=0.5, y=0.5, showarrow=False)

    # --------------------------
    # 5) Attack Vector distribution (counts)
    # --------------------------
    if "attack_vector" in dff.columns and dff["attack_vector"].notna().any():
        av_df = (
            dff[["attack_vector", "cve_id"]]
            .dropna(subset=["attack_vector"])
            .groupby("attack_vector")
            .agg(total=("cve_id", "nunique"))
            .reset_index()
            .sort_values("total", ascending=False)
        )
        # safe plotting: use explicit columns
        fig_attack = px.bar(av_df, x="attack_vector", y="total", title="Attack Vector Distribution (by distinct CVE)")
        fig_attack.update_layout(xaxis_title="Attack Vector (AV)", yaxis_title="Distinct CVE Count")
    else:
        fig_attack = go.Figure().add_annotation(text=f"No '{VECTOR_COL}' / 'attack_vector' data available", x=0.5, y=0.5, showarrow=False)

    # --------------------------
    # 6) P(KEV | Attack Vector)
    # --------------------------
    if "attack_vector" in dff.columns and dff["attack_vector"].notna().any():
        av_stats = (
            dff[["attack_vector", "is_kev"]]
            .dropna(subset=["attack_vector"])
            .groupby("attack_vector")
            .agg(total=("is_kev", "count"), kev_count=("is_kev", "sum"))
            .reset_index()
        )
        av_stats["p_kev_given_av"] = av_stats["kev_count"] / av_stats["total"]
        av_stats = av_stats.sort_values("p_kev_given_av", ascending=False)
        # add total to hovertext
        fig_avprob = px.bar(av_stats, x="attack_vector", y="p_kev_given_av", 
                            hover_data=["total", "kev_count"],
                            title="P(KEV | Attack Vector) — probability that a CVE with this AV is in KEV")
        fig_avprob.update_layout(xaxis_title="Attack Vector (AV)", yaxis_title="P(KEV | AV)", yaxis=dict(tickformat=".2f"))
    else:
        fig_avprob = go.Figure().add_annotation(text="No parsed attack_vector information to compute probabilities", x=0.5, y=0.5, showarrow=False)

    # --------------------------
    # 7) Top cvss_vectorstring by P(KEV | vectorstring) (min support to avoid noise)
    # --------------------------
    if VECTOR_COL in dff.columns:
        vs_counts = (
            dff[[VECTOR_COL, "is_kev"]]
            .dropna(subset=[VECTOR_COL])
            .groupby(VECTOR_COL)
            .agg(total=("is_kev", "count"), kev_count=("is_kev", "sum"))
            .reset_index()
        )
        min_support = 20  # tuneable
        vs_top = vs_counts[vs_counts["total"] >= min_support].copy()
        if not vs_top.empty:
            vs_top["p_kev_given_vector"] = vs_top["kev_count"] / vs_top["total"]
            vs_top = vs_top.sort_values("p_kev_given_vector", ascending=False).head(40)
            fig_vs = px.bar(vs_top, x="p_kev_given_vector", y=VECTOR_COL, orientation="h",
                            hover_data=["total", "kev_count"],
                            title=f"Top {len(vs_top)} cvss_vectorstring by P(KEV | vector) (support >= {min_support})")
            fig_vs.update_layout(xaxis_title="P(KEV | vectorstring)", yaxis_title="cvss_vectorstring")
        else:
            fig_vs = go.Figure().add_annotation(text=f"No vectorstrings with support >= {min_support}", x=0.5, y=0.5, showarrow=False)
    else:
        fig_vs = go.Figure().add_annotation(text=f"No '{VECTOR_COL}' column found", x=0.5, y=0.5, showarrow=False)

    return fig_cross, fig_lag, fig_corr, fig_lead, fig_attack, fig_avprob, fig_vs

# Top risk table callback (unchanged)
@app.callback(
    Output("top-risk-table", "data"),
    Input("cve-search", "value"),
    Input("year-slider", "value"),
    Input("severity-dropdown", "value"),
    Input("kev-filter", "value"),
)
def update_top_risk_table(search_value, year_range, severities, kev_mode):
    dff = filter_df(df, year_range, severities, kev_mode)
    if RISK_SCORE_COL not in dff.columns:
        return []
    dff = dff.dropna(subset=[RISK_SCORE_COL])
    if search_value and "cve_id" in dff.columns:
        mask = dff["cve_id"].astype(str).str.contains(str(search_value), case=False, na=False)
        dff = dff[mask]
    dff = dff.sort_values(RISK_SCORE_COL, ascending=False)
    dff = dff.head(5 if not search_value else 20)
    cols = ["cve_id", RISK_SCORE_COL, "base_score", "severity", "is_kev", "days_to_kev"]
    cols = [c for c in cols if c in dff.columns]
    return dff[cols].to_dict("records")

if __name__ == "__main__":
    app.run(debug=True, port=8050)
