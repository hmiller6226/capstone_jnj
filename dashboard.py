import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objs as go

# ==============================
# CONFIG
# ==============================
CSV_PATH = "kde_scores.csv"
RISK_SCORE_COL = "risk_score"

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(CSV_PATH, low_memory=False)

# ---- Basic typing / cleanup ----
if "cve_year" in df.columns:
    df["cve_year"] = pd.to_numeric(df["cve_year"], errors="coerce")

if "is_kev" in df.columns:
    df["is_kev"] = pd.to_numeric(df["is_kev"], errors="coerce")
else:
    df["is_kev"] = 0

if "base_score" in df.columns:
    df["base_score"] = pd.to_numeric(df["base_score"], errors="coerce")
else:
    df["base_score"] = 0.0

if RISK_SCORE_COL in df.columns:
    df[RISK_SCORE_COL] = pd.to_numeric(df[RISK_SCORE_COL], errors="coerce")
else:
    df[RISK_SCORE_COL] = df["base_score"]

df["is_high_risk"] = df["base_score"] >= 7.0

# ---- Severity labels ----
def map_severity(score):
    if pd.isna(score):
        return None
    score = float(score)
    if score <= 3.9:
        return "LOW"
    elif score <= 6.9:
        return "MEDIUM"
    elif score <= 8.9:
        return "HIGH"
    else:
        return "CRITICAL"

if "severity" not in df.columns:
    df["severity"] = df["base_score"].apply(map_severity)

# ---- Clean repo lag ----
if "repo_publication_lag" in df.columns:
    df["repo_publication_lag"] = pd.to_numeric(df["repo_publication_lag"], errors="coerce")
    df["repo_publication_lag_clean"] = df["repo_publication_lag"].where(
        (df["repo_publication_lag"] >= 0) & (df["repo_publication_lag"] <= 3650)
    )
else:
    df["repo_publication_lag_clean"] = None

# ---- CVSS base-score columns ----
for col in ["nvd_base_score", "jvn_cvss_score", "euvd_basescore"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---- Attack Vector Cleanup ----
if "cvss_attackvector" in df.columns:
    df["cvss_attackvector"] = df["cvss_attackvector"].astype(str).str.upper().replace({
        "N": "NETWORK",
        "A": "ADJACENT",
        "L": "LOCAL",
        "P": "PHYSICAL",
    })

# ---- Year / Severity options ----
if "cve_year" in df.columns and df["cve_year"].notna().any():
    year_min = int(df["cve_year"].min())
    year_max = int(df["cve_year"].max())
else:
    year_min, year_max = 2000, 2025

severity_options = sorted(df["severity"].dropna().unique()) if "severity" in df.columns else []


# ==============================
# APP
# ==============================
app = Dash(__name__)
app.title = "CVE Risk & Coverage Dashboard"

app.layout = html.Div(
    style={"margin": "20px"},
    children=[
        html.H1("CVE Risk & Coverage Dashboard"),

        # FILTERS
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
                            marks={y: str(y) for y in range(year_min, year_max + 1, max(1, (year_max - year_min) // 6 or 1))},
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

        # MAIN CHARTS (ADDED ATTACK VECTOR CHART)
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
                dcc.Graph(id="attackvector-chart"),  # NEW CHART
            ],
        ),

        html.Hr(),

        # TOP 5 TABLE
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
                        {"name": "Days to KEV", "id": "days_to_kev"},
                    ],
                    page_size=10,
                    sort_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "padding": "5px"},
                    style_header={"backgroundColor": "#f0f0f0", "fontWeight": "bold"},
                ),
            ],
        ),
    ],
)


# ==============================
# FILTER HELPER
# ==============================
def filter_df(df_in, year_range, severities, kev_mode):
    dff = df_in.copy()

    dff = dff[
        (dff["cve_year"] >= year_range[0]) &
        (dff["cve_year"] <= year_range[1])
    ]

    if severities:
        dff = dff[dff["severity"].isin(severities)]

    if kev_mode == "kev":
        dff = dff[dff["is_kev"] == 1]
    elif kev_mode == "nonkev":
        dff = dff[dff["is_kev"] == 0]

    return dff


# ==============================
# CALLBACK — MAIN CHARTS
# ==============================
@app.callback(
    Output("cross-listing-chart", "figure"),
    Output("registration-lag-chart", "figure"),
    Output("risk-correlation-chart", "figure"),
    Output("kev-leadtime-chart", "figure"),
    Output("attackvector-chart", "figure"),
    Input("year-slider", "value"),
    Input("severity-dropdown", "value"),
    Input("kev-filter", "value"),
    Input("score-pair", "value"),
)
def update_charts(year_range, severities, kev_mode, score_pair):
    dff = filter_df(df, year_range, severities, kev_mode)

    # ---------- 1) Cross-Listing ----------
    if "cross_listing_count" in dff.columns:
        tmp = dff.copy()
        tmp["Cross-Listed?"] = tmp["cross_listing_count"].ge(2)
        summary = tmp.groupby("Cross-Listed?").agg(CVE_Count=("cve_id", "nunique")).reset_index()
        summary["Cross-Listed?"] = summary["Cross-Listed?"].map({True: "≥ 2 repos", False: "< 2 repos"})
        fig_cross = px.bar(summary, x="Cross-Listed?", y="CVE_Count",
                           title="CVE IDs in ≥ 2 Repositories", text="CVE_Count")
    else:
        fig_cross = go.Figure().add_annotation(text="Missing cross_listing_count", x=0.5, y=0.5, showarrow=False)

    # ---------- 2) Registration Lag ----------
    lag_df = dff.dropna(subset=["repo_publication_lag_clean"])
    lag_df = lag_df[lag_df["repo_publication_lag_clean"] > 0]

    if not lag_df.empty:
        fig_lag = px.histogram(lag_df, x="repo_publication_lag_clean", nbins=40,
                               title="Registration Lag (Δ Days)")
    else:
        fig_lag = go.Figure().add_annotation(text="No registration-lag data", x=0.5, y=0.5, showarrow=False)

    # ---------- 3) Risk Correlation ----------
    pair_map = {
        "nvd_jvn": ("nvd_base_score", "jvn_cvss_score", "NVD Base", "JVN Score"),
        "nvd_euvd": ("nvd_base_score", "euvd_basescore", "NVD Base", "EUVD Base"),
        "jvn_euvd": ("jvn_cvss_score", "euvd_basescore", "JVN Score", "EUVD Base"),
    }
    x_col, y_col, x_label, y_label = pair_map[score_pair]
    corr_df = dff.dropna(subset=[x_col, y_col])

    if not corr_df.empty:
        fig_corr = px.scatter(
            corr_df, x=x_col, y=y_col, color="severity",
            title=f"{x_label} vs {y_label}", hover_data=["cve_id"]
        )
    else:
        fig_corr = go.Figure().add_annotation(text="No correlation data", x=0.5, y=0.5, showarrow=False)

    # ---------- 4) KEV Lead Time ----------
    kev_df = dff[(dff["is_kev"] == 1) & (dff["is_high_risk"] == True)].dropna(subset=["days_to_kev"])
    if not kev_df.empty:
        fig_lead = px.box(kev_df, x="severity", y="days_to_kev",
                          title="KEV Lead Time (Days)")
    else:
        fig_lead = go.Figure().add_annotation(text="No KEV lead-time data", x=0.5, y=0.5, showarrow=False)

    # ---------- 5) Attack Vector Distribution (NEW) ----------
    if "cvss_attackvector" in dff.columns:
        av_df = dff["cvss_attackvector"].value_counts().reset_index()
        av_df.columns = ["Attack Vector", "Count"]

        fig_attack = px.bar(
            av_df,
            x="Attack Vector",
            y="Count",
            title="Attack Vector Distribution",
            text="Count"
        )
    else:
        fig_attack = go.Figure().add_annotation(
            text="cvss_attackvector column missing", x=0.5, y=0.5, showarrow=False
        )

    return fig_cross, fig_lag, fig_corr, fig_lead, fig_attack


# ==============================
# CALLBACK — TOP TABLE
# ==============================
@app.callback(
    Output("top-risk-table", "data"),
    Input("cve-search", "value"),
    Input("year-slider", "value"),
    Input("severity-dropdown", "value"),
    Input("kev-filter", "value"),
)
def update_top_risk_table(search_value, year_range, severities, kev_mode):
    dff = filter_df(df, year_range, severities, kev_mode)
    dff = dff.dropna(subset=[RISK_SCORE_COL])

    if search_value:
        dff = dff[dff["cve_id"].str.contains(str(search_value), case=False, na=False)]

    dff = dff.sort_values(RISK_SCORE_COL, ascending=False)
    dff = dff.head(5 if not search_value else 20)

    keep_cols = ["cve_id", RISK_SCORE_COL, "base_score", "severity", "is_kev", "days_to_kev"]
    keep_cols = [c for c in keep_cols if c in dff.columns]

    return dff[keep_cols].to_dict("records")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    app.run(debug=True, port=8050)
