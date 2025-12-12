import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objs as go
import numpy as np

#Config
CSV_PATH = "data/model/model_scores.csv"
RISK_SCORE_XGBOOST = "risk_score_xgb"          # XGB risk score
RISK_SCORE_KDE = "risk_score_kde"
HAZARD_RISK_COL = "hazard_risk_score"  # hazard-model risk
VECTOR_COL = "cvss_vectorstring"       # CVSS vectorstring column


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

AV_MAP = {
    "N": "NETWORK",
    "A": "ADJACENT",
    "L": "LOCAL",
    "P": "PHYSICAL",
}

#Load Data
df = pd.read_csv(CSV_PATH, low_memory=False)

try:
    hazard_df=pd.read_csv("data/hazard/hazard_outputs.csv")

    # rename risk_score so we don't overwrite KDE risk_score
    hazard_df = hazard_df.rename(columns={
        "risk_score": HAZARD_RISK_COL
    })

    df = df.merge(hazard_df, on="cve_id", how="left")
    print("Merged hazard_outputs.csv into main dataframe.")
except FileNotFoundError:
    print("hazard_outputs.csv not found — survival plot & columns will be empty.")

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

if RISK_SCORE_KDE in df.columns:
    df[RISK_SCORE_KDE] = pd.to_numeric(df[RISK_SCORE_KDE], errors="coerce")
else:
    df[RISK_SCORE_KDE] = df["base_score"]

if RISK_SCORE_XGBOOST in df.columns:
    df[RISK_SCORE_XGBOOST] = pd.to_numeric(df[RISK_SCORE_XGBOOST], errors="coerce")
else:
    df[RISK_SCORE_XGBOOST] = df["base_score"]

if HAZARD_RISK_COL in df.columns:
    df[HAZARD_RISK_COL] = pd.to_numeric(df[HAZARD_RISK_COL], errors="coerce")

if "predicted_day_to_kev_quantile" in df.columns:
    disp = df["predicted_day_to_kev_quantile"]
    df["predicted_day_to_kev_quantile_display"] = disp.fillna("No Prediction")
    df["predicted_day_to_kev_quantile_numeric"] = pd.to_numeric(
        df["predicted_day_to_kev_quantile"], errors="coerce"
    )

df["is_high_risk"] = df["base_score"] >= 7.0

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

if "repo_publication_lag" in df.columns:
    df["repo_publication_lag"] = pd.to_numeric(df["repo_publication_lag"], errors="coerce")
    df["repo_publication_lag_clean"] = df["repo_publication_lag"].where(
        (df["repo_publication_lag"] >= 0) & (df["repo_publication_lag"] <= 3650)
    )
else:
    df["repo_publication_lag_clean"] = None

for col in ["nvd_base_score", "jvn_base_score", "eu_base_score"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# Parse cvss_vectorstring into components and attack_vector
if VECTOR_COL in df.columns:
    parsed = df[VECTOR_COL].apply(parse_cvss_vector)

    all_keys = set()
    for d in parsed.dropna().tolist():
        if isinstance(d, dict):
            all_keys.update(d.keys())

    for k in sorted(all_keys):
        df[f"vec_{k}"] = parsed.apply(
            lambda x: x.get(k) if isinstance(x, dict) else np.nan
        )

    if "AV" in all_keys:
        df["attack_vector_code"] = df["vec_AV"]
        df["attack_vector"] = df["vec_AV"].map(
            lambda x: AV_MAP.get(x, x) if pd.notna(x) else np.nan
        )
    else:
        df["attack_vector_code"] = np.nan
        df["attack_vector"] = np.nan
else:
    df["attack_vector_code"] = np.nan
    df["attack_vector"] = np.nan

if "cvss_attackvector" in df.columns:
    df["cvss_attackvector"] = (
        df["cvss_attackvector"]
        .astype(str)
        .str.upper()
        .replace({
            "N": "NETWORK",
            "A": "ADJACENT",
            "L": "LOCAL",
            "P": "PHYSICAL",
        })
    )
else:
    df["cvss_attackvector"] = df["attack_vector"]

if "cve_year" in df.columns and df["cve_year"].notna().any():
    year_min = int(df["cve_year"].min())
    year_max = int(df["cve_year"].max())
else:
    year_min, year_max = 2000, 2025

severity_options = sorted(df["severity"].dropna().unique()) if "severity" in df.columns else []

#APP
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
                            marks={y: str(y) for y in range(
                                year_min,
                                year_max + 1,
                                max(1, (year_max - year_min) // 6 or 1)
                            )},
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
                html.Div(
                    style={"width": "260px"},
                    children=[
                        html.Label("Hazard Prediction Filter"),
                        dcc.RadioItems(
                            id="hazard-filter",
                            options=[
                                {"label": "All CVEs", "value": "all"},
                                {"label": "With Hazard Prediction", "value": "hazard"},
                            ],
                            value="all",
                            inline=True,
                        ),
                    ],
                ),
            ],
        ),

        html.Hr(),

        # MAIN charts + survival + CVSS
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
                dcc.Graph(id="attackvector-chart"),
                dcc.Graph(id="hazard-scatter-chart"),
                dcc.Graph(id="attack-vector-kev-prob"),
                dcc.Graph(id="vectorstring-top-kevprob"),
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
                            f"Empty search = show top 5 by {RISK_SCORE_XGBOOST}",
                            style={"fontSize": "12px", "color": "#555"},
                        ),
                    ],
                ),
                dash_table.DataTable(
                    id="top-risk-table",
                    columns=[
                        {"name": "CVE ID", "id": "cve_id"},
                        {"name": "Risk Score (XGB)", "id": RISK_SCORE_XGBOOST},
                        {"name": "Risk Score (KDE)", "id": RISK_SCORE_KDE},
                        #{"name": "Hazard Risk Score", "id": HAZARD_RISK_COL},
                        {
                            "name": "Predicted Days to KEV (Quantile)",
                            "id": "predicted_day_to_kev_quantile_display",
                        },
                        {"name": "Base Score", "id": "base_score"},
                        {"name": "Severity", "id": "severity"},
                        {"name": "Is KEV", "id": "is_kev"},
                        {"name": "Days to KEV (Observed)", "id": "days_to_kev"},
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


def filter_df(df_in, year_range, severities, kev_mode, hazard_mode):
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

    # Hazard prediction filter
    if hazard_mode == "hazard":
        # Keep only rows that actually have a hazard prediction
        if "predicted_day_to_kev_quantile_display" in dff.columns:
            dff = dff[dff["predicted_day_to_kev_quantile_display"] != "No Prediction"]

    return dff


def make_hazard_figure(dff):
    """
    Predicted Days to KEV vs Hazard Model Risk — Critical Non-KEVs,
    using quantile-based prediction; rows with "Never" or "No Prediction"
    are excluded from the plot via the numeric column.
    """
    if HAZARD_RISK_COL not in dff.columns or "predicted_day_to_kev_quantile_numeric" not in dff.columns:
        fig = go.Figure().add_annotation(
            text=f"Need '{HAZARD_RISK_COL}' and 'predicted_day_to_kev_quantile_numeric' columns",
            x=0.5, y=0.5, showarrow=False,
        )
        return fig

    hazard_df = dff.copy()

    if "severity" in hazard_df.columns:
        hazard_df = hazard_df[hazard_df["severity"] == "CRITICAL"]
    if "is_kev" in hazard_df.columns:
        hazard_df = hazard_df[hazard_df["is_kev"] == 0]

    hazard_df = hazard_df.dropna(
        subset=[HAZARD_RISK_COL, "predicted_day_to_kev_quantile_numeric"]
    )

    if hazard_df.empty:
        fig = go.Figure().add_annotation(
            text="No data for Hazard Model plot under current filters",
            x=0.5, y=0.5, showarrow=False,
        )
        return fig

    hover_cols = [
        c for c in [
            "cve_id",
            HAZARD_RISK_COL,
            "predicted_day_to_kev_quantile_display",
        ]
        if c in hazard_df.columns
    ]

    fig = px.scatter(
        hazard_df,
        x=HAZARD_RISK_COL,
        y="predicted_day_to_kev_quantile_numeric",
        color=HAZARD_RISK_COL,
        color_continuous_scale="Reds",
        hover_data=hover_cols,
        labels={
            HAZARD_RISK_COL: "Hazard Model Risk Score",
            "predicted_day_to_kev_quantile_numeric": "Predicted Days to KEV (Quantile Threshold)",
        },
        title="Predicted Days to KEV vs Hazard Model Risk — Critical Non-KEVs",
    )

    x_min = hazard_df[HAZARD_RISK_COL].min()
    x_max = hazard_df[HAZARD_RISK_COL].max()
    span = x_max - x_min if x_max > x_min else 1.0

    q50 = hazard_df[HAZARD_RISK_COL].quantile(0.5)
    q75 = hazard_df[HAZARD_RISK_COL].quantile(0.75)
    q90 = hazard_df[HAZARD_RISK_COL].quantile(0.9)

    for q, label in [(q50, "50%"), (q75, "75%"), (q90, "90%")]:
        fig.add_vline(
            x=q,
            line=dict(color="gray", dash="dash"),
            annotation_text=label,
            annotation_position="top",
        )

    fig.add_vrect(
        x0=x_min,
        x1=q90,
        fillcolor="lightblue",
        opacity=0.08,
        layer="below",
        line_width=0,
    )

    fig.add_vrect(
        x0=q90,
        x1=x_max,
        fillcolor="yellow",
        opacity=0.2,
        layer="below",
        line_width=0,
        annotation_text="High hazard-risk region (>90th percentile)",
        annotation_position="top right",
    )

    fig.update_xaxes(range=[x_min - 0.05 * span, x_max + 0.05 * span])
    fig.update_layout(coloraxis_colorbar=dict(title="Hazard Risk"))

    return fig

#Callbacks
@app.callback(
    Output("cross-listing-chart", "figure"),
    Output("registration-lag-chart", "figure"),
    Output("risk-correlation-chart", "figure"),
    Output("kev-leadtime-chart", "figure"),
    Output("attackvector-chart", "figure"),
    Output("hazard-scatter-chart", "figure"),
    Output("attack-vector-kev-prob", "figure"),
    Output("vectorstring-top-kevprob", "figure"),
    Input("year-slider", "value"),
    Input("severity-dropdown", "value"),
    Input("kev-filter", "value"),
    Input("score-pair", "value"),
    Input("hazard-filter", "value"),
)
def update_charts(year_range, severities, kev_mode, score_pair, hazard_mode):
    dff = filter_df(df, year_range, severities, kev_mode, hazard_mode)

    # Cross listing
    if "cross_listing_count" in dff.columns:
        tmp = dff.copy()
        tmp["Cross-Listed?"] = tmp["cross_listing_count"].ge(2)
        summary = tmp.groupby("Cross-Listed?").agg(CVE_Count=("cve_id", "nunique")).reset_index()
        summary["Cross-Listed?"] = summary["Cross-Listed?"].map({True: "≥ 2 repos", False: "< 2 repos"})
        fig_cross = px.bar(summary, x="Cross-Listed?", y="CVE_Count",
                           title="CVE IDs in ≥ 2 Repositories", text="CVE_Count")
    else:
        fig_cross = go.Figure().add_annotation(text="Missing cross_listing_count", x=0.5, y=0.5, showarrow=False)

    # Registration Lag
    if "repo_publication_lag_clean" in dff.columns:
        lag_df = dff.dropna(subset=["repo_publication_lag_clean"]).copy()

        if "cross_listing_count" in lag_df.columns:
            cross_listed = lag_df[lag_df["cross_listing_count"] >= 2]
            if not cross_listed.empty:
                lag_df = cross_listed

        lag_df = lag_df[lag_df["repo_publication_lag_clean"] >= 0]

        if not lag_df.empty:
            fig_lag = px.histogram(
                lag_df,
                x="repo_publication_lag_clean",
                nbins=40,
                title="Registration Lag (Δ Days)",
            )
            fig_lag.update_layout(
                xaxis_title="Repo Publication Lag (days)",
                yaxis_title="Number of CVEs",
            )
        else:
            fig_lag = go.Figure().add_annotation(
                text="No valid registration lag data under current filters",
                x=0.5, y=0.5, showarrow=False,
            )
    else:
        fig_lag = go.Figure().add_annotation(
            text="Need 'repo_publication_lag' column",
            x=0.5, y=0.5, showarrow=False,
        )

    # Risk Correlation
    pair_map = {
        "nvd_jvn": ("nvd_base_score", "jvn_base_score", "NVD Base", "JVN Score"),
        "nvd_euvd": ("nvd_base_score", "eu_base_score", "NVD Base", "EUVD Base"),
        "jvn_euvd": ("jvn_base_score", "eu_base_score", "JVN Score", "EUVD Base"),
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

    # KEV Lead Time
    kev_df = dff[(dff["is_kev"] == 1) & (dff["is_high_risk"] == True)].dropna(subset=["days_to_kev"])
    kev_df = kev_df[kev_df['severity'] != 'NAN']
    if not kev_df.empty:
        fig_lead = px.box(kev_df, x="severity", y="days_to_kev",
                          title="KEV Lead Time (Days)")
    else:
        fig_lead = go.Figure().add_annotation(text="No KEV lead-time data", x=0.5, y=0.5, showarrow=False)

    # Attack Vector Distribution
    if "cvss_attackvector" in dff.columns and dff["cvss_attackvector"].notna().any():
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
            text="cvss_attackvector / attack_vector data missing", x=0.5, y=0.5, showarrow=False
        )

    # Hazard Model
    fig_hazard = make_hazard_figure(dff)

    # P(KEV | Attack Vector)
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

        fig_avprob = px.bar(
            av_stats,
            x="attack_vector",
            y="p_kev_given_av",
            hover_data=["total", "kev_count"],
            title="P(KEV | Attack Vector)",
        )
        fig_avprob.update_layout(
            xaxis_title="Attack Vector (AV)",
            yaxis_title="P(KEV | AV)",
            yaxis=dict(tickformat=".2f"),
        )
    else:
        fig_avprob = go.Figure().add_annotation(
            text="No parsed attack_vector information to compute probabilities",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    # Top cvss_vectorstring by P(KEV | vectorstring)
    if VECTOR_COL in dff.columns:
        vs_counts = (
            dff[[VECTOR_COL, "is_kev"]]
            .dropna(subset=[VECTOR_COL])
            .groupby(VECTOR_COL)
            .agg(total=("is_kev", "count"), kev_count=("is_kev", "sum"))
            .reset_index()
        )
        min_support = 20
        vs_top = vs_counts[vs_counts["total"] >= min_support].copy()
        if not vs_top.empty:
            vs_top["p_kev_given_vector"] = vs_top["kev_count"] / vs_top["total"]
            vs_top = vs_top.sort_values("p_kev_given_vector", ascending=False).head(40)

            fig_vs = px.bar(
                vs_top,
                x="p_kev_given_vector",
                y=VECTOR_COL,
                orientation="h",
                hover_data=["total", "kev_count"],
                title=f"Top cvss_vectorstring by P(KEV | vector) (support ≥ {min_support})",
            )
            fig_vs.update_layout(
                xaxis_title="P(KEV | vectorstring)",
                yaxis_title="cvss_vectorstring",
            )
        else:
            fig_vs = go.Figure().add_annotation(
                text=f"No vectorstrings with support ≥ {min_support}",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
    else:
        fig_vs = go.Figure().add_annotation(
            text=f"No '{VECTOR_COL}' column found",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    return fig_cross, fig_lag, fig_corr, fig_lead, fig_attack, fig_hazard, fig_avprob, fig_vs


#Callback
@app.callback(
    Output("top-risk-table", "data"),
    Input("cve-search", "value"),
    Input("year-slider", "value"),
    Input("severity-dropdown", "value"),
    Input("kev-filter", "value"),
    Input("hazard-filter", "value"),
)
def update_top_risk_table(search_value, year_range, severities, kev_mode, hazard_mode):
    dff = filter_df(df, year_range, severities, kev_mode, hazard_mode)
    dff = dff.dropna(subset=[RISK_SCORE_XGBOOST])

    if search_value:
        dff = dff[dff["cve_id"].astype(str).str.contains(str(search_value), case=False, na=False)]

    dff = dff.sort_values(RISK_SCORE_XGBOOST, ascending=False)
    dff = dff.head(5 if not search_value else 20)

    keep_cols = [
        "cve_id",
        RISK_SCORE_XGBOOST,
        HAZARD_RISK_COL,
        RISK_SCORE_KDE,
        "predicted_day_to_kev_quantile_display",
        "base_score",
        "severity",
        "is_kev",
        "days_to_kev",
    ]
    keep_cols = [c for c in keep_cols if c in dff.columns]

    return dff[keep_cols].to_dict("records")

if __name__ == "__main__":
    app.run(debug=True, port=8050)

