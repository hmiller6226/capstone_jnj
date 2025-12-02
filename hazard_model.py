import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("C:/Users/jvgat/Downloads/Hazard/df_after_feature_engineering.csv")
date_cols = ['published_date','kev_published']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)

df['time_to_event'] = (df['kev_published'] - df['published_date']).dt.total_seconds() / (60*60*24)
df['event'] = df['kev_present']
df.loc[df['event'] == 0, 'time_to_event'] = (pd.Timestamp.now(tz='UTC') - df.loc[df['event'] == 0, 'published_date']).dt.total_seconds() / (60*60*24)

features = ['base_score', 'repo_publication_lag', 'cross_listing_count','cross_listing_variance', 'cwe_risk_factor']
df_model = df[features + ['time_to_event', 'event']].dropna().copy()

# Apply transformations
df_model['repo_publication_lag_rank'] = df_model['repo_publication_lag'].rank()
df_model['cross_listing_variance_sqrt'] = np.sqrt(df_model['cross_listing_variance'])

# FIXED: Better categorization based on actual data distribution
print("Cross-listing count distribution:")
print(df_model['cross_listing_count'].describe())
print("\nCWE risk factor distribution:")
print(df_model['cwe_risk_factor'].describe())

# Create categories that actually have data
df_model['cross_listing_count_cat'] = pd.cut(
    df_model['cross_listing_count'], 
    bins=[0, 1, 2, float('inf')],  # Adjusted bins
    labels=['1', '2', '3']
)

df_model['cwe_risk_category'] = pd.qcut(
    df_model['cwe_risk_factor'], 
    q=4,  # Reduced to 3 categories
    labels=['low', 'medium', 'high','critical'],
    duplicates='drop'
)

# Verify the new distribution
print("\nNew strata distribution:")
strata_counts = df_model.groupby(['cross_listing_count_cat', 'cwe_risk_category'], observed=True).size()
print(strata_counts)
print(f"\nNumber of valid strata: {len(strata_counts[strata_counts > 0])}")

# Define transformed features and columns to include
transformed_features = ['base_score', 'repo_publication_lag_rank', 
                       'cross_listing_variance_sqrt']

model_cols = transformed_features + ['time_to_event', 'event', 
                                     'cross_listing_count_cat', 'cwe_risk_category']

df_final = df_model[model_cols].copy()

# Fit model with fixed stratification
cph = CoxPHFitter()
cph.fit(df_final, 
        duration_col='time_to_event', 
        event_col='event',
        strata=['cross_listing_count_cat', 'cwe_risk_category'])

print("\n" + "="*50)
print("Model Summary:")
print("="*50)
print(cph.summary)

# Check assumptions
cph.check_assumptions(df_final, 
                      columns=transformed_features, 
                      p_value_threshold=0.05, 
                      show_plots=True)

print(f"\nConcordance index: {cph.concordance_index_}")

df_final[df_final['event'] == 0].groupby( ['cross_listing_count_cat', 'cwe_risk_category'] ).size()

df_final[df_final['event'] == 1].groupby( ['cross_listing_count_cat', 'cwe_risk_category'] ).size()

# ------------------------------
# 1) Keep only non-KEVs
# ------------------------------
df_nonkev = df_final[df_final['event'] == 0].copy()

# ------------------------------
# 2) Compute partial hazard for non-KEVs
# ------------------------------
df_nonkev['risk_score'] = cph.predict_partial_hazard(df_nonkev)

# ------------------------------
# 3) Merge metadata
# ------------------------------
df_nonkev = df_nonkev.merge(
    df[['vendorProject', 'product', 'vulnerabilityName','cve_id','description_nvd','description_jvn','description']],
    left_index=True,
    right_index=True,
    how='left'
)

# ------------------------------
# 4) Create stratum column
# ------------------------------
df_nonkev['stratum'] = (
    df_nonkev['cross_listing_count_cat'].astype(str) + '/' +
    df_nonkev['cwe_risk_category'].astype(str)
)

# ------------------------------
# 5) Focus on selected strata
# ------------------------------
strata_focus = ['2/low', '2/medium', '2/high', '2/critical']
df_focus = df_nonkev[df_nonkev['stratum'].isin(strata_focus)]

# ------------------------------
# 6) Average survival curves per stratum
# ------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

for stratum in strata_focus:
    df_stratum = df_focus[df_focus['stratum'] == stratum]
    
    if df_stratum.empty:
        continue
    
    # Predict survival functions for all rows in this stratum
    surv_funcs = cph.predict_survival_function(df_stratum)
    
    # Compute average survival at each time point
    avg_surv = surv_funcs.mean(axis=1)
    
    ax.plot(avg_surv.index, avg_surv.values, label=stratum, linewidth=2)

ax.set_title("Average Predicted Survival Functions — Medium Cross-Listing (Non-KEVs)")
ax.set_xlabel("Time")
ax.set_ylabel("Average Survival Probability")
ax.legend(title="Stratum")
plt.tight_layout()
plt.show()

sns.histplot(df_focus, x='risk_score', hue='stratum', kde=True, bins=30)
plt.title("Hazard Score Distribution — Medium Cross-Listing (Non-KEVs)")
plt.xlabel("Predicted Partial Hazard (Risk Score)")
plt.ylabel("Count")
plt.show()

for stratum in strata_focus:
    df_stratum = df_focus[df_focus['stratum'] == stratum]
    cumhaz = cph.predict_cumulative_hazard(df_stratum).mean(axis=1)
    plt.plot(cumhaz.index, cumhaz.values, label=stratum)

plt.title("Average Cumulative Hazard — Medium Cross-Listing (Non-KEVs) ")
plt.xlabel("Time")
plt.ylabel("Cumulative Hazard")
plt.legend(title="Stratum")
plt.show()


sns.boxplot(data=df_focus, x='stratum', y='risk_score')
plt.title("Hazard Score Distribution - Medium Cross-Listing (Non-KEVs)")
plt.xlabel("Stratum")
plt.ylabel("Predicted Partial Hazard")
plt.show()

# ------------------------------
# Non-KEVs in 2/critical — Predicted Survival & leveling-off time
# ------------------------------

# 1) Filter non-KEVs in stratum 2/critical
df_critical_nonkev = df_nonkev[df_nonkev['stratum'] == '2/critical'].copy()

if df_critical_nonkev.empty:
    print("No non-KEVs in the '2/critical' stratum.")
else:
    # 2) Predict survival functions for these rows
    surv_funcs = cph.predict_survival_function(df_critical_nonkev)

    # 3) Predicted survival at final observed time
    final_time = surv_funcs.index[-1]
    df_critical_nonkev['predicted_survival_probability'] = surv_funcs.loc[final_time].values

    # 4) Compute "leveling-off" time: last time the survival curve decreases
    leveling_times = []
    for idx in df_critical_nonkev.index:
        surv = surv_funcs[idx]
        decreasing_times = surv[surv.diff() < 0].index  # times where survival decreases
        leveling_times.append(decreasing_times[-1] if len(decreasing_times) > 0 else np.nan)

    df_critical_nonkev['survival plateau date'] = leveling_times

    # 5) Build table
    table = df_critical_nonkev[[
        'cve_id', 'risk_score', 'predicted_survival_probability', 'survival plateau date'
    ]].sort_values('predicted_survival_probability')

    print("\nPredicted Survival & Leveling-Off Time — Non-KEVs in 2/critical:")
    print(table.to_string(index=False))


# --- Step 0: Focus on non-KEVs in critical stratum ---
df_critical_nonkev = df_focus[(df_focus['stratum'] == '2/critical') & (df_focus['event'] == 0)].copy()

# --- Step 1: Compute risk score ---
df_critical_nonkev['risk_score'] = cph.predict_partial_hazard(df_critical_nonkev)

# --- Step 2: Predict survival functions (up to 1691 days) ---
surv_funcs = cph.predict_survival_function(df_critical_nonkev)
surv_funcs = surv_funcs[surv_funcs.index <= 1691]

# --- Step 3a: Fixed survival threshold (absolute risk) ---
fixed_threshold = 0.90  # e.g., survival drops below 90%
predicted_days_fixed = []
for idx in surv_funcs.columns:
    surv_series = surv_funcs[idx]
    below = surv_series[surv_series <= fixed_threshold]
    if not below.empty:
        predicted_days_fixed.append(below.index[0])
    else:
        predicted_days_fixed.append(np.nan)
df_critical_nonkev['predicted_day_to_kev_fixed'] = predicted_days_fixed

# --- Step 3b: Data-driven survival threshold (quantile-based) ---
all_surv_values = surv_funcs.values.flatten()
quantile_threshold = np.quantile(all_surv_values, 0.05)  # 5th percentile
print("Data-driven survival threshold (5th percentile):", quantile_threshold)

predicted_days_quantile = []
for idx in surv_funcs.columns:
    surv_series = surv_funcs[idx]
    below = surv_series[surv_series <= quantile_threshold]
    if not below.empty:
        predicted_days_quantile.append(below.index[0])
    else:
        predicted_days_quantile.append(np.nan)
df_critical_nonkev['predicted_day_to_kev_quantile'] = predicted_days_quantile

# --- Step 4: Final table ---
table = df_critical_nonkev[['cve_id', 'risk_score', 
                            'predicted_day_to_kev_fixed', 
                            'predicted_day_to_kev_quantile', 'description','vendorProject', 'product', 'vulnerabilityName']] \
        .sort_values('risk_score', ascending=False)

print(table.head(50).to_string(index=False))

table.to_csv("hazard_outputs.csv",index=False)

import plotly.express as px

# Prepare a combined dataframe with both threshold predictions
df_plot = df_critical_nonkev.copy()
df_plot['predicted_day_to_kev_fixed'] = df_plot['predicted_day_to_kev_fixed']
df_plot['predicted_day_to_kev_quantile'] = df_plot['predicted_day_to_kev_quantile']

# Interactive scatter plot
fig = px.scatter(
    df_plot,
    x='risk_score',
    y='predicted_day_to_kev_fixed',
    color='risk_score',
    color_continuous_scale='Reds',
    hover_data={
        'cve_id': True,
        'risk_score': True,
        'predicted_day_to_kev_fixed': True,
        'predicted_day_to_kev_quantile': True
    },
    labels={
        'risk_score': 'Predicted Risk Score',
        'predicted_day_to_kev_fixed': 'Predicted Days to KEV (Fixed Threshold)'
    },
    title='Predicted Days to KEV vs Risk Score — Critical Stratum Non-KEVs',
    opacity=0.7
)

# Add quantile threshold lines (optional)
quantiles = [0.5, 0.75, 0.9]
for q in quantiles:
    q_value = df_plot['risk_score'].quantile(q)
    fig.add_vline(x=q_value, line=dict(color='gray', dash='dash'), annotation_text=f"{int(q*100)}%", 
                  annotation_position="top right")

# Highlight high-risk region (>90th percentile)
high_risk_threshold = df_plot['risk_score'].quantile(0.9)
fig.add_vrect(x0=high_risk_threshold, x1=df_plot['risk_score'].max(), 
              fillcolor="yellow", opacity=0.2, layer="below", line_width=0, 
              annotation_text="High-risk region (>90th percentile)", annotation_position="top left")

fig.update_layout(coloraxis_colorbar=dict(title="Risk Score"))
fig.show()
