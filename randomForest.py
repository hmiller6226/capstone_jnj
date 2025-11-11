# === random_forest_top5_features.py ===

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate  # ← ADD THIS
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    average_precision_score, roc_auc_score
)

# 1️⃣ Load dataset
df = pd.read_csv("df_after_feature_engineering.csv", low_memory=False)
print(f"Loaded {len(df):,} rows")

# 2️⃣ Ensure label exists
assert "is_kev" in df.columns, "Expected column 'is_kev'"
df["isKEV"] = df["is_kev"].astype(int)
print(df["isKEV"].value_counts())

# 3️⃣ Select features
features = [
    "base_score",
    "repo_publication_lag",
    "cross_listing_count",
    "cross_listing_variance",
    "cwe_risk_factor",
]

X = df[features].copy()
y = df["isKEV"]

# Add this after step 2 (after loading data)
print("\n📊 Feature Correlation with KEV:")
print("="*60)
for feat in features:
    corr = df[feat].corr(df['isKEV'])
    stars = "★" * int(abs(corr) * 10) if abs(corr) > 0.1 else "⚠️ WEAK"
    print(f"  {feat:30s}: {corr:7.4f}  {stars}")
print("="*60)


# 4️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# 5️⃣ Preprocessing (impute missing + scale)
pre = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
X_train_t = pre.fit_transform(X_train)
X_test_t = pre.transform(X_test)

# 6️⃣ Random Forest training
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=5,
    min_samples_split=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_t, y_train)

# ═══════════════════════════════════════════════════════════════════════════
# ✨ CROSS-VALIDATION (ADD THIS SECTION HERE - AFTER TRAINING, BEFORE EVAL) ✨
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("🔄 5-FOLD CROSS-VALIDATION")
print("="*60)

# Create stratified k-fold splitter
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_results = cross_validate(
    rf,                  # Your trained model
    X_train_t,          # Training features
    y_train,            # Training labels
    cv=skf,             # 5-fold stratified
    scoring=['precision', 'recall', 'f1', 'roc_auc'],  # Metrics
    n_jobs=-1,          # Use all cores
    return_train_score=False  # We only care about validation scores
)

# Print results
print("\nCross-Validation Results (Mean ± Std):")
print("-" * 60)
for metric in ['precision', 'recall', 'f1', 'roc_auc']:
    scores = cv_results[f'test_{metric}']
    mean_score = scores.mean()
    std_score = scores.std()
    print(f"  {metric.upper():12s}: {mean_score:.4f} ± {std_score:.4f}")
    
    # Print individual fold scores
    print(f"               Folds: ", end="")
    for i, score in enumerate(scores, 1):
        print(f"[{i}:{score:.3f}] ", end="")
    print()  # New line

print("="*60)

# ═══════════════════════════════════════════════════════════════════════════
# ✨ END OF CROSS-VALIDATION SECTION ✨
# ═══════════════════════════════════════════════════════════════════════════

# 7️⃣ Evaluation (continues as before)
probs = rf.predict_proba(X_test_t)[:, 1]
preds = (probs >= 0.5).astype(int)

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, preds))

print("\n📋 Classification Report (KEV = positive):")
print(classification_report(y_test, preds, digits=3))

ap = average_precision_score(y_test, probs)
roc = roc_auc_score(y_test, probs)
print(f"Average Precision (PR-AUC): {ap:.3f}")
print(f"ROC-AUC: {roc:.3f}")

# 🔟 KEV Capture @ top-k%
test_eval = pd.DataFrame({
    "isKEV": y_test.values,
    "probability": probs
}).sort_values("probability", ascending=False)

for pct in (0.01, 0.02, 0.05, 0.10):
    k = max(1, int(len(test_eval) * pct))
    capture = test_eval.head(k)["isKEV"].mean() * 100
    print(f"KEV capture in top {int(pct*100)}%: {capture:.2f}%")

# 11️⃣ Save predictions on full dataset
X_all_t = pre.transform(X)
df["kev_probability"] = rf.predict_proba(X_all_t)[:, 1]
df["kev_rank_pct"] = (
    df["kev_probability"].rank(method="average", ascending=False) / len(df)
) * 100

df.sort_values("kev_probability", ascending=False).to_csv(
    "randomforest_scored_top5.csv", index=False
)

print("\n✅ Saved: randomforest_scored_top5.csv")
