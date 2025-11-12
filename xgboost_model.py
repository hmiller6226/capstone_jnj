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

from xgboost import XGBClassifier


# 6️⃣ Train XGBoost model
xgb_model = XGBClassifier(
    n_estimators=300,        # number of boosting rounds (trees)
    learning_rate=0.05,      # shrinkage rate — smaller = slower but more accurate
    max_depth=4,             # controls model complexity
    subsample=0.8,           # row sampling to reduce overfitting
    colsample_bytree=0.8,    # feature sampling per tree
    random_state=42,
    n_jobs=-1,               # use all cores
    reg_lambda=1.0,          # L2 regularization
    reg_alpha=0.0,           # L1 regularization (can tune)
    use_label_encoder=False,
    eval_metric="logloss"    # for binary classification
)

xgb_model.fit(X_train_t, y_train)

# 7️⃣ Predictions and evaluation
y_pred = xgb_model.predict(X_test_t)
y_prob = xgb_model.predict_proba(X_test_t)[:, 1]

# 8️⃣ Evaluation metrics
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📈 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
avg_prec = average_precision_score(y_test, y_prob)
print(f"\nROC AUC: {roc_auc:.4f}")
print(f"Average Precision (PR-AUC): {avg_prec:.4f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    xgb_model,
    pre.fit_transform(X),  # apply preprocessing
    y,
    cv=cv,
    scoring=["roc_auc", "average_precision", "f1"],
    n_jobs=-1
)

print("\n📊 Cross-Validation Results:")
for metric in ["test_roc_auc", "test_average_precision", "test_f1"]:
    print(f"{metric}: {cv_results[metric].mean():.4f} ± {cv_results[metric].std():.4f}")

