"""
Irrigation ETc Prediction Model — Retraining Script
=====================================================
Retrains the ML model on the corrected dataset (fainal.xlsx)
using only raw environmental inputs (no data leakage).

Target: ETc (crop evapotranspiration)
Model: XGBoost Regressor
Pipeline: sklearn ColumnTransformer + Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import os
import json
import warnings

warnings.filterwarnings("ignore")

# ── 1. Load Dataset ────────────────────────────────────────────────────────────
DATA_PATH = "fainal.xlsx"
print(f"Loading dataset from {DATA_PATH} ...")
df = pd.read_excel(DATA_PATH)

# Rename the messy radiation column for convenience
df.rename(columns={"Net radiation (Rn ) (MJ/m^2/day) ": "Rn"}, inplace=True)

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Drop rows with any missing values (only 1 row has NaN across all columns)
df.dropna(inplace=True)
print(f"\nRows after dropping NaN: {len(df)}")

# ── 2. Feature / Target Separation ────────────────────────────────────────────
#
# RAW MEASURED INPUTS (X):
#   - T2M_MAX, T2M_MIN, T_mean   → Temperature measurements
#   - RH2M                        → Relative humidity
#   - WS2M                        → Wind speed
#   - PS                          → Atmospheric pressure
#   - Rn                          → Net radiation
#   - G                           → Soil heat flux density
#   - FieldCapacity               → θFC (soil property)
#   - WiltingPoint                → θWP (soil property)
#   - RootDepth                   → Root zone depth
#   - GrowthStage                 → Crop growth stage (categorical)
#   - SOIL_TEX                    → Soil texture (categorical)
#   - CROP                        → Crop type (categorical, currently 1 value)
#
# EXCLUDED (calculated / intermediate / output — data leakage if used):
#   - Cp, e, lambda, Y_kPa        → Intermediate PM equation constants
#   - eo, ea, delta               → Derived vapor pressure values
#   - ET0                         → Reference ET (output of PM equation)
#   - KC                          → Crop coefficient (derived from crop + stage)
#   - TAW, RAW                    → Derived from soil props: TAW = 1000*(θFC-θWP)*RootDepth
#   - II_day                      → Calculated irrigation interval
#   - WaterAmount_m3              → Final water requirement (downstream calculation)
#   - Etc                         → TARGET: ETc = ETo × Kc
#
# TARGET (y): ETc
#   Justification: ETc is the core agronomic output that directly represents
#   crop water needs. WaterAmount_m3 depends on site-specific factors (area,
#   irrigation efficiency) that reduce generalizability. ETc is the physically
#   meaningful quantity from which all downstream irrigation decisions derive.

NUMERICAL_FEATURES = [
    "T2M_MAX", "T2M_MIN", "T_mean",
    "RH2M", "WS2M", "PS",
    "Rn", "G",
    "FieldCapacity", "WiltingPoint", "RootDepth",
]

CATEGORICAL_FEATURES = [
    "CROP",
    "GrowthStage",
    "SOIL_TEX",
]

TARGET = "Etc"

FEATURE_COLS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

print(f"\n{'='*60}")
print("FEATURE / TARGET SEPARATION")
print(f"{'='*60}")
print(f"\nNumerical features ({len(NUMERICAL_FEATURES)}):")
for f in NUMERICAL_FEATURES:
    print(f"  - {f}")
print(f"\nCategorical features ({len(CATEGORICAL_FEATURES)}):")
for f in CATEGORICAL_FEATURES:
    print(f"  - {f}")
print(f"\nTarget: {TARGET}")
print(f"\nExcluded columns (data leakage risk):")
excluded = [c for c in df.columns if c not in FEATURE_COLS and c != TARGET]
for c in excluded:
    print(f"  - {c}")

X = df[FEATURE_COLS].copy()
y = df[TARGET].copy()

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"y stats: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")

# ── 3. Preprocessing Pipeline ─────────────────────────────────────────────────
# StandardScaler on numerical features
# OneHotEncoder on categorical features (handles unknown categories gracefully)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ],
    remainder="drop",
)

# ── 4. Model Selection ────────────────────────────────────────────────────────
# XGBoost is chosen over RandomForest because:
#   - Generally superior performance on structured/tabular data
#   - Built-in regularization (L1/L2) reduces overfitting
#   - Handles feature interactions natively via gradient boosting
#   - More efficient training on large datasets (153K+ rows)
#   - Native handling of feature importance

xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

# Full pipeline: preprocessing → model
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", xgb_model),
])

# ── 5. Train / Test Split ─────────────────────────────────────────────────────
# 80/20 split, no data leakage — preprocessing is inside the pipeline

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\n{'='*60}")
print("TRAIN / TEST SPLIT")
print(f"{'='*60}")
print(f"Training samples : {len(X_train):,}")
print(f"Test samples     : {len(X_test):,}")

# ── 6. Training ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("TRAINING XGBoost MODEL")
print(f"{'='*60}")
print("Fitting pipeline (preprocessing + XGBoost) on training data ...")

pipeline.fit(X_train, y_train)

print("Training complete.")

# ── 7. Evaluation ─────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    print(f"\n  [{label}] Evaluation:")
    print(f"    MAE   = {mae:.4f}")
    print(f"    RMSE  = {rmse:.4f}")
    print(f"    R²    = {r2:.4f}")
    print(f"    MAPE  = {mape:.2f}%")
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4), "MAPE": round(mape, 2)}

# Training set evaluation (sanity check)
y_train_pred = pipeline.predict(X_train)
train_metrics = evaluate(y_train, y_train_pred, label="TRAIN")

# Test set evaluation (true performance)
y_test_pred = pipeline.predict(X_test)
test_metrics = evaluate(y_test, y_test_pred, label="TEST")

# Cross-validation on full data (5-fold for robustness)
print(f"\n{'='*60}")
print("5-FOLD CROSS-VALIDATION (on full dataset)")
print(f"{'='*60}")
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2", n_jobs=-1)
print(f"  CV R² scores : {cv_scores}")
print(f"  Mean CV R²   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 8. Feature Importance ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FEATURE IMPORTANCE (XGBoost)")
print(f"{'='*60}")

# Get feature names after preprocessing
ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
all_feature_names = NUMERICAL_FEATURES + cat_feature_names

importances = pipeline.named_steps["regressor"].feature_importances_
sorted_idx = np.argsort(importances)[::-1]

for i in sorted_idx:
    print(f"  {all_feature_names[i]:30s} : {importances[i]:.4f}")

# ── 9. Save Model Artifacts ───────────────────────────────────────────────────
MODEL_DIR = "saved_models_v2"
os.makedirs(MODEL_DIR, exist_ok=True)

# Save the full pipeline (preprocessing + model) — single file, no separate scaler/encoders needed
joblib.dump(pipeline, os.path.join(MODEL_DIR, "pipeline_etc.pkl"))
print(f"\nSaved pipeline → {MODEL_DIR}/pipeline_etc.pkl")

# Save metadata for deployment
metadata = {
    "target": TARGET,
    "numerical_features": NUMERICAL_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "all_features_after_transform": all_feature_names,
    "train_metrics": train_metrics,
    "test_metrics": test_metrics,
    "cv_r2_mean": round(cv_scores.mean(), 4),
    "cv_r2_std": round(cv_scores.std(), 4),
    "feature_importance": {
        all_feature_names[i]: round(float(importances[i]), 4) for i in sorted_idx
    },
    "model_type": "XGBRegressor",
    "training_samples": len(X_train),
    "test_samples": len(X_test),
}
with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved metadata → {MODEL_DIR}/metadata.json")

# ── 10. Bonus: Load & Predict Demo ────────────────────────────────────────────
print(f"\n{'='*60}")
print("DEMO: LOAD SAVED MODEL & PREDICT")
print(f"{'='*60}")

loaded_pipeline = joblib.load(os.path.join(MODEL_DIR, "pipeline_etc.pkl"))

# Create a sample input (using first test row)
sample = X_test.iloc[[0]]
prediction = loaded_pipeline.predict(sample)
actual = y_test.iloc[0]
print(f"\n  Sample input:")
for col in FEATURE_COLS:
    print(f"    {col:20s} = {sample[col].values[0]}")
print(f"\n  Predicted ETc : {prediction[0]:.4f}")
print(f"  Actual ETc    : {actual:.4f}")
print(f"  Error          : {abs(prediction[0] - actual):.4f}")

print(f"\n{'='*60}")
print("RETRAINING COMPLETE — Model ready for deployment")
print(f"{'='*60}")
