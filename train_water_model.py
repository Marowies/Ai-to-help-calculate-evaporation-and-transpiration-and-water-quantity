"""
Water Amount Prediction Model — Training Script
=================================================
Trains a separate XGBoost model for WaterAmount_m3 prediction.
Same raw environmental features as ETc model — no data leakage.

Target: WaterAmount_m3
Features: Same 11 numerical + 3 categorical as ETc model
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
df.rename(columns={"Net radiation (Rn ) (MJ/m^2/day) ": "Rn"}, inplace=True)
df.dropna(inplace=True)
print(f"Dataset shape: {df.shape}")

# ── 2. Feature / Target Separation ────────────────────────────────────────────
# Same raw features as ETc model — no leakage
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

TARGET = "WaterAmount_m3"
FEATURE_COLS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

X = df[FEATURE_COLS].copy()
y = df[TARGET].copy()

print(f"X shape: {X.shape}")
print(f"y stats: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")

# ── 3. Preprocessing Pipeline ─────────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ],
    remainder="drop",
)

# ── 4. Model ────────────────────────────────────────────────────────────────────
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

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", xgb_model),
])

# ── 5. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Training samples : {len(X_train):,}")
print(f"Test samples     : {len(X_test):,}")

# ── 6. Training ───────────────────────────────────────────────────────────────
print("Training XGBoost WaterAmount model ...")
pipeline.fit(X_train, y_train)
print("Training complete.")

# ── 7. Evaluation ─────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    print(f"  [{label}] MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, MAPE={mape:.2f}%")
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4), "MAPE": round(mape, 2)}

train_metrics = evaluate(y_train, pipeline.predict(X_train), "TRAIN")
test_metrics = evaluate(y_test, pipeline.predict(X_test), "TEST")

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2", n_jobs=-1)
print(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 8. Feature Importance ─────────────────────────────────────────────────────
ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
all_feature_names = NUMERICAL_FEATURES + cat_feature_names
importances = pipeline.named_steps["regressor"].feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("\nFeature Importance:")
for i in sorted_idx:
    print(f"  {all_feature_names[i]:30s} : {importances[i]:.4f}")

# ── 9. Save Model ─────────────────────────────────────────────────────────────
MODEL_DIR = "saved_models_v2"
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(pipeline, os.path.join(MODEL_DIR, "water_model.pkl"))
print(f"\nSaved water model → {MODEL_DIR}/water_model.pkl")

water_metadata = {
    "target": TARGET,
    "numerical_features": NUMERICAL_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
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
    "area_m2": 4200,
    "conversion_factor": 4.2,
    "note": "WaterAmount_m3 = Etc * 4.2 (area=4200m2, /1000 for mm->m3 conversion)",
}

with open(os.path.join(MODEL_DIR, "water_metadata.json"), "w") as f:
    json.dump(water_metadata, f, indent=2)
print(f"Saved water metadata → {MODEL_DIR}/water_metadata.json")

# ── 10. Demo ───────────────────────────────────────────────────────────────────
sample = X_test.iloc[[0]]
pred = pipeline.predict(sample)[0]
actual = y_test.iloc[0]
print(f"\nDemo: Predicted={pred:.4f}, Actual={actual:.4f}, Error={abs(pred-actual):.4f}")
print("\nWater model training complete!")
