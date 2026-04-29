"""
Batch Prediction Script (v2)
==============================
Uses the new sklearn Pipeline to predict ETc on the full fainal.xlsx dataset
and saves results to a new Excel file.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings

warnings.filterwarnings("ignore")

# ── Load pipeline & metadata ──────────────────────────────────────────────────
MODEL_DIR = "saved_models_v2"
pipeline = joblib.load(os.path.join(MODEL_DIR, "pipeline_etc.pkl"))

with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as f:
    metadata = json.load(f)

FEATURE_COLS = metadata["numerical_features"] + metadata["categorical_features"]
TARGET = metadata["target"]

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset ...")
df = pd.read_excel("fainal.xlsx")
df.rename(columns={"Net radiation (Rn ) (MJ/m^2/day) ": "Rn"}, inplace=True)
print(f"Total rows: {len(df):,}")

# ── Predict on rows with valid features ───────────────────────────────────────
valid_mask = df[FEATURE_COLS].notna().all(axis=1)

input_df = df.loc[valid_mask, FEATURE_COLS].copy()
predictions = pipeline.predict(input_df)

df["Predicted_Etc"] = np.nan
df.loc[valid_mask, "Predicted_Etc"] = predictions.round(4)

skipped = (~valid_mask).sum()
print(f"Predicted : {valid_mask.sum():,} rows")
print(f"Skipped   : {skipped:,} rows (missing features)")

# ── Save to Excel ─────────────────────────────────────────────────────────────
out_path = "fainal_with_predictions.xlsx"
print(f"\nSaving to {out_path} ...")
df.to_excel(out_path, index=False)
print("Done!")

# ── Quick accuracy on valid rows ──────────────────────────────────────────────
valid_df = df.loc[valid_mask].dropna(subset=[TARGET, "Predicted_Etc"])
if len(valid_df):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = valid_df[TARGET].values
    y_pred = valid_df["Predicted_Etc"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\nAccuracy on full dataset:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")
