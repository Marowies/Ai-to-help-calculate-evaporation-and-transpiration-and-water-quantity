import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── Load saved artifacts ──────────────────────────────────────────────────────
model_water = joblib.load('saved_models/model_WaterAmount_m3.pkl')
model_etc   = joblib.load('saved_models/model_Etc.pkl')
scaler      = joblib.load('saved_models/scaler.pkl')
le_crop     = joblib.load('saved_models/le_crop.pkl')
le_growth   = joblib.load('saved_models/le_growth.pkl')
le_soil     = joblib.load('saved_models/le_soil.pkl')
feature_cols = joblib.load('saved_models/feature_cols.pkl')

# ── Load & preprocess data (same as training) ─────────────────────────────────
df = pd.read_excel('Data.xlsx', header=None, skiprows=2)
col_names = ['T2M_MAX','T2M_MIN','T_mean','RH2M','WS2M','PS','WSC',
             'ALLSKY_SFC_SW_DWN','G','Cp','e','lambda','Y_kPa','eo','ea',
             'delta','ET0','CROP','GrowthStage','KC','Etc','SOIL_TEX',
             'FieldCapacity','WiltingPoint','RootDepth','TAW','RAW','II_day',
             'WaterAmount_m3']
df.columns = col_names

df['CROP_enc']        = le_crop.transform(df['CROP'].astype(str))
df['GrowthStage_enc'] = le_growth.transform(df['GrowthStage'].astype(str))
df['SOIL_TEX_enc']    = le_soil.transform(df['SOIL_TEX'].astype(str))

targets = ['WaterAmount_m3', 'Etc']
df_clean = df[feature_cols + targets].dropna()

X  = df_clean[feature_cols].values.astype(float)
y1 = df_clean['WaterAmount_m3'].values.astype(float)
y2 = df_clean['Etc'].values.astype(float)

# ── Split 80% train / 20% test ────────────────────────────────────────────────
X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    X, y1, y2, test_size=0.20, random_state=42
)

X_train_sc = scaler.transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train samples : {len(X_train):,}")
print(f"Test  samples : {len(X_test):,}")

# ── Evaluate ──────────────────────────────────────────────────────────────────
def report(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    print(f"\n  Target  : {name}")
    print(f"  R²      : {r2:.4f}")
    print(f"  MAE     : {mae:.4f}")
    print(f"  RMSE    : {rmse:.4f}")
    print(f"  MAPE    : {mape:.2f}%")
    return r2

print("\n" + "="*55)
print("  Random Forest — Test Set Results (20% holdout)")
print("="*55)

r2_water = report("WaterAmount_m3", y1_test, model_water.predict(X_test_sc))
r2_etc   = report("Etc",            y2_test, model_etc.predict(X_test_sc))

print("\n" + "="*55)
print(f"  Overall avg R² : {(r2_water + r2_etc) / 2:.4f}")
print("="*55)
