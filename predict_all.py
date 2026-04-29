import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Load saved artifacts ──────────────────────────────────────────────────────
model_water  = joblib.load('saved_models/model_WaterAmount_m3.pkl')
model_etc    = joblib.load('saved_models/model_Etc.pkl')
scaler       = joblib.load('saved_models/scaler.pkl')
le_crop      = joblib.load('saved_models/le_crop.pkl')
le_growth    = joblib.load('saved_models/le_growth.pkl')
le_soil      = joblib.load('saved_models/le_soil.pkl')
feature_cols = joblib.load('saved_models/feature_cols.pkl')

# ── Load original data ────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_excel('Data.xlsx', header=None, skiprows=2)
col_names = ['T2M_MAX','T2M_MIN','T_mean','RH2M','WS2M','PS','WSC',
             'ALLSKY_SFC_SW_DWN','G','Cp','e','lambda','Y_kPa','eo','ea',
             'delta','ET0','CROP','GrowthStage','KC','Etc','SOIL_TEX',
             'FieldCapacity','WiltingPoint','RootDepth','TAW','RAW','II_day',
             'WaterAmount_m3']
df.columns = col_names
print(f"Total rows: {len(df):,}")

# ── Encode categoricals (handle unseen labels gracefully) ─────────────────────
def safe_encode(encoder, series):
    known = set(encoder.classes_)
    return series.apply(lambda x: encoder.transform([str(x)])[0] if str(x) in known else -1)

df['CROP_enc']        = safe_encode(le_crop,   df['CROP'].astype(str))
df['GrowthStage_enc'] = safe_encode(le_growth, df['GrowthStage'].astype(str))
df['SOIL_TEX_enc']    = safe_encode(le_soil,   df['SOIL_TEX'].astype(str))

# ── Predict on all rows, NaN rows get NaN prediction ─────────────────────────
df['Predicted_WaterAmount_m3'] = np.nan
df['Predicted_Etc']            = np.nan

valid_mask = df[feature_cols].notna().all(axis=1) & (df['CROP_enc'] != -1) \
             & (df['GrowthStage_enc'] != -1) & (df['SOIL_TEX_enc'] != -1)

X_valid   = df.loc[valid_mask, feature_cols].values.astype(float)
X_scaled  = scaler.transform(X_valid)

df.loc[valid_mask, 'Predicted_WaterAmount_m3'] = model_water.predict(X_scaled).round(4)
df.loc[valid_mask, 'Predicted_Etc']            = model_etc.predict(X_scaled).round(4)

skipped = (~valid_mask).sum()
print(f"Predicted : {valid_mask.sum():,} rows")
print(f"Skipped   : {skipped:,} rows (NaN or unknown category)")

# ── Drop helper encoded columns before saving ─────────────────────────────────
df.drop(columns=['CROP_enc','GrowthStage_enc','SOIL_TEX_enc'], inplace=True)

# ── Save to new Excel file ────────────────────────────────────────────────────
out_path = 'Data_with_predictions.xlsx'
print(f"\nSaving to {out_path} ...")
df.to_excel(out_path, index=False)
print("Done!")

# ── Quick accuracy on valid rows ──────────────────────────────────────────────
valid_df = df[valid_mask.values].dropna(subset=['WaterAmount_m3','Predicted_WaterAmount_m3'])
if len(valid_df):
    err_water = (valid_df['WaterAmount_m3'] - valid_df['Predicted_WaterAmount_m3']).abs()
    err_etc   = (valid_df['Etc']            - valid_df['Predicted_Etc']).abs()
    print(f"\nMAE WaterAmount_m3 : {err_water.mean():.4f}")
    print(f"MAE Etc            : {err_etc.mean():.4f}")
