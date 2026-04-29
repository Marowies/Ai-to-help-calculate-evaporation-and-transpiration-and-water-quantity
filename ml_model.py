import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_excel('Data.xlsx', header=None, skiprows=2)

# Column names from row 1 (English)
col_names = ['T2M_MAX', 'T2M_MIN', 'T_mean', 'RH2M', 'WS2M', 'PS', 'WSC', 
             'ALLSKY_SFC_SW_DWN', 'G', 'Cp', 'e', 'lambda', 'Y_kPa', 'eo', 'ea', 
             'delta', 'ET0', 'CROP', 'GrowthStage', 'KC', 'Etc', 'SOIL_TEX', 
             'FieldCapacity', 'WiltingPoint', 'RootDepth', 'TAW', 'RAW', 'II_day', 
             'WaterAmount_m3']
df.columns = col_names

print(f"Dataset shape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst 3 rows:\n{df.head(3)}")

# --- Target columns ---
target1 = 'WaterAmount_m3'  # كمية المياه الواجب اضافتها بالمتر مكعب
target2 = 'Etc'              # البخر نتح الفعلي

# --- Feature Engineering ---
# Encode categorical columns
le_crop = LabelEncoder()
le_growth = LabelEncoder()
le_soil = LabelEncoder()

df['CROP_enc'] = le_crop.fit_transform(df['CROP'].astype(str))
df['GrowthStage_enc'] = le_growth.fit_transform(df['GrowthStage'].astype(str))
df['SOIL_TEX_enc'] = le_soil.fit_transform(df['SOIL_TEX'].astype(str))

# Select features - exclude derived/intermediate columns and targets
# We keep the primary meteorological + crop + soil features
feature_cols = ['T2M_MAX', 'T2M_MIN', 'T_mean', 'RH2M', 'WS2M', 'PS', 'WSC',
                'ALLSKY_SFC_SW_DWN', 'CROP_enc', 'GrowthStage_enc', 'KC', 
                'SOIL_TEX_enc', 'FieldCapacity', 'WiltingPoint', 'RootDepth']

df_clean = df[feature_cols + [target1, target2]].dropna()
print(f"Rows after dropping NaN: {len(df_clean)} (removed {len(df) - len(df_clean)})")

X = df_clean[feature_cols].values.astype(float)
y1 = df_clean[target1].values.astype(float)  # Water Amount
y2 = df_clean[target2].values.astype(float)  # ETc

print(f"\nFeatures used ({len(feature_cols)}):")
for i, f in enumerate(feature_cols):
    print(f"  {i+1}. {f}")

print(f"\nTarget 1: {target1} (كمية المياه الواجب اضافتها)")
print(f"Target 2: {target2} (البخر نتح الفعلي)")

# --- Models to compare ---
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'KNN (k=3)': KNeighborsRegressor(n_neighbors=3),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
}

# --- Use 10-Fold CV ---
loo = KFold(n_splits=10, shuffle=True, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results = {}

for target_name, y in [(target1, y1), (target2, y2)]:
    print(f"\n{'='*60}")
    print(f"TARGET: {target_name}")
    print(f"{'='*60}")
    
    target_results = {}
    
    for name, model in models.items():
        y_pred = cross_val_predict(model, X_scaled, y, cv=loo)
        
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        
        target_results[name] = {
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'R2': round(r2, 4),
            'MAPE%': round(mape, 2),
            'y_true': y.tolist(),
            'y_pred': y_pred.tolist()
        }
        
        print(f"\n  {name}:")
        print(f"    MAE  = {mae:.4f}")
        print(f"    RMSE = {rmse:.4f}")
        print(f"    R²   = {r2:.4f}")
        print(f"    MAPE = {mape:.2f}%")
    
    results[target_name] = target_results

# Find best model for each target
print(f"\n{'='*60}")
print("BEST MODELS SUMMARY")
print(f"{'='*60}")

best_models = {}
for target_name in [target1, target2]:
    best_name = max(results[target_name].keys(), 
                    key=lambda k: results[target_name][k]['R2'])
    best = results[target_name][best_name]
    best_models[target_name] = {'name': best_name, 'metrics': best}
    print(f"\n{target_name}:")
    print(f"  Best Model: {best_name}")
    print(f"  R² = {best['R2']}, RMSE = {best['RMSE']}, MAE = {best['MAE']}")

# --- Save best models trained on full data ---
os.makedirs('saved_models', exist_ok=True)

for target_name, y in [(target1, y1), (target2, y2)]:
    best_name = best_models[target_name]['name']
    best_model = models[best_name]
    best_model.fit(X_scaled, y)
    safe_name = target_name.replace('/', '_')
    joblib.dump(best_model, f'saved_models/model_{safe_name}.pkl')
    print(f"\nSaved best model for {target_name}: {best_name} -> saved_models/model_{safe_name}.pkl")

# Save scaler and encoders (needed for app preprocessing)
joblib.dump(scaler,    'saved_models/scaler.pkl')
joblib.dump(le_crop,   'saved_models/le_crop.pkl')
joblib.dump(le_growth, 'saved_models/le_growth.pkl')
joblib.dump(le_soil,   'saved_models/le_soil.pkl')
joblib.dump(feature_cols, 'saved_models/feature_cols.pkl')
print("\nSaved: scaler, label encoders, feature_cols -> saved_models/")

# Save results for visualization
with open('ml_results.json', 'w') as f:
    json.dump({
        'results': {k: {mk: {kk: vv for kk, vv in mv.items() if kk not in ['y_true', 'y_pred']} 
                        for mk, mv in v.items()} for k, v in results.items()},
        'predictions': {k: {mk: {'y_true': mv['y_true'], 'y_pred': mv['y_pred']} 
                           for mk, mv in v.items()} for k, v in results.items()},
        'best_models': {k: v['name'] for k, v in best_models.items()},
        'feature_cols': feature_cols
    }, f)

# Feature importance from best RF model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_scaled, y1)
importances1 = rf.feature_importances_

rf.fit(X_scaled, y2)
importances2 = rf.feature_importances_

print(f"\nFeature Importance (Random Forest) for {target1}:")
for f, imp in sorted(zip(feature_cols, importances1), key=lambda x: -x[1]):
    print(f"  {f}: {imp:.4f}")

print(f"\nFeature Importance (Random Forest) for {target2}:")
for f, imp in sorted(zip(feature_cols, importances2), key=lambda x: -x[1]):
    print(f"  {f}: {imp:.4f}")

# Save feature importances
fi_data = {
    target1: {f: round(float(imp), 4) for f, imp in zip(feature_cols, importances1)},
    target2: {f: round(float(imp), 4) for f, imp in zip(feature_cols, importances2)}
}
with open('feature_importance.json', 'w') as f:
    json.dump(fi_data, f)

print("\nDone! Results saved.")
