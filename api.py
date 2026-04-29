from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load models and preprocessors once at startup
models = {
    'WaterAmount_m3': joblib.load('saved_models/model_WaterAmount_m3.pkl'),
    'Etc':            joblib.load('saved_models/model_Etc.pkl'),
}
scaler       = joblib.load('saved_models/scaler.pkl')
le_crop      = joblib.load('saved_models/le_crop.pkl')
le_growth    = joblib.load('saved_models/le_growth.pkl')
le_soil      = joblib.load('saved_models/le_soil.pkl')
feature_cols = joblib.load('saved_models/feature_cols.pkl')

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Return valid categorical values for dropdowns."""
    return jsonify({
        'crops':         le_crop.classes_.tolist(),
        'growth_stages': le_growth.classes_.tolist(),
        'soil_types':    le_soil.classes_.tolist(),
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        crop_enc   = int(le_crop.transform([data['CROP']])[0])
        growth_enc = int(le_growth.transform([data['GrowthStage']])[0])
        soil_enc   = int(le_soil.transform([data['SOIL_TEX']])[0])
    except Exception as e:
        return jsonify({'error': f'Invalid category value: {e}'}), 400

    # Build feature vector in the same order as training
    X = np.array([[
        float(data['T2M_MAX']),
        float(data['T2M_MIN']),
        float(data['T_mean']),
        float(data['RH2M']),
        float(data['WS2M']),
        float(data['PS']),
        float(data['WSC']),
        float(data['ALLSKY_SFC_SW_DWN']),
        crop_enc,
        growth_enc,
        float(data['KC']),
        soil_enc,
        float(data['FieldCapacity']),
        float(data['WiltingPoint']),
        float(data['RootDepth']),
    ]])

    X_scaled = scaler.transform(X)

    return jsonify({
        'WaterAmount_m3': round(float(models['WaterAmount_m3'].predict(X_scaled)[0]), 4),
        'Etc':            round(float(models['Etc'].predict(X_scaled)[0]), 4),
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
