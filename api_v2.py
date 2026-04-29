"""
Flask API for Irrigation Prediction (v2)
==========================================
Uses two sklearn Pipelines (ETc + WaterAmount) trained on fainal.xlsx.
Returns both ETc (mm/day) and Water Amount (m³).
Falls back to manual calculation if Water model fails.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json
import os
import math

app = Flask(__name__)
CORS(app)

# ── Load pipelines & metadata ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models_v2")

# ETc model
etc_pipeline = joblib.load(os.path.join(MODEL_DIR, "pipeline_etc.pkl"))
with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as f:
    etc_metadata = json.load(f)

# Water model
water_pipeline = None
water_metadata = None
water_model_path = os.path.join(MODEL_DIR, "water_model.pkl")
if os.path.exists(water_model_path):
    water_pipeline = joblib.load(water_model_path)
    with open(os.path.join(MODEL_DIR, "water_metadata.json"), "r") as f:
        water_metadata = json.load(f)
    print("Water model loaded successfully.")
else:
    print("WARNING: water_model.pkl not found — will use fallback formula only.")

CATEGORICAL_FEATURES = etc_metadata["categorical_features"]
NUMERICAL_FEATURES = etc_metadata["numerical_features"]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Area constant (m²) — from dataset analysis: WaterAmount_m3 = Etc * Area/1000
AREA_M2 = 4200

# Extract valid categorical values from the OneHotEncoder inside the pipeline
ohe = etc_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
cat_values = {}
for i, cat_col in enumerate(CATEGORICAL_FEATURES):
    cat_values[cat_col] = ohe.categories_[i].tolist()


@app.route("/api/classes", methods=["GET"])
def get_classes():
    """Return valid categorical values for dropdowns."""
    return jsonify(cat_values)


VALIDATION_RULES = {
    "T2M_MAX":       {"min": -40, "max": 60},
    "T2M_MIN":       {"min": -40, "max": 60},
    "T_mean":        {"min": -40, "max": 60},
    "RH2M":          {"min": 0,   "max": 100},
    "WS2M":          {"min": 0,   "max": 15},
    "PS":            {"min": 50,  "max": 110},
    "Rn":            {"min": -5,  "max": 30},
    "G":             {"min": -5,  "max": 5},
    "FieldCapacity": {"min": 0,   "max": 1},
    "WiltingPoint":  {"min": 0,   "max": 1},
    "RootDepth":     {"min": 0,   "max": 3},
}


def calculate_water_fallback(etc_value, field_capacity, wilting_point, root_depth):
    """
    Calculate Water Amount using the agronomic fallback formula:

    TAW = 1000 * (FC - WP) * RootDepth   (mm)
    RAW = TAW * 0.4                       (mm)
    Irrigation Interval = RAW / ETc        (days)
    Water = TAW / Irrigation Interval      (mm/day)
    """
    if etc_value <= 0:
        return 0.0

    taw = 1000.0 * (field_capacity - wilting_point) * root_depth
    if taw <= 0:
        taw = 1.0  # fallback to avoid division by zero
        
    raw = taw * 0.4
    if raw <= 0:
        raw = 0.4
        
    irrigation_interval = raw / etc_value
    if irrigation_interval <= 0:
        irrigation_interval = 1.0
        
    water = taw / irrigation_interval

    return round(water, 4)

def predict_irrigation(input_df, etc_model, water_model, fallback_kwargs):
    etc = etc_model.predict(input_df)[0]
    
    water = None
    water_source = "ml_model"
    
    if water_model is not None:
        try:
            water = water_model.predict(input_df)[0]
            if water < 0:
                water = 0.0
        except Exception as e:
            print(f"Water model prediction failed: {e}")
            water = None
            
    if water is None or math.isnan(water) or math.isinf(water):
        water = calculate_water_fallback(etc, **fallback_kwargs)
        water_source = "fallback_formula"
        
    return {
        "ETc": float(etc),
        "Water": float(water),
        "water_source": water_source
    }


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict ETc and Water Amount from raw environmental inputs."""
    data = request.json

    try:
        # Build input DataFrame with correct column names and types
        input_dict = {}
        errors = []

        for col in NUMERICAL_FEATURES:
            if col not in data or data[col] == "" or data[col] is None:
                errors.append(f"Missing feature: {col}")
                continue
            val = float(data[col])
            rule = VALIDATION_RULES.get(col, {})
            if rule and (val < rule["min"] or val > rule["max"]):
                errors.append(f"{col} = {val} out of range [{rule['min']}, {rule['max']}]")
            input_dict[col] = [val]

        for col in CATEGORICAL_FEATURES:
            if col not in data or data[col] == "" or data[col] is None:
                errors.append(f"Missing feature: {col}")
                continue
            input_dict[col] = [str(data[col])]

        if errors:
            return jsonify({"error": "; ".join(errors)}), 400

        # Sanity: T_mean should roughly equal (T2M_MAX + T2M_MIN) / 2
        t_max = input_dict["T2M_MAX"][0]
        t_min = input_dict["T2M_MIN"][0]
        t_mean = input_dict["T_mean"][0]
        expected_tmean = (t_max + t_min) / 2
        if abs(t_mean - expected_tmean) > 5.0:
            return jsonify({"error": f"T_mean ({t_mean}) is far from expected ({expected_tmean:.2f}). Check inputs."}), 400

        # Sanity: WiltingPoint must be less than FieldCapacity
        fc = input_dict["FieldCapacity"][0]
        wp = input_dict["WiltingPoint"][0]
        if wp >= fc:
            return jsonify({"error": f"WiltingPoint ({wp}) must be less than FieldCapacity ({fc})"}), 400

        input_df = pd.DataFrame(input_dict)

        # ── Predict using the prediction function ─────────────────────────────
        fallback_kwargs = {
            "field_capacity": fc,
            "wilting_point": wp,
            "root_depth": input_dict["RootDepth"][0]
        }
        
        result = predict_irrigation(input_df, etc_pipeline, water_pipeline, fallback_kwargs)

        return jsonify({
            "ETc": round(result["ETc"], 4),
            "Water": round(result["Water"], 4),
            "water_source": result["water_source"],
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid number: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    """Return model metadata (features, metrics, etc.)."""
    response = {**etc_metadata}
    if water_metadata:
        response["water_model"] = water_metadata
    return jsonify(response)


if __name__ == "__main__":
    import os
    # Use the port assigned by Render/Cloud or 5000 for local development
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
