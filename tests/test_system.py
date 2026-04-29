"""
Comprehensive Test Suite for Irrigation ETc Prediction System
==============================================================
Tests: T_mean calculation, input validation, ML pipeline, API endpoints,
       data leakage prevention, and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import json
import os

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_DIR = "saved_models_v2"
DATA_PATH = "fainal.xlsx"

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def pipeline():
    """Load the trained pipeline once for all tests."""
    path = os.path.join(MODEL_DIR, "pipeline_etc.pkl")
    if not os.path.exists(path):
        pytest.skip(f"Pipeline not found at {path} — run train_model.py first")
    return joblib.load(path)


@pytest.fixture(scope="session")
def metadata():
    """Load model metadata."""
    path = os.path.join(MODEL_DIR, "metadata.json")
    if not os.path.exists(path):
        pytest.skip(f"Metadata not found at {path}")
    with open(path, "r") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def dataset():
    """Load the dataset once for all tests."""
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Dataset not found at {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)
    df.rename(columns={"Net radiation (Rn ) (MJ/m^2/day) ": "Rn"}, inplace=True)
    df.dropna(inplace=True)
    return df


@pytest.fixture
def sample_input():
    """Valid sample input for prediction."""
    return pd.DataFrame([{
        "T2M_MAX": 32.38,
        "T2M_MIN": 17.6,
        "T_mean": 24.99,
        "RH2M": 53.04,
        "WS2M": 2.76,
        "PS": 99.97,
        "Rn": 13.35,
        "G": 0.0,
        "FieldCapacity": 0.36,
        "WiltingPoint": 0.22,
        "RootDepth": 0.25,
        "CROP": "Spinach",
        "GrowthStage": "Flowering",
        "SOIL_TEX": "CLAY",
    }])


# ══════════════════════════════════════════════════════════════════════════════
# 1. T_mean Calculation Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTMeanCalculation:
    """Test Tavg = (Tmax + Tmin) / 2 formula."""

    def test_basic_calculation(self):
        assert round((35.2 + 18.5) / 2, 2) == 26.85

    def test_equal_values(self):
        assert round((25.0 + 25.0) / 2, 2) == 25.0

    def test_negative_temperatures(self):
        assert round((-5.0 + 5.0) / 2, 2) == 0.0

    def test_both_negative(self):
        assert round((-10.0 + -2.0) / 2, 2) == -6.0

    def test_extreme_values(self):
        result = (60.0 + -40.0) / 2
        assert result == 10.0

    def test_decimal_precision(self):
        result = (32.38 + 17.6) / 2
        assert abs(result - 24.99) < 0.01

    def test_dataset_tmean_consistency(self, dataset):
        """Verify T_mean in dataset matches (T2M_MAX + T2M_MIN) / 2."""
        expected = (dataset["T2M_MAX"] + dataset["T2M_MIN"]) / 2
        diff = (dataset["T_mean"] - expected).abs()
        # Allow small floating point tolerance
        assert (diff < 0.1).all(), f"Max T_mean deviation: {diff.max():.4f}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Input Validation Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestInputValidation:
    """Test validation rules for model inputs."""

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

    def test_valid_ranges(self, dataset):
        """Verify dataset values fall within validation ranges."""
        for col, rule in self.VALIDATION_RULES.items():
            if col in dataset.columns:
                assert dataset[col].min() >= rule["min"] - 1, f"{col} below min"
                assert dataset[col].max() <= rule["max"] + 1, f"{col} above max"

    def test_wilting_point_less_than_field_capacity(self, dataset):
        """WiltingPoint must always be less than FieldCapacity."""
        assert (dataset["WiltingPoint"] < dataset["FieldCapacity"]).all()

    def test_tmax_greater_than_tmin(self, dataset):
        """T2M_MAX must be greater than T2M_MIN."""
        assert (dataset["T2M_MAX"] >= dataset["T2M_MIN"]).all()

    def test_negative_wind_speed(self):
        """Wind speed cannot be negative."""
        assert self.VALIDATION_RULES["WS2M"]["min"] == 0

    def test_humidity_range(self):
        """Humidity must be 0-100%."""
        assert self.VALIDATION_RULES["RH2M"]["min"] == 0
        assert self.VALIDATION_RULES["RH2M"]["max"] == 100


# ══════════════════════════════════════════════════════════════════════════════
# 3. ML Pipeline Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMLPipeline:
    """Test the ML prediction pipeline."""

    def test_pipeline_loads(self, pipeline):
        """Pipeline should load without error."""
        assert pipeline is not None

    def test_pipeline_has_preprocessor(self, pipeline):
        """Pipeline must have a preprocessor step."""
        assert "preprocessor" in pipeline.named_steps

    def test_pipeline_has_regressor(self, pipeline):
        """Pipeline must have a regressor step."""
        assert "regressor" in pipeline.named_steps

    def test_prediction_returns_float(self, pipeline, sample_input):
        """Prediction should return a float value."""
        result = pipeline.predict(sample_input)
        assert isinstance(result[0], (float, np.floating))

    def test_prediction_reasonable_range(self, pipeline, sample_input):
        """ETc prediction should be in a realistic range (0-20 mm/day)."""
        result = pipeline.predict(sample_input)[0]
        assert 0 <= result <= 20, f"ETc = {result} out of realistic range"

    def test_prediction_non_negative(self, pipeline, sample_input):
        """ETc should never be negative."""
        result = pipeline.predict(sample_input)[0]
        assert result >= 0

    def test_multiple_predictions(self, pipeline, dataset, metadata):
        """Pipeline should handle batch predictions."""
        features = metadata["numerical_features"] + metadata["categorical_features"]
        X = dataset[features].iloc[:100]
        predictions = pipeline.predict(X)
        assert len(predictions) == 100
        assert all(p >= 0 for p in predictions)

    def test_preprocessor_has_numerical_scaler(self, pipeline):
        """Preprocessor should include StandardScaler for numerical features."""
        preprocessor = pipeline.named_steps["preprocessor"]
        num_transformer = preprocessor.named_transformers_["num"]
        assert hasattr(num_transformer, "mean_"), "StandardScaler not fitted"

    def test_preprocessor_has_categorical_encoder(self, pipeline):
        """Preprocessor should include OneHotEncoder for categorical features."""
        preprocessor = pipeline.named_steps["preprocessor"]
        cat_transformer = preprocessor.named_transformers_["cat"]
        assert hasattr(cat_transformer, "categories_"), "OneHotEncoder not fitted"


# ══════════════════════════════════════════════════════════════════════════════
# 4. Data Leakage Prevention Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDataLeakagePrevention:
    """Ensure no output/derived columns are used as model inputs."""

    LEAKAGE_COLUMNS = [
        "Cp", "e", "lambda", "Y_kPa",     # Intermediate PM equation constants
        "eo", "ea", "delta",               # Derived vapor pressure values
        "ET0",                              # Reference ET (output of PM equation)
        "KC",                               # Crop coefficient (derived from crop + stage)
        "TAW", "RAW",                       # Derived from soil properties
        "II_day",                           # Calculated irrigation interval
        "WaterAmount_m3",                   # Final water requirement
    ]

    def test_no_leakage_in_features(self, metadata):
        """Model features must NOT include any output/derived columns."""
        features = set(metadata["numerical_features"] + metadata["categorical_features"])
        for col in self.LEAKAGE_COLUMNS:
            assert col not in features, f"LEAKAGE: {col} is in model features!"

    def test_etc_not_in_features(self, metadata):
        """ETc (the target) must NOT be in the feature list."""
        features = set(metadata["numerical_features"] + metadata["categorical_features"])
        assert "Etc" not in features, "LEAKAGE: Target Etc is in features!"

    def test_et0_not_in_features(self, metadata):
        """ETo (reference ET) must NOT be in features — it's a direct calculation output."""
        features = set(metadata["numerical_features"] + metadata["categorical_features"])
        assert "ET0" not in features, "LEAKAGE: ET0 is in features!"

    def test_kc_not_in_features(self, metadata):
        """Kc must NOT be in features — it's derived from crop + growth stage."""
        features = set(metadata["numerical_features"] + metadata["categorical_features"])
        assert "KC" not in features, "LEAKAGE: KC is in features!"

    def test_only_raw_inputs(self, metadata):
        """Verify features are only raw measured environmental variables."""
        expected_num = {"T2M_MAX", "T2M_MIN", "T_mean", "RH2M", "WS2M", "PS", "Rn", "G",
                        "FieldCapacity", "WiltingPoint", "RootDepth"}
        expected_cat = {"CROP", "GrowthStage", "SOIL_TEX"}
        actual_num = set(metadata["numerical_features"])
        actual_cat = set(metadata["categorical_features"])
        assert actual_num == expected_num, f"Numerical mismatch: {actual_num - expected_num}"
        assert actual_cat == expected_cat, f"Categorical mismatch: {actual_cat - expected_cat}"


# ══════════════════════════════════════════════════════════════════════════════
# 5. Model Performance Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestModelPerformance:
    """Verify model meets minimum performance thresholds."""

    def test_r2_above_threshold(self, metadata):
        """Test R² should be above 0.95."""
        assert metadata["test_metrics"]["R2"] >= 0.95, f"R² = {metadata['test_metrics']['R2']}"

    def test_mae_below_threshold(self, metadata):
        """Test MAE should be below 1.0 mm/day."""
        assert metadata["test_metrics"]["MAE"] < 1.0, f"MAE = {metadata['test_metrics']['MAE']}"

    def test_rmse_below_threshold(self, metadata):
        """Test RMSE should be below 1.0 mm/day."""
        assert metadata["test_metrics"]["RMSE"] < 1.0, f"RMSE = {metadata['test_metrics']['RMSE']}"

    def test_cv_r2_consistent(self, metadata):
        """CV R² should be close to test R² (no overfitting)."""
        test_r2 = metadata["test_metrics"]["R2"]
        cv_r2 = metadata["cv_r2_mean"]
        assert abs(test_r2 - cv_r2) < 0.05, f"Test R²={test_r2}, CV R²={cv_r2} — possible overfitting"

    def test_train_test_r2_close(self, metadata):
        """Train and test R² should be close (no severe overfitting)."""
        train_r2 = metadata["train_metrics"]["R2"]
        test_r2 = metadata["test_metrics"]["R2"]
        assert abs(train_r2 - test_r2) < 0.05, f"Train R²={train_r2}, Test R²={test_r2}"


# ══════════════════════════════════════════════════════════════════════════════
# 6. API Endpoint Tests (requires Flask test client)
# ══════════════════════════════════════════════════════════════════════════════

class TestAPIEndpoints:
    """Test Flask API endpoints."""

    @pytest.fixture(scope="class")
    def client(self):
        """Create Flask test client."""
        try:
            from api_v2 import app
            app.config["TESTING"] = True
            with app.test_client() as client:
                yield client
        except ImportError:
            pytest.skip("api_v2 module not importable")

    def test_classes_endpoint(self, client):
        """GET /api/classes should return categorical values."""
        resp = client.get("/api/classes")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "CROP" in data
        assert "GrowthStage" in data
        assert "SOIL_TEX" in data

    def test_predict_valid_input(self, client):
        """POST /api/predict with valid input should return Etc."""
        payload = {
            "T2M_MAX": 32.38, "T2M_MIN": 17.6, "T_mean": 24.99,
            "RH2M": 53.04, "WS2M": 2.76, "PS": 99.97,
            "Rn": 13.35, "G": 0.0,
            "FieldCapacity": 0.36, "WiltingPoint": 0.22, "RootDepth": 0.25,
            "CROP": "Spinach", "GrowthStage": "Flowering", "SOIL_TEX": "CLAY",
        }
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "Etc" in data
        assert data["Etc"] > 0

    def test_predict_missing_feature(self, client):
        """POST /api/predict with missing feature should return 400."""
        payload = {"T2M_MAX": 30}  # Missing most features
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 400

    def test_predict_out_of_range(self, client):
        """POST /api/predict with out-of-range value should return 400."""
        payload = {
            "T2M_MAX": 999, "T2M_MIN": 17.6, "T_mean": 508.3,
            "RH2M": 53.04, "WS2M": 2.76, "PS": 99.97,
            "Rn": 13.35, "G": 0.0,
            "FieldCapacity": 0.36, "WiltingPoint": 0.22, "RootDepth": 0.25,
            "CROP": "Spinach", "GrowthStage": "Flowering", "SOIL_TEX": "CLAY",
        }
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 400

    def test_predict_wp_greater_than_fc(self, client):
        """POST /api/predict with WP >= FC should return 400."""
        payload = {
            "T2M_MAX": 32.38, "T2M_MIN": 17.6, "T_mean": 24.99,
            "RH2M": 53.04, "WS2M": 2.76, "PS": 99.97,
            "Rn": 13.35, "G": 0.0,
            "FieldCapacity": 0.15, "WiltingPoint": 0.35, "RootDepth": 0.25,
            "CROP": "Spinach", "GrowthStage": "Flowering", "SOIL_TEX": "CLAY",
        }
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 400

    def test_metadata_endpoint(self, client):
        """GET /api/metadata should return model metadata."""
        resp = client.get("/api/metadata")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "target" in data
        assert "test_metrics" in data


# ══════════════════════════════════════════════════════════════════════════════
# 7. Edge Case Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_extreme_cold_prediction(self, pipeline):
        """Prediction with very low temperatures should not crash."""
        input_df = pd.DataFrame([{
            "T2M_MAX": -5.0, "T2M_MIN": -15.0, "T_mean": -10.0,
            "RH2M": 80.0, "WS2M": 1.0, "PS": 101.0,
            "Rn": 1.0, "G": 0.0,
            "FieldCapacity": 0.36, "WiltingPoint": 0.22, "RootDepth": 0.25,
            "CROP": "Spinach", "GrowthStage": "Seedling", "SOIL_TEX": "CLAY",
        }])
        result = pipeline.predict(input_df)
        assert not np.isnan(result[0])
        assert not np.isinf(result[0])

    def test_extreme_hot_prediction(self, pipeline):
        """Prediction with very high temperatures should not crash."""
        input_df = pd.DataFrame([{
            "T2M_MAX": 55.0, "T2M_MIN": 35.0, "T_mean": 45.0,
            "RH2M": 10.0, "WS2M": 10.0, "PS": 95.0,
            "Rn": 25.0, "G": 2.0,
            "FieldCapacity": 0.36, "WiltingPoint": 0.22, "RootDepth": 0.25,
            "CROP": "Spinach", "GrowthStage": "Flowering", "SOIL_TEX": "SAND",
        }])
        result = pipeline.predict(input_df)
        assert not np.isnan(result[0])
        assert not np.isinf(result[0])

    def test_zero_radiation(self, pipeline):
        """Prediction with zero radiation should not crash."""
        input_df = pd.DataFrame([{
            "T2M_MAX": 20.0, "T2M_MIN": 10.0, "T_mean": 15.0,
            "RH2M": 60.0, "WS2M": 2.0, "PS": 100.0,
            "Rn": 0.0, "G": 0.0,
            "FieldCapacity": 0.36, "WiltingPoint": 0.22, "RootDepth": 0.25,
            "CROP": "Spinach", "GrowthStage": "Vegetative", "SOIL_TEX": "Silt",
        }])
        result = pipeline.predict(input_df)
        assert not np.isnan(result[0])

    def test_unknown_crop_handled(self, pipeline):
        """Pipeline with handle_unknown='ignore' should not crash on unknown crop."""
        input_df = pd.DataFrame([{
            "T2M_MAX": 30.0, "T2M_MIN": 15.0, "T_mean": 22.5,
            "RH2M": 50.0, "WS2M": 3.0, "PS": 100.0,
            "Rn": 10.0, "G": 0.0,
            "FieldCapacity": 0.36, "WiltingPoint": 0.22, "RootDepth": 0.25,
            "CROP": "UnknownCrop", "GrowthStage": "Flowering", "SOIL_TEX": "CLAY",
        }])
        # Should not raise an error due to handle_unknown="ignore"
        result = pipeline.predict(input_df)
        assert not np.isnan(result[0])

    def test_dataset_no_nulls_in_features(self, dataset, metadata):
        """Feature columns in dataset should have no null values after dropna."""
        features = metadata["numerical_features"] + metadata["categorical_features"]
        for col in features:
            if col in dataset.columns:
                assert dataset[col].notna().all(), f"Nulls found in {col}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
