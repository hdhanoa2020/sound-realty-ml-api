# app.py
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURES_PATH = MODEL_DIR / "model_features.json"
DEMOGRAPHICS_PATH = DATA_DIR / "zipcode_demographics.csv"

# ---------- Load model & metadata at startup ----------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(FEATURES_PATH, "r") as f:
    model_features: List[str] = json.load(f)

demographics_df = pd.read_csv(DEMOGRAPHICS_PATH)
# assume there is a 'zipcode' column
demographics_df = demographics_df.set_index("zipcode")

app = FastAPI(title="Sound Realty Housing Price API")


def prepare_features(input_row: Dict[str, Any]) -> pd.DataFrame:
    """
    1. Convert incoming JSON dict to a Series.
    2. Join zipcode demographics on the backend.
    3. Reorder columns to match model_features.
    4. RETURN A DATAFRAME (required by the ColumnTransformer pipeline).
    """
    try:
        # Step 1: base features from the request
        row_series = pd.Series(input_row)

        # Make sure zipcode is present
        if "zipcode" not in row_series:
            raise KeyError("zipcode")

        zipcode_val = row_series["zipcode"]

        # Step 2: join demographics
        if zipcode_val not in demographics_df.index:
            raise ValueError(f"zipcode {zipcode_val} not found in demographics table")

        demo_series = demographics_df.loc[zipcode_val]

        # Combine base + demographics
        full_series = pd.concat([row_series, demo_series])

        # Step 3: keep only the features the model expects, in order
        missing_features = [f for f in model_features if f not in full_series.index]
        if missing_features:
            raise KeyError(f"Missing required features: {missing_features}")

        # 🔴 OLD CODE (for the old, non-pipeline model)
        # ordered_values = full_series[model_features].values.astype(float)
        # return ordered_values.reshape(1, -1)

        # ✅ NEW CODE: return a 1-row DataFrame with the correct columns
        ordered_df = full_series[model_features].to_frame().T
        return ordered_df

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature preparation error: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "ok", "model_features_count": len(model_features)}


@app.post("/predict_full")
def predict_full(payload: Dict[str, Any]):
    """
    Expects a JSON object with all columns from data/future_unseen_examples.csv
    (no demographics – those are joined automatically).
    """
    X = prepare_features(payload)   # <-- now a DataFrame
    try:
        pred = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {str(e)}")

    prediction_value = float(pred[0])

    return {
        "prediction": prediction_value,
        "currency": "USD",
        "model_type": type(model).__name__,
        "requested_zipcode": payload.get("zipcode"),
        "feature_count": len(model_features),
    }


@app.post("/predict_required")
def predict_required(payload: Dict[str, Any]):
    """
    BONUS endpoint:
    Expects only the minimal set of features required by the model
    (those listed in model_features). You still don't send demographics;
    we still join on zipcode if it is part of the features.
    """
    X = prepare_features(payload)   # <-- also a DataFrame here
    try:
        pred = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {str(e)}")

    prediction_value = float(pred[0])

    return {
        "prediction": prediction_value,
        "currency": "USD",
        "model_type": type(model).__name__,
        "input_keys": list(payload.keys()),
        "feature_count": len(model_features),
    }
