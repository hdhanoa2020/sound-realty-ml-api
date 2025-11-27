# evaluate_model.py
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

HOUSING_PATH = DATA_DIR / "kc_house_data.csv"
DEMOGRAPHICS_PATH = DATA_DIR / "zipcode_demographics.csv"
MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURES_PATH = MODEL_DIR / "model_features.json"


def main():
    print("BASE_DIR:", BASE_DIR)
    print("HOUSING_PATH:", HOUSING_PATH)
    print("DEMOGRAPHICS_PATH:", DEMOGRAPHICS_PATH)
    print("MODEL_PATH:", MODEL_PATH)
    print("FEATURES_PATH:", FEATURES_PATH)

    # ---- Load data ----
    try:
        df = pd.read_csv(HOUSING_PATH)
    except Exception as e:
        print("ERROR reading housing data:", e)
        return

    try:
        demo = pd.read_csv(DEMOGRAPHICS_PATH).set_index("zipcode")
    except Exception as e:
        print("ERROR reading demographics data:", e)
        return

    print("Housing data shape:", df.shape)
    print("Demographics data shape:", demo.shape)

    # ---- Join demographics ----
    if "zipcode" not in df.columns:
        print("ERROR: 'zipcode' column not found in housing data.")
        return

    df = df.join(demo, on="zipcode")
    print("Combined data shape after join:", df.shape)

    # ---- Load feature list ----
    try:
        with open(FEATURES_PATH, "r") as f:
            model_features = json.load(f)
    except Exception as e:
        print("ERROR reading model_features.json:", e)
        return

    print("Number of model features:", len(model_features))
    print("First few features:", model_features[:5])

    # ---- Build X, y ----
    missing = [col for col in model_features if col not in df.columns]
    if missing:
        print("ERROR: These features are missing in df:", missing)
        return

    X = df[model_features]
    if "price" not in df.columns:
        print("ERROR: 'price' column not found in combined data.")
        return

    y = df["price"].values

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Train size:", X_train.shape[0])
    print("Test size:", X_test.shape[0])

    # ---- Load model ----
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print("ERROR loading model.pkl:", e)
        return

    # ---- Evaluate ----
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        print("ERROR during model.predict:", e)
        return

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n===== EVALUATION RESULTS =====")
    print(f"Test RMSE: {rmse:,.2f}")
    print(f"Test R^2:  {r2:.4f}")


if __name__ == "__main__":
    main()
