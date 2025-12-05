# improve_model.py
print(">>> Starting improve_model.py ...")  # top-level sanity check

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Paths ---
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

    # ---------- Load data ----------
    try:
        print("Loading housing data...")
        df = pd.read_csv(HOUSING_PATH)
        print("Housing data shape:", df.shape)
    except Exception as e:
        print("ERROR reading housing data:", e)
        return

    try:
        print("Loading demographics data...")
        demo = pd.read_csv(DEMOGRAPHICS_PATH).set_index("zipcode")
        print("Demographics data shape:", demo.shape)
    except Exception as e:
        print("ERROR reading demographics data:", e)
        return

    # Join demographics on zipcode
    if "zipcode" not in df.columns:
        print("ERROR: 'zipcode' column not found in housing data.")
        return

    print("Joining demographics on zipcode ...")
    df = df.join(demo, on="zipcode")
    print("Combined data shape after join:", df.shape)

    # ---------- Define features ----------

    # Numeric columns from kc_house_data (adjust if needed)
    numeric_features = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
        "lat",
        "long",
        "sqft_living15",
        "sqft_lot15",
    ]

    # Demographic numeric columns
    demo_numeric_features = [
        "medn_hshld_incm_amt",
        "medn_incm_per_prsn_amt",
        "hous_val_amt",
        "per_urbn",
        "per_sbrbn",
        "per_farm",
        "per_non_farm",
        "per_less_than_9",
        "per_9_to_12",
        "per_hsd",
        "per_some_clg",
        "per_assoc",
        "per_bchlr",
        "per_prfsnl"
    ]


    categorical_features = [
        "zipcode",
        "waterfront",
        "view",
        "condition",
        "grade",
    ]

    model_features = numeric_features + demo_numeric_features + categorical_features

    print("Total number of model features:", len(model_features))
    print("First few features:", model_features[:10])

    # Check all features exist
    missing = [col for col in model_features if col not in df.columns]
    if missing:
        print("ERROR: These features are missing in the dataframe:")
        print(missing)
        return

    # ---------- Build X, y ----------
    if "price" not in df.columns:
        print("ERROR: 'price' column not found in data.")
        return

    X = df[model_features]    # keep as DataFrame for ColumnTransformer
    y = df["price"].values

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Train/test split
    print("Splitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Train size:", X_train.shape[0])
    print("Test size:", X_test.shape[0])

    # ---------- Preprocessing ----------
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features + demo_numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # ---------- Model ----------
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # ---------- Training ----------
    print("Fitting model...")
    clf.fit(X_train, y_train)
    print("Model fit complete.")

    # ---------- Cross-validation ----------
    print("Running 3-fold cross-validation on training set (R^2)...")
    try:
        cv_scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="r2")
        print("CV R^2 scores:", cv_scores)
        print(f"CV R^2 mean: {cv_scores.mean():.4f}")
    except Exception as e:
        print("ERROR during cross_val_score:", e)

    # ---------- Evaluation ----------
    print("Evaluating on test set...")
    y_pred = clf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n===== IMPROVED MODEL RESULTS =====")
    print(f"Test RMSE: {rmse:,.2f}")
    print(f"Test R^2:  {r2:.4f}")

    # ---------- Save model & features ----------
    MODEL_DIR.mkdir(exist_ok=True)

    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(clf, f)
        print(f"Saved improved model to: {MODEL_PATH}")
    except Exception as e:
        print("ERROR saving model:", e)
        return

    try:
        with open(FEATURES_PATH, "w") as f:
            json.dump(model_features, f)
        print(f"Saved model feature list to: {FEATURES_PATH}")
    except Exception as e:
        print("ERROR saving model_features.json:", e)
        return

    print(">>> improve_model.py finished successfully.")


if __name__ == "__main__":
    print(">>> Entering main() ...")
    main()
