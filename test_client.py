# test_client.py
import json
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "future_unseen_examples.csv"

API_URL = "http://127.0.0.1:8000/predict_full"  # endpoint to test


def main():
    print(f"Loading data from: {DATA_PATH}")

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("ERROR: CSV file not found. Please check the path:")
        print(f"  {DATA_PATH}")
        return
    except Exception as e:
        print("ERROR: Failed to read CSV:", e)
        return

    print(f"Loaded {len(df)} rows from CSV.")

    if df.empty:
        print("The CSV has no rows. Nothing to send to the API.")
        return

    # take first 3 rows as examples
    examples = df.head(3).to_dict(orient="records")
    print(f"Prepared {len(examples)} example(s) to send to the API.")

    for i, row in enumerate(examples, start=1):
        print(f"\n--- Example {i} ---")
        print(f"Input zipcode: {row.get('zipcode')}")

        try:
            resp = requests.post(API_URL, json=row, timeout=10)
        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
            continue

        print(f"HTTP status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"Error response body: {resp.text}")
        else:
            try:
                data = resp.json()
            except ValueError:
                print("ERROR: Response is not valid JSON:")
                print(resp.text)
            else:
                print("Response JSON:")
                print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
