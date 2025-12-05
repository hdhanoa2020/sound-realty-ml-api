# 🏠 Sound Realty – ML Home Price Prediction API

A complete end-to-end machine learning prototype that trains a regression model to predict **home sale prices** and exposes the model as a **FastAPI-powered REST API**.  
The service runs locally or inside Docker and is designed as a starting point for deploying ML models as microservices.

---

## 📁 Project Structure



sound-realty-ml-api/
├── app.py                  # FastAPI app exposing the prediction endpoint
├── create_model.py         # Script to train and save the regression model
├── evaluate_model.py       # Script to evaluate the trained model (optional)
├── improve_model.py        # For feature engineering and tuning (optional)
├── test_client.py          # Quick script to test API calls
├── conda_environment.yml   # (Optional) Conda environment specification
├── Dockerfile              # Dockerized version of the API
├── data/
│   └── kc_house_data.csv   # Dataset used for training
└── README.md               # Documentation (this file)




# 🎯 Project Objective

Sound Realty wants to reduce manual effort in estimating property prices.  
This project demonstrates how to:

1. Train an ML regression model using Seattle home-sale data.
2. Save the trained model as an artifact.
3. Serve predictions via a REST API.
4. Deploy locally or packaged inside Docker.

This mirrors real enterprise ML deployment patterns:  
**Train offline → deploy model artifact → serve predictions via a lightweight API.**

---

# ⚙️ Local Development Setup (Non-Docker)

### 1. Clone the repo

```bash
git clone https://github.com/hdhanoa2020/sound-realty-ml-api.git
cd sound-realty-ml-api
````

### 2. Install dependencies

```bash
pip install fastapi uvicorn pandas scikit-learn
```

Or using conda:

```bash
conda env create -f conda_environment.yml
conda activate <env_name>
```

### 3. Train the model

```bash
python create_model.py
```

This generates a saved model file (e.g., `model.pkl`).

### 3.1 . Imrove the  model
```bash
python improve_model.py
```

This generates a saved model file (e.g., `model.pkl`).

### 4. Run the API

```bash
uvicorn app:app --reload
```

Access in browser:

```
http://localhost:8000
```

Interactive API docs (Swagger UI):

```
http://localhost:8000/docs
```

---

# 🐳 Docker Deployment

The Dockerfile runs FastAPI using **Uvicorn** on port **80** inside the container.

### Build the Docker image

```bash
docker build -t sound-realty-ml-api .
```

### Run the container

```bash
docker run -p 80:80 sound-realty-ml-api
```

Now access the API at:

```
http://localhost
http://localhost/docs
```

---

# 📡 API Usage

## ➤ POST `/predict`

### Example Request

```json
 {
    "bedrooms": 3,
    "bathrooms": 1.0,
    "sqft_living": 1180,
    "sqft_lot": 5650,
    "floors": 1.0,
    "sqft_above": 1180,
    "sqft_basement": 0,
    "yr_built": 1955,
    "yr_renovated": 0,
    "lat": 47.5112,
    "long": -122.257,
    "sqft_living15": 1340,
    "sqft_lot15": 5650,
    "zipcode": 98178,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "grade": 7
 }

```

### Example Response

```json
{
  "predicted_price": 625000.0,
  "currency": "USD"
}
```

### 🎯 Notes

* Field names depend on the model features inside `app.py`.
* FastAPI automatically validates types if Pydantic models are used.

---

# 🧪 Testing the API

### 1. Using the provided test client

```bash
python test_client.py
```

### 2. Using curl

```bash
curl -X POST http://localhost/predict \
  -H "Content-Type: application/json" \
  -d '{"bedrooms": 3, "bathrooms": 2, "sqft_living": 1800, "zipcode": 98103}'
```

### 3. Using Python

```python
import requests

payload = {
    "bedrooms": 3,
    "bathrooms": 2,
    "sqft_living": 1800,
    "zipcode": 98103
}

res = requests.post("http://localhost/predict", json=payload)
print(res.json())
```

---






```


