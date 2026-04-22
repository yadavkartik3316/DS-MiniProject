# 🌤 Weather Prediction System

A complete end-to-end Machine Learning web application that predicts:
- **Temperature** using Regression models
- **Rainfall (Yes/No)** using Classification models

Built with Python, Flask, scikit-learn, and a modern HTML/CSS/JS frontend.

---

## 📁 Project Structure

```
PROJECT 222/
├── app.py                  # Flask REST API backend
├── train_models.py         # ML training script (all steps)
├── model_temperature.pkl   # Best regression model (auto-generated)
├── model_rain.pkl          # Best classification model (auto-generated)
├── scaler.pkl              # StandardScaler (auto-generated)
├── features.pkl            # Feature list (auto-generated)
├── requirements.txt        # Python dependencies
├── Procfile                # Deployment config (Render/Heroku)
├── README.md
├── templates/
│   └── index.html          # Frontend UI
└── static/
    └── plots/              # Generated visualization images
        ├── heatmap.png
        ├── histogram_temperature.png
        ├── scatter_humidity_temp.png
        ├── boxplot.png
        ├── outliers_before_after.png
        └── model_comparison.png
```

---

## ✅ Requirements Coverage

| # | Requirement | File | Status |
|---|-------------|------|--------|
| 1 | Data Preprocessing (missing values, duplicates, encoding) | train_models.py L40–70 | ✅ |
| 2 | Data Wrangling | train_models.py L40–70 | ✅ |
| 3 | Outlier Handling (IQR, before/after) | train_models.py L73–100 | ✅ |
| 4 | Data Normalization (StandardScaler) | train_models.py L130–145 | ✅ |
| 5 | Descriptive Statistics | train_models.py L103–115 | ✅ |
| 6 | Visualizations (heatmap, histogram, scatter, boxplot) | train_models.py L118–155 | ✅ |
| 7 | 3 Regression + 3 Classification Models | train_models.py L160–230 | ✅ |
| 8 | Hyperparameter Tuning (GridSearchCV) | train_models.py L170–180 | ✅ |
| 9 | Model Evaluation & Comparison | train_models.py L185–230 | ✅ |
| 10 | Best model saved with pickle | train_models.py L195, L225 | ✅ |
| 11 | Flask REST API (/predict-temperature, /predict-rain) | app.py L50–95 | ✅ |
| 12 | Input validation & error handling | app.py L25–55 | ✅ |
| 13 | Frontend UI with all inputs and buttons | templates/index.html | ✅ |
| 14 | Postman testing guide | See below | ✅ |
| 15 | Deployment ready (Procfile, requirements.txt) | Procfile, requirements.txt | ✅ |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (generates .pkl files and plots)
```bash
python train_models.py
```

### 3. Run Flask App
```bash
python app.py
```

### 4. Open Browser
```
http://localhost:5000
```

---

## 🌐 API Endpoints

### POST `/predict-temperature`
Predicts temperature in °C.

**Request Body:**
```json
{
  "min_temp": 12,
  "max_temp": 26,
  "humidity": 65,
  "wind_speed": 20,
  "pressure": 1015,
  "sunshine": 7,
  "evaporation": 5,
  "cloud": 3,
  "wind_dir": "N",
  "location": "Sydney"
}
```

**Response:**
```json
{
  "status": "success",
  "predicted_temperature": 20.34,
  "unit": "°C"
}
```

---

### POST `/predict-rain`
Predicts whether it will rain tomorrow.

**Request Body:** *(same as above)*

**Response:**
```json
{
  "status": "success",
  "rain_tomorrow": "Yes",
  "probability": {
    "no_rain": 32.5,
    "rain": 67.5
  }
}
```

---

## 🧪 Postman Testing Guide

### Setup
1. Download & install [Postman](https://www.postman.com/downloads/)
2. Start the Flask app: `python app.py`

### Test 1 — Predict Temperature
- Method: `POST`
- URL: `http://localhost:5000/predict-temperature`
- Headers: `Content-Type: application/json`
- Body (raw JSON):
```json
{
  "min_temp": 12, "max_temp": 26, "humidity": 65,
  "wind_speed": 20, "pressure": 1015, "sunshine": 7,
  "evaporation": 5, "cloud": 3, "wind_dir": "N", "location": "Sydney"
}
```
- Expected: `200 OK` with `predicted_temperature`

### Test 2 — Predict Rain
- Method: `POST`
- URL: `http://localhost:5000/predict-rain`
- Same body as above
- Expected: `200 OK` with `rain_tomorrow` and `probability`

### Test 3 — Validation Error
- Send `"humidity": 150` (out of range)
- Expected: `400 Bad Request` with `error` message

### Test 4 — Missing Field
- Remove `"pressure"` from body
- Expected: `400 Bad Request` — `"Missing required field: 'pressure'"`

---

## ☁️ Deployment on Render

1. Push project to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt && python train_models.py`
   - **Start Command:** `gunicorn app:app`
5. Deploy!

> **Note:** Run `python train_models.py` as part of the build step so `.pkl` files are generated on the server.

---

## 📊 ML Models Used

### Regression (Temperature)
| Model | Metric |
|-------|--------|
| Linear Regression | R², MAE, MSE |
| Decision Tree Regressor | R², MAE, MSE |
| **Random Forest Regressor** ✅ | R², MAE, MSE |

### Classification (Rain)
| Model | Metric |
|-------|--------|
| Logistic Regression | Accuracy, Precision, Recall, F1 |
| Decision Tree Classifier | Accuracy, Precision, Recall, F1 |
| **Random Forest Classifier** ✅ | Accuracy, Precision, Recall, F1 |

Best models are selected automatically and saved as `.pkl` files.

---

## 🛠 Tech Stack

- **Backend:** Python 3.10+, Flask 3.0
- **ML:** scikit-learn, NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Deployment:** Gunicorn, Render/Heroku
