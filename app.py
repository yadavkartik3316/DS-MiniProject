"""
Weather Prediction System - Flask Backend API
"""

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

app = Flask(__name__)

# ── Load artifacts ──────────────────────────────────────────────────
def load_artifact(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

scaler   = load_artifact('scaler.pkl')
features = load_artifact('features.pkl')

# Load all 6 trained models
reg_models = {
    'linear_regression':   load_artifact('models/reg_linear.pkl'),
    'decision_tree':       load_artifact('models/reg_tree.pkl'),
    'random_forest':       load_artifact('models/reg_rf.pkl'),
}
clf_models = {
    'logistic_regression': load_artifact('models/clf_logistic.pkl'),
    'decision_tree':       load_artifact('models/clf_tree.pkl'),
    'random_forest':       load_artifact('models/clf_rf.pkl'),
}

WIND_DIRS = {'E': 0, 'N': 1, 'NE': 2, 'NW': 3, 'S': 4, 'SE': 5, 'SW': 6, 'W': 7}

# ── Validation ──────────────────────────────────────────────────────
def validate_and_parse(data):
    required = ['min_temp', 'max_temp', 'humidity', 'wind_speed',
                'pressure', 'sunshine', 'cloud']
    for field in required:
        if field not in data:
            raise ValueError(f"Missing required field: '{field}'")

    values = {k: float(data[k]) for k in required}

    if not (-10 <= values['min_temp'] <= 50):   raise ValueError("min_temp must be between -10 and 50")
    if not (-10 <= values['max_temp'] <= 60):   raise ValueError("max_temp must be between -10 and 60")
    if not (0 <= values['humidity'] <= 100):    raise ValueError("humidity must be between 0 and 100")
    if not (0 <= values['wind_speed'] <= 150):  raise ValueError("wind_speed must be between 0 and 150")
    if not (950 <= values['pressure'] <= 1060): raise ValueError("pressure must be between 950 and 1060")
    if not (0 <= values['sunshine'] <= 14):     raise ValueError("sunshine must be between 0 and 14")
    if not (0 <= values['cloud'] <= 8):         raise ValueError("cloud must be between 0 and 8")

    wind_dir = data.get('wind_dir', 'N').upper()
    if wind_dir not in WIND_DIRS:
        raise ValueError(f"wind_dir must be one of {list(WIND_DIRS.keys())}")

    return [values['min_temp'], values['max_temp'], values['humidity'],
            values['wind_speed'], values['pressure'], values['sunshine'],
            values['cloud'], WIND_DIRS[wind_dir]]

# ── Routes ──────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict-temperature', methods=['POST'])
def predict_temperature():
    try:
        data = request.get_json(force=True)
        if not data: return jsonify({'error': 'No JSON body provided'}), 400

        algo = data.get('algorithm', 'linear_regression')
        if algo not in reg_models:
            return jsonify({'error': f"Invalid algorithm. Choose from: {list(reg_models.keys())}"}), 400

        fv     = validate_and_parse(data)
        scaled = scaler.transform([fv])
        pred   = reg_models[algo].predict(scaled)[0]

        return jsonify({
            'status': 'success',
            'algorithm': algo,
            'predicted_temperature': round(float(pred), 2),
            'unit': '°C'
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/predict-rain', methods=['POST'])
def predict_rain():
    try:
        data = request.get_json(force=True)
        if not data: return jsonify({'error': 'No JSON body provided'}), 400

        algo = data.get('algorithm', 'logistic_regression')
        if algo not in clf_models:
            return jsonify({'error': f"Invalid algorithm. Choose from: {list(clf_models.keys())}"}), 400

        fv     = validate_and_parse(data)
        scaled = scaler.transform([fv])
        pred   = clf_models[algo].predict(scaled)[0]
        proba  = clf_models[algo].predict_proba(scaled)[0]

        return jsonify({
            'status': 'success',
            'algorithm': algo,
            'rain_tomorrow': 'Yes' if pred == 1 else 'No',
            'probability': {
                'no_rain': round(float(proba[0]) * 100, 1),
                'rain':    round(float(proba[1]) * 100, 1)
            }
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/compare', methods=['POST'])
def compare():
    """Run all algorithms and return results for comparison."""
    try:
        data = request.get_json(force=True)
        if not data: return jsonify({'error': 'No JSON body provided'}), 400

        fv     = validate_and_parse(data)
        scaled = scaler.transform([fv])

        reg_results = {}
        for name, model in reg_models.items():
            pred = model.predict(scaled)[0]
            reg_results[name] = round(float(pred), 2)

        clf_results = {}
        for name, model in clf_models.items():
            pred  = model.predict(scaled)[0]
            proba = model.predict_proba(scaled)[0]
            clf_results[name] = {
                'rain_tomorrow': 'Yes' if pred == 1 else 'No',
                'rain_prob': round(float(proba[1]) * 100, 1)
            }

        return jsonify({
            'status': 'success',
            'temperature_comparison': reg_results,
            'rain_comparison': clf_results
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
