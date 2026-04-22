"""
Weather Prediction System - Model Training Script
Covers: Data Preprocessing, Wrangling, Outlier Handling, Normalization,
        Descriptive Stats, Visualization, ML Models, Hyperparameter Tuning,
        Model Evaluation, and saving best models.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, precision_score, recall_score, f1_score,
                             classification_report)
import os

os.makedirs('static/plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1: Generate Synthetic Real-World-Like Dataset
# (Mimics Kaggle weather dataset structure)
# ─────────────────────────────────────────────
np.random.seed(42)
n = 2000

data = pd.DataFrame({
    'MinTemp':      np.random.normal(12, 6, n),
    'MaxTemp':      np.random.normal(24, 7, n),
    'Humidity':     np.random.normal(65, 20, n).clip(10, 100),
    'WindSpeed':    np.random.normal(20, 10, n).clip(0, 80),
    'Pressure':     np.random.normal(1015, 10, n),
    'Sunshine':     np.random.normal(7, 3, n).clip(0, 14),
    'Cloud':        np.random.randint(0, 9, n).astype(float),
    'WindDir':      np.random.choice(['N','S','E','W','NE','NW','SE','SW'], n),
})

# Target: Temperature (regression) — derived with noise
data['Temperature'] = (0.6 * data['MaxTemp'] + 0.4 * data['MinTemp']
                       + np.random.normal(0, 1.5, n))

# Target: RainTomorrow (classification) — balanced ~40% rain days
rain_prob = 1 / (1 + np.exp(-(
    0.5 + 0.04 * (data['Humidity'] - 65) - 0.03 * (data['Pressure'] - 1015)
    + 0.15 * data['Cloud'] - 0.08 * data['Sunshine']
)))
data['RainTomorrow'] = (np.random.rand(n) < rain_prob).astype(int)

# Inject ~5% missing values for realism
for col in ['Humidity', 'WindSpeed', 'Pressure', 'Sunshine']:
    idx = np.random.choice(n, size=int(0.05 * n), replace=False)
    data.loc[idx, col] = np.nan

# Inject duplicates
data = pd.concat([data, data.sample(50, random_state=1)], ignore_index=True)

print("=" * 55)
print("STEP 1: Raw Dataset")
print(f"Shape: {data.shape}")
print(data.head(3))

# ─────────────────────────────────────────────
# STEP 2: Data Preprocessing & Wrangling
# REQ: Handle missing values, remove duplicates, encode categoricals
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2: Preprocessing & Wrangling")

# Remove duplicates
before_dup = len(data)
data.drop_duplicates(inplace=True)
print(f"Duplicates removed: {before_dup - len(data)}")

# Fill missing values with median (robust to outliers)
num_cols = data.select_dtypes(include=np.number).columns
data[num_cols] = data[num_cols].fillna(data[num_cols].median())
print(f"Missing values after fill: {data.isnull().sum().sum()}")

# Encode categorical variables
le_wind = LabelEncoder()
data['WindDir_enc']  = le_wind.fit_transform(data['WindDir'])

print("Categorical encoding done.")

# ─────────────────────────────────────────────
# STEP 3: Outlier Handling (IQR Method)
# REQ: Show before vs after comparison
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Outlier Handling (IQR)")

outlier_cols = ['Temperature', 'Humidity', 'WindSpeed', 'Pressure']

fig, axes = plt.subplots(2, len(outlier_cols), figsize=(16, 8))
fig.suptitle('Outlier Detection: Before vs After IQR Removal', fontsize=14)

for i, col in enumerate(outlier_cols):
    axes[0, i].boxplot(data[col])
    axes[0, i].set_title(f'{col} - Before')

before_shape = len(data)
for col in outlier_cols:
    Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
    IQR = Q3 - Q1
    data = data[(data[col] >= Q1 - 1.5 * IQR) & (data[col] <= Q3 + 1.5 * IQR)]

print(f"Rows before outlier removal: {before_shape}, after: {len(data)}")

for i, col in enumerate(outlier_cols):
    axes[1, i].boxplot(data[col])
    axes[1, i].set_title(f'{col} - After')

plt.tight_layout()
plt.savefig('static/plots/outliers_before_after.png', dpi=100)
plt.close()
print("Saved: outliers_before_after.png")

# ─────────────────────────────────────────────
# STEP 4: Descriptive Statistics
# REQ: Mean, median, mode, std
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4: Descriptive Statistics")
stats_cols = ['Temperature', 'Humidity', 'WindSpeed', 'Pressure']
stats = data[stats_cols].agg(['mean', 'median', 'std'])
stats.loc['mode'] = data[stats_cols].mode().iloc[0]
print(stats.round(2))

# ─────────────────────────────────────────────
# STEP 5: Data Visualization
# REQ: Heatmap, Histogram, Scatter, Boxplot
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 5: Data Visualization")

numeric_data = data.select_dtypes(include=np.number)

# 5a. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('static/plots/heatmap.png', dpi=100)
plt.close()

# 5b. Temperature Histogram
plt.figure(figsize=(8, 5))
sns.histplot(data['Temperature'], bins=40, kde=True, color='steelblue')
plt.title('Temperature Distribution')
plt.xlabel('Temperature (°C)')
plt.savefig('static/plots/histogram_temperature.png', dpi=100)
plt.close()

# 5c. Humidity vs Temperature Scatter
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='Humidity', y='Temperature',
                hue='RainTomorrow', palette='Set1', alpha=0.5)
plt.title('Humidity vs Temperature')
plt.savefig('static/plots/scatter_humidity_temp.png', dpi=100)
plt.close()

# 5d. Boxplot for outlier detection
plt.figure(figsize=(10, 5))
data[stats_cols].boxplot()
plt.title('Boxplot - Feature Distribution')
plt.savefig('static/plots/boxplot.png', dpi=100)
plt.close()

print("All 4 plots saved.")

# ─────────────────────────────────────────────
# STEP 6: Feature Engineering & Normalization
# REQ: StandardScaler
# ─────────────────────────────────────────────
features = ['MinTemp', 'MaxTemp', 'Humidity', 'WindSpeed',
            'Pressure', 'Sunshine', 'Cloud',
            'WindDir_enc']

X = data[features].copy()
y_temp = data['Temperature']
y_rain = data['RainTomorrow']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

# Save scaler for Flask API
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("\nScaler saved.")

# Train/Test Split
X_train, X_test, yt_train, yt_test = train_test_split(X_scaled, y_temp, test_size=0.2, random_state=42)
_, _, yr_train, yr_test = train_test_split(X_scaled, y_rain, test_size=0.2, random_state=42)

# ─────────────────────────────────────────────
# STEP 7 & 8: Regression Models + Hyperparameter Tuning
# REQ: Linear Regression, Decision Tree, Random Forest + GridSearchCV
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 7-8: Regression Models (Temperature Prediction)")

reg_models = {
    'Linear Regression':    LinearRegression(),
    'Decision Tree Reg':    DecisionTreeRegressor(random_state=42),
    'Random Forest Reg':    RandomForestRegressor(random_state=42, n_jobs=-1),
}

# Hyperparameter tuning for Random Forest Regressor
print("  GridSearchCV on Random Forest Regressor...")
rf_param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
rf_gs = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1),
                     rf_param_grid, cv=3, scoring='r2', n_jobs=-1)
rf_gs.fit(X_train, yt_train)
reg_models['Random Forest Reg'] = rf_gs.best_estimator_
print(f"  Best RF Reg params: {rf_gs.best_params_}")

reg_file_map = {
    'Linear Regression':  'models/reg_linear.pkl',
    'Decision Tree Reg':  'models/reg_tree.pkl',
    'Random Forest Reg':  'models/reg_rf.pkl',
}

reg_results = {}
best_reg_score, best_reg_model, best_reg_name = -np.inf, None, ''

for name, model in reg_models.items():
    model.fit(X_train, yt_train)
    preds = model.predict(X_test)
    r2  = r2_score(yt_test, preds)
    mae = mean_absolute_error(yt_test, preds)
    mse = mean_squared_error(yt_test, preds)
    reg_results[name] = {'R2': round(r2, 4), 'MAE': round(mae, 4), 'MSE': round(mse, 4)}
    print(f"  {name}: R2={r2:.4f}, MAE={mae:.4f}, MSE={mse:.4f}")
    with open(reg_file_map[name], 'wb') as f:
        pickle.dump(model, f)
    if r2 > best_reg_score:
        best_reg_score, best_reg_model, best_reg_name = r2, model, name

print(f"\n  Best Regression Model: {best_reg_name} (R2={best_reg_score:.4f})")
with open('model_temperature.pkl', 'wb') as f:
    pickle.dump(best_reg_model, f)
print("  Saved: all regression models")

# ─────────────────────────────────────────────
# STEP 9: Classification Models + Hyperparameter Tuning
# REQ: Logistic Regression, Decision Tree, Random Forest + GridSearchCV
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 9: Classification Models (Rain Prediction)")

clf_models = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'Decision Tree Clf':   DecisionTreeClassifier(random_state=42),
    'Random Forest Clf':   RandomForestClassifier(random_state=42, n_jobs=-1),
}

# Hyperparameter tuning for Random Forest Classifier
print("  GridSearchCV on Random Forest Classifier...")
rf_clf_gs = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                         {'n_estimators': [50, 100], 'max_depth': [None, 10]},
                         cv=3, scoring='f1', n_jobs=-1)
rf_clf_gs.fit(X_train, yr_train)
clf_models['Random Forest Clf'] = rf_clf_gs.best_estimator_
print(f"  Best RF Clf params: {rf_clf_gs.best_params_}")

clf_file_map = {
    'Logistic Regression': 'models/clf_logistic.pkl',
    'Decision Tree Clf':   'models/clf_tree.pkl',
    'Random Forest Clf':   'models/clf_rf.pkl',
}

clf_results = {}
best_clf_score, best_clf_model, best_clf_name = -np.inf, None, ''

for name, model in clf_models.items():
    model.fit(X_train, yr_train)
    preds = model.predict(X_test)
    acc  = accuracy_score(yr_test, preds)
    prec = precision_score(yr_test, preds, zero_division=0)
    rec  = recall_score(yr_test, preds, zero_division=0)
    f1   = f1_score(yr_test, preds, zero_division=0)
    clf_results[name] = {'Accuracy': round(acc, 4), 'Precision': round(prec, 4),
                         'Recall': round(rec, 4), 'F1': round(f1, 4)}
    print(f"  {name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    with open(clf_file_map[name], 'wb') as f:
        pickle.dump(model, f)
    if f1 > best_clf_score:
        best_clf_score, best_clf_model, best_clf_name = f1, model, name

print(f"\n  Best Classification Model: {best_clf_name} (F1={best_clf_score:.4f})")
with open('model_rain.pkl', 'wb') as f:
    pickle.dump(best_clf_model, f)
print("  Saved: all classification models")

# ─────────────────────────────────────────────
# STEP 10: Model Comparison Plot
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Regression comparison
reg_df = pd.DataFrame(reg_results).T
reg_df['R2'].plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
axes[0].set_title('Regression Model Comparison (R² Score)')
axes[0].set_ylabel('R² Score')
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis='x', rotation=15)

# Classification comparison
clf_df = pd.DataFrame(clf_results).T
clf_df[['Accuracy', 'F1']].plot(kind='bar', ax=axes[1], edgecolor='black')
axes[1].set_title('Classification Model Comparison')
axes[1].set_ylabel('Score')
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('static/plots/model_comparison.png', dpi=100)
plt.close()
print("\nSaved: model_comparison.png")

# Save feature list for API
with open('features.pkl', 'wb') as f:
    pickle.dump(features, f)

print("\n" + "=" * 55)
print("[DONE] Training Complete! All models and plots saved.")
print("=" * 55)
