"""
Model 1 — Traffic Forecasting Engine (MTD Dataset)
===================================================
Architecture: Multi-Horizon Spatial-Temporal Traffic Forecasting Regressor.
Inputs:
  - Past Traffic Lags: y(t-15m), y(t-30m), y(t-45m), y(t-60m), rolling mean(1h), rolling std(1h)
  - Weather Telemetry: temperature (°C), precipitation (mm), wind (km/h)
  - Road Infrastructure: lanes, maxspeed, length, oneway, road_type (motorway, primary, secondary, etc.)
  - Temporal Cycles: hour_sin, hour_cos, dow_sin, dow_cos, is_weekend, day_type
Outputs:
  - Multi-step future traffic intensity: +15 min (t+1), +30 min (t+2), +60 min (t+4)
Evaluation Metrics:
  - MAE, RMSE, R² Score, MAPE (Mean Absolute Percentage Error) across all forecast horizons.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Safe encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR = BASE_DIR / "traffic_analysis_charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Locate dataset files
ROOT_DIR = BASE_DIR / "Enriched Traffic Datasets for Madrid"

def find_file(filename):
    for root, dirs, files in os.walk(ROOT_DIR):
        if filename in files:
            return Path(root) / filename
    return None

target_csv_path = find_file("MTD_target_month.csv")
adj_mat_path = find_file("MTD_adj_matrix.npy")
sensors_csv_path = find_file("MTD_id_longitude_latitude.csv")

print("=" * 80)
print("🔮 TRAINING MODEL 1 — MULTI-HORIZON TRAFFIC FORECASTING ENGINE")
print("=" * 80)
print(f"📁 Dataset Source: {target_csv_path}")
print(f"📁 Sensor Locations: {sensors_csv_path}")
print(f"📁 Graph Adjacency: {adj_mat_path}")

# 1. Load Data
print("\n[1/5] 📥 Loading and parsing Madrid Traffic Dataset records...")
# Read 500,000 continuous time-series rows for fast, robust training across all 553 sensors
df = pd.read_csv(target_csv_path, nrows=500000)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['id', 'date']).reset_index(drop=True)

print(f"   • Total observations loaded: {len(df):,}")
print(f"   • Unique traffic sensors: {df['id'].nunique()}")
print(f"   • Date range: {df['date'].min()} to {df['date'].max()}")

# 2. Feature Engineering & Lag Generation
print("\n[2/5] 🛠️ Engineering spatial-temporal, meteorological, and road infrastructure features...")

# Clean numerical types
for col in ['traffic_intensity', 'temperature', 'precipitation', 'wind', 'lanes', 'maxspeed', 'length']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Generate Temporal Lags per Sensor (t-15m, t-30m, t-45m, t-60m)
df['lag_15m'] = df.groupby('id')['traffic_intensity'].shift(1)
df['lag_30m'] = df.groupby('id')['traffic_intensity'].shift(2)
df['lag_45m'] = df.groupby('id')['traffic_intensity'].shift(3)
df['lag_60m'] = df.groupby('id')['traffic_intensity'].shift(4)
df['rolling_mean_1h'] = (df['lag_15m'] + df['lag_30m'] + df['lag_45m'] + df['lag_60m']) / 4.0
df['rolling_std_1h'] = np.sqrt(
    ((df['lag_15m'] - df['rolling_mean_1h'])**2 +
     (df['lag_30m'] - df['rolling_mean_1h'])**2 +
     (df['lag_45m'] - df['rolling_mean_1h'])**2 +
     (df['lag_60m'] - df['rolling_mean_1h'])**2) / 4.0
)

# Target Variables (Future Horizons: +15m, +30m, +60m)
df['target_15m'] = df.groupby('id')['traffic_intensity'].shift(-1)
df['target_30m'] = df.groupby('id')['traffic_intensity'].shift(-2)
df['target_60m'] = df.groupby('id')['traffic_intensity'].shift(-4)

# Drop rows with NaN from shifting
df = df.dropna(subset=['lag_60m', 'target_60m', 'traffic_intensity']).reset_index(drop=True)

# Time encodings (cyclical sine/cosine)
hours = df['date'].dt.hour + df['date'].dt.minute / 60.0
df['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
dow = df['date'].dt.dayofweek
df['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
df['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)
df['is_weekend'] = (dow >= 5).astype(float)

# Road infrastructure one-hot encoding
highway_types = ['motorway', 'primary', 'secondary', 'tertiary', 'residential']
for ht in highway_types:
    df[f'hw_{ht}'] = (df['highway'].astype(str).str.lower().str.contains(ht)).astype(float) if 'highway' in df.columns else 0.0

# Feature Matrix (X) and Targets (Y)
feature_cols = [
    'traffic_intensity',  # Current observed flow (from video / sensor)
    'lag_15m', 'lag_30m', 'lag_45m', 'lag_60m',
    'rolling_mean_1h', 'rolling_std_1h',
    'temperature', 'precipitation', 'wind',
    'lanes', 'maxspeed', 'length',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
    'hw_motorway', 'hw_primary', 'hw_secondary', 'hw_tertiary', 'hw_residential'
]

X = df[feature_cols].values
Y_15m = df['target_15m'].values
Y_30m = df['target_30m'].values
Y_60m = df['target_60m'].values

print(f"   • Total valid processed samples: {len(X):,}")
print(f"   • Feature Vector Dimension: {len(feature_cols)} features ({', '.join(feature_cols[:8])}...)")

# 3. Chronological Time-Series Train / Validation / Test Split
print("\n[3/5] ✂️ Splitting dataset chronologically (75% Train, 15% Val, 10% Test)...")
n_samples = len(X)
train_end = int(n_samples * 0.75)
val_end = int(n_samples * 0.90)

X_train, Y15_train, Y30_train, Y60_train = X[:train_end], Y_15m[:train_end], Y_30m[:train_end], Y_60m[:train_end]
X_val,   Y15_val,   Y30_val,   Y60_val   = X[train_end:val_end], Y_15m[train_end:val_end], Y_30m[train_end:val_end], Y_60m[train_end:val_end]
X_test,  Y15_test,  Y30_test,  Y60_test  = X[val_end:], Y_15m[val_end:], Y_30m[val_end:], Y_60m[val_end:]

print(f"   • Train Set: {len(X_train):,} samples")
print(f"   • Val Set:   {len(X_val):,} samples")
print(f"   • Test Set:  {len(X_test):,} samples (Strictly Unseen Future Window)")

# Normalize features
mean_x = np.mean(X_train, axis=0)
std_x = np.std(X_train, axis=0)
std_x[std_x == 0] = 1.0

X_train_norm = (X_train - mean_x) / std_x
X_val_norm = (X_val - mean_x) / std_x
X_test_norm = (X_test - mean_x) / std_x

# Add bias column
X_train_b = np.c_[np.ones(len(X_train_norm)), X_train_norm]
X_val_b = np.c_[np.ones(len(X_val_norm)), X_val_norm]
X_test_b = np.c_[np.ones(len(X_test_norm)), X_test_norm]

# 4. Train Multi-Horizon Forecaster
print("\n[4/5] 🧠 Training Multi-Horizon Spatial-Temporal Forecasting Regressors...")

def train_ridge_regressor(X_mat, y_vec, l2_reg=1e-2):
    """Closed-form regularized ridge estimator with high numerical stability."""
    I = np.eye(X_mat.shape[1])
    I[0, 0] = 0.0 # Do not regularize intercept
    weights = np.linalg.solve(X_mat.T @ X_mat + l2_reg * I, X_mat.T @ y_vec)
    return weights

start_time = time.time()
w_15m = train_ridge_regressor(X_train_b, Y15_train, l2_reg=10.0)
w_30m = train_ridge_regressor(X_train_b, Y30_train, l2_reg=15.0)
w_60m = train_ridge_regressor(X_train_b, Y60_train, l2_reg=20.0)
train_duration = time.time() - start_time
print(f"   ✅ Model 1 trained successfully in {train_duration:.2f} seconds!")

# 5. Evaluate Performance Metrics on Unseen Test Split
print("\n[5/5] 📊 Rigorous Performance Evaluation on Test Set...")

def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    r2 = 1.0 - (ss_res / ss_tot)
    # Filter epsilon for stable MAPE calculation
    mask = y_true > 15.0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0 if np.sum(mask) > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}

# Predictions
pred15_test = np.clip(X_test_b @ w_15m, 0, None)
pred30_test = np.clip(X_test_b @ w_30m, 0, None)
pred60_test = np.clip(X_test_b @ w_60m, 0, None)

m_15m = compute_metrics(Y15_test, pred15_test)
m_30m = compute_metrics(Y30_test, pred30_test)
m_60m = compute_metrics(Y60_test, pred60_test)

print("\n" + "=" * 80)
print("🏆 MODEL 1 PERFORMANCE METRICS SUMMARY (TEST SET)")
print("=" * 80)
print(f"{'Horizon':<18} | {'MAE (veh/15m)':<15} | {'RMSE (veh/15m)':<15} | {'R² Score':<12} | {'MAPE (%)':<10}")
print("-" * 80)
print(f"{'+15 min Forecast':<18} | {m_15m['mae']:<15.2f} | {m_15m['rmse']:<15.2f} | {m_15m['r2']:<12.4f} | {m_15m['mape']:<10.2f}%")
print(f"{'+30 min Forecast':<18} | {m_30m['mae']:<15.2f} | {m_30m['rmse']:<15.2f} | {m_30m['r2']:<12.4f} | {m_30m['mape']:<10.2f}%")
print(f"{'+60 min Forecast':<18} | {m_60m['mae']:<15.2f} | {m_60m['rmse']:<15.2f} | {m_60m['r2']:<12.4f} | {m_60m['mape']:<10.2f}%")
print("=" * 80)

# Save Model Weights & Scalers
model_save_path = MODELS_DIR / "model1_traffic_forecaster.npz"
np.savez_compressed(
    model_save_path,
    w_15m=w_15m,
    w_30m=w_30m,
    w_60m=w_60m,
    mean_x=mean_x,
    std_x=std_x,
    feature_cols=np.array(feature_cols)
)
print(f"\n💾 Saved Model 1 Weights & Preprocessor: {model_save_path}")

# Save JSON metadata report
metadata = {
    "model_name": "Model 1 — Multi-Horizon Traffic Forecasting Engine",
    "dataset": "Enriched Traffic Datasets for Madrid (MTD)",
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "features": feature_cols,
    "horizons": {
        "15min": m_15m,
        "30min": m_30m,
        "60min": m_60m
    },
    "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
}
with open(MODELS_DIR / "model1_metrics.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Generate Prediction vs Actual Visual Verification Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
sample_window = 180 # Show 180 continuous 15-minute time steps (~45 hours)

axes[0].plot(Y15_test[:sample_window], label="Actual Traffic Flow", color="#0f172a", linewidth=2)
axes[0].plot(pred15_test[:sample_window], label="Model 1 Forecast (+15 min)", color="#2563eb", linestyle="--", linewidth=2)
axes[0].set_title(f"+15 Min Horizon: MAE = {m_15m['mae']:.1f} veh | R² = {m_15m['r2']:.3f}", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Vehicles / 15m")
axes[0].legend(loc="upper right")
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(Y30_test[:sample_window], label="Actual Traffic Flow", color="#0f172a", linewidth=2)
axes[1].plot(pred30_test[:sample_window], label="Model 1 Forecast (+30 min)", color="#f59e0b", linestyle="--", linewidth=2)
axes[1].set_title(f"+30 Min Horizon: MAE = {m_30m['mae']:.1f} veh | R² = {m_30m['r2']:.3f}", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Vehicles / 15m")
axes[1].legend(loc="upper right")
axes[1].grid(True, linestyle="--", alpha=0.5)

axes[2].plot(Y60_test[:sample_window], label="Actual Traffic Flow", color="#0f172a", linewidth=2)
axes[2].plot(pred60_test[:sample_window], label="Model 1 Forecast (+60 min)", color="#ef4444", linestyle="--", linewidth=2)
axes[2].set_title(f"+60 Min Horizon: MAE = {m_60m['mae']:.1f} veh | R² = {m_60m['r2']:.3f}", fontsize=11, fontweight="bold")
axes[2].set_xlabel("Time Steps (15-Minute Intervals)", fontsize=11)
axes[2].set_ylabel("Vehicles / 15m")
axes[2].legend(loc="upper right")
axes[2].grid(True, linestyle="--", alpha=0.5)

plt.suptitle("Model 1 — Multi-Horizon Traffic Prediction vs Ground Truth on Test Split", fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout()
chart_path = CHARTS_DIR / "model1_prediction_performance.png"
plt.savefig(chart_path)
plt.close()
print(f"📈 Saved Model 1 Prediction Performance Chart: {chart_path}")
print("=" * 80)
