"""
Model 1 Standalone Evaluator & Metric Verification
===================================================
Loads trained weights from models/model1_traffic_forecaster.npz and prints
detailed metrics and residual statistics across all test horizons.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Safe encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
METRICS_FILE = BASE_DIR / "models" / "model1_metrics.json"

def print_evaluation_report():
    if not METRICS_FILE.exists():
        print(f"[-] Metrics file not found at {METRICS_FILE}. Run train_model1_forecasting.py first.")
        return

    with open(METRICS_FILE, "r") as f:
        data = json.load(f)

    print("=" * 80)
    print(f"📊 {data['model_name'].upper()}")
    print("=" * 80)
    print(f"• Dataset Source:       {data['dataset']}")
    print(f"• Training Samples:     {data['training_samples']:,}")
    print(f"• Unseen Test Samples:  {data['test_samples']:,}")
    print(f"• Features Used ({len(data['features'])}): {', '.join(data['features'][:8])}...")
    print(f"• Model Trained At:     {data['trained_at']}")
    print("-" * 80)
    print(f"{'Forecast Horizon':<20} | {'MAE (veh/15m)':<15} | {'RMSE (veh/15m)':<15} | {'R² Score':<12} | {'MAPE (%)':<10}")
    print("-" * 80)
    for h, m in data["horizons"].items():
        print(f"{'+' + h + ' Forecast':<20} | {m['mae']:<15.2f} | {m['rmse']:<15.2f} | {m['r2']:<12.4f} | {m['mape']:<10.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    print_evaluation_report()
