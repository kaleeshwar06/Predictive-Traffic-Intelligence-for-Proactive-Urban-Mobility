"""
Module 5 — Local Domain Adaptation Module 🌍
============================================
Calibrates and adapts the base Model 1 (trained on Madrid MTD) 
to a new city (e.g., Bengaluru) in real-time, using live camera 
vehicle counts (from Model 2) as ground truth.
"""

import sys
import numpy as np
from typing import Dict, List, Tuple

# Safe encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class CityAdaptationModule:
    """
    Dynamically calibrates base predictions using a rolling memory 
    of live camera observations (Model 2) vs Base Model Forecasts (Model 1).
    """
    def __init__(self, target_city_name: str, memory_size: int = 50):
        self.target_city = target_city_name
        self.memory_size = memory_size  # How many recent intervals to remember
        
        self.historical_base_preds = []
        self.historical_actuals = []
        
        # Calibration parameters (Y_calibrated = alpha * Y_base + beta)
        self.alpha = 1.0  # Multiplicative scaling factor
        self.beta = 0.0   # Additive bias shift

    def update_with_live_camera_data(self, base_forecast: float, actual_camera_count: float):
        """
        Receives the live vehicle count from Model 2 and the forecast Model 1 *originally* made.
        Stores it in the rolling buffer.
        """
        self.historical_base_preds.append(base_forecast)
        self.historical_actuals.append(actual_camera_count)
        
        # Keep buffer within memory size
        if len(self.historical_base_preds) > self.memory_size:
            self.historical_base_preds.pop(0)
            self.historical_actuals.pop(0)
            
        # Re-calibrate on every new data point once we have a minimum threshold
        if len(self.historical_base_preds) >= 5:
            self._recalibrate()

    def _recalibrate(self):
        """
        Calculates optimal alpha and beta using Ordinary Least Squares (OLS) regression
        to map the Base Model's domain to the New City's domain.
        """
        X = np.array(self.historical_base_preds)
        Y = np.array(self.historical_actuals)
        
        # Prevent division by zero if all predictions are identical
        if np.std(X) == 0:
            self.alpha = 1.0
            self.beta = np.mean(Y) - np.mean(X)
        else:
            # Linear Regression: Y = alpha * X + beta
            # alpha = Covariance(X, Y) / Variance(X)
            covariance_matrix = np.cov(X, Y)
            self.alpha = covariance_matrix[0, 1] / covariance_matrix[0, 0]
            
            # Keep alpha within sensible bounds (don't invert trends or scale ridiculously)
            self.alpha = max(0.2, min(self.alpha, 5.0))
            
            # beta = Mean(Y) - alpha * Mean(X)
            self.beta = np.mean(Y) - self.alpha * np.mean(X)

    def adapt_prediction(self, base_forecast: float) -> float:
        """
        Applies the learned calibration parameters to a new forecast.
        """
        calibrated = (base_forecast * self.alpha) + self.beta
        return max(0.0, round(calibrated, 1)) # Traffic can't be negative

    def get_calibration_stats(self) -> Dict[str, float]:
        return {
            "alpha_multiplier": round(self.alpha, 3),
            "beta_bias_shift": round(self.beta, 3),
            "data_points_learned": len(self.historical_base_preds)
        }


if __name__ == "__main__":
    print("==================================================================")
    print("🌍 MODULE 5: LOCAL DOMAIN ADAPTATION MODULE")
    print("==================================================================")
    
    # 1. Initialize adaptation module for Bengaluru
    target_city = "Bengaluru, India"
    adapter = CityAdaptationModule(target_city_name=target_city)
    
    print(f"[*] Base Model trained on: Madrid (MTD Dataset)")
    print(f"[*] Target Deployment City: {target_city}")
    print("[*] Simulating live camera data ingestion to adapt the model...\n")
    
    # Scenario: Madrid traffic is generally lighter than Bengaluru.
    # We simulate a situation where actual Bengaluru traffic (from Model 2 camera)
    # is roughly 2.5x heavier than what the Madrid base model expects, plus an offset of +50 vehicles.
    
    uncalibrated_errors = []
    calibrated_errors = []
    
    print("--- LIVE DOMAIN ADAPTATION PHASE ---")
    # Simulate receiving 15 updates (e.g., 15 intervals of 15-minutes)
    for i in range(1, 16):
        # Model 1 (Madrid) predicts based on raw features
        madrid_base_pred = float(np.random.normal(120, 20))
        
        # Model 2 (Live Camera in Bengaluru) counts actual vehicles
        # Ground reality: Bengaluru is much busier!
        bengaluru_actual_count = (madrid_base_pred * 2.5) + 50.0 + np.random.normal(0, 15)
        
        # How would the model have performed if we DIDN'T calibrate?
        uncalib_err = abs(madrid_base_pred - bengaluru_actual_count)
        uncalibrated_errors.append(uncalib_err)
        
        # How does the Calibrated Model perform on this new data?
        calibrated_pred = adapter.adapt_prediction(madrid_base_pred)
        calib_err = abs(calibrated_pred - bengaluru_actual_count)
        calibrated_errors.append(calib_err)
        
        # Feed the ground truth back into the adapter so it learns!
        adapter.update_with_live_camera_data(madrid_base_pred, bengaluru_actual_count)
        
        # Print progress every 3 steps
        if i % 3 == 0:
            stats = adapter.get_calibration_stats()
            print(f"  [Interval {i}] Learned Multiplier (α): {stats['alpha_multiplier']} | Learned Bias (β): {stats['beta_bias_shift']}")

    print("\n==================================================================")
    print("📊 ADAPTATION PERFORMANCE RESULTS")
    print("==================================================================")
    
    mae_uncalibrated = np.mean(uncalibrated_errors)
    mae_calibrated = np.mean(calibrated_errors)
    improvement = ((mae_uncalibrated - mae_calibrated) / mae_uncalibrated) * 100
    
    print(f"• Baseline Madrid Model Error in Bengaluru:   {mae_uncalibrated:.1f} vehicles")
    print(f"• Locally Adapted Model Error in Bengaluru:   {mae_calibrated:.1f} vehicles")
    print(f"• Error Reduction (Improvement):              {improvement:.1f}%")
    print("\n✅ Module 5 successfully learned the new city's traffic profile in real-time!")
    print("==================================================================")
