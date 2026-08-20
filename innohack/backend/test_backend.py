import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import bmd45_pipeline
    import iot_simulator
    import predictor
    import route_optimizer
    
    print("[SUCCESS] All AI Backend modules imported successfully!")

    pipe = bmd45_pipeline.BMD45Pipeline()
    sample = pipe.get_cctv_sample("bmd_silk_board_01")
    print(f"[SUCCESS] BMD-45 CCTV Sample: {sample['corridor']} ({sample['congestion_status']})")
    print(f"          PCU Index: {sample['congestion_index_pct']}%, Total Vehicles: {sample['total_vehicles']}")

    sim = iot_simulator.IoTSimulator()
    telemetry = sim.get_live_telemetry()
    print(f"[SUCCESS] IoT Telemetry simulated across {len(telemetry)} corridors.")

    pred = predictor.TrafficPredictor()
    forecast = pred.get_citywide_forecast(telemetry)
    print(f"[SUCCESS] Citywide 15m/30m/60m Forecast computed. Hotspots: {len(forecast['critical_hotspots'])}")

    opt = route_optimizer.RouteOptimizer()
    route = opt.calculate_optimized_route()
    print(f"[SUCCESS] Route Optimizer: {route['origin']} -> {route['destination']}")
    print(f"          Time Saved: {route['ai_optimized_route']['time_saved_min']} mins")

    print("\n--- ALL BACKEND MODULE VERIFICATIONS PASSED ---")

except Exception as e:
    print(f"[ERROR] Verification failed: {e}")
