"""
FastAPI AI Backend for Real-Time Traffic Congestion Prediction & Management
Problem Statement: AI/ML-09 (Innohack Project)
"""

import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from bmd45_pipeline import BMD45Pipeline
from iot_simulator import IoTSimulator
from predictor import TrafficPredictor
from route_optimizer import RouteOptimizer

app = FastAPI(
    title="AI/ML-09: Real-Time Traffic Congestion Prediction & Management System",
    description="Backend API powering multi-source IoT ingestion, BMD-45 visual detection, time-series forecasting, and dynamic route optimization.",
    version="1.0.0"
)

# Enable CORS for Web Dashboard frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core AI Modules
bmd_pipeline = BMD45Pipeline()
iot_simulator = IoTSimulator()
predictor = TrafficPredictor()
route_optimizer = RouteOptimizer()

class EmergencyRequest(BaseModel):
    corridor: str = "Silk Board Junction"
    hospital_name: str = "St. John's Hospital (Koramangala)"

@app.get("/")
def read_root():
    return {
        "status": "ONLINE",
        "system": "Traffic Congestion AI Engine (AI/ML-09)",
        "dataset": "iisc-aim/BMD-45 (Bengaluru Mobility Dataset)",
        "sdgs": ["SDG 9: Industry & Infrastructure", "SDG 11: Sustainable Cities", "SDG 13: Climate Action"],
        "endpoints": [
            "/api/traffic/live",
            "/api/traffic/predict",
            "/api/camera/sample",
            "/api/route/optimize",
            "/api/emergency/greenwave"
        ]
    }

@app.get("/api/traffic/live")
def get_live_traffic():
    """Returns real-time IoT corridor telemetry."""
    telemetry = iot_simulator.get_live_telemetry()
    return {
        "corridors_count": len(telemetry),
        "telemetry": telemetry
    }

@app.get("/api/traffic/predict")
def get_traffic_forecast():
    """Returns 15m, 30m, and 60m predictive congestion forecast."""
    telemetry = iot_simulator.get_live_telemetry()
    forecast = predictor.get_citywide_forecast(telemetry)
    return forecast

@app.get("/api/camera/sample")
def get_camera_sample(sample_id: Optional[str] = Query(None)):
    """Returns BMD-45 dataset CCTV detection frame with PCU density."""
    sample = bmd_pipeline.get_cctv_sample(sample_id)
    return sample

@app.get("/api/route/optimize")
def optimize_route(route_id: Optional[str] = Query(None)):
    """Computes AI-optimized dynamic route vs standard route."""
    result = route_optimizer.calculate_optimized_route(route_id)
    return result

@app.post("/api/emergency/greenwave")
def activate_green_wave(req: EmergencyRequest):
    """Activates Emergency Ambulance Corridor Green Wave Mode."""
    result = route_optimizer.activate_emergency_green_wave(req.corridor, req.hospital_name)
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
