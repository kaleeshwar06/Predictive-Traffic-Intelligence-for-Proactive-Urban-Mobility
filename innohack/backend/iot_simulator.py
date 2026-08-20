"""
IoT Telemetry Simulator across 10 Major Bengaluru Traffic Corridors
Simulates CCTV Camera visual counts, IoT Loop Detector Speeds, and GPS Probe Travel Times.
"""

import random
import time
from typing import List, Dict, Any

BENGALURU_CORRIDORS = [
    {
        "id": "CORR-01",
        "name": "Silk Board Junction",
        "area": "South Bengaluru",
        "base_speed_kmh": 12.0,
        "free_flow_kmh": 50.0,
        "length_km": 3.2,
        "lat": 12.9172,
        "lng": 77.6228,
        "cctv_cam": "CAM-3049"
    },
    {
        "id": "CORR-02",
        "name": "Outer Ring Road (Bellandur - Marathahalli)",
        "area": "East IT Corridor",
        "base_speed_kmh": 18.0,
        "free_flow_kmh": 60.0,
        "length_km": 6.5,
        "lat": 12.9304,
        "lng": 77.6784,
        "cctv_cam": "CAM-1923"
    },
    {
        "id": "CORR-03",
        "name": "MG Road & Trinity Circle",
        "area": "Central Bengaluru",
        "base_speed_kmh": 22.0,
        "free_flow_kmh": 45.0,
        "length_km": 2.8,
        "lat": 12.9756,
        "lng": 77.6066,
        "cctv_cam": "CAM-0812"
    },
    {
        "id": "CORR-04",
        "name": "Tin Factory (KR Puram)",
        "area": "East Gate",
        "base_speed_kmh": 9.5,
        "free_flow_kmh": 55.0,
        "length_km": 4.1,
        "lat": 12.9984,
        "lng": 77.6698,
        "cctv_cam": "CAM-4120"
    },
    {
        "id": "CORR-05",
        "name": "Hebbal Flyover Junction",
        "area": "North Bengaluru (Airport Highway)",
        "base_speed_kmh": 34.0,
        "free_flow_kmh": 70.0,
        "length_km": 5.0,
        "lat": 13.0358,
        "lng": 77.5970,
        "cctv_cam": "CAM-1055"
    },
    {
        "id": "CORR-06",
        "name": "Koramangala 80ft Road",
        "area": "South-East",
        "base_speed_kmh": 26.0,
        "free_flow_kmh": 45.0,
        "length_km": 2.2,
        "lat": 12.9352,
        "lng": 77.6245,
        "cctv_cam": "CAM-0544"
    },
    {
        "id": "CORR-07",
        "name": "Electronic City Elevated Expressway",
        "area": "South IT Hub",
        "base_speed_kmh": 48.0,
        "free_flow_kmh": 80.0,
        "length_km": 9.8,
        "lat": 12.8452,
        "lng": 77.6602,
        "cctv_cam": "CAM-7711"
    },
    {
        "id": "CORR-08",
        "name": "Indiranagar 100ft Road",
        "area": "East Commercial Hub",
        "base_speed_kmh": 20.0,
        "free_flow_kmh": 45.0,
        "length_km": 3.0,
        "lat": 12.9784,
        "lng": 77.6408,
        "cctv_cam": "CAM-2219"
    },
    {
        "id": "CORR-09",
        "name": "Goraguntepalya Junction (Tumkur Rd)",
        "area": "North-West Industrial",
        "base_speed_kmh": 14.0,
        "free_flow_kmh": 60.0,
        "length_km": 4.5,
        "lat": 13.0285,
        "lng": 77.5402,
        "cctv_cam": "CAM-3388"
    },
    {
        "id": "CORR-10",
        "name": "Bannerghatta Road (Dairy Circle)",
        "area": "South Corridor",
        "base_speed_kmh": 15.5,
        "free_flow_kmh": 50.0,
        "length_km": 4.0,
        "lat": 12.9380,
        "lng": 77.5980,
        "cctv_cam": "CAM-0941"
    }
]

class IoTSimulator:
    def __init__(self):
        self.corridors = BENGALURU_CORRIDORS

    def get_live_telemetry(self) -> List[Dict[str, Any]]:
        """Generates real-time IoT metrics for all corridors."""
        telemetry = []
        for corr in self.corridors:
            # Introduce small noise to simulate real sensor variance
            jitter = random.uniform(-3.0, 3.0)
            current_speed = max(5.0, min(corr["free_flow_kmh"], corr["base_speed_kmh"] + jitter))
            
            speed_ratio = current_speed / corr["free_flow_kmh"]
            congestion_pct = round((1.0 - speed_ratio) * 100.0, 1)
            
            # Estimated travel time and delay
            ideal_time_min = (corr["length_km"] / corr["free_flow_kmh"]) * 60.0
            actual_time_min = (corr["length_km"] / current_speed) * 60.0
            delay_min = round(max(0.0, actual_time_min - ideal_time_min), 1)

            if congestion_pct < 35.0:
                status = "CLEAR"
                color = "#10b981"
            elif congestion_pct < 65.0:
                status = "MODERATE"
                color = "#f59e0b"
            elif congestion_pct < 85.0:
                status = "HEAVY"
                color = "#f97316"
            else:
                status = "SEVERE"
                color = "#ef4444"

            telemetry.append({
                "corridor_id": corr["id"],
                "name": corr["name"],
                "area": corr["area"],
                "camera_id": corr["cctv_cam"],
                "lat": corr["lat"],
                "lng": corr["lng"],
                "current_speed_kmh": round(current_speed, 1),
                "free_flow_speed_kmh": corr["free_flow_kmh"],
                "congestion_pct": congestion_pct,
                "status": status,
                "status_color": color,
                "travel_time_min": round(actual_time_min, 1),
                "delay_min": delay_min,
                "active_iot_sensors": {
                    "cctv_cameras": random.randint(3, 8),
                    "induction_loops": random.randint(12, 30),
                    "gps_probes": random.randint(450, 1200)
                }
            })
        return telemetry
