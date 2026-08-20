"""
BMD-45 Dataset Pipeline & PCU Traffic Density Engine
Dataset: https://huggingface.co/datasets/iisc-aim/BMD-45 (Bengaluru Mobility Dataset)
"""

import os
import random
from typing import Dict, List, Any

# Indian Passenger Car Unit (PCU) Standard Weighting Factors (IRC Standards)
PCU_WEIGHTS = {
    "Auto-rickshaw": 0.8,
    "Motorbike": 0.5,
    "Car": 1.0,
    "Bus (BMTC / Private)": 3.0,
    "LCV (Light Commercial)": 1.5,
    "Truck": 3.0,
    "Rider (Motorcyclist)": 0.5,
    "Minibus": 1.8,
    "Tractor": 2.5,
    "Van": 1.2,
    "Bicycle": 0.2,
    "Two-wheeler passenger": 0.5,
    "Construction Equipment": 3.5,
    "Other Vehicle": 1.0
}

# Pre-cached sample CCTV traffic scenes from Bengaluru corridors
DEMO_SAMPLES = [
    {
        "id": "bmd_silk_board_01",
        "corridor": "Silk Board Junction",
        "camera_id": "CAM-3049",
        "time": "09:42:15 AM",
        "resolution": "1920x1080",
        "vehicle_counts": {
            "Auto-rickshaw": 14,
            "Motorbike": 32,
            "Car": 18,
            "Bus (BMTC / Private)": 5,
            "LCV (Light Commercial)": 4,
            "Truck": 2
        },
        "boxes": [
            {"class": "Auto-rickshaw", "color": "#f59e0b", "x": 0.15, "y": 0.45, "w": 0.14, "h": 0.22, "conf": 0.94},
            {"class": "Motorbike", "color": "#06b6d4", "x": 0.32, "y": 0.55, "w": 0.08, "h": 0.18, "conf": 0.98},
            {"class": "Bus (BMTC / Private)", "color": "#ef4444", "x": 0.45, "y": 0.30, "w": 0.25, "h": 0.40, "conf": 0.96},
            {"class": "Car", "color": "#6366f1", "x": 0.72, "y": 0.50, "w": 0.18, "h": 0.25, "conf": 0.91},
            {"class": "Auto-rickshaw", "color": "#f59e0b", "x": 0.05, "y": 0.60, "w": 0.12, "h": 0.20, "conf": 0.89},
            {"class": "Motorbike", "color": "#06b6d4", "x": 0.62, "y": 0.58, "w": 0.07, "h": 0.16, "conf": 0.93}
        ]
    },
    {
        "id": "bmd_mg_road_02",
        "corridor": "MG Road Signal",
        "camera_id": "CAM-0812",
        "time": "06:15:30 PM",
        "resolution": "1920x1080",
        "vehicle_counts": {
            "Auto-rickshaw": 8,
            "Motorbike": 24,
            "Car": 22,
            "Bus (BMTC / Private)": 2,
            "LCV (Light Commercial)": 3
        },
        "boxes": [
            {"class": "Car", "color": "#6366f1", "x": 0.10, "y": 0.40, "w": 0.20, "h": 0.30, "conf": 0.95},
            {"class": "LCV (Light Commercial)", "color": "#10b981", "x": 0.35, "y": 0.35, "w": 0.18, "h": 0.32, "conf": 0.92},
            {"class": "Motorbike", "color": "#06b6d4", "x": 0.55, "y": 0.52, "w": 0.09, "h": 0.20, "conf": 0.97},
            {"class": "Motorbike", "color": "#06b6d4", "x": 0.66, "y": 0.54, "w": 0.08, "h": 0.19, "conf": 0.94},
            {"class": "Auto-rickshaw", "color": "#f59e0b", "x": 0.78, "y": 0.45, "w": 0.15, "h": 0.26, "conf": 0.93}
        ]
    },
    {
        "id": "bmd_orr_marathahalli_03",
        "corridor": "Outer Ring Road (Marathahalli)",
        "camera_id": "CAM-1923",
        "time": "02:30:00 PM",
        "resolution": "1920x1080",
        "vehicle_counts": {
            "Truck": 6,
            "Car": 16,
            "Auto-rickshaw": 5,
            "Motorbike": 18
        },
        "boxes": [
            {"class": "Truck", "color": "#8b5cf6", "x": 0.20, "y": 0.28, "w": 0.30, "h": 0.45, "conf": 0.98},
            {"class": "Car", "color": "#6366f1", "x": 0.54, "y": 0.48, "w": 0.18, "h": 0.24, "conf": 0.93},
            {"class": "Auto-rickshaw", "color": "#f59e0b", "x": 0.74, "y": 0.52, "w": 0.13, "h": 0.21, "conf": 0.90}
        ]
    }
]

class BMD45Pipeline:
    def __init__(self):
        self.hf_dataset_id = "iisc-aim/BMD-45"
        self.pcu_weights = PCU_WEIGHTS

    def calculate_pcu_density(self, vehicle_counts: Dict[str, int]) -> Dict[str, Any]:
        """Calculates total Passenger Car Units (PCU) and Congestion Index."""
        total_pcu = 0.0
        total_vehicles = 0
        
        for v_type, count in vehicle_counts.items():
            weight = self.pcu_weights.get(v_type, 1.0)
            total_pcu += count * weight
            total_vehicles += count

        # Max capacity threshold for typical 4-lane CCTV frame is 60 PCU
        capacity_threshold = 60.0
        congestion_index = min(100.0, round((total_pcu / capacity_threshold) * 100, 1))
        
        if congestion_index < 35.0:
            status = "CLEAR"
            color = "#10b981"
        elif congestion_index < 70.0:
            status = "MODERATE"
            color = "#f59e0b"
        elif congestion_index < 88.0:
            status = "HEAVY"
            color = "#f97316"
        else:
            status = "SEVERE"
            color = "#ef4444"

        return {
            "total_vehicles": total_vehicles,
            "total_pcu": round(total_pcu, 2),
            "congestion_index_pct": congestion_index,
            "congestion_status": status,
            "status_color": color
        }

    def get_cctv_sample(self, sample_id: str = None) -> Dict[str, Any]:
        """Returns CCTV detection frame payload."""
        if sample_id:
            sample = next((s for s in DEMO_SAMPLES if s["id"] == sample_id), DEMO_SAMPLES[0])
        else:
            sample = random.choice(DEMO_SAMPLES)

        density_info = self.calculate_pcu_density(sample["vehicle_counts"])
        return {
            **sample,
            **density_info
        }

    def load_from_huggingface_streaming(self):
        """Attempts live HF streaming load if online."""
        try:
            from datasets import load_dataset
            ds = load_dataset(self.hf_dataset_id, streaming=True)
            return True, "HF Dataset Connected"
        except Exception as e:
            return False, f"Using Local Pre-cached Pipeline ({str(e)})"
