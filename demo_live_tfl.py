"""
Live TfL JamCam API Demo - Module 2
===================================
Fetches live traffic camera frames from London's Transport for London API,
runs them through our YOLO + Tracking Computer Vision Engine, and extracts
real-time traffic flow metrics to be fed into Model 1.
"""

import os
import sys
import time
from pathlib import Path

# Safe encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "innohack" / "backend"))
from model2_cv_engine import VideoTrafficCVEngine

CHARTS_DIR = BASE_DIR / "traffic_analysis_charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

print("=====================================================================")
print("📹 MODULE 2: LIVE TFL JAMCAM API TRAFFIC TRACKING")
print("=====================================================================")

# Initialize Engine with TfL API Enabled
print("[*] Initializing Computer Vision Engine and connecting to TfL API...")
engine = VideoTrafficCVEngine(use_api=True)

# Give it a moment to fetch the camera list
time.sleep(2)
num_cams = len(engine.api_client.cached_tfl_cams) if engine.api_client else 0
print(f"[+] Successfully connected to TfL API. Cached {num_cams} live cameras.")

if num_cams == 0:
    print("[-] Error: No cameras loaded from API. Check network connection.")
    sys.exit(1)

# Pick a specific camera to track over a few frames
target_cam = engine.api_client.cached_tfl_cams[0]
print(f"[*] Selected Camera: {target_cam['name']} (ID: {target_cam['id']})")
print(f"[*] Live Image URL: {target_cam['imageUrl']}")
print("\n[*] Running Live Inference Loop...\n")

num_frames_to_process = 3

for i in range(num_frames_to_process):
    print(f"--- Processing Frame {i+1}/{num_frames_to_process} ---")
    
    # Process the live frame (automatically fetches from API inside the engine)
    # We pass camera_idx=0 indirectly by relying on engine.frame_count logic or directly fetching
    frame_img = engine.api_client.fetch_live_camera_frame(camera_idx=0)
    
    if frame_img is None:
        print("    [!] Failed to fetch live frame. Skipping...")
        continue
        
    res = engine.process_video_frame(frame_img=frame_img)
    
    # Extract Metrics
    vehicles = res["vehicle_counts"]
    flow_rate = res["traffic_flow_veh_min"]
    pcu = res["total_pcu_load"]
    latency = res["latency_ms"]
    
    print(f"    • Detected Vehicles: {res['total_vehicles_in_view']} {vehicles}")
    print(f"    • Flow Rate:         {flow_rate} veh/min")
    print(f"    • PCU Load:          {pcu}")
    print(f"    • Inference Latency: {latency} ms ({res['processing_fps']} FPS)")
    
    # Save the annotated frame
    out_path = CHARTS_DIR / f"tfl_live_tracking_frame_{i+1}.jpg"
    res["annotated_image"].save(out_path, "JPEG", quality=90)
    print(f"    • Saved Annotated Frame: {out_path}\n")
    
    # Small delay between frames to simulate live feed polling
    if i < num_frames_to_process - 1:
        time.sleep(1.5)

print("=====================================================================")
print("✅ Live API Tracking Complete!")
print("The output payload 'model1_input_payload' is now ready to be fed ")
print("directly into Model 1 (Forecasting Engine).")
print("=====================================================================")
