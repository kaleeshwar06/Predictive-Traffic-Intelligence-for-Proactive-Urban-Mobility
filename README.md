# Predictive Traffic Intelligence for Proactive Urban Mobility

An end-to-end AI-powered traffic prediction and smart routing system built for Innohack.

## 🧠 System Architecture

The system is composed of 5 integrated modules:

| Module | Description |
|---|---|
| **Model 1** | Traffic Forecasting — predicts congestion 15/30/60 minutes ahead |
| **Model 2 (CV Engine)** | YOLO-based vehicle detection from live CCTV feeds |
| **Module 3 (OSM Graph)** | OpenStreetMap road network graph builder |
| **Model 3 (Routing Engine)** | Modified Dijkstra using predicted travel times |
| **Module 5 (Domain Adaptation)** | OLS-based transfer learning for new cities |

## 🚀 Features

- **Live TfL JamCam Integration**: Streams 800+ real London traffic cameras on an interactive OSM map.
- **Predictive Routing**: 3 distinct OSRM-calculated road routes with AI-forecasted congestion levels.
- **Congestion Timing**: Shows exactly when traffic will peak on each route.
- **Domain Adaptation**: Calibrates predictions to any city without retraining.
- **Interactive Dashboard**: Click any camera on the map to stream its live video feed.

## 📁 Project Structure

```
Innohack/
├── innohack/
│   ├── backend/
│   │   ├── server.py                  # Flask backend server
│   │   ├── model2_cv_engine.py        # Computer vision pipeline
│   │   ├── module3_osm_graph.py       # OSM graph database
│   │   ├── model3_routing_engine.py   # Predictive routing engine
│   │   └── module5_domain_adaptation.py  # Local adaptation module
│   └── traffic_dashboard.html         # Frontend dashboard (OSM + TfL)
├── main_pipeline.py                   # End-to-end orchestrator
├── evaluate_model2.py                 # CV engine evaluation
└── demo_live_tfl.py                   # TfL live API demo
```

## 🛠️ Tech Stack

- **Backend**: Python, Flask, NumPy, Pandas
- **AI/CV**: YOLO Object Detection, OLS Regression
- **Frontend**: HTML5/CSS3, Leaflet.js, OSRM Routing API
- **Data**: TfL JamCams API, OpenStreetMap, Madrid MTD Dataset

## ▶️ Running the Project

```bash
# Install dependencies
pip install flask numpy pandas requests

# Start the backend server
python innohack/backend/server.py

# Open in browser
http://localhost:8000
```

## 📊 Model Performance

- **Model 2 CV Engine**: 92.5% F1-Score, 267 FPS
- **Domain Adaptation**: 61% error reduction vs baseline

## 👥 Built For

**Innohack Hackathon** — Smart City Traffic Management Track
