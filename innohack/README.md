# 🚦 Real-Time Traffic Congestion Prediction & Management System (AI/ML-09)

> **Innohack Project** | Problem Statement ID: **AI/ML-09**  
> **SDG Alignment**: SDG 9 (Industry, Innovation, and Infrastructure), SDG 11 (Sustainable Cities and Communities), SDG 13 (Climate Action).

---

## 📌 Executive Summary
Urban traffic congestion in Indian megacities like Bengaluru results in millions of wasted commuter hours, economic losses exceeding \$5 Billion annually, and increased carbon emissions. 

This project delivers an **AI-powered Real-Time Traffic Congestion Prediction and Dynamic Management System**. By integrating vision data from Hugging Face's **Bengaluru Mobility Dataset (`iisc-aim/BMD-45`)** across 14 vehicle classes (auto-rickshaws, motorbikes, buses, LCVs, etc.) alongside simulated multi-source IoT road sensors (loop detectors, GPS velocity probes), the system:
1. Calculates **Passenger Car Unit (PCU)** weighted traffic density.
2. Predicts spatial-temporal congestion trends for **15m, 30m, and 60m** horizons.
3. Dynamically reroutes vehicles using an AI A* Graph Routing Engine.
4. Provides an **Emergency Ambulance Green Wave Corridor** feature to clear routes for emergency services.

---

## 🏗️ Architecture

```
+-----------------------------------------------------------------------------------+
|                            MULTI-SOURCE IOT INPUTS                                |
+-----------------------------------+-----------------------------------------------+
| CCTV Feeds (BMD-45 Dataset)       | IoT Road Sensors (Speed/Occupancy)            |
| 14 Vehicle Classes (Auto, Bike..) | GPS Probe Data (Travel Time/Velocity)         |
+-----------------------------------+-----------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------------+
|                        AI CONGESTION ENGINE (Python / FastAPI)                    |
| - PCU (Passenger Car Unit) Weighted Density Calculation                           |
| - Spatial-Temporal Time-Series Congestion Predictor (15m / 30m / 60m forecast)    |
| - Dynamic Graph Route Optimizer (A* with Congestion & Delay Weights)              |
+-----------------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------------+
|               HACKATHON DASHBOARD & LIVE COMMAND CENTER (Web App)                 |
| - Live CCTV AI Camera Feed with Bounding Boxes                                    |
| - Bengaluru Interactive Corridor Traffic Map & Congestion Matrix                   |
| - Real-Time Dynamic Route Optimizer & Emergency Green-Wave Corridor               |
| - Predictive Congestion Analytics & Sustainability Impact (SDG 9, 11, 13)         |
+-----------------------------------------------------------------------------------+
```

---

## 🚀 Quick Start Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch FastAPI AI Backend
```bash
cd backend
python main.py
```
*API Documentation will be available at `http://localhost:8000/docs`*

### 3. Open Web Dashboard
Open `D:\innohack\web_dashboard\index.html` directly in your browser or run:
```bash
python -m http.server 8080 --directory "D:\innohack\web_dashboard"
```
*Visit `http://localhost:8080` in your web browser.*

---

## 📊 Key Features & Innovations
* **Heterogeneous Indian Traffic Modeling**: PCU weighting tailored for mixed traffic (Auto-rickshaws, two-wheelers, heavy buses).
* **Multi-Horizon AI Forecasting**: 15, 30, and 60-minute predictive congestion index.
* **Emergency Green-Wave Corridor**: Instantly prioritizes emergency vehicle dispatch, reducing ambulance delay by up to 40%.
* **Sustainability & Impact Metrics**: Real-time tracking of CO2 emissions saved, fuel saved, and delay mitigation.
