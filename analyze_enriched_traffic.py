"""
Comprehensive Exploratory Data Analysis (EDA) for Enriched Traffic Datasets for Madrid (MTD)
=============================================================================================
Performs statistical profiling, spatial analysis, temporal pattern extraction,
meteorological correlation, road infrastructure impact, and graph network analysis.
Generates publication-quality charts into 'traffic_analysis_charts/'.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Safe encoding for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR / "Enriched Traffic Datasets for Madrid"
CHARTS_DIR = BASE_DIR / "traffic_analysis_charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

def find_file(filename):
    for root, dirs, files in os.walk(ROOT_DIR):
        if filename in files:
            return Path(root) / filename
    return None

sensors_file = find_file("MTD_id_longitude_latitude.csv")
adj_file = find_file("MTD_adj_matrix.npy")
target_file = find_file("MTD_target_month.csv")

print("=" * 80)
print("🚦 ENRICHED TRAFFIC DATASETS FOR MADRID (MTD) — DEEP COMPREHENSIVE ANALYSIS")
print("=" * 80)
print(f"📁 Sensors File:  {sensors_file}")
print(f"📁 Graph Adj Mat: {adj_file}")
print(f"📁 Target Month:  {target_file}")

# 1. Geographic & Sensor Topology Analysis
print("\n[1/6] 🗺️ SENSOR NETWORK & SPATIAL TOPOLOGY")
df_sensors = pd.read_csv(sensors_file)
print(f"   • Total Monitored Sensors: {len(df_sensors):,}")
print(f"   • Latitude Range:  [{df_sensors['latitude'].min():.4f}, {df_sensors['latitude'].max():.4f}] (Center: {df_sensors['latitude'].mean():.4f}°N)")
print(f"   • Longitude Range: [{df_sensors['longitude'].min():.4f}, {df_sensors['longitude'].max():.4f}] (Center: {df_sensors['longitude'].mean():.4f}°W)")

# 2. Graph Adjacency Matrix Analysis
print("\n[2/6] 🕸️ SPATIAL-TEMPORAL ROAD GRAPH ADJACENCY MATRIX")
adj_matrix = np.load(adj_file)
print(f"   • Adjacency Matrix Shape: {adj_matrix.shape} ({adj_matrix.shape[0]} nodes)")
total_possible_edges = adj_matrix.shape[0] * adj_matrix.shape[1]
non_zero_edges = np.count_nonzero(adj_matrix)
graph_density = (non_zero_edges / total_possible_edges) * 100
avg_degree = non_zero_edges / adj_matrix.shape[0]
print(f"   • Connected Road Links (Edges): {non_zero_edges:,}")
print(f"   • Graph Network Density: {graph_density:.2f}%")
print(f"   • Average Node Degree: {avg_degree:.1f} connections per traffic sensor")

# 3. Time-Series & Target Month Traffic Analysis
print("\n[3/6] ⏱️ TEMPORAL & MULTI-VARIABLE DATASET PROFILING")

# Read 300,000 representative records
print("   • Loading 300,000 representative records from MTD_target_month.csv...")
df_target = pd.read_csv(target_file, nrows=300000)
df_target['date'] = pd.to_datetime(df_target['date'])
df_target['hour'] = df_target['date'].dt.hour
df_target['day_name'] = df_target['date'].dt.day_name()
df_target['is_weekend'] = df_target['date'].dt.dayofweek >= 5

print(f"   • Features in Dataset ({len(df_target.columns)} columns): {', '.join(df_target.columns.tolist())}")
print(f"   • Sampled Date Range: {df_target['date'].min()} to {df_target['date'].max()}")

# Traffic Intensity Stats
ti = df_target['traffic_intensity']
print("\n   📊 Traffic Intensity (Vehicles/Interval) Statistics:")
print(f"      - Mean Intensity:   {ti.mean():.1f} vehicles")
print(f"      - Std Dev:          {ti.std():.1f}")
print(f"      - Median:           {ti.median():.1f} vehicles")
print(f"      - 25th - 75th IQR:  {ti.quantile(0.25):.1f} to {ti.quantile(0.75):.1f} vehicles")
print(f"      - 95th Percentile:  {ti.quantile(0.95):.1f} vehicles (Congestion Peak)")
print(f"      - 99th Percentile:  {ti.quantile(0.99):.1f} vehicles (Severe Gridlock)")
print(f"      - Maximum Recorded: {ti.max():.1f} vehicles")

# 4. Road Infrastructure Distribution
print("\n[4/6] 🛣️ ROAD INFRASTRUCTURE & OPENSTREETMAP (OSM) ENRICHMENT")
if 'highway' in df_target.columns:
    hw_counts = df_target['highway'].value_counts()
    print("   • Highway Class Distribution:")
    for hw, cnt in hw_counts.head(6).items():
        avg_flow = df_target[df_target['highway'] == hw]['traffic_intensity'].mean()
        print(f"      - {str(hw):<20}: {cnt:>6,} records ({cnt/len(df_target)*100:.1f}%) | Avg Flow: {avg_flow:.1f} veh")

if 'lanes' in df_target.columns:
    print(f"   • Number of Lanes Distribution: {df_target['lanes'].dropna().value_counts().head(4).to_dict()}")

if 'maxspeed' in df_target.columns:
    print(f"   • Speed Limits: {df_target['maxspeed'].dropna().value_counts().head(5).to_dict()}")

# 5. Weather Correlation
print("\n[5/6] 🌦️ METEOROLOGICAL IMPACT ON TRAFFIC FLOW")
weather_cols = [c for c in ['traffic_intensity', 'temperature', 'wind', 'precipitation'] if c in df_target.columns]
if len(weather_cols) > 1:
    corr = df_target[weather_cols].corr()
    print("   • Correlation Matrix with Traffic Intensity:")
    for col in weather_cols:
        if col != 'traffic_intensity':
            print(f"      - {col.capitalize():<15}: correlation = {corr.loc['traffic_intensity', col]:+.4f}")

# 6. Generate Analytical Charts
print("\n[6/6] 🎨 GENERATING COMPREHENSIVE DATASET VISUALIZATIONS...")

# Chart 1: Diurnal Traffic Profile (Weekday vs Weekend)
fig, ax = plt.subplots(figsize=(10, 5))
hourly_summary = df_target.groupby(['hour', 'is_weekend'])['traffic_intensity'].mean().unstack()
ax.plot(hourly_summary.index, hourly_summary[False], color='#2563eb', linewidth=2.5, marker='o', label='Weekday (Working Day)')
ax.plot(hourly_summary.index, hourly_summary[True], color='#f59e0b', linewidth=2.5, marker='s', linestyle='--', label='Weekend (Sat / Sun)')
ax.set_title('Hourly Traffic Intensity Profile in Madrid: Weekday vs Weekend', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Hour of Day (00:00 - 23:00)', fontsize=11)
ax.set_ylabel('Average Traffic Intensity (Vehicles / 15m Interval)', fontsize=11)
ax.set_xticks(range(0, 24))
ax.axvspan(7.5, 9.5, color='#ef4444', alpha=0.12, label='Morning Rush (08:00 - 09:30)')
ax.axvspan(18.0, 20.5, color='#ef4444', alpha=0.12, label='Evening Rush (18:00 - 20:30)')
ax.legend(loc='upper left', frameon=True)
plt.tight_layout()
c1_path = CHARTS_DIR / "01_diurnal_traffic_profile.png"
plt.savefig(c1_path)
plt.close()
print(f"   [+] Saved: {c1_path}")

# Chart 2: Highway Infrastructure vs Traffic Flow
if 'highway' in df_target.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    top_hw = df_target['highway'].value_counts().head(7).index
    df_hw_top = df_target[df_target['highway'].isin(top_hw)]
    hw_order = df_hw_top.groupby('highway')['traffic_intensity'].mean().sort_values(ascending=False).index
    
    sns.barplot(data=df_hw_top, x='highway', y='traffic_intensity', order=hw_order, palette='viridis', ax=ax, capsize=0.1)
    ax.set_title('Traffic Volume by OpenStreetMap Road Hierarchy', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Road Classification (Highway Tag)', fontsize=11)
    ax.set_ylabel('Mean Traffic Flow (Vehicles / Interval)', fontsize=11)
    plt.xticks(rotation=25)
    plt.tight_layout()
    c2_path = CHARTS_DIR / "02_highway_hierarchy_flow.png"
    plt.savefig(c2_path)
    plt.close()
    print(f"   [+] Saved: {c2_path}")

# Chart 3: Spatial Sensor Network Map
fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(df_sensors['longitude'], df_sensors['latitude'], c='#3b82f6', s=35, alpha=0.7, edgecolors='black', linewidth=0.5)
ax.set_title(f'Madrid Intelligent Transportation Network ({len(df_sensors)} Sensors)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Longitude (°W)', fontsize=11)
ax.set_ylabel('Latitude (°N)', fontsize=11)
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
c3_path = CHARTS_DIR / "03_sensor_spatial_network.png"
plt.savefig(c3_path)
plt.close()
print(f"   [+] Saved: {c3_path}")

# Chart 4: Correlation Matrix Heatmap
fig, ax = plt.subplots(figsize=(7, 6))
# Convert lanes / maxspeed to numeric where possible
for num_c in ['lanes', 'temperature', 'wind', 'precipitation', 'traffic_intensity', 'hour']:
    if num_c in df_target.columns:
        df_target[num_c] = pd.to_numeric(df_target[num_c], errors='coerce')
numeric_cols = [c for c in ['traffic_intensity', 'temperature', 'wind', 'precipitation', 'lanes', 'hour'] if c in df_target.columns]
corr_matrix = df_target[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5, ax=ax)
ax.set_title('Cross-Feature Correlation Matrix (Traffic & Exogenous Features)', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
c4_path = CHARTS_DIR / "04_correlation_heatmap.png"
plt.savefig(c4_path)
plt.close()
print(f"   [+] Saved: {c4_path}")

# Chart 5: Traffic Intensity Probability Distribution
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(df_target['traffic_intensity'].dropna(), bins=50, kde=True, color='#6366f1', ax=ax)
ax.axvline(df_target['traffic_intensity'].mean(), color='#ef4444', linestyle='--', linewidth=2, label=f"Mean ({df_target['traffic_intensity'].mean():.1f})")
ax.axvline(df_target['traffic_intensity'].median(), color='#10b981', linestyle='-', linewidth=2, label=f"Median ({df_target['traffic_intensity'].median():.1f})")
ax.axvline(df_target['traffic_intensity'].quantile(0.95), color='#f59e0b', linestyle=':', linewidth=2, label=f"95th %ile ({df_target['traffic_intensity'].quantile(0.95):.1f})")
ax.set_title('Probability Distribution of Traffic Intensity (Right-Skewed Flow)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Observed Traffic Intensity (Vehicles)', fontsize=11)
ax.set_ylabel('Observation Frequency', fontsize=11)
ax.legend(loc='upper right')
plt.tight_layout()
c5_path = CHARTS_DIR / "05_intensity_distribution.png"
plt.savefig(c5_path)
plt.close()
print(f"   [+] Saved: {c5_path}")

print("\n" + "=" * 80)
print("✅ ALL 5 ANALYTICAL CHARTS GENERATED & SAVED IN 'traffic_analysis_charts/'")
print("=" * 80)
