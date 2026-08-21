# DESIGN.md — ROUTES BY VENOM (Professional Navigation System)

## Product Vision & Visual Direction
The visual visual system for **ROUTES BY VENOM** models real-world professional navigation software and modern SaaS mobility products (such as Google Maps, Waze, and enterprise traffic operations platforms). 

It emphasizes visual clarity, map dominance, readable typography, and actionable routing data. It avoids futuristic gimmicks, cyberpunk neon glows, HUD elements, glassmorphism, or excessive monospaced technical telemetry.

---

## 1. Color Palette

### Base & Backgrounds
- **Application Background (`--bg-main`):** `#0F172A` (Slate 900 — Clean Dark Slate)
- **Panel & Card Fill (`--bg-surface`):** `#1E293B` (Slate 800 — Solid Card Background)
- **Hover / Active State (`--bg-subtle`):** `#334155` (Slate 700 — Interactive Elevation)
- **Borders & Dividers (`--border`):** `#334155` (Slate 700 — Crisp 1px Borders)

### Primary & Route Colors
- **Brand Primary (`--primary-blue`):** `#2563EB` (Professional Royal Blue)
- **Primary Hover (`--primary-blue-hover`):** `#1D4ED8`
- **Recommended Route (`--route-optimal`):** `#16A34A` (Clean Emerald Green)
- **Alternative Route (`--route-alt`):** `#3B82F6` (Solid Navy Blue)
- **Moderate Traffic (`--traffic-mod`):** `#EAB308` (Clean Amber/Yellow)
- **Heavy Traffic (`--traffic-heavy`):** `#F97316` (Solid Orange)
- **Severe Congestion / Accident (`--traffic-severe`):** `#DC2626` (Solid Red)

### Neutral Text Hierarchy
- **Primary Text (`--text-main`):** `#F8FAFC` (Slate 50 — Crisp High Contrast White)
- **Secondary Text (`--text-sub`):** `#94A3B8` (Slate 400 — Muted Subtitles)
- **Tertiary Text (`--text-muted`):** `#64748B` (Slate 500 — Subtle Labels)

---

## 2. Typography

- **Single Universal Typeface:** `'Inter', system-ui, -apple-system, sans-serif`
- **Case Rules:** Sentence case throughout the UI (e.g. "Find Best Route", "Traffic Volume Increase", "Search history"). Avoid forced uppercase.

### Font Weight & Hierarchy
- **Header Brand:** `18px` | Weight: `700` | Line Height: `1.2`
- **Section Headers:** `15px` | Weight: `600` | Line Height: `1.3`
- **Card Titles:** `14px` | Weight: `600` | Line Height: `1.4`
- **Body Text:** `13px` | Weight: `400` | Line Height: `1.5`
- **Subtext & Badges:** `11px` | Weight: `500` | Line Height: `1.2`

---

## 3. Spacing & Layout System

- **Grid System:** Standard 8px linear rhythm (`4px`, `8px`, `12px`, `16px`, `24px`, `32px`).
- **Map Viewport Dominance:** The interactive map occupies 70%–80% of screen width.
- **Sidebar Width:** Fixed 380px on desktop docked to the right side.
- **Navigation Tabs:** Simple clean tabs: `Traffic`, `Planner`, `History`.

---

## 4. Component Design Guidelines

### Header
- Clean horizontal bar (`#0F172A`) containing:
  - Left: **ROUTES BY VENOM** • Predictive Traffic Intelligence
  - Right: Current location/system status (e.g., `System Online`) and live time clock.

### Route Cards
- Emphasized recommended route with solid green border badge (`RECOMMENDED`) and clear ETA, distance, and arrival time.
- Clean alternative route cards with subtle slate background and border.

### CCTV & YOLO Detection Panel
- Natural video frame with crisp 1px slate border (`#334155`).
- Clean vehicle classification table below video displaying vehicle counts, class breakdown (Cars, Motorcycles, Vans, Buses, Trucks), and PCU load.

### What-If Scenario Simulator
- Clean compact horizontal bar above the map with standard range sliders and crisp checkboxes for Rain, Accident, and Closure scenarios.

### 24-Hour Congestion Chart
- Clean SVG line chart displaying congestion trend curves over 24 hours with a vertical marker for current time (`NOW`).

---

## 5. Responsive Behavior
- **Desktop (>1024px):** Fixed 380px sidebar, dominant 80% map viewport.
- **Tablet & Mobile (<900px):** Responsive bottom drawer sheet for route planner and traffic analytics.
