"""
Standalone Zero-Dependency HTTP API & UI Server for AI/ML-09 Traffic Congestion System.
Runs CCTV & UA-DETRAC Video Processor Engine to calculate live vehicle counts, PCU density,
and congestion levels frame-by-frame directly from traffic video monitoring.
Serves the RoutePulse HTML Dashboard directly at http://localhost:8000
"""

import json
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import bmd45_pipeline
import detrac_pipeline
import iot_simulator
import predictor
import route_optimizer
import cctv_video_processor

pipe = bmd45_pipeline.BMD45Pipeline()
detrac_pipe = detrac_pipeline.DETRACPipeline()
sim = iot_simulator.IoTSimulator()
pred = predictor.TrafficPredictor()
opt = route_optimizer.RouteOptimizer()
video_proc = cctv_video_processor.CCTVVideoProcessor()

HTML_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "traffic_dashboard.html")

class TrafficAPIHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # Serve HTML Dashboard UI at root /
        if path == "/" or path == "/index.html" or path == "/dashboard":
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self._send_cors_headers()
            self.end_headers()
            try:
                with open(HTML_FILE_PATH, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                self.wfile.write(html_content.encode('utf-8'))
            except Exception as e:
                self.wfile.write(f"<h1>Error loading dashboard HTML: {e}</h1>".encode('utf-8'))
            return

        # Serve Live Video Stream (MJPEG format)
        if path == "/api/video/feed":
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self._send_cors_headers()
            self.end_headers()
            try:
                for _ in range(50):
                    frame_data = video_proc.process_next_frame()
                    jpg_bytes = frame_data["jpeg_bytes"]
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(jpg_bytes)))
                    self.end_headers()
                    self.wfile.write(jpg_bytes)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.04) # ~25 FPS
            except Exception:
                pass
            return

        # Serve Live Single Video Frame JPEG
        if path == "/api/video/frame":
            frame_data = video_proc.process_next_frame()
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(frame_data["jpeg_bytes"])
            return

        # Serve Live Video Computed Telemetry JSON
        if path == "/api/video/telemetry":
            frame_data = video_proc.process_next_frame()
            telemetry = {k: v for k, v in frame_data.items() if k != "jpeg_bytes"}
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(telemetry, indent=2).encode('utf-8'))
            return

        # Serve API endpoints
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self._send_cors_headers()
        self.end_headers()

        response = {}

        if path == "/api" or path == "/api/info":
            response = {
                "status": "ONLINE",
                "server": "CCTV & UA-DETRAC Traffic Processing Server",
                "system": "Traffic Congestion AI Engine (AI/ML-09)",
                "endpoints": [
                    "/api/video/feed",
                    "/api/video/telemetry",
                    "/api/traffic/live",
                    "/api/traffic/predict",
                    "/api/detrac/sample",
                    "/api/detrac/sequences",
                    "/api/camera/sample",
                    "/api/route/optimize"
                ]
            }
        elif path == "/api/traffic/live":
            frame_data = video_proc.process_next_frame()
            video_telemetry = {k: v for k, v in frame_data.items() if k != "jpeg_bytes"}

            telemetry = sim.get_live_telemetry()
            telemetry[0]["congestion_pct"] = video_telemetry["congestion_pct"]
            telemetry[0]["current_speed_kmh"] = video_telemetry["avg_speed_kmh"]
            telemetry[0]["delay_min"] = video_telemetry["delay_min"]
            telemetry[0]["status"] = video_telemetry["status"]
            telemetry[0]["status_color"] = video_telemetry["status_color"]

            response = {
                "corridors_count": len(telemetry),
                "video_analytics": video_telemetry,
                "telemetry": telemetry
            }
        elif path == "/api/detrac/sample":
            seq = query.get("seq", ["MVI_20011"])[0]
            f_num = int(query.get("frame", [1])[0])
            response = detrac_pipe.get_sample_telemetry(seq, f_num)
        elif path == "/api/detrac/sequences":
            response = {
                "dataset": "UA-DETRAC Benchmark",
                "available_sequences": detrac_pipe.cached_sequences,
                "base_dir": str(detrac_pipe.base_dir)
            }
        elif path == "/api/traffic/predict":
            telemetry = sim.get_live_telemetry()
            response = pred.get_citywide_forecast(telemetry)
        elif path == "/api/camera/sample":
            sample_id = query.get("sample_id", [None])[0]
            response = pipe.get_cctv_sample(sample_id)
        elif path == "/api/route/optimize":
            route_id = query.get("route_id", [None])[0]
            response = opt.calculate_optimized_route(route_id)
        else:
            response = {"error": "Endpoint not found", "path": path}

        self.wfile.write(json.dumps(response, indent=2).encode('utf-8'))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/emergency/greenwave":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'
            try:
                body = json.loads(post_data.decode('utf-8'))
            except Exception:
                body = {}

            corridor = body.get("corridor", "Silk Board Junction")
            hospital = body.get("hospital_name", "St. John's Hospital")

            result = opt.activate_emergency_green_wave(corridor, hospital)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(result, indent=2).encode('utf-8'))

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, TrafficAPIHandler)
    print("=================================================================")
    print(f" [+] CCTV & UA-DETRAC Video Processing Server running at http://localhost:{port}")
    print(" Press Ctrl+C to stop the server.")
    print("=================================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")

if __name__ == "__main__":
    run_server(8000)
