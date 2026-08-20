import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

print("Testing Public Free Traffic Video & Camera APIs...")

# 1. TfL JamCams API (London) - Returns live MP4 video clips & JPEGs
print("\n[1] Testing Transport for London (TfL) JamCams API:")
try:
    url = "https://api.tfl.gov.uk/Place/Type/JamCam"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
        data = json.loads(resp.read().decode())
        print(f"   [OK] TfL JamCams Active! Found {len(data)} live traffic cameras across London.")
        sample = data[0]
        print(f"        - Camera: {sample.get('commonName')}")
        props = {p.get('key'): p.get('value') for p in sample.get('additionalProperties', [])}
        print(f"        - Live Video MP4 URL: {props.get('videoUrl')}")
        print(f"        - Live Snapshot URL:  {props.get('imageUrl')}")
except Exception as e:
    print(f"   [-] TfL JamCams: {e}")

# 2. Singapore LTA DataMall Traffic Images API
print("\n[2] Testing Singapore Government Traffic Camera API:")
try:
    sg_url = "https://api.data.gov.sg/v1/transport/traffic-images"
    req_sg = urllib.request.Request(sg_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req_sg, timeout=10, context=ctx) as resp_sg:
        data_sg = json.loads(resp_sg.read().decode())
        cameras = data_sg.get("items", [])[0].get("cameras", [])
        print(f"   [OK] Singapore OpenGov Active! Found {len(cameras)} live highway expressway cameras.")
        sample_sg = cameras[0]
        print(f"        - Camera ID: {sample_sg.get('camera_id')}")
        print(f"        - Live Image URL: {sample_sg.get('image')}")
except Exception as e:
    print(f"   [-] Singapore API: {e}")

# 3. Madrid Open Data Traffic Cameras
print("\n[3] Testing Madrid City Council Traffic Cameras (Ayuntamiento de Madrid):")
try:
    madrid_url = "https://informo.madrid.es/informo/tmadrid/CCTV.kml"
    req_m = urllib.request.Request(madrid_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req_m, timeout=10, context=ctx) as resp_m:
        content = resp_m.read()
        print(f"   [OK] Madrid Traffic Camera KML Feed Active! Response Size: {len(content):,} bytes.")
except Exception as e:
    print(f"   [-] Madrid Camera Feed: {e}")
