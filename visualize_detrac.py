"""
UA-DETRAC Data Viewer & Visualizer
==================================
Easily view, inspect, and extract bounding boxes from UA-DETRAC frames.
Saves annotated preview frames to 'detrac_preview/' folder and opens the image.
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageDraw

# Windows console safe utf-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DETRAC_COLORS = {
    "car": (99, 102, 241),       # Indigo
    "van": (16, 185, 129),      # Green
    "bus": (239, 68, 68),       # Red
    "others": (245, 158, 11)    # Amber
}

def visualize_frame(sequence="MVI_20011", frame_num=1, base_dir="d:/Innohack/detrac_sample_data", out_dir="d:/Innohack/detrac_preview", auto_open=True):
    """Loads an image frame and overlays bounding boxes from DETRAC XML annotations."""
    base = Path(base_dir)
    img_path = base / "DETRAC-train-data" / sequence / f"img{frame_num:05d}.jpg"
    xml_path = base / "DETRAC-Train-Annotations-XML" / f"{sequence}.xml"
    
    if not img_path.exists():
        print(f"[-] Image not found: {img_path}")
        return None
    if not xml_path.exists():
        print(f"[-] XML not found: {xml_path}")
        return None

    # Open image
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Parse XML for target boxes at frame_num
    tree = ET.parse(xml_path)
    root = tree.getroot()

    found_targets = []
    for frame in root.findall("frame"):
        if int(frame.attrib.get("num", 0)) == frame_num:
            t_list = frame.find("target_list")
            if t_list is not None:
                for target in t_list.findall("target"):
                    tid = target.attrib.get("id", "0")
                    box = target.find("box")
                    attr = target.find("attribute")

                    if box is not None:
                        left = float(box.attrib.get("left", 0))
                        top = float(box.attrib.get("top", 0))
                        w = float(box.attrib.get("width", 0))
                        h = float(box.attrib.get("height", 0))

                        vtype = "car"
                        speed = 1.0
                        if attr is not None:
                            vtype = attr.attrib.get("vehicle_type", "car").lower()
                            if vtype not in DETRAC_COLORS:
                                vtype = "others"
                            speed = float(attr.attrib.get("speed", 1.0))

                        color = DETRAC_COLORS.get(vtype, (255, 255, 255))
                        
                        # Draw bounding box
                        draw.rectangle([left, top, left + w, top + h], outline=color, width=3)
                        # Draw label
                        label = f"{vtype.upper()} #{tid} (Spd: {speed})"
                        draw.rectangle([left, max(0, top - 18), left + 140, top], fill=color)
                        draw.text((left + 4, max(0, top - 16)), label, fill=(255, 255, 255))

                        found_targets.append({"id": tid, "class": vtype, "box": (left, top, w, h), "speed": speed})

    # Save output preview image
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_file = out_path / f"{sequence}_frame_{frame_num:05d}_annotated.jpg"
    img.save(save_file, "JPEG", quality=95)

    print("=" * 65)
    print(f"Visualized Sequence: {sequence} | Frame: {frame_num:05d}")
    print(f"Image Path: {img_path}")
    print(f"Annotation: {xml_path}")
    print(f"Saved Annotated Preview: {save_file}")
    print(f"Total Detections in Frame: {len(found_targets)}")
    for t in found_targets:
        print(f"   * {t['class'].upper()} (ID: {t['id']}) at [x={t['box'][0]}, y={t['box'][1]}, w={t['box'][2]}, h={t['box'][3]}]")
    print("=" * 65)

    # Automatically open the image on Windows Photos if requested
    if auto_open and sys.platform == "win32":
        try:
            os.startfile(str(save_file))
        except Exception:
            pass

    return save_file

if __name__ == "__main__":
    visualize_frame("MVI_20011", 1)
