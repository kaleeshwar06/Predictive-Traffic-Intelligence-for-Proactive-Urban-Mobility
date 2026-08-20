import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / 'innohack' / 'backend'))
import detrac_pipeline, cctv_video_processor

print("1. Testing DETRAC pipeline...")
detrac_dir = BASE_DIR / 'detrac_sample_data'
dp = detrac_pipeline.DETRACPipeline(str(detrac_dir))
print("   - Discovered sequences:", dp.cached_sequences)
seq = dp.parse_sequence_xml('MVI_20011')
print("   - Sequence XML loaded, frame count:", seq['total_frames'])

print("2. Testing Video Processor with DETRAC frames...")
vp = cctv_video_processor.CCTVVideoProcessor(detrac_dir=str(detrac_dir / 'DETRAC-train-data' / 'MVI_20011'))
for i in range(3):
    f = vp.process_next_frame()
    print(f"   - Frame {f['frame_index']}: Vehicles={f['total_vehicles']} | PCU={f['total_pcu']} | Congestion={f['congestion_pct']}% ({f['status']})")

print("3. Pipeline verification SUCCESSFUL!")
