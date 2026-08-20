import sys
sys.path.append('d:/Innohack/innohack/backend')
import detrac_pipeline, cctv_video_processor

print("1. Testing DETRAC pipeline...")
dp = detrac_pipeline.DETRACPipeline('d:/Innohack/detrac_sample_data')
print("   - Discovered sequences:", dp.cached_sequences)
seq = dp.parse_sequence_xml('MVI_20011')
print("   - Sequence XML loaded, frame count:", seq['total_frames'])

print("2. Testing Video Processor with DETRAC frames...")
vp = cctv_video_processor.CCTVVideoProcessor(detrac_dir='d:/Innohack/detrac_sample_data/DETRAC-train-data/MVI_20011')
for i in range(3):
    f = vp.process_next_frame()
    print(f"   - Frame {f['frame_index']}: Vehicles={f['total_vehicles']} | PCU={f['total_pcu']} | Congestion={f['congestion_pct']}% ({f['status']})")

print("3. Pipeline verification SUCCESSFUL!")
