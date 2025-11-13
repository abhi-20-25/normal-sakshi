#!/usr/bin/env python3
"""
Download YOLOv11n model for better tracking performance
"""
from ultralytics import YOLO
import os

print("=" * 60)
print("Downloading YOLOv11n model...")
print("=" * 60)

# Download YOLO11n (this will auto-download from Ultralytics)
# Just specify the model name without .pt, it will download automatically
model = YOLO('yolo11n')  # This triggers auto-download

# Verify model exists
if os.path.exists('yolo11n.pt'):
    file_size = os.path.getsize('yolo11n.pt') / (1024 * 1024)  # MB
    print(f"\n✅ YOLOv11n downloaded successfully!")
    print(f"📦 File: yolo11n.pt")
    print(f"📊 Size: {file_size:.1f} MB")
    print("\n" + "=" * 60)
    print("YOLO11 advantages over YOLO8:")
    print("=" * 60)
    print("✅ Improved tracking accuracy (better ID persistence)")
    print("✅ Better occlusion handling (people passing each other)")
    print("✅ Faster inference on CPU")
    print("✅ More stable bounding boxes (less jitter)")
    print("✅ Better small object detection")
    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print("1. Update edit-004.py to use 'yolo11n.pt'")
    print("2. Restart the application")
    print("3. Test counting with 2-3 people passing together")
    print("=" * 60)
else:
    print("❌ Download failed!")
