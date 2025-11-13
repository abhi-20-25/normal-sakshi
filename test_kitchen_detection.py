#!/usr/bin/env python3
"""
Test script to verify Kitchen Compliance detection is working correctly.
Checks:
1. GPU availability and usage
2. Model loading
3. Detection capabilities (gloves, apron, cap, uniform, phone)
"""

import torch
import cv2
import sys
from ultralytics import YOLO

def test_kitchen_compliance():
    print("=" * 70)
    print("KITCHEN COMPLIANCE DETECTION TEST")
    print("=" * 70)
    
    # 1. Check GPU availability
    print("\n1. GPU Status:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   Current GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
    else:
        print("   ⚠️  WARNING: No GPU available, will use CPU")
    
    # 2. Load models
    print("\n2. Loading Models:")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")
    
    try:
        print("   Loading yolo11n.pt (general person detection)...", end=" ")
        general_model = YOLO('yolo11n.pt')
        print("✅ Loaded")
        
        print("   Loading apron-cap.pt (apron/cap detection)...", end=" ")
        apron_cap_model = YOLO('apron-cap.pt')
        print("✅ Loaded")
        
        print("   Loading gloves.pt (gloves detection)...", end=" ")
        gloves_model = YOLO('gloves.pt')
        print("✅ Loaded")
        
    except Exception as e:
        print(f"\n   ❌ ERROR loading models: {e}")
        return False
    
    # 3. Test RTSP connection
    print("\n3. Testing RTSP Connection:")
    rtsp_url = "rtsp://admin:cctv%231234@182.65.205.121:554/cam/realmonitor?channel=10&subtype=1"
    print(f"   Connecting to: {rtsp_url}")
    
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("   ❌ ERROR: Could not connect to RTSP stream")
        return False
    
    print("   ✅ Connected successfully")
    
    # 4. Test detection on a single frame
    print("\n4. Testing Detection on Single Frame:")
    ret, frame = cap.read()
    if not ret:
        print("   ❌ ERROR: Could not read frame from stream")
        cap.release()
        return False
    
    print(f"   Frame size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Test person detection
    print("   Running person detection...", end=" ")
    person_results = general_model.track(frame, persist=True, classes=[0], conf=0.5, 
                                        verbose=False, device=device)
    num_persons = len(person_results[0].boxes) if person_results[0].boxes is not None else 0
    print(f"✅ Detected {num_persons} person(s)")
    
    # Test apron/cap detection
    print("   Running apron/cap detection...", end=" ")
    apron_results = apron_cap_model(frame, conf=0.5, verbose=False, device=device)
    num_apron_cap = len(apron_results[0].boxes) if apron_results[0].boxes is not None else 0
    print(f"✅ Detected {num_apron_cap} apron/cap item(s)")
    
    # Test gloves detection
    print("   Running gloves detection...", end=" ")
    gloves_results = gloves_model(frame, conf=0.5, verbose=False, device=device)
    num_gloves = len(gloves_results[0].boxes) if gloves_results[0].boxes is not None else 0
    print(f"✅ Detected {num_gloves} glove(s)")
    
    # Test phone detection (class 67 = cell phone in COCO)
    print("   Running phone detection...", end=" ")
    phone_results = general_model(frame, classes=[67], conf=0.5, verbose=False, device=device)
    num_phones = len(phone_results[0].boxes) if phone_results[0].boxes is not None else 0
    print(f"✅ Detected {num_phones} phone(s)")
    
    cap.release()
    
    # 5. GPU memory after inference
    if torch.cuda.is_available():
        print("\n5. GPU Memory After Inference:")
        print(f"   Memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
        print(f"   Memory cached: {torch.cuda.memory_reserved(0) / 1e6:.2f} MB")
    
    # 6. Summary
    print("\n" + "=" * 70)
    print("DETECTION SUMMARY:")
    print("=" * 70)
    print(f"   Persons detected: {num_persons}")
    print(f"   Apron/Cap items: {num_apron_cap}")
    print(f"   Gloves detected: {num_gloves}")
    print(f"   Phones detected: {num_phones}")
    print(f"   Device used: {device.upper()}")
    
    if num_persons > 0:
        print("\n✅ Kitchen Compliance detection is WORKING!")
        if num_gloves == 0 and num_persons > 0:
            print("⚠️  Violation Alert: Person detected without gloves!")
        if num_apron_cap == 0 and num_persons > 0:
            print("⚠️  Violation Alert: Person detected without apron/cap!")
    else:
        print("\nℹ️  No persons detected in current frame (may be empty kitchen)")
    
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_kitchen_compliance()
    sys.exit(0 if success else 1)
