#!/usr/bin/env python3
"""
Diagnostic script to check camera feeds and processor status
"""

import cv2
import time

def test_camera_feed(url, name="Camera"):
    """Test if a camera feed is accessible"""
    print(f"\n{'='*60}")
    print(f"Testing {name}: {url}")
    print('='*60)
    
    try:
        cap = cv2.VideoCapture(url)
        
        if not cap.isOpened():
            print(f"❌ Failed to open camera feed")
            return False
        
        print(f"✅ Camera opened successfully")
        
        # Try to read a frame
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print(f"❌ Failed to read frame from camera")
            cap.release()
            return False
        
        print(f"✅ Frame captured successfully")
        print(f"   Frame shape: {frame.shape}")
        print(f"   Frame size: {frame.shape[1]}x{frame.shape[0]}")
        
        # Try to read a few more frames
        success_count = 0
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                success_count += 1
            time.sleep(0.1)
        
        print(f"✅ Successfully read {success_count}/5 additional frames")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Error testing camera: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CAMERA FEED DIAGNOSTIC")
    print("="*60)
    
    # Test with 0 (default webcam)
    test_camera_feed(0, "Webcam (device 0)")
    
    # Add your RTSP/HTTP camera URLs here
    # Example:
    # test_camera_feed("rtsp://admin:password@192.168.1.100:554/stream1", "Camera 1")
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    print("\nIf all cameras show ✅, then the issue is in edit-004.py")
    print("If cameras show ❌, check:")
    print("  - Camera URLs are correct")
    print("  - Network connectivity")
    print("  - Camera credentials")
    print("  - Firewall settings")
