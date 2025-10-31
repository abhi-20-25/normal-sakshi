#!/usr/bin/env python3
"""
Quick diagnostic script to check if Gunicorn setup is working
Run this after starting gunicorn service
"""

import requests
import sys

def check_status():
    """Check various endpoints to diagnose issues"""
    base_url = "http://localhost:5001"
    
    print("=" * 60)
    print("Gunicorn Service Diagnostic Check")
    print("=" * 60)
    
    # 1. Check if server is running
    print("\n1. Checking if server is running...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"   ✓ Server is running (Status: {response.status_code})")
    except requests.exceptions.ConnectionError:
        print("   ✗ Server is NOT running - check gunicorn service")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # 2. Check CUDA status (no auth needed after your changes)
    print("\n2. Checking CUDA status...")
    try:
        response = requests.get(f"{base_url}/api/cuda_status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ CUDA Status API working")
            print(f"   - CUDA Available: {data.get('cuda_available')}")
            print(f"   - Disabled Processors: {len(data.get('disabled_processors', []))}")
            print(f"   - Error Counts: {data.get('error_counts', {})}")
        else:
            print(f"   ⚠ CUDA Status returned: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 3. Check login page
    print("\n3. Checking login page...")
    try:
        response = requests.get(f"{base_url}/login", timeout=5)
        if response.status_code == 200:
            print("   ✓ Login page accessible")
        else:
            print(f"   ⚠ Login page returned: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 4. Try to get video feed info (will need auth, but check if endpoint exists)
    print("\n4. Checking video feed endpoint...")
    try:
        response = requests.get(f"{base_url}/video_feed/PeopleCounter/cam_test", timeout=5, allow_redirects=False)
        if response.status_code == 302 or response.status_code == 401:
            print("   ✓ Video feed endpoint exists (requires authentication)")
        elif response.status_code == 404:
            print("   ✓ Video feed endpoint exists (channel not found, but endpoint works)")
        else:
            print(f"   ⚠ Video feed returned: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("1. Check gunicorn logs: sudo journalctl -u sakshi-ai -f")
    print("2. Look for 'Application initialized' message")
    print("3. Check for 'Total processors started' message")
    print("4. Verify RTSP streams are accessible from server")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    check_status()

