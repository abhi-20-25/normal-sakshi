#!/usr/bin/env python3
"""
Quick test script to check if server responds
Run this while gunicorn is starting
"""

import requests
import time
import sys

def test_server(max_wait=30):
    """Test if server responds within max_wait seconds"""
    url = "http://localhost:5001/"
    start_time = time.time()
    
    print(f"Testing server at {url} (max wait: {max_wait}s)...")
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(url, timeout=2)
            elapsed = time.time() - start_time
            print(f"✓ Server responded in {elapsed:.1f}s (Status: {response.status_code})")
            return True
        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            print(f"  Still waiting... ({elapsed:.1f}s)")
        except requests.exceptions.ConnectionError:
            elapsed = time.time() - start_time
            print(f"  Connection refused... ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  Error: {e}")
        
        time.sleep(1)
    
    print(f"✗ Server did not respond within {max_wait}s")
    return False

if __name__ == "__main__":
    success = test_server()
    sys.exit(0 if success else 1)

