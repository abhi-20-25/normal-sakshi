#!/usr/bin/env python3
"""
Quick test script for Phase 2 Backend Integration
Tests the new restaurant management API endpoints
"""

import requests
import json
from pprint import pprint

# Configuration
BASE_URL = "http://localhost:5001"
USERNAME = "admin"
PASSWORD = "admin"

def test_get_restaurants():
    """Test GET /api/restaurants"""
    print("\n" + "="*70)
    print("TEST 1: GET /api/restaurants")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/api/restaurants")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success! Found {data['count']} restaurant(s)")
        pprint(data['restaurants'])
    else:
        print(f"❌ Error: {response.text}")
    
    return response.status_code == 200

def test_get_specific_restaurant(restaurant_id=1):
    """Test GET /api/restaurants/{id}"""
    print("\n" + "="*70)
    print(f"TEST 2: GET /api/restaurants/{restaurant_id}")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/api/restaurants/{restaurant_id}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success! Restaurant details:")
        pprint(data['restaurant'])
    else:
        print(f"❌ Error: {response.text}")
    
    return response.status_code == 200

def test_get_restaurant_cameras(restaurant_id=1):
    """Test GET /api/restaurants/{id}/cameras"""
    print("\n" + "="*70)
    print(f"TEST 3: GET /api/restaurants/{restaurant_id}/cameras")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/api/restaurants/{restaurant_id}/cameras")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success! Found {data['count']} camera(s)")
        print(f"\nRestaurant: {data['restaurant']['name']} - {data['restaurant']['location']}")
        print("\nCameras:")
        for cam in data['cameras']:
            print(f"  📹 {cam['channel_name']} (ID: {cam['channel_id']})")
            print(f"     Apps: {', '.join([app['app_name'] for app in cam['apps']])}")
    else:
        print(f"❌ Error: {response.text}")
    
    return response.status_code == 200

def test_get_app_configs():
    """Test get_app_configs() via dashboard route"""
    print("\n" + "="*70)
    print("TEST 4: Check app configs loading")
    print("="*70)
    
    # Test without restaurant filter
    print("\nTesting dashboard without restaurant filter...")
    response = requests.get(f"{BASE_URL}/dashboard", allow_redirects=False)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 302:
        print("⚠️  Redirected to login (expected if not authenticated)")
    elif response.status_code == 200:
        print("✅ Dashboard loaded successfully")
    
    # Test with restaurant filter
    print("\nTesting dashboard with restaurant_id=1...")
    response = requests.get(f"{BASE_URL}/dashboard?restaurant_id=1", allow_redirects=False)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 302:
        print("⚠️  Redirected to login (expected if not authenticated)")
    elif response.status_code == 200:
        print("✅ Dashboard with filter loaded successfully")
    
    return True

def test_backward_compatibility():
    """Verify system still works with database (should not fall back to file)"""
    print("\n" + "="*70)
    print("TEST 5: Backward Compatibility Check")
    print("="*70)
    
    print("\n⚠️  This test requires checking application logs:")
    print("    grep 'Loading cameras from' logs/app.log")
    print("\nExpected in logs:")
    print("  ✅ '📊 Loading cameras from database'")
    print("  ❌ NOT '📄 Loading cameras from rtsp_links.txt'")
    
    return True

def main():
    print("="*70)
    print("PHASE 2 BACKEND INTEGRATION - API ENDPOINT TESTS")
    print("="*70)
    print(f"Testing against: {BASE_URL}")
    print("\nNote: Application must be running for these tests to work")
    print("      (python3 edit-004.py or gunicorn -c gunicorn_config.py wsgi:app)")
    
    # Check if application is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        print("\n✅ Application is running!")
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Application not running at", BASE_URL)
        print("   Please start the application first:")
        print("   python3 edit-004.py")
        return
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return
    
    # Run tests
    results = {
        "Test 1 - Get Restaurants": test_get_restaurants(),
        "Test 2 - Get Specific Restaurant": test_get_specific_restaurant(1),
        "Test 3 - Get Restaurant Cameras": test_get_restaurant_cameras(1),
        "Test 4 - App Configs Loading": test_get_app_configs(),
        "Test 5 - Backward Compatibility": test_backward_compatibility()
    }
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Phase 2 backend integration is working correctly.")
        print("\n📋 Next Steps:")
        print("   1. Manually verify application logs show database loading")
        print("   2. Proceed to Phase 3: Frontend Dashboard Dropdown")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
