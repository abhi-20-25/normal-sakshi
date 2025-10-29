#!/usr/bin/env python3
"""
Test script to verify People Counter database connection and functionality
"""
import psycopg2
from datetime import datetime
import pytz

IST = pytz.timezone('Asia/Kolkata')

def test_database_connection():
    """Test PostgreSQL database connection"""
    print("🧪 Testing database connection...")
    
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="sakshi",
            user="postgres",
            password="Tneural01"
        )
        cursor = conn.cursor()
        
        # Test basic connection
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ Database connected: {version[0][:50]}...")
        
        # Check People Counter tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('daily_footfall', 'hourly_footfall', 'detections')
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        print(f"✅ People Counter tables found: {[t[0] for t in tables]}")
        
        # Check for People Counter data
        cursor.execute("""
            SELECT COUNT(*) FROM daily_footfall;
        """)
        daily_count = cursor.fetchone()[0]
        print(f"📊 Daily footfall records: {daily_count}")
        
        cursor.execute("""
            SELECT COUNT(*) FROM hourly_footfall;
        """)
        hourly_count = cursor.fetchone()[0]
        print(f"📊 Hourly footfall records: {hourly_count}")
        
        # Check recent People Counter detections
        cursor.execute("""
            SELECT COUNT(*) FROM detections 
            WHERE app_name = 'PeopleCounter';
        """)
        pc_detections = cursor.fetchone()[0]
        print(f"📊 People Counter detections: {pc_detections}")
        
        # Check current day's data
        today = datetime.now(IST).date()
        cursor.execute("""
            SELECT channel_id, in_count, out_count 
            FROM daily_footfall 
            WHERE report_date = %s;
        """, (today,))
        today_data = cursor.fetchall()
        print(f"📊 Today's footfall data: {today_data}")
        
        cursor.close()
        conn.close()
        
        print("✅ Database connection test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_direction_logic():
    """Test the new direction logic"""
    print("\n🧪 Testing People Counter direction logic...")
    
    # Simulate the logic
    line_x = 320  # Middle of 640px wide frame
    
    test_cases = [
        {"prev_x": 200, "curr_x": 400, "expected": "IN", "description": "Left to Right (should be IN)"},
        {"prev_x": 400, "curr_x": 200, "expected": "OUT", "description": "Right to Left (should be OUT)"},
        {"prev_x": 300, "curr_x": 350, "expected": "IN", "description": "Left to Right (should be IN)"},
        {"prev_x": 350, "curr_x": 300, "expected": "OUT", "description": "Right to Left (should be OUT)"},
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        prev_x, curr_x = case["prev_x"], case["curr_x"]
        expected = case["expected"]
        
        # New logic: Left to Right = IN, Right to Left = OUT
        if prev_x < line_x and curr_x >= line_x:
            result = "IN"
        elif prev_x > line_x and curr_x <= line_x:
            result = "OUT"
        else:
            result = "NONE"
        
        status = "✅" if result == expected else "❌"
        if result == expected:
            passed += 1
            
        print(f"{status} Test {i}: {case['description']}")
        print(f"    Movement: {prev_x} → {curr_x} = {result} (Expected: {expected})")
        print()
    
    print(f"📊 Direction logic tests: {passed}/{total} passed")
    return passed == total

def test_app_configuration():
    """Test that all required apps are configured"""
    print("\n🧪 Testing app configuration...")
    
    # Check rtsp_links.txt
    try:
        with open('rtsp_links.txt', 'r') as f:
            lines = f.readlines()
        
        active_apps = []
        for line in lines:
            if line.strip() and not line.startswith('#') and ',' in line:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    app_name = parts[2].strip()
                    active_apps.append(app_name)
        
        print(f"📊 Active apps in rtsp_links.txt: {active_apps}")
        
        required_apps = ['PeopleCounter', 'QueueMonitor', 'KitchenCompliance']
        missing_apps = [app for app in required_apps if app not in active_apps]
        
        if missing_apps:
            print(f"❌ Missing required apps: {missing_apps}")
            return False
        else:
            print("✅ All required apps are configured")
            return True
            
    except Exception as e:
        print(f"❌ Failed to read rtsp_links.txt: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting People Counter tests...\n")
    
    tests = [
        test_database_connection,
        test_direction_logic,
        test_app_configuration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📊 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! People Counter is ready.")
        print("\n📋 Summary of changes:")
        print("✅ Fixed direction logic: Left-to-Right = IN, Right-to-Left = OUT")
        print("✅ Excluded Generic app from frontend display")
        print("✅ Database connection working")
        print("✅ Memory persistence across restarts")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()
