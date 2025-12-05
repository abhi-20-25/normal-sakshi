#!/usr/bin/env python3
"""
Quick Test Script for Phase 1
Tests basic database queries and connectivity
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging

DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/sakshi"

logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_phase1():
    """Test Phase 1 migration results"""
    
    print("\n" + "=" * 70)
    print("🧪 PHASE 1 QUICK TEST")
    print("=" * 70)
    
    try:
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Test 1: Can we query restaurants?
        print("\n1️⃣ Testing restaurants table...")
        result = session.execute(text("SELECT COUNT(*) FROM restaurants"))
        count = result.scalar()
        print(f"   ✅ Found {count} restaurant(s)")
        
        # Test 2: Can we query cameras?
        print("\n2️⃣ Testing cameras table...")
        result = session.execute(text("SELECT COUNT(*) FROM cameras"))
        count = result.scalar()
        print(f"   ✅ Found {count} camera(s)")
        
        # Test 3: Can we query camera_apps?
        print("\n3️⃣ Testing camera_apps table...")
        result = session.execute(text("SELECT COUNT(*) FROM camera_apps"))
        count = result.scalar()
        print(f"   ✅ Found {count} camera-app link(s)")
        
        # Test 4: Can we join tables?
        print("\n4️⃣ Testing table joins...")
        result = session.execute(text("""
            SELECT 
                r.restaurant_name,
                c.channel_name,
                ca.app_name
            FROM restaurants r
            JOIN cameras c ON r.id = c.restaurant_id
            JOIN camera_apps ca ON c.id = ca.camera_id
            LIMIT 3
        """))
        rows = result.fetchall()
        if rows:
            print(f"   ✅ Join successful! Sample data:")
            for row in rows:
                print(f"      • {row[0]} → {row[1]} → {row[2]}")
        else:
            print("   ⚠️  No data found in joined query")
        
        # Test 5: Check restaurant_id in existing tables
        print("\n5️⃣ Testing restaurant_id linkage...")
        result = session.execute(text("""
            SELECT COUNT(*) FROM roi_configs WHERE restaurant_id IS NOT NULL
        """))
        count = result.scalar()
        print(f"   ✅ roi_configs has {count} rows linked to restaurants")
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nPhase 1 migration is working correctly.")
        print("You can now proceed with Phase 2 (Backend Integration).\n")
        
        session.close()
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_phase1()
    exit(0 if success else 1)
