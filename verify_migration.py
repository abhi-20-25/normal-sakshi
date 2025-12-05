#!/usr/bin/env python3
"""
Verification Script for Phase 1 Migration
Displays the migrated data in a readable format
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
import logging

# Configuration
DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/sakshi"

logging.basicConfig(level=logging.INFO, format='%(message)s')

def verify_migration():
    """Verify the migration by displaying all migrated data"""
    
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        print("\n" + "=" * 80)
        print("🔍 PHASE 1 MIGRATION VERIFICATION")
        print("=" * 80)
        
        # 1. Show Restaurants
        print("\n📍 RESTAURANTS:")
        print("-" * 80)
        result = session.execute(text("""
            SELECT id, restaurant_code, restaurant_name, location, dvr_ip, is_active
            FROM restaurants
            ORDER BY id
        """))
        rows = result.fetchall()
        if rows:
            headers = ['ID', 'Code', 'Name', 'Location', 'DVR IP', 'Active']
            print(tabulate(rows, headers=headers, tablefmt='grid'))
        else:
            print("❌ No restaurants found!")
        
        # 2. Show Cameras
        print("\n📹 CAMERAS:")
        print("-" * 80)
        result = session.execute(text("""
            SELECT 
                c.id,
                r.restaurant_name,
                c.channel_number,
                c.channel_name,
                c.subtype,
                c.channel_id,
                c.is_active
            FROM cameras c
            JOIN restaurants r ON c.restaurant_id = r.id
            ORDER BY r.id, c.channel_number
        """))
        rows = result.fetchall()
        if rows:
            headers = ['ID', 'Restaurant', 'Ch#', 'Name', 'SubType', 'Channel ID', 'Active']
            print(tabulate(rows, headers=headers, tablefmt='grid'))
        else:
            print("❌ No cameras found!")
        
        # 3. Show Camera Apps
        print("\n🤖 CAMERA APPS:")
        print("-" * 80)
        result = session.execute(text("""
            SELECT 
                c.channel_name,
                c.channel_number,
                ca.app_name,
                ca.is_active,
                ca.config
            FROM camera_apps ca
            JOIN cameras c ON ca.camera_id = c.id
            ORDER BY c.channel_number, ca.app_name
        """))
        rows = result.fetchall()
        if rows:
            # Format config JSON for display
            formatted_rows = []
            for row in rows:
                formatted_rows.append([
                    row[0],  # channel_name
                    row[1],  # channel_number
                    row[2],  # app_name
                    '✓' if row[3] else '✗',  # is_active
                    str(row[4])[:50] if row[4] else '-'  # config (truncated)
                ])
            headers = ['Camera', 'Ch#', 'App', 'Active', 'Config']
            print(tabulate(formatted_rows, headers=headers, tablefmt='grid'))
        else:
            print("❌ No camera apps found!")
        
        # 4. Statistics
        print("\n📊 STATISTICS:")
        print("-" * 80)
        
        result = session.execute(text("""
            SELECT 
                COUNT(DISTINCT r.id) as restaurants,
                COUNT(DISTINCT c.id) as cameras,
                COUNT(DISTINCT ca.id) as camera_apps,
                COUNT(DISTINCT ca.app_name) as unique_apps
            FROM restaurants r
            LEFT JOIN cameras c ON r.id = c.restaurant_id
            LEFT JOIN camera_apps ca ON c.id = ca.camera_id
        """))
        stats = result.fetchone()
        
        print(f"  • Restaurants: {stats[0]}")
        print(f"  • Cameras: {stats[1]}")
        print(f"  • Camera-App Links: {stats[2]}")
        print(f"  • Unique Apps: {stats[3]}")
        
        # 5. Apps per Camera Summary
        print("\n📋 APPS PER CAMERA:")
        print("-" * 80)
        result = session.execute(text("""
            SELECT 
                c.channel_name,
                STRING_AGG(ca.app_name, ', ' ORDER BY ca.app_name) as apps
            FROM cameras c
            LEFT JOIN camera_apps ca ON c.id = ca.camera_id
            GROUP BY c.channel_name
            ORDER BY c.channel_name
        """))
        rows = result.fetchall()
        if rows:
            headers = ['Camera', 'Linked Apps']
            print(tabulate(rows, headers=headers, tablefmt='grid'))
        
        # 6. Check for issues
        print("\n⚠️  VALIDATION CHECKS:")
        print("-" * 80)
        
        # Check for cameras without apps
        result = session.execute(text("""
            SELECT c.channel_name
            FROM cameras c
            LEFT JOIN camera_apps ca ON c.id = ca.camera_id
            WHERE ca.id IS NULL
        """))
        orphaned = result.fetchall()
        if orphaned:
            print(f"  ⚠️  Cameras without apps: {len(orphaned)}")
            for cam in orphaned:
                print(f"     - {cam[0]}")
        else:
            print("  ✅ All cameras have apps assigned")
        
        # Check for duplicate channel IDs
        result = session.execute(text("""
            SELECT channel_id, COUNT(*) as count
            FROM cameras
            GROUP BY channel_id
            HAVING COUNT(*) > 1
        """))
        duplicates = result.fetchall()
        if duplicates:
            print(f"  ⚠️  Duplicate channel IDs: {len(duplicates)}")
            for dup in duplicates:
                print(f"     - {dup[0]} (appears {dup[1]} times)")
        else:
            print("  ✅ No duplicate channel IDs")
        
        # Check restaurant_id in existing tables
        print("\n🔗 EXISTING DATA LINKAGE:")
        print("-" * 80)
        
        tables = [
            'roi_configs',
            'detections',
            'daily_footfall',
            'hourly_footfall',
            'queue_logs',
            'kitchen_violations',
            'occupancy_logs',
            'occupancy_schedules'
        ]
        
        for table in tables:
            try:
                result = session.execute(text(f"""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(restaurant_id) as linked
                    FROM {table}
                """))
                counts = result.fetchone()
                status = "✅" if counts[0] == counts[1] else "⚠️"
                print(f"  {status} {table}: {counts[1]}/{counts[0]} rows linked to restaurant")
            except Exception as e:
                print(f"  ❌ {table}: Error - {e}")
        
        print("\n" + "=" * 80)
        print("✅ Verification Complete!")
        print("=" * 80)
        print("\nIf everything looks good, proceed to Phase 2 (Backend Integration)")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()

if __name__ == "__main__":
    verify_migration()
