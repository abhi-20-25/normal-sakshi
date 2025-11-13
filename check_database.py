#!/usr/bin/env python3
"""
Script to check what's in the database
"""

import psycopg2
from datetime import datetime, time
import pytz

# Database configuration
DB_CONFIG = {
    'dbname': 'sakshi',
    'user': 'postgres',
    'password': 'Tneural01',
    'host': '127.0.0.1',
    'port': 5432
}

# IST timezone
IST = pytz.timezone('Asia/Kolkata')

try:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    today = datetime.now(IST).date()
    start_datetime = IST.localize(datetime.combine(today, time(0, 0, 0)))
    end_datetime = IST.localize(datetime.combine(today, time(23, 59, 59)))
    
    print("=" * 70)
    print("DATABASE INSPECTION - Today's Records")
    print("=" * 70)
    print(f"Date: {today}")
    print()
    
    # 1. Show all app_names and their counts
    cursor.execute("""
        SELECT app_name, COUNT(*) as count
        FROM detections 
        WHERE timestamp >= %s AND timestamp <= %s
        GROUP BY app_name
        ORDER BY count DESC
    """, (start_datetime, end_datetime))
    
    app_names = cursor.fetchall()
    print("App Names in database (today):")
    for app_name, count in app_names:
        print(f"  - {app_name}: {count} records")
    print()
    
    # 2. Show sample messages for each app_name
    print("Sample messages from each app_name:")
    print("-" * 70)
    for app_name, _ in app_names:
        cursor.execute("""
            SELECT message, timestamp
            FROM detections 
            WHERE timestamp >= %s AND timestamp <= %s
            AND app_name = %s
            ORDER BY timestamp DESC
            LIMIT 3
        """, (start_datetime, end_datetime, app_name))
        
        samples = cursor.fetchall()
        print(f"\n{app_name}:")
        for msg, ts in samples:
            print(f"  [{ts}] {msg}")
    
    print()
    print("=" * 70)
    
    # 3. Search for IN/OUT messages
    cursor.execute("""
        SELECT app_name, message, timestamp
        FROM detections 
        WHERE timestamp >= %s AND timestamp <= %s
        AND (message LIKE '%IN%' OR message LIKE '%OUT%' OR message LIKE '%entered%' OR message LIKE '%exited%')
        ORDER BY timestamp DESC
        LIMIT 20
    """, (start_datetime, end_datetime))
    
    in_out_records = cursor.fetchall()
    if in_out_records:
        print("\nRecords containing IN/OUT/entered/exited:")
        print("-" * 70)
        for app_name, msg, ts in in_out_records:
            print(f"  [{app_name}] {ts}: {msg}")
    else:
        print("\n⚠️  No records found containing IN/OUT/entered/exited")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
