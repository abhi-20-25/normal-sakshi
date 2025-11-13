#!/usr/bin/env python3
"""
One-time script to reset IN/OUT counts between 6 AM to 11 AM
Run this once and then delete it.
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

def reset_counts_6am_to_11am():
    """Reset IN/OUT counts between 6 PM to 11 PM for today"""
    try:
        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Get today's date in IST
        today = datetime.now(IST).date()
        
        # Define time range: 6 PM to 11 PM
        start_time = time(18, 0, 0)    # 6:00 PM
        end_time = time(23, 0, 0)      # 11:00 PM
        
        print(f"Resetting counts between 6 PM to 11 PM for today: {today}")
        print(f"  Time range: {start_time} to {end_time}")
        
        # First, show ALL hourly records to understand the data
        cursor.execute("""
            SELECT hour, in_count, out_count 
            FROM hourly_footfall 
            WHERE report_date = %s
            ORDER BY hour
        """, (today,))
        
        all_records = cursor.fetchall()
        print(f"\n📊 ALL hourly records for today:")
        all_in = 0
        all_out = 0
        for hour, in_count, out_count in all_records:
            if in_count > 0 or out_count > 0:
                print(f"  - Hour {hour}:00: IN={in_count}, OUT={out_count}")
                all_in += in_count
                all_out += out_count
        print(f"  Total from hourly: IN={all_in}, OUT={all_out}")
        
        # Step 1: Check and delete hourly_footfall records for 6 PM - 11 PM
        cursor.execute("""
            SELECT hour, in_count, out_count 
            FROM hourly_footfall 
            WHERE report_date = %s AND hour >= 18 AND hour < 23
            ORDER BY hour
        """, (today,))
        
        hourly_records = cursor.fetchall()
        print(f"\n📊 Found {len(hourly_records)} hourly records between 6 PM - 11 PM:")
        total_in = 0
        total_out = 0
        for hour, in_count, out_count in hourly_records:
            print(f"  - Hour {hour}:00: IN={in_count}, OUT={out_count}")
            total_in += in_count
            total_out += out_count
        
        if len(hourly_records) == 0:
            print("\n⚠️  No hourly records to delete for 6 PM - 11 PM.")
        else:
            print(f"\n  Total to subtract: IN={total_in}, OUT={total_out}")
        
        # Step 2: Check current daily_footfall
        cursor.execute("""
            SELECT in_count, out_count 
            FROM daily_footfall 
            WHERE report_date = %s
        """, (today,))
        
        daily_record = cursor.fetchone()
        if daily_record:
            current_in, current_out = daily_record
            print(f"\n📈 Current daily totals: IN={current_in}, OUT={current_out}")
            new_in = max(0, current_in - total_in)
            new_out = max(0, current_out - total_out)
            print(f"   After reset: IN={new_in}, OUT={new_out}")
        else:
            print(f"\n⚠️  No daily_footfall record found for {today}")
            return
        
        # Ask for confirmation
        if len(hourly_records) > 0:
            response = input(f"\n⚠️  Delete {len(hourly_records)} hourly records and update daily totals? (yes/no): ")
            if response.lower() != 'yes':
                print("Operation cancelled.")
                cursor.close()
                conn.close()
                return
            
            # Delete hourly records for 6 PM - 11 PM
            cursor.execute("""
                DELETE FROM hourly_footfall 
                WHERE report_date = %s AND hour >= 18 AND hour < 23
            """, (today,))
            deleted_hourly = cursor.rowcount
            
            # Update daily_footfall by subtracting the deleted counts
            cursor.execute("""
                UPDATE daily_footfall 
                SET in_count = %s, out_count = %s
                WHERE report_date = %s
            """, (new_in, new_out, today))
            
            conn.commit()
            
            print(f"\n✅ Successfully deleted {deleted_hourly} hourly records")
            print(f"✅ Updated daily totals: IN={new_in}, OUT={new_out}")
            print(f"\nThe dashboard will now show the corrected counts (excluding 6 PM - 11 PM)")
        else:
            print("\n✅ Nothing to reset - no records found in that time range")
        
        cursor.close()
        conn.close()
        
        print("\n⚠️  IMPORTANT: Please delete this script after running it!")
        print("Command: rm reset_counts_one_time.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("ONE-TIME RESET: Delete People Counter Data (6 PM to 11 PM)")
    print("=" * 60)
    reset_counts_6am_to_11am()
