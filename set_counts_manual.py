#!/usr/bin/env python3
"""
Manually set the IN/OUT counts for today to specific values
"""

import psycopg2
from datetime import datetime
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

def set_manual_counts():
    """Manually set IN/OUT counts for today"""
    try:
        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Get today's date in IST
        today = datetime.now(IST).date()
        
        # Step 1: Show current counts
        cursor.execute("""
            SELECT in_count, out_count 
            FROM daily_footfall 
            WHERE report_date = %s
        """, (today,))
        
        daily_record = cursor.fetchone()
        if daily_record:
            current_in, current_out = daily_record
            print(f"📈 Current daily totals for {today}:")
            print(f"   IN={current_in}, OUT={current_out}")
        else:
            print(f"⚠️  No daily_footfall record found for {today}")
            return
        
        # Step 2: Show hourly breakdown
        cursor.execute("""
            SELECT hour, in_count, out_count 
            FROM hourly_footfall 
            WHERE report_date = %s
            ORDER BY hour
        """, (today,))
        
        hourly_records = cursor.fetchall()
        print(f"\n📊 Hourly breakdown:")
        hourly_in = 0
        hourly_out = 0
        for hour, in_count, out_count in hourly_records:
            if in_count > 0 or out_count > 0:
                print(f"   {hour:2d}:00 - IN={in_count:3d}, OUT={out_count:3d}")
                hourly_in += in_count
                hourly_out += out_count
        
        print(f"\n   Hourly total: IN={hourly_in}, OUT={hourly_out}")
        print(f"   Daily total:  IN={current_in}, OUT={current_out}")
        print(f"   Difference:   IN={current_in - hourly_in}, OUT={current_out - hourly_out}")
        print(f"\n   (Difference = counts from current/incomplete hours)")
        
        # Step 3: Ask for new values
        print(f"\n" + "="*50)
        print("Enter new values to set:")
        print("="*50)
        
        try:
            new_in = int(input(f"New IN count (current={current_in}): "))
            new_out = int(input(f"New OUT count (current={current_out}): "))
        except ValueError:
            print("❌ Invalid input. Must be integers.")
            return
        
        if new_in < 0 or new_out < 0:
            print("❌ Counts cannot be negative.")
            return
        
        # Step 4: Confirm
        print(f"\n⚠️  Confirm changes:")
        print(f"   Current: IN={current_in}, OUT={current_out}")
        print(f"   New:     IN={new_in}, OUT={new_out}")
        print(f"   Change:  IN={new_in - current_in:+d}, OUT={new_out - current_out:+d}")
        
        response = input(f"\nApply these changes? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled.")
            cursor.close()
            conn.close()
            return
        
        # Step 5: Update the database
        cursor.execute("""
            UPDATE daily_footfall 
            SET in_count = %s, out_count = %s
            WHERE report_date = %s
        """, (new_in, new_out, today))
        
        conn.commit()
        
        print(f"\n✅ Successfully updated daily counts!")
        print(f"   New totals: IN={new_in}, OUT={new_out}")
        print(f"\n🔄 IMPORTANT: Restart the application to see changes!")
        print(f"   Command: sudo systemctl restart sakshi-ai.service")
        print(f"   Or: pkill -f edit-004.py && python3 edit-004.py")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("MANUAL COUNT SETTER: Set exact IN/OUT values for today")
    print("=" * 60)
    set_manual_counts()
