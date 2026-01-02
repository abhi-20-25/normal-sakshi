#!/usr/bin/env python3
"""
Script to backup ROI configurations from the database to roi.txt
"""
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable is not set")
    sys.exit(1)

# Create database connection
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def backup_roi_data():
    """Fetch ROI data from database and save to roi.txt"""
    db = SessionLocal()
    
    try:
        # Query all ROI configurations (includes QueueMonitor ROIs and PeopleCounter counting lines)
        query = text("""
            SELECT id, channel_id, app_name, roi_points, restaurant_id
            FROM roi_configs
            ORDER BY app_name, id
        """)
        
        result = db.execute(query)
        rows = result.fetchall()
        
        if not rows:
            print("⚠️  No ROI configurations found in database")
            return
        
        # Prepare backup data
        backup_data = {
            "backup_timestamp": datetime.now().isoformat(),
            "total_records": len(rows),
            "roi_configs": []
        }
        
        for row in rows:
            roi_config = {
                "id": row[0],
                "channel_id": row[1],
                "app_name": row[2],
                "roi_points": json.loads(row[3]) if row[3] else None,
                "restaurant_id": row[4]
            }
            backup_data["roi_configs"].append(roi_config)
        
        # Write to roi.txt
        output_file = "roi.txt"
        with open(output_file, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        print("=" * 60)
        print("✅ ROI Backup Completed Successfully")
        print("=" * 60)
        print(f"📁 Output file: {output_file}")
        print(f"📊 Total records backed up: {len(rows)}")
        print(f"⏰ Backup timestamp: {backup_data['backup_timestamp']}")
        print()
        
        # Display summary of each ROI config
        for idx, config in enumerate(backup_data["roi_configs"], 1):
            print(f"{idx}. ID: {config['id']}")
            print(f"   Channel: {config['channel_id']}")
            print(f"   App: {config['app_name']}")
            print(f"   Restaurant ID: {config['restaurant_id']}")
            
            if config['roi_points']:
                roi_data = config['roi_points']
                
                # Handle PeopleCounter (counting line position)
                if config['app_name'] == 'PeopleCounter' and isinstance(roi_data, dict):
                    if 'line_position' in roi_data:
                        line_pos = roi_data['line_position']
                        print(f"   📏 Counting Line Position: {line_pos*100:.1f}%")
                        print(f"      (LEFT: 0-{line_pos*100:.1f}% = IN, RIGHT: {line_pos*100:.1f}%-100% = OUT)")
                
                # Handle QueueMonitor (main and secondary ROI)
                elif isinstance(roi_data, dict):
                    main_count = len(roi_data.get('main', []))
                    secondary_count = len(roi_data.get('secondary', []))
                    print(f"   🔵 Main ROI points: {main_count}")
                    print(f"   🟢 Secondary ROI points: {secondary_count}")
                else:
                    print(f"   ROI points: {len(roi_data) if isinstance(roi_data, list) else 'N/A'}")
            print()
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error backing up ROI data: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    backup_roi_data()
