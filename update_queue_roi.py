#!/usr/bin/env python3
"""
Update Queue Monitor ROI Configuration
This script updates the ROI polygons for the Queue Monitor in the database.
Coordinates should be normalized between 0.0 and 1.0.
"""

import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Database configuration
DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/sakshi"
CHANNEL_ID = "cam_f822b0bf4e"

# NEW ROI Configuration (normalized coordinates 0.0 to 1.0)
# Adjust these coordinates to match your desired queue and counter areas
NEW_ROI_CONFIG = {
    "main": [
        # Main queue area polygon (example - adjust these points)
        [0.35, 0.40],  # Point 1
        [0.65, 0.40],  # Point 2
        [0.70, 0.80],  # Point 3
        [0.30, 0.80]   # Point 4
    ],
    "secondary": [
        # Counter/cashier area polygon (example - adjust these points)
        [0.60, 0.30],  # Point 1
        [0.80, 0.30],  # Point 2
        [0.80, 0.60],  # Point 3
        [0.60, 0.60]   # Point 4
    ]
}

def update_queue_roi():
    """Update the Queue Monitor ROI in the database"""
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Convert to JSON string
        roi_json = json.dumps(NEW_ROI_CONFIG)
        
        # Update the database
        query = text("""
            UPDATE roi_configs 
            SET roi_points = :roi_points
            WHERE channel_id = :channel_id AND app_name = 'QueueMonitor'
        """)
        
        result = session.execute(query, {
            'roi_points': roi_json,
            'channel_id': CHANNEL_ID
        })
        session.commit()
        
        if result.rowcount > 0:
            print(f"✅ Successfully updated Queue Monitor ROI for channel {CHANNEL_ID}")
            print(f"📍 Main ROI points: {len(NEW_ROI_CONFIG['main'])}")
            print(f"📍 Secondary ROI points: {len(NEW_ROI_CONFIG['secondary'])}")
            print("\n⚠️  IMPORTANT: Restart the application for changes to take effect!")
            print("   Run: sudo systemctl restart sakshi-ai.service")
        else:
            print(f"❌ No ROI configuration found for channel {CHANNEL_ID}")
            print("   Creating new entry...")
            
            # Insert new entry if not exists
            insert_query = text("""
                INSERT INTO roi_configs (channel_id, app_name, roi_points)
                VALUES (:channel_id, 'QueueMonitor', :roi_points)
            """)
            session.execute(insert_query, {
                'channel_id': CHANNEL_ID,
                'roi_points': roi_json
            })
            session.commit()
            print(f"✅ Created new Queue Monitor ROI for channel {CHANNEL_ID}")
            
    except Exception as e:
        print(f"❌ Error updating ROI: {e}")
        session.rollback()
    finally:
        session.close()

def view_current_roi():
    """View the current ROI configuration"""
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        query = text("""
            SELECT channel_id, app_name, roi_points 
            FROM roi_configs 
            WHERE channel_id = :channel_id AND app_name = 'QueueMonitor'
        """)
        
        result = session.execute(query, {'channel_id': CHANNEL_ID}).fetchone()
        
        if result:
            print(f"\n📊 Current ROI Configuration for {CHANNEL_ID}:")
            print(f"   App Name: {result[1]}")
            roi_data = json.loads(result[2])
            print(f"   Main ROI: {len(roi_data.get('main', []))} points")
            print(f"   Secondary ROI: {len(roi_data.get('secondary', []))} points")
            print(f"\n   Full config:")
            print(f"   {json.dumps(roi_data, indent=2)}")
        else:
            print(f"❌ No ROI configuration found for {CHANNEL_ID}")
            
    except Exception as e:
        print(f"❌ Error viewing ROI: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Queue Monitor ROI Configuration Tool")
    print("=" * 60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "view":
        view_current_roi()
    else:
        print(f"\n🎯 Target Channel: {CHANNEL_ID}")
        print(f"\n📝 New ROI Configuration:")
        print(f"   Main ROI (Queue): {len(NEW_ROI_CONFIG['main'])} points")
        print(f"   Secondary ROI (Counter): {len(NEW_ROI_CONFIG['secondary'])} points")
        print("\nCoordinates are normalized (0.0 to 1.0):")
        print(json.dumps(NEW_ROI_CONFIG, indent=2))
        
        print("\n" + "=" * 60)
        response = input("\nProceed with update? (yes/no): ")
        
        if response.lower() in ['yes', 'y']:
            update_queue_roi()
        else:
            print("❌ Update cancelled")
    
    print("\n" + "=" * 60)
    print("\n💡 Tip: Run 'python update_queue_roi.py view' to see current config")
    print("=" * 60)
