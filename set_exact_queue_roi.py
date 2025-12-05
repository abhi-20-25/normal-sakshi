#!/usr/bin/env python3
"""
Update Queue Monitor ROI with Exact Coordinates
"""

import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Database configuration
DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/sakshi"
CHANNEL_ID = "cam_f822b0bf4e"

# EXACT ROI Configuration (as provided)
EXACT_ROI_CONFIG = {
    "main": [
        [0.5549999952316285, 0.5744444105360244],
        [0.4456249952316284, 0.5272221883138021],
        [0.3081249952316284, 0.3105555216471354],
        [0.08624999523162842, 0.4272221883138021],
        [0.19249999523162842, 0.7938888549804688]
    ],
    "secondary": [
        [0.5924999952316284, 0.5355555216471354],
        [0.49874999523162844, 0.502222188313802],
        [0.3487499952316284, 0.31333329942491317],
        [0.38156249523162844, 0.3105555216471354],
        [0.3940624952316284, 0.2883332994249132],
        [0.5003124952316285, 0.26888885498046877],
        [0.6721874952316285, 0.4633332994249132]
    ]
}

def update_roi():
    """Update the Queue Monitor ROI with exact coordinates"""
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Convert to JSON string
        roi_json = json.dumps(EXACT_ROI_CONFIG)
        
        print("=" * 70)
        print("🎯 Updating Queue Monitor ROI Configuration")
        print("=" * 70)
        print(f"\nTarget Channel: {CHANNEL_ID}")
        print(f"\n🔵 Main ROI (Queue Area): {len(EXACT_ROI_CONFIG['main'])} points")
        for i, point in enumerate(EXACT_ROI_CONFIG['main'], 1):
            print(f"   Point {i}: [{point[0]}, {point[1]}]")
        
        print(f"\n🟢 Secondary ROI (Counter Area): {len(EXACT_ROI_CONFIG['secondary'])} points")
        for i, point in enumerate(EXACT_ROI_CONFIG['secondary'], 1):
            print(f"   Point {i}: [{point[0]}, {point[1]}]")
        
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
            print("\n" + "=" * 70)
            print("✅ Successfully updated Queue Monitor ROI!")
            print("=" * 70)
            print("\n⚠️  IMPORTANT: Restart the application for changes to take effect!")
            print("\n   Option 1 (if running as service):")
            print("   sudo systemctl restart sakshi-ai.service")
            print("\n   Option 2 (if running manually):")
            print("   Stop the current edit-004.py process (Ctrl+C)")
            print("   Then run: python3 edit-004.py")
            print("\n" + "=" * 70)
            return True
        else:
            print("\n❌ Error: No ROI configuration found for this channel")
            print("   The channel might not exist in the database")
            return False
            
    except Exception as e:
        print(f"\n❌ Error updating ROI: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def verify_update():
    """Verify the ROI was updated correctly"""
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
            print("\n" + "=" * 70)
            print("✅ Verification: ROI Configuration in Database")
            print("=" * 70)
            roi_data = json.loads(result[2])
            
            print(f"\n🔵 Main ROI: {len(roi_data.get('main', []))} points")
            for i, point in enumerate(roi_data.get('main', []), 1):
                print(f"   Point {i}: [{point[0]}, {point[1]}]")
            
            print(f"\n🟢 Secondary ROI: {len(roi_data.get('secondary', []))} points")
            for i, point in enumerate(roi_data.get('secondary', []), 1):
                print(f"   Point {i}: [{point[0]}, {point[1]}]")
            
            # Verify exact match
            if roi_data == EXACT_ROI_CONFIG:
                print("\n✅ Perfect match! ROI coordinates are exactly as specified.")
            else:
                print("\n⚠️  Warning: Saved ROI doesn't exactly match (possibly formatting)")
            
            print("=" * 70)
            return True
        else:
            print(f"\n❌ Could not verify: No ROI found for {CHANNEL_ID}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        return False
    finally:
        session.close()

if __name__ == "__main__":
    print("\n")
    if update_roi():
        verify_update()
    print("\n")
