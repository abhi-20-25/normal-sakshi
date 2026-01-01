#!/usr/bin/env python3
"""
Channel ID Mismatch Fix Script
Diagnoses and fixes channel_id mismatches between cameras and footfall data
"""

import sys
import hashlib
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

DATABASE_URL = "postgresql://postgres:Tneural01@127.0.0.1:5432/sakshi"

def get_stable_channel_id(rtsp_url: str) -> str:
    """Generate stable hash-based channel ID from URL"""
    parsed = rtsp_url.lower().strip()
    hash_obj = hashlib.sha256(parsed.encode('utf-8'))
    return f"cam_{hash_obj.hexdigest()[:12]}"

def diagnose_channel_mismatch():
    """Diagnose channel_id mismatches"""
    
    logging.info("=" * 80)
    logging.info("🔍 CHANNEL ID MISMATCH DIAGNOSTIC")
    logging.info("=" * 80)
    
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # 1. Get all cameras
        logging.info("\n📹 Current Cameras in Database:")
        logging.info("-" * 80)
        cameras = session.execute(text("""
            SELECT c.id, c.restaurant_id, r.restaurant_name, c.channel_number, 
                   c.channel_name, c.channel_id, c.rtsp_url
            FROM cameras c
            JOIN restaurants r ON c.restaurant_id = r.id
            WHERE c.is_active = true
            ORDER BY r.restaurant_name, c.channel_number
        """)).fetchall()
        
        for cam in cameras:
            logging.info(f"ID: {cam[0]} | Restaurant: {cam[2]} | Ch{cam[3]}: {cam[4]}")
            logging.info(f"   Channel ID: {cam[5]}")
            logging.info(f"   RTSP: {cam[6][:60]}...")
        
        # 2. Get all channel_ids in footfall data
        logging.info("\n📊 Channel IDs in Footfall Data:")
        logging.info("-" * 80)
        footfall_channels = session.execute(text("""
            SELECT channel_id, 
                   COUNT(*) as record_count,
                   MIN(report_date) as first_date,
                   MAX(report_date) as last_date,
                   SUM(in_count) as total_in,
                   SUM(out_count) as total_out
            FROM hourly_footfall
            GROUP BY channel_id
            ORDER BY first_date, channel_id
        """)).fetchall()
        
        camera_ids = {cam[5] for cam in cameras}
        orphaned_data = []
        
        for fc in footfall_channels:
            status = "✅ LINKED" if fc[0] in camera_ids else "❌ ORPHANED"
            logging.info(f"{status} | {fc[0]}")
            logging.info(f"   Records: {fc[1]} | Date Range: {fc[2]} to {fc[3]}")
            logging.info(f"   Total In: {fc[4]} | Total Out: {fc[5]}")
            
            if fc[0] not in camera_ids:
                orphaned_data.append(fc)
        
        # 3. Identify orphaned data
        if orphaned_data:
            logging.info("\n⚠️  ORPHANED FOOTFALL DATA FOUND:")
            logging.info("-" * 80)
            for od in orphaned_data:
                logging.info(f"Channel ID: {od[0]}")
                logging.info(f"   {od[1]} records | {od[2]} to {od[3]}")
                logging.info(f"   Total footfall: {od[4]} in / {od[5]} out")
        
        # 4. Try to match orphaned data to cameras
        logging.info("\n🔗 Attempting to Match Orphaned Data:")
        logging.info("-" * 80)
        
        # Check if cam_3df702bb28 might be the old Main Entrance
        main_entrance_cameras = [cam for cam in cameras if 'Main Entrance' in cam[4] or cam[3] == 1]
        
        logging.info("\nPossible matches for cam_3df702bb28 (main historical data):")
        for cam in main_entrance_cameras:
            logging.info(f"   Restaurant: {cam[2]}")
            logging.info(f"   Camera: Ch{cam[3]} - {cam[4]}")
            logging.info(f"   Current ID: {cam[5]}")
            logging.info(f"   Camera DB ID: {cam[0]}")
        
        # 5. Suggest fix options
        logging.info("\n💡 FIX OPTIONS:")
        logging.info("-" * 80)
        logging.info("Option 1: Update camera to use old channel_id (preserves link)")
        logging.info("   - Best if you want to keep historical continuity")
        logging.info("   - Update main store's Main Entrance camera to use 'cam_3df702bb28'")
        logging.info("")
        logging.info("Option 2: Update footfall data to use new channel_id (migrates data)")
        logging.info("   - Best if you want consistent channel_id generation")
        logging.info("   - Update all footfall records from 'cam_3df702bb28' to new ID")
        logging.info("")
        logging.info("Option 3: Keep both (creates duplicate tracking)")
        logging.info("   - Historical data stays with old ID")
        logging.info("   - New data uses new ID")
        
        return {
            'cameras': cameras,
            'footfall_channels': footfall_channels,
            'orphaned_data': orphaned_data,
            'main_entrance_cameras': main_entrance_cameras
        }
        
    finally:
        session.close()

def apply_fix_option_1(camera_db_id: int, old_channel_id: str):
    """Update camera to use old channel_id"""
    
    logging.info("\n🔧 APPLYING FIX OPTION 1: Update Camera Channel ID")
    logging.info("-" * 80)
    
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        result = session.execute(text("""
            UPDATE cameras 
            SET channel_id = :old_channel_id,
                updated_at = NOW()
            WHERE id = :camera_id
            RETURNING id, channel_name, channel_id
        """), {'old_channel_id': old_channel_id, 'camera_id': camera_db_id})
        
        updated = result.fetchone()
        session.commit()
        
        logging.info(f"✅ Updated Camera ID {updated[0]}: {updated[1]}")
        logging.info(f"   New channel_id: {updated[2]}")
        logging.info("\n⚠️  IMPORTANT: Restart the application!")
        logging.info("   sudo systemctl restart sakshi-ai.service")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def apply_fix_option_2(old_channel_id: str, new_channel_id: str):
    """Update footfall data to use new channel_id"""
    
    logging.info("\n🔧 APPLYING FIX OPTION 2: Migrate Footfall Data")
    logging.info("-" * 80)
    
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Update hourly_footfall
        result_hourly = session.execute(text("""
            UPDATE hourly_footfall 
            SET channel_id = :new_channel_id
            WHERE channel_id = :old_channel_id
            RETURNING id
        """), {'old_channel_id': old_channel_id, 'new_channel_id': new_channel_id})
        
        hourly_count = len(result_hourly.fetchall())
        
        # Update daily_footfall
        result_daily = session.execute(text("""
            UPDATE daily_footfall 
            SET channel_id = :new_channel_id
            WHERE channel_id = :old_channel_id
            RETURNING id
        """), {'old_channel_id': old_channel_id, 'new_channel_id': new_channel_id})
        
        daily_count = len(result_daily.fetchall())
        
        session.commit()
        
        logging.info(f"✅ Updated {hourly_count} hourly footfall records")
        logging.info(f"✅ Updated {daily_count} daily footfall records")
        logging.info(f"   Old channel_id: {old_channel_id}")
        logging.info(f"   New channel_id: {new_channel_id}")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        session.rollback()
        return False
    finally:
        session.close()

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 22 + "CHANNEL ID MISMATCH FIX SCRIPT" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    # Run diagnostics
    results = diagnose_channel_mismatch()
    
    print("\n")
    print("=" * 80)
    print("INTERACTIVE FIX")
    print("=" * 80)
    print("\nWhich fix would you like to apply?")
    print("1. Update Main Store's Main Entrance camera to use old channel_id 'cam_3df702bb28'")
    print("2. Update historical footfall data to use new channel_id")
    print("3. Just show diagnostics (no changes)")
    print("\n")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice == "1":
        # Find Main Store's Main Entrance camera
        main_camera = None
        for cam in results['main_entrance_cameras']:
            if 'Main' in cam[2] and 'Mumbai' in cam[2]:  # Main store
                main_camera = cam
                break
        
        if main_camera:
            confirm = input(f"\nUpdate camera '{main_camera[4]}' (ID: {main_camera[0]}) to use 'cam_3df702bb28'? (yes/no): ")
            if confirm.lower() == 'yes':
                apply_fix_option_1(main_camera[0], 'cam_3df702bb28')
        else:
            print("❌ Could not find Main Store's Main Entrance camera")
    
    elif choice == "2":
        # Find the new channel_id for Main Store's Main Entrance
        main_camera = None
        for cam in results['main_entrance_cameras']:
            if 'Main' in cam[2] and 'Mumbai' in cam[2]:  # Main store
                main_camera = cam
                break
        
        if main_camera:
            confirm = input(f"\nUpdate footfall data from 'cam_3df702bb28' to '{main_camera[5]}'? (yes/no): ")
            if confirm.lower() == 'yes':
                apply_fix_option_2('cam_3df702bb28', main_camera[5])
        else:
            print("❌ Could not find Main Store's Main Entrance camera")
    
    else:
        print("\n✅ Diagnostics complete. No changes made.")
    
    print("\n")
