#!/usr/bin/env python3
"""
Main Tea Toast Store Onboarding Script
Automatically adds main Tea Toast restaurant with all cameras and use cases
"""

import sys
import hashlib
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import logging
import pytz


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
IST = pytz.timezone('Asia/Kolkata')
DATABASE_URL = "postgresql://postgres:Tneural01@127.0.0.1:5432/sakshi"

Base = declarative_base()

# ============================================================================
# Database Models
# ============================================================================

class Restaurant(Base):
    __tablename__ = "restaurants"
    id = Column(Integer, primary_key=True)
    restaurant_code = Column(String(50), unique=True, nullable=False)
    restaurant_name = Column(String(200), nullable=False)
    location = Column(String(200))
    dvr_ip = Column(String(50))
    dvr_username = Column(String(100))
    dvr_password = Column(String(100))
    telegram_chat_id = Column(String(50))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))
    updated_at = Column(DateTime, default=lambda: datetime.now(IST))

class Camera(Base):
    __tablename__ = "cameras"
    id = Column(Integer, primary_key=True)
    restaurant_id = Column(Integer, ForeignKey('restaurants.id'))
    channel_number = Column(Integer)
    channel_name = Column(String(255), nullable=False)
    rtsp_url = Column(Text, nullable=False)
    channel_id = Column(String(50), unique=True, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))
    updated_at = Column(DateTime, default=lambda: datetime.now(IST))

class CameraApp(Base):
    __tablename__ = "camera_apps"
    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey('cameras.id'))
    app_name = Column(String(50), nullable=False)
    is_active = Column(Boolean, default=True)
    config = Column(JSONB)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))

# ============================================================================
# Main Tea Toast Store Configuration
# ============================================================================

MAIN_STORE_CONFIG = {
    'restaurant': {
        'restaurant_code': 'main_store',
        'restaurant_name': 'Tea Toast - Main',
        'location': 'Mumbai, Maharashtra',
        'dvr_ip': '182.65.205.121',
        'dvr_username': 'admin',
        'dvr_password': 'cctv#1234',
        'telegram_chat_id': '-4835836048',
        'is_active': True
    },
    'cameras': [
        {
            'channel_number': 1,
            'channel_name': 'Main Entrance',
            'rtsp_url': 'rtsp://admin:cctv%231234@182.65.205.121:554/cam/realmonitor?channel=1&subtype=1',
            'apps': [
                {'name': 'PeopleCounter', 'config': {'confidence': 0.15, 'model_path': 'models/yolo11n.pt'}}
            ]
        },
        {
            'channel_number': 4,
            'channel_name': 'Checkout Queue',
            'rtsp_url': 'rtsp://admin:cctv%231234@182.65.205.121:554/cam/realmonitor?channel=4&subtype=0',
            'apps': [
                {'name': 'QueueMonitor', 'config': {'confidence': 0.15, 'model_path': 'models/yolo11n.pt', 'alert_threshold': 3}}
            ]
        },
        {
            'channel_number': 5,
            'channel_name': 'Front Office Violation',
            'rtsp_url': 'rtsp://admin:cctv%231234@182.65.205.121:554/cam/realmonitor?channel=5&subtype=0',
            'apps': [
                {'name': 'Generic', 'config': {'confidence': 0.3, 'model_path': 'models/kitchen_violation_30_12_2025.pt', 'target_class_id': [1, 2, 3, 4, 5, 6, 7]}}
            ]
        },
        {
            'channel_number': 10,
            'channel_name': 'Kitchen Camera',
            'rtsp_url': 'rtsp://admin:cctv%231234@182.65.205.121:554/cam/realmonitor?channel=10&subtype=1',
            'apps': [
                {'name': 'KitchenCompliance', 'config': {'confidence': 0.3, 'model_path': 'models/kitchen_violation_30_12_2025.pt'}}
            ]
        },
        {
            'channel_number': 10,
            'channel_name': 'Kitchen Area - Occupancy',
            'rtsp_url': 'rtsp://admin:cctv%231234@182.65.205.121:554/cam/realmonitor?channel=10&subtype=0',
            'apps': [
                {'name': 'OccupancyMonitor', 'config': {'confidence': 0.15, 'model_path': 'models/yolo11n.pt'}}
            ]
        }
    ]
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_stable_channel_id(rtsp_url: str) -> str:
    """Generate stable hash-based channel ID from URL"""
    parsed = rtsp_url.lower().strip()
    hash_obj = hashlib.sha256(parsed.encode('utf-8'))
    return f"cam_{hash_obj.hexdigest()[:12]}"

# ============================================================================
# Main Onboarding Function
# ============================================================================

def onboard_main_store():
    """Main function to onboard main Tea Toast store"""
    
    logging.info("=" * 80)
    logging.info("🚀 MAIN TEA TOAST STORE ONBOARDING - STARTING")
    logging.info("=" * 80)
    logging.info("")
    
    # Create database connection
    try:
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()
        logging.info("✅ Database connection established")
    except Exception as e:
        logging.error(f"❌ Failed to connect to database: {e}")
        return False
    
    try:
        # =====================================================================
        # STEP 1: Create Restaurant
        # =====================================================================
        logging.info("")
        logging.info("📍 STEP 1: Creating Main Tea Toast Restaurant")
        logging.info("-" * 80)
        
        # Check if restaurant already exists
        existing = session.query(Restaurant).filter_by(
            restaurant_code=MAIN_STORE_CONFIG['restaurant']['restaurant_code']
        ).first()
        
        if existing:
            logging.warning(f"⚠️  Restaurant '{existing.restaurant_name}' already exists (ID: {existing.id})")
            restaurant = existing
        else:
            restaurant = Restaurant(**MAIN_STORE_CONFIG['restaurant'])
            session.add(restaurant)
            session.flush()
            logging.info(f"✅ Restaurant created: {restaurant.restaurant_name}")
            logging.info(f"   ID: {restaurant.id}")
            logging.info(f"   Code: {restaurant.restaurant_code}")
            logging.info(f"   Location: {restaurant.location}")
            logging.info(f"   DVR IP: {restaurant.dvr_ip}")
        
        # =====================================================================
        # STEP 2: Add Cameras
        # =====================================================================
        logging.info("")
        logging.info("📹 STEP 2: Adding Cameras")
        logging.info("-" * 80)
        
        cameras_added = 0
        cameras_updated = 0
        
        for cam_config in MAIN_STORE_CONFIG['cameras']:
            channel_id = get_stable_channel_id(cam_config['rtsp_url'])
            
            # Check if camera exists
            existing_cam = session.query(Camera).filter_by(channel_id=channel_id).first()
            
            if existing_cam:
                logging.info(f"ℹ️  Camera '{cam_config['channel_name']}' already exists (updating)")
                camera = existing_cam
                camera.is_active = True
                cameras_updated += 1
            else:
                camera = Camera(
                    restaurant_id=restaurant.id,
                    channel_number=cam_config['channel_number'],
                    channel_name=cam_config['channel_name'],
                    rtsp_url=cam_config['rtsp_url'],
                    channel_id=channel_id,
                    is_active=True
                )
                session.add(camera)
                session.flush()
                cameras_added += 1
                
            logging.info(f"✅ Channel {cam_config['channel_number']}: {cam_config['channel_name']}")
            logging.info(f"   Camera ID: {channel_id}")
            logging.info(f"   RTSP: {cam_config['rtsp_url'][:50]}...")
            
            # =====================================================================
            # STEP 3: Link Apps to Camera
            # =====================================================================
            for app_config in cam_config['apps']:
                # Check if app already linked
                existing_app = session.query(CameraApp).filter_by(
                    camera_id=camera.id,
                    app_name=app_config['name']
                ).first()
                
                if existing_app:
                    logging.info(f"   ℹ️  App '{app_config['name']}' already linked")
                    continue
                
                camera_app = CameraApp(
                    camera_id=camera.id,
                    app_name=app_config['name'],
                    is_active=True,
                    config=app_config['config']
                )
                session.add(camera_app)
                logging.info(f"   ✅ Linked: {app_config['name']}")
                logging.info(f"      Model: {app_config['config'].get('model_path', 'default')}")
                logging.info(f"      Confidence: {app_config['config'].get('confidence', 'default')}")
        
        # Commit all changes
        session.commit()
        
        # =====================================================================
        # STEP 4: Verification
        # =====================================================================
        logging.info("")
        logging.info("🔍 STEP 4: Verification")
        logging.info("-" * 80)
        
        # Count cameras
        active_cameras = session.query(Camera).filter_by(
            restaurant_id=restaurant.id,
            is_active=True
        ).count()
        
        # Count apps
        active_apps = session.query(CameraApp).join(Camera).filter(
            Camera.restaurant_id == restaurant.id,
            Camera.is_active == True,
            CameraApp.is_active == True
        ).count()
        
        logging.info(f"✅ Total Active Cameras: {active_cameras}")
        logging.info(f"✅ Total Active Apps: {active_apps}")
        
        # Show camera and app summary
        cameras_list = session.query(Camera).filter_by(
            restaurant_id=restaurant.id,
            is_active=True
        ).all()
        
        logging.info("")
        logging.info("📋 Camera & App Summary:")
        for cam in cameras_list:
            apps = session.query(CameraApp).filter_by(
                camera_id=cam.id,
                is_active=True
            ).all()
            app_names = ', '.join([app.app_name for app in apps])
            logging.info(f"   • Ch{cam.channel_number}: {cam.channel_name}")
            logging.info(f"     Apps: {app_names}")
        
        logging.info("")
        logging.info("=" * 80)
        logging.info("✅ MAIN TEA TOAST STORE ONBOARDING - COMPLETED SUCCESSFULLY!")
        logging.info("=" * 80)
        logging.info("")
        logging.info("📊 Summary:")
        logging.info(f"   • Restaurant: {restaurant.restaurant_name}")
        logging.info(f"   • Location: {restaurant.location}")
        logging.info(f"   • Cameras Added: {cameras_added}")
        logging.info(f"   • Cameras Updated: {cameras_updated}")
        logging.info(f"   • Total Active Cameras: {active_cameras}")
        logging.info(f"   • Total Apps Configured: {active_apps}")
        logging.info("")
        logging.info("⚠️  IMPORTANT NEXT STEPS:")
        logging.info("   1. Restart the application to load new cameras:")
        logging.info("      sudo systemctl restart sakshi-ai.service")
        logging.info("")
        logging.info("   2. Configure ROI (Region of Interest) for:")
        logging.info("      • PeopleCounter (Channel 1) - Draw counting line")
        logging.info("      • QueueMonitor (Channel 4) - Draw queue area")
        logging.info("      • OccupancyMonitor (Channel 10) - Draw monitoring zone")
        logging.info("")
        logging.info("   3. Test dashboard filtering:")
        logging.info(f"      http://localhost:5001/dashboard?restaurant_id={restaurant.id}")
        logging.info("")
        logging.info("   4. Verify cameras are streaming:")
        logging.info("      Check dashboard for 'Main' in restaurant dropdown")
        logging.info("")
        logging.info("=" * 80)
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Error during onboarding: {e}")
        session.rollback()
        return False
    finally:
        session.close()

# ============================================================================
# Script Entry Point
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "MAIN TEA TOAST STORE ONBOARDING SCRIPT" + " " * 20 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    success = onboard_main_store()
    
    if success:
        print("\n🎉 Onboarding completed successfully!\n")
        sys.exit(0)
    else:
        print("\n❌ Onboarding failed. Check logs above for details.\n")
        sys.exit(1)
