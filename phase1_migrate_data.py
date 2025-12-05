#!/usr/bin/env python3
"""
Phase 1: Data Migration Script
Migrates rtsp_links.txt data to the new database schema
"""

import re
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
DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/sakshi"
RTSP_LINKS_FILE = 'rtsp_links.txt'

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
    channel_number = Column(Integer, nullable=False)
    channel_name = Column(String(100), nullable=False)
    subtype = Column(Integer, default=0)
    rtsp_url = Column(Text)
    channel_id = Column(String(100), unique=True)
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
# Helper Functions
# ============================================================================

def get_stable_channel_id(url: str) -> str:
    """Generate stable hash-based channel ID from URL"""
    parsed = url.lower().strip()
    hash_obj = hashlib.sha256(parsed.encode('utf-8'))
    return f"cam_{hash_obj.hexdigest()[:12]}"

def parse_rtsp_url(rtsp_url: str):
    """Parse RTSP URL to extract components"""
    # Example: rtsp://admin:cctv%231234@182.65.205.121:554/cam/realmonitor?channel=10&subtype=1
    
    pattern = r'rtsp://([^:]+):([^@]+)@([^:]+):(\d+)/.*channel=(\d+).*subtype=(\d+)'
    match = re.search(pattern, rtsp_url)
    
    if match:
        return {
            'username': match.group(1),
            'password': match.group(2).replace('%23', '#'),  # URL decode
            'ip': match.group(3),
            'port': int(match.group(4)),
            'channel': int(match.group(5)),
            'subtype': int(match.group(6))
        }
    return None

def get_app_config(app_name: str):
    """Get default configuration for each app"""
    configs = {
        'PeopleCounter': {'confidence': 0.15},
        'QueueMonitor': {'confidence': 0.15, 'alert_threshold': 3},
        'KitchenCompliance': {'confidence': 0.35},
        'Generic': {'confidence': 0.35, 'target_class_id': [0, 2, 4, 6, 7, 8]},
        'OccupancyMonitor': {'confidence': 0.15},
    }
    return configs.get(app_name, {})

# ============================================================================
# Migration Functions
# ============================================================================

def migrate_rtsp_links():
    """Main migration function"""
    
    logging.info("=" * 70)
    logging.info("🚀 Starting Phase 1 Data Migration")
    logging.info("=" * 70)
    
    # Create database connection
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Get Tea Toast restaurant (should exist from schema creation)
        restaurant = session.query(Restaurant).filter_by(
            restaurant_code='tea_toast'
        ).first()
        
        if not restaurant:
            logging.error("❌ Tea Toast restaurant not found in database!")
            logging.error("   Please run phase1_create_schema.sql first")
            return False
        
        logging.info(f"✅ Found restaurant: {restaurant.restaurant_name} (ID: {restaurant.id})")
        
        # Check if rtsp_links.txt exists
        try:
            with open(RTSP_LINKS_FILE, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            logging.error(f"❌ File not found: {RTSP_LINKS_FILE}")
            return False
        
        logging.info(f"📄 Reading from {RTSP_LINKS_FILE}...")
        
        # Parse and migrate each line
        cameras_added = 0
        apps_added = 0
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse line: RTSP_URL, Channel_Name, App1, App2, ...
            parts = [p.strip() for p in line.split(',')]
            
            if len(parts) < 3:
                logging.warning(f"⚠️  Line {line_num}: Not enough parts, skipping")
                continue
            
            rtsp_url = parts[0]
            channel_name = parts[1]
            app_names = parts[2:]
            
            # Parse RTSP URL
            parsed = parse_rtsp_url(rtsp_url)
            if not parsed:
                logging.warning(f"⚠️  Line {line_num}: Could not parse RTSP URL, skipping")
                continue
            
            # Generate channel_id
            channel_id = get_stable_channel_id(rtsp_url)
            
            # Check if camera already exists
            existing_camera = session.query(Camera).filter_by(
                channel_id=channel_id
            ).first()
            
            if existing_camera:
                logging.info(f"ℹ️  Camera '{channel_name}' already exists, updating...")
                camera = existing_camera
            else:
                # Create new camera
                camera = Camera(
                    restaurant_id=restaurant.id,
                    channel_number=parsed['channel'],
                    channel_name=channel_name,
                    subtype=parsed['subtype'],
                    rtsp_url=rtsp_url,
                    channel_id=channel_id,
                    is_active=True
                )
                session.add(camera)
                session.flush()  # Get the camera.id
                cameras_added += 1
                logging.info(f"✅ Added camera: {channel_name} (Channel {parsed['channel']}, Subtype {parsed['subtype']})")
            
            # Add camera apps
            for app_name in app_names:
                # Check if app already exists for this camera
                existing_app = session.query(CameraApp).filter_by(
                    camera_id=camera.id,
                    app_name=app_name
                ).first()
                
                if existing_app:
                    logging.info(f"   ℹ️  App '{app_name}' already linked to this camera")
                    continue
                
                # Create camera app
                app_config = get_app_config(app_name)
                camera_app = CameraApp(
                    camera_id=camera.id,
                    app_name=app_name,
                    is_active=True,
                    config=app_config
                )
                session.add(camera_app)
                apps_added += 1
                logging.info(f"   ✅ Linked app: {app_name}")
        
        # Commit all changes
        session.commit()
        
        # Display summary
        logging.info("")
        logging.info("=" * 70)
        logging.info("✅ Migration Completed Successfully!")
        logging.info("=" * 70)
        logging.info(f"📊 Summary:")
        logging.info(f"   • Cameras added: {cameras_added}")
        logging.info(f"   • Apps linked: {apps_added}")
        logging.info(f"   • Restaurant: {restaurant.restaurant_name}")
        logging.info("")
        
        # Display verification query
        total_cameras = session.query(Camera).filter_by(
            restaurant_id=restaurant.id
        ).count()
        total_apps = session.query(CameraApp).join(Camera).filter(
            Camera.restaurant_id == restaurant.id
        ).count()
        
        logging.info("📋 Verification:")
        logging.info(f"   • Total cameras in DB: {total_cameras}")
        logging.info(f"   • Total apps in DB: {total_apps}")
        logging.info("")
        
        # Show sample data
        logging.info("🔍 Sample Data:")
        cameras = session.query(Camera).filter_by(
            restaurant_id=restaurant.id
        ).limit(5).all()
        
        for cam in cameras:
            apps = session.query(CameraApp).filter_by(camera_id=cam.id).all()
            app_list = ', '.join([a.app_name for a in apps])
            logging.info(f"   • {cam.channel_name} (Channel {cam.channel_number})")
            logging.info(f"     Apps: {app_list}")
        
        logging.info("")
        logging.info("=" * 70)
        logging.info("⚠️  Next Steps:")
        logging.info("   1. Verify data: python3 verify_migration.py")
        logging.info("   2. Update edit-004.py to use database (Phase 2)")
        logging.info("   3. Keep rtsp_links.txt as backup")
        logging.info("=" * 70)
        
        return True
        
    except Exception as e:
        session.rollback()
        logging.error(f"❌ Migration failed: {e}", exc_info=True)
        return False
    finally:
        session.close()

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    success = migrate_rtsp_links()
    exit(0 if success else 1)
