"""
Script to add Idle People Violation camera configuration to the database
Run this script to add channel_id 5 (Front Office) for idle people monitoring
"""

import os
import sys
import hashlib
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text, ForeignKey, func
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import pytz

# Configuration
DATABASE_URL = "postgresql://postgres:Tneural01@127.0.0.1:5432/sakshi"
IST = pytz.timezone('Asia/Kolkata')
RTSP_URL = 'rtsp://admin:cctv%231234@182.65.205.121:554/cam/realmonitor?channel=5&subtype=1'

def get_stable_channel_id(link):
    """Generate stable channel ID from RTSP URL using MD5 hash"""
    return f"cam_{hashlib.md5(link.encode()).hexdigest()[:8]}"

# Database models
Base = declarative_base()

class Restaurant(Base):
    __tablename__ = "restaurants"
    id = Column(Integer, primary_key=True)
    restaurant_code = Column(String(50), nullable=False, unique=True)
    restaurant_name = Column(String(200), nullable=False)
    location = Column(String(200))
    dvr_ip = Column(String(50))
    dvr_username = Column(String(100))
    dvr_password = Column(String(100))
    telegram_chat_id = Column(String(50))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))
    updated_at = Column(DateTime, default=lambda: datetime.now(IST), onupdate=lambda: datetime.now(IST))

class Camera(Base):
    __tablename__ = "cameras"
    id = Column(Integer, primary_key=True)
    restaurant_id = Column(Integer, ForeignKey('restaurants.id'))
    channel_id = Column(String(50), unique=True, nullable=False)
    channel_name = Column(String(255), nullable=False)
    channel_number = Column(Integer)
    rtsp_url = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))
    updated_at = Column(DateTime, default=lambda: datetime.now(IST), onupdate=lambda: datetime.now(IST))

class CameraApp(Base):
    __tablename__ = "camera_apps"
    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey('cameras.id'))
    app_name = Column(String(50), nullable=False)
    is_active = Column(Boolean, default=True)
    config = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))


def add_idle_people_camera():
    """Add camera configuration for idle people violation monitoring"""
    
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Generate channel_id from RTSP URL
    channel_id = get_stable_channel_id(RTSP_URL)
    print(f"Generated channel_id: {channel_id}")
    
    with SessionLocal() as db:
        try:
            # Use restaurant_id = 2
            restaurant = db.query(Restaurant).filter_by(id=2).first()
            if not restaurant:
                print(f"❌ Restaurant with ID=2 not found in database!")
                print("Available restaurants:")
                all_restaurants = db.query(Restaurant).all()
                for r in all_restaurants:
                    print(f"  ID: {r.id}, Code: {r.restaurant_code}, Name: {r.restaurant_name}")
                return
            
            print(f"✅ Using restaurant: {restaurant.restaurant_name} (ID: {restaurant.id})")
            
            # Check if camera already exists
            existing_camera = db.query(Camera).filter_by(channel_id=channel_id).first()
            
            if existing_camera:
                print(f"⚠️  Camera with channel_id='{channel_id}' already exists:")
                print(f"   Name: {existing_camera.channel_name}")
                print(f"   RTSP: {existing_camera.rtsp_url}")
                print(f"   Active: {existing_camera.is_active}")
                
                # Update configuration if needed
                if existing_camera.rtsp_url != RTSP_URL or existing_camera.channel_name != 'Front Office':
                    print(f"\n📝 Updating camera configuration...")
                    existing_camera.rtsp_url = RTSP_URL
                    existing_camera.channel_name = 'Front Office'
                    existing_camera.restaurant_id = restaurant.id
                    existing_camera.is_active = True
                    db.commit()
                    print(f"✅ Updated camera configuration")
                
                camera_id = existing_camera.id
            else:
                # Create new camera
                print("Creating new camera for Idle People Violation monitoring...")
                new_camera = Camera(
                    restaurant_id=restaurant.id,
                    channel_id=channel_id,
                    channel_name='Front Office',
                    channel_number=5,
                    rtsp_url=RTSP_URL,
                    is_active=True
                )
                db.add(new_camera)
                db.commit()
                db.refresh(new_camera)
                camera_id = new_camera.id
                print(f"✅ Created camera: {new_camera.channel_name} (ID: {camera_id})")
            
            # Check if IdlePeopleViolation app is configured for this camera
            existing_app = db.query(CameraApp).filter_by(
                camera_id=camera_id,
                app_name='IdlePeopleViolation'
            ).first()
            
            if existing_app:
                print(f"\n⚠️  IdlePeopleViolation app already configured for this camera")
                print(f"   Active: {existing_app.is_active}")
                if not existing_app.is_active:
                    existing_app.is_active = True
                    db.commit()
                    print(f"✅ Activated IdlePeopleViolation app")
            else:
                # Add IdlePeopleViolation app
                print("\nConfiguring IdlePeopleViolation app for this camera...")
                new_app = CameraApp(
                    camera_id=camera_id,
                    app_name='IdlePeopleViolation',
                    is_active=True,
                    config='{"confidence": 0.3, "idle_frame_threshold": 15, "model_path": "models/yolo11n.pt"}'
                )
                db.add(new_app)
                db.commit()
                print(f"✅ Configured IdlePeopleViolation app")
            
            print("\n" + "="*60)
            print("✅ Setup Complete!")
            print("="*60)
            print(f"Restaurant ID: {restaurant.id}")
            print(f"Restaurant Name: {restaurant.restaurant_name}")
            print(f"Channel ID: {channel_id}")
            print(f"Channel Name: Front Office")
            print(f"RTSP URL: {RTSP_URL}")
            print(f"App: IdlePeopleViolation")
            print(f"Status: Active")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            db.rollback()
            raise


if __name__ == "__main__":
    print("Adding Idle People Violation camera configuration to database...\n")
    add_idle_people_camera()
