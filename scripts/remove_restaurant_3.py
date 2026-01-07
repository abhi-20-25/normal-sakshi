"""
Script to remove restaurant ID=3 and all associated data from the database
This will delete:
- Camera apps associated with restaurant 3's cameras
- Cameras associated with restaurant 3
- The restaurant record itself
"""

import os
import sys
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import pytz

# Configuration
DATABASE_URL = "postgresql://postgres:Tneural01@127.0.0.1:5432/sakshi"
IST = pytz.timezone('Asia/Kolkata')

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


def remove_restaurant_3():
    """Remove restaurant ID=3 and all associated data"""
    
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    with SessionLocal() as db:
        try:
            # Find restaurant ID=3
            restaurant = db.query(Restaurant).filter_by(id=3).first()
            
            if not restaurant:
                print("❌ Restaurant ID=3 not found in database")
                return
            
            print(f"\n{'='*60}")
            print(f"Found Restaurant ID=3:")
            print(f"  Code: {restaurant.restaurant_code}")
            print(f"  Name: {restaurant.restaurant_name}")
            print(f"  Location: {restaurant.location}")
            print(f"{'='*60}\n")
            
            # Find all cameras for this restaurant
            cameras = db.query(Camera).filter_by(restaurant_id=3).all()
            
            print(f"Found {len(cameras)} camera(s) associated with Restaurant ID=3:")
            for cam in cameras:
                print(f"  📹 {cam.channel_name} (channel_id: {cam.channel_id}, camera_id: {cam.id})")
            
            # Delete camera apps for each camera
            total_apps_deleted = 0
            for cam in cameras:
                camera_apps = db.query(CameraApp).filter_by(camera_id=cam.id).all()
                for app in camera_apps:
                    print(f"  🗑️  Deleting camera_app: {app.app_name} for camera {cam.channel_name}")
                    db.delete(app)
                    total_apps_deleted += 1
            
            # Commit camera apps deletion first
            db.commit()
            print(f"\n✅ Deleted {total_apps_deleted} camera app(s)")
            
            # Delete cameras
            cameras_deleted = 0
            for cam in cameras:
                print(f"  🗑️  Deleting camera: {cam.channel_name} (channel_id: {cam.channel_id})")
                db.delete(cam)
                cameras_deleted += 1
            
            # Commit cameras deletion
            db.commit()
            print(f"✅ Deleted {cameras_deleted} camera(s)")
            
            # Delete restaurant
            print(f"\n🗑️  Deleting restaurant: {restaurant.restaurant_name}")
            db.delete(restaurant)
            
            # Commit restaurant deletion
            db.commit()
            
            print(f"\n{'='*60}")
            print("✅ Successfully removed Restaurant ID=3 and all associated data")
            print(f"{'='*60}\n")
            
            # Show remaining restaurants
            print("Remaining restaurants in database:")
            remaining = db.query(Restaurant).all()
            for r in remaining:
                print(f"  ID: {r.id}, Code: {r.restaurant_code}, Name: {r.restaurant_name}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            db.rollback()
            raise


if __name__ == "__main__":
    print("Removing Restaurant ID=3 from database...\n")
    
    # Confirm deletion
    response = input("⚠️  This will delete Restaurant ID=3 and all associated cameras and apps. Continue? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        remove_restaurant_3()
    else:
        print("❌ Operation cancelled")
