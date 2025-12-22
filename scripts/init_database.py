#!/usr/bin/env python3
"""
Database Initialization Script
Creates all required tables for the Sakshi AI application
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Text, Float, UniqueConstraint, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Tneural01@127.0.0.1:5432/sakshi")

Base = declarative_base()

# Define all tables from the application

# 0. Restaurants table (new in Tea-toast-new-v1)
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
    created_at = Column(DateTime, server_default='now()')
    updated_at = Column(DateTime, server_default='now()')

# 0a. Cameras table (new in Tea-toast-new-v1)
class Camera(Base):
    __tablename__ = "cameras"
    id = Column(Integer, primary_key=True)
    restaurant_id = Column(Integer, ForeignKey('restaurants.id'))
    channel_id = Column(String(50), unique=True, nullable=False)
    channel_name = Column(String(255), nullable=False)
    channel_number = Column(Integer)
    rtsp_url = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default='now()')
    updated_at = Column(DateTime, server_default='now()')

# 0b. Camera Apps table (new in Tea-toast-new-v1)
class CameraApp(Base):
    __tablename__ = "camera_apps"
    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey('cameras.id'))
    app_name = Column(String(50), nullable=False)
    is_active = Column(Boolean, default=True)
    config = Column(JSONB)
    created_at = Column(DateTime, server_default='now()')

# 1. PetPooja Webhook Events (from main.py and fastapi_app.py)
class PetpoojaWebhookEvent(Base):
    __tablename__ = "petpooja_webhook_events"
    id = Column(Integer, primary_key=True)
    content = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default='now()')

# 2. Detections (from edit-004.py)
class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    app_name = Column(String, index=True)
    channel_id = Column(String, index=True)
    timestamp = Column(DateTime)
    message = Column(Text)
    media_path = Column(String)
    __table_args__ = (UniqueConstraint('media_path', name='_media_path_uc'),)

# 3. Daily Footfall (from edit-004.py)
class DailyFootfall(Base):
    __tablename__ = "daily_footfall"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    report_date = Column(Date, index=True)
    in_count = Column(Integer, default=0)
    out_count = Column(Integer, default=0)

# 4. Hourly Footfall (from edit-004.py)
class HourlyFootfall(Base):
    __tablename__ = "hourly_footfall"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    report_date = Column(Date, index=True)
    hour = Column(Integer, index=True)
    in_count = Column(Integer, default=0)
    out_count = Column(Integer, default=0)
    __table_args__ = (UniqueConstraint('channel_id', 'report_date', 'hour', name='_channel_date_hour_uc'),)

# 5. Queue Logs (from edit-004.py)
class QueueLog(Base):
    __tablename__ = "queue_logs"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    queue_count = Column(Integer)

# 6. ROI Configs (from edit-004.py)
class RoiConfig(Base):
    __tablename__ = "roi_configs"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    app_name = Column(String, index=True)
    roi_points = Column(Text)  # Storing as JSON string
    restaurant_id = Column(Integer, ForeignKey('restaurants.id'), nullable=True)
    __table_args__ = (UniqueConstraint('channel_id', 'app_name', name='_roi_uc'),)

# 7. Kitchen Violations (from kitchen_compliance_monitor.py and edit-004.py)
class KitchenViolation(Base):
    __tablename__ = "kitchen_violations"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    channel_name = Column(String)
    timestamp = Column(DateTime)
    violation_type = Column(String)
    details = Column(String)
    media_path = Column(String)
    __table_args__ = (UniqueConstraint('media_path', name='_kitchen_media_path_uc'),)

# 8. Occupancy Logs (from occupancy_monitor_processor.py and edit-004.py)
class OccupancyLog(Base):
    __tablename__ = "occupancy_logs"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    timestamp = Column(DateTime)
    time_slot = Column(String)
    day_of_week = Column(String)
    live_count = Column(Integer)
    required_count = Column(Integer)
    status = Column(String)  # 'OK', 'BELOW_REQUIREMENT', 'NO_SCHEDULE', 'PAUSED'

# 9. Occupancy Schedules (from occupancy_monitor_processor.py and edit-004.py)
class OccupancySchedule(Base):
    __tablename__ = "occupancy_schedules"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    time_slot = Column(String)  # e.g., "9:00"
    day_of_week = Column(String)  # e.g., "Monday"
    required_count = Column(Integer)
    __table_args__ = (UniqueConstraint('channel_id', 'time_slot', 'day_of_week', name='_occupancy_schedule_uc'),)


def init_database():
    """Initialize all database tables"""
    try:
        logging.info(f"Connecting to database: {DATABASE_URL}")
        engine = create_engine(DATABASE_URL)
        
        logging.info("Creating all tables...")
        Base.metadata.create_all(bind=engine)
        
        logging.info("✅ Database initialization complete!")
        logging.info("\nCreated tables:")
        for table in Base.metadata.sorted_tables:
            logging.info(f"  - {table.name}")
        
        # Test connection
        with engine.connect() as conn:
            logging.info("\n✅ Database connection test successful!")
        
        return True
    
    except Exception as e:
        logging.error(f"❌ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = init_database()
    sys.exit(0 if success else 1)
