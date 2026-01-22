# edit-004.py (People Counter, Queue Detection, and Generic Detection)

import cv2
import torch
from ultralytics import YOLO
import threading
import time
import json
from datetime import datetime, date, timedelta, time as dt_time
from collections import defaultdict
import os
import requests
import imageio
from flask import Flask, Response, render_template, jsonify, url_for, request, stream_with_context, session, redirect
from flask_socketio import SocketIO
from functools import wraps
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Text, text, UniqueConstraint, Boolean, ForeignKey, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError
from urllib.parse import urlparse, urlunparse
import logging
import pytz
import numpy as np
import atexit
import io
import csv
import hashlib
import subprocess
from apscheduler.schedulers.background import BackgroundScheduler
from shapely.geometry import Point, Polygon
import pandas as pd
from queue import Queue, Empty

# --- CUDA/Backend Tuning ---
# Enable CUDA auto-detection
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    logging.info("✅ CUDA ENABLED - Using GPU for processing")
else:
    logging.info("⚠️  CUDA not available - Running in CPU mode")

# --- Frame Downscale Settings ---
# Reduce resolution early in the pipeline to speed up processing/streaming
# Set to None to preserve original camera resolution
TARGET_WIDTH = 640
TARGET_HEIGHT = 360

# --- Module Imports ---
from kitchen_compliance_monitor import KitchenComplianceProcessor
from idle_people_violation import IdlePeopleViolationProcessor
from petpooja_integration import (
    create_petpooja_services,
    PetPoojaDatabase,
    get_petpooja_restaurant_id
)
from queue_monitor import QueueMonitorProcessor
from occupancy_monitor_processor import run_occupancy_monitor, get_occupancy_tables, OccupancyMonitorProcessor
from people_counter import PeopleCounterProcessor

# --- Basic Logging Setup ---
import sys
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,  # Force to stdout instead of stderr
    force=True  # Override any existing config
)
# Force flush after each log
for handler in logging.root.handlers:
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    if hasattr(handler.stream, 'reconfigure'):
        handler.stream.reconfigure(line_buffering=True)
        
logging.getLogger('ultralytics').setLevel(logging.WARNING)
logging.getLogger('apscheduler').setLevel(logging.WARNING)

# --- Master Configuration ---
IST = pytz.timezone('Asia/Kolkata')
DATABASE_URL = "postgresql://postgres:Tneural01@127.0.0.1:5432/sakshi" 

RTSP_LINKS_FILE = 'data/rtsp_links.txt'

# Initialize PetPooja services (will be used in Flask routes)
pp_sales_analytics, pp_conversion_analytics, pp_time_based_menu, pp_promotion_effectiveness, pp_staffing_recommendations = create_petpooja_services()
STATIC_FOLDER = 'static'
DETECTIONS_SUBFOLDER = 'detections'
TELEGRAM_BOT_TOKEN = "7843300957:AAGVv866cPiDPVD0Wrk_wwEEHDSD64Pgaqs"
TELEGRAM_CHAT_ID = "-4835836048"

# --- Authentication Configuration ---
LOGIN_USERNAME = "user"
LOGIN_PASSWORD = "Tneural123"
os.makedirs(os.path.join(STATIC_FOLDER, DETECTIONS_SUBFOLDER), exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, DETECTIONS_SUBFOLDER, 'shutter_videos'), exist_ok=True)

#server

# --- App Task Configuration ---
APP_TASKS_CONFIG = {
    'Generic': {'model_path': 'models/kitchen_violation_18_01_2026.pt', 'target_class_id': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'confidence': 0.3, 'is_gif': False},
    'PeopleCounter': {'model_path': 'models/yolo11n.pt' , 'confidence': 0.15},
    'QueueMonitor': {'model_path': 'models/yolo11n.pt' , 'confidence': 0.15},
    'KitchenCompliance': {'model_path': 'models/kitchen_violation_18_01_2026.pt', 'confidence': 0.3},  # Kitchen violation model (person detection via yolo11n.pt)
    'OccupancyMonitor': {'model_path': 'models/yolo11n.pt', 'confidence': 0.15},
    'IdlePeopleViolation': {'model_path': 'models/yolo11n.pt', 'confidence': 0.3}
}

# --- YOLO tracking helper (CPU-only mode) ---
def safe_track_persons(model, frame, conf=0.25, iou=0.5, processor_name=None):
    # Validate frame before processing
    if frame is None:
        logging.warning("safe_track_persons: frame is None, returning empty result")
        return []
    
    # Check if frame is a numpy array with valid shape
    if not hasattr(frame, 'shape'):
        logging.warning(f"safe_track_persons: frame has no shape attribute (type: {type(frame)}), returning empty result")
        return []
    
    # Validate frame dimensions (must have height, width, and at least 1 channel)
    if len(frame.shape) < 2:
        logging.warning(f"safe_track_persons: invalid frame shape {frame.shape}, returning empty result")
        return []
    
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        logging.warning(f"safe_track_persons: invalid frame dimensions (h={h}, w={w}), returning empty result")
        return []
    
    # Minimum size check (frames should be at least 32x32 pixels)
    if h < 32 or w < 32:
        logging.warning(f"safe_track_persons: frame too small (h={h}, w={w}), returning empty result")
        return []
    
    # FORCE CPU MODE - Disable all CUDA usage
    proc_name = processor_name if processor_name else 'unknown'
    device_to_use = 'cpu'
    use_half = False  # No half precision on CPU
    
    try:
        with torch.inference_mode():
            result = model.track(
                frame,
                persist=True,
                classes=[0],
                conf=conf,
                iou=iou,
                verbose=False,
                device=device_to_use,
                half=use_half
                # Removed tracker='bytetrack.yaml' - use default tracker to fix tracking ID generation
            )
            return result
    except RuntimeError as e:
        # CPU-only mode - no CUDA error handling needed
        logging.warning(f"Runtime error in safe_track_persons: {e}")
        logging.warning("Attempting fallback to predict() instead of track()")
        try:
            fallback_result = model.predict(
                frame,
                classes=[0],
                conf=conf,
                iou=iou,
                verbose=False,
                device='cpu',
                half=False
            )
            return fallback_result
        except Exception as fallback_e:
            logging.error(f"Fallback predict() also failed: {fallback_e}")
            return []
    except Exception as e:
        logging.error(f"Unexpected error in safe_track_persons: {e}")
        return []


# --- QUEUE MONITOR CONFIGURATION ---
# Queue monitor configuration moved to queue_monitor.py module

# --- Flask and SocketIO Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-very-secret-key-for-sakshi-ai'
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True
# Force threading async mode to avoid eventlet/gevent interfering with streaming responses
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- Global State Management ---
stream_processors = {}

# Shared RTSP frame provider per camera to minimize latency and duplicate decoding
# Shared RTSP frame provider using a thread-safe queue
class FrameHub(threading.Thread):
    def __init__(self, rtsp_url, name):
        super().__init__(name=f"FrameHub-{name}", daemon=True)
        self.rtsp_url = rtsp_url
        
        # A queue of size 1 is the perfect "latest frame" buffer
        self.frame_queue = Queue(maxsize=1) 
        self.is_running = True
        logging.info(f"FrameHub {self.name} initialized for {self.rtsp_url}")

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            logging.error(f"FrameHub {self.name} could not open stream: {self.rtsp_url}")
            return
            
        # We only set properties that are relevant.
        # BUFFERSIZE is a good hint to OpenCV.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Best effort request for smaller frames (some RTSP servers ignore this)
        try:
            if TARGET_WIDTH:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
            if TARGET_HEIGHT:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
        except Exception:
            pass

        while self.is_running:
            # Use cap.read() which combines grab() and retrieve()
            ret, frame = cap.read()

            if not ret:
                logging.warning(f"FrameHub {self.name} disconnected. Reconnecting...")
                cap.release()
                time.sleep(5)  # Wait 5 seconds before retrying
                cap = cv2.VideoCapture(self.rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            
            # Validate frame before processing
            if frame is None or not hasattr(frame, 'shape') or len(frame.shape) < 2:
                logging.warning(f"FrameHub {self.name}: Invalid frame received, skipping")
                continue
            
            h, w = frame.shape[:2]
            if h <= 0 or w <= 0:
                logging.warning(f"FrameHub {self.name}: Invalid frame dimensions (h={h}, w={w}), skipping")
                continue
            
            # Downscale the frame to speed up processing and streaming
            try:
                if TARGET_WIDTH and TARGET_HEIGHT:
                    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
                elif TARGET_WIDTH:
                    h, w = frame.shape[:2]
                    new_h = int(h * (TARGET_WIDTH / float(w)))
                    frame = cv2.resize(frame, (TARGET_WIDTH, new_h), interpolation=cv2.INTER_AREA)
            except Exception as e:
                logging.warning(f"FrameHub {self.name}: Error resizing frame: {e}")
                continue
            
            # Final validation after resize
            if frame is None or not hasattr(frame, 'shape') or len(frame.shape) < 2:
                logging.warning(f"FrameHub {self.name}: Frame became invalid after resize, skipping")
                continue
            
            h, w = frame.shape[:2]
            if h < 32 or w < 32:
                logging.warning(f"FrameHub {self.name}: Frame too small after resize (h={h}, w={w}), skipping")
                continue
            
            # --- This is the key logic ---
            # If the queue is full (i.e., it has 1 frame), 
            # we first clear it to make space for the new frame.
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()  # Discard the old frame
                except Empty:
                    pass # Should not happen, but safe to include
            
            # Put the new, latest frame into the queue (non-blocking)
            try:
                self.frame_queue.put_nowait(frame)
            except:
                # Queue full (shouldn't happen as we clear it above, but handle it anyway)
                pass
            # --- End of key logic ---

        cap.release()
        logging.info(f"FrameHub {self.name} stopped.")

    def get_latest(self):
        """Gets the latest frame from the queue without blocking."""
        try:
            # Get frame from queue. copy() is good practice
            # to prevent the processing thread from locking the frame.
            return self.frame_queue.get_nowait().copy()
        except Empty:
            # If the queue is empty, just return None
            return None

    def stop(self):
        logging.info(f"Stopping FrameHub {self.name}...")
        self.is_running = False

# Simple passthrough processor: no detection, just relays frames from FrameHub
## PassthroughProcessor removed; restoring detection processors

# --- Database Setup ---
Base = declarative_base()
db_connected = False
engine = None
SessionLocal = None

class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    app_name = Column(String, index=True)
    channel_id = Column(String, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(IST))
    message = Column(Text)
    media_path = Column(String)
    __table_args__ = (UniqueConstraint('media_path', name='_media_path_uc'),)

class DailyFootfall(Base):
    __tablename__ = "daily_footfall"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    report_date = Column(Date, index=True)
    in_count = Column(Integer, default=0)
    out_count = Column(Integer, default=0)

class HourlyFootfall(Base):
    __tablename__ = "hourly_footfall"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    report_date = Column(Date, index=True)
    hour = Column(Integer, index=True)
    in_count = Column(Integer, default=0)
    out_count = Column(Integer, default=0)
    __table_args__ = (UniqueConstraint('channel_id', 'report_date', 'hour', name='_channel_date_hour_uc'),)

class QueueLog(Base):
    __tablename__ = "queue_logs"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(IST), index=True)
    queue_count = Column(Integer)



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
    petpooja_rest_id = Column(String(50))  # PetPooja restaurant ID for API filtering

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
    config = Column(JSONB)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))

class RoiConfig(Base):
    __tablename__ = "roi_configs"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    app_name = Column(String, index=True)
    roi_points = Column(Text) # Storing as JSON string
    restaurant_id = Column(Integer, ForeignKey('restaurants.id'))
    __table_args__ = (UniqueConstraint('channel_id', 'app_name', name='_roi_uc'),)

class KitchenViolation(Base):
    __tablename__ = "kitchen_violations"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    channel_name = Column(String)
    timestamp = Column(DateTime, default=lambda: datetime.now(IST))
    violation_type = Column(String)
    details = Column(String)
    media_path = Column(String)
    __table_args__ = (UniqueConstraint('media_path', name='_kitchen_media_path_uc'),)

class OccupancyLog(Base):
    __tablename__ = "occupancy_logs"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(IST))
    time_slot = Column(String)
    day_of_week = Column(String)
    live_count = Column(Integer)
    required_count = Column(Integer)
    status = Column(String)  # 'OK', 'BELOW_REQUIREMENT', 'NO_SCHEDULE', 'PAUSED'
    
class OccupancySchedule(Base):
    __tablename__ = "occupancy_schedules"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    time_slot = Column(String)  # e.g., "9:00"
    day_of_week = Column(String)  # e.g., "Monday"
    required_count = Column(Integer)
    __table_args__ = (UniqueConstraint('channel_id', 'time_slot', 'day_of_week', name='_occupancy_schedule_uc'),)

def get_stable_channel_id(link):
    return f"cam_{hashlib.md5(link.encode()).hexdigest()[:10]}"

def login_required(f):
    """Decorator to require login for protected routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

def initialize_database():
    global db_connected, engine, SessionLocal, OccupancyLog, OccupancySchedule
    try:
        logging.info("Initializing database connection...")
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        logging.info("Creating occupancy table classes...")
        # Initialize occupancy tables BEFORE creating all tables
        OccupancyLog, OccupancySchedule = get_occupancy_tables(Base)
        
        logging.info("Creating all database tables...")
        # Now create all tables including occupancy tables
        Base.metadata.create_all(bind=engine)
        
        db_connected = True
        logging.info("✅ Database connection successful.")
        return True
    except OperationalError as e:
        logging.error(f"❌ Database connection failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False
    except Exception as e:
        logging.error(f"❌ Unexpected error during DB init: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def send_telegram_notification(message):
    if not TELEGRAM_BOT_TOKEN or "YOUR_TELEGRAM" in TELEGRAM_BOT_TOKEN:
        logging.warning("Telegram token not configured.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        logging.error(f"Error sending Telegram notification: {e}")

def handle_detection(app_name, channel_id, frames, message, is_gif=False):
    timestamp = datetime.now(IST)
    ts_string = timestamp.strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{app_name}_{channel_id}_{ts_string}.{'gif' if is_gif else 'jpg'}"
    media_path = os.path.join(DETECTIONS_SUBFOLDER, filename)
    full_path = os.path.join(STATIC_FOLDER, media_path)
    try:
        if is_gif and isinstance(frames, list) and len(frames) > 1:
            rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
            imageio.mimsave(full_path, rgb_frames, fps=10, duration=0.1)
        else:
            frame_to_save = frames[0] if isinstance(frames, list) else frames
            cv2.imwrite(full_path, frame_to_save)
    except Exception as e:
        logging.error(f"Failed to save media file '{full_path}': {e}")
        return
    if db_connected:
        with SessionLocal() as db:
            try:
                exists = db.query(Detection).filter(Detection.media_path == media_path).first()
                if not exists:
                    db.add(Detection(app_name=app_name, channel_id=channel_id, timestamp=timestamp, message=message, media_path=media_path))
                    db.commit()
            except Exception as e:
                logging.error(f"Failed to save detection to DB: {e}")
                db.rollback()
    with app.test_request_context():
        media_url = url_for('static', filename=media_path)
    socketio.emit('new_detection', {'app_name': app_name, 'channel_id': channel_id, 'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"), 'message': message, 'media_url': media_url})
    # Return the relative media_path so callers (e.g., KitchenCompliance) can persist it
    return media_path

class MultiModelProcessor(threading.Thread):
    def __init__(self, rtsp_url, channel_id, channel_name, tasks, detection_callback):
        super().__init__()
        self.rtsp_url, self.channel_id, self.channel_name, self.tasks, self.detection_callback = rtsp_url, channel_id, channel_name, tasks, detection_callback
        self.is_running = True
        self.last_detection_times = {task['app_name']: 0 for task in self.tasks}
        self.cooldown, self.gif_duration_seconds, self.fps = 120, 3, 10
        self.expected_frame_shape = None  # Track expected frame dimensions
        self.consecutive_invalid_frames = 0  # Track consecutive invalid frames
        self.consecutive_errors = 0  # Track consecutive CUDA errors
        self.max_consecutive_errors = 10
        self.latest_frame = None  # For video streaming
        self.lock = threading.Lock()  # Thread-safe frame access
        
        # ROI filtering for Generic app
        self.roi_polygon = None
        self._load_roi()

    def stop(self): self.is_running = False
    def shutdown(self):
        logging.info(f"Shutting down MultiModel for {self.channel_name} ({self.channel_id})")
        self.is_running = False
    
    def _load_roi(self):
        """Load ROI configuration from database for Generic app"""
        try:
            from sqlalchemy import text, create_engine
            from sqlalchemy.orm import sessionmaker
            from shapely.geometry import Polygon
            import json
            
            # Get database connection from environment
            database_url = os.environ.get('DATABASE_URL')
            if not database_url:
                logging.info(f"No DATABASE_URL found for Generic {self.channel_name} - skipping ROI load")
                return
            
            engine = create_engine(database_url, pool_pre_ping=True)
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            
            with SessionLocal() as db:
                query = text("""
                    SELECT roi_points FROM roi_configs 
                    WHERE channel_id = :channel_id AND app_name = 'Generic'
                """)
                result = db.execute(query, {"channel_id": self.channel_id}).fetchone()
                
                if result and result[0]:
                    roi_data = result[0] if isinstance(result[0], dict) else json.loads(result[0])
                    points = roi_data.get('points', [])
                    
                    if points and len(points) >= 3:
                        self.roi_polygon = Polygon(points)
                        logging.info(f"✅ Generic ROI loaded for {self.channel_name}: {len(points)} points")
                    else:
                        logging.info(f"No valid ROI configured for Generic {self.channel_name} - monitoring entire frame")
                else:
                    logging.info(f"No ROI configured for Generic {self.channel_name} - monitoring entire frame")
        except Exception as e:
            logging.error(f"Error loading Generic ROI for {self.channel_name}: {e}")
            self.roi_polygon = None

    def _is_in_roi(self, x1, y1, x2, y2, frame_width, frame_height, overlap_threshold=0.5):
        """
        Check if at least 50% (or specified threshold) of bounding box is inside ROI.
        
        Args:
            x1, y1, x2, y2: Bounding box pixel coordinates
            frame_width, frame_height: Frame dimensions for normalization
            overlap_threshold: Minimum percentage of bbox that must be in ROI (default: 0.5 = 50%)
        
        Returns:
            True if bbox overlap with ROI >= threshold, False otherwise
        """
        if self.roi_polygon is None:
            return True  # No ROI means monitor entire frame
        
        try:
            from shapely.geometry import box as shapely_box
            
            # Normalize bbox coordinates to 0-1 range (ROI is stored normalized)
            norm_x1 = x1 / frame_width
            norm_y1 = y1 / frame_height
            norm_x2 = x2 / frame_width
            norm_y2 = y2 / frame_height
            
            # Create a polygon from the normalized bounding box coordinates
            bbox_polygon = shapely_box(norm_x1, norm_y1, norm_x2, norm_y2)
            
            # Calculate intersection area
            intersection = self.roi_polygon.intersection(bbox_polygon)
            bbox_area = bbox_polygon.area
            
            if bbox_area == 0:
                return False
            
            # Calculate overlap percentage
            overlap_percentage = intersection.area / bbox_area
            
            return overlap_percentage >= overlap_threshold
        except Exception as e:
            logging.error(f"Error checking ROI for Generic {self.channel_name}: {e}")
            return True  # On error, allow detection
    
    def apply_smart_validation(self, detections, model):
        """
        Apply smart validation logic using complementary pairs:
        Compare confidence scores and keep the higher confidence detection.
        NEW MODEL CLASS IDs:
        - Cap_present (0) vs Without_cap (1)
        - With_apron (2) vs Without_apron (3)
        - With_gloves (4) vs Without_gloves (5)
        - Uniform (6) vs Without_uniform (7)
        - Using_phone (8) always triggers as violation
        """
        if not detections:
            return detections
        
        # Extract confidence scores by class
        class_confidences = {}
        for det in detections:
            class_id = int(det['class_id'])
            conf = det['confidence']
            if class_id not in class_confidences:
                class_confidences[class_id] = []
            class_confidences[class_id].append(conf)
        
        # Get max confidence for each class
        max_conf_by_class = {cls_id: max(confs) for cls_id, confs in class_confidences.items()}
        
        # Apply smart logic - compare confidences
        filtered_detections = []
        for det in detections:
            class_id = int(det['class_id'])
            conf = det['confidence']
            should_keep = True
            
            # Without_cap (1) vs Cap_present (0) - keep higher confidence
            if class_id == 1 and 0 in max_conf_by_class:
                if max_conf_by_class[0] > conf:  # Cap_present has higher confidence
                    should_keep = False
            elif class_id == 0 and 1 in max_conf_by_class:
                if max_conf_by_class[1] > conf:  # Without_cap has higher confidence
                    should_keep = False
            
            # Without_apron (3) vs With_apron (2) - keep higher confidence
            elif class_id == 3 and 2 in max_conf_by_class:
                if max_conf_by_class[2] > conf:  # With_apron has higher confidence
                    should_keep = False
            elif class_id == 2 and 3 in max_conf_by_class:
                if max_conf_by_class[3] > conf:  # Without_apron has higher confidence
                    should_keep = False
            
            # Without_gloves (5) vs With_gloves (4) - keep higher confidence
            elif class_id == 5 and 4 in max_conf_by_class:
                if max_conf_by_class[4] > conf:  # With_gloves has higher confidence
                    should_keep = False
            elif class_id == 4 and 5 in max_conf_by_class:
                if max_conf_by_class[5] > conf:  # Without_gloves has higher confidence
                    should_keep = False
            
            # Without_uniform (7) vs Uniform (6) - keep higher confidence
            elif class_id == 7 and 6 in max_conf_by_class:
                if max_conf_by_class[6] > conf:  # Uniform has higher confidence
                    should_keep = False
            elif class_id == 6 and 7 in max_conf_by_class:
                if max_conf_by_class[7] > conf:  # Without_uniform has higher confidence
                    should_keep = False
            
            if should_keep:
                filtered_detections.append(det)
        
        return filtered_detections
    
    def get_frame(self):
        """Get the latest frame for video streaming"""
        with self.lock:
            if self.latest_frame is None:
                placeholder = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(placeholder, 'Connecting...', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (201, 209, 217), 2)
                _, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()
            _, jpeg = cv2.imencode('.jpg', self.latest_frame)
            return jpeg.tobytes()

    def _validate_frame(self, frame):
        """Validate frame before processing to prevent CUDA errors"""
        if frame is None:
            return False
        
        # Check if frame is a numpy array with valid shape
        if not hasattr(frame, 'shape'):
            self.consecutive_invalid_frames += 1
            if self.consecutive_invalid_frames % 100 == 0:
                logging.warning(f"MultiModel {self.channel_name}: Invalid frame type (no shape attribute)")
            return False
        
        # Validate frame dimensions
        if len(frame.shape) < 2:
            self.consecutive_invalid_frames += 1
            if self.consecutive_invalid_frames % 100 == 0:
                logging.warning(f"MultiModel {self.channel_name}: Invalid frame shape {frame.shape}")
            return False
        
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            self.consecutive_invalid_frames += 1
            if self.consecutive_invalid_frames % 100 == 0:
                logging.warning(f"MultiModel {self.channel_name}: Invalid frame dimensions (h={h}, w={w})")
            return False
        
        # Minimum size check
        if h < 32 or w < 32:
            self.consecutive_invalid_frames += 1
            if self.consecutive_invalid_frames % 100 == 0:
                logging.warning(f"MultiModel {self.channel_name}: Frame too small (h={h}, w={w})")
            return False
        
        # Check dimension consistency - detect if frame source changed
        current_shape = (h, w)
        if self.expected_frame_shape is None:
            self.expected_frame_shape = current_shape
            logging.info(f"MultiModel {self.channel_name}: Expected frame shape set to {current_shape}")
        elif self.expected_frame_shape != current_shape:
            # Frame dimensions changed - might be from different source
            logging.warning(f"MultiModel {self.channel_name}: Frame dimension mismatch! Expected {self.expected_frame_shape}, got {current_shape}. "
                         f"This might indicate feed overlap/mixing. Skipping frame.")
            self.consecutive_invalid_frames += 1
            return False
        
        # Reset invalid frame counter on valid frame
        if self.consecutive_invalid_frames > 0:
            self.consecutive_invalid_frames = 0
        
        return True

    def run(self):
        frame_count = 0
        while self.is_running:
            frame = getattr(self, 'frame_hub', None).get_latest() if hasattr(self, 'frame_hub') else None
            if frame is None:
                continue

            # Validate frame before processing
            if not self._validate_frame(frame):
                continue
            
            frame_count += 1
            # Log every 30 frames to show detection is running continuously
            if frame_count % 30 == 0:
                logging.info(f"🎬 MultiModel {self.channel_name}: Processing frame {frame_count}")

            # Store frame for video streaming (even if no detection)
            with self.lock:
                self.latest_frame = frame.copy()

            current_time = time.time()

            for task in self.tasks:
                app_name = task['app_name']
                if app_name in ['PeopleCounter', 'QueueMonitor']: continue

                # Run detection every frame (no cooldown for detection itself)
                model_args = {'conf': task['confidence'], 'verbose': False}
                if task.get('target_class_id') is not None:
                    model_args['classes'] = task['target_class_id']

                # CPU-only mode for non-Generic apps, CUDA for Generic
                try:
                    # Use CUDA for Generic app if available, CPU for others
                    device_to_use = DEVICE if app_name == 'Generic' else 'cpu'
                    with torch.inference_mode():
                        results = task['model'](
                            frame,
                            device=device_to_use,
                            half=False,
                            **model_args
                        )
                    self.consecutive_errors = 0  # Reset on success
                    
                except RuntimeError as e:
                    error_msg = str(e)
                    self.consecutive_errors += 1
                    logging.error(f"Runtime error in MultiModel {self.channel_name} for {app_name}: {e}. Frame shape: {frame.shape}. Error count: {self.consecutive_errors}")
                    
                    # If too many errors, log and continue
                    if self.consecutive_errors >= self.max_consecutive_errors:
                        logging.error(f"Too many errors for {self.channel_name} MultiModel. Resetting counter.")
                        self.consecutive_errors = 0
                    continue
                except Exception as e:
                    logging.error(f"Unexpected error in MultiModel {self.channel_name} for {app_name}: {e}")
                    break
                
                if not results:
                    continue

                # Process results - ALWAYS update display, even if no boxes detected
                if app_name == 'Generic':
                    # Log that we're processing this frame
                    if frame_count % 10 == 0:  # Log every 10 frames
                        logging.info(f"🎯 Generic {self.channel_name} processing frame {frame_count}, boxes: {len(results[0].boxes) if results and results[0].boxes else 0}")
                    
                    # Extract all detections
                    all_detections = []
                    if results and len(results[0].boxes) > 0:
                        for box in results[0].boxes:
                            class_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            # Log ALL detections above 0.15 for debugging
                            if conf >= 0.15:
                                logging.info(f"🔍 Generic {self.channel_name} RAW: class_id={class_id}, name={task['model'].names.get(class_id, f'Class_{class_id}')}, conf={conf:.3f}")
                            
                            all_detections.append({
                                'class_id': class_id,
                                'confidence': conf,
                                'box': box
                            })
                    
                    # Apply smart validation
                    filtered_detections = self.apply_smart_validation(all_detections, task['model'])
                    
                    # Apply higher confidence threshold for phone detection (reduce false positives)
                    PHONE_MIN_CONFIDENCE = 0.5  # Require 50% confidence for phone to reduce paper/false detections
                    final_detections = []
                    for det in filtered_detections:
                        if det['class_id'] == 8:  # Using_phone (class 8 in new model)
                            if det['confidence'] >= PHONE_MIN_CONFIDENCE:
                                final_detections.append(det)
                            else:
                                logging.debug(f"Rejected phone detection (conf={det['confidence']:.2f} < {PHONE_MIN_CONFIDENCE})")
                        else:
                            final_detections.append(det)
                    
                    # Debug logging (ALWAYS log what we see)
                    if all_detections:
                        detected_classes_debug = [f"{task['model'].names[det['class_id']]}({det['confidence']:.2f})" for det in all_detections]
                        filtered_classes_debug = [f"{task['model'].names[det['class_id']]}({det['confidence']:.2f})" for det in final_detections]
                        logging.info(f"🔍 Generic {self.channel_name} Frame {frame_count}: Raw: {detected_classes_debug} | After filtering: {filtered_classes_debug}")
                        
                        # Special alert for phone detection
                        phone_detections = [det for det in final_detections if det['class_id'] == 8]  # Using_phone is class 8
                        if phone_detections:
                            logging.warning(f"📱 PHONE DETECTED in {self.channel_name}! Confidence: {phone_detections[0]['confidence']:.2f}")
                    
                    # Define class sets and colors
                    # NEW MODEL: Only "Without_" classes and "Using_phone" are violations
                    violation_classes = {1, 3, 5, 7, 8}  # Without_cap, Without_apron, Without_gloves, Without_uniform, Using_phone
                    compliance_classes = {0, 2, 4, 6}  # Cap_present, With_apron, With_gloves, Uniform
                    COLOR_GREEN = (0, 255, 0)  # Compliance
                    COLOR_RED = (0, 0, 255)    # Violations
                    
                    # Create annotated frame - START with frame copy (ALWAYS process every frame)
                    annotated_frame = frame.copy()
                    h, w = annotated_frame.shape[:2]
                    violation_detected_classes = []
                    
                    # Draw ALL detections (violations in red, compliance in green) - use final_detections
                    for det in final_detections:
                        box = det['box']
                        class_id = det['class_id']
                        class_name = task['model'].names[class_id]
                        conf = det['confidence']
                        
                        # Get box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        
                        # Check if detection is inside ROI (only for violations)
                        is_in_roi = True
                        if class_id in violation_classes:
                            is_in_roi = self._is_in_roi(x1, y1, x2, y2, w, h)
                            if not is_in_roi:
                                # Skip this violation - it's outside ROI
                                continue
                        
                        # Determine color based on class type
                        if class_id in violation_classes:
                            color = COLOR_RED
                            violation_detected_classes.append(class_name)
                        elif class_id in compliance_classes:
                            color = COLOR_GREEN
                        else:
                            color = (128, 128, 128)  # Gray for unknown
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)  # Thicker boxes
                        
                        # Draw label with background
                        label = f"{class_name} {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + label_size[0] + 5, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1 + 2, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw ROI polygon if configured
                    if self.roi_polygon is not None:
                        try:
                            # Get ROI coordinates and convert from normalized to pixel coordinates
                            roi_coords = list(self.roi_polygon.exterior.coords)
                            pixel_coords = np.array([[int(x * w), int(y * h)] for x, y in roi_coords], dtype=np.int32)
                            
                            # Draw ROI polygon
                            cv2.polylines(annotated_frame, [pixel_coords], isClosed=True, color=(255, 255, 0), thickness=2)
                            cv2.putText(annotated_frame, "ROI Zone", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        except Exception as e:
                            logging.debug(f"Error drawing ROI: {e}")
                    
                    # ALWAYS add frame info overlay - this proves continuous processing
                    violation_count = len(violation_detected_classes)
                    info_text = f"Frame: {frame_count} | Detections: {len(final_detections)} | Violations: {violation_count}"
                    # Black background for better visibility
                    cv2.rectangle(annotated_frame, (5, 5), (650, 45), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Store annotated frame for streaming (ALWAYS update, even with no detections)
                    with self.lock:
                        self.latest_frame = annotated_frame
                    
                    # Save screenshot/send alert based on violation type
                    if violation_detected_classes:
                        # Check if Using_phone (class 7) is in violations - 30 second cooldown for phone
                        has_phone_violation = any('Using_phone' in cls_name for cls_name in violation_detected_classes)
                        
                        if has_phone_violation:
                            # Phone usage: 30 second cooldown
                            phone_cooldown = 30
                            if current_time - self.last_detection_times[app_name] > phone_cooldown:
                                message = f"Front Office Violation: {', '.join(set(violation_detected_classes))}"
                                logging.warning(f"📱 PHONE VIOLATION DETECTED! Taking screenshot - {self.channel_name}")
                                self.last_detection_times[app_name] = current_time
                                self.detection_callback(app_name, self.channel_id, [annotated_frame], message, False)
                        elif current_time - self.last_detection_times[app_name] > self.cooldown:
                            # Other violations: apply 120 second cooldown
                            message = f"Front Office Violation: {', '.join(set(violation_detected_classes))}"
                            logging.info(f"📸 {self.channel_name} - {message}")
                            self.last_detection_times[app_name] = current_time
                            self.detection_callback(app_name, self.channel_id, [annotated_frame], message, False)
                
                elif results and len(results[0].boxes) > 0:
                        # Original logic for non-Generic apps
                        # Store annotated frame for streaming
                        with self.lock:
                            self.latest_frame = results[0].plot()
                        
                        # Extract detected class names for better message
                        detected_classes = []
                        for box in results[0].boxes:
                            class_id = int(box.cls[0])
                            class_name = task['model'].names[class_id]
                            detected_classes.append(class_name)
                        
                        # Create descriptive message
                        unique_classes = list(set(detected_classes))
                        message = f"{app_name}: {', '.join(unique_classes)}"
                        
                        # Only save screenshot/send alert if cooldown has passed
                        if current_time - self.last_detection_times[app_name] > self.cooldown:
                            logging.info(f"📸 {self.channel_name} - {message}")
                            self.last_detection_times[app_name] = current_time
                            if task['is_gif']:
                                frames_to_capture = self.gif_duration_seconds * self.fps
                                gif_frames = [results[0].plot()]
                                # Collect additional frames from frame_hub for GIF
                                for _ in range(frames_to_capture - 1):
                                    time.sleep(1 / self.fps)
                                    frame_gif = getattr(self, 'frame_hub', None).get_latest() if hasattr(self, 'frame_hub') else None
                                    if frame_gif is None:
                                        break
                                    
                                    # Validate GIF frame before processing
                                    if not self._validate_frame(frame_gif):
                                        gif_frames.append(frame_gif.copy())  # Use frame even if invalid to maintain frame count
                                        continue
                                
                                    # Run detection on this frame to get annotated version
                                    try:
                                        # CPU-only mode
                                        with torch.inference_mode():
                                            gif_results = task['model'](
                                                frame_gif,
                                                device='cpu',
                                                half=False,
                                                **model_args
                                            )
                                        if gif_results and len(gif_results[0].boxes) > 0:
                                            gif_frames.append(gif_results[0].plot())
                                        else:
                                            gif_frames.append(frame_gif.copy())
                                    except Exception as e:
                                        logging.warning(f"Error processing GIF frame: {e}")
                                        gif_frames.append(frame_gif.copy())
                                # Only create GIF if we have multiple frames
                                if len(gif_frames) > 1:
                                    self.detection_callback(app_name, self.channel_id, gif_frames, message, True)
                                else:
                                    # Fallback to single frame if we couldn't capture enough frames
                                    self.detection_callback(app_name, self.channel_id, gif_frames, message, False)
                            else:
                                annotated_frame = results[0].plot()
                                self.detection_callback(app_name, self.channel_id, [annotated_frame], message, False)
        # No cap to release when using FrameHub

class RawFeedProcessor(threading.Thread):
    """Raw video feed without any AI detection - just displays the RTSP stream"""
    def __init__(self, rtsp_url, channel_id, channel_name):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.is_running = True
        self.latest_frame = None
        self.lock = threading.Lock()
        logging.info(f"RawFeedProcessor initialized for {channel_name} (no AI detection)")

    def stop(self):
        self.is_running = False

    def shutdown(self):
        logging.info(f"Shutting down RawFeed for {self.channel_name}")
        self.is_running = False

    def get_frame(self):
        """Get the latest raw frame for video streaming"""
        with self.lock:
            if self.latest_frame is None:
                placeholder = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(placeholder, 'Connecting...', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (201, 209, 217), 2)
                _, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()
            _, jpeg = cv2.imencode('.jpg', self.latest_frame)
            return jpeg.tobytes()

    def run(self):
        """Simply fetch and store frames without any AI processing"""
        while self.is_running:
            frame = getattr(self, 'frame_hub', None).get_latest() if hasattr(self, 'frame_hub') else None
            if frame is None:
                time.sleep(0.01)
                continue

            # Just store the raw frame for video streaming
            with self.lock:
                self.latest_frame = frame.copy()
            
            time.sleep(0.03)  # ~30 FPS

# QueueMonitorProcessor class moved to queue_monitor.py module
# PeopleCounterProcessor class moved to people_counter.py module

# OccupancyMonitorProcessor class moved to occupancy_monitor_processor.py module


def get_app_configs(restaurant_id=None):
    """Get application configs, optionally filtered by restaurant
    
    First tries to load from database (new multi-restaurant structure).
    Falls back to rtsp_links.txt if database is empty (backward compatibility).
    """
    app_configs = defaultdict(lambda: {'channels': [], 'online_count': 0})
    
    # Try loading from database first
    if db_connected:
        try:
            with SessionLocal() as db:
                # Check if we have any cameras in database
                camera_count = db.query(Camera).count()
                
                if camera_count > 0:
                    logging.debug(f"Loading config from database (found {camera_count} cameras)")
                    
                    # Build query
                    query = db.query(Camera, CameraApp, Restaurant).\
                        join(CameraApp, Camera.id == CameraApp.camera_id).\
                        join(Restaurant, Camera.restaurant_id == Restaurant.id).\
                        filter(Camera.is_active == True, CameraApp.is_active == True, Restaurant.is_active == True)
                    
                    # Filter by restaurant if specified
                    if restaurant_id:
                        query = query.filter(Camera.restaurant_id == restaurant_id)
                        logging.debug(f"Filtering by restaurant_id: {restaurant_id}")
                    
                    results = query.all()
                    logging.info(f"Database query returned {len(results)} camera-app combinations")
                    
                    # Build app configs structure
                    for camera, camera_app, restaurant in results:
                        app_name = camera_app.app_name
                        
                        # Check if channel is online
                        processors = stream_processors.get(camera.channel_id, [])
                        is_alive = any(p.is_alive() for p in processors) if processors else False
                        
                        # Add channel to app config (avoid duplicates)
                        if not any(d['id'] == camera.channel_id for d in app_configs[app_name]['channels']):
                            app_configs[app_name]['channels'].append({
                                'id': camera.channel_id,
                                'name': camera.channel_name,
                                'restaurant_id': restaurant.id,
                                'restaurant_name': restaurant.restaurant_name,
                                'is_alive': is_alive
                            })
                    
                    # Calculate online counts
                    for app_name, config in app_configs.items():
                        online_count = sum(1 for ch in config['channels'] if ch.get('is_alive', False))
                        config['online_count'] = online_count
                    
                    logging.info(f"Loaded config from database: {len(app_configs)} apps")
                    return dict(app_configs)
                else:
                    logging.info("No cameras in database, falling back to rtsp_links.txt")
        except Exception as e:
            logging.warning(f"Error loading from database, falling back to file: {e}")
    
    # Fallback to rtsp_links.txt (backward compatibility)
    logging.info("Loading config from rtsp_links.txt (legacy mode)")
    if not os.path.exists(RTSP_LINKS_FILE): 
        logging.warning(f"Neither database nor {RTSP_LINKS_FILE} available")
        return {}
    
    channel_status = {}
    all_channel_ids = set()
    
    with open(RTSP_LINKS_FILE, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 3: continue
                link, channel_name = parts[0], parts[1]
                channel_id = get_stable_channel_id(link)
                all_channel_ids.add(channel_id)
                processors = stream_processors.get(channel_id, [])
                is_alive = any(p.is_alive() for p in processors) if processors else False
                channel_status[channel_id] = {'name': channel_name, 'is_alive': is_alive}
    
    with open(RTSP_LINKS_FILE, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 3: continue
                link, _, app_names = parts[0], parts[1], parts[2:]
                channel_id = get_stable_channel_id(link)
                for app_name in app_names:
                    if app_name in APP_TASKS_CONFIG:
                        if not any(d['id'] == channel_id for d in app_configs[app_name]['channels']):
                            app_configs[app_name]['channels'].append({
                                'id': channel_id,
                                'name': channel_status[channel_id]['name']
                            })
    
    for app_name, config in app_configs.items():
        online_count = sum(1 for ch in config['channels'] if channel_status.get(ch['id'], {}).get('is_alive', False))
        config['online_count'] = online_count
    
    return dict(app_configs)

def log_queue_counts():
    if not db_connected: return
    with SessionLocal() as db:
        for channel_id, processors in stream_processors.items():
            for p in processors:
                if isinstance(p, QueueMonitorProcessor):
                    db.add(QueueLog(channel_id=channel_id, queue_count=p.current_queue_count))
        db.commit()
    logging.info("Scheduled job: Saved current queue counts to database.")


@app.route('/')
def landing_page(): 
    return render_template('landing.html')

@app.route('/display/<channel_id>')
def display_feed(channel_id):
    processors = stream_processors.get(channel_id)
    if not processors:
        return ("Stream not found", 404)
    proc = processors[0]
    return Response(gen_video_feed(proc), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        # If already logged in, redirect to dashboard
        if session.get('logged_in'):
            return redirect('/dashboard')
        return render_template('login.html')
    
    elif request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if username == LOGIN_USERNAME and password == LOGIN_PASSWORD:
            session['logged_in'] = True
            session['username'] = username
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.route('/conversion-analytics')
@login_required
def conversion_analytics():
    """Footfall-to-Sales Conversion Analytics Page"""
    restaurant_id = request.args.get('restaurant_id', type=int)
    
    # Get restaurants list for selector
    with SessionLocal() as db:
        restaurants_data = db.execute(text("""
            SELECT id, restaurant_code, restaurant_name, location
            FROM restaurants
            WHERE is_active = true
            ORDER BY id
        """)).fetchall()
        
        restaurants = [
            {
                'id': r[0],
                'code': r[1],
                'name': r[2],
                'location': r[3],
                'display_name': f"{r[2]} - {r[3]}"
            }
            for r in restaurants_data
        ]
        
        # Add "All Restaurants" option at the beginning
        all_option = {
            'id': None,
            'code': 'all',
            'name': 'All Restaurants',
            'location': 'Combined',
            'display_name': '🏢 All Restaurants'
        }
        restaurants.insert(0, all_option)
        
        # Get selected restaurant details
        selected_restaurant = None
        if restaurant_id:
            selected = [r for r in restaurants if r['id'] == restaurant_id]
            selected_restaurant = selected[0] if selected else None
    
    return render_template('conversion_analytics.html',
                         restaurants=restaurants,
                         selected_restaurant=selected_restaurant,
                         restaurant_id=restaurant_id)

@app.route('/debug-conversion')
@login_required
def debug_conversion():
    """Debug page for conversion analytics"""
    restaurant_id = request.args.get('restaurant_id', type=int)
    
    # Get restaurants list for selector
    with SessionLocal() as db:
        restaurants_data = db.execute(text("""
            SELECT id, restaurant_code, restaurant_name, location
            FROM restaurants
            WHERE is_active = true
            ORDER BY id
        """)).fetchall()
        
        restaurants = [
            {
                'id': r[0],
                'code': r[1],
                'name': r[2],
                'location': r[3],
                'display_name': f"{r[2]} - {r[3]}"
            }
            for r in restaurants_data
        ]
        
        # Add "All Restaurants" option
        all_option = {
            'id': None,
            'code': 'all',
            'name': 'All Restaurants',
            'location': 'Combined',
            'display_name': '🏢 All Restaurants'
        }
        restaurants.insert(0, all_option)
        
        selected_restaurant = None
        if restaurant_id:
            selected = [r for r in restaurants if r['id'] == restaurant_id]
            selected_restaurant = selected[0] if selected else None
    
    return render_template('debug_conversion.html',
                         restaurants=restaurants,
                         selected_restaurant=selected_restaurant,
                         restaurant_id=restaurant_id)

@app.route('/idle-people-violation')
@login_required
def idle_people_violation():
    """Idle People Violation monitoring page"""
    restaurant_id = request.args.get('restaurant_id', type=int)
    
    # Get app configs to find IdlePeopleViolation channels
    app_configs = get_app_configs()
    idle_channels = app_configs.get('IdlePeopleViolation', {}).get('channels', [])
    
    # Filter by restaurant if specified
    if restaurant_id:
        idle_channels = [ch for ch in idle_channels if ch.get('restaurant_id') == restaurant_id]
    
    return render_template('idle_people_violation.html', channels=idle_channels, restaurant_id=restaurant_id)

@app.route('/roi_editor_idle_people')
@login_required
def roi_editor_idle_people():
    """ROI Editor for Idle People Violation"""
    channel_id = request.args.get('channel_id')
    if not channel_id:
        # Default to first available channel
        app_configs = get_app_configs()
        idle_channels = app_configs.get('IdlePeopleViolation', {}).get('channels', [])
        if idle_channels:
            # Extract channel id from the first channel dict
            channel_id = idle_channels[0]['id'] if isinstance(idle_channels[0], dict) else idle_channels[0]
        else:
            return "No channels configured for Idle People Violation", 404
    
    return render_template('roi_editor_idle_people.html', channel_id=channel_id)

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard - supports optional restaurant filtering via query parameter"""
    restaurant_id = request.args.get('restaurant_id', type=int)
    
    # Get list of all restaurants for dropdown (if database connected)
    restaurants = []
    selected_restaurant = None
    default_restaurant_id = None
    
    if db_connected:
        try:
            with SessionLocal() as db:
                restaurant_query = db.query(Restaurant).filter(Restaurant.is_active == True).order_by(Restaurant.restaurant_name).all()
                restaurants = [
                    {
                        'id': r.id,
                        'restaurant_name': r.restaurant_name,
                        'location': r.location,
                        'display_name': f"{r.restaurant_name} - {r.location}" if r.location else r.restaurant_name,
                        'petpooja_rest_id': r.petpooja_rest_id
                    }
                    for r in restaurant_query
                ]
                
                # Default to main_store (restaurant_id=2) if no restaurant specified
                if not restaurant_id and restaurants:
                    # Find main_store (id=2) and set as default
                    main_store = next((r for r in restaurants if r['id'] == 2), None)
                    if main_store:
                        default_restaurant_id = 2
                    elif restaurants:
                        # Fallback to first restaurant if main_store not found
                        default_restaurant_id = restaurants[0]['id']
                    
                    # Redirect to dashboard with default restaurant_id parameter
                    if default_restaurant_id:
                        logging.info(f"No restaurant selected - redirecting to default restaurant {default_restaurant_id}")
                        return redirect(url_for('dashboard', restaurant_id=default_restaurant_id))
                
                # Get selected restaurant details
                if restaurant_id:
                    selected = db.query(Restaurant).filter_by(id=restaurant_id, is_active=True).first()
                    if selected:
                        selected_restaurant = {
                            'id': selected.id,
                            'restaurant_name': selected.restaurant_name,
                            'location': selected.location,
                            'petpooja_rest_id': selected.petpooja_rest_id
                        }
                        logging.info(f"Dashboard loaded for restaurant: {selected_restaurant['restaurant_name']} (PetPooja ID: {selected_restaurant['petpooja_rest_id']})")
        except Exception as e:
            logging.warning(f"Could not load restaurants for dashboard: {e}")
    
    # Get app configs (filtered by restaurant if specified, otherwise show all)
    app_configs = get_app_configs(restaurant_id=restaurant_id)
    
    import time
    timestamp = int(time.time())
    
    return render_template(
        'dashboard.html',
        app_configs=app_configs,
        restaurants=restaurants,
        selected_restaurant=selected_restaurant,
        timestamp=timestamp
    )

def gen_video_feed(processor):
    """Generator function for video feed - works with both direct run and gunicorn"""
    while True:
        try:
            # ~30 FPS pacing, and tolerate None frames
            time.sleep(0.03)
            frame_bytes = processor.get_frame()
            if not frame_bytes:
                continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        except GeneratorExit:
            # Client disconnected
            break
        except Exception as e:
            logging.error(f"Error in video feed generator: {e}")
            time.sleep(0.1)
            continue

@app.route('/video_feed/<app_name>/<channel_id>')
@login_required
def video_feed(app_name, channel_id):
    """Video feed endpoint - works with both direct run and gunicorn"""
    # Ensure initialization (non-blocking check)
    if not _initialized:
        # Try to ensure initialization, but don't block
        _ensure_initialized(background=True)
        # Wait a bit for initialization if still in progress
        if _initialization_thread and _initialization_thread.is_alive():
            _initialization_thread.join(timeout=2.0)  # Wait max 2 seconds
    
    processors = stream_processors.get(channel_id)
    if not processors:
        logging.warning(f"Video feed requested for channel {channel_id} but processors not found")
        return (f"Stream not found for channel {channel_id}", 404)
    
    target_processor, target_class = None, None
    if app_name == 'PeopleCounter': target_class = PeopleCounterProcessor
    elif app_name == 'QueueMonitor': target_class = QueueMonitorProcessor
    elif app_name == 'KitchenCompliance': target_class = KitchenComplianceProcessor
    elif app_name == 'OccupancyMonitor': target_class = OccupancyMonitorProcessor
    elif app_name == 'IdlePeopleViolation': target_class = IdlePeopleViolationProcessor
    elif app_name == 'Generic': target_class = MultiModelProcessor
    elif app_name == 'RawFeed': target_class = RawFeedProcessor
    
    if target_class:
        target_processor = next((p for p in processors if isinstance(p, target_class)), None)
    
    if target_processor:
        if target_processor.is_alive():
            logging.info(f"Streaming video feed for {app_name} on channel {channel_id}")
            return Response(gen_video_feed(target_processor), mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            logging.warning(f"{app_name} processor for channel {channel_id} is not alive")
            return (f"{app_name} processor is not running for this channel", 503)
    else:
        logging.warning(f"{app_name} processor not found for channel {channel_id}")
        available = [type(p).__name__ for p in processors]
        return (f"{app_name} stream not found for channel {channel_id}. Available: {available}", 404)


@app.route('/history/<app_name>')
@login_required
def get_history(app_name):
    if not db_connected: return jsonify({"error": "Database not connected"}), 500
    try:
        page, limit = int(request.args.get('page', 1)), int(request.args.get('limit', 10))
        channel_id, start_date_str, end_date_str = request.args.get('channel_id'), request.args.get('start_date'), request.args.get('end_date')
        violation_type = request.args.get('violation_type')  # Add violation type filter
        restaurant_id = request.args.get('restaurant_id', type=int)  # Add restaurant filter
    except (ValueError, TypeError): return jsonify({"error": "Invalid page or limit parameter"}), 400
    offset = (page - 1) * limit
    with SessionLocal() as db:
        try:
            # Join with cameras table to filter by restaurant
            query = db.query(Detection).join(Camera, Detection.channel_id == Camera.channel_id).filter(Detection.app_name == app_name)
            
            if channel_id and channel_id != 'null': query = query.filter(Detection.channel_id == channel_id)
            if restaurant_id: query = query.filter(Camera.restaurant_id == restaurant_id)
            
            if start_date_str and end_date_str:
                try:
                    start_date, end_date = datetime.strptime(start_date_str, '%Y-%m-%d').date(), datetime.strptime(end_date_str, '%Y-%m-%d').date()
                    query = query.filter(Detection.timestamp.between(start_date, datetime.combine(end_date, datetime.max.time())))
                except ValueError: return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400
            # Filter by violation type for Kitchen Compliance and Generic (Front Office)
            if violation_type and app_name in ['KitchenCompliance', 'Generic']:
                query = query.filter(Detection.message.like(f'%{violation_type}%'))
            total_detections, detections = query.count(), query.order_by(Detection.timestamp.desc()).offset(offset).limit(limit).all()
            total_pages = (total_detections + limit - 1) // limit  # Calculate total pages
            return jsonify({'detections': [{'timestamp': d.timestamp.strftime("%Y-%m-%d %H:%M:%S"),'message': d.message,'channel_id': d.channel_id,'media_url': url_for('static', filename=d.media_path)} for d in detections],'total': total_detections, 'page': page, 'limit': limit, 'total_pages': total_pages})
        except Exception as e:
            logging.error(f"Error fetching history: {e}")
            return jsonify({"error": "Could not fetch history from database"}), 500

@app.route('/roi_editor')
@login_required
def roi_editor():
    """ROI Editor page for drawing queue monitor regions"""
    # Get channel_id from query params or show channel selector
    channel_id = request.args.get('channel_id')
    if not channel_id:
        # Show available channels
        app_configs = get_app_configs()
        queue_channels = app_configs.get('QueueMonitor', {}).get('channels', [])
        if len(queue_channels) == 1:
            # Auto-redirect to the only available channel
            return redirect(f'/roi_editor?channel_id={queue_channels[0]["id"]}')
        return render_template('roi_editor.html', channels=queue_channels, show_selector=True)
    
    return render_template('roi_editor.html', channel_id=channel_id, show_selector=False)

@app.route('/roi_editor_people')
@login_required
def roi_editor_people():
    """ROI Editor page for PeopleCounter counting line"""
    channel_id = request.args.get('channel_id')
    if not channel_id:
        # Show available channels
        app_configs = get_app_configs()
        people_channels = app_configs.get('PeopleCounter', {}).get('channels', [])
        if len(people_channels) == 1:
            # Auto-redirect to the only available channel
            return redirect(f'/roi_editor_people?channel_id={people_channels[0]["id"]}')
        return render_template('roi_editor_people.html', channels=people_channels, show_selector=True)
    
    return render_template('roi_editor_people.html', channel_id=channel_id, show_selector=False)

@app.route('/roi_editor_occupancy')
@login_required
def roi_editor_occupancy():
    """ROI Editor page for OccupancyMonitor area"""
    channel_id = request.args.get('channel_id')
    if not channel_id:
        # Show available channels
        app_configs = get_app_configs()
        occupancy_channels = app_configs.get('OccupancyMonitor', {}).get('channels', [])
        if len(occupancy_channels) == 1:
            # Auto-redirect to the only available channel
            return redirect(f'/roi_editor_occupancy?channel_id={occupancy_channels[0]["id"]}')
        return render_template('roi_editor_occupancy.html', channels=occupancy_channels, show_selector=True)
    
    return render_template('roi_editor_occupancy.html', channel_id=channel_id, show_selector=False)

@app.route('/api/set_roi', methods=['POST'])
@login_required
def set_roi():
    if not db_connected: return jsonify({"error": "Database not connected"}), 500
    data = request.json
    channel_id, app_name, roi_points = data.get('channel_id'), data.get('app_name'), data.get('roi_points')
    if not all([channel_id, app_name, isinstance(roi_points, dict)]):
        return jsonify({"error": "Missing or invalid data"}), 400
    with SessionLocal() as db:
        try:
            stmt = text("""
                INSERT INTO roi_configs (channel_id, app_name, roi_points) VALUES (:cid, :an, :rp)
                ON CONFLICT (channel_id, app_name) DO UPDATE SET roi_points = EXCLUDED.roi_points;
            """)
            db.execute(stmt, {'cid': channel_id, 'an': app_name, 'rp': json.dumps(roi_points)})
            db.commit()

            processors = stream_processors.get(channel_id, [])
            target_class = None
            if app_name == 'QueueMonitor': 
                target_class = QueueMonitorProcessor
            elif app_name == 'PeopleCounter':
                target_class = PeopleCounterProcessor
            elif app_name == 'OccupancyMonitor':
                target_class = OccupancyMonitorProcessor
            
            if target_class:
                for p in processors:
                    if isinstance(p, target_class):
                        if app_name == 'QueueMonitor' and hasattr(p, 'update_roi'):
                            p.update_roi(roi_points)
                            logging.info(f"Sent live ROI update to {app_name} for {channel_id}")
                        elif app_name == 'PeopleCounter' and hasattr(p, 'update_line_position'):
                            line_pos = roi_points.get('line_position', 0.38)
                            p.update_line_position(line_pos)
                            logging.info(f"Sent live line position update to {app_name} for {channel_id}")
                        elif app_name == 'OccupancyMonitor' and hasattr(p, 'update_roi'):
                            p.update_roi(roi_points)
                            logging.info(f"Sent live ROI update to {app_name} for {channel_id}")
                        break

            return jsonify({"success": True, "message": "ROI updated successfully."})
        except Exception as e:
            db.rollback()
            logging.error(f"Error saving ROI: {e}")
            return jsonify({"error": "Could not save ROI to database"}), 500

@app.route('/api/get_roi', methods=['GET'])
@login_required
def get_roi():
    if not db_connected: return jsonify({"error": "Database not connected"}), 500
    channel_id = request.args.get('channel_id')
    app_name = request.args.get('app_name')
    if not all([channel_id, app_name]):
        return jsonify({"error": "Missing channel_id or app_name"}), 400
    
    with SessionLocal() as db:
        try:
            roi_record = db.query(RoiConfig).filter_by(channel_id=channel_id, app_name=app_name).first()
            if roi_record and roi_record.roi_points:
                roi_points = json.loads(roi_record.roi_points)
                return jsonify({"success": True, "roi_points": roi_points})
            else:
                return jsonify({"success": False, "message": "No ROI configuration found"})
        except Exception as e:
            logging.error(f"Error fetching ROI: {e}")
            return jsonify({"error": "Could not fetch ROI from database"}), 500

@app.route('/report/<channel_id>/<date_str>')
@login_required
def get_report(channel_id, date_str):
    if not db_connected: return jsonify({"error": "DB not connected"}), 500
    try: report_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError: return jsonify({"error": "Invalid date format"}), 400
    hourly_data = {h: {'in': 0, 'out': 0, 'total': 0} for h in range(24)}
    with SessionLocal() as db:
        for r in db.query(HourlyFootfall).filter_by(channel_id=channel_id, report_date=report_date).order_by(HourlyFootfall.hour).all():
            hourly_data[r.hour].update({'in': r.in_count, 'out': r.out_count, 'total': r.in_count + r.out_count})
    # Note: Hourly counts are now updated in real-time, so database already has the latest data
    # No need to calculate from processor counts anymore
    if not any(v['total'] > 0 for v in hourly_data.values()):
        with SessionLocal() as db:
            daily_record = db.query(DailyFootfall).filter_by(channel_id=channel_id, report_date=report_date).first()
            if not daily_record or (daily_record.in_count == 0 and daily_record.out_count == 0):
                 return jsonify({"error": "No data for this date"}), 404
    return jsonify({'hourly_data': hourly_data})

@app.route('/generate_report/<channel_id>')
@login_required
def generate_report(channel_id):
    if not db_connected: return jsonify({"error": "DB not connected"}), 500
    period, report_format = request.args.get('period', '7days'), request.args.get('format', 'json')
    end_date = date.today()
    if period == '7days': start_date = end_date - timedelta(days=6)
    elif period == '30days': start_date = end_date - timedelta(days=29)
    else: start_date = end_date - timedelta(days=6)
    with SessionLocal() as db:
        daily_records = db.query(DailyFootfall).filter(DailyFootfall.channel_id == channel_id, DailyFootfall.report_date.between(start_date, end_date)).order_by(DailyFootfall.report_date).all()
        hourly_records = db.query(HourlyFootfall).filter(HourlyFootfall.channel_id == channel_id, HourlyFootfall.report_date.between(start_date, end_date)).all()
        if not daily_records: return jsonify({"error": f"No data found for the last {period.replace('days', '')} days."})
        labels = [(start_date + timedelta(days=i)).strftime("%b %d") for i in range((end_date - start_date).days + 1)]
        daily_totals, hourly_totals, total_traffic = {label: 0 for label in labels}, defaultdict(int), 0
        for record in daily_records:
            label, total = record.report_date.strftime("%b %d"), record.in_count
            if label in daily_totals: daily_totals[label] = total
            total_traffic += total
        for record in hourly_records: hourly_totals[record.hour] += record.in_count
        busiest_day_label, peak_hour_label = "N/A", "N/A"
        if any(daily_totals.values()):
            busiest_day_value = max(daily_totals.values())
            busiest_day = [day for day, total in daily_totals.items() if total == busiest_day_value][0]
            busiest_day_label = f"{busiest_day} ({busiest_day_value} visitors)"
        if hourly_totals:
            peak_hour_24 = max(hourly_totals, key=hourly_totals.get)
            peak_hour_label = f"{datetime.strptime(str(peak_hour_24), '%H').strftime('%I %p')} ({hourly_totals[peak_hour_24]} avg)"
        summary = {"total_footfall": total_traffic, "busiest_day": busiest_day_label, "peak_hour": peak_hour_label}
        if report_format == 'csv':
            def generate_csv():
                data = io.StringIO(); writer = csv.writer(data)
                writer.writerow(['Date', 'Total Visitors (In Count)']); yield data.getvalue(); data.seek(0); data.truncate(0)
                for label, total in daily_totals.items(): writer.writerow([label, total]); yield data.getvalue(); data.seek(0); data.truncate(0)
            return Response(stream_with_context(generate_csv()), mimetype='text/csv', headers={"Content-Disposition": f"attachment;filename=report_{channel_id}_in_count.csv"})
        else:
            return jsonify({"labels": list(daily_totals.keys()), "data": list(daily_totals.values()), "summary": summary})

@app.route('/queue_report/<channel_id>')
@login_required
def get_queue_report(channel_id):
    if not db_connected: return jsonify({"error": "DB not connected"}), 500
    period, start_date_str, end_date_str = request.args.get('period'), request.args.get('start_date'), request.args.get('end_date')
    now = datetime.now(IST)
    if start_date_str and end_date_str:
        start_dt, end_dt = IST.localize(datetime.strptime(start_date_str, '%Y-%m-%d')), IST.localize(datetime.combine(datetime.strptime(end_date_str, '%Y-%m-%d'), datetime.max.time()))
    elif period == 'today': start_dt, end_dt = now.replace(hour=0, minute=0, second=0, microsecond=0), now
    elif period == 'yesterday':
        yesterday = now - timedelta(days=1)
        start_dt, end_dt = yesterday.replace(hour=0, minute=0, second=0, microsecond=0), yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
    else: start_dt, end_dt = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=6), now
    with SessionLocal() as db:
        records = db.query(QueueLog).filter(QueueLog.channel_id == channel_id, QueueLog.timestamp.between(start_dt, end_dt)).order_by(QueueLog.timestamp).all()
        if not records: return jsonify({"error": "No data found for the selected period."})
        labels = [r.timestamp.strftime('%H:%M' if period in ['today', 'yesterday'] else '%d %b %H:%M') for r in records]
        data = [r.queue_count for r in records]
        max_queue, avg_queue = (max(data), round(sum(data) / len(data), 1)) if data else (0, 0)
        hourly_counts = defaultdict(list)
        for r in records: hourly_counts[r.timestamp.hour].append(r.queue_count)
        peak_hour = "N/A"
        if hourly_counts: peak_hour = datetime.strptime(str(max({h: sum(c)/len(c) for h, c in hourly_counts.items()}, key=lambda h: sum(hourly_counts[h])/len(hourly_counts[h]))), '%H').strftime('%I %p')
        summary = { 'max_queue_length': max_queue, 'avg_queue_length': avg_queue, 'peak_hour': peak_hour }
        return jsonify({'labels': labels, 'data': data, 'summary': summary})

@app.route('/api/peak_analytics/<channel_id>')
@login_required
def get_peak_analytics(channel_id):
    """Get peak day (current week) and peak hour (today) with hourly chart data"""
    if not db_connected: return jsonify({"error": "Database not connected"}), 500
    try:
        with SessionLocal() as db:
            today = date.today()
            # Current week: Monday to Sunday
            week_start = today - timedelta(days=today.weekday())
            week_end = week_start + timedelta(days=6)
            
            # Get daily data for current week
            daily_records = db.query(DailyFootfall).filter(
                DailyFootfall.channel_id == channel_id,
                DailyFootfall.report_date.between(week_start, week_end)
            ).order_by(DailyFootfall.report_date).all()
            
            # Calculate peak day of week
            peak_day_name = "N/A"
            peak_day_count = 0
            week_data = []
            
            if daily_records:
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                for i in range(7):
                    current_date = week_start + timedelta(days=i)
                    record = next((r for r in daily_records if r.report_date == current_date), None)
                    count = record.in_count if record else 0
                    week_data.append({'day': day_names[i], 'count': count})
                    
                    if count > peak_day_count:
                        peak_day_count = count
                        peak_day_name = day_names[i]
            
            # Get hourly data for today
            hourly_records = db.query(HourlyFootfall).filter(
                HourlyFootfall.channel_id == channel_id,
                HourlyFootfall.report_date == today
            ).order_by(HourlyFootfall.hour).all()
            
            # Calculate peak hour of today
            peak_hour_label = "N/A"
            peak_hour_count = 0
            hourly_data = []
            
            if hourly_records:
                for record in hourly_records:
                    count = record.in_count
                    hour_12 = datetime.strptime(str(record.hour), '%H').strftime('%I %p').lstrip('0')
                    hourly_data.append({'hour': hour_12, 'count': count})
                    
                    if count > peak_hour_count:
                        peak_hour_count = count
                        peak_hour_label = hour_12
            
            return jsonify({
                'peak_day': {
                    'name': peak_day_name,
                    'count': peak_day_count,
                    'week_data': week_data
                },
                'peak_hour': {
                    'label': peak_hour_label,
                    'count': peak_hour_count,
                    'hourly_data': hourly_data
                }
            })
    except Exception as e:
        logging.error(f"Error in peak analytics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sales-analytics/daily')
@login_required
def get_sales_analytics_daily():
    """
    Get daily sales analytics with proper restaurant filtering
    REFACTORED: Business logic moved to petpooja_integration.SalesAnalytics
    """
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        days = request.args.get('days', default=7, type=int)
        restaurant_id = request.args.get('restaurant_id', type=int)
        
        end_date = datetime.now(IST).date()
        start_date = end_date - timedelta(days=days - 1)
        
        logging.info(f"📊 Sales Analytics Daily - days={days}, start={start_date}, end={end_date}, restaurant_id={restaurant_id}")
        
        with SessionLocal() as db:
            # Delegate to service layer
            results = pp_sales_analytics.get_daily_sales(db, start_date, end_date, days, restaurant_id)
            return jsonify(results)
            
    except Exception as e:
        logging.error(f"❌ Error in sales analytics daily: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/sales-analytics/payment-modes')
@login_required
def get_sales_analytics_payment_modes():
    """
    Get payment modes breakdown with proper restaurant filtering
    REFACTORED: Business logic moved to petpooja_integration.SalesAnalytics
    """
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        restaurant_id = request.args.get('restaurant_id', type=int)
        
        logging.info(f"📊 Payment Modes - restaurant_id={restaurant_id}")
        
        # Get current month start
        today = datetime.now(IST).date()
        start_date = datetime(today.year, today.month, 1).date()
        end_date = today
        
        with SessionLocal() as db:
            # Delegate to service layer
            results = pp_sales_analytics.get_payment_modes(db, start_date, end_date, restaurant_id)
            logging.info(f"✅ Returning {len(results)} payment mode records")
            return jsonify(results)
            
    except Exception as e:
        logging.error(f"❌ Error in payment modes: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/sales-analytics/order-types')
@login_required
def get_sales_analytics_order_types():
    """
    Get order types breakdown with proper restaurant filtering
    REFACTORED: Business logic moved to petpooja_integration.SalesAnalytics
    """
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        restaurant_id = request.args.get('restaurant_id', type=int)
        
        logging.info(f"📊 Order Types - restaurant_id={restaurant_id}")
        
        # Get current month start
        today = datetime.now(IST).date()
        start_date = datetime(today.year, today.month, 1).date()
        end_date = today
        
        with SessionLocal() as db:
            # Delegate to service layer
            results = pp_sales_analytics.get_order_types(db, start_date, end_date, restaurant_id)
            logging.info(f"✅ Returning {len(results)} order type records")
            return jsonify(results)
            
    except Exception as e:
        logging.error(f"❌ Error in order types: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/sales-analytics/top-items')
@login_required
def get_sales_analytics_top_items():
    """
    Get top selling items with proper restaurant filtering
    REFACTORED: Business logic moved to petpooja_integration.SalesAnalytics
    """
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        restaurant_id = request.args.get('restaurant_id', type=int)
        limit = request.args.get('limit', default=5, type=int)
        
        logging.info(f"📊 Top Items - restaurant_id={restaurant_id}, limit={limit}")
        
        # Get current month start
        today = datetime.now(IST).date()
        start_date = datetime(today.year, today.month, 1).date()
        end_date = today
        
        with SessionLocal() as db:
            # Delegate to service layer
            results = pp_sales_analytics.get_top_items(db, start_date, end_date, limit, restaurant_id)
            logging.info(f"✅ Returning {len(results)} top items")
            return jsonify(results)
            
    except Exception as e:
        logging.error(f"❌ Error in top items: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/analytics/footfall-conversion')
@login_required
def get_footfall_conversion():
    """
    Calculate footfall-to-sales conversion metrics
    
    REFACTORED: Thin wrapper - all business logic in petpooja_integration module
    """
    logging.info(f"Conversion API called - Session: {session.get('logged_in')}, User: {session.get('username')}")
    
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        # Parse parameters
        days = request.args.get('days', default=7, type=int)
        date_str = request.args.get('date')
        restaurant_id = request.args.get('restaurant_id', type=int)
        
        # Delegate to service
        with SessionLocal() as db:
            result = pp_conversion_analytics.get_footfall_conversion_analytics(
                db, days=days, date_str=date_str, restaurant_id=restaurant_id
            )
            return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error in footfall conversion analytics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/conversion-debug')
@login_required
def conversion_debug():
    """Debug endpoint to show raw data for troubleshooting conversion analytics"""
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        days = request.args.get('days', default=7, type=int)
        end_date = datetime.now(IST).date()
        start_date = end_date - timedelta(days=days - 1)
        
        with SessionLocal() as db:
            # Check footfall data
            footfall_count = db.query(HourlyFootfall).filter(
                HourlyFootfall.report_date >= start_date,
                HourlyFootfall.report_date <= end_date
            ).count()
            
            footfall_sample = db.query(HourlyFootfall).filter(
                HourlyFootfall.report_date >= start_date,
                HourlyFootfall.report_date <= end_date
            ).order_by(HourlyFootfall.report_date.desc(), HourlyFootfall.hour.desc()).limit(5).all()
            
            # Check PetPooja data using PetPoojaDatabase helper
            petpooja_count = PetPoojaDatabase.get_petpooja_count(db)
            petpooja_sample_raw = PetPoojaDatabase.get_petpooja_sample(db, limit=5)
            
            return jsonify({
                'debug_info': 'This shows what data is available for conversion analytics',
                'date_range': {
                    'start': start_date.strftime('%Y-%m-%d'),
                    'end': end_date.strftime('%Y-%m-%d'),
                    'days': days
                },
                'footfall': {
                    'total_records_in_range': footfall_count,
                    'sample_records': [
                        {
                            'date': f.report_date.strftime('%Y-%m-%d'),
                            'hour': f.hour,
                            'in_count': f.in_count,
                            'out_count': f.out_count,
                            'channel_id': f.channel_id
                        } for f in footfall_sample
                    ]
                },
                'petpooja_orders': {
                    'total_records': petpooja_count,
                    'sample_records': [
                        {
                            'id': row.id,
                            'webhook_received_at': str(row.created_at),
                            'order_time_used': str(row.effective_time),
                            'order_id': row.order_id,
                            'total': row.total,
                            'in_date_range': start_date <= row.effective_time.date() <= end_date if row.effective_time else False
                        } for row in petpooja_sample_raw
                    ]
                },
                'explanation': {
                    'footfall_data': 'Footfall from hourly_footfall table',
                    'order_data': 'Orders from petpooja_webhook_events table',
                    'note': 'Conversion analytics matches orders to footfall by date and hour. Both must exist for the same time period.'
                }
            })
    except Exception as e:
        logging.error(f"Error in conversion debug: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/staffing-recommendations')
@login_required
def get_staffing_recommendations():
    """
    Provides detailed staffing and inventory recommendations based on historical demand patterns
    Analyzes footfall + billing data to suggest optimal resource allocation per hour
    
    REFACTORED: Business logic moved to petpooja_integration.StaffingRecommendations
    """
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        # Get query parameters
        days = request.args.get('days', default=14, type=int)
        restaurant_id = request.args.get('restaurant_id', type=int)
        
        end_date = datetime.now(IST).date()
        start_date = end_date - timedelta(days=days - 1)
        
        with SessionLocal() as db:
            # Delegate to service layer
            result = pp_staffing_recommendations.calculate_recommendations(
                db, start_date, end_date, restaurant_id
            )
            
            # Add date range info
            result['date_range'] = {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'days': days
            }
            
            return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error in staffing recommendations: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/occupancy/today/<channel_id>')
@login_required
def get_occupancy_today(channel_id):
    """Get today's occupancy statistics"""
    if not db_connected: return jsonify({"error": "Database not connected"}), 500
    try:
        with SessionLocal() as db:
            today = datetime.now(IST).date()
            today_start = datetime.combine(today, datetime.min.time()).replace(tzinfo=IST)
            today_end = datetime.combine(today, datetime.max.time()).replace(tzinfo=IST)
            
            # Get all logs for today
            logs = db.query(OccupancyLog).filter(
                OccupancyLog.channel_id == channel_id,
                OccupancyLog.timestamp >= today_start,
                OccupancyLog.timestamp <= today_end
            ).all()
            
            if not logs:
                return jsonify({
                    "max_today": 0,
                    "avg_today": 0,
                    "current": 0
                })
            
            # Calculate statistics
            counts = [log.live_count for log in logs]
            max_today = max(counts) if counts else 0
            avg_today = round(sum(counts) / len(counts)) if counts else 0
            current = logs[-1].live_count if logs else 0
            
            return jsonify({
                "max_today": max_today,
                "avg_today": avg_today,
                "current": current
            })
    except Exception as e:
        logging.error(f"Error getting occupancy stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/occupancy/schedule/<channel_id>')
@login_required
def get_occupancy_schedule(channel_id):
    """Get occupancy schedule for a channel"""
    if not db_connected: return jsonify({"error": "Database not connected"}), 500
    try:
        with SessionLocal() as db:
            records = db.query(OccupancySchedule).filter_by(channel_id=channel_id).all()
            schedule = {}
            for record in records:
                if record.time_slot not in schedule:
                    schedule[record.time_slot] = {}
                schedule[record.time_slot][record.day_of_week] = record.required_count
            return jsonify(schedule)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/occupancy/schedule/<channel_id>', methods=['POST'])
@login_required
def update_occupancy_schedule(channel_id):
    """Update occupancy schedule for a channel"""
    if not db_connected: return jsonify({"error": "Database not connected"}), 500
    try:
        schedule_data = request.get_json()
        if not schedule_data:
            return jsonify({"error": "No schedule data provided"}), 400
        
        # Find the OccupancyMonitor processor for this channel
        processors = stream_processors.get(channel_id, [])
        om_processor = next((p for p in processors if isinstance(p, OccupancyMonitorProcessor)), None)
        
        if om_processor:
            success = om_processor.update_schedule(schedule_data)
            if success:
                return jsonify({"success": True, "message": "Schedule updated successfully"})
            else:
                return jsonify({"success": False, "error": "Failed to update schedule"}), 500
        else:
            return jsonify({"success": False, "error": "OccupancyMonitor processor not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/occupancy-logs/<channel_id>')
@login_required
def get_occupancy_logs(channel_id):
    """Download occupancy logs as CSV for a channel"""
    if not db_connected: return jsonify({"error": "Database not connected"}), 500
    try:
        with SessionLocal() as db:
            logs = db.query(OccupancyLog).filter_by(channel_id=channel_id).order_by(OccupancyLog.timestamp.desc()).limit(1000).all()
            
            # Generate CSV content
            def generate_csv():
                data = io.StringIO()
                writer = csv.writer(data)
                
                # Write header
                writer.writerow(['Timestamp', 'Time Slot', 'Day of Week', 'Live Count', 'Required Count', 'Status'])
                yield data.getvalue()
                data.seek(0)
                data.truncate(0)
                
                # Write log entries
                for log in logs:
                    writer.writerow([
                        log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        log.time_slot or '',
                        log.day_of_week or '',
                        log.live_count or 0,
                        log.required_count or 0,
                        log.status or ''
                    ])
                    yield data.getvalue()
                    data.seek(0)
                    data.truncate(0)
            
            # Get channel name for filename
            channel_name = channel_id
            processors = stream_processors.get(channel_id, [])
            if processors:
                for p in processors:
                    if hasattr(p, 'channel_name'):
                        channel_name = p.channel_name
                        break
            
            filename = f"occupancy_logs_{channel_name}_{datetime.now(IST).strftime('%Y%m%d')}.csv"
            filename = filename.replace(' ', '_').replace('/', '_')
            
            return Response(
                stream_with_context(generate_csv()),
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={filename}'}
            )
    except Exception as e:
        logging.error(f"Error generating occupancy logs CSV: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/occupancy/schedule/template')
@login_required
def download_schedule_template():
    """Download CSV template for schedule upload"""
    try:
        # Create template data
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        # Use non-padded hour keys to match processor expectations (H:00)
        hours = [f"{i}:00" for i in range(24)]
        
        # Create template with sample data
        template_data = []
        for day in days:
            row = [day]
            for hour in hours:
                if 8 <= int(hour.split(':')[0]) <= 20:  # Business hours
                    if day in ['Saturday', 'Sunday']:
                        row.append(1)  # Weekend: 1 person
                    else:
                        row.append(2)  # Weekday: 2 people
                else:
                    row.append(0)  # Off hours: 0 people
            template_data.append(row)
        
        # Create CSV
        df = pd.DataFrame(template_data, columns=['Day'] + hours)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        return Response(
            csv_content,
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=occupancy_schedule_template.csv'}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/restart_gunicorn', methods=['GET'])
def restart_gunicorn():
    """API endpoint to restart gunicorn.service - No authentication required"""
    try:
        logging.info("Restarting gunicorn.service via API request")
        
        # Use full paths to ensure commands are found
        systemctl_path = '/usr/bin/systemctl'
        sudo_path = '/usr/bin/sudo'
        
        # Check if commands exist
        if not os.path.exists(systemctl_path):
            # Try to find systemctl in PATH
            systemctl_path = 'systemctl'
        
        # Build command: try with sudo first, then without (if running as root)
        if os.path.exists(sudo_path):
            cmd = [sudo_path, systemctl_path, 'restart', 'gunicorn.service']
        elif os.path.exists('/bin/sudo'):
            cmd = ['/bin/sudo', systemctl_path, 'restart', 'gunicorn.service']
        else:
            # Try without sudo (if running as root or user has permissions)
            cmd = [systemctl_path, 'restart', 'gunicorn.service']
            logging.info("sudo not found, trying systemctl directly")
        
        # Execute systemctl restart command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            logging.info("gunicorn.service restarted successfully")
            return jsonify({
                "success": True,
                "message": "gunicorn.service restarted successfully",
                "output": result.stdout.strip() if result.stdout else "No output",
                "command_used": " ".join(cmd)
            })
        else:
            error_msg = result.stderr.strip() if result.stderr else result.stdout.strip() if result.stdout else 'Unknown error'
            logging.error(f"Failed to restart gunicorn.service: {error_msg}")
            return jsonify({
                "success": False,
                "error": f"Failed to restart gunicorn.service: {error_msg}",
                "returncode": result.returncode,
                "command_used": " ".join(cmd)
            }), 500
            
    except subprocess.TimeoutExpired:
        logging.error("Timeout while restarting gunicorn.service")
        return jsonify({
            "success": False,
            "error": "Timeout while restarting gunicorn.service"
        }), 500
    except FileNotFoundError as e:
        logging.error(f"Command not found while restarting gunicorn.service: {e}")
        return jsonify({
            "success": False,
            "error": f"Command not found: {str(e)}. Please ensure systemctl is available."
        }), 500
    except Exception as e:
        logging.error(f"Error restarting gunicorn.service: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/occupancy/schedule/upload/<channel_id>', methods=['POST'])
@login_required
def upload_schedule_file(channel_id):
    """Upload and process schedule file (CSV/Excel)"""
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Read file based on extension
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file.read()))
        else:
            return jsonify({"error": "Unsupported file format. Use CSV or Excel files."}), 400
        
        # Validate format
        if 'Day' not in df.columns:
            return jsonify({"error": "CSV must have 'Day' column"}), 400
        
        # Process schedule data
        schedule = {}
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # Helper to normalize timeslot keys to H:00 (e.g., '09:00' -> '9:00')
        def _normalize_timeslot(ts):
            try:
                h, m = str(ts).split(':', 1)
                return f"{int(h)}:{m}"
            except Exception:
                return str(ts)
        
        for _, row in df.iterrows():
            day = row['Day']
            if day not in days:
                continue
            for col in df.columns:
                if col != 'Day' and ':' in str(col):  # Time column
                    time_slot = _normalize_timeslot(col)
                    val = row[col]
                    required_count = int(val) if (pd.notna(val) and str(val).strip() != '') else 0
                    if time_slot not in schedule:
                        schedule[time_slot] = {}
                    schedule[time_slot][day] = required_count
        
        # Find the OccupancyMonitor processor for this channel
        processors = stream_processors.get(channel_id, [])
        om_processor = next((p for p in processors if isinstance(p, OccupancyMonitorProcessor)), None)
        
        if om_processor:
            success = om_processor.update_schedule(schedule)
            if success:
                return jsonify({
                    "success": True, 
                    "message": f"Schedule uploaded successfully for {len(schedule)} time slots",
                    "schedule": schedule
                })
            else:
                return jsonify({"success": False, "error": "Failed to update schedule"}), 500
        else:
            return jsonify({"success": False, "error": "OccupancyMonitor processor not found"}), 404
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@socketio.on('connect')
def handle_connect(): logging.info('Frontend client connected')

# ============================================================================
# RESTAURANT MANAGEMENT API ENDPOINTS (Phase 2)
# ============================================================================

@app.route('/api/restaurants', methods=['GET'])
def get_restaurants():
    """Get all restaurants (optionally filter by active status)"""
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        with SessionLocal() as db:
            active_only = request.args.get('active_only', 'true').lower() == 'true'
            
            query = db.query(Restaurant)
            if active_only:
                query = query.filter(Restaurant.is_active == True)
            
            restaurants = query.order_by(Restaurant.restaurant_name).all()
            
            result = []
            for r in restaurants:
                result.append({
                    'id': r.id,
                    'restaurant_code': r.restaurant_code,
                    'restaurant_name': r.restaurant_name,
                    'location': r.location,
                    'dvr_ip': r.dvr_ip,
                    'dvr_username': r.dvr_username,
                    'telegram_chat_id': r.telegram_chat_id,
                    'is_active': r.is_active,
                    'created_at': r.created_at.isoformat() if r.created_at else None
                })
            
            return jsonify({
                'success': True,
                'restaurants': result,
                'count': len(result)
            })
    except Exception as e:
        logging.error(f"Error fetching restaurants: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/restaurants/<int:restaurant_id>', methods=['GET'])
def get_restaurant(restaurant_id):
    """Get details of a specific restaurant"""
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        with SessionLocal() as db:
            restaurant = db.query(Restaurant).filter_by(id=restaurant_id).first()
            
            if not restaurant:
                return jsonify({'error': 'Restaurant not found'}), 404
            
            return jsonify({
                'success': True,
                'restaurant': {
                    'id': restaurant.id,
                    'restaurant_code': restaurant.restaurant_code,
                    'restaurant_name': restaurant.restaurant_name,
                    'location': restaurant.location,
                    'dvr_ip': restaurant.dvr_ip,
                    'dvr_username': restaurant.dvr_username,
                    'telegram_chat_id': restaurant.telegram_chat_id,
                    'is_active': restaurant.is_active,
                    'created_at': restaurant.created_at.isoformat() if restaurant.created_at else None,
                    'updated_at': restaurant.updated_at.isoformat() if restaurant.updated_at else None
                }
            })
    except Exception as e:
        logging.error(f"Error fetching restaurant {restaurant_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/restaurants/<int:restaurant_id>/cameras', methods=['GET'])
def get_restaurant_cameras(restaurant_id):
    """Get all cameras for a specific restaurant with their assigned apps"""
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        with SessionLocal() as db:
            # Verify restaurant exists
            restaurant = db.query(Restaurant).filter_by(id=restaurant_id).first()
            if not restaurant:
                return jsonify({'error': 'Restaurant not found'}), 404
            
            # Get all cameras for this restaurant
            cameras = db.query(Camera).filter_by(restaurant_id=restaurant_id).order_by(Camera.channel_number).all()
            
            result = []
            for cam in cameras:
                # Get apps assigned to this camera
                camera_apps = db.query(CameraApp).filter_by(camera_id=cam.id).all()
                apps = [{'app_name': ca.app_name, 'is_active': ca.is_active} for ca in camera_apps]
                
                result.append({
                    'id': cam.id,
                    'channel_id': cam.channel_id,
                    'channel_name': cam.channel_name,
                    'channel_number': cam.channel_number,
                    'rtsp_url': cam.rtsp_url,
                    'is_active': cam.is_active,
                    'apps': apps
                })
            
            return jsonify({
                'success': True,
                'restaurant': {
                    'id': restaurant.id,
                    'restaurant_name': restaurant.restaurant_name,
                    'location': restaurant.location
                },
                'cameras': result,
                'count': len(result)
            })
    except Exception as e:
        logging.error(f"Error fetching cameras for restaurant {restaurant_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/restaurants', methods=['POST'])
@login_required
def create_restaurant():
    """Create a new restaurant (admin only)"""
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['restaurant_code', 'restaurant_name']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        with SessionLocal() as db:
            # Check if restaurant already exists
            existing = db.query(Restaurant).filter_by(restaurant_code=data['restaurant_code']).first()
            if existing:
                return jsonify({'error': 'Restaurant with this code already exists'}), 400
            
            # Create new restaurant
            new_restaurant = Restaurant(
                restaurant_code=data['restaurant_code'],
                restaurant_name=data['restaurant_name'],
                location=data.get('location'),
                dvr_ip=data.get('dvr_ip'),
                dvr_username=data.get('dvr_username'),
                dvr_password=data.get('dvr_password'),
                telegram_chat_id=data.get('telegram_chat_id'),
                is_active=data.get('is_active', True)
            )
            
            db.add(new_restaurant)
            db.commit()
            db.refresh(new_restaurant)
            
            logging.info(f"Created new restaurant: {new_restaurant.restaurant_name} - {new_restaurant.location}")
            
            return jsonify({
                'success': True,
                'message': 'Restaurant created successfully',
                'restaurant': {
                    'id': new_restaurant.id,
                    'restaurant_code': new_restaurant.restaurant_code,
                    'restaurant_name': new_restaurant.restaurant_name,
                    'location': new_restaurant.location
                }
            }), 201
    except Exception as e:
        logging.error(f"Error creating restaurant: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/restaurants/<int:restaurant_id>', methods=['PUT'])
@login_required
def update_restaurant(restaurant_id):
    """Update restaurant details (admin only)"""
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        data = request.get_json()
        
        with SessionLocal() as db:
            restaurant = db.query(Restaurant).filter_by(id=restaurant_id).first()
            if not restaurant:
                return jsonify({'error': 'Restaurant not found'}), 404
            
            # Update fields if provided
            if 'restaurant_code' in data:
                restaurant.restaurant_code = data['restaurant_code']
            if 'restaurant_name' in data:
                restaurant.restaurant_name = data['restaurant_name']
            if 'location' in data:
                restaurant.location = data['location']
            if 'dvr_ip' in data:
                restaurant.dvr_ip = data['dvr_ip']
            if 'dvr_username' in data:
                restaurant.dvr_username = data['dvr_username']
            if 'dvr_password' in data:
                restaurant.dvr_password = data['dvr_password']
            if 'telegram_chat_id' in data:
                restaurant.telegram_chat_id = data['telegram_chat_id']
            if 'is_active' in data:
                restaurant.is_active = data['is_active']
            
            restaurant.updated_at = datetime.now()
            
            db.commit()
            
            logging.info(f"Updated restaurant {restaurant_id}: {restaurant.restaurant_name}")
            
            return jsonify({
                'success': True,
                'message': 'Restaurant updated successfully'
            })
    except Exception as e:
        logging.error(f"Error updating restaurant {restaurant_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/menu-time-popularity')
@login_required
def menu_time_popularity():
    """
    Analyzes menu item popularity by time of day (Morning, Afternoon, Evening)
    Uses REMOTE FastAPI server for sales data + LOCAL database for footfall
    Morning: 6 AM - 12 PM (hours 6-11)
    Afternoon: 12 PM - 5 PM (hours 12-16)
    Evening: 5 PM - 6 AM (hours 17-23, 0-5)
    
    REFACTORED: Business logic moved to petpooja_integration.TimeBasedMenu
    """
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        # Get query parameters
        days = request.args.get('days', 30, type=int)
        restaurant_id = request.args.get('restaurant_id', type=int)
        end_date_ist = datetime.now(IST).date()
        start_date_ist = end_date_ist - timedelta(days=days - 1)
        
        logging.info(f"📅 IST Date Range: {start_date_ist} to {end_date_ist}")
        logging.info(f"📅 Days requested: {days}")
        logging.info(f"🏪 Restaurant ID filter: {restaurant_id}")
        
        with SessionLocal() as db:
            # Delegate to service layer
            result = pp_time_based_menu.analyze_menu_popularity(
                db, start_date_ist, end_date_ist, days, restaurant_id
            )
            
            return jsonify(result)
            
    except Exception as e:
        logging.error(f"Error in menu_time_popularity: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500
# ============================================================================

_MODEL_CACHE = {}

def load_model(model_path: str, force_device=None):
    cache_key = f"{model_path}_{force_device or 'auto'}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return None
    try:
        model = YOLO(model_path)
        # Use force_device if provided, otherwise use DEVICE
        device = force_device if force_device is not None else DEVICE
        model.to(device)
        
        # Enable optimizations for CUDA only
        if device == 'cuda':
            try:
                model.fuse()
            except Exception:
                pass
        
        # Warmup
        try:
            import numpy as _np
            dummy = _np.zeros((640, 640, 3), dtype=_np.uint8)
            with torch.inference_mode():
                warmup_iterations = 3 if device == 'cuda' else 2
                for _ in range(warmup_iterations):
                    _ = model(dummy, conf=0.25, iou=0.45, imgsz=640, device=device, verbose=False)
        except Exception:
            pass
        
        logging.info(f"Loaded '{model_path}' on {device}")
        _MODEL_CACHE[cache_key] = model
        return model
    except Exception as e:
        logging.error(f"Failed to load model '{model_path}': {e}")
        return None

def start_streams():
    """Initialize all video stream processors
    
    First tries to load from database (new multi-restaurant structure).
    Falls back to rtsp_links.txt if database is empty (backward compatibility).
    """
    logging.info("=" * 70)
    logging.info("🚀 Initializing stream processors...")
    
    # Try loading from database first
    if db_connected:
        try:
            with SessionLocal() as db:
                # Check if we have any cameras in database
                camera_count = db.query(Camera).count()
                
                if camera_count > 0:
                    logging.info(f"📊 Loading cameras from database ({camera_count} cameras found)")
                    
                    # Get all active cameras with their restaurants and apps
                    cameras_data = db.query(Camera, Restaurant).\
                        join(Restaurant, Camera.restaurant_id == Restaurant.id).\
                        filter(Camera.is_active == True, Restaurant.is_active == True).all()
                    
                    logging.info(f"✅ Found {len(cameras_data)} active cameras")
                    
                    # Group cameras by RTSP URL for stream assignments
                    stream_assignments = defaultdict(lambda: {'apps': set(), 'name': '', 'id': '', 'restaurant': None})
                    
                    for camera, restaurant in cameras_data:
                        # Use existing RTSP URL or generate if needed
                        rtsp_url = camera.rtsp_url
                        channel_id = camera.channel_id
                        
                        # Get apps for this camera
                        camera_apps = db.query(CameraApp).filter_by(
                            camera_id=camera.id,
                            is_active=True
                        ).all()
                        
                        app_names = [ca.app_name for ca in camera_apps]
                        
                        stream_assignments[rtsp_url]['apps'].update(app_names)
                        stream_assignments[rtsp_url]['name'] = camera.channel_name
                        stream_assignments[rtsp_url]['id'] = channel_id
                        stream_assignments[rtsp_url]['restaurant'] = restaurant
                        
                        logging.info(f"  📹 {camera.channel_name} → {', '.join(app_names)}")
                    
                    # Start streams from database data
                    _start_streams_from_data(stream_assignments)
                    logging.info("=" * 70)
                    return
                else:
                    logging.info("No cameras in database, falling back to rtsp_links.txt")
        except Exception as e:
            logging.warning(f"Error loading from database, falling back to file: {e}")
            import traceback
            traceback.print_exc()
    
    # Fallback to rtsp_links.txt (backward compatibility)
    logging.info("📄 Loading cameras from rtsp_links.txt (legacy mode)")
    if not os.path.exists(RTSP_LINKS_FILE):
        logging.error(f"'{RTSP_LINKS_FILE}' not found and no database data available.")
        logging.info("=" * 70)
        return
    
    stream_assignments = defaultdict(lambda: {'apps': set(), 'name': '', 'id': '', 'restaurant': None})
    
    with open(RTSP_LINKS_FILE, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 3: continue
                link, name, app_names = parts[0], parts[1], parts[2:]
                channel_id = get_stable_channel_id(link)
                stream_assignments[link]['apps'].update(app_names)
                stream_assignments[link]['name'] = name
                stream_assignments[link]['id'] = channel_id
                logging.info(f"  📹 {name} → {', '.join(app_names)}")
    
    _start_streams_from_data(stream_assignments)
    logging.info("=" * 70)

def _start_streams_from_data(stream_assignments):
    """Helper function to start streams from parsed data (database or file)"""
    for link, assignment in stream_assignments.items():
        channel_id, channel_name, app_names = assignment['id'], assignment['name'], list(assignment['apps'])
        restaurant = assignment.get('restaurant')
        
        if channel_id not in stream_processors: 
            stream_processors[channel_id] = []
        
        active_app_names = app_names[:]
        
        # Start a shared FrameHub per link
        hub = FrameHub(link, channel_name)
        hub.start()
        atexit.register(hub.stop)
        if 'PeopleCounter' in active_app_names:
            model_obj = load_model(APP_TASKS_CONFIG['PeopleCounter']['model_path'], force_device='cpu')
            if model_obj:
                pc_processor = PeopleCounterProcessor(
                    link, channel_id, channel_name, model_obj, handle_detection, socketio,
                    db_session_factory=SessionLocal, db_connected=db_connected, 
                    timezone=IST, safe_track_persons_func=safe_track_persons
                )
                pc_processor.frame_hub = hub
                stream_processors[channel_id].append(pc_processor); pc_processor.start()
                logging.info(f"Started PeopleCounter for {channel_id} ({channel_name}).")
                atexit.register(pc_processor.shutdown); active_app_names.remove('PeopleCounter')
        if 'QueueMonitor' in active_app_names:
            model_obj = load_model(APP_TASKS_CONFIG['QueueMonitor']['model_path'], force_device='cpu')
            if model_obj:
                # Pass restaurant_id to QueueMonitor processor
                restaurant_id = restaurant.id if restaurant else None
                qm_processor = QueueMonitorProcessor(
                    rtsp_url=link,
                    channel_id=channel_id,
                    channel_name=channel_name,
                    model=model_obj,
                    restaurant_id=restaurant_id,
                    db_session_factory=SessionLocal,
                    socketio=socketio,
                    detection_handler=handle_detection,
                    notification_sender=send_telegram_notification,
                    tracking_function=safe_track_persons,
                    timezone=IST
                )
                qm_processor.frame_hub = hub
                stream_processors[channel_id].append(qm_processor); qm_processor.start()
                logging.info(f"Started QueueMonitor for {channel_id} ({channel_name}) - Restaurant ID: {restaurant_id}")
                atexit.register(qm_processor.shutdown); active_app_names.remove('QueueMonitor')
        if 'KitchenCompliance' in active_app_names:
            # KitchenComplianceProcessor loads its own unified model internally
            kc_processor = KitchenComplianceProcessor(
                link, channel_id, channel_name, SessionLocal, socketio, 
                send_telegram_notification, handle_detection
            )
            # KitchenComplianceProcessor should read frames from hub if implemented to do so.
            if hasattr(kc_processor, 'frame_hub'):
                kc_processor.frame_hub = hub
            stream_processors[channel_id].append(kc_processor)
            kc_processor.start()
            logging.info(f"Started KitchenCompliance for {channel_id} ({channel_name}).")
            atexit.register(kc_processor.shutdown)
            active_app_names.remove('KitchenCompliance')
        if 'OccupancyMonitor' in active_app_names:
            model_obj = load_model(APP_TASKS_CONFIG['OccupancyMonitor']['model_path'], force_device=DEVICE)
            if model_obj:
                om_processor = run_occupancy_monitor(
                    config={
                        'rtsp_url': link,
                        'channel_id': channel_id,
                        'channel_name': channel_name,
                        'model': model_obj,
                        'socketio': socketio,
                        'session_factory': SessionLocal,
                        'notification_sender': send_telegram_notification,
                        'OccupancyLog': OccupancyLog,
                        'OccupancySchedule': OccupancySchedule,
                        'timezone': IST,
                        'database_url': DATABASE_URL,
                        'device': DEVICE  # Use CUDA if available
                    },
                    frame_hub=hub
                )
                stream_processors[channel_id].append(om_processor)
                logging.info(f"Started OccupancyMonitor for {channel_id} ({channel_name}).")
                atexit.register(om_processor.shutdown)
                active_app_names.remove('OccupancyMonitor')
        
        if 'IdlePeopleViolation' in active_app_names:
            # IdlePeopleViolationProcessor loads its own model internally
            # Pass restaurant_id for store-specific behavior
            restaurant_id = restaurant.id if restaurant else None
            ipv_processor = IdlePeopleViolationProcessor(
                link, channel_id, channel_name, SessionLocal, socketio,
                send_telegram_notification, handle_detection, restaurant_id=restaurant_id
            )
            if hasattr(ipv_processor, 'frame_hub'):
                ipv_processor.frame_hub = hub
            stream_processors[channel_id].append(ipv_processor)
            ipv_processor.start()
            logging.info(f"Started IdlePeopleViolation for {channel_id} ({channel_name}) with restaurant_id={restaurant_id}.")
            atexit.register(ipv_processor.shutdown)
            active_app_names.remove('IdlePeopleViolation')
        
        if active_app_names:
            tasks_for_multi_model = []
            for app_name in active_app_names:
                config = APP_TASKS_CONFIG.get(app_name)
                if config and 'model_path' in config:
                    logging.info(f"Loading model for {app_name}: {config['model_path']}")
                    # Generic runs on CUDA for better performance
                    device_for_app = DEVICE if app_name == 'Generic' else 'cpu'
                    model_obj = load_model(config['model_path'], force_device=device_for_app)
                    if model_obj: 
                        tasks_for_multi_model.append({'app_name': app_name, 'model': model_obj, **config})
                        logging.info(f"✅ Model loaded for {app_name}. Target classes: {config.get('target_class_id', 'all')}, Confidence: {config.get('confidence', 0.5)}")
                    else: 
                        logging.warning(f"❌ Skipping '{app_name}' for {channel_id}; model failed to load.")
                else:
                    logging.warning(f"❌ No config found for {app_name}")
            if tasks_for_multi_model:
                multi_processor = MultiModelProcessor(link, channel_id, channel_name, tasks_for_multi_model, handle_detection)
                multi_processor.frame_hub = hub
                stream_processors[channel_id].append(multi_processor); multi_processor.start()
                task_names = [t['app_name'] for t in tasks_for_multi_model]
                logging.info(f"🚀 Started MultiModel for {channel_id} ({channel_name}) with tasks: {task_names}.")
                atexit.register(multi_processor.shutdown)
            else:
                logging.warning(f"⚠️ No tasks loaded for MultiModel on {channel_name}")


def restart_processor(processor_info):
    """Restart a failed processor"""
    channel_id, processor_type, channel_name = processor_info['channel_id'], processor_info['type'], processor_info['name']
    rtsp_url = processor_info.get('rtsp_url')
    frame_hub = processor_info.get('frame_hub')
    
    logging.warning(f"Attempting to restart {processor_type} for {channel_name} ({channel_id})")
    
    try:
        # Find and stop the old processor
        processors = stream_processors.get(channel_id, [])
        old_processor = None
        for p in processors:
            if isinstance(p, processor_type):
                old_processor = p
                break
        
        if old_processor:
            old_processor.is_running = False
            old_processor.shutdown()
            # Wait a bit for it to stop
            time.sleep(2)
            processors.remove(old_processor)
        
        # Create new processor instance
        if processor_type == PeopleCounterProcessor:
            model_obj = load_model(APP_TASKS_CONFIG['PeopleCounter']['model_path'])
            if model_obj:
                new_processor = PeopleCounterProcessor(
                    rtsp_url, channel_id, channel_name, model_obj, handle_detection, socketio,
                    db_session_factory=SessionLocal, db_connected=db_connected,
                    timezone=IST, safe_track_persons_func=safe_track_persons
                )
                new_processor.frame_hub = frame_hub
                new_processor.start()
                processors.append(new_processor)
                logging.info(f"Successfully restarted PeopleCounter for {channel_name}")
                return True
        elif processor_type == QueueMonitorProcessor:
            model_obj = load_model(APP_TASKS_CONFIG['QueueMonitor']['model_path'])
            if model_obj:
                new_processor = QueueMonitorProcessor(rtsp_url, channel_id, channel_name, model_obj)
                new_processor.frame_hub = frame_hub
                new_processor.start()
                processors.append(new_processor)
                logging.info(f"Successfully restarted QueueMonitor for {channel_name}")
                return True
        elif processor_type == MultiModelProcessor:
            # MultiModelProcessor restart would need tasks info
            logging.warning(f"Cannot auto-restart MultiModelProcessor - requires task configuration")
            return False
            
    except Exception as e:
        logging.error(f"Failed to restart {processor_type} for {channel_name}: {e}")
        return False
    

# Global flag to track if initialization has been done
_initialized = False

def initialize_app():
    """Initialize the application (scheduler, processors, etc.)"""
    global _initialized
    if _initialized:
        logging.info("Application already initialized, skipping...")
        return  # Already initialized
    
    try:
        logging.info("Starting application initialization...")
        
        # Initialize database
        initialize_database()
        
        # Initialize Kitchen Compliance tables (separate Base object)
        if db_connected:
            try:
                from kitchen_compliance_monitor import KitchenComplianceProcessor
                KitchenComplianceProcessor.initialize_tables(engine)
                logging.info("Kitchen Compliance tables initialized")
            except Exception as e:
                logging.error(f"Failed to initialize Kitchen tables: {e}")
            
            # Initialize Idle People Violation tables
            try:
                from idle_people_violation import IdlePeopleViolationProcessor
                IdlePeopleViolationProcessor.initialize_tables(engine)
                logging.info("Idle People Violation tables initialized")
            except Exception as e:
                logging.error(f"Failed to initialize Idle People Violation tables: {e}")
        
        # Start the scheduler if database is connected
        if db_connected:
            scheduler = BackgroundScheduler(timezone=str(IST))
            # scheduler.add_job(log_queue_counts, 'interval', minutes=5)  # disabled queue_logs periodic write
            scheduler.start()
            atexit.register(lambda: scheduler.shutdown())
            logging.info("Scheduler started successfully")
        else:
            logging.warning("Database not connected, scheduler not started")
        
        # Start all stream processors
        start_streams()
        
        _initialized = True
        logging.info("✓ Application initialized successfully - processors and scheduler started")
        
        # Log processor status
        total_processors = sum(len(procs) for procs in stream_processors.values())
        logging.info(f"Total processors started: {total_processors} across {len(stream_processors)} channels")
        
    except Exception as e:
        logging.error(f"Error during application initialization: {e}", exc_info=True)
        raise

# Initialize when module is imported (for gunicorn)
# Use threading lock to prevent multiple initializations in multi-worker setup
_init_lock = threading.Lock()
_initialization_thread = None

def _ensure_initialized(background=True):
    """Ensure initialization happens, with locking for multi-worker safety
    
    Args:
        background: If True, run initialization in background thread (non-blocking)
                   If False, run synchronously (blocking)
    """
    global _initialized, _initialization_thread
    
    with _init_lock:
        if _initialized:
            return True  # Already initialized
        
        if _initialization_thread is not None and _initialization_thread.is_alive():
            return False  # Initialization in progress
        
        def _init_wrapper():
            """Wrapper to run initialization in background"""
            try:
                initialize_app()
            except Exception as e:
                logging.error(f"Background initialization error: {e}", exc_info=True)
        
        if background:
            # Run initialization in background thread so server can start responding
            _initialization_thread = threading.Thread(target=_init_wrapper, daemon=True, name="AppInitializer")
            _initialization_thread.start()
            logging.info("Initialization started in background thread - server will respond immediately")
            return False  # Still initializing
        else:
            # Run synchronously (for direct run mode)
            _init_wrapper()
            return _initialized

# For gunicorn: initialize in background (non-blocking)
# For direct run: initialize synchronously in __main__
_ensure_initialized(background=True)

if __name__ == "__main__":
    # For direct run, initialize synchronously (blocking until ready)
    logging.info("Direct run mode - initializing synchronously...")
    _ensure_initialized(background=False)
    
    logging.info("Starting Flask-SocketIO server on http://0.0.0.0:5001")
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)