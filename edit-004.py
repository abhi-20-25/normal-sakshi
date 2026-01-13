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
# FORCE CPU MODE - Disable CUDA to avoid GPU errors on server
DEVICE = 'cpu'
logging.info("🚫 CUDA DISABLED - Running in CPU-only mode for stability")

# Keep CUDA code commented out for future use
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# if DEVICE == 'cuda':
#     torch.backends.cudnn.benchmark = True
#     try:
#         torch.set_float32_matmul_precision('high')
#     except Exception:
#         pass

# --- Frame Downscale Settings ---
# Reduce resolution early in the pipeline to speed up processing/streaming
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
pp_sales_analytics, pp_conversion_analytics, pp_time_based_menu, pp_promotion_effectiveness = create_petpooja_services()
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
    'Generic': {'model_path': 'models/02_01_2026_teatost_best.pt', 'target_class_id': [1, 2, 3, 4, 5, 6, 7, 8], 'confidence': 0.3, 'is_gif': False},
    'PeopleCounter': {'model_path': 'models/yolo11n.pt' , 'confidence': 0.15},
    'QueueMonitor': {'model_path': 'models/yolo11n.pt' , 'confidence': 0.15},
    'KitchenCompliance': {'model_path': 'models/02_01_2026_teatost_best.pt', 'confidence': 0.3},  # Unified model with person detection
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
# THIS IS NOW A FALLBACK if no ROI is in the database.
QUEUE_MONITOR_ROI_CONFIG = {
    "Checkout Queue": {
        "roi_points": [[0.5549999952316285, 0.5744444105360244], [0.4456249952316284, 0.5272221883138021], [0.3081249952316284, 0.3105555216471354], [0.08624999523162842, 0.4272221883138021], [0.19249999523162842, 0.7938888549804688]],
        "secondary_roi_points": [[0.5924999952316284, 0.5355555216471354], [0.49874999523162844, 0.502222188313802], [0.3487499952316284, 0.31333329942491317], [0.38156249523162844, 0.3105555216471354], [0.3940624952316284, 0.2883332994249132], [0.5003124952316285, 0.26888885498046877], [0.6721874952316285, 0.4633332994249132]],
    }
}
QUEUE_DWELL_TIME_SEC = 0.05        # How long a person must stay in queue to be counted (reduced to 0.05 seconds)
QUEUE_SCREENSHOT_DWELL_TIME_SEC = 5.0  # How long a person must stay in queue to trigger screenshot (5 seconds)
QUEUE_ALERT_THRESHOLD = 3          # Regular alert: 2+ people with NO cashier
QUEUE_OVERQUEUE_THRESHOLD = 4      # Overqueue alert: 4+ people WITH cashier
QUEUE_HIGH_COUNT_THRESHOLD = 3     # Screenshot threshold: queue count > 3
QUEUE_COUNTER_PERSISTENCE_SEC = 8.0  # How long to keep counter as "occupied" after last detection (8 seconds)
QUEUE_ALERT_COOLDOWN_SEC = 6      # 60-second cooldown between alerts

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
    global db_connected, engine, SessionLocal
    try:
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
        db_connected = True
        logging.info("Database connection successful.")
        return True
    except OperationalError as e:
        logging.error(f"Database connection failed: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during DB init: {e}")
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
        - Cap_present (1) vs Without_cap (2)
        - With_apron (3) vs Without_apron (4)
        - With_gloves (5) vs Without_gloves (6)
        - Using_phone (7) always triggers as violation
        - Without_uniform (8) triggers as violation
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
            
            # Without_cap (2) vs Cap_present (1) - keep higher confidence
            if class_id == 2 and 1 in max_conf_by_class:
                if max_conf_by_class[1] > conf:  # Cap_present has higher confidence
                    should_keep = False
            elif class_id == 1 and 2 in max_conf_by_class:
                if max_conf_by_class[2] > conf:  # Without_cap has higher confidence
                    should_keep = False
            
            # Without_apron (4) vs With_apron (3) - keep higher confidence
            elif class_id == 4 and 3 in max_conf_by_class:
                if max_conf_by_class[3] > conf:  # With_apron has higher confidence
                    should_keep = False
            elif class_id == 3 and 4 in max_conf_by_class:
                if max_conf_by_class[4] > conf:  # Without_apron has higher confidence
                    should_keep = False
            
            # Without_gloves (6) vs With_gloves (5) - keep higher confidence
            elif class_id == 6 and 5 in max_conf_by_class:
                if max_conf_by_class[5] > conf:  # With_gloves has higher confidence
                    should_keep = False
            elif class_id == 5 and 6 in max_conf_by_class:
                if max_conf_by_class[6] > conf:  # Without_gloves has higher confidence
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

                # CPU-only mode
                try:
                    with torch.inference_mode():
                        results = task['model'](
                            frame,
                            device='cpu',
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
                        if det['class_id'] == 7:  # Using_phone
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
                        phone_detections = [det for det in final_detections if det['class_id'] == 7]
                        if phone_detections:
                            logging.warning(f"📱 PHONE DETECTED in {self.channel_name}! Confidence: {phone_detections[0]['confidence']:.2f}")
                    
                    # Define class sets and colors
                    violation_classes = {2, 4, 6, 7, 8}  # Without_cap, Without_apron, Without_gloves, Using_phone, Without_uniform
                    compliance_classes = {1, 3, 5}  # Cap_present, With_apron, With_gloves
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

class PeopleCounterProcessor(threading.Thread):
    def __init__(self, rtsp_url, channel_id, channel_name, model, detection_callback, socketio):
        super().__init__()
        self.rtsp_url, self.channel_id, self.model, self.detection_callback = rtsp_url, channel_id, model, detection_callback
        self.channel_name, self.app_name = channel_name, "PeopleCounter"
        self.socketio = socketio
        self.is_running, self.lock = True, threading.Lock()
        
        # LINE CROSSING APPROACH - Simple & Reliable!
        self.previous_centroids = []  # List of (x, y) from previous frame
        self.counting_line_position = 0.38  # Default: Line at 38% (LEFT=0-38%, RIGHT=38-100%)
        self.cooldown_zones = {}  # {(approx_x, approx_y): timestamp} to prevent double counting
        self.cooldown_duration = 0.8  # 800ms cooldown per zone
        
        # Load counting line position from database
        self._load_line_position_from_db()
        
        self.counts = {'in': 0, 'out': 0}
        self.current_hour = datetime.now(IST).hour
        self.tracking_date = datetime.now(IST).date()
        self.latest_frame = None
        
        self._load_initial_counts()

        hourly_data = self._get_hourly_data()
        self.socketio.emit('count_update', {
            'channel_id': self.channel_id, 
            'in_count': self.counts['in'], 
            'out_count': self.counts['out'],
            'hourly_data': hourly_data
        })

    def update_line_position(self, new_position):
        """Update counting line position from ROI editor"""
        with self.lock:
            self.counting_line_position = new_position
            logging.info(f"🎯 PeopleCounter {self.channel_name} line position updated to {new_position*100:.0f}%")
    
    def stop(self): self.is_running = False
    def shutdown(self):
        logging.info(f"Shutting down PeopleCounter for {self.channel_name}. Saving final counts...")
        self._update_and_log_counts()
        self.is_running = False

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                placeholder = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(placeholder, 'Connecting...', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (201, 209, 217), 2)
                _, jpeg = cv2.imencode('.jpg', placeholder); return jpeg.tobytes()
            _, jpeg = cv2.imencode('.jpg', self.latest_frame); return jpeg.tobytes()

    def _get_hourly_data(self):
        """Get today's hourly IN counts for the bar chart"""
        hourly_data = [0] * 24  # Initialize 24 hours with 0
        if not db_connected: return hourly_data
        
        with SessionLocal() as db:
            try:
                today_ist = datetime.now(IST).date()
                records = db.query(HourlyFootfall).filter_by(
                    channel_id=self.channel_id, 
                    report_date=today_ist
                ).all()
                
                for record in records:
                    if 0 <= record.hour < 24:
                        hourly_data[record.hour] = record.in_count
            except Exception as e:
                logging.error(f"Failed to fetch hourly data: {e}")
        
        return hourly_data

    def _load_line_position_from_db(self):
        """Load counting line position from database"""
        if not db_connected: return
        try:
            with SessionLocal() as db:
                roi_record = db.query(RoiConfig).filter_by(channel_id=self.channel_id, app_name='PeopleCounter').first()
                if roi_record and roi_record.roi_points:
                    points = json.loads(roi_record.roi_points)
                    if 'line_position' in points:
                        self.counting_line_position = points['line_position']
                        logging.info(f"✅ Loaded counting line position: {self.counting_line_position*100:.0f}% for {self.channel_name}")
                    else:
                        logging.info(f"Using default counting line position: 45% for {self.channel_name}")
                else:
                    logging.info(f"No saved line position found, using default: 45% for {self.channel_name}")
        except Exception as e:
            logging.error(f"Error loading line position: {e}. Using default 45%")

    def _load_initial_counts(self):
        if not db_connected: return
        with SessionLocal() as db:
            try:
                today_ist = datetime.now(IST).date()
                self.tracking_date = today_ist
                record = db.query(DailyFootfall).filter_by(channel_id=self.channel_id, report_date=today_ist).first()
                if record: self.counts = {'in': record.in_count, 'out': record.out_count}
                else: self._reset_counts_for_new_day(db, today_ist)
            except Exception as e: logging.error(f"Failed to load initial counts: {e}")

    def _reset_counts_for_new_day(self, db, new_date):
        self.counts = {'in': 0, 'out': 0}
        self.tracking_date = new_date
        db.add(DailyFootfall(channel_id=self.channel_id, report_date=new_date, in_count=0, out_count=0))
        db.commit()

    def _update_and_log_counts(self):
        """Update daily counts in database"""
        if not db_connected: return
        with SessionLocal() as db, self.lock:
            try:
                db.query(DailyFootfall).filter_by(channel_id=self.channel_id, report_date=self.tracking_date).update({'in_count': self.counts['in'], 'out_count': self.counts['out']})
                db.commit()
            except Exception as e:
                logging.error(f"Error updating daily counts in DB: {e}"); db.rollback()
    
    def _update_hourly_count_realtime(self, count_type):
        """Update hourly count in database in real-time when in/out is detected"""
        if not db_connected: return
        current_time = datetime.now(IST)
        current_hour_ist = current_time.hour
        current_date_ist = current_time.date()
        
        # Check if hour changed - if so, update current_hour
        if current_hour_ist != self.current_hour:
            self.current_hour = current_hour_ist
        
        # Check if day changed - if so, update tracking_date
        if current_date_ist != self.tracking_date:
            self.tracking_date = current_date_ist
        
        with SessionLocal() as db:
            try:
                # Increment hourly count for current hour
                stmt = text("""
                    INSERT INTO hourly_footfall (channel_id, report_date, hour, in_count, out_count)
                    VALUES (:cid, :rdate, :hour, :inc, :outc)
                    ON CONFLICT (channel_id, report_date, hour)
                    DO UPDATE SET 
                        in_count = hourly_footfall.in_count + EXCLUDED.in_count,
                        out_count = hourly_footfall.out_count + EXCLUDED.out_count;
                """)
                inc = 1 if count_type == 'in' else 0
                outc = 1 if count_type == 'out' else 0
                db.execute(stmt, {
                    'cid': self.channel_id,
                    'rdate': self.tracking_date,
                    'hour': self.current_hour,
                    'inc': inc,
                    'outc': outc
                })
                db.commit()
                logging.info(f"PeopleCounter {self.channel_name}: Updated hourly count in real-time - Hour {self.current_hour:02d}:00, {count_type.upper()}+1")
            except Exception as e:
                logging.error(f"Error updating hourly count in DB: {e}"); db.rollback()

    def _check_for_new_day(self):
        current_date_ist = datetime.now(IST).date()
        current_hour_ist = datetime.now(IST).hour
        if current_date_ist > self.tracking_date:
            logging.info("New day detected. Resetting people counter.")
            self._update_and_log_counts()
            with SessionLocal() as db:
                self._reset_counts_for_new_day(db, current_date_ist)
                self.current_hour = current_hour_ist

    def run(self):
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_running:
            try:
                self._check_for_new_day()
                frame = getattr(self, 'frame_hub', None).get_latest() if hasattr(self, 'frame_hub') else None
                if frame is None:
                    time.sleep(0.01)
                    continue
                # Ensure we have a fresh copy to prevent any cross-contamination with other processors
                frame = frame.copy()
                
                # Apply adaptive histogram equalization for better detection in varying lighting
                # Convert to LAB color space and equalize the L channel
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l_equalized = clahe.apply(l)
                enhanced_frame = cv2.merge([l_equalized, a, b])
                enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_LAB2BGR)
                
                # Use YOLO predict (no tracking needed for line-crossing!)
                results = safe_track_persons(self.model, enhanced_frame, conf=0.20, iou=0.5, processor_name=f"{self.channel_name}-PeopleCounter")
                consecutive_errors = 0  # Reset on successful frame
                r0 = results[0] if (results and len(results) > 0) else None
                
                # LINE CROSSING DETECTION - Counting line at 55%
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]
                counting_line_x = int(frame_width * self.counting_line_position)  # 55% line
                
                if r0 is not None and getattr(r0, 'boxes', None) is not None:
                    boxes_xyxy = r0.boxes.xyxy.cpu()
                    boxes_conf = r0.boxes.conf.cpu()
                    
                    # Calculate frame dimensions for size filtering
                    frame_area = frame_width * frame_height
                    min_box_area = frame_area * 0.003  # Minimum 0.3% of frame area
                    max_box_area = frame_area * 0.9    # Maximum 90% of frame area
                    min_confidence = 0.20  # Balanced threshold
                    
                    # Collect current frame centroids
                    current_centroids = []
                    
                    for i, box in enumerate(boxes_xyxy):
                        # Get box dimensions
                        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                        box_width = float(x2 - x1)
                        box_height = float(y2 - y1)
                        box_area = box_width * box_height
                        confidence = float(boxes_conf[i])
                        
                        # Filter: person-shaped, valid size, high confidence
                        aspect_ratio = box_height / box_width if box_width > 0 else 0
                        is_person_shaped = 1.2 <= aspect_ratio <= 4.0
                        is_valid_size = min_box_area <= box_area <= max_box_area
                        is_confident = confidence >= min_confidence
                        
                        if is_person_shaped and is_valid_size and is_confident:
                            # Calculate centroid (center point)
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            current_centroids.append((center_x, center_y))
                    
                    # LINE CROSSING DETECTION
                    current_time = time.time()
                    
                    # Clean up old cooldown zones
                    expired_zones = [zone for zone, timestamp in self.cooldown_zones.items() 
                                    if current_time - timestamp > self.cooldown_duration]
                    for zone in expired_zones:
                        del self.cooldown_zones[zone]
                    
                    # For each current centroid, find closest match in previous frame
                    for curr_x, curr_y in current_centroids:
                        # Find closest previous centroid (within 100px threshold)
                        closest_prev = None
                        min_distance = float('inf')
                        
                        for prev_x, prev_y in self.previous_centroids:
                            distance = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2)**0.5
                            if distance < min_distance and distance < 100:  # Max 100px movement per frame
                                min_distance = distance
                                closest_prev = (prev_x, prev_y)
                        
                        # Check if line was crossed
                        if closest_prev is not None:
                            prev_x, prev_y = closest_prev
                            
                            # Check cooldown zone (approximate location to prevent double-counting)
                            zone_key = (int(curr_x / 80) * 80, int(curr_y / 80) * 80)  # 80px grid
                            if zone_key in self.cooldown_zones:
                                continue  # Skip - recently counted in this area
                            
                            # Detect crossing: previous position on one side, current on other
                            crossed_left_to_right = prev_x < counting_line_x and curr_x >= counting_line_x
                            crossed_right_to_left = prev_x >= counting_line_x and curr_x < counting_line_x
                            
                            if crossed_left_to_right:
                                # Person entered (LEFT → RIGHT)
                                with self.lock:
                                    self.counts['in'] += 1
                                self._update_hourly_count_realtime('in')
                                self._update_and_log_counts()
                                self.cooldown_zones[zone_key] = current_time
                                logging.info(f"✅ IN: Person crossed line LEFT→RIGHT at ({curr_x},{curr_y}). Total IN: {self.counts['in']}")
                            
                            elif crossed_right_to_left:
                                # Person exited (RIGHT → LEFT)
                                with self.lock:
                                    self.counts['out'] += 1
                                self._update_hourly_count_realtime('out')
                                self._update_and_log_counts()
                                self.cooldown_zones[zone_key] = current_time
                                logging.info(f"✅ OUT: Person crossed line RIGHT→LEFT at ({curr_x},{curr_y}). Total OUT: {self.counts['out']}")
                    
                    # Update previous centroids for next frame
                    self.previous_centroids = current_centroids
                
                # Create annotated frame with person bounding boxes only
                annotated_frame = frame.copy()
                
                # Counting line visualization removed for cleaner view
                # frame_width = frame.shape[1]
                # frame_height = frame.shape[0]
                # counting_line_x = int(frame_width * self.counting_line_position)
                # cv2.line(annotated_frame, (counting_line_x, 0), (counting_line_x, frame_height), (0, 0, 255), 3)
                # cv2.putText(annotated_frame, "COUNTING LINE", (counting_line_x + 10, 30),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # cv2.putText(annotated_frame, "IN ->", (counting_line_x - 80, frame_height // 2),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # cv2.putText(annotated_frame, "<- OUT", (counting_line_x + 10, frame_height // 2),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Person bounding boxes removed for cleaner view
                # if r0 is not None and getattr(r0, 'boxes', None) is not None:
                #     boxes = r0.boxes
                #     frame_area = frame.shape[0] * frame.shape[1]
                #     min_box_area = frame_area * 0.003
                #     max_box_area = frame_area * 0.9
                #     min_confidence = 0.20
                #     
                #     for i in range(len(boxes)):
                #         box = boxes.xyxy[i].cpu().numpy()
                #         x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                #         conf = float(boxes.conf[i].cpu())
                #         
                #         box_width = x2 - x1
                #         box_height = y2 - y1
                #         box_area = box_width * box_height
                #         aspect_ratio = box_height / box_width if box_width > 0 else 0
                #         
                #         is_person_shaped = 1.2 <= aspect_ratio <= 4.0
                #         is_valid_size = min_box_area <= box_area <= max_box_area
                #         is_confident = conf >= min_confidence
                #         
                #         # Draw only valid person boxes
                #         if is_person_shaped and is_valid_size and is_confident:
                #             cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #             label = f"Person {conf:.2f}"
                #             cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                with self.lock: self.latest_frame = annotated_frame.copy()
                hourly_data = self._get_hourly_data()
                socketio.emit('count_update', {
                    'channel_id': self.channel_id, 
                    'in_count': self.counts['in'], 
                    'out_count': self.counts['out'],
                    'hourly_data': hourly_data
                })
            except RuntimeError as e:
                logging.error(f"Runtime error in PeopleCounter {self.channel_name} run loop: {e}. Error count: {consecutive_errors}")
                
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logging.error(f"Too many consecutive errors for {self.channel_name}. Pausing for recovery...")
                    time.sleep(10)  # Pause for recovery
                    consecutive_errors = 0
                else:
                    time.sleep(2)  # Short pause before retry
            except Exception as e:
                logging.error(f"Unexpected error in PeopleCounter {self.channel_name}: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logging.error(f"Too many consecutive errors for {self.channel_name}. Pausing...")
                    time.sleep(10)
                    consecutive_errors = 0
                else:
                    time.sleep(1)
        # No cap to release when using FrameHub

class QueueMonitorProcessor(threading.Thread):
    def __init__(self, rtsp_url, channel_id, channel_name, model, restaurant_id=None):
        super().__init__(name=channel_name)
        self.rtsp_url = rtsp_url
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.model = model
        self.restaurant_id = restaurant_id  # Store restaurant ID
        self.is_running = True
        self.lock = threading.Lock()
        self.latest_frame = None
        self.queue_tracker = defaultdict(lambda: {'entry_time': 0})
        self.current_queue_count = 0
        self.secondary_queue_tracker = defaultdict(lambda: {'entry_time': 0})
        self.current_secondary_count = 0
        self.last_counter_detection_time = 0  # Track last time someone was detected in counter area
        self.last_alert_time = 0
        self.last_overqueue_time = 0  # Track overqueue alerts separately
        self.last_screenshot_time = 0  # Track screenshot alerts to avoid spam
        self.screenshot_cooldown = 10  # 10 seconds cooldown between screenshots
        self.roi_poly = Polygon([])
        self.secondary_roi_poly = Polygon([])
        self._load_roi_from_db()
        
        # Cache for queue stats to avoid frequent DB queries
        self.cached_served_today = 0
        self.cached_peak_count = 0
        self.last_stats_update = 0
        self.stats_update_interval = 30  # Update stats every 30 seconds

    # def _load_roi_from_db(self):
    #     with SessionLocal() as db:
    #         roi_record = db.query(RoiConfig).filter_by(channel_id=self.channel_id, app_name='QueueMonitor').first()
    #         if roi_record and roi_record.roi_points:
    #             try:
    #                 points = json.loads(roi_record.roi_points)
    #                 self.roi_poly = Polygon(points.get("main", []))
    #                 self.secondary_roi_poly = Polygon(points.get("secondary", []))
    #                 logging.info(f"Loaded custom ROI for QueueMonitor {self.channel_name} from DB.")
    #             except (json.JSONDecodeError, TypeError):
    #                 logging.error("Failed to parse ROI JSON from DB. Using fallback.")
    #                 self._use_fallback_roi()
    #         else:
    #             logging.warning(f"No custom ROI in DB for QueueMonitor {self.channel_name}. Using fallback.")
    #             self._use_fallback_roi()
    def _load_roi_from_db(self):
        """Load ROI from database first, fallback to hardcoded values if not found
        
        Priority: Database ROI > Hardcoded ROI
        This ensures server and local use the same ROI from database.
        """
        # Always try to load from database first
        logging.info(f"🏪 Restaurant ID {self.restaurant_id} - attempting to load ROI from database for {self.channel_name}")
        with SessionLocal() as db:
            roi_record = db.query(RoiConfig).filter_by(channel_id=self.channel_id, app_name='QueueMonitor').first()
            if roi_record and roi_record.roi_points:
                try:
                    points = json.loads(roi_record.roi_points) if isinstance(roi_record.roi_points, str) else roi_record.roi_points
                    self.normalized_main_roi = points.get("main", [])
                    self.normalized_secondary_roi = points.get("secondary", [])
                    
                    # Initialize with empty polygons - will be converted to pixels in run() method
                    self.roi_poly = Polygon([])
                    self.secondary_roi_poly = Polygon([])
                    
                    logging.info(f"✅ Loaded custom ROI for QueueMonitor {self.channel_name} from database.")
                    logging.info(f"   Main ROI: {len(self.normalized_main_roi)} points")
                    logging.info(f"   Secondary ROI: {len(self.normalized_secondary_roi)} points")
                    return
                except (json.JSONDecodeError, TypeError) as e:
                    logging.error(f"Failed to parse ROI JSON from DB: {e}. Using fallback.")
                    self._use_fallback_roi()
            else:
                logging.warning(f"No custom ROI in DB for QueueMonitor {self.channel_name}. Using hardcoded fallback.")
                self._use_fallback_roi()

    def _use_fallback_roi(self):
        fallback_config = QUEUE_MONITOR_ROI_CONFIG.get(self.channel_name, {})
        # If channel name doesn't match, try to use the first available config
        if not fallback_config and QUEUE_MONITOR_ROI_CONFIG:
            first_key = list(QUEUE_MONITOR_ROI_CONFIG.keys())[0]
            fallback_config = QUEUE_MONITOR_ROI_CONFIG[first_key]
            logging.info(f"No ROI config found for '{self.channel_name}', using first available config: '{first_key}'")
        
        # Store normalized coordinates for later conversion to pixels
        self.normalized_main_roi = fallback_config.get("roi_points", [])
        self.normalized_secondary_roi = fallback_config.get("secondary_roi_points", [])
        
        # Log what we got
        if self.normalized_main_roi:
            logging.info(f"Loaded main ROI with {len(self.normalized_main_roi)} points for {self.channel_name}")
        else:
            logging.warning(f"No main ROI points found for {self.channel_name}")
        if self.normalized_secondary_roi:
            logging.info(f"Loaded secondary ROI with {len(self.normalized_secondary_roi)} points for {self.channel_name}")
        else:
            logging.warning(f"No secondary ROI points found for {self.channel_name}")
        
        # Initialize with empty polygons - will be converted to pixels in run() method
        self.roi_poly = Polygon([])
        self.secondary_roi_poly = Polygon([])

    def update_roi(self, new_roi_points):
        with self.lock:
            try:
                self.normalized_main_roi = new_roi_points.get("main", [])
                self.normalized_secondary_roi = new_roi_points.get("secondary", [])
                
                # Force immediate polygon update
                if self.latest_frame is not None:
                    h, w = self.latest_frame.shape[:2]
                    
                    # Update main ROI polygon
                    if self.normalized_main_roi and len(self.normalized_main_roi) >= 3:
                        pixel_coords = [(int(p[0]*w), int(p[1]*h)) for p in self.normalized_main_roi]
                        self.roi_poly = Polygon(pixel_coords)
                        if not self.roi_poly.is_valid:
                            self.roi_poly = self.roi_poly.buffer(0)
                        logging.info(f"✅ Updated main ROI: {len(pixel_coords)} points")
                    
                    # Update secondary ROI polygon
                    if self.normalized_secondary_roi and len(self.normalized_secondary_roi) >= 3:
                        pixel_coords = [(int(p[0]*w), int(p[1]*h)) for p in self.normalized_secondary_roi]
                        self.secondary_roi_poly = Polygon(pixel_coords)
                        if not self.secondary_roi_poly.is_valid:
                            self.secondary_roi_poly = self.secondary_roi_poly.buffer(0)
                        logging.info(f"✅ Updated secondary ROI: {len(pixel_coords)} points")
                
                logging.info(f"🎯 QueueMonitor {self.channel_name} ROI updated successfully!")
                
                # Reset the flag so ROI polygons will be logged again
                if hasattr(self, '_roi_logged_once'):
                    delattr(self, '_roi_logged_once')
                if hasattr(self, '_roi_warning_logged'):
                    delattr(self, '_roi_warning_logged')
                if hasattr(self, '_secondary_roi_warning_logged'):
                    delattr(self, '_secondary_roi_warning_logged')
            except Exception as e:
                logging.error(f"Error updating ROI for {self.channel_name}: {e}")

    def shutdown(self):
        logging.info(f"Shutting down QueueMonitor for {self.channel_name}.")
        self.is_running = False

    def _get_queue_stats(self):
        """Get served count and peak count for today with caching"""
        current_time = time.time()
        
        # Return cached values if updated recently
        if current_time - self.last_stats_update < self.stats_update_interval:
            return self.cached_served_today, self.cached_peak_count
        
        served_today = 0
        peak_count = 0
        
        if not db_connected:
            return served_today, peak_count
        
        try:
            with SessionLocal() as db:
                today_ist = datetime.now(IST).date()
                start_of_day = datetime.combine(today_ist, datetime.min.time()).replace(tzinfo=IST)
                end_of_day = datetime.combine(today_ist, datetime.max.time()).replace(tzinfo=IST)
                
                # Get all queue logs for today, ordered by time
                records = db.query(QueueLog).filter(
                    QueueLog.channel_id == self.channel_id,
                    QueueLog.timestamp >= start_of_day,
                    QueueLog.timestamp <= end_of_day
                ).order_by(QueueLog.timestamp).all()
                
                if records:
                    # Calculate peak count
                    peak_count = max(record.queue_count for record in records)
                    
                    # Calculate served count (count people who entered counter area)
                    # Count transitions where queue decreases
                    prev_count = 0
                    for record in records:
                        if prev_count > 0 and record.queue_count < prev_count:
                            # People moved from queue (likely to counter)
                            served_today += (prev_count - record.queue_count)
                        prev_count = record.queue_count
                    
                    logging.debug(f"Queue stats for {self.channel_name}: Served={served_today}, Peak={peak_count}, Records={len(records)}")
                
                # Update cache
                self.cached_served_today = served_today
                self.cached_peak_count = peak_count
                self.last_stats_update = current_time
                
        except Exception as e:
            logging.error(f"Failed to get queue stats for {self.channel_name}: {e}")
        
        return served_today, peak_count

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                placeholder = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(placeholder, 'Connecting...', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (201, 209, 217), 2)
                _, jpeg = cv2.imencode('.jpg', placeholder); return jpeg.tobytes()
            _, jpeg = cv2.imencode('.jpg', self.latest_frame); return jpeg.tobytes()

    # Non-blocking DB persistence to avoid adding latency in the frame loop
    def _persist_queue_count(self, count: int) -> None:
        if not db_connected:
            return
        try:
            with SessionLocal() as db:
                db.add(QueueLog(channel_id=self.channel_id, queue_count=count))
                db.commit()
        except Exception as e:
            logging.error(f"Failed to save queue count to DB for {self.channel_name}: {e}")

    def _update_roi_polygons(self, frame):
        """Update ROI polygons from normalized coordinates based on current frame dimensions"""
        h, w = frame.shape[:2]
        if hasattr(self, 'normalized_main_roi') and self.normalized_main_roi and len(self.normalized_main_roi) >= 3:
            try:
                pixel_coords = [(int(p[0]*w), int(p[1]*h)) for p in self.normalized_main_roi]
                self.roi_poly = Polygon(pixel_coords)
                if not self.roi_poly.is_valid:
                    logging.warning(f"Main ROI polygon is invalid for {self.channel_name}. Coords: {pixel_coords}")
                    self.roi_poly = self.roi_poly.buffer(0)  # Try to fix invalid polygon
                logging.info(f"Updated main ROI for {self.channel_name}: {len(pixel_coords)} points, valid: {self.roi_poly.is_valid}, empty: {self.roi_poly.is_empty}")
            except Exception as e:
                logging.error(f"Error creating main ROI polygon for {self.channel_name}: {e}")
        else:
            logging.warning(f"No valid normalized_main_roi for {self.channel_name}")
            
        if hasattr(self, 'normalized_secondary_roi') and self.normalized_secondary_roi and len(self.normalized_secondary_roi) >= 3:
            try:
                pixel_coords = [(int(p[0]*w), int(p[1]*h)) for p in self.normalized_secondary_roi]
                self.secondary_roi_poly = Polygon(pixel_coords)
                if not self.secondary_roi_poly.is_valid:
                    logging.warning(f"Secondary ROI polygon is invalid for {self.channel_name}. Coords: {pixel_coords}")
                    self.secondary_roi_poly = self.secondary_roi_poly.buffer(0)  # Try to fix invalid polygon
                logging.info(f"Updated secondary ROI for {self.channel_name}: {len(pixel_coords)} points, valid: {self.secondary_roi_poly.is_valid}, empty: {self.secondary_roi_poly.is_empty}")
            except Exception as e:
                logging.error(f"Error creating secondary ROI polygon for {self.channel_name}: {e}")
        else:
            logging.warning(f"No valid normalized_secondary_roi for {self.channel_name}")

    def run(self):
        first_frame = True
        consecutive_errors = 0
        max_consecutive_errors = 10
        last_error_time = 0
        
        while self.is_running:
            try:
                frame = getattr(self, 'frame_hub', None).get_latest() if hasattr(self, 'frame_hub') else None
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                if first_frame or not self.roi_poly.is_valid or self.roi_poly.is_empty:
                    self._update_roi_polygons(frame)
                    first_frame = False

                self.process_frame(frame.copy())
                consecutive_errors = 0  # Reset on successful frame
                
            except RuntimeError as e:
                error_msg = str(e)
                if 'CUDA' in error_msg or 'cuda' in error_msg:
                    consecutive_errors += 1
                    last_error_time = time.time()
                    logging.error(f"CUDA error in QueueMonitor {self.channel_name} run loop: {e}. Error count: {consecutive_errors}")
                    
                logging.error(f"Runtime error in QueueMonitor {self.channel_name} run loop: {e}. Error count: {consecutive_errors}")
                
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logging.error(f"Too many consecutive errors for {self.channel_name}. Pausing for recovery...")
                    time.sleep(15)  # Longer pause for recovery
                    consecutive_errors = 0
                else:
                    time.sleep(3)  # Short pause before retry
            except Exception as e:
                logging.error(f"Unexpected error in QueueMonitor {self.channel_name}: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logging.error(f"Too many consecutive errors for {self.channel_name}. Pausing...")
                    time.sleep(15)
                    consecutive_errors = 0
                else:
                    time.sleep(1)

        # No cap to release when using FrameHub

    def process_frame(self, frame):
        current_time = time.time()
        # Use lower confidence for better detection of partially occluded people (especially in counter area)
        results = safe_track_persons(self.model, frame, conf=0.20, iou=0.5, processor_name=f"{self.channel_name}-QueueMonitor")
        current_tracks_in_main_roi, current_tracks_in_secondary_roi = set(), set()

        r0 = results[0] if (results and len(results) > 0) else None
        
        # Debug: Log ROI status
        if not hasattr(self, '_roi_logged_once'):
            logging.info(f"ROI Status for {self.channel_name}: Main ROI valid={self.roi_poly.is_valid}, empty={self.roi_poly.is_empty}, "
                        f"Secondary ROI valid={self.secondary_roi_poly.is_valid}, empty={self.secondary_roi_poly.is_empty}")
            if hasattr(self, 'normalized_main_roi'):
                logging.info(f"Normalized main ROI: {self.normalized_main_roi}")
            if hasattr(self, 'normalized_secondary_roi'):
                logging.info(f"Normalized secondary ROI: {self.normalized_secondary_roi}")
            self._roi_logged_once = True
        
        # FALLBACK: If tracking IDs are not available, use detection indices as pseudo-IDs
        if r0 is not None and getattr(r0, 'boxes', None) is not None:
            boxes = r0.boxes.xyxy.cpu()
            
            # Try to get tracking IDs, fallback to using detection indices
            if getattr(r0.boxes, 'id', None) is not None:
                track_ids = r0.boxes.id.int().cpu().tolist()
            else:
                # Tracking failed - use hash of box coordinates as pseudo-ID for this frame
                track_ids = []
                for i, box in enumerate(boxes):
                    # Create a stable ID based on box position (will be consistent across frames for stationary objects)
                    pseudo_id = hash((int(box[0]/10)*10, int(box[1]/10)*10, int(box[2]/10)*10, int(box[3]/10)*10)) % 100000
                    track_ids.append(pseudo_id)
                if len(boxes) > 0 and not hasattr(self, '_tracking_fallback_logged'):
                    logging.warning(f"{self.channel_name}: Tracking IDs not available, using position-based pseudo-IDs")
                    self._tracking_fallback_logged = True
            
            # Debug: Log number of detections
            if len(boxes) > 0:
                logging.debug(f"Detected {len(boxes)} persons in frame for {self.channel_name}")
            
            for box, track_id in zip(boxes, track_ids):
                # Calculate the true center point of the bounding box (center X and center Y)
                # box format: [x1, y1, x2, y2] where (x1,y1) is top-left and (x2,y2) is bottom-right
                center_x = int((box[0] + box[2]) / 2)  # Center X coordinate
                center_y = int((box[1] + box[3]) / 2)  # Center Y coordinate (true center, not bottom)
                person_point = Point(center_x, center_y)
                
                # Check main ROI (queue area) - count if center point is inside ROI
                if self.roi_poly.is_valid and not self.roi_poly.is_empty:
                    contains_main = self.roi_poly.contains(person_point)
                    if contains_main:
                        current_tracks_in_main_roi.add(track_id)
                        tracker = self.queue_tracker[track_id]
                        if tracker['entry_time'] == 0: 
                            tracker['entry_time'] = current_time
                            logging.info(f"Person {track_id} entered queue ROI at {current_time} - Center Point: ({center_x}, {center_y})")
                    else:
                        # Log when person is detected but not in ROI (for debugging)
                        if track_id not in current_tracks_in_main_roi and len(current_tracks_in_main_roi) == 0:
                            logging.debug(f"Person {track_id} center point ({center_x}, {center_y}) NOT in queue ROI")
                else:
                    if not hasattr(self, '_roi_warning_logged'):
                        logging.warning(f"Main ROI is invalid or empty for {self.channel_name} - cannot count persons. Valid: {self.roi_poly.is_valid}, Empty: {self.roi_poly.is_empty}")
                        self._roi_warning_logged = True
                
                # Check secondary ROI (counter area) - count if center point is inside ROI
                if self.secondary_roi_poly.is_valid and not self.secondary_roi_poly.is_empty:
                    contains_secondary = self.secondary_roi_poly.contains(person_point)
                    if contains_secondary:
                        current_tracks_in_secondary_roi.add(track_id)
                        sec_tracker = self.secondary_queue_tracker[track_id]
                        if sec_tracker['entry_time'] == 0: 
                            sec_tracker['entry_time'] = current_time
                            logging.info(f"Person {track_id} entered counter ROI at {current_time} - Center Point: ({center_x}, {center_y})")
                        # Update last detection time whenever someone is detected in counter
                        self.last_counter_detection_time = current_time
                    else:
                        # Log when person is detected but not in ROI (for debugging)
                        if track_id not in current_tracks_in_secondary_roi and len(current_tracks_in_secondary_roi) == 0:
                            logging.debug(f"Person {track_id} center point ({center_x}, {center_y}) NOT in counter ROI")
                else:
                    if not hasattr(self, '_secondary_roi_warning_logged'):
                        logging.warning(f"Secondary ROI is invalid or empty for {self.channel_name} - cannot count persons. Valid: {self.secondary_roi_poly.is_valid}, Empty: {self.secondary_roi_poly.is_empty}")
                        self._secondary_roi_warning_logged = True

        # Clean up trackers for persons who left the ROI
        track_ids_to_remove = [tid for tid in list(self.queue_tracker.keys()) if tid not in current_tracks_in_main_roi]
        for tid in track_ids_to_remove:
            self.queue_tracker.pop(tid, None)
        
        track_ids_to_remove_sec = [tid for tid in list(self.secondary_queue_tracker.keys()) if tid not in current_tracks_in_secondary_roi]
        for tid in track_ids_to_remove_sec:
            self.secondary_queue_tracker.pop(tid, None)

        # Count persons in queue ROI who have been there long enough
        valid_queue_count = 0
        for track_id in current_tracks_in_main_roi:
            if track_id in self.queue_tracker:
                entry_time = self.queue_tracker[track_id]['entry_time']
                if entry_time > 0 and (current_time - entry_time) >= QUEUE_DWELL_TIME_SEC:
                    valid_queue_count += 1
        updated = False
        if self.current_queue_count != valid_queue_count:
            self.current_queue_count = valid_queue_count
            updated = True
            # Persist queue count to database for statistics
            self._persist_queue_count(valid_queue_count)

        # Validate secondary (counter area) count with dwell
        valid_secondary_count = 0
        for track_id in current_tracks_in_secondary_roi:
            if track_id in self.secondary_queue_tracker:
                entry_time = self.secondary_queue_tracker[track_id]['entry_time']
                if entry_time > 0 and (current_time - entry_time) >= QUEUE_DWELL_TIME_SEC:
                    valid_secondary_count += 1
        
        # Persistence mechanism: If no one is currently detected in counter but someone was detected
        # within the persistence window, still consider counter as occupied to prevent false alerts
        if valid_secondary_count == 0 and self.last_counter_detection_time > 0:
            time_since_last_detection = current_time - self.last_counter_detection_time
            if time_since_last_detection <= QUEUE_COUNTER_PERSISTENCE_SEC:
                valid_secondary_count = 1  # Assume counter is still occupied
                logging.debug(f"Counter area persistence active: last detection was {time_since_last_detection:.1f}s ago (within {QUEUE_COUNTER_PERSISTENCE_SEC}s window)")
            else:
                # Reset if persistence window expired
                self.last_counter_detection_time = 0

        if self.current_secondary_count != valid_secondary_count:
            self.current_secondary_count = valid_secondary_count
            updated = True
            # Persist queue count whenever it changes
            self._persist_queue_count(self.current_queue_count)
            # No screenshot when counter area becomes occupied - only when queue present and counter empty

        # Emit live counts (no DB persistence) if either changed
        if updated:
            served_today, peak_count = self._get_queue_stats()
            socketio.emit('queue_update', {
                'channel_id': self.channel_id,
                'queue': self.current_queue_count,
                'counter': self.current_secondary_count,
                'count': self.current_queue_count,  # backward compat
                'served_today': served_today,
                'peak_count': peak_count
            })

        # Check for persons who have been in queue for more than 5 seconds with no one in counter area
        persons_in_queue_5sec = []
        for track_id in current_tracks_in_main_roi:
            if track_id in self.queue_tracker:
                dwell_time = current_time - self.queue_tracker[track_id]['entry_time']
                if dwell_time >= QUEUE_SCREENSHOT_DWELL_TIME_SEC:
                    persons_in_queue_5sec.append(track_id)
        
        # Screenshot trigger 1: person waiting > 5 seconds AND counter is empty (HIGHEST PRIORITY)
        should_screenshot_5sec = (
            len(persons_in_queue_5sec) > 0 and  # Someone waited > 5 seconds
            valid_secondary_count == 0 and  # Counter is empty
            (current_time - self.last_screenshot_time) > self.screenshot_cooldown
        )
        
        # Screenshot trigger 2: queue count > 3 (MEDIUM PRIORITY)
        should_screenshot_high_count = (
            valid_queue_count > QUEUE_HIGH_COUNT_THRESHOLD and  # Queue count > 3
            (current_time - self.last_screenshot_time) > self.screenshot_cooldown
        )
        
        # Screenshot trigger 3: counter is empty AND queue has people (FALLBACK)
        # Take screenshot every 10 seconds if counter empty and queue has people
        should_screenshot_counter_empty = (
            valid_queue_count > 0 and  # Queue has people
            valid_secondary_count == 0 and  # Counter is empty
            (current_time - self.last_screenshot_time) > 10.0  # 10-second cooldown
        )

        # Alert when cashier area is empty and queue has 2+ people (with cooldown)
        should_alert = (
            valid_queue_count >= QUEUE_ALERT_THRESHOLD and
            self.current_secondary_count == 0 and
            (current_time - self.last_alert_time) > QUEUE_ALERT_COOLDOWN_SEC
        )
        
        # Overqueue detection: when cashier is present and queue has 4+ people
        should_overqueue_alert = (
            valid_queue_count >= QUEUE_OVERQUEUE_THRESHOLD and
            self.current_secondary_count > 0 and
            (current_time - self.last_overqueue_time) > QUEUE_ALERT_COOLDOWN_SEC
        )

        # Create annotated frame with person bounding boxes
        annotated_frame = frame.copy()
        
        # Draw ROI polygons on the frame for monitoring
        # Main ROI (Queue area) - Blue polygon
        if self.roi_poly.is_valid and not self.roi_poly.is_empty:
            try:
                roi_points = np.array(list(self.roi_poly.exterior.coords), dtype=np.int32)
                cv2.polylines(annotated_frame, [roi_points], isClosed=True, color=(255, 0, 0), thickness=2)
                # Add label for main ROI
                if len(roi_points) > 0:
                    label_pos = tuple(roi_points[0])
                    cv2.putText(annotated_frame, 'Queue ROI', label_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            except Exception as e:
                logging.warning(f"Could not draw main ROI: {e}")
        
        # Secondary ROI (Counter area) - Green polygon
        if self.secondary_roi_poly.is_valid and not self.secondary_roi_poly.is_empty:
            try:
                roi_points = np.array(list(self.secondary_roi_poly.exterior.coords), dtype=np.int32)
                cv2.polylines(annotated_frame, [roi_points], isClosed=True, color=(0, 255, 0), thickness=2)
                # Add label for secondary ROI
                if len(roi_points) > 0:
                    label_pos = tuple(roi_points[0])
                    cv2.putText(annotated_frame, 'Counter ROI', label_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception as e:
                logging.warning(f"Could not draw secondary ROI: {e}")
        
        # Draw person bounding boxes
        if r0 is not None and getattr(r0, 'boxes', None) is not None and getattr(r0.boxes, 'id', None) is not None:
            boxes_xyxy = r0.boxes.xyxy.cpu()
            track_ids = r0.boxes.id.int().cpu().tolist()
            
            for i, track_id in enumerate(track_ids):
                if track_id is not None:
                    box = boxes_xyxy[i]
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    
                    # Color based on location: Blue for queue, Green for counter
                    if track_id in current_tracks_in_main_roi:
                        color = (255, 0, 0)  # Blue for queue
                        label = f"Queue #{track_id}"
                    elif track_id in current_tracks_in_secondary_roi:
                        color = (0, 255, 0)  # Green for counter
                        label = f"Counter #{track_id}"
                    else:
                        color = (128, 128, 128)  # Gray for others
                        label = f"Person #{track_id}"
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Take screenshot if person in queue > 5 seconds
        if should_screenshot_5sec:
            self.last_screenshot_time = current_time
            screenshot_message = f"Person waiting in queue for more than {QUEUE_SCREENSHOT_DWELL_TIME_SEC} seconds. Queue count: {valid_queue_count}, Counter: {valid_secondary_count}"
            logging.warning(f"5-SEC WAIT SCREENSHOT on {self.channel_name}: {screenshot_message}")
            try:
                media_path = handle_detection('QueueMonitor', self.channel_id, [annotated_frame], screenshot_message, is_gif=False)
                if media_path:
                    logging.info(f"Screenshot saved successfully: {media_path}")
                else:
                    logging.error(f"Failed to save screenshot for {self.channel_name}")
            except Exception as e:
                logging.error(f"Error saving screenshot for {self.channel_name}: {e}")
        
        # Take screenshot if queue count > 3
        elif should_screenshot_high_count:
            self.last_screenshot_time = current_time
            high_count_message = f"High queue count: {valid_queue_count} people in queue. Counter: {valid_secondary_count}"
            logging.warning(f"HIGH QUEUE COUNT SCREENSHOT on {self.channel_name}: {high_count_message}")
            try:
                media_path = handle_detection('QueueMonitor', self.channel_id, [annotated_frame], high_count_message, is_gif=False)
                if media_path:
                    logging.info(f"Screenshot saved successfully: {media_path}")
                else:
                    logging.error(f"Failed to save screenshot for {self.channel_name}")
            except Exception as e:
                logging.error(f"Error saving screenshot for {self.channel_name}: {e}")
        
        # Take screenshot if counter empty and queue has people
        elif should_screenshot_counter_empty:
            self.last_screenshot_time = current_time
            counter_empty_message = f"Counter is empty but queue has {valid_queue_count} people waiting"
            logging.info(f"COUNTER EMPTY SCREENSHOT on {self.channel_name}: {counter_empty_message}")
            try:
                media_path = handle_detection('QueueMonitor', self.channel_id, [annotated_frame], counter_empty_message, is_gif=False)
                if media_path:
                    logging.info(f"Screenshot saved successfully: {media_path}")
                else:
                    logging.error(f"Failed to save screenshot for {self.channel_name}")
            except Exception as e:
                logging.error(f"Error saving screenshot for {self.channel_name}: {e}")
        
        if should_alert:
            self.last_alert_time = current_time
            alert_message = f"Queue is full ({valid_queue_count} people), but the counter is free."
            logging.warning(f"QUEUE ALERT on {self.channel_name}: {alert_message}")
            send_telegram_notification(f"🚨 **Queue Alert: {self.channel_name}** 🚨\n{alert_message}")
            handle_detection('QueueMonitor', self.channel_id, [annotated_frame], alert_message, is_gif=False)
        
        if should_overqueue_alert:
            self.last_overqueue_time = current_time
            overqueue_message = f"OVERQUEUE: {valid_queue_count} people in queue with cashier present!"
            logging.warning(f"OVERQUEUE ALERT on {self.channel_name}: {overqueue_message}")
            send_telegram_notification(f"⚠️ **Overqueue Alert: {self.channel_name}** ⚠️\n{overqueue_message}")
            handle_detection('QueueMonitor', self.channel_id, [annotated_frame], overqueue_message, is_gif=False)

        # Count display text removed for clean transparent view
        # Log count changes for debugging
        queue_display_count = valid_queue_count
        counter_display_count = valid_secondary_count
        if queue_display_count > 0 or counter_display_count > 0:
            logging.info(f"QueueMonitor {self.channel_name}: Queue={queue_display_count}, Counter={counter_display_count}, "
                       f"Tracks in main ROI: {len(current_tracks_in_main_roi)}, "
                       f"Tracks in secondary ROI: {len(current_tracks_in_secondary_roi)}")
        
        with self.lock: self.latest_frame = annotated_frame.copy()


class OccupancyMonitorProcessor(threading.Thread):
    """
    Enhanced Occupancy Monitor - CUDA enabled, accurate detection, scheduled operation
    """
    
    def __init__(self, rtsp_url, channel_id, channel_name, model, socketio, SessionLocal, send_notification):
        super().__init__(name=f"OccupancyMonitor-{channel_name}")
        self.rtsp_url = rtsp_url
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.model = model
        self.socketio = socketio
        self.SessionLocal = SessionLocal
        self.send_notification = send_notification
        
        # FORCE CPU MODE - Disable all CUDA usage
        self.device = 'cpu'
        self.model.to(self.device)
        logging.info(f"🎯 Using device: {self.device.upper()} (FORCED CPU-ONLY)")
        
        self.is_running = True
        self.lock = threading.Lock()
        self.latest_frame = None
        
        self.schedule = {}  # {time_slot: {day: required_count}}
        self.live_count = 0
        self.required_count = 0
        self.current_time_slot = ""
        self.last_alert_time = 0
        self.alert_cooldown = 300  # 5 minutes between alerts
        
        # Track if requirement is met
        self.requirement_met = False
        self.requirement_met_time = 0
        self.pause_after_met_duration = 300  # Pause for 5 minutes after requirement met
        
        # Load schedule from database
        self._load_schedule_from_db()
        
        logging.info(f"✅ Occupancy Monitor initialized for {self.channel_name}")
    
    @staticmethod
    def initialize_tables(engine):
        """Initialize database tables"""
        try:
            Base.metadata.create_all(bind=engine)
            logging.info("Tables 'occupancy_logs' and 'occupancy_schedules' checked/created.")
        except Exception as e:
            logging.error(f"Could not create OccupancyMonitor tables: {e}")
    
    def _load_schedule_from_db(self):
        """Load schedule from database for this channel"""
        try:
            with self.SessionLocal() as db:
                records = db.query(OccupancySchedule).filter_by(channel_id=self.channel_id).all()
                self.schedule = {}
                for record in records:
                    if record.time_slot not in self.schedule:
                        self.schedule[record.time_slot] = {}
                    self.schedule[record.time_slot][record.day_of_week] = record.required_count
                logging.info(f"Loaded {len(records)} schedule entries for {self.channel_name}")
        except Exception as e:
            logging.error(f"Error loading schedule: {e}")
    
    def update_schedule(self, schedule_data):
        """Update schedule for this channel"""
        try:
            with self.SessionLocal() as db:
                db.query(OccupancySchedule).filter_by(channel_id=self.channel_id).delete()
                
                for time_slot, days in schedule_data.items():
                    for day_name, required_count in days.items():
                        db.add(OccupancySchedule(
                            channel_id=self.channel_id,
                            time_slot=time_slot,
                            day_of_week=day_name,
                            required_count=required_count
                        ))
                db.commit()
                
                self._load_schedule_from_db()
                logging.info(f"Schedule updated for {self.channel_name}")
                return True
        except Exception as e:
            logging.error(f"Error updating schedule: {e}")
            return False
    
    def get_frame(self):
        """Return latest frame as JPEG bytes - zero-lag optimized"""
        with self.lock:
            if self.latest_frame is None:
                placeholder = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(placeholder, 'Connecting...', (180, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (201, 209, 217), 2)
                _, jpeg = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 50])
                return jpeg.tobytes()
            
            # Zero-lag: Aggressive JPEG compression for instant encoding
            success, jpeg = cv2.imencode('.jpg', self.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            return jpeg.tobytes() if success else b''
    
    def _is_within_schedule(self):
        """Check if current time is within a scheduled slot"""
        now = datetime.now(IST)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        current_day = days[now.weekday()]
        current_hour = f"{now.hour}:00"
        
        # Check if schedule exists for this time
        if current_hour in self.schedule and current_day in self.schedule[current_hour]:
            return True, current_hour, current_day, self.schedule[current_hour][current_day]
        return False, current_hour, current_day, 0
    
    def _detect_people(self, frame):
        """Enhanced YOLO detection with CUDA support and maximum accuracy"""
        try:
            # Enhanced detection with very low confidence for maximum recall
            with torch.inference_mode():
                results = self.model(
                    frame, 
                    conf=0.15,
                    iou=0.40,
                    classes=[0],
                    verbose=False,
                    device=self.device,
                    imgsz=640,
                    max_det=100,
                    agnostic_nms=True,
                    half=False  # CPU mode - no FP16
                )
            person_count = 0
            detections = []
            
            annotated_frame = frame.copy()
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    
                    # Very low threshold - catch everyone!
                    if conf > 0.15:
                        person_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append({'conf': conf, 'bbox': (x1, y1, x2, y2)})
                        
                        # Enhanced color coding based on confidence
                        if conf > 0.6:
                            color = (0, 255, 0)    # Bright green - very confident
                            thickness = 4
                        elif conf > 0.4:
                            color = (0, 220, 0)    # Green - confident
                            thickness = 3
                        elif conf > 0.25:
                            color = (0, 255, 255)  # Yellow - moderate
                            thickness = 3
                        else:
                            color = (255, 165, 0)  # Orange - low confidence
                            thickness = 2
                        
                        # Draw bounding box with better visibility
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                        # Add label background for better readability
                        label = f'Person {person_count} ({conf:.2f})'
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1-label_size[1]-10), (x1+label_size[0]+5, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Log detections for debugging
            if person_count > 0:
                conf_list = [f"{d['conf']:.2f}" for d in detections]
                logging.info(f"Detected {person_count} people with confidences: {', '.join(conf_list)}")
            
            return person_count, annotated_frame
        except Exception as e:
            logging.error(f"Detection error: {e}")
            return 0, frame
    
    def _check_occupancy_requirement(self):
        """Check if live count meets schedule requirement"""
        now = datetime.now(IST)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        current_day = days[now.weekday()]
        current_hour = f"{now.hour}:00"
        
        self.current_time_slot = f"{current_day} {current_hour}"
        
        # Get required count from schedule
        self.required_count = 0
        if current_hour in self.schedule and current_day in self.schedule[current_hour]:
            self.required_count = self.schedule[current_hour][current_day]
        
        # Determine status
        status = 'NO_SCHEDULE'
        if self.required_count > 0:
            if self.live_count >= self.required_count:
                status = 'OK'
                # Mark requirement as met
                if not self.requirement_met:
                    self.requirement_met = True
                    self.requirement_met_time = time.time()
                    logging.info(f"✅ Requirement MET for {self.channel_name}: {self.live_count}/{self.required_count}")
            else:
                status = 'BELOW_REQUIREMENT'
                self.requirement_met = False
                
                # Send alert if cooldown period has passed
                current_time = time.time()
                if current_time - self.last_alert_time > self.alert_cooldown:
                    shortage = self.required_count - self.live_count
                    message = (f"⚠️ *OCCUPANCY ALERT* - {self.channel_name}\n"
                             f"Time: {current_hour} ({current_day})\n"
                             f"Required: {self.required_count} people\n"
                             f"Detected: {self.live_count} people\n"
                             f"Shortage: {shortage} people")
                    self.send_notification(message)
                    self.last_alert_time = current_time
                    logging.warning(f"🚨 Occupancy alert: {shortage} people short at {self.channel_name}")
        
        # Log to database
        try:
            with self.SessionLocal() as db:
                db.add(OccupancyLog(
                    channel_id=self.channel_id,
                    time_slot=current_hour,
                    day_of_week=current_day,
                    live_count=self.live_count,
                    required_count=self.required_count,
                    status=status
                ))
                db.commit()
        except Exception as e:
            logging.error(f"Error logging occupancy: {e}")
        
        # Calculate today's statistics
        max_today = self.live_count  # Default to current
        avg_today = self.live_count  # Default to current
        
        try:
            with self.SessionLocal() as db:
                today = datetime.now(IST).date()
                today_start = datetime.combine(today, datetime.min.time()).replace(tzinfo=IST)
                today_end = datetime.combine(today, datetime.max.time()).replace(tzinfo=IST)
                
                logs = db.query(OccupancyLog).filter(
                    OccupancyLog.channel_id == self.channel_id,
                    OccupancyLog.timestamp >= today_start,
                    OccupancyLog.timestamp <= today_end
                ).all()
                
                if logs:
                    counts = [log.live_count for log in logs]
                    max_today = max(counts) if counts else self.live_count
                    avg_today = round(sum(counts) / len(counts)) if counts else self.live_count
        except Exception as e:
            logging.debug(f"Error calculating occupancy stats: {e}")
        
        # Generate banner text based on status
        banner_text = ""
        if status == 'OK':
            banner_text = f"✅ REQUIREMENT MET - {self.live_count}/{self.required_count} people present"
        elif status == 'BELOW_REQUIREMENT':
            shortage = self.required_count - self.live_count
            banner_text = f"⚠️ ALERT: {shortage} people short! ({self.live_count}/{self.required_count} present)"
        elif status == 'PAUSED':
            banner_text = f"✅ REQUIREMENT MET - Monitoring paused"
        elif status == 'NO_SCHEDULE':
            banner_text = "ℹ️ No schedule configured for this time"
        
        # Emit to dashboard via SocketIO
        self.socketio.emit('occupancy_update', {
            'channel_id': self.channel_id,
            'channel_name': self.channel_name,
            'time_slot': self.current_time_slot,
            'live_count': self.live_count,
            'required_count': self.required_count,
            'status': status,
            'max_today': max_today,
            'avg_today': avg_today,
            'banner_text': banner_text
        })
        
        return status
    
    def _should_run_detection(self):
        """
        Determine if detection should run based on:
        1. Schedule availability (only run during scheduled times)
        2. Requirement status (pause if already met)
        """
        is_scheduled, current_hour, current_day, required = self._is_within_schedule()
        
        # If no schedule for this time, don't run detection
        if not is_scheduled or required == 0:
            return False, "NO_SCHEDULE"
        
        # If requirement is met and we're still in pause period
        if self.requirement_met:
            time_since_met = time.time() - self.requirement_met_time
            if time_since_met < self.pause_after_met_duration:
                # Still paused
                return False, "PAUSED_REQ_MET"
            else:
                # Pause period over, resume detection
                self.requirement_met = False
                logging.info(f"🔄 Resuming detection for {self.channel_name} after pause period")
        
        return True, "ACTIVE"
    
    def run(self):
        """Enhanced processing loop - SMOOTH STREAMING with continuous detection"""
        logging.info(f"Starting Enhanced Occupancy Monitor for {self.channel_name}...")
        logging.info(f"Device: {self.device.upper()}, Confidence: 0.15, Mode: CONTINUOUS (Smooth streaming)")
        
        # Use FrameHub for frames
        frame_delay = 0.01
        
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        last_schedule_check = 0
        last_detection_time = 0
        detection_cooldown = 1.0  # Run YOLO detection once per second (avoid GPU overload)
        
        while self.is_running:
            frame_start_time = time.time()
            
            frame = getattr(self, 'frame_hub', None).get_latest() if hasattr(self, 'frame_hub') else None
            if frame is None:
                time.sleep(0.01)
                continue
            
            reconnect_attempts = 0
            current_time = time.time()
            
            # Check schedule every 10 seconds
            if current_time - last_schedule_check > 10:
                should_detect, detection_status = self._should_run_detection()
                last_schedule_check = current_time
                
                # Update schedule info
                is_scheduled, current_hour, current_day, required = self._is_within_schedule()
                self.required_count = required
                self.current_time_slot = f"{current_day} {current_hour}"
            else:
                should_detect, detection_status = self._should_run_detection()
            
            # CONTINUOUS DETECTION (with 1-second cooldown for YOLO)
            time_since_last_detection = current_time - last_detection_time
            
            if should_detect and time_since_last_detection >= detection_cooldown:
                # RUN YOLO DETECTION
                last_detection_time = current_time
                
                self.live_count, annotated_frame = self._detect_people(frame)
                
                # Clean video feed - no overlays for better user experience
                with self.lock:
                    self.latest_frame = annotated_frame
                
                # Check requirement
                self._check_occupancy_requirement()
                
            elif should_detect:
                # Between YOLO detections - smooth video without overlays
                display_frame = frame.copy()
                
                # Clean video feed - no overlays
                with self.lock:
                    self.latest_frame = display_frame
                    
            else:
                # PAUSED/NO SCHEDULE - Clean video feed
                display_frame = frame.copy()
                
                # No overlays - clean video
                with self.lock:
                    self.latest_frame = display_frame
            
            # Maintain smooth FPS - NO FRAME SKIPPING
            elapsed = time.time() - frame_start_time
            sleep_time = max(0, frame_delay - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        logging.info(f"Occupancy Monitor stopped for {self.channel_name}")
    
    def stop(self):
        """Stop the processor"""
        logging.info(f"Stopping Occupancy Monitor for {self.channel_name}...")
        self.is_running = False
    
    def shutdown(self):
        """Shutdown method for compatibility"""
        self.stop()


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
    Uses the same backend approach as conversion analytics for consistency
    Supports restaurant filtering via restaurant_id query parameter
    """
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        # Get query parameters
        days = request.args.get('days', default=7, type=int)
        restaurant_id = request.args.get('restaurant_id', type=int)
        
        # Calculate date range
        end_date = datetime.now(IST).date()
        start_date = end_date - timedelta(days=days - 1)
        
        logging.info(f"📊 Sales Analytics Daily - days={days}, start={start_date}, end={end_date}, restaurant_id={restaurant_id}")
        
        with SessionLocal() as db:
            # Convert database restaurant_id to PetPooja restID
            petpooja_restaurant_id = get_petpooja_restaurant_id(restaurant_id, db)
            
            # Use the same approach as conversion analytics
            sales_data, remote_used = pp_sales_analytics.get_sales_data(
                db, start_date, end_date, use_remote=True, restaurant_id=restaurant_id
            )
            
            if not sales_data:
                return jsonify([])
            
            # Aggregate by date
            daily_aggregates = defaultdict(lambda: {'orders': 0, 'revenue': 0.0})
            for row in sales_data:
                daily_aggregates[row.sale_date]['orders'] += row.orders
                daily_aggregates[row.sale_date]['revenue'] += float(row.revenue) if row.revenue else 0
            
            # Convert to list format
            results = [
                {
                    'date': date.strftime('%Y-%m-%d'),
                    'total_sales': round(data['revenue'], 2),
                    'order_count': data['orders']
                }
                for date, data in sorted(daily_aggregates.items())
            ]
            
            logging.info(f"✅ Returning {len(results)} daily sales records (source: {'REMOTE' if remote_used else 'LOCAL'})")
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
    Uses backend processing with restaurant_id filter
    """
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        restaurant_id = request.args.get('restaurant_id', type=int)
        
        logging.info(f"📊 Payment Modes - restaurant_id={restaurant_id}")
        
        with SessionLocal() as db:
            petpooja_restaurant_id = get_petpooja_restaurant_id(restaurant_id, db)
            
            # Get current month start
            today = datetime.now(IST).date()
            start_date = datetime(today.year, today.month, 1).date()
            end_date = today
            
            # Get sales data
            sales_data, remote_used = pp_sales_analytics.get_sales_data(
                db, start_date, end_date, use_remote=True, restaurant_id=restaurant_id
            )
            
            if not sales_data:
                return jsonify([])
            
            # We need to get the raw events to extract payment modes
            # Use the client to fetch raw events and process them
            response = requests.get(
                f"{pp_sales_analytics.client.base_url}/webhook/events/search/date-range",
                params={
                    'start_date': (start_date - timedelta(days=1)).isoformat(),
                    'end_date': (end_date + timedelta(days=1)).isoformat(),
                    'token': pp_sales_analytics.client.api_token
                },
                timeout=30
            )
            
            if response.status_code != 200:
                return jsonify([])
            
            all_events = response.json()
            payment_aggregates = defaultdict(lambda: {'amount': 0.0, 'count': 0})
            seen_orders = set()
            
            for event in all_events:
                if event.get('content', {}).get('event') == 'orderdetails':
                    properties = event.get('content', {}).get('properties', {})
                    
                    # Filter by restaurant if specified
                    if petpooja_restaurant_id:
                        event_rest_id = properties.get('Restaurant', {}).get('restID')
                        if event_rest_id != petpooja_restaurant_id:
                            continue
                    
                    order = properties.get('Order', {})
                    order_id = order.get('orderID')
                    
                    if not order_id or order_id in seen_orders:
                        continue
                    
                    seen_orders.add(order_id)
                    
                    # Parse date and filter
                    try:
                        if order.get('created_on'):
                            order_time_utc = datetime.fromisoformat(order['created_on'].replace('Z', '+00:00'))
                        else:
                            created_at = event.get('created_at', '')
                            if not created_at:
                                continue
                            order_time_utc = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        
                        order_time_ist = order_time_utc.astimezone(IST)
                        order_date_ist = order_time_ist.date()
                        
                        if order_date_ist < start_date or order_date_ist > end_date:
                            continue
                        
                        payment_type = order.get('payment_type', 'Unknown')
                        total = float(order.get('total', 0))
                        
                        payment_aggregates[payment_type]['amount'] += total
                        payment_aggregates[payment_type]['count'] += 1
                    
                    except Exception as parse_error:
                        continue
            
            results = [
                {
                    'method': method,
                    'amount': round(data['amount'], 2),
                    'count': data['count']
                }
                for method, data in payment_aggregates.items()
            ]
            
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
    """
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        restaurant_id = request.args.get('restaurant_id', type=int)
        
        logging.info(f"📊 Order Types - restaurant_id={restaurant_id}")
        
        with SessionLocal() as db:
            petpooja_restaurant_id = get_petpooja_restaurant_id(restaurant_id, db)
            
            # Get current month start
            today = datetime.now(IST).date()
            start_date = datetime(today.year, today.month, 1).date()
            end_date = today
            
            # Get raw events
            response = requests.get(
                f"{pp_sales_analytics.client.base_url}/webhook/events/search/date-range",
                params={
                    'start_date': (start_date - timedelta(days=1)).isoformat(),
                    'end_date': (end_date + timedelta(days=1)).isoformat(),
                    'token': pp_sales_analytics.client.api_token
                },
                timeout=30
            )
            
            if response.status_code != 200:
                return jsonify([])
            
            all_events = response.json()
            order_type_aggregates = defaultdict(lambda: {'count': 0, 'total_sales': 0.0})
            seen_orders = set()
            
            for event in all_events:
                if event.get('content', {}).get('event') == 'orderdetails':
                    properties = event.get('content', {}).get('properties', {})
                    
                    # Filter by restaurant
                    if petpooja_restaurant_id:
                        event_rest_id = properties.get('Restaurant', {}).get('restID')
                        if event_rest_id != petpooja_restaurant_id:
                            continue
                    
                    order = properties.get('Order', {})
                    order_id = order.get('orderID')
                    
                    if not order_id or order_id in seen_orders:
                        continue
                    
                    seen_orders.add(order_id)
                    
                    try:
                        if order.get('created_on'):
                            order_time_utc = datetime.fromisoformat(order['created_on'].replace('Z', '+00:00'))
                        else:
                            created_at = event.get('created_at', '')
                            if not created_at:
                                continue
                            order_time_utc = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        
                        order_time_ist = order_time_utc.astimezone(IST)
                        order_date_ist = order_time_ist.date()
                        
                        if order_date_ist < start_date or order_date_ist > end_date:
                            continue
                        
                        order_source = order.get('order_from', 'Unknown')
                        order_type = order.get('order_type', 'Unknown')
                        total = float(order.get('total', 0))
                        
                        key = (order_source, order_type)
                        order_type_aggregates[key]['count'] += 1
                        order_type_aggregates[key]['total_sales'] += total
                    
                    except Exception:
                        continue
            
            results = [
                {
                    'order_source': source,
                    'order_type': otype,
                    'count': data['count'],
                    'total_sales': round(data['total_sales'], 2)
                }
                for (source, otype), data in order_type_aggregates.items()
            ]
            
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
    """
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        restaurant_id = request.args.get('restaurant_id', type=int)
        limit = request.args.get('limit', default=5, type=int)
        
        logging.info(f"📊 Top Items - restaurant_id={restaurant_id}, limit={limit}")
        
        with SessionLocal() as db:
            petpooja_restaurant_id = get_petpooja_restaurant_id(restaurant_id, db)
            
            # Get current month start
            today = datetime.now(IST).date()
            start_date = datetime(today.year, today.month, 1).date()
            end_date = today
            
            # Get raw events
            response = requests.get(
                f"{pp_sales_analytics.client.base_url}/webhook/events/search/date-range",
                params={
                    'start_date': (start_date - timedelta(days=1)).isoformat(),
                    'end_date': (end_date + timedelta(days=1)).isoformat(),
                    'token': pp_sales_analytics.client.api_token
                },
                timeout=30
            )
            
            if response.status_code != 200:
                return jsonify([])
            
            all_events = response.json()
            item_aggregates = defaultdict(lambda: {'quantity': 0, 'revenue': 0.0})
            seen_orders = set()
            
            for event in all_events:
                if event.get('content', {}).get('event') == 'orderdetails':
                    properties = event.get('content', {}).get('properties', {})
                    
                    # Filter by restaurant
                    if petpooja_restaurant_id:
                        event_rest_id = properties.get('Restaurant', {}).get('restID')
                        if event_rest_id != petpooja_restaurant_id:
                            continue
                    
                    order = properties.get('Order', {})
                    order_id = order.get('orderID')
                    
                    if not order_id or order_id in seen_orders:
                        continue
                    
                    seen_orders.add(order_id)
                    
                    try:
                        if order.get('created_on'):
                            order_time_utc = datetime.fromisoformat(order['created_on'].replace('Z', '+00:00'))
                        else:
                            created_at = event.get('created_at', '')
                            if not created_at:
                                continue
                            order_time_utc = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        
                        order_time_ist = order_time_utc.astimezone(IST)
                        order_date_ist = order_time_ist.date()
                        
                        if order_date_ist < start_date or order_date_ist > end_date:
                            continue
                        
                        # Process order items
                        order_items = properties.get('OrderItem', [])
                        if not isinstance(order_items, list):
                            continue
                        
                        order_total = float(order.get('total', 0))
                        items_subtotal = sum(float(item.get('total', 0)) for item in order_items if isinstance(item, dict))
                        
                        for item in order_items:
                            if not isinstance(item, dict):
                                continue
                            
                            item_name = item.get('name', '').strip()
                            if not item_name:
                                continue
                            
                            quantity = float(item.get('quantity', 0))
                            item_subtotal = float(item.get('total', 0))
                            
                            # Allocate order total proportionally
                            if items_subtotal > 0:
                                allocation_ratio = item_subtotal / items_subtotal
                                revenue = order_total * allocation_ratio
                            else:
                                revenue = item_subtotal
                            
                            item_aggregates[item_name]['quantity'] += quantity
                            item_aggregates[item_name]['revenue'] += revenue
                    
                    except Exception:
                        continue
            
            # Sort by quantity and get top items
            sorted_items = sorted(
                item_aggregates.items(),
                key=lambda x: x[1]['quantity'],
                reverse=True
            )[:limit]
            
            results = [
                {
                    'name': item_name,
                    'quantity_sold': int(data['quantity']),
                    'total_revenue': round(data['revenue'], 2)
                }
                for item_name, data in sorted_items
            ]
            
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
    Fetches footfall from local DB and sales from remote FastAPI (or local DB if available)
    Returns hourly breakdown with conversion rates
    Supports multi-restaurant filtering via restaurant_id query parameter
    
    REFACTORED: Now uses petpooja_integration module for clean separation of concerns
    """
    logging.info(f"Conversion API called - Session: {session.get('logged_in')}, User: {session.get('username')}")
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        # Get query parameters
        days = request.args.get('days', default=7, type=int)
        date_str = request.args.get('date')  # Optional: specific date (YYYY-MM-DD)
        restaurant_id = request.args.get('restaurant_id', type=int)  # Optional: filter by restaurant
        
        # Calculate date range
        if date_str:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            start_date = target_date
            end_date = target_date
        else:
            end_date = datetime.now(IST).date()
            start_date = end_date - timedelta(days=days - 1)
        
        logging.info(f"📅 Date Range Query: days={days}, start={start_date}, end={end_date}, restaurant_id={restaurant_id}")
        
        with SessionLocal() as db:
            # STEP 1: Get footfall data from local database (hourly_footfall)
            try:
                footfall_query = db.query(
                    HourlyFootfall.report_date,
                    HourlyFootfall.hour,
                    func.sum(HourlyFootfall.in_count).label('visitors')
                )
                
                # Only join with cameras if filtering by restaurant AND cameras table has data
                if restaurant_id:
                    camera_count = db.query(Camera).count()
                    if camera_count > 0:
                        footfall_query = footfall_query.join(
                            Camera, HourlyFootfall.channel_id == Camera.channel_id
                        ).filter(Camera.restaurant_id == restaurant_id)
                        logging.info(f"Filtering footfall data by restaurant_id: {restaurant_id}")
                    else:
                        logging.warning(f"⚠️ restaurant_id={restaurant_id} provided but cameras table is empty - showing all footfall data")
                
                footfall_query = footfall_query.filter(
                    HourlyFootfall.report_date >= start_date,
                    HourlyFootfall.report_date <= end_date
                ).group_by(
                    HourlyFootfall.report_date,
                    HourlyFootfall.hour
                ).order_by(
                    HourlyFootfall.report_date,
                    HourlyFootfall.hour
                ).all()
            except Exception as footfall_error:
                logging.error(f"❌ Error querying footfall data: {footfall_error}")
                import traceback
                traceback.print_exc()
                return jsonify({"error": f"Failed to query footfall data: {str(footfall_error)}"}), 500
            
            logging.info(f"📊 Footfall Query Results: {len(footfall_query)} records from {start_date} to {end_date}")
            if footfall_query:
                logging.info(f"   First record: {footfall_query[0].report_date} hour {footfall_query[0].hour} - {footfall_query[0].visitors} visitors")
                logging.info(f"   Last record: {footfall_query[-1].report_date} hour {footfall_query[-1].hour} - {footfall_query[-1].visitors} visitors")
            
            # STEP 2: Get sales data using PetPooja integration module
            try:
                sales_query, remote_api_used = pp_sales_analytics.get_sales_data(
                    db, start_date, end_date, use_remote=True, restaurant_id=restaurant_id
                )
            except Exception as sales_error:
                logging.error(f"❌ Error getting sales data: {sales_error}")
                import traceback
                traceback.print_exc()
                return jsonify({"error": f"Failed to get sales data: {str(sales_error)}"}), 500
            
            logging.info(f"📊 Sales Query Results: {len(sales_query) if sales_query else 0} records (source: {'REMOTE' if remote_api_used else 'LOCAL'})")
            if sales_query:
                logging.info(f"   First sale: {sales_query[0].sale_date} hour {sales_query[0].sale_hour} - {sales_query[0].orders} orders, ₹{sales_query[0].revenue}")
                logging.info(f"   Last sale: {sales_query[-1].sale_date} hour {sales_query[-1].sale_hour} - {sales_query[-1].orders} orders, ₹{sales_query[-1].revenue}")
            
            # If remote API failed, try daily distribution
            if not sales_query and not remote_api_used:
                # Convert database restaurant_id to PetPooja restID using database lookup
                petpooja_restaurant_id = get_petpooja_restaurant_id(restaurant_id, db)
                daily_sales = pp_sales_analytics.client.get_daily_sales(start_date, end_date, petpooja_restaurant_id)
                if daily_sales:
                    # Build footfall distribution for smart allocation
                    footfall_by_date_hour = defaultdict(lambda: defaultdict(int))
                    footfall_by_date_total = defaultdict(int)
                    
                    for f in footfall_query:
                        footfall_by_date_hour[f.report_date][f.hour] = f.visitors or 0
                        footfall_by_date_total[f.report_date] += f.visitors or 0
                    
                    # Use conversion analytics to distribute daily to hourly
                    sales_query = pp_conversion_analytics.distribute_daily_to_hourly(
                        daily_sales, footfall_by_date_hour, footfall_by_date_total, end_date
                    )
                    remote_api_used = True
            
            # Ensure sales_query is a list (handle None case)
            if sales_query is None:
                sales_query = []
            
            logging.info(f"📊 Data Summary: {len(footfall_query)} footfall records, {len(sales_query) if sales_query else 0} sales records")
            logging.info(f"Using {'REMOTE FastAPI' if remote_api_used else 'LOCAL DATABASE'} for sales data")
            
            # STEP 3: Calculate conversion metrics using PetPooja integration module
            try:
                results = pp_conversion_analytics.calculate_conversion_metrics(footfall_query, sales_query)
                logging.info(f"✅ Calculated {len(results)} hourly conversion records")
            except Exception as calc_error:
                logging.error(f"❌ Error calculating conversion metrics: {calc_error}")
                import traceback
                traceback.print_exc()
                return jsonify({"error": f"Failed to calculate conversion metrics: {str(calc_error)}"}), 500
            
            # Calculate summary statistics
            try:
                total_visitors = sum(r['visitors'] for r in results)
                total_orders = sum(r['orders'] for r in results)
                total_revenue = sum(r['revenue'] for r in results)
            except KeyError as ke:
                logging.error(f"❌ Missing key in results dictionary: {ke}")
                logging.error(f"Results structure: {results[:2] if results else 'empty'}")
                return jsonify({"error": f"Invalid results structure: missing key {ke}"}), 500
            except Exception as summary_error:
                logging.error(f"❌ Error calculating summary: {summary_error}")
                import traceback
                traceback.print_exc()
                return jsonify({"error": f"Failed to calculate summary: {str(summary_error)}"}), 500
            
            overall_conversion = (total_orders / total_visitors * 100) if total_visitors > 0 else 0
            overall_revenue_per_visitor = (total_revenue / total_visitors) if total_visitors > 0 else 0
            overall_avg_order_value = (total_revenue / total_orders) if total_orders > 0 else 0
            
            logging.info(f"✅ Final conversion totals: {total_orders} orders, ₹{total_revenue:.2f} revenue")
            
            # STEP 4: Identify busiest hours and peak demand patterns
            # Group by hour to find average metrics per hour of day
            hourly_aggregates = defaultdict(lambda: {'visitors': [], 'orders': [], 'revenue': []})
            for row in results:
                hour = row['hour']
                hourly_aggregates[hour]['visitors'].append(row['visitors'])
                hourly_aggregates[hour]['orders'].append(row['orders'])
                hourly_aggregates[hour]['revenue'].append(row['revenue'])
            
            # Calculate averages and identify peaks - show all business hours 6am-11pm
            hourly_stats = []
            for hour in range(6, 24):  # Only business hours 6am-11pm
                # Always include hour, even with no data (show 0, 0)
                if hour in hourly_aggregates:
                    agg = hourly_aggregates[hour]
                    avg_visitors = sum(agg['visitors']) / len(agg['visitors']) if agg['visitors'] else 0
                    avg_orders = sum(agg['orders']) / len(agg['orders']) if agg['orders'] else 0
                    avg_revenue = sum(agg['revenue']) / len(agg['revenue']) if agg['revenue'] else 0
                    total_volume = sum(agg['visitors']) + sum(agg['orders'])
                else:
                    # No data for this hour - show zeros
                    avg_visitors = 0
                    avg_orders = 0
                    avg_revenue = 0
                    total_volume = 0
                    
                hourly_stats.append({
                    'hour': hour,
                    'hour_label': datetime.strptime(str(hour), '%H').strftime('%I %p').lstrip('0'),
                    'avg_visitors': round(avg_visitors, 1),
                    'avg_orders': round(avg_orders, 1),
                    'avg_revenue': round(avg_revenue, 2),
                    'total_volume': total_volume
                })
            
            # Find busiest hour by visitors and hour with most orders
            busiest_by_visitors = max(hourly_stats, key=lambda x: x['avg_visitors']) if hourly_stats else None
            busiest_by_orders = max(hourly_stats, key=lambda x: x['avg_orders']) if hourly_stats else None
            
            # Sort by visitors to find peak hours based on footfall
            hourly_stats_by_visitors = sorted(hourly_stats, key=lambda x: x['avg_visitors'], reverse=True)
            top_visitor_hours = hourly_stats_by_visitors[:5]  # Top 5 by visitors for peak period calculation
            
            # Calculate staffing recommendations based on demand
            # Base staff: 2, +1 for every 20 visitors/hour
            matched_hours = 0  # Initialize counter for matched hours
            for hour_stat in hourly_stats:
                visitors_per_hour = hour_stat['avg_visitors']
                orders_per_hour = hour_stat['avg_orders']
                
                # Staffing recommendation (minimum 2, scale with traffic)
                base_staff = 2
                additional_staff = int(visitors_per_hour / 20)  # +1 staff per 20 visitors
                recommended_staff = max(base_staff, base_staff + additional_staff)
                
                # Inventory recommendation (scale with orders)
                # Assume 1.5x buffer for peak hours
                inventory_multiplier = 1.5 if hour_stat in top_visitor_hours[:3] else 1.2
                recommended_inventory_units = int(orders_per_hour * inventory_multiplier)
                
                hour_stat['recommended_staff'] = recommended_staff
                hour_stat['recommended_inventory'] = recommended_inventory_units
                matched_hours += 1  # Count this as a successful match
            
            logging.info(f"✅ Successfully matched {matched_hours} out of {len(footfall_query)} footfall records with sales data")
            
            # Re-sort by hour for display
            hourly_stats.sort(key=lambda x: x['hour'])
            
            # Identify peak periods based on visitor numbers only
            peak_morning = [h for h in top_visitor_hours if 6 <= h['hour'] < 12]
            peak_afternoon = [h for h in top_visitor_hours if 12 <= h['hour'] < 17]
            peak_evening = [h for h in top_visitor_hours if 17 <= h['hour'] < 24]
            
            peak_periods = {
                'morning': [h['hour_label'] for h in peak_morning],
                'afternoon': [h['hour_label'] for h in peak_afternoon],
                'evening': [h['hour_label'] for h in peak_evening]
            }
            
            return jsonify({
                'date_range': {
                    'start': start_date.strftime('%Y-%m-%d'),
                    'end': end_date.strftime('%Y-%m-%d'),
                    'days': days
                },
                'summary': {
                    'total_visitors': total_visitors,
                    # Display total orders as an integer (rounded) while keeping internal precision
                    'total_orders': int(round(total_orders)),
                    'total_revenue': round(total_revenue, 2),
                    'conversion_rate': round(overall_conversion, 2),
                    'revenue_per_visitor': round(overall_revenue_per_visitor, 2),
                    'avg_order_value': round(overall_avg_order_value, 2)
                },
                'hourly_data': results,
                'peak_analysis': {
                    'busiest_by_visitors': busiest_by_visitors,
                    'busiest_by_orders': busiest_by_orders,
                    'peak_periods': peak_periods,
                    'hourly_recommendations': hourly_stats
                }
            })
    
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
    """
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        # Get query parameters
        days = request.args.get('days', default=14, type=int)  # Default to 2 weeks for better patterns
        restaurant_id = request.args.get('restaurant_id', type=int)
        
        end_date = datetime.now(IST).date()
        start_date = end_date - timedelta(days=days - 1)
        
        with SessionLocal() as db:
            # Get footfall data
            footfall_query = db.query(
                HourlyFootfall.hour,
                func.avg(HourlyFootfall.in_count).label('avg_visitors'),
                func.max(HourlyFootfall.in_count).label('max_visitors'),
                func.count(HourlyFootfall.id).label('data_points')
            ).filter(
                HourlyFootfall.report_date >= start_date,
                HourlyFootfall.report_date <= end_date,
                HourlyFootfall.in_count > 0  # Only non-zero hours
            )
            
            # Only join with cameras if filtering by restaurant AND cameras table has data
            if restaurant_id:
                camera_count = db.query(Camera).count()
                if camera_count > 0:
                    footfall_query = footfall_query.join(
                        Camera, HourlyFootfall.channel_id == Camera.channel_id
                    ).filter(Camera.restaurant_id == restaurant_id)
                else:
                    logging.warning(f"⚠️ restaurant_id={restaurant_id} provided but cameras table is empty ({camera_count} records). Showing all footfall data.")
            
            footfall_query = footfall_query.group_by(HourlyFootfall.hour).all()
            
            # Get sales data grouped by hour
            sales_query = db.execute(text("""
                SELECT 
                    sale_hour,
                    AVG(order_count) as avg_orders,
                    MAX(order_count) as max_orders,
                    AVG(revenue) as avg_revenue
                FROM (
                    SELECT 
                        DATE(created_at AT TIME ZONE 'Asia/Kolkata') as sale_date,
                        EXTRACT(HOUR FROM created_at AT TIME ZONE 'Asia/Kolkata')::INTEGER as sale_hour,
                        COUNT(DISTINCT content->'properties'->'Order'->>'orderID') as order_count,
                        SUM(CAST(content->'properties'->'Order'->>'total' AS DECIMAL)) as revenue
                    FROM petpooja_webhook_events
                    WHERE DATE(created_at AT TIME ZONE 'Asia/Kolkata') >= :start_date 
                      AND DATE(created_at AT TIME ZONE 'Asia/Kolkata') <= :end_date
                    GROUP BY sale_date, sale_hour
                ) daily_sales
                GROUP BY sale_hour
                ORDER BY sale_hour
            """), {
                'start_date': start_date,
                'end_date': end_date
            }).fetchall()
            
            # Build hourly recommendations for business hours 6am-11pm (6-23)
            recommendations = []
            for hour in range(6, 24):  # Only 6am to 11pm
                # Get footfall data for this hour
                footfall_data = next((f for f in footfall_query if f.hour == hour), None)
                sales_data = next((s for s in sales_query if s.sale_hour == hour), None)
                
                # Always include the hour, even if no data (show 0, 0)
                # if not footfall_data and not sales_data:
                #     continue  # OLD: Skip hours with no data
                
                avg_visitors = float(footfall_data.avg_visitors) if footfall_data else 0
                max_visitors = int(footfall_data.max_visitors) if footfall_data else 0
                avg_orders = float(sales_data.avg_orders) if sales_data else 0
                max_orders = int(sales_data.max_orders) if sales_data else 0
                avg_revenue = float(sales_data.avg_revenue) if sales_data else 0
                
                # Calculate demand score (0-100)
                demand_score = min(100, int((avg_visitors / 50 * 60) + (avg_orders / 30 * 40)))
                
                # Determine demand level
                if demand_score >= 75:
                    demand_level = "Very High"
                    demand_color = "#ef4444"  # Red
                elif demand_score >= 50:
                    demand_level = "High"
                    demand_color = "#f59e0b"  # Orange
                elif demand_score >= 25:
                    demand_level = "Moderate"
                    demand_color = "#3b82f6"  # Blue
                else:
                    demand_level = "Low"
                    demand_color = "#22c55e"  # Green
                
                # Staffing recommendations
                # Base: 2 staff, +1 per 15 visitors, +1 per 10 orders
                base_staff = 2
                visitor_based_staff = int(avg_visitors / 15)
                order_based_staff = int(avg_orders / 10)
                recommended_staff = max(base_staff, base_staff + visitor_based_staff + order_based_staff)
                recommended_staff = min(recommended_staff, 12)  # Cap at 12
                
                # Inventory recommendations (in units/servings)
                # Assume each order = 2 items average, add 30% buffer for peak
                base_inventory = int(avg_orders * 2.3)
                peak_buffer = int(max_orders * 0.5) if demand_score >= 50 else 0
                recommended_inventory = base_inventory + peak_buffer
                
                hour_label = datetime.strptime(str(hour), '%H').strftime('%I %p').lstrip('0')
                
                recommendations.append({
                    'hour': hour,
                    'hour_label': hour_label,
                    'hour_range': f"{hour}:00 - {hour}:59",
                    'avg_visitors': round(avg_visitors, 1),
                    'max_visitors': max_visitors,
                    'avg_orders': round(avg_orders, 1),
                    'max_orders': max_orders,
                    'avg_revenue': round(avg_revenue, 2),
                    'demand_score': demand_score,
                    'demand_level': demand_level,
                    'demand_color': demand_color,
                    'recommended_staff': recommended_staff,
                    'recommended_inventory': recommended_inventory,
                    'notes': f"Plan for {recommended_staff} staff members and stock {recommended_inventory} units"
                })
            
            # Calculate summary statistics
            total_avg_visitors = sum(r['avg_visitors'] for r in recommendations)
            total_avg_orders = sum(r['avg_orders'] for r in recommendations)
            peak_hours = sorted(recommendations, key=lambda x: x['demand_score'], reverse=True)[:5]
            
            return jsonify({
                'date_range': {
                    'start': start_date.strftime('%Y-%m-%d'),
                    'end': end_date.strftime('%Y-%m-%d'),
                    'days': days
                },
                'summary': {
                    'total_hours_analyzed': len(recommendations),
                    'avg_daily_visitors': round(total_avg_visitors, 1),
                    'avg_daily_orders': round(total_avg_orders, 1),
                    'peak_hours': [h['hour_label'] for h in peak_hours],
                    'max_concurrent_staff_needed': max(r['recommended_staff'] for r in recommendations) if recommendations else 0
                },
                'hourly_recommendations': recommendations,
                'peak_hours_detail': peak_hours
            })
    
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
    
    FIXED: Corrected hour-to-period mapping to match sales analytics
    """
    if not db_connected:
        return jsonify({"error": "Database not connected"}), 500
    
    try:
        # Get date range (default: last 30 days)
        days = request.args.get('days', 30, type=int)
        restaurant_id = request.args.get('restaurant_id', type=int)
        end_date_ist = datetime.now(IST).date()
        start_date_ist = end_date_ist - timedelta(days=days - 1)
        
        logging.info(f"📅 IST Date Range: {start_date_ist} to {end_date_ist}")
        logging.info(f"📅 Days requested: {days}")
        logging.info(f"🏪 Restaurant ID filter: {restaurant_id}")
        
        # Get current time for filtering TODAY's future orders
        current_time_ist = datetime.now(IST)
        current_date = current_time_ist.date()
        current_hour = current_time_ist.hour
        is_viewing_today = end_date_ist == current_date
        
        logging.info(f"🕐 Current IST: {current_time_ist}, Hour: {current_hour}")
        logging.info(f"📍 Viewing today: {is_viewing_today}")
        
        # Fetch from REMOTE API (same as sales analytics and conversion analytics)
        FASTAPI_URL = 'http://13.202.92.108:8000'
        API_TOKEN = 'Z4N8T2W9L3H6Q1P'
        
        seen_order_ids = {}
        
        try:
            # Query remote API with expanded date range to account for timezone differences
            # Add 1 day buffer on both sides since remote API might be in UTC
            api_start_date = start_date_ist - timedelta(days=1)
            api_end_date = end_date_ist + timedelta(days=1)
            
            logging.info(f"🌐 Querying remote API with buffer: {api_start_date} to {api_end_date}")
            
            # Build params - NOTE: Remote API doesn't support restaurant_id, we filter client-side
            params = {
                'start_date': api_start_date.isoformat(),
                'end_date': api_end_date.isoformat(),
                'token': API_TOKEN
            }
            
            # Get PetPooja restaurant ID for client-side filtering
            petpooja_rest_id = None
            if restaurant_id:
                with SessionLocal() as db:
                    petpooja_rest_id = get_petpooja_restaurant_id(restaurant_id, db)
                    if petpooja_rest_id:
                        logging.info(f"🏪 Will filter by PetPooja restaurant_id: {petpooja_rest_id} (client-side)")
            
            response = requests.get(
                f'{FASTAPI_URL}/webhook/events/search/date-range',
                params=params,
                timeout=30
            )
            response.raise_for_status()
            all_events = response.json()
            logging.info(f"✅ Fetched {len(all_events)} webhook events from remote API")
            
            # First pass: Filter and extract order data
            filtered_orders = {}
            restaurant_filtered_count = 0  # Track how many were filtered by restaurant
            
            for event in all_events:
                if event.get('content', {}).get('event') == 'orderdetails':
                    # Client-side restaurant filtering (since remote API doesn't support it)
                    if petpooja_rest_id:
                        event_rest_id = event.get('content', {}).get('properties', {}).get('Restaurant', {}).get('restID')
                        if event_rest_id != petpooja_rest_id:
                            restaurant_filtered_count += 1
                            continue  # Skip orders from other restaurants
                    
                    order = event.get('content', {}).get('properties', {}).get('Order', {})
                    order_id = order.get('orderID')
                    
                    if order_id:
                        try:
                            # Parse order time and convert to IST for filtering
                            if order.get('created_on'):
                                order_time_utc = datetime.fromisoformat(order['created_on'].replace('Z', '+00:00'))
                                order_time_ist = order_time_utc.astimezone(IST)
                            else:
                                created_at = event.get('created_at', '')
                                if created_at:
                                    order_time_utc = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                                    order_time_ist = order_time_utc.astimezone(IST)
                                else:
                                    continue  # Skip if no timestamp
                            
                            order_date_ist = order_time_ist.date()
                            order_hour_ist = order_time_ist.hour
                            
                            # Filter by IST date range FIRST
                            if order_date_ist < start_date_ist or order_date_ist > end_date_ist:
                                continue
                            
                            # REMOVED: Future hour filtering to match Sales Analytics behavior
                            # All orders within date range are now included
                            
                            # NOW deduplicate - keep highest event_id for each order_id
                            event_id = event.get('id', 0)
                            if order_id not in filtered_orders or event_id > filtered_orders[order_id]['event_id']:
                                filtered_orders[order_id] = {
                                    'event_id': event_id,
                                    'order': order,
                                    'items': event.get('content', {}).get('properties', {}).get('OrderItem', []),
                                    'created_on': order.get('created_on'),
                                    'created_at': event.get('created_at'),
                                    'order_hour': order_hour_ist,
                                    'order_date': order_date_ist
                                }
                        except Exception as parse_error:
                            logging.warning(f"Error parsing order {order_id} timestamp: {parse_error}")
                            continue
            
            # Use filtered and deduplicated orders
            seen_order_ids = filtered_orders
            if petpooja_rest_id:
                logging.info(f"🏪 Restaurant filter applied: Excluded {restaurant_filtered_count} orders from other restaurants")
            logging.info(f"✅ Processed {len(seen_order_ids)} unique orders after IST filtering and deduplication")
            
        except Exception as e:
            logging.error(f"❌ Failed to fetch from remote API: {e}")
            import traceback
            traceback.print_exc()
            seen_order_ids = {}
        
        # Initialize item collections by time period
        morning_items = {}   # hours 6-11
        afternoon_items = {} # hours 12-16
        evening_items = {}   # hours 17-23, 0-5
        
        # Track orders by date
        orders_by_date = {}
        actual_start_date = None
        actual_end_date = None
        hour_distribution = defaultdict(int)
        
        logging.info(f"📊 Processing {len(seen_order_ids)} orders for time period analysis...")
        
        # Process each order and categorize by hour
        for order_id, order_data in seen_order_ids.items():
            try:
                # Use the hour and date already extracted
                hour = order_data.get('order_hour', 12)
                order_date = order_data.get('order_date', end_date_ist)
                
                hour_distribution[hour] += 1
                
                # Map early-morning orders (00:00 - 05:59) to the previous calendar date
                # so they are attributed to the previous day's "evening" period.
                period_date = order_date
                if 0 <= hour <= 5:
                    period_date = order_date - timedelta(days=1)
                
                # Use period_date when computing actual start/end and orders_by_date
                if actual_start_date is None or period_date < actual_start_date:
                    actual_start_date = period_date
                if actual_end_date is None or period_date > actual_end_date:
                    actual_end_date = period_date
                    
                if period_date not in orders_by_date:
                    orders_by_date[period_date] = 0
                orders_by_date[period_date] += 1
            except Exception as order_error:
                logging.warning(f"Error processing order {order_id}: {order_error}")
                hour = 12  # Default to noon if parsing fails
            
            # Calculate order-level revenue for accurate allocation
            order_total = float(order_data.get('order', {}).get('total', 0))
            order_items = order_data.get('items', [])
            
            # Calculate sum of item totals to find allocation ratio
            items_subtotal = sum(float(item.get('total', 0)) for item in order_items if isinstance(item, dict))
            
            # Process each item in the order
            for item in order_items:
                if isinstance(item, dict):
                    item_name = item.get('name', 'Unknown')
                    quantity = float(item.get('quantity', 0))
                    item_subtotal = float(item.get('total', 0))
                    
                    if not item_name or item_name == 'Unknown':
                        continue
                    
                    # Allocate order total proportionally to match sales analytics
                    # This accounts for taxes, discounts, and delivery charges
                    if items_subtotal > 0:
                        allocation_ratio = item_subtotal / items_subtotal
                        revenue = order_total * allocation_ratio
                    else:
                        revenue = item_subtotal
                    
                    # FIXED: Correct time period categorization
                    # Morning: 6-11 (6 AM to 11:59 AM)
                    # Afternoon: 12-16 (12 PM to 4:59 PM)
                    # Evening: 17-23, 0-5 (5 PM to 5:59 AM next day)
                    
                    if 6 <= hour <= 11:  # Morning
                        if item_name not in morning_items:
                            morning_items[item_name] = {'quantity': 0, 'revenue': 0}
                        morning_items[item_name]['quantity'] += quantity
                        morning_items[item_name]['revenue'] += revenue
                    
                    elif 12 <= hour <= 16:  # Afternoon
                        if item_name not in afternoon_items:
                            afternoon_items[item_name] = {'quantity': 0, 'revenue': 0}
                        afternoon_items[item_name]['quantity'] += quantity
                        afternoon_items[item_name]['revenue'] += revenue
                    
                    else:  # Evening (17-23, 0-5)
                        if item_name not in evening_items:
                            evening_items[item_name] = {'quantity': 0, 'revenue': 0}
                        evening_items[item_name]['quantity'] += quantity
                        evening_items[item_name]['revenue'] += revenue
        
        logging.info(f"💰 Revenue Summary - Morning: ₹{sum(i['revenue'] for i in morning_items.values()):.2f}, Afternoon: ₹{sum(i['revenue'] for i in afternoon_items.values()):.2f}, Evening: ₹{sum(i['revenue'] for i in evening_items.values()):.2f}")
        
        # Convert to sorted lists (top 5)
        morning_list = [{'name': k, 'quantity': v['quantity'], 'revenue': v['revenue']} for k, v in morning_items.items()]
        afternoon_list = [{'name': k, 'quantity': v['quantity'], 'revenue': v['revenue']} for k, v in afternoon_items.items()]
        evening_list = [{'name': k, 'quantity': v['quantity'], 'revenue': v['revenue']} for k, v in evening_items.items()]
        
        morning_top5 = sorted(morning_list, key=lambda x: x['revenue'], reverse=True)[:5]
        afternoon_top5 = sorted(afternoon_list, key=lambda x: x['revenue'], reverse=True)[:5]
        evening_top5 = sorted(evening_list, key=lambda x: x['revenue'], reverse=True)[:5]
        
        # Get footfall data from LOCAL database (using IST dates)
        with SessionLocal() as db:
            footfall_query = db.execute(text("""
                SELECT 
                    CASE 
                        WHEN hour BETWEEN 6 AND 11 THEN 'morning'
                        WHEN hour BETWEEN 12 AND 16 THEN 'afternoon'
                        ELSE 'evening'
                    END as time_period,
                    SUM(in_count) as total_visitors,
                    COUNT(DISTINCT report_date) as days_count
                FROM hourly_footfall
                WHERE report_date >= :start_date
                  AND report_date <= :end_date
                GROUP BY 
                    CASE 
                        WHEN hour BETWEEN 6 AND 11 THEN 'morning'
                        WHEN hour BETWEEN 12 AND 16 THEN 'afternoon'
                        ELSE 'evening'
                    END
            """), {
                'start_date': start_date_ist,
                'end_date': end_date_ist
            }).fetchall()
            
            footfall_data = {
                'morning': {'visitors': 0, 'days_count': 0},
                'afternoon': {'visitors': 0, 'days_count': 0},
                'evening': {'visitors': 0, 'days_count': 0}
            }
            
            logging.info(f"📊 Raw footfall query results: {len(footfall_query)} rows")
            for row in footfall_query:
                logging.info(f"  - {row.time_period}: {row.total_visitors} visitors, {row.days_count} days")
                if row.time_period in footfall_data:
                    footfall_data[row.time_period]['visitors'] = int(row.total_visitors) if row.total_visitors else 0
                    footfall_data[row.time_period]['days_count'] = int(row.days_count) if row.days_count else 0
        
        # Calculate metrics - IMPORTANT: Adjust for current time when viewing today
        # If viewing only today and current hour hasn't reached that time period yet, don't count today in the average
        is_viewing_today = is_viewing_today and (end_date_ist == current_date)
        
        morning_days = footfall_data['morning']['days_count']
        afternoon_days = footfall_data['afternoon']['days_count']
        evening_days = footfall_data['evening']['days_count']
        
        if is_viewing_today and days == 1:
            # If current time is before 6 AM, morning hasn't started today
            if current_hour < 6:
                morning_days = max(0, morning_days - 1)
                logging.info(f"⏰ Current time {current_hour}:xx is before morning (6 AM) - excluding today from morning average")
            
            # If current time is before 12 PM, afternoon hasn't started today
            if current_hour < 12:
                afternoon_days = max(0, afternoon_days - 1)
                logging.info(f"⏰ Current time {current_hour}:xx is before afternoon (12 PM) - excluding today from afternoon average")
            
            # If current time is before 5 PM (17:00), evening hasn't started today
            if current_hour < 17:
                evening_days = max(0, evening_days - 1)
                logging.info(f"⏰ Current time {current_hour}:xx is before evening (5 PM) - excluding today from evening average")
        
        morning_avg_footfall = (footfall_data['morning']['visitors'] / morning_days) if morning_days > 0 else 0
        afternoon_avg_footfall = (footfall_data['afternoon']['visitors'] / afternoon_days) if afternoon_days > 0 else 0
        evening_avg_footfall = (footfall_data['evening']['visitors'] / evening_days) if evening_days > 0 else 0
        
        morning_total_sales = sum(item['revenue'] for item in morning_list)
        afternoon_total_sales = sum(item['revenue'] for item in afternoon_list)
        evening_total_sales = sum(item['revenue'] for item in evening_list)
        
        total_all_sales = morning_total_sales + afternoon_total_sales + evening_total_sales
        
        # Log breakdown by hour for verification
        logging.info(f"💰 Revenue Breakdown by Time Period:")
        logging.info(f"  - Morning (6-11):     ₹{morning_total_sales:.2f} ({len(morning_list)} unique items)")
        logging.info(f"  - Afternoon (12-16):  ₹{afternoon_total_sales:.2f} ({len(afternoon_list)} unique items)")
        logging.info(f"  - Evening (others):   ₹{evening_total_sales:.2f} ({len(evening_list)} unique items)")
        logging.info(f"  - TOTAL:              ₹{total_all_sales:.2f}")
        
        logging.info(f"📊 Hour Distribution in Orders:")
        for hour in sorted(hour_distribution.keys()):
            period = "Morning" if 6 <= hour <= 11 else "Afternoon" if 12 <= hour <= 16 else "Evening"
            logging.info(f"  - Hour {hour:02d} ({period}): {hour_distribution[hour]} orders")
        
        # Generate intelligent suggestions based on actual menu data
        suggestions = []
        
        # Helper function to detect item type and suggest appropriate variations
        def get_item_suggestions(item_name, revenue):
            name_lower = item_name.lower()
            
            # Detect beverages
            beverage_keywords = ['coffee', 'cappuccino', 'latte', 'espresso', 'tea', 'chai', 'juice', 
                                'shake', 'smoothie', 'mojito', 'lassi', 'milk', 'frappe', 'americano', 
                                'mocha', 'macchiato', 'hot chocolate', 'cold coffee']
            is_beverage = any(keyword in name_lower for keyword in beverage_keywords)
            
            # Detect sandwiches/burgers
            sandwich_keywords = ['sandwich', 'burger', 'toast', 'panini', 'wrap', 'roll']
            is_sandwich = any(keyword in name_lower for keyword in sandwich_keywords)
            
            # Detect pizza/pasta
            italian_keywords = ['pizza', 'pasta', 'lasagna', 'ravioli']
            is_italian = any(keyword in name_lower for keyword in italian_keywords)
            
            if is_beverage:
                return {
                    'variations': 'size options (Regular/Large), flavor shots (Vanilla/Caramel/Hazelnut), temperature (Hot/Iced)',
                    'combo_with': 'breakfast items or pastries',
                    'upsell': 'extra shot, whipped cream, or cookie pairing'
                }
            elif is_sandwich:
                return {
                    'variations': 'spice levels (mild/medium/spicy), cheese options (regular/premium), bread choices',
                    'combo_with': 'fries, beverage, or salad',
                    'upsell': 'extra cheese, bacon, or make it a combo'
                }
            elif is_italian:
                return {
                    'variations': 'size (personal/medium/large), crust types, topping combinations',
                    'combo_with': 'garlic bread, beverage, or dessert',
                    'upsell': 'extra toppings, stuffed crust, or side salad'
                }
            else:
                # Generic food items
                return {
                    'variations': 'portion sizes, spice levels, or add-on toppings',
                    'combo_with': 'beverage or side dish',
                    'upsell': 'extra portions or premium ingredients'
                }
        
        # Calculate metrics per period
        morning_orders = len([o for o in seen_order_ids.values() if 6 <= o.get('order_hour', 0) <= 11])
        afternoon_orders = len([o for o in seen_order_ids.values() if 12 <= o.get('order_hour', 0) <= 16])
        evening_orders = len([o for o in seen_order_ids.values() if o.get('order_hour', 0) >= 17 or o.get('order_hour', 0) <= 5])
        
        morning_aov = morning_total_sales / morning_orders if morning_orders > 0 else 0
        afternoon_aov = afternoon_total_sales / afternoon_orders if afternoon_orders > 0 else 0
        evening_aov = evening_total_sales / evening_orders if evening_orders > 0 else 0
        
        logging.info(f"📊 Order Analysis - Morning: {morning_orders} orders (AOV: ₹{morning_aov:.0f}), Afternoon: {afternoon_orders} orders (AOV: ₹{afternoon_aov:.0f}), Evening: {evening_orders} orders (AOV: ₹{evening_aov:.0f})")
        
        # Morning Analysis (6 AM - 12 PM)
        if morning_orders > 0:
            top_morning = sorted(morning_list, key=lambda x: x['quantity'], reverse=True)[:3]
            top_item = top_morning[0]['name']
            item_suggestions = get_item_suggestions(top_item, top_morning[0]['revenue'])
            
            if morning_aov < 250:
                # Low AOV - suggest combos with popular items
                top_items_str = ", ".join([item['name'] for item in top_morning])
                suggestions.append({
                    'period': 'Morning (6 AM - 12 PM)',
                    'type': 'combo',
                    'icon': '☕',
                    'reason': f'Average order value is ₹{morning_aov:.0f}. Top sellers: {top_items_str}',
                    'suggestion': f'Create breakfast combos: {top_item} + {item_suggestions["combo_with"]} at 15% discount to increase AOV to ₹300+'
                })
            elif morning_aov >= 250 and morning_orders < 15:
                # Good AOV but low orders - attract more customers
                suggestions.append({
                    'period': 'Morning (6 AM - 12 PM)',
                    'type': 'promotion',
                    'icon': '🎁',
                    'reason': f'Good order value (₹{morning_aov:.0f}) but only {morning_orders} orders',
                    'suggestion': f'Launch "Early Bird Special" (before 10 AM): Get 20% off on {top_item} to drive morning traffic'
                })
            else:
                # Strong performance - upsell opportunities
                suggestions.append({
                    'period': 'Morning (6 AM - 12 PM)',
                    'type': 'upsell',
                    'icon': '⬆️',
                    'reason': f'Strong performance: {morning_orders} orders at ₹{morning_aov:.0f} AOV',
                    'suggestion': f'Upsell strategy: Offer {item_suggestions["upsell"]} with {top_item} to push AOV to ₹350+'
                })
        
        # Afternoon Analysis (12 PM - 5 PM)
        if afternoon_orders > 0:
            top_afternoon = sorted(afternoon_list, key=lambda x: x['quantity'], reverse=True)[:3]
            top_item = top_afternoon[0]['name']
            item_suggestions = get_item_suggestions(top_item, top_afternoon[0]['revenue'])
            
            if afternoon_aov < 300:
                top_items_str = ", ".join([item['name'] for item in top_afternoon])
                suggestions.append({
                    'period': 'Afternoon (12 PM - 5 PM)',
                    'type': 'combo',
                    'icon': '🍱',
                    'reason': f'Average order value is ₹{afternoon_aov:.0f}. Most ordered: {top_items_str}',
                    'suggestion': f'Create "Lunch Deal": {top_item} + {item_suggestions["combo_with"]} at bundled price to boost AOV'
                })
            elif afternoon_orders < 20:
                suggestions.append({
                    'period': 'Afternoon (12 PM - 5 PM)',
                    'type': 'promotion',
                    'icon': '⏰',
                    'reason': f'Peak lunch hours but only {afternoon_orders} orders',
                    'suggestion': f'Introduce "Express Lunch" (12-2 PM): Fast service guarantee + {top_item} combo deals to capture office crowd'
                })
            else:
                # Peak period - maximize revenue
                suggestions.append({
                    'period': 'Afternoon (12 PM - 5 PM)',
                    'type': 'premium',
                    'icon': '⭐',
                    'reason': f'Peak period: {afternoon_orders} orders, ₹{afternoon_aov:.0f} AOV',
                    'suggestion': f'Launch premium option: {top_item} with {item_suggestions["upsell"]} at ₹{int(afternoon_aov * 1.3)} to target high-value customers'
                })
        
        # Evening Analysis (5 PM - 6 AM)
        if evening_orders > 0:
            top_evening = sorted(evening_list, key=lambda x: x['quantity'], reverse=True)[:3]
            top_item = top_evening[0]['name'] if top_evening else "menu items"
            item_suggestions = get_item_suggestions(top_item, top_evening[0]['revenue']) if top_evening else None
            
            if evening_aov < 350:
                top_items_str = ", ".join([item['name'] for item in top_evening])
                combo_suggestion = item_suggestions["combo_with"] if item_suggestions else "sides and drinks"
                suggestions.append({
                    'period': 'Evening (5 PM - 6 AM)',
                    'type': 'combo',
                    'icon': '🌙',
                    'reason': f'Dinner period with ₹{evening_aov:.0f} AOV. Popular: {top_items_str}',
                    'suggestion': f'Create "Dinner For Two": 2x {top_item} + {combo_suggestion} at ₹{int(evening_aov * 2.2)} value price'
                })
            elif evening_orders < 10:
                suggestions.append({
                    'period': 'Evening (5 PM - 6 AM)',
                    'type': 'promotion',
                    'icon': '🎉',
                    'reason': f'Evening potential untapped - only {evening_orders} orders',
                    'suggestion': f'Happy Hours (5-7 PM): Buy {top_item}, get 50% off second item + free beverage'
                })
            else:
                suggestions.append({
                    'period': 'Evening (5 PM - 6 AM)',
                    'type': 'family',
                    'icon': '👨‍👩‍👧',
                    'reason': f'Dinner rush: {evening_orders} orders at ₹{evening_aov:.0f} AOV',
                    'suggestion': f'Family Bundle: 4x {top_item} + family-size {item_suggestions["combo_with"] if item_suggestions else "sides"} at ₹{int(evening_aov * 3.5)}'
                })
        
        # Add cross-period insights
        if len(seen_order_ids) > 0:
            # Find most consistent seller across all periods
            all_items = {}
            for period_items in [morning_items, afternoon_items, evening_items]:
                for name, data in period_items.items():
                    if name not in all_items:
                        all_items[name] = {'count': 0, 'revenue': 0}
                    all_items[name]['count'] += 1  # Present in how many periods
                    all_items[name]['revenue'] += data['revenue']
            
            if all_items:
                consistent_sellers = sorted(
                    [(name, data) for name, data in all_items.items() if data['count'] >= 2],
                    key=lambda x: x[1]['revenue'],
                    reverse=True
                )
                
                if consistent_sellers:
                    bestseller = consistent_sellers[0]
                    bestseller_name = bestseller[0]
                    item_suggestions = get_item_suggestions(bestseller_name, bestseller[1]['revenue'])
                    
                    suggestions.append({
                        'period': 'All Day Strategy',
                        'type': 'signature',
                        'icon': '🏆',
                        'reason': f'{bestseller_name} is popular across multiple time periods (₹{bestseller[1]["revenue"]:.0f} total)',
                        'suggestion': f'Make {bestseller_name} your "Signature Item" - feature it prominently and offer {item_suggestions["variations"]} to boost sales further'
                    })
        
        # Use actual date range from data
        display_start_date = actual_start_date if actual_start_date else start_date_ist
        display_end_date = actual_end_date if actual_end_date else end_date_ist
        actual_days = len(orders_by_date) if orders_by_date else 0
        
        # Add note if viewing today's data (up to current time)
        time_note = f"Data up to {current_time_ist.strftime('%I:%M %p').lstrip('0')}" if is_viewing_today else "Full day data"
        
        logging.info(f"📅 Actual date range: {display_start_date} to {display_end_date} (IST)")
        logging.info(f"⏰ {time_note}")
        
        # Return comprehensive data
        return jsonify({
            'success': True,
            'date_range': {
                'start': display_start_date.strftime('%Y-%m-%d'),
                'end': display_end_date.strftime('%Y-%m-%d'),
                'days': actual_days,
                'requested_days': days,
                'orders_by_date': {date.strftime('%Y-%m-%d'): count for date, count in orders_by_date.items()},
                'timezone': 'IST',
                'viewing_today': is_viewing_today,
                'current_time': current_time_ist.strftime('%Y-%m-%d %I:%M %p').lstrip('0') if is_viewing_today else None,
                'note': time_note
            },
            'morning': {
                'period': '6 AM - 12 PM',
                'top_items': morning_top5,
                'total_items': len(morning_list),
                'total_revenue': round(morning_total_sales, 2),
                'total_quantity': sum(item['quantity'] for item in morning_list),
                'footfall': {
                    'total_visitors': footfall_data['morning']['visitors'],
                    'avg_per_day': round(morning_avg_footfall),
                    'days_tracked': footfall_data['morning']['days_count']
                }
            },
            'afternoon': {
                'period': '12 PM - 5 PM',
                'top_items': afternoon_top5,
                'total_items': len(afternoon_list),
                'total_revenue': round(afternoon_total_sales, 2),
                'total_quantity': sum(item['quantity'] for item in afternoon_list),
                'footfall': {
                    'total_visitors': footfall_data['afternoon']['visitors'],
                    'avg_per_day': round(afternoon_avg_footfall),
                    'days_tracked': footfall_data['afternoon']['days_count']
                }
            },
            'evening': {
                'period': '5 PM - 6 AM (Evening + Late Night)',
                'top_items': evening_top5,
                'total_items': len(evening_list),
                'total_revenue': round(evening_total_sales, 2),
                'total_quantity': sum(item['quantity'] for item in evening_list),
                'footfall': {
                    'total_visitors': footfall_data['evening']['visitors'],
                    'avg_per_day': round(evening_avg_footfall),
                    'days_tracked': footfall_data['evening']['days_count']
                },
                'note': 'Includes late-night orders (11 PM - 6 AM)'
            },
            'suggestions': suggestions,
            'has_data': len(morning_list) > 0 or len(afternoon_list) > 0 or len(evening_list) > 0,
            'data_source': 'remote_api'
        })
            
    except Exception as e:
        logging.error(f"Error in menu_time_popularity: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500
# ============================================================================

_MODEL_CACHE = {}

def load_model(model_path: str):
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return None
    try:
        model = YOLO(model_path)
        # FORCE CPU MODE - No CUDA
        model.to('cpu')
        
        # No half precision or fuse on CPU
        # try:
        #     model.fuse()
        # except Exception:
        #     pass
        
        # Warmup with CPU
        try:
            import numpy as _np
            dummy = _np.zeros((640, 640, 3), dtype=_np.uint8)
            with torch.inference_mode():
                for _ in range(2):  # Reduced warmup iterations for CPU
                    _ = model(dummy, conf=0.25, iou=0.45, imgsz=640, device='cpu', verbose=False)
        except Exception:
            pass
        
        logging.info(f"Loaded '{model_path}' on cpu (half=False)")
        _MODEL_CACHE[model_path] = model
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
            model_obj = load_model(APP_TASKS_CONFIG['PeopleCounter']['model_path'])
            if model_obj:
                pc_processor = PeopleCounterProcessor(link, channel_id, channel_name, model_obj, handle_detection, socketio)
                pc_processor.frame_hub = hub
                stream_processors[channel_id].append(pc_processor); pc_processor.start()
                logging.info(f"Started PeopleCounter for {channel_id} ({channel_name}).")
                atexit.register(pc_processor.shutdown); active_app_names.remove('PeopleCounter')
        if 'QueueMonitor' in active_app_names:
            model_obj = load_model(APP_TASKS_CONFIG['QueueMonitor']['model_path'])
            if model_obj:
                # Pass restaurant_id to QueueMonitor processor
                restaurant_id = restaurant.id if restaurant else None
                qm_processor = QueueMonitorProcessor(link, channel_id, channel_name, model_obj, restaurant_id=restaurant_id)
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
            model_obj = load_model(APP_TASKS_CONFIG['OccupancyMonitor']['model_path'])
            if model_obj:
                om_processor = OccupancyMonitorProcessor(
                    link, channel_id, channel_name, model_obj, socketio, 
                    SessionLocal, send_telegram_notification
                )
                om_processor.frame_hub = hub
                stream_processors[channel_id].append(om_processor)
                om_processor.start()
                logging.info(f"Started OccupancyMonitor for {channel_id} ({channel_name}).")
                atexit.register(om_processor.shutdown)
                active_app_names.remove('OccupancyMonitor')
        
        if 'IdlePeopleViolation' in active_app_names:
            # IdlePeopleViolationProcessor loads its own model internally
            ipv_processor = IdlePeopleViolationProcessor(
                link, channel_id, channel_name, SessionLocal, socketio,
                send_telegram_notification, handle_detection
            )
            if hasattr(ipv_processor, 'frame_hub'):
                ipv_processor.frame_hub = hub
            stream_processors[channel_id].append(ipv_processor)
            ipv_processor.start()
            logging.info(f"Started IdlePeopleViolation for {channel_id} ({channel_name}).")
            atexit.register(ipv_processor.shutdown)
            active_app_names.remove('IdlePeopleViolation')
        
        if active_app_names:
            tasks_for_multi_model = []
            for app_name in active_app_names:
                config = APP_TASKS_CONFIG.get(app_name)
                if config and 'model_path' in config:
                    logging.info(f"Loading model for {app_name}: {config['model_path']}")
                    model_obj = load_model(config['model_path'])
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
                new_processor = PeopleCounterProcessor(rtsp_url, channel_id, channel_name, model_obj, handle_detection, socketio)
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
            logging.info("Scheduler started (CPU-only mode - no CUDA recovery needed)")
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