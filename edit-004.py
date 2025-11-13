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
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Text, text, UniqueConstraint
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

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('ultralytics').setLevel(logging.WARNING)
logging.getLogger('apscheduler').setLevel(logging.WARNING)

# --- Master Configuration ---
IST = pytz.timezone('Asia/Kolkata')
DATABASE_URL = "postgresql://postgres:Tneural01@127.0.0.1:5432/sakshi"
RTSP_LINKS_FILE = 'rtsp_links.txt'
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
    'Generic': {'model_path': 'best_generic.pt', 'target_class_id': [1, 2, 3, 4, 5, 6, 7], 'confidence': 0.8, 'is_gif': True},
    'PeopleCounter': {'model_path': 'yolo11n.pt' , 'confidence': 0.15},
    'QueueMonitor': {'model_path': 'yolov8n.pt' , 'confidence': 0.15},
    'KitchenCompliance': {'model_path': 'yolov8n.pt', 'apron_cap_model': 'apron-cap.pt', 'gloves_model': 'gloves.pt', 'confidence': 0.5},
    'OccupancyMonitor': {'model_path': 'yolo11n.pt', 'confidence': 0.15}
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
        "roi_points": [[0.391,0.206],[0.732,0.553],[0.356,0.668],[0.247,0.311]], #[[0.436, 0.288], [0.624, 0.509], [0.846, 0.438], [0.643, 0.19]],
        "secondary_roi_points":[[0.451,0.273],[0.598,0.214],[0.803,0.371],[0.635,0.508]],#[[0.399, 0.181], [0.163, 0.425], [0.361, 0.931], [0.861, 0.653]],
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



class RoiConfig(Base):
    __tablename__ = "roi_configs"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    app_name = Column(String, index=True)
    roi_points = Column(Text) # Storing as JSON string
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
        self.cooldown, self.gif_duration_seconds, self.fps = 30, 3, 10
        self.expected_frame_shape = None  # Track expected frame dimensions
        self.consecutive_invalid_frames = 0  # Track consecutive invalid frames
        self.consecutive_errors = 0  # Track consecutive CUDA errors
        self.max_consecutive_errors = 10

    def stop(self): self.is_running = False
    def shutdown(self):
        logging.info(f"Shutting down MultiModel for {self.channel_name} ({self.channel_id})")
        self.is_running = False

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
        while self.is_running:
            frame = getattr(self, 'frame_hub', None).get_latest() if hasattr(self, 'frame_hub') else None
            if frame is None:
                time.sleep(0.01)
                continue

            # Validate frame before processing
            if not self._validate_frame(frame):
                time.sleep(0.01)
                continue

            current_time = time.time()

            for task in self.tasks:
                app_name = task['app_name']
                if app_name in ['PeopleCounter', 'QueueMonitor']: continue

                else:
                    if current_time - self.last_detection_times[app_name] > self.cooldown:
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

                        if results and len(results[0].boxes) > 0:
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
                                    self.detection_callback(app_name, self.channel_id, gif_frames, f"{app_name} detected.", True)
                                else:
                                    # Fallback to single frame if we couldn't capture enough frames
                                    self.detection_callback(app_name, self.channel_id, gif_frames, f"{app_name} detected.", False)
                            else:
                                annotated_frame = results[0].plot()
                                self.detection_callback(app_name, self.channel_id, [annotated_frame], f"{app_name} detected.", False)
        # No cap to release when using FrameHub

class PeopleCounterProcessor(threading.Thread):
    def __init__(self, rtsp_url, channel_id, channel_name, model, detection_callback, socketio):
        super().__init__()
        self.rtsp_url, self.channel_id, self.model, self.detection_callback = rtsp_url, channel_id, model, detection_callback
        self.channel_name, self.app_name = channel_name, "PeopleCounter"
        self.socketio = socketio
        self.is_running, self.lock = True, threading.Lock()
        
        # LINE CROSSING APPROACH - Simple & Reliable!
        self.previous_centroids = []  # List of (x, y) from previous frame
        self.counting_line_position = 0.45  # Line at 45% (LEFT=0-45%, RIGHT=45-100%)
        self.cooldown_zones = {}  # {(approx_x, approx_y): timestamp} to prevent double counting
        self.cooldown_duration = 0.8  # 800ms cooldown per zone
        
        self.counts = {'in': 0, 'out': 0}
        self.current_hour = datetime.now(IST).hour
        self.tracking_date = datetime.now(IST).date()
        self.latest_frame = None
        
        self._load_initial_counts()

        self.socketio.emit('count_update', {'channel_id': self.channel_id, 'in_count': self.counts['in'], 'out_count': self.counts['out']})

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
                
                # Create a completely fresh annotated frame to prevent cross-contamination
                # Start with a deep copy of the original frame
                annotated_frame = frame.copy()
                
                # Draw only valid person detections on our own copy (filtered)
                if r0 is not None and getattr(r0, 'boxes', None) is not None:
                    boxes = r0.boxes
                    frame_area = frame.shape[0] * frame.shape[1]
                    min_box_area = frame_area * 0.003  # Match counting filter
                    max_box_area = frame_area * 0.9    # Match counting filter
                    min_confidence = 0.30              # Match counting filter
                    
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        conf = float(boxes.conf[i].cpu())
                        
                        # Apply same filtering as counting logic
                        box_width = x2 - x1
                        box_height = y2 - y1
                        box_area = box_width * box_height
                        aspect_ratio = box_height / box_width if box_width > 0 else 0
                        
                        is_person_shaped = 1.2 <= aspect_ratio <= 4.0
                        is_valid_size = min_box_area <= box_area <= max_box_area
                        is_confident = conf >= min_confidence
                        
                        # Only draw if it passes person filters
                        if is_person_shaped and is_valid_size and is_confident:
                            # Draw bounding box (GREEN for valid person)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # Draw confidence label
                            label = f"Person {conf:.2f}"
                            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            # Draw rejected detections in RED with reason
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                            reason = "Small" if box_area < min_box_area else "Large" if box_area > max_box_area else "Shape" if not is_person_shaped else "LowConf"
                            cv2.putText(annotated_frame, f"X {reason}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                # Draw counting line and zones
                # Counting line at 45% (LEFT zone: 0-45%, RIGHT zone: 45-100%)
                cv2.line(annotated_frame, (counting_line_x, 0), (counting_line_x, frame_height), (0, 255, 255), 2)  # Yellow line
                cv2.putText(annotated_frame, "COUNTING LINE (45%)", (counting_line_x + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Draw LEFT zone label
                cv2.putText(annotated_frame, "LEFT (45%)", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                # Draw RIGHT zone label
                cv2.putText(annotated_frame, "RIGHT (55%)", (counting_line_x + 20, frame_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                frame_height = annotated_frame.shape[0]
                frame_width = annotated_frame.shape[1]
                center_line = int(frame_width * 0.45)  # 45% boundary

                # Draw CENTER LINE (GREEN) to separate the two zones
                cv2.line(annotated_frame, (center_line, 0), (center_line, frame_height), (0, 255, 0), 3)
                
                # Draw LEFT ZONE (BLUE) border - 40% of frame
                cv2.rectangle(annotated_frame, (0, 0), (center_line, frame_height), (255, 0, 0), 3)
                cv2.putText(annotated_frame, "LEFT 45%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Draw RIGHT ZONE (RED) border - 60% of frame
                cv2.rectangle(annotated_frame, (center_line, 0), (frame_width, frame_height), (0, 0, 255), 3)
                cv2.putText(annotated_frame, "RIGHT 55%", (center_line + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display both total daily count and current hour count (matching DB)
                current_hour_ist = datetime.now(IST).hour
                current_date_ist = datetime.now(IST).date()
                
                # Update current_hour if hour changed
                if current_hour_ist != self.current_hour:
                    self.current_hour = current_hour_ist
                
                # Fetch current hour counts from database (updated in real-time)
                hourly_in_current = 0
                hourly_out_current = 0
                try:
                    with SessionLocal() as db:
                        saved_hourly = db.query(HourlyFootfall).filter_by(
                            channel_id=self.channel_id,
                            report_date=current_date_ist,
                            hour=current_hour_ist
                        ).first()
                        if saved_hourly:
                            hourly_in_current = saved_hourly.in_count
                            hourly_out_current = saved_hourly.out_count
                except Exception as e:
                    logging.warning(f"Could not fetch hourly count from DB: {e}")
                
                # Display total daily counts
                cv2.putText(annotated_frame, f"TOTAL - IN: {self.counts['in']}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"TOTAL - OUT: {self.counts['out']}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # Display current hour counts (matching what's stored in DB for current hour)
                #cv2.putText(annotated_frame, f"HOUR {current_hour_ist:02d}:00 - IN: {hourly_in_current}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                #cv2.putText(annotated_frame, f"HOUR {current_hour_ist:02d}:00 - OUT: {hourly_out_current}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                with self.lock: self.latest_frame = annotated_frame.copy()
                socketio.emit('count_update', {'channel_id': self.channel_id, 'in_count': self.counts['in'], 'out_count': self.counts['out']})
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
    def __init__(self, rtsp_url, channel_id, channel_name, model):
        super().__init__(name=channel_name)
        self.rtsp_url = rtsp_url
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.model = model
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
    # REPLACE IT WITH THIS
    def _load_roi_from_db(self):
        logging.info(f"BACKEND_FIX: Forcing use of hardcoded ROI for {self.channel_name}")
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
                logging.info(f"QueueMonitor {self.channel_name} received new ROI config.")
                # Reset the flag so ROI polygons will be updated on next frame
                if hasattr(self, '_roi_logged_once'):
                    delattr(self, '_roi_logged_once')
            except Exception as e:
                logging.error(f"Error updating ROI for {self.channel_name}: {e}")

    def shutdown(self):
        logging.info(f"Shutting down QueueMonitor for {self.channel_name}.")
        self.is_running = False

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
        
        if r0 is not None and getattr(r0, 'boxes', None) is not None and getattr(r0.boxes, 'id', None) is not None:
            boxes = r0.boxes.xyxy.cpu()
            track_ids = r0.boxes.id.int().cpu().tolist()
            
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
            # No screenshot when counter area becomes occupied - only when queue present and counter empty

        # Emit live counts (no DB persistence) if either changed
        if updated:
            socketio.emit('queue_update', {
                'channel_id': self.channel_id,
                'queue': self.current_queue_count,
                'counter': self.current_secondary_count,
                'count': self.current_queue_count  # backward compat
            })

        # Check for persons who have been in queue for more than 5 seconds with no one in counter area
        persons_in_queue_5sec = []
        for track_id in current_tracks_in_main_roi:
            if track_id in self.queue_tracker:
                dwell_time = current_time - self.queue_tracker[track_id]['entry_time']
                if dwell_time >= QUEUE_SCREENSHOT_DWELL_TIME_SEC:
                    persons_in_queue_5sec.append(track_id)
        
        # Screenshot trigger 1: person waiting > 5 seconds (HIGHEST PRIORITY)
        should_screenshot_5sec = (
            len(persons_in_queue_5sec) > 0 and  # Someone waited > 5 seconds
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

        # Create annotated frame - draw bounding boxes with colors based on ROI
        annotated_frame = frame.copy()
        
        # Draw bounding boxes with color based on which ROI the person is in
        if r0 is not None and getattr(r0, 'boxes', None) is not None:
            boxes = r0.boxes.xyxy.cpu()
            track_ids = r0.boxes.id.int().cpu().tolist() if getattr(r0.boxes, 'id', None) is not None else []
            confidences = r0.boxes.conf.cpu() if hasattr(r0.boxes, 'conf') else None
            
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                track_id = track_ids[idx] if idx < len(track_ids) else None
                
                # Determine box color based on ROI membership
                box_color = (255, 255, 255)  # Default white if not in any ROI
                label_color = (255, 255, 255)
                
                if track_id is not None:
                    if track_id in current_tracks_in_main_roi:
                        box_color = (0, 255, 255)  # Yellow (BGR format) for queue area
                        label_color = (0, 255, 255)
                    elif track_id in current_tracks_in_secondary_roi:
                        box_color = (255, 255, 0)  # Cyan (BGR format) for counter area
                        label_color = (255, 255, 0)
                    else:
                        box_color = (128, 128, 128)  # Gray for persons not in any ROI
                        label_color = (128, 128, 128)
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Draw label with track ID and confidence
                if confidences is not None and idx < len(confidences):
                    conf = float(confidences[idx])
                    label = f"ID:{track_id}" if track_id is not None else f"Conf:{conf:.2f}"
                    if track_id is not None:
                        label = f"ID:{track_id} ({conf:.2f})"
                else:
                    label = f"ID:{track_id}" if track_id is not None else "Person"
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 5), 
                            (x1 + label_size[0] + 5, y1), box_color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw main ROI (queue area) - Yellow
        if self.roi_poly.is_valid and not self.roi_poly.is_empty:
            cv2.polylines(annotated_frame, [np.array(self.roi_poly.exterior.coords, dtype=np.int32)], True, (255, 255, 0), 2)
            logging.debug(f"Drawing main ROI with {len(self.roi_poly.exterior.coords)} points")
        else:
            logging.warning(f"Main ROI is invalid or empty. Valid: {self.roi_poly.is_valid}, Empty: {self.roi_poly.is_empty}")
        
        # Draw secondary ROI (cashier area) - Cyan
        if self.secondary_roi_poly.is_valid and not self.secondary_roi_poly.is_empty:
            cv2.polylines(annotated_frame, [np.array(self.secondary_roi_poly.exterior.coords, dtype=np.int32)], True, (0, 255, 255), 2)
            logging.debug(f"Drawing secondary ROI with {len(self.secondary_roi_poly.exterior.coords)} points")
        else:
            logging.warning(f"Secondary ROI is invalid or empty. Valid: {self.secondary_roi_poly.is_valid}, Empty: {self.secondary_roi_poly.is_empty}")
        
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

        # Display counts prominently on the frame with better visibility
        # Background rectangles for better text visibility
        cv2.rectangle(annotated_frame, (45, 25), (280, 95), (0, 0, 0), -1)  # Black background
        cv2.rectangle(annotated_frame, (45, 25), (280, 95), (255, 255, 255), 2)  # White border
        
        # Queue count with larger, more visible font - use valid_queue_count for real-time display
        queue_display_count = valid_queue_count  # Show actual current count (main ROI = queue area)
        counter_display_count = valid_secondary_count  # Show actual current count (secondary ROI = counter area)
        
        cv2.putText(annotated_frame, f"Queue: {queue_display_count}", (50, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)  # Yellow, thicker
        cv2.putText(annotated_frame, f"Counter: {counter_display_count}", (50, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)  # Cyan, thicker
        
        # Log count changes for debugging
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
        
        # Emit to dashboard via SocketIO
        self.socketio.emit('occupancy_update', {
            'channel_id': self.channel_id,
            'channel_name': self.channel_name,
            'time_slot': self.current_time_slot,
            'live_count': self.live_count,
            'required_count': self.required_count,
            'status': status
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
                
                # Add comprehensive info overlay
                cv2.putText(annotated_frame, f"Live: {self.live_count} | Required: {self.required_count}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(annotated_frame, self.current_time_slot, (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(annotated_frame, f"Device: {self.device.upper()} | Conf: 0.15", (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # Add alert overlay if below requirement
                if self.required_count > 0 and self.live_count < self.required_count:
                    cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 120), (0, 0, 255), -1)
                    cv2.putText(annotated_frame, f"ALERT: {self.required_count - self.live_count} people short!", 
                              (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add "OK" indicator if requirement met
                elif self.required_count > 0 and self.live_count >= self.required_count:
                    cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 120), (0, 255, 0), -1)
                    cv2.putText(annotated_frame, f"REQUIREMENT MET! ({self.live_count}/{self.required_count})", 
                              (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                with self.lock:
                    self.latest_frame = annotated_frame
                
                # Check requirement
                self._check_occupancy_requirement()
                
            elif should_detect:
                # Between YOLO detections - still show smooth video with last detection overlay
                # This ensures smooth streaming without frame skip
                display_frame = frame.copy()
                
                # Reapply last detection info (smooth display)
                cv2.putText(display_frame, f"Live: {self.live_count} | Required: {self.required_count}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_frame, self.current_time_slot, (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(display_frame, f"Device: {self.device.upper()} | Conf: 0.15", (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # Add status banner
                if self.required_count > 0 and self.live_count < self.required_count:
                    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 120), (0, 0, 255), -1)
                    cv2.putText(display_frame, f"ALERT: {self.required_count - self.live_count} people short!", 
                              (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                elif self.required_count > 0:
                    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 120), (0, 255, 0), -1)
                    cv2.putText(display_frame, f"REQUIREMENT MET! ({self.live_count}/{self.required_count})", 
                              (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                with self.lock:
                    self.latest_frame = display_frame
                    
            else:
                # PAUSED/NO SCHEDULE - Show status on frame
                display_frame = frame.copy()
                
                if detection_status == "NO_SCHEDULE":
                    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 120), (100, 100, 100), -1)
                    cv2.putText(display_frame, "NO SCHEDULE FOR THIS TIME", 
                              (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                elif detection_status == "PAUSED_REQ_MET":
                    time_paused = int(time.time() - self.requirement_met_time)
                    time_remaining = self.pause_after_met_duration - time_paused
                    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 120), (0, 200, 0), -1)
                    cv2.putText(display_frame, f"REQUIREMENT MET - PAUSED", 
                              (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Resuming in {time_remaining}s", 
                              (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(display_frame, f"Time: {self.current_time_slot}", (10, 110),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
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


def get_app_configs():
    app_configs = defaultdict(lambda: {'channels': [], 'online_count': 0})
    if not os.path.exists(RTSP_LINKS_FILE): return {}
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
                            app_configs[app_name]['channels'].append({'id': channel_id, 'name': channel_status[channel_id]['name']})
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

@app.route('/dashboard')
@login_required
def dashboard(): 
    return render_template('dashboard.html', app_configs=get_app_configs())

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
    except (ValueError, TypeError): return jsonify({"error": "Invalid page or limit parameter"}), 400
    offset = (page - 1) * limit
    with SessionLocal() as db:
        try:
            query = db.query(Detection).filter(Detection.app_name == app_name)
            if channel_id and channel_id != 'null': query = query.filter(Detection.channel_id == channel_id)
            if start_date_str and end_date_str:
                try:
                    start_date, end_date = datetime.strptime(start_date_str, '%Y-%m-%d').date(), datetime.strptime(end_date_str, '%Y-%m-%d').date()
                    query = query.filter(Detection.timestamp.between(start_date, datetime.combine(end_date, datetime.max.time())))
                except ValueError: return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400
            total_detections, detections = query.count(), query.order_by(Detection.timestamp.desc()).offset(offset).limit(limit).all()
            return jsonify({'detections': [{'timestamp': d.timestamp.strftime("%Y-%m-%d %H:%M:%S"),'message': d.message,'channel_id': d.channel_id,'media_url': url_for('static', filename=d.media_path)} for d in detections],'total': total_detections, 'page': page, 'limit': limit})
        except Exception as e:
            logging.error(f"Error fetching history: {e}")
            return jsonify({"error": "Could not fetch history from database"}), 500

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
            if app_name == 'QueueMonitor': target_class = QueueMonitorProcessor
            
            if target_class:
                for p in processors:
                    if isinstance(p, target_class) and hasattr(p, 'update_roi'):
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
    if not os.path.exists(RTSP_LINKS_FILE):
        logging.error(f"'{RTSP_LINKS_FILE}' not found.")
        return
    stream_assignments = defaultdict(lambda: {'apps': set(), 'name': ''})
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
    for link, assignment in stream_assignments.items():
        channel_id, channel_name, app_names = assignment['id'], assignment['name'], list(assignment['apps'])
        if channel_id not in stream_processors: stream_processors[channel_id] = []
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
                qm_processor = QueueMonitorProcessor(link, channel_id, channel_name, model_obj)
                qm_processor.frame_hub = hub
                stream_processors[channel_id].append(qm_processor); qm_processor.start()
                logging.info(f"Started QueueMonitor for {channel_id} ({channel_name}).")
                atexit.register(qm_processor.shutdown); active_app_names.remove('QueueMonitor')
        if 'KitchenCompliance' in active_app_names:
            config = APP_TASKS_CONFIG['KitchenCompliance']
            general_model = load_model(config['model_path'])
            apron_cap_model = load_model(config['apron_cap_model'])
            gloves_model = load_model(config['gloves_model'])
            if general_model and apron_cap_model and gloves_model:
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
        if active_app_names:
            tasks_for_multi_model = []
            for app_name in active_app_names:
                config = APP_TASKS_CONFIG.get(app_name)
                if config and 'model_path' in config:
                    model_obj = load_model(config['model_path'])
                    if model_obj: tasks_for_multi_model.append({'app_name': app_name, 'model': model_obj, **config})
                    else: logging.warning(f"Skipping '{app_name}' for {channel_id}; model failed to load.")
            if tasks_for_multi_model:
                multi_processor = MultiModelProcessor(link, channel_id, channel_name, tasks_for_multi_model, handle_detection)
                multi_processor.frame_hub = hub
                stream_processors[channel_id].append(multi_processor); multi_processor.start()
                task_names = [t['app_name'] for t in tasks_for_multi_model]
                logging.info(f"Started MultiModel for {channel_id} ({channel_name}) with tasks: {task_names}.")
                atexit.register(multi_processor.shutdown)


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