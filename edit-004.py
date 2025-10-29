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
from apscheduler.schedulers.background import BackgroundScheduler
from shapely.geometry import Point, Polygon

# --- Module Imports ---
from kitchen_compliance_monitor import KitchenComplianceProcessor

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('ultralytics').setLevel(logging.WARNING)
logging.getLogger('apscheduler').setLevel(logging.WARNING)

# --- Master Configuration ---
IST = pytz.timezone('Asia/Kolkata')
DATABASE_URL = "sqlite:///./cctv_dashboard.db"
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


# --- App Task Configuration ---
APP_TASKS_CONFIG = {
    'Generic': {'model_path': 'best_generic.pt', 'target_class_id': [1, 2, 3, 4, 5, 6, 7], 'confidence': 0.8, 'is_gif': True},
    'PeopleCounter': {'model_path': 'yolov8n.pt' , 'confidence': 0.2},
    'QueueMonitor': {'model_path': 'yolov8n.pt' , 'confidence': 0.2},
    'KitchenCompliance': {'model_path': 'yolov8n.pt', 'apron_cap_model': 'apron-cap.pt', 'gloves_model': 'gloves.pt', 'confidence': 0.5}
}


# --- QUEUE MONITOR CONFIGURATION ---
# THIS IS NOW A FALLBACK if no ROI is in the database.
QUEUE_MONITOR_ROI_CONFIG = {
    "Checkout Queue": {
        "roi_points": [[0.399, 0.181], [0.163, 0.425], [0.361, 0.931], [0.761, 0.653]],
        "secondary_roi_points": [[0.436, 0.288], [0.624, 0.509], [0.846, 0.438], [0.643, 0.19]],
    }
}
QUEUE_DWELL_TIME_SEC = 1.0
QUEUE_ALERT_THRESHOLD = 2
QUEUE_ALERT_COOLDOWN_SEC = 60

# --- Flask and SocketIO Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-very-secret-key-for-sakshi-ai'
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Global State Management ---
stream_processors = {}

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
    ts_string = timestamp.strftime("%Y%m%d_%H%M%S")
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

class MultiModelProcessor(threading.Thread):
    def __init__(self, rtsp_url, channel_id, channel_name, tasks, detection_callback):
        super().__init__()
        self.rtsp_url, self.channel_id, self.channel_name, self.tasks, self.detection_callback = rtsp_url, channel_id, channel_name, tasks, detection_callback
        self.is_running = True
        self.last_detection_times = {task['app_name']: 0 for task in self.tasks}
        self.cooldown, self.gif_duration_seconds, self.fps = 30, 3, 10

    def stop(self): self.is_running = False
    def shutdown(self):
        logging.info(f"Shutting down MultiModel for {self.channel_name} ({self.channel_id})")
        self.is_running = False

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            logging.error(f"Could not open stream for {self.channel_id}: {self.rtsp_url}")
            return
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Reconnecting to stream {self.channel_id}..."); time.sleep(5)
                cap.release(); cap = cv2.VideoCapture(self.rtsp_url); continue

            current_time = time.time()

            for task in self.tasks:
                app_name = task['app_name']
                if app_name in ['PeopleCounter', 'QueueMonitor']: continue

                else:
                    if current_time - self.last_detection_times[app_name] > self.cooldown:
                        model_args = {'conf': task['confidence'], 'verbose': False}
                        if task.get('target_class_id') is not None:
                            model_args['classes'] = task['target_class_id']

                        results = task['model'](frame, **model_args)

                        if results and len(results[0].boxes) > 0:
                            self.last_detection_times[app_name] = current_time
                            if task['is_gif']:
                                frames_to_capture = self.gif_duration_seconds * self.fps
                                gif_frames = [results[0].plot()]
                                for _ in range(frames_to_capture - 1):
                                    ret_gif, frame_gif = cap.read()
                                    if not ret_gif: break
                                    gif_frames.append(frame_gif.copy())
                                    time.sleep(1 / self.fps)
                                self.detection_callback(app_name, self.channel_id, gif_frames, f"{app_name} detected.", True)
                            else:
                                annotated_frame = results[0].plot()
                                self.detection_callback(app_name, self.channel_id, [annotated_frame], f"{app_name} detected.", False)
        cap.release()

class PeopleCounterProcessor(threading.Thread):
    def __init__(self, rtsp_url, channel_id, channel_name, model, detection_callback, socketio):
        super().__init__()
        self.rtsp_url, self.channel_id, self.model, self.detection_callback = rtsp_url, channel_id, model, detection_callback
        self.channel_name, self.app_name = channel_name, "PeopleCounter"
        self.socketio = socketio
        self.is_running, self.lock = True, threading.Lock()
        self.track_history = defaultdict(list)
        self.counts = {'in': 0, 'out': 0}
        self.last_saved_total_counts = {'in': 0, 'out': 0}
        self.last_saved_hour = datetime.now(IST).hour
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
                self.last_saved_total_counts = self.counts.copy()
            except Exception as e: logging.error(f"Failed to load initial counts: {e}")

    def _reset_counts_for_new_day(self, db, new_date):
        self.counts = {'in': 0, 'out': 0}
        self.tracking_date = new_date
        db.add(DailyFootfall(channel_id=self.channel_id, report_date=new_date, in_count=0, out_count=0))
        db.commit()

    def _update_and_log_counts(self):
        if not db_connected: return
        with SessionLocal() as db, self.lock:
            try:
                db.query(DailyFootfall).filter_by(channel_id=self.channel_id, report_date=self.tracking_date).update({'in_count': self.counts['in'], 'out_count': self.counts['out']})
                current_hour_ist = datetime.now(IST).hour
                if current_hour_ist != self.last_saved_hour:
                    hourly_in, hourly_out = self.counts['in'] - self.last_saved_total_counts['in'], self.counts['out'] - self.last_saved_total_counts['out']
                    if hourly_in > 0 or hourly_out > 0:
                        stmt = text("""
                            INSERT INTO hourly_footfall (channel_id, report_date, hour, in_count, out_count)
                            VALUES (:cid, :rdate, :hour, :inc, :outc)
                            ON CONFLICT (channel_id, report_date, hour)
                            DO UPDATE SET in_count = hourly_footfall.in_count + EXCLUDED.in_count,
                                          out_count = hourly_footfall.out_count + EXCLUDED.out_count;
                        """)
                        db.execute(stmt, {'cid': self.channel_id, 'rdate': self.tracking_date, 'hour': self.last_saved_hour, 'inc': hourly_in, 'outc': hourly_out})
                    self.last_saved_total_counts, self.last_saved_hour = self.counts.copy(), current_hour_ist
                db.commit()
            except Exception as e:
                logging.error(f"Error updating counts in DB: {e}"); db.rollback()

    def _check_for_new_day(self):
        current_date_ist = datetime.now(IST).date()
        if current_date_ist > self.tracking_date:
            logging.info("New day detected. Resetting people counter.")
            self._update_and_log_counts()
            with SessionLocal() as db:
                self._reset_counts_for_new_day(db, current_date_ist)
                self.last_saved_total_counts = self.counts.copy()
                self.last_saved_hour = datetime.now(IST).hour

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        while self.is_running:
            self._check_for_new_day()
            ret, frame = cap.read()
            if not ret:
                time.sleep(5); cap.release(); cap = cv2.VideoCapture(self.rtsp_url); continue
            results = self.model.track(frame, persist=True, classes=[0], conf=0.5, iou=0.5, verbose=False)
            if results and results[0].boxes.id is not None:
                boxes, track_ids = results[0].boxes.xywh.cpu(), results[0].boxes.id.int().cpu().tolist()
                line_x, count_changed = int(frame.shape[1] * 0.5), False
                for box, track_id in zip(boxes, track_ids):
                    center_x = int(box[0])
                    history = self.track_history[track_id]
                    history.append(center_x)
                    if len(history) > 2: history.pop(0)
                    if len(history) == 2:
                        prev_x, curr_x = history
                        if prev_x < line_x and curr_x >= line_x: self.counts['out'] += 1; count_changed = True
                        elif prev_x > line_x and curr_x <= line_x: self.counts['in'] += 1; count_changed = True
                if count_changed: self._update_and_log_counts()
            annotated_frame = results[0].plot() if results else frame
            line_x = int(annotated_frame.shape[1] * 0.5)
            cv2.line(annotated_frame, (line_x, 0), (line_x, annotated_frame.shape[0]), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"IN: {self.counts['in']}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"OUT: {self.counts['out']}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            with self.lock: self.latest_frame = annotated_frame.copy()
            socketio.emit('count_update', {'channel_id': self.channel_id, 'in_count': self.counts['in'], 'out_count': self.counts['out']})
        cap.release()

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
        self.last_alert_time = 0
        self.roi_poly = Polygon([])
        self.secondary_roi_poly = Polygon([])
        self._load_roi_from_db()

    def _load_roi_from_db(self):
        with SessionLocal() as db:
            roi_record = db.query(RoiConfig).filter_by(channel_id=self.channel_id, app_name='QueueMonitor').first()
            if roi_record and roi_record.roi_points:
                try:
                    points = json.loads(roi_record.roi_points)
                    self.roi_poly = Polygon(points.get("main", []))
                    self.secondary_roi_poly = Polygon(points.get("secondary", []))
                    logging.info(f"Loaded custom ROI for QueueMonitor {self.channel_name} from DB.")
                except (json.JSONDecodeError, TypeError):
                    logging.error("Failed to parse ROI JSON from DB. Using fallback.")
                    self._use_fallback_roi()
            else:
                logging.warning(f"No custom ROI in DB for QueueMonitor {self.channel_name}. Using fallback.")
                self._use_fallback_roi()

    def _use_fallback_roi(self):
        fallback_config = QUEUE_MONITOR_ROI_CONFIG.get(self.channel_name, {})
        # Store normalized coordinates for later conversion to pixels
        self.normalized_main_roi = fallback_config.get("roi_points", [])
        self.normalized_secondary_roi = fallback_config.get("secondary_roi_points", [])
        # Initialize with empty polygons - will be converted to pixels in run() method
        self.roi_poly = Polygon([])
        self.secondary_roi_poly = Polygon([])

    def update_roi(self, new_roi_points):
        with self.lock:
            try:
                self.normalized_main_roi = new_roi_points.get("main", [])
                self.normalized_secondary_roi = new_roi_points.get("secondary", [])
                logging.info(f"QueueMonitor {self.channel_name} received new ROI config.")
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

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            logging.error(f"Could not open QueueMonitor stream for {self.channel_name}")
            return

        first_frame = True
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Reconnecting to QueueMonitor stream {self.channel_name}..."); time.sleep(5)
                cap.release(); cap = cv2.VideoCapture(self.rtsp_url); continue
            
            if first_frame:
                h, w, _ = frame.shape
                if hasattr(self, 'normalized_main_roi') and self.normalized_main_roi:
                    self.roi_poly = Polygon([(int(p[0]*w), int(p[1]*h)) for p in self.normalized_main_roi])
                if hasattr(self, 'normalized_secondary_roi') and self.normalized_secondary_roi:
                    self.secondary_roi_poly = Polygon([(int(p[0]*w), int(p[1]*h)) for p in self.normalized_secondary_roi])
                first_frame = False

            self.process_frame(frame.copy())

        cap.release()

    def process_frame(self, frame):
        current_time = time.time()
        results = self.model.track(frame, persist=True, classes=[0], verbose=False, conf=0.4)
        current_tracks_in_main_roi, people_in_secondary_roi = set(), 0

        if results and results[0].boxes.id is not None:
            for box, track_id in zip(results[0].boxes.xyxy.cpu(), results[0].boxes.id.int().cpu().tolist()):
                person_point = Point(int((box[0] + box[2]) / 2), int(box[3]))
                if self.roi_poly.is_valid and self.roi_poly.contains(person_point):
                    current_tracks_in_main_roi.add(track_id)
                    tracker = self.queue_tracker[track_id]
                    if tracker['entry_time'] == 0: tracker['entry_time'] = current_time
                if self.secondary_roi_poly.is_valid and self.secondary_roi_poly.contains(person_point):
                    people_in_secondary_roi += 1

        valid_queue_count = sum(1 for track_id in list(self.queue_tracker.keys()) if (track_id in current_tracks_in_main_roi and (current_time - self.queue_tracker[track_id]['entry_time']) >= QUEUE_DWELL_TIME_SEC) or (self.queue_tracker.pop(track_id) and False))
        if self.current_queue_count != valid_queue_count:
            self.current_queue_count = valid_queue_count
            socketio.emit('queue_update', {'channel_id': self.channel_id, 'count': self.current_queue_count})

        # Alert when cashier area is empty and queue has 2+ people (with cooldown)
        should_alert = (
            valid_queue_count >= 2 and
            people_in_secondary_roi == 0 and
            (current_time - self.last_alert_time) > QUEUE_ALERT_COOLDOWN_SEC
        )

        annotated_frame = results[0].plot() if results and results[0] else frame
        
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
        if should_alert:
            self.last_alert_time = current_time
            alert_message = f"Queue is full ({valid_queue_count} people), but the counter is free."
            logging.warning(f"QUEUE ALERT on {self.channel_name}: {alert_message}")
            send_telegram_notification(f"🚨 **Queue Alert: {self.channel_name}** 🚨\n{alert_message}")
            handle_detection('QueueMonitor', self.channel_id, [annotated_frame], alert_message, is_gif=False)

        cv2.putText(annotated_frame, f"Queue: {self.current_queue_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"Counter Area: {people_in_secondary_roi}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        with self.lock: self.latest_frame = annotated_frame.copy()


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
    while True:
        time.sleep(0.05)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + processor.get_frame() + b'\r\n\r\n')

@app.route('/video_feed/<app_name>/<channel_id>')
@login_required
def video_feed(app_name, channel_id):
    processors = stream_processors.get(channel_id)
    if not processors: return ("Stream not found", 404)
    target_processor, target_class = None, None
    if app_name == 'PeopleCounter': target_class = PeopleCounterProcessor
    elif app_name == 'QueueMonitor': target_class = QueueMonitorProcessor
    elif app_name == 'KitchenCompliance': target_class = KitchenComplianceProcessor
    if target_class: target_processor = next((p for p in processors if isinstance(p, target_class)), None)
    if target_processor and target_processor.is_alive():
        return Response(gen_video_feed(target_processor), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return (f"{app_name} stream not found or is not running for this channel", 404)


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
    if report_date == datetime.now(IST).date():
        pc_processor = next((p for p in stream_processors.get(channel_id, []) if isinstance(p, PeopleCounterProcessor)), None)
        if pc_processor:
            with pc_processor.lock:
                current_hour = datetime.now(IST).hour
                hourly_data[current_hour]['in'] += pc_processor.counts['in'] - pc_processor.last_saved_total_counts['in']
                hourly_data[current_hour]['out'] += pc_processor.counts['out'] - pc_processor.last_saved_total_counts['out']
                hourly_data[current_hour]['total'] = hourly_data[current_hour]['in'] + hourly_data[current_hour]['out']
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



@socketio.on('connect')
def handle_connect(): logging.info('Frontend client connected')

def load_model(model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists(model_path): logging.error(f"Model file not found: {model_path}"); return None
    try:
        model = YOLO(model_path); model.to(device)
        logging.info(f"Successfully loaded model '{model_path}' on '{device}'")
        return model
    except Exception as e: logging.error(f"Failed to load model '{model_path}': {e}"); return None

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
        if 'PeopleCounter' in active_app_names:
            model_obj = load_model(APP_TASKS_CONFIG['PeopleCounter']['model_path'])
            if model_obj:
                pc_processor = PeopleCounterProcessor(link, channel_id, channel_name, model_obj, handle_detection, socketio)
                stream_processors[channel_id].append(pc_processor); pc_processor.start()
                logging.info(f"Started PeopleCounter for {channel_id} ({channel_name}).")
                atexit.register(pc_processor.shutdown); active_app_names.remove('PeopleCounter')
        if 'QueueMonitor' in active_app_names:
            model_obj = load_model(APP_TASKS_CONFIG['QueueMonitor']['model_path'])
            if model_obj:
                qm_processor = QueueMonitorProcessor(link, channel_id, channel_name, model_obj)
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
                stream_processors[channel_id].append(kc_processor)
                kc_processor.start()
                logging.info(f"Started KitchenCompliance for {channel_id} ({channel_name}).")
                atexit.register(kc_processor.shutdown)
                active_app_names.remove('KitchenCompliance')
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
                stream_processors[channel_id].append(multi_processor); multi_processor.start()
                task_names = [t['app_name'] for t in tasks_for_multi_model]
                logging.info(f"Started MultiModel for {channel_id} ({channel_name}) with tasks: {task_names}.")
                atexit.register(multi_processor.shutdown)

if __name__ == "__main__":
    if initialize_database():
        scheduler = BackgroundScheduler(timezone=str(IST))
        scheduler.add_job(log_queue_counts, 'interval', minutes=5)
        scheduler.start()
        atexit.register(lambda: scheduler.shutdown())
        start_streams()
        logging.info("Starting Flask-SocketIO server on http://0.0.0.0:5004")
        socketio.run(app, host='0.0.0.0', port=5006, debug=False, allow_unsafe_werkzeug=True)

