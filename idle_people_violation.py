import cv2
import torch
from ultralytics import YOLO
import threading
import time
from datetime import datetime
from collections import defaultdict
import os
import logging
import pytz
import numpy as np
from sqlalchemy import Column, Integer, String, DateTime, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import Point, Polygon

# --- Basic Configuration ---
IST = pytz.timezone('Asia/Kolkata')
Base = declarative_base()

# --- Model Configuration ---
MODEL_PATH = 'models/yolo11n.pt'  # YOLO11n - Better for detecting people at all distances
PERSON_CLASS_ID = 0  # Person class in COCO dataset
CONFIDENCE_THRESHOLD = 0.1  # Detection confidence threshold (0.1 = very sensitive, catches distant people too)
IDLE_FRAME_THRESHOLD = 15  # Number of frames a person must be detected to be considered idle (adjustable)
FRAME_SKIP_RATE = 1  # Process every frame for maximum accuracy (was 2)
ALERT_COOLDOWN_SECONDS = 60  # Cooldown between alerts for same person

# --- Database Table Definition ---
class IdlePeopleViolation(Base):
    __tablename__ = "idle_people_violations"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    channel_name = Column(String)
    timestamp = Column(DateTime, default=lambda: datetime.now(IST))
    person_id = Column(Integer)  # Tracked person ID
    frame_count = Column(Integer)  # Number of frames person was idle
    details = Column(String)
    media_path = Column(String)
    __table_args__ = (UniqueConstraint('media_path', name='_idle_media_path_uc'),)


class IdlePeopleViolationProcessor(threading.Thread):
    """
    Processor for detecting idle people in camera feeds.
    Tracks people using YOLO object detection and flags violations when a person
    is detected for more than IDLE_FRAME_THRESHOLD consecutive frames.
    """
    
    def __init__(self, rtsp_url, channel_id, channel_name, SessionLocal, socketio, telegram_sender, detection_callback):
        super().__init__(name=f"IdlePeople-{channel_name}")
        self.rtsp_url = rtsp_url
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.is_running = True
        self.error_message = None
        self.latest_frame = None
        self.lock = threading.Lock()

        self.SessionLocal = SessionLocal
        self.socketio = socketio
        self.send_telegram_notification = telegram_sender
        self.handle_main_detection = detection_callback

        # Person tracking dictionary: {track_id: frame_count}
        self.person_tracker = defaultdict(int)
        self.last_alert_time = defaultdict(float)  # Track last alert time per person
        
        # ROI (Region of Interest) configuration
        self.roi_polygon = None
        self._load_roi()
        
        # Load YOLO model
        try:
            self.device = 'cpu'  # Use CPU for stability
            logging.info(f"Idle People Monitor {self.channel_name} using device: CPU")
            
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
            
            self.model = YOLO(MODEL_PATH)
            self.model.to(self.device)
            
            logging.info(f"✅ Idle People {self.channel_name}: Loaded model {MODEL_PATH}")
            logging.info(f"   Monitoring for idle people (threshold: {IDLE_FRAME_THRESHOLD} frames)")
            
        except Exception as e:
            self.error_message = f"Model Error: {e}"
            logging.error(f"FATAL: Failed to initialize Idle People model for {self.channel_name}. Error: {e}")

        self.alert_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="IdlePeopleAlert")
        
        # FPS tracking
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        self.frame_counter = 0  # For frame skipping

    @staticmethod
    def initialize_tables(engine):
        """Create database table for idle people violations"""
        try:
            Base.metadata.create_all(bind=engine)
            logging.info("Table 'idle_people_violations' checked/created.")
        except Exception as e:
            logging.error(f"Could not create 'idle_people_violations' table: {e}")

    def _load_roi(self):
        """Load ROI configuration from database"""
        try:
            from sqlalchemy import text
            with self.SessionLocal() as db:
                query = text("""
                    SELECT roi_points FROM roi_configs 
                    WHERE channel_id = :channel_id AND app_name = 'IdlePeopleViolation'
                """)
                result = db.execute(query, {"channel_id": self.channel_id}).fetchone()
                
                if result and result[0]:
                    import json
                    roi_data = result[0] if isinstance(result[0], dict) else json.loads(result[0])
                    points = roi_data.get('points', [])
                    
                    if points and len(points) >= 3:
                        self.roi_polygon = Polygon(points)
                        logging.info(f"✅ Loaded ROI for {self.channel_name}: {len(points)} points")
                    else:
                        logging.info(f"No valid ROI configured for {self.channel_name} - monitoring entire frame")
                else:
                    logging.info(f"No ROI configured for {self.channel_name} - monitoring entire frame")
        except Exception as e:
            logging.error(f"Error loading ROI for {self.channel_name}: {e}")
            self.roi_polygon = None

    def _is_in_roi(self, x1, y1, x2, y2):
        """Check if bounding box intersects/overlaps with the ROI polygon"""
        if self.roi_polygon is None:
            return True  # No ROI means monitor entire frame
        
        try:
            # Create a polygon from the bounding box coordinates
            bbox_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            # Check if bounding box intersects or overlaps with ROI
            return self.roi_polygon.intersects(bbox_polygon)
        except Exception as e:
            logging.error(f"Error checking ROI: {e}")
            return True  # Default to allowing detection on error

    def _draw_roi(self, frame):
        """Draw ROI polygon on frame"""
        if self.roi_polygon is None:
            return
        
        try:
            points = np.array(self.roi_polygon.exterior.coords, dtype=np.int32)
            cv2.polylines(frame, [points], isClosed=True, color=(255, 255, 0), thickness=2)
            # Fill with semi-transparent overlay
            overlay = frame.copy()
            cv2.fillPoly(overlay, [points], color=(255, 255, 0))
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        except Exception as e:
            logging.error(f"Error drawing ROI: {e}")

    def stop(self):
        self.is_running = False

    def shutdown(self):
        logging.info(f"Shutting down Idle People processor for {self.channel_name}.")
        self.is_running = False
        if hasattr(self, 'alert_executor'):
            self.alert_executor.shutdown(wait=False)

    def get_frame(self):
        """Get latest frame with annotations for display"""
        with self.lock:
            if self.error_message:
                placeholder = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(placeholder, f'Error: {self.error_message}', (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                _, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()
            
            if self.latest_frame is not None:
                success, jpeg = cv2.imencode('.jpg', self.latest_frame)
                return jpeg.tobytes() if success else b''
            else:
                placeholder = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(placeholder, 'Connecting...', (180, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (201, 209, 217), 2)
                _, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()

    def _save_violation_to_db(self, person_id, frame_count, details, media_path):
        """Save violation to database"""
        with self.SessionLocal() as db:
            try:
                # Avoid duplicate entries
                existing = db.query(IdlePeopleViolation).filter_by(media_path=media_path).first()
                if existing:
                    return
                violation = IdlePeopleViolation(
                    channel_id=self.channel_id, 
                    channel_name=self.channel_name,
                    person_id=person_id,
                    frame_count=frame_count,
                    details=details, 
                    media_path=media_path
                )
                db.add(violation)
                db.commit()
                logging.info(f"💾 Saved idle person violation to DB: Person {person_id}, Frames: {frame_count}")
            except Exception as e:
                logging.error(f"Failed to save idle person violation to DB: {e}")
                db.rollback()

    def _trigger_alert(self, frame, person_id, frame_count, bbox):
        """Trigger alert for idle person violation"""
        details = f"Person ID {person_id} idle for {frame_count} frames"
        logging.warning(f"🚨 IDLE PERSON ALERT on {self.channel_name}: {details}")
        
        # Run telegram and screenshot saving in background thread pool
        def async_alert():
            try:
                telegram_message = f"🚨 Idle Person Alert: {self.channel_name}\nPerson ID: {person_id}\nIdle Duration: {frame_count} frames\nLocation: {bbox}"
                self.send_telegram_notification(telegram_message)
                
                # Create frame with bounding box annotation (without ROI polygon)
                annotated_frame = frame.copy()
                x1, y1, x2, y2 = bbox
                
                # Draw red bounding box for violation
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw label with background
                label = f"IDLE! ID:{person_id} Frames:{frame_count}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), 
                            (x1 + label_w, y1), (0, 0, 255), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                media_path = self.handle_main_detection(
                    'IdlePeopleViolation', self.channel_id, [annotated_frame], details, is_gif=False
                )
                
                if media_path:
                    self._save_violation_to_db(person_id, frame_count, details, media_path)
                    
                logging.info(f"Idle People: Alert saved for Person {person_id}")
            except Exception as e:
                logging.error(f"Idle People alert background task failed: {e}")
        
        self.alert_executor.submit(async_alert)

    def _update_fps(self):
        """Update FPS calculation"""
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = time.time()

    def run(self):
        """Main processing loop"""
        logging.info(f"🚀 Idle People Violation thread starting for {self.channel_name}")
        
        if self.error_message:
            logging.error(f"Thread exiting early due to error: {self.error_message}")
            return

        cap = None
        reconnect_delay = 5
        
        try:
            while self.is_running:
                try:
                    if cap is None or not cap.isOpened():
                        logging.info(f"Connecting to RTSP: {self.rtsp_url}")
                        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        if not cap.isOpened():
                            self.error_message = "Failed to connect to RTSP stream"
                            logging.error(f"{self.channel_name}: {self.error_message}")
                            time.sleep(reconnect_delay)
                            continue
                        
                        logging.info(f"✅ Connected to {self.channel_name}")
                        self.error_message = None

                    ret, frame = cap.read()
                    if not ret:
                        logging.warning(f"{self.channel_name}: Failed to read frame")
                        cap.release()
                        cap = None
                        time.sleep(reconnect_delay)
                        continue

                    self.frame_counter += 1
                    self._update_fps()

                    # Make a copy for display
                    display_frame = frame.copy()
                    
                    # Draw ROI if configured
                    self._draw_roi(display_frame)
                    
                    # Run YOLO inference with tracking
                    results = self.model.track(frame, persist=True, classes=[PERSON_CLASS_ID], 
                                              conf=CONFIDENCE_THRESHOLD, verbose=False)

                    current_tracked_ids = set()
                    
                    if results and len(results) > 0:
                        result = results[0]
                        
                        if result.boxes is not None and len(result.boxes) > 0:
                            boxes = result.boxes.cpu().numpy()
                            
                            for box in boxes:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])
                                
                                # Check if bounding box intersects with ROI
                                if not self._is_in_roi(x1, y1, x2, y2):
                                    continue  # Skip people whose box doesn't intersect ROI
                                
                                # Get track ID if available
                                track_id = int(box.id[0]) if box.id is not None else None
                                
                                if track_id is not None:
                                    current_tracked_ids.add(track_id)
                                    
                                    # Increment frame count for this person
                                    self.person_tracker[track_id] += 1
                                    frame_count = self.person_tracker[track_id]
                                    
                                    # Determine color based on idle status
                                    if frame_count >= IDLE_FRAME_THRESHOLD:
                                        # VIOLATION: Person is idle
                                        color = (0, 0, 255)  # Red
                                        label = f"IDLE! ID:{track_id} Frames:{frame_count}"
                                        
                                        # Check if we should trigger alert (cooldown)
                                        current_time = time.time()
                                        last_alert = self.last_alert_time.get(track_id, 0)
                                        
                                        if current_time - last_alert > ALERT_COOLDOWN_SECONDS:
                                            self._trigger_alert(frame, track_id, frame_count, (x1, y1, x2, y2))
                                            self.last_alert_time[track_id] = current_time
                                    else:
                                        # Normal tracking
                                        color = (0, 255, 0)  # Green
                                        label = f"ID:{track_id} Frames:{frame_count}/{IDLE_FRAME_THRESHOLD}"
                                    
                                    # Draw bounding box
                                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                                    
                                    # Draw label with background
                                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                    cv2.rectangle(display_frame, (x1, y1 - label_h - 10), 
                                                (x1 + label_w, y1), color, -1)
                                    cv2.putText(display_frame, label, (x1, y1 - 5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Clean up tracking for people who left the frame
                    disappeared_ids = set(self.person_tracker.keys()) - current_tracked_ids
                    for track_id in disappeared_ids:
                        del self.person_tracker[track_id]
                        if track_id in self.last_alert_time:
                            del self.last_alert_time[track_id]

                    # Add FPS and info overlay
                    cv2.putText(display_frame, f"FPS: {self.current_fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"People Tracked: {len(current_tracked_ids)}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Idle Threshold: {IDLE_FRAME_THRESHOLD} frames", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Update latest frame for streaming
                    with self.lock:
                        self.latest_frame = display_frame

                except Exception as e:
                    logging.error(f"Error in idle people processing loop: {e}", exc_info=True)
                    time.sleep(1)

        finally:
            if cap:
                cap.release()
            logging.info(f"Idle People thread for {self.channel_name} has stopped.")
