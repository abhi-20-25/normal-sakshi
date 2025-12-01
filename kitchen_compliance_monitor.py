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

# --- Basic Configuration ---
IST = pytz.timezone('Asia/Kolkata')
Base = declarative_base()

# --- Model Paths (Unified Model) ---
UNIFIED_MODEL_PATH = 'final_best.pt'  # Single model for all violations
VIOLATION_CLASSES = [2, 4, 6, 7, 8]  # Classes to detect: without_uniform, without_cap, without_apron, without_gloves, using_phone

# --- Detection Configuration ---
CONFIDENCE_THRESHOLD = 0.50
FRAME_SKIP_RATE = 5
PHONE_PERSISTENCE_SECONDS = 3
ALERT_COOLDOWN_SECONDS = 20  # Reduced from 60 to capture more violations

# --- Uniform Color Ranges (HSV) ---
YELLOW_LOWER = np.array([18, 80, 80])
YELLOW_UPPER = np.array([35, 255, 255])
BLACK_LOWER = np.array([0, 0, 0])
BLACK_UPPER = np.array([180, 255, 50])

# --- Database Table Definition ---
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

class KitchenComplianceProcessor(threading.Thread):
    def __init__(self, rtsp_url, channel_id, channel_name, SessionLocal, socketio, telegram_sender, detection_callback):
        super().__init__(name=f"Kitchen-{channel_name}")
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

        try:
            # Force CPU mode - CUDA causes "double free" crashes with OpenCV cleanup
            self.device = 'cpu'
            logging.info(f"Kitchen channel {self.channel_name} using device: CPU")
            
            # Load unified model
            if not os.path.exists(UNIFIED_MODEL_PATH):
                raise FileNotFoundError(f"Missing model file: {UNIFIED_MODEL_PATH}")
            
            self.unified_model = YOLO(UNIFIED_MODEL_PATH)
            self.unified_model.to(self.device)
            
            logging.info(f"✅ Kitchen {self.channel_name}: Loaded unified model {UNIFIED_MODEL_PATH}")
            logging.info(f"   Model classes: {self.unified_model.names}")
            logging.info(f"   Monitoring violation classes: {VIOLATION_CLASSES}")
            
        except Exception as e:
            self.error_message = f"Model Error: {e}"
            logging.error(f"FATAL: Failed to initialize Kitchen models for {self.channel_name}. Error: {e}")

        self.last_alert_time = defaultdict(float)  # Track last alert time per violation type
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.last_socketio_emit = 0  # Track last SocketIO emit time
        self.alert_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="KitchenAlert")  # Limit concurrent alerts
        
        # FPS tracking
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0

    @staticmethod
    def initialize_tables(engine):
        try:
            Base.metadata.create_all(bind=engine)
            logging.info("Table 'kitchen_violations' checked/created.")
        except Exception as e:
            logging.error(f"Could not create 'kitchen_violations' table: {e}")

    def stop(self):
        self.is_running = False

    def shutdown(self):
        logging.info(f"Shutting down Kitchen Compliance processor for {self.channel_name}.")
        self.is_running = False
        if hasattr(self, 'alert_executor'):
            self.alert_executor.shutdown(wait=False)

    def get_frame(self):
        with self.lock:
            if self.error_message:
                placeholder = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(placeholder, f'Error: {self.error_message}', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                _, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()
            
            if self.latest_frame is not None:
                success, jpeg = cv2.imencode('.jpg', self.latest_frame)
                return jpeg.tobytes() if success else b''
            else:
                placeholder = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(placeholder, 'Connecting...', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (201, 209, 217), 2)
                _, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()

    def _save_violation_to_db(self, violation_type, details, media_path):
        with self.SessionLocal() as db:
            try:
                # Avoid duplicate unique media_path entries
                existing = db.query(KitchenViolation).filter_by(media_path=media_path).first()
                if existing:
                    return
                violation = KitchenViolation(
                    channel_id=self.channel_id, channel_name=self.channel_name,
                    violation_type=violation_type, details=details, media_path=media_path
                )
                db.add(violation)
                db.commit()
            except Exception as e:
                logging.error(f"Failed to save kitchen violation to DB: {e}")
                db.rollback()

    def _save_violation_screenshot(self, frame, violation_type):
        """Save screenshot of violation to static/detections folder"""
        try:
            timestamp = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
            filename = f"kitchen_{self.channel_name}_{violation_type}_{timestamp}.jpg"
            filepath = os.path.join('static', 'detections', filename)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save screenshot
            cv2.imwrite(filepath, frame)
            logging.info(f"📸 Saved violation screenshot: {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"Failed to save screenshot: {e}")
            return None

    def _trigger_alert(self, frame, violation_type, details):
        logging.warning(f"ALERT on {self.channel_name}: {details}")
        
        # Run telegram and screenshot saving in background thread pool to avoid blocking
        def async_alert():
            try:
                telegram_message = f"🚨 Kitchen Alert: {self.channel_name}\nViolation: {violation_type}\nDetails: {details}"
                self.send_telegram_notification(telegram_message)
                media_path = self.handle_main_detection(
                    'KitchenCompliance', self.channel_id, [frame], details, is_gif=False
                )
                if media_path:
                    self._save_violation_to_db(violation_type, details, media_path)
                logging.info(f"Kitchen: Alert saved for {violation_type}")
            except Exception as e:
                logging.error(f"Kitchen alert background task failed: {e}")
        
        # Submit alert to thread pool (max 2 concurrent alerts)
        self.alert_executor.submit(async_alert)

    def run(self):
        logging.info(f"🚀 Kitchen Compliance thread starting for {self.channel_name}")
        if self.error_message: 
            logging.error(f"Kitchen thread exiting early due to error: {self.error_message}")
            return
        
        # Check for test mode
        use_placeholder = os.environ.get('USE_PLACEHOLDER_FEED', 'false').lower() == 'true'
        
        if not use_placeholder:
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|timeout;5000000'
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            
            if not cap.isOpened():
                logging.warning(f"Could not open Kitchen stream for {self.channel_name}, using placeholder")
                use_placeholder = True
            else:
                is_file = any(self.rtsp_url.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov'])
        
        if use_placeholder:
            logging.info(f"Using placeholder feed for Kitchen {self.channel_name}")
            frame_counter = 0
            while self.is_running:
                frame = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(frame, f'{self.channel_name}', (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (201, 209, 217), 2)
                cv2.putText(frame, f'Camera Offline - Test Mode', (120, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 150, 255), 2)
                cv2.putText(frame, f'Frame: {frame_counter}', (230, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                
                with self.lock:
                    self.latest_frame = frame
                frame_counter += 1
                time.sleep(0.1)
            return

        is_file = any(self.rtsp_url.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov'])
        frame_count = 0
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        phone_persistence_frames = int(PHONE_PERSISTENCE_SECONDS * video_fps)

        while self.is_running:
            success, frame = cap.read()
            if not success:
                if is_file:
                    logging.info(f"Restarting video file for Kitchen {self.channel_name}...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logging.warning(f"Reconnecting to Kitchen stream {self.channel_name}...")
                    time.sleep(5)
                    cap.release()
                    cap = cv2.VideoCapture(self.rtsp_url)
                    continue

            frame_count += 1
            current_time = time.time()
            annotated_frame = frame.copy()
            
            # Calculate FPS
            self.fps_frame_count += 1
            elapsed_time = current_time - self.fps_start_time
            if elapsed_time >= 1.0:  # Update FPS every second
                self.current_fps = self.fps_frame_count / elapsed_time
                self.fps_frame_count = 0
                self.fps_start_time = current_time
            
            # Log every 100 frames (about every 3 seconds at 30 FPS)
            if frame_count % 100 == 0:
                logging.info(f"Kitchen {self.channel_name}: ✅ ALIVE - Processing frame {frame_count} | FPS: {self.current_fps:.1f}")

            # --- Run Inferences (Unified Model - Single Pass Detection) ---
            try:
                # Run model ONCE to detect both people AND violations
                results = self.unified_model(
                    frame, 
                    classes=[0] + VIOLATION_CLASSES,  # Detect person (0) + violations (2,4,6,7,8)
                    conf=0.35,  # 35% confidence threshold
                    verbose=False
                )
                
                # Separate person boxes from violation boxes
                person_boxes = []
                violation_boxes = []
                
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        if cls_id == 0:  # Person class
                            person_boxes.append(box)
                        else:  # Violation classes
                            violation_boxes.append(box)
                
                # Log detection results every 100 frames
                if frame_count % 100 == 0:
                    logging.info(f"Kitchen {self.channel_name}: Detected {len(person_boxes)} people, {len(violation_boxes)} raw violations in frame {frame_count}")
                    
            except Exception as e:
                logging.error(f"❌ Kitchen {self.channel_name}: Model inference error at frame {frame_count}: {e}")
                person_boxes = []
                violation_boxes = []

            # Draw header info
            h, w = annotated_frame.shape[:2]

            # --- Process Violations from Unified Model (Human-Verified) ---
            violation_count = 0
            violations_found = []
            
            # Get person boxes for verification
            person_box_coords = []
            if len(person_boxes) > 0:
                for person_box in person_boxes:
                    px1, py1, px2, py2 = map(int, person_box.xyxy[0].cpu().numpy())
                    person_box_coords.append([px1, py1, px2, py2])
                    
                    # Draw green boxes around detected people
                    cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 255, 0), 1)
            
            # Process violations
            if len(violation_boxes) > 0:
                for box in violation_boxes:
                    # Get box coordinates
                    vx1, vy1, vx2, vy2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    # Get class name
                    class_name = self.unified_model.names[cls_id]
                    
                    # Calculate violation box center
                    v_center_x = (vx1 + vx2) / 2
                    v_center_y = (vy1 + vy2) / 2
                    
                    # VERIFY: Check if violation is near/inside any person box
                    is_human_violation = False
                    if len(person_box_coords) > 0:
                        for person_box in person_box_coords:
                            px1, py1, px2, py2 = person_box
                            
                            # Expand person box by 20% to account for slight misalignments
                            width = px2 - px1
                            height = py2 - py1
                            expanded_px1 = px1 - width * 0.2
                            expanded_py1 = py1 - height * 0.2
                            expanded_px2 = px2 + width * 0.2
                            expanded_py2 = py2 + height * 0.2
                            
                            # Check if violation center is inside expanded person box
                            if (expanded_px1 <= v_center_x <= expanded_px2 and 
                                expanded_py1 <= v_center_y <= expanded_py2):
                                is_human_violation = True
                                break
                    
                    # Only process violations associated with people
                    if is_human_violation:
                        violations_found.append(class_name)
                        violation_count += 1
                        
                        # Draw red bounding box for verified human violation
                        cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), (0, 0, 255), 2)
                        
                        # Trigger alert if not in cooldown
                        time_since_last = current_time - self.last_alert_time[class_name]
                        if time_since_last > ALERT_COOLDOWN_SECONDS:
                            self.last_alert_time[class_name] = current_time
                            details = f"Human violation: {class_name} (confidence: {conf:.2%})"
                            
                            # Save screenshot
                            screenshot_path = self._save_violation_screenshot(frame.copy(), class_name)
                            
                            # Send alert
                            self._trigger_alert(frame.copy(), class_name, details)
                            
                            logging.info(f"🚨 Kitchen {self.channel_name}: Human {class_name} violation detected! Screenshot: {screenshot_path}")
                    else:
                        # Draw gray box for non-human violations (filtered out)
                        cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), (128, 128, 128), 1)
                
                # Log violations every 100 frames
                if frame_count % 100 == 0 and violation_count > 0:
                    logging.info(f"Kitchen {self.channel_name}: Found {violation_count} human violations - {', '.join(set(violations_found))}")
            
            # Emit SocketIO update every 2 seconds
            if current_time - self.last_socketio_emit >= 2.0:
                try:
                    metrics = {
                        'channel_id': self.channel_id,
                        'channel_name': self.channel_name,
                        'violation_count': violation_count,
                        'violations_detected': list(set(violations_found)) if violations_found else [],
                        'timestamp': datetime.now(IST).isoformat()
                    }
                    self.socketio.emit('kitchen_update', metrics, namespace='/')
                    self.last_socketio_emit = current_time
                except Exception as e:
                    logging.error(f"Kitchen: SocketIO emit failed for {self.channel_name}: {e}")

            
            # Add footer indicator
            # Footer text removed
            
            # Update latest frame AFTER all annotations
            with self.lock:
                self.latest_frame = annotated_frame.copy()
        
        cap.release()
        logging.info(f"✅ Kitchen Compliance thread stopped for {self.channel_name}")

