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

# --- Basic Configuration ---
IST = pytz.timezone('Asia/Kolkata')
Base = declarative_base()

# --- Model Paths (Standardized) ---
APRON_CAP_MODEL_PATH = 'apron-cap.pt'
GLOVES_MODEL_PATH = 'gloves.pt'
GENERAL_MODEL_PATH = 'yolo11n.pt'  # Updated to YOLO 11

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
            # FORCE CPU MODE - Disable all CUDA usage for stability
            self.device = 'cpu'
            logging.info(f"🚫 CUDA DISABLED - Using CPU-only for Kitchen channel {self.channel_name}")
            
            for model_path in [APRON_CAP_MODEL_PATH, GLOVES_MODEL_PATH, GENERAL_MODEL_PATH]:
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Missing model file: {model_path}")
            
            self.apron_cap_model = YOLO(APRON_CAP_MODEL_PATH)
            self.gloves_model = YOLO(GLOVES_MODEL_PATH)
            self.general_model = YOLO(GENERAL_MODEL_PATH)
            self.apron_cap_model.to(self.device)
            self.gloves_model.to(self.device)
            self.general_model.to(self.device)
            logging.info(f"Successfully loaded Kitchen Compliance models for {self.channel_name} (CPU mode)")
        except Exception as e:
            self.error_message = f"Model Error: {e}"
            logging.error(f"FATAL: Failed to initialize Kitchen models for {self.channel_name}. Error: {e}")

        self.person_violation_tracker = defaultdict(lambda: defaultdict(float))
        self.phone_tracker = {}
        self.last_apron_cap_results = []
        self.last_gloves_results = []
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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

    def _trigger_alert(self, frame, violation_type, details):
        logging.warning(f"ALERT on {self.channel_name}: {details}")
        telegram_message = f"🚨 Kitchen Alert: {self.channel_name}\nViolation: {violation_type}\nDetails: {details}"
        self.send_telegram_notification(telegram_message)
        media_path = self.handle_main_detection(
            'KitchenCompliance', self.channel_id, [frame], details, is_gif=False
        )
        if media_path:
            self._save_violation_to_db(violation_type, details, media_path)

    def run(self):
        if self.error_message: return
        
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

            # --- Run Inferences ---
            try:
                person_results = self.general_model.track(frame, persist=True, classes=[0], conf=CONFIDENCE_THRESHOLD, verbose=False)
                phone_results = self.general_model(frame, classes=[67], conf=CONFIDENCE_THRESHOLD, verbose=False)

                if frame_count % FRAME_SKIP_RATE == 0:
                    self.last_apron_cap_results = self.apron_cap_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
                    self.last_gloves_results = self.gloves_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            except Exception as e:
                logging.error(f"Model inference error: {e}")
                person_results = None
                phone_results = None

            # Draw header info
            h, w = annotated_frame.shape[:2]
            cv2.putText(annotated_frame, "Kitchen Compliance Monitor", (15, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(annotated_frame, "Checking: Gloves, Apron, Cap, Uniform, Phone", (15, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(annotated_frame, f"Device: {self.device.upper()} | Conf: {CONFIDENCE_THRESHOLD}", (15, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # --- Process Each Person ---
            if person_results and person_results[0].boxes.id is not None:
                track_ids = person_results[0].boxes.id.int().cpu().tolist()
                person_boxes = person_results[0].boxes.xyxy.cpu()
                confidences = person_results[0].boxes.conf.cpu().numpy()
                logging.debug(f"Detected {len(track_ids)} people in frame {frame_count}")

                # Get detected gloves boxes (FIXED LOGIC)
                detected_gloves_boxes = []
                for r in self.last_gloves_results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        for box in r.boxes:
                            if int(box.cls[0]) < len(self.gloves_model.names):
                                class_name = self.gloves_model.names[int(box.cls[0])]
                                if 'glove' in class_name.lower() or 'surgical' in class_name.lower():
                                    detected_gloves_boxes.append(box.xyxy[0].cpu().numpy())
            else:
                # No people detected - show status
                cv2.putText(annotated_frame, "No people detected", (50, h-50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)

            if person_results and person_results[0].boxes.id is not None:
                for person_box, track_id, conf in zip(person_boxes, track_ids, confidences):
                    px1, py1, px2, py2 = map(int, person_box)
                    
                    # Default: Green box = compliant
                    box_color = (0, 255, 0)  # Green
                    status_text = "OK"
                    violations = []
                    
                    # 1. Check for Apron/Cap Violations
                    for r in self.last_apron_cap_results:
                        if r.boxes is not None and len(r.boxes) > 0:
                            for box in r.boxes:
                                if int(box.cls[0]) < len(self.apron_cap_model.names):
                                    violation_class = self.apron_cap_model.names[int(box.cls[0])]
                                    if 'without' in violation_class.lower() or 'no' in violation_class.lower():
                                        violations.append(violation_class)
                                        box_color = (0, 0, 255)  # Red
                                        if current_time - self.person_violation_tracker[track_id][violation_class] > ALERT_COOLDOWN_SECONDS:
                                            self.person_violation_tracker[track_id][violation_class] = current_time
                                            details = f"Person ID {track_id} detected with '{violation_class}'."
                                            self._trigger_alert(frame.copy(), violation_class, details)

                    # 2. Check for Gloves Violation (FIXED LOGIC)
                    has_gloves = False
                    for g_box in detected_gloves_boxes:
                        gx1, gy1, gx2, gy2 = map(int, g_box)
                        # Check if glove box overlaps with person box
                        if (gx1 < px2 and gx2 > px1 and gy1 < py2 and gy2 > py1):
                            has_gloves = True
                            # Draw glove indicator
                            cv2.rectangle(annotated_frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, "GLOVES", (gx1, gy1-5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                            break
                    
                    if not has_gloves:
                        violations.append("No-Gloves")
                        box_color = (0, 0, 255)  # Red
                        if current_time - self.person_violation_tracker[track_id]['No-Gloves'] > ALERT_COOLDOWN_SECONDS:
                            self.person_violation_tracker[track_id]['No-Gloves'] = current_time
                            details = f"Person ID {track_id} has no gloves."
                            self._trigger_alert(frame.copy(), "No-Gloves", details)
                    
                    # 3. Check for Uniform Color Violation
                    torso_crop = frame[py1 + int((py2-py1)*0.1):py1 + int((py2-py1)*0.7), px1:px2]
                    if torso_crop.size > 0:
                        try:
                            lab_torso = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2LAB)
                            l, a, b = cv2.split(lab_torso)
                            equalized_l = self.clahe.apply(l)
                            merged_lab = cv2.merge((equalized_l, a, b))
                            equalized_torso = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
                            hsv_torso = cv2.cvtColor(equalized_torso, cv2.COLOR_BGR2HSV)
                            
                            mask_yellow = cv2.inRange(hsv_torso, YELLOW_LOWER, YELLOW_UPPER)
                            mask_black = cv2.inRange(hsv_torso, BLACK_LOWER, BLACK_UPPER)
                            compliant_mask = cv2.bitwise_or(mask_yellow, mask_black)
                            
                            total_pixels = torso_crop.shape[0] * torso_crop.shape[1]
                            compliant_ratio = np.count_nonzero(compliant_mask) / total_pixels if total_pixels > 0 else 0

                            if compliant_ratio < 0.30: # If less than 30% of torso is compliant color
                                violations.append("Uniform-Violation")
                                box_color = (0, 0, 255)  # Red
                                if current_time - self.person_violation_tracker[track_id]['Uniform-Violation'] > ALERT_COOLDOWN_SECONDS:
                                    self.person_violation_tracker[track_id]['Uniform-Violation'] = current_time
                                    details = f"Person ID {track_id} has a uniform color violation."
                                    self._trigger_alert(frame.copy(), "Uniform-Violation", details)
                        except Exception as e:
                            logging.error(f"Uniform detection error: {e}")
                    
                    # Draw person bounding box
                    cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), box_color, 3)
                    
                    # Prepare label
                    if violations:
                        status_text = f"ID:{track_id} VIOLATION"
                        label_color = (0, 0, 255)
                    else:
                        status_text = f"ID:{track_id} COMPLIANT"
                        label_color = (0, 255, 0)
                    
                    # Draw label background
                    label_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)[0]
                    cv2.rectangle(annotated_frame, (px1, py1-label_size[1]-10), 
                                 (px1+label_size[0]+5, py1), box_color, -1)
                    cv2.putText(annotated_frame, status_text, (px1, py1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                    
                    # Show violations below box
                    if violations:
                        for idx, violation in enumerate(violations[:3]):  # Show max 3
                            cv2.putText(annotated_frame, f"- {violation}", (px1, py2+20+idx*20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                    
                    # Show confidence
                    cv2.putText(annotated_frame, f"{conf:.2f}", (px2-50, py1+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # --- 4. Detect and Track Mobile Phones ---
            if phone_results and phone_results[0].boxes is not None:
                current_phones = [box.xyxy[0].cpu().numpy() for r in phone_results for box in r.boxes]
                new_phone_tracker = {}

                for phone_box in current_phones:
                    cx, cy = int((phone_box[0] + phone_box[2]) / 2), int((phone_box[1] + phone_box[3]) / 2)
                    found_match = False
                    for phone_id, data in self.phone_tracker.items():
                        dist = np.sqrt((cx - data['center'][0])**2 + (cy - data['center'][1])**2)
                        if dist < 50:
                            new_phone_tracker[phone_id] = {'box': phone_box, 'frames': data['frames'] + 1, 'center': (cx, cy), 'alerted': data.get('alerted', False)}
                            found_match = True
                            break
                    if not found_match:
                        new_id = max(self.phone_tracker.keys(), default=0) + 1
                        new_phone_tracker[new_id] = {'box': phone_box, 'frames': 1, 'center': (cx, cy), 'alerted': False}

                self.phone_tracker = new_phone_tracker

                for phone_id, data in self.phone_tracker.items():
                    phone_box = data['box']
                    p_x1, p_y1, p_x2, p_y2 = map(int, phone_box)
                    
                    # Draw phone detection
                    cv2.rectangle(annotated_frame, (p_x1, p_y1), (p_x2, p_y2), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, f"PHONE {data['frames']}f", (p_x1, p_y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                    
                    if data['frames'] > phone_persistence_frames and not data['alerted']:
                        data['alerted'] = True # Mark as alerted to prevent spamming
                        details = f"Mobile phone detected in restricted area for {PHONE_PERSISTENCE_SECONDS} seconds."
                        self._trigger_alert(frame.copy(), "Mobile-Phone", details)
            
            # Add footer indicator
            cv2.putText(annotated_frame, "YOLO Kitchen Compliance Active", (w-350, h-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            
            # Update latest frame AFTER all annotations
            with self.lock:
                self.latest_frame = annotated_frame.copy()
        
        cap.release()

