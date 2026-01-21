"""
People Counter Module
Handles person counting, entry/exit detection, and footfall aggregation
"""

import threading
import time
import logging
import json
import cv2
import numpy as np
from datetime import datetime
from sqlalchemy import text


class PeopleCounterProcessor(threading.Thread):
    """
    People counter processor using line-crossing detection
    Tracks people entering/exiting based on centroid movement across a counting line
    """
    
    def __init__(self, rtsp_url, channel_id, channel_name, model, detection_callback, socketio,
                 db_session_factory, db_connected, timezone, safe_track_persons_func):
        """
        Initialize People Counter Processor
        
        Args:
            rtsp_url: RTSP stream URL
            channel_id: Unique channel identifier
            channel_name: Human-readable channel name
            model: YOLO model for person detection
            detection_callback: Callback for handling detections
            socketio: SocketIO instance for real-time updates
            db_session_factory: SQLAlchemy session factory
            db_connected: Database connection status flag
            timezone: Timezone for date/time handling (IST)
            safe_track_persons_func: Function for safe person tracking
        """
        super().__init__()
        self.rtsp_url = rtsp_url
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.model = model
        self.detection_callback = detection_callback
        self.socketio = socketio
        self.SessionLocal = db_session_factory
        self.db_connected = db_connected
        self.IST = timezone
        self.safe_track_persons = safe_track_persons_func
        
        self.app_name = "PeopleCounter"
        self.is_running = True
        self.lock = threading.Lock()
        
        # LINE CROSSING APPROACH - Simple & Reliable!
        self.previous_centroids = []  # List of (x, y) from previous frame
        self.counting_line_position = 0.38  # Default: Line at 38% (LEFT=0-38%, RIGHT=38-100%)
        self.cooldown_zones = {}  # {(approx_x, approx_y): timestamp} to prevent double counting
        self.cooldown_duration = 0.8  # 800ms cooldown per zone
        
        # Load counting line position from database
        self._load_line_position_from_db()
        
        self.counts = {'in': 0, 'out': 0}
        self.current_hour = datetime.now(self.IST).hour
        self.tracking_date = datetime.now(self.IST).date()
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
    
    def stop(self):
        self.is_running = False
        
    def shutdown(self):
        logging.info(f"Shutting down PeopleCounter for {self.channel_name}. Saving final counts...")
        self._update_and_log_counts()
        self.is_running = False

    def get_frame(self):
        """Get current frame for video streaming"""
        with self.lock:
            if self.latest_frame is None:
                placeholder = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(placeholder, 'Connecting...', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (201, 209, 217), 2)
                _, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()
            _, jpeg = cv2.imencode('.jpg', self.latest_frame)
            return jpeg.tobytes()

    def _get_hourly_data(self):
        """Get today's hourly IN counts for the bar chart"""
        hourly_data = [0] * 24  # Initialize 24 hours with 0
        if not self.db_connected:
            return hourly_data
        
        with self.SessionLocal() as db:
            try:
                today_ist = datetime.now(self.IST).date()
                query = text("""
                    SELECT hour, in_count 
                    FROM hourly_footfall 
                    WHERE channel_id = :channel_id 
                    AND report_date = :report_date
                """)
                records = db.execute(query, {
                    'channel_id': self.channel_id,
                    'report_date': today_ist
                }).fetchall()
                
                for record in records:
                    hour, in_count = record
                    if 0 <= hour < 24:
                        hourly_data[hour] = in_count
            except Exception as e:
                logging.error(f"Failed to fetch hourly data: {e}")
        
        return hourly_data

    def _load_line_position_from_db(self):
        """Load counting line position from database"""
        if not self.db_connected:
            return
            
        try:
            with self.SessionLocal() as db:
                query = text("""
                    SELECT roi_points 
                    FROM roi_configs 
                    WHERE channel_id = :channel_id 
                    AND app_name = :app_name
                """)
                result = db.execute(query, {
                    'channel_id': self.channel_id,
                    'app_name': 'PeopleCounter'
                }).fetchone()
                
                if result and result[0]:
                    points = json.loads(result[0])
                    if 'line_position' in points:
                        self.counting_line_position = points['line_position']
                        logging.info(f"✅ Loaded counting line position: {self.counting_line_position*100:.0f}% for {self.channel_name}")
                    else:
                        logging.info(f"Using default counting line position: 38% for {self.channel_name}")
                else:
                    logging.info(f"No saved line position found, using default: 38% for {self.channel_name}")
        except Exception as e:
            logging.error(f"Error loading line position: {e}. Using default 38%")

    def _load_initial_counts(self):
        """Load today's counts from database on startup"""
        if not self.db_connected:
            return
            
        with self.SessionLocal() as db:
            try:
                today_ist = datetime.now(self.IST).date()
                self.tracking_date = today_ist
                
                query = text("""
                    SELECT in_count, out_count 
                    FROM daily_footfall 
                    WHERE channel_id = :channel_id 
                    AND report_date = :report_date
                """)
                result = db.execute(query, {
                    'channel_id': self.channel_id,
                    'report_date': today_ist
                }).fetchone()
                
                if result:
                    self.counts = {'in': result[0], 'out': result[1]}
                else:
                    self._reset_counts_for_new_day(db, today_ist)
            except Exception as e:
                logging.error(f"Failed to load initial counts: {e}")

    def _reset_counts_for_new_day(self, db, new_date):
        """Reset counts for a new day"""
        self.counts = {'in': 0, 'out': 0}
        self.tracking_date = new_date
        
        query = text("""
            INSERT INTO daily_footfall (channel_id, report_date, in_count, out_count)
            VALUES (:channel_id, :report_date, 0, 0)
        """)
        db.execute(query, {
            'channel_id': self.channel_id,
            'report_date': new_date
        })
        db.commit()

    def _update_and_log_counts(self):
        """Update daily counts in database"""
        if not self.db_connected:
            return
            
        with self.SessionLocal() as db, self.lock:
            try:
                query = text("""
                    UPDATE daily_footfall 
                    SET in_count = :in_count, out_count = :out_count
                    WHERE channel_id = :channel_id 
                    AND report_date = :report_date
                """)
                db.execute(query, {
                    'in_count': self.counts['in'],
                    'out_count': self.counts['out'],
                    'channel_id': self.channel_id,
                    'report_date': self.tracking_date
                })
                db.commit()
            except Exception as e:
                logging.error(f"Error updating daily counts in DB: {e}")
                db.rollback()
    
    def _update_hourly_count_realtime(self, count_type):
        """Update hourly count in database in real-time when in/out is detected"""
        if not self.db_connected:
            return
            
        current_time = datetime.now(self.IST)
        current_hour_ist = current_time.hour
        current_date_ist = current_time.date()
        
        # Check if hour changed - if so, update current_hour
        if current_hour_ist != self.current_hour:
            self.current_hour = current_hour_ist
        
        # Check if day changed - if so, update tracking_date
        if current_date_ist != self.tracking_date:
            self.tracking_date = current_date_ist
        
        with self.SessionLocal() as db:
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
                logging.error(f"Error updating hourly count in DB: {e}")
                db.rollback()

    def _check_for_new_day(self):
        """Check if a new day has started and reset counts if needed"""
        current_date_ist = datetime.now(self.IST).date()
        current_hour_ist = datetime.now(self.IST).hour
        
        if current_date_ist > self.tracking_date:
            logging.info("New day detected. Resetting people counter.")
            self._update_and_log_counts()
            with self.SessionLocal() as db:
                self._reset_counts_for_new_day(db, current_date_ist)
                self.current_hour = current_hour_ist

    def run(self):
        """Main processing loop for people counting"""
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
                results = self.safe_track_persons(
                    self.model, enhanced_frame, conf=0.20, iou=0.5, 
                    processor_name=f"{self.channel_name}-PeopleCounter"
                )
                consecutive_errors = 0  # Reset on successful frame
                r0 = results[0] if (results and len(results) > 0) else None
                
                # LINE CROSSING DETECTION
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]
                counting_line_x = int(frame_width * self.counting_line_position)
                
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
                    expired_zones = [
                        zone for zone, timestamp in self.cooldown_zones.items() 
                        if current_time - timestamp > self.cooldown_duration
                    ]
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
                
                # Create annotated frame with bounding boxes
                annotated_frame = frame.copy()
                
                # Draw bounding boxes around detected persons
                if r0 is not None and getattr(r0, 'boxes', None) is not None:
                    boxes_xyxy = r0.boxes.xyxy.cpu()
                    boxes_conf = r0.boxes.conf.cpu()
                    
                    for i, box in enumerate(boxes_xyxy):
                        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                        box_width = float(x2 - x1)
                        box_height = float(y2 - y1)
                        box_area = box_width * box_height
                        confidence = float(boxes_conf[i])
                        
                        # Apply same filters as detection
                        aspect_ratio = box_height / box_width if box_width > 0 else 0
                        is_person_shaped = 1.2 <= aspect_ratio <= 4.0
                        is_valid_size = min_box_area <= box_area <= max_box_area
                        is_confident = confidence >= min_confidence
                        
                        if is_person_shaped and is_valid_size and is_confident:
                            # Draw bounding box
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Draw confidence label
                            label = f'Person {confidence:.2f}'
                            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                        (x1 + label_size[0], y1), (0, 255, 0), -1)
                            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Draw counting line
                cv2.line(annotated_frame, (counting_line_x, 0), 
                        (counting_line_x, frame_height), (255, 0, 0), 2)
                
                with self.lock:
                    self.latest_frame = annotated_frame.copy()
                    
                hourly_data = self._get_hourly_data()
                self.socketio.emit('count_update', {
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
