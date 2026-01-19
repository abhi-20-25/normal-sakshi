"""
Queue Monitor Module
Monitors queue status, detects waiting people, and sends alerts.
Extracted from monolithic edit-004.py for better modularity.
"""

import cv2
import torch
import threading
import time
import json
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict
from shapely.geometry import Point, Polygon
from sqlalchemy import text


# --- Queue Monitor Configuration ---
# Default ROI configuration (fallback if not in database)
DEFAULT_QUEUE_MONITOR_ROI_CONFIG = {
    "Checkout Queue": {
        "roi_points": [[0.5549999952316285, 0.5744444105360244], [0.4456249952316284, 0.5272221883138021], [0.3081249952316284, 0.3105555216471354], [0.08624999523162842, 0.4272221883138021], [0.19249999523162842, 0.7938888549804688]],
        "secondary_roi_points": [[0.5924999952316284, 0.5355555216471354], [0.49874999523162844, 0.502222188313802], [0.3487499952316284, 0.31333329942491317], [0.38156249523162844, 0.3105555216471354], [0.3940624952316284, 0.2883332994249132], [0.5003124952316285, 0.26888885498046877], [0.6721874952316285, 0.4633332994249132]],
    }
}

# Queue detection thresholds
DEFAULT_QUEUE_DWELL_TIME_SEC = 0.05  # How long a person must stay in queue to be counted
DEFAULT_QUEUE_SCREENSHOT_DWELL_TIME_SEC = 5.0  # How long before screenshot triggered
DEFAULT_QUEUE_ALERT_THRESHOLD = 3  # Regular alert: 3+ people with NO cashier
DEFAULT_QUEUE_OVERQUEUE_THRESHOLD = 4  # Overqueue alert: 4+ people WITH cashier
DEFAULT_QUEUE_HIGH_COUNT_THRESHOLD = 3  # Screenshot threshold: queue count > 3
DEFAULT_QUEUE_COUNTER_PERSISTENCE_SEC = 8.0  # Counter occupied persistence time
DEFAULT_QUEUE_ALERT_COOLDOWN_SEC = 6  # Cooldown between alerts


class QueueMonitorProcessor(threading.Thread):
    """
    Queue Monitor Processor - Tracks queue and counter areas, sends alerts.
    
    This processor monitors two ROI areas:
    1. Main ROI (queue area) - where people wait
    2. Secondary ROI (counter area) - where cashier serves
    
    Alerts are triggered when queue builds up without cashier present.
    """
    
    def __init__(self, rtsp_url, channel_id, channel_name, model, restaurant_id=None,
                 db_session_factory=None, socketio=None, detection_handler=None,
                 notification_sender=None, tracking_function=None, timezone=None,
                 config=None):
        """
        Initialize Queue Monitor Processor
        
        Args:
            rtsp_url: Camera RTSP stream URL
            channel_id: Unique channel identifier
            channel_name: Human-readable channel name
            model: YOLO model for person detection
            restaurant_id: Restaurant ID (optional)
            db_session_factory: SQLAlchemy session factory for database access
            socketio: SocketIO instance for real-time updates
            detection_handler: Function to handle detections (screenshots/GIFs)
            notification_sender: Function to send telegram notifications
            tracking_function: Function for safe person tracking
            timezone: Timezone for timestamps
            config: Configuration dictionary with thresholds and settings
        """
        super().__init__(name=channel_name)
        self.rtsp_url = rtsp_url
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.model = model
        self.restaurant_id = restaurant_id
        self.db_session_factory = db_session_factory
        self.socketio = socketio
        self.detection_handler = detection_handler
        self.notification_sender = notification_sender
        self.tracking_function = tracking_function
        self.timezone = timezone
        
        # Apply configuration
        config = config or {}
        self.queue_dwell_time = config.get('queue_dwell_time', DEFAULT_QUEUE_DWELL_TIME_SEC)
        self.screenshot_dwell_time = config.get('screenshot_dwell_time', DEFAULT_QUEUE_SCREENSHOT_DWELL_TIME_SEC)
        self.alert_threshold = config.get('alert_threshold', DEFAULT_QUEUE_ALERT_THRESHOLD)
        self.overqueue_threshold = config.get('overqueue_threshold', DEFAULT_QUEUE_OVERQUEUE_THRESHOLD)
        self.high_count_threshold = config.get('high_count_threshold', DEFAULT_QUEUE_HIGH_COUNT_THRESHOLD)
        self.counter_persistence_sec = config.get('counter_persistence_sec', DEFAULT_QUEUE_COUNTER_PERSISTENCE_SEC)
        self.alert_cooldown_sec = config.get('alert_cooldown_sec', DEFAULT_QUEUE_ALERT_COOLDOWN_SEC)
        self.roi_config = config.get('roi_config', DEFAULT_QUEUE_MONITOR_ROI_CONFIG)
        
        # Thread control
        self.is_running = True
        self.lock = threading.Lock()
        self.latest_frame = None
        
        # Queue tracking state
        self.queue_tracker = defaultdict(lambda: {'entry_time': 0})
        self.current_queue_count = 0
        self.secondary_queue_tracker = defaultdict(lambda: {'entry_time': 0})
        self.current_secondary_count = 0
        self.last_counter_detection_time = 0
        
        # Alert management
        self.last_alert_time = 0
        self.last_overqueue_time = 0
        self.last_screenshot_time = 0
        self.screenshot_cooldown = 10  # 10 seconds cooldown between screenshots
        
        # ROI polygons
        self.roi_poly = Polygon([])
        self.secondary_roi_poly = Polygon([])
        
        # Load ROI from database
        self._load_roi_from_db()
        
        # Cache for queue stats
        self.cached_served_today = 0
        self.cached_peak_count = 0
        self.last_stats_update = 0
        self.stats_update_interval = 30  # Update stats every 30 seconds

    def _load_roi_from_db(self):
        """
        Load ROI from database first, fallback to hardcoded values if not found
        
        Priority: Database ROI > Hardcoded ROI
        This ensures server and local use the same ROI from database.
        """
        if not self.db_session_factory:
            logging.warning(f"No database session factory for {self.channel_name}, using fallback ROI")
            self._use_fallback_roi()
            return
        
        logging.info(f"🏪 Restaurant ID {self.restaurant_id} - attempting to load ROI from database for {self.channel_name}")
        
        try:
            with self.db_session_factory() as db:
                # Query ROI config from database
                query = text("""
                    SELECT roi_points FROM roi_configs 
                    WHERE channel_id = :channel_id AND app_name = 'QueueMonitor'
                """)
                result = db.execute(query, {"channel_id": self.channel_id}).fetchone()
                
                if result and result[0]:
                    points = result[0] if isinstance(result[0], dict) else json.loads(result[0])
                    self.normalized_main_roi = points.get("main", [])
                    self.normalized_secondary_roi = points.get("secondary", [])
                    
                    # Initialize with empty polygons - will be converted to pixels in run() method
                    self.roi_poly = Polygon([])
                    self.secondary_roi_poly = Polygon([])
                    
                    logging.info(f"✅ Loaded custom ROI for QueueMonitor {self.channel_name} from database.")
                    logging.info(f"   Main ROI: {len(self.normalized_main_roi)} points")
                    logging.info(f"   Secondary ROI: {len(self.normalized_secondary_roi)} points")
                    return
                else:
                    logging.warning(f"No custom ROI in DB for QueueMonitor {self.channel_name}. Using hardcoded fallback.")
                    self._use_fallback_roi()
        except Exception as e:
            logging.error(f"Failed to load ROI from DB: {e}. Using fallback.")
            self._use_fallback_roi()

    def _use_fallback_roi(self):
        """Use hardcoded fallback ROI configuration"""
        fallback_config = self.roi_config.get(self.channel_name, {})
        
        # If channel name doesn't match, try to use the first available config
        if not fallback_config and self.roi_config:
            first_key = list(self.roi_config.keys())[0]
            fallback_config = self.roi_config[first_key]
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
        """Update ROI configuration dynamically"""
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
        """Gracefully shutdown the processor"""
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
        
        if not self.db_session_factory:
            return served_today, peak_count
        
        try:
            with self.db_session_factory() as db:
                if self.timezone:
                    today = datetime.now(self.timezone).date()
                    start_of_day = datetime.combine(today, datetime.min.time()).replace(tzinfo=self.timezone)
                    end_of_day = datetime.combine(today, datetime.max.time()).replace(tzinfo=self.timezone)
                else:
                    today = datetime.now().date()
                    start_of_day = datetime.combine(today, datetime.min.time())
                    end_of_day = datetime.combine(today, datetime.max.time())
                
                # Get all queue logs for today, ordered by time
                query = text("""
                    SELECT queue_count, timestamp FROM queue_logs
                    WHERE channel_id = :channel_id 
                    AND timestamp >= :start_time 
                    AND timestamp <= :end_time
                    ORDER BY timestamp
                """)
                records = db.execute(query, {
                    "channel_id": self.channel_id,
                    "start_time": start_of_day,
                    "end_time": end_of_day
                }).fetchall()
                
                if records:
                    # Calculate peak count
                    peak_count = max(record[0] for record in records)
                    
                    # Calculate served count (count people who entered counter area)
                    # Count transitions where queue decreases
                    prev_count = 0
                    for record in records:
                        if prev_count > 0 and record[0] < prev_count:
                            # People moved from queue (likely to counter)
                            served_today += (prev_count - record[0])
                        prev_count = record[0]
                    
                    logging.debug(f"Queue stats for {self.channel_name}: Served={served_today}, Peak={peak_count}, Records={len(records)}")
                
                # Update cache
                self.cached_served_today = served_today
                self.cached_peak_count = peak_count
                self.last_stats_update = current_time
                
        except Exception as e:
            logging.error(f"Failed to get queue stats for {self.channel_name}: {e}")
        
        return served_today, peak_count

    def get_frame(self):
        """Get the latest annotated frame for streaming"""
        with self.lock:
            if self.latest_frame is None:
                placeholder = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(placeholder, 'Connecting...', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (201, 209, 217), 2)
                _, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()
            _, jpeg = cv2.imencode('.jpg', self.latest_frame)
            return jpeg.tobytes()

    def _persist_queue_count(self, count: int) -> None:
        """Non-blocking DB persistence to avoid adding latency in the frame loop"""
        if not self.db_session_factory:
            return
        try:
            with self.db_session_factory() as db:
                query = text("""
                    INSERT INTO queue_logs (channel_id, queue_count, timestamp)
                    VALUES (:channel_id, :queue_count, :timestamp)
                """)
                timestamp = datetime.now(self.timezone) if self.timezone else datetime.now()
                db.execute(query, {
                    "channel_id": self.channel_id,
                    "queue_count": count,
                    "timestamp": timestamp
                })
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
        """Main thread loop - process frames continuously"""
        first_frame = True
        consecutive_errors = 0
        max_consecutive_errors = 10
        
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

    def process_frame(self, frame):
        """Process a single frame - detect persons, update counts, send alerts"""
        current_time = time.time()
        
        # Use tracking function to detect persons
        if self.tracking_function:
            results = self.tracking_function(self.model, frame, conf=0.20, iou=0.5, processor_name=f"{self.channel_name}-QueueMonitor")
        else:
            # Fallback to direct model inference
            with torch.inference_mode():
                results = self.model.track(frame, persist=True, classes=[0], conf=0.20, iou=0.5, verbose=False)
        
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
        
        # Process detected persons
        if r0 is not None and getattr(r0, 'boxes', None) is not None:
            boxes = r0.boxes.xyxy.cpu()
            
            # Try to get tracking IDs, fallback to using detection indices
            if getattr(r0.boxes, 'id', None) is not None:
                track_ids = r0.boxes.id.int().cpu().tolist()
            else:
                # Tracking failed - use hash of box coordinates as pseudo-ID for this frame
                track_ids = []
                for i, box in enumerate(boxes):
                    # Create a stable ID based on box position
                    pseudo_id = hash((int(box[0]/10)*10, int(box[1]/10)*10, int(box[2]/10)*10, int(box[3]/10)*10)) % 100000
                    track_ids.append(pseudo_id)
                if len(boxes) > 0 and not hasattr(self, '_tracking_fallback_logged'):
                    logging.warning(f"{self.channel_name}: Tracking IDs not available, using position-based pseudo-IDs")
                    self._tracking_fallback_logged = True
            
            # Debug: Log number of detections
            if len(boxes) > 0:
                logging.debug(f"Detected {len(boxes)} persons in frame for {self.channel_name}")
            
            for box, track_id in zip(boxes, track_ids):
                # Calculate the true center point of the bounding box
                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)
                person_point = Point(center_x, center_y)
                
                # Check main ROI (queue area)
                if self.roi_poly.is_valid and not self.roi_poly.is_empty:
                    contains_main = self.roi_poly.contains(person_point)
                    if contains_main:
                        current_tracks_in_main_roi.add(track_id)
                        tracker = self.queue_tracker[track_id]
                        if tracker['entry_time'] == 0: 
                            tracker['entry_time'] = current_time
                            logging.info(f"Person {track_id} entered queue ROI at {current_time} - Center Point: ({center_x}, {center_y})")
                    else:
                        if track_id not in current_tracks_in_main_roi and len(current_tracks_in_main_roi) == 0:
                            logging.debug(f"Person {track_id} center point ({center_x}, {center_y}) NOT in queue ROI")
                else:
                    if not hasattr(self, '_roi_warning_logged'):
                        logging.warning(f"Main ROI is invalid or empty for {self.channel_name}")
                        self._roi_warning_logged = True
                
                # Check secondary ROI (counter area)
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
                        if track_id not in current_tracks_in_secondary_roi and len(current_tracks_in_secondary_roi) == 0:
                            logging.debug(f"Person {track_id} center point ({center_x}, {center_y}) NOT in counter ROI")
                else:
                    if not hasattr(self, '_secondary_roi_warning_logged'):
                        logging.warning(f"Secondary ROI is invalid or empty for {self.channel_name}")
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
                if entry_time > 0 and (current_time - entry_time) >= self.queue_dwell_time:
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
                if entry_time > 0 and (current_time - entry_time) >= self.queue_dwell_time:
                    valid_secondary_count += 1
        
        # Persistence mechanism: If no one is currently detected in counter but someone was detected
        # within the persistence window, still consider counter as occupied to prevent false alerts
        if valid_secondary_count == 0 and self.last_counter_detection_time > 0:
            time_since_last_detection = current_time - self.last_counter_detection_time
            if time_since_last_detection <= self.counter_persistence_sec:
                valid_secondary_count = 1  # Assume counter is still occupied
                logging.debug(f"Counter area persistence active: last detection was {time_since_last_detection:.1f}s ago (within {self.counter_persistence_sec}s window)")
            else:
                # Reset if persistence window expired
                self.last_counter_detection_time = 0

        if self.current_secondary_count != valid_secondary_count:
            self.current_secondary_count = valid_secondary_count
            updated = True
            # Persist queue count whenever it changes
            self._persist_queue_count(self.current_queue_count)

        # Emit live counts (no DB persistence) if either changed
        if updated and self.socketio:
            served_today, peak_count = self._get_queue_stats()
            self.socketio.emit('queue_update', {
                'channel_id': self.channel_id,
                'queue': self.current_queue_count,
                'counter': self.current_secondary_count,
                'count': self.current_queue_count,  # backward compat
                'served_today': served_today,
                'peak_count': peak_count
            })

        # Check for persons who have been in queue for more than screenshot dwell time
        persons_in_queue_5sec = []
        for track_id in current_tracks_in_main_roi:
            if track_id in self.queue_tracker:
                dwell_time = current_time - self.queue_tracker[track_id]['entry_time']
                if dwell_time >= self.screenshot_dwell_time:
                    persons_in_queue_5sec.append(track_id)
        
        # Screenshot trigger 1: person waiting > 5 seconds AND counter is empty (HIGHEST PRIORITY)
        should_screenshot_5sec = (
            len(persons_in_queue_5sec) > 0 and
            valid_secondary_count == 0 and
            (current_time - self.last_screenshot_time) > self.screenshot_cooldown
        )
        
        # Screenshot trigger 2: queue count > threshold (MEDIUM PRIORITY)
        should_screenshot_high_count = (
            valid_queue_count > self.high_count_threshold and
            (current_time - self.last_screenshot_time) > self.screenshot_cooldown
        )
        
        # Screenshot trigger 3: counter is empty AND queue has people (FALLBACK)
        should_screenshot_counter_empty = (
            valid_queue_count > 0 and
            valid_secondary_count == 0 and
            (current_time - self.last_screenshot_time) > 10.0  # 10-second cooldown
        )

        # Alert when cashier area is empty and queue has threshold+ people (with cooldown)
        should_alert = (
            valid_queue_count >= self.alert_threshold and
            self.current_secondary_count == 0 and
            (current_time - self.last_alert_time) > self.alert_cooldown_sec
        )
        
        # Overqueue detection: when cashier is present and queue has overqueue threshold+ people
        should_overqueue_alert = (
            valid_queue_count >= self.overqueue_threshold and
            self.current_secondary_count > 0 and
            (current_time - self.last_overqueue_time) > self.alert_cooldown_sec
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
        
        # Handle screenshots and alerts
        if should_screenshot_5sec and self.detection_handler:
            self.last_screenshot_time = current_time
            screenshot_message = f"Person waiting in queue for more than {self.screenshot_dwell_time} seconds. Queue count: {valid_queue_count}, Counter: {valid_secondary_count}"
            logging.warning(f"5-SEC WAIT SCREENSHOT on {self.channel_name}: {screenshot_message}")
            try:
                media_path = self.detection_handler('QueueMonitor', self.channel_id, [annotated_frame], screenshot_message, is_gif=False)
                if media_path:
                    logging.info(f"Screenshot saved successfully: {media_path}")
                else:
                    logging.error(f"Failed to save screenshot for {self.channel_name}")
            except Exception as e:
                logging.error(f"Error saving screenshot for {self.channel_name}: {e}")
        
        elif should_screenshot_high_count and self.detection_handler:
            self.last_screenshot_time = current_time
            high_count_message = f"High queue count: {valid_queue_count} people in queue. Counter: {valid_secondary_count}"
            logging.warning(f"HIGH QUEUE COUNT SCREENSHOT on {self.channel_name}: {high_count_message}")
            try:
                media_path = self.detection_handler('QueueMonitor', self.channel_id, [annotated_frame], high_count_message, is_gif=False)
                if media_path:
                    logging.info(f"Screenshot saved successfully: {media_path}")
            except Exception as e:
                logging.error(f"Error saving screenshot for {self.channel_name}: {e}")
        
        elif should_screenshot_counter_empty and self.detection_handler:
            self.last_screenshot_time = current_time
            counter_empty_message = f"Counter is empty but queue has {valid_queue_count} people waiting"
            logging.info(f"COUNTER EMPTY SCREENSHOT on {self.channel_name}: {counter_empty_message}")
            try:
                media_path = self.detection_handler('QueueMonitor', self.channel_id, [annotated_frame], counter_empty_message, is_gif=False)
                if media_path:
                    logging.info(f"Screenshot saved successfully: {media_path}")
            except Exception as e:
                logging.error(f"Error saving screenshot for {self.channel_name}: {e}")
        
        if should_alert:
            self.last_alert_time = current_time
            alert_message = f"Queue is full ({valid_queue_count} people), but the counter is free."
            logging.warning(f"QUEUE ALERT on {self.channel_name}: {alert_message}")
            if self.notification_sender:
                self.notification_sender(f"🚨 **Queue Alert: {self.channel_name}** 🚨\n{alert_message}")
            if self.detection_handler:
                self.detection_handler('QueueMonitor', self.channel_id, [annotated_frame], alert_message, is_gif=False)
        
        if should_overqueue_alert:
            self.last_overqueue_time = current_time
            overqueue_message = f"OVERQUEUE: {valid_queue_count} people in queue with cashier present!"
            logging.warning(f"OVERQUEUE ALERT on {self.channel_name}: {overqueue_message}")
            if self.notification_sender:
                self.notification_sender(f"⚠️ **Overqueue Alert: {self.channel_name}** ⚠️\n{overqueue_message}")
            if self.detection_handler:
                self.detection_handler('QueueMonitor', self.channel_id, [annotated_frame], overqueue_message, is_gif=False)

        # Log count changes for debugging
        queue_display_count = valid_queue_count
        counter_display_count = valid_secondary_count
        if queue_display_count > 0 or counter_display_count > 0:
            logging.info(f"QueueMonitor {self.channel_name}: Queue={queue_display_count}, Counter={counter_display_count}, "
                       f"Tracks in main ROI: {len(current_tracks_in_main_roi)}, "
                       f"Tracks in secondary ROI: {len(current_tracks_in_secondary_roi)}")
        
        with self.lock:
            self.latest_frame = annotated_frame.copy()


def run_queue_monitor(config, camera_stream, db_client):
    """
    Public function to run queue monitor.
    
    This is the main entry point for running the queue monitor processor.
    It creates and starts a QueueMonitorProcessor instance.
    
    Args:
        config: Configuration dictionary containing:
            - rtsp_url: Camera RTSP stream URL
            - channel_id: Unique channel identifier
            - channel_name: Human-readable channel name
            - model: YOLO model for person detection
            - restaurant_id: Restaurant ID (optional)
            - db_session_factory: SQLAlchemy session factory
            - socketio: SocketIO instance for real-time updates
            - detection_handler: Function to handle detections
            - notification_sender: Function to send notifications
            - tracking_function: Function for safe person tracking
            - timezone: Timezone for timestamps
            - queue_config: Queue-specific configuration (thresholds, etc.)
        camera_stream: FrameHub instance providing camera frames
        db_client: Database client (unused, kept for compatibility)
    
    Returns:
        QueueMonitorProcessor: The running queue monitor processor instance
    """
    processor = QueueMonitorProcessor(
        rtsp_url=config.get('rtsp_url'),
        channel_id=config.get('channel_id'),
        channel_name=config.get('channel_name'),
        model=config.get('model'),
        restaurant_id=config.get('restaurant_id'),
        db_session_factory=config.get('db_session_factory'),
        socketio=config.get('socketio'),
        detection_handler=config.get('detection_handler'),
        notification_sender=config.get('notification_sender'),
        tracking_function=config.get('tracking_function'),
        timezone=config.get('timezone'),
        config=config.get('queue_config', {})
    )
    
    # Attach frame hub for frame access
    processor.frame_hub = camera_stream
    
    # Start the processor thread
    processor.start()
    
    return processor
