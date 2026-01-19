#!/usr/bin/env python3
"""
Live Occupancy Monitor Processor - Enhanced Version
Features:
- CUDA auto-detection
- Improved person detection (detects all people accurately)
- Scheduled operation (only runs during configured times)
- Auto-pause when requirement met
"""

import cv2
import threading
import time
import numpy as np
import logging
import pytz
import torch
import json
import os
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, UniqueConstraint, text, create_engine
from sqlalchemy.orm import sessionmaker
from shapely.geometry import Point, Polygon


# Database tables (Base will be provided by caller)
def get_occupancy_tables(Base):
    """Define occupancy database tables using provided Base"""
    
    class OccupancyLog(Base):
        __tablename__ = "occupancy_logs"
        __table_args__ = {'extend_existing': True}
        id = Column(Integer, primary_key=True, index=True)
        channel_id = Column(String, index=True)
        timestamp = Column(DateTime)
        time_slot = Column(String)
        day_of_week = Column(String)
        live_count = Column(Integer)
        required_count = Column(Integer)
        status = Column(String)  # 'OK', 'BELOW_REQUIREMENT', 'NO_SCHEDULE', 'PAUSED'
    
    class OccupancySchedule(Base):
        __tablename__ = "occupancy_schedules"
        __table_args__ = (
            UniqueConstraint('channel_id', 'time_slot', 'day_of_week', name='_occupancy_schedule_uc'),
            {'extend_existing': True}
        )
        id = Column(Integer, primary_key=True, index=True)
        channel_id = Column(String, index=True)
        time_slot = Column(String)  # e.g., "9:00"
        day_of_week = Column(String)  # e.g., "Monday"
        required_count = Column(Integer)
    
    return OccupancyLog, OccupancySchedule


class OccupancyMonitorProcessor(threading.Thread):
    """
    Enhanced Occupancy Monitor - CUDA enabled, accurate detection, scheduled operation
    """
    
    def __init__(self, rtsp_url, channel_id, channel_name, model, socketio, SessionLocal, send_notification, 
                 OccupancyLog, OccupancySchedule, timezone=None, database_url=None, device='cuda'):
        super().__init__(name=f"OccupancyMonitor-{channel_name}")
        self.rtsp_url = rtsp_url
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.model = model
        self.socketio = socketio
        self.SessionLocal = SessionLocal
        self.send_notification = send_notification
        self.OccupancyLog = OccupancyLog
        self.OccupancySchedule = OccupancySchedule
        self.timezone = timezone or pytz.timezone('Asia/Kolkata')
        self.database_url = database_url
        
        # Use provided device setting
        self.device = device
        self.model.to(self.device)
        logging.info(f"🎯 Using device: {self.device.upper()}")
        
        self.is_running = True
        self.lock = threading.Lock()
        self.latest_frame = None
        
        self.schedule = {}  # {time_slot: {day: required_count}}
        self.live_count = 0
        self.required_count = 0
        self.current_time_slot = ""
        self.last_alert_time = 0
        self.alert_cooldown = 3  # 5 minutes between alerts
        
        # Track if requirement is met
        self.requirement_met = False
        self.requirement_met_time = 0
        self.pause_after_met_duration = 3  # Pause for 5 minutes after requirement met
        
        # ROI Configuration
        self.roi_polygon = None
        self._load_roi_from_db()
        
        # Load schedule from database
        self._load_schedule_from_db()
        
        logging.info(f"✅ Occupancy Monitor initialized for {self.channel_name}")
    
    def _load_roi_from_db(self):
        """Load ROI configuration from database for this channel"""
        try:
            if not self.database_url:
                logging.info(f"No DATABASE_URL found for OccupancyMonitor {self.channel_name} - skipping ROI load")
                return
            
            database_url = self.database_url
            
            engine = create_engine(database_url, pool_pre_ping=True)
            SessionLocal_roi = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            
            with SessionLocal_roi() as db:
                query = text("""
                    SELECT roi_points FROM roi_configs 
                    WHERE channel_id = :channel_id AND app_name = 'OccupancyMonitor'
                """)
                result = db.execute(query, {"channel_id": self.channel_id}).fetchone()
                
                if result and result[0]:
                    roi_data = result[0] if isinstance(result[0], dict) else json.loads(result[0])
                    points = roi_data.get('points', [])
                    
                    if points and len(points) >= 3:
                        self.roi_polygon = Polygon(points)
                        logging.info(f"✅ OccupancyMonitor ROI loaded for {self.channel_name}: {len(points)} points")
                    else:
                        logging.info(f"No valid ROI configured for OccupancyMonitor {self.channel_name} - monitoring entire frame")
                else:
                    logging.info(f"No ROI configured for OccupancyMonitor {self.channel_name} - monitoring entire frame")
        except Exception as e:
            logging.error(f"Error loading OccupancyMonitor ROI for {self.channel_name}: {e}")
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
            
            # Avoid division by zero
            if bbox_area == 0:
                return False
            
            # Calculate overlap percentage
            overlap_ratio = intersection.area / bbox_area
            
            return overlap_ratio >= overlap_threshold
        except Exception as e:
            logging.error(f"Error checking ROI overlap: {e}")
            return True  # Default to include on error
    
    def _load_schedule_from_db(self):
        """Load schedule from database for this channel"""
        try:
            with self.SessionLocal() as db:
                records = db.query(self.OccupancySchedule).filter_by(channel_id=self.channel_id).all()
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
                db.query(self.OccupancySchedule).filter_by(channel_id=self.channel_id).delete()
                
                for time_slot, days in schedule_data.items():
                    for day_name, required_count in days.items():
                        db.add(self.OccupancySchedule(
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
    
    def update_roi(self, roi_data):
        """Update ROI configuration in real-time"""
        try:
            points = roi_data.get('points', [])
            if points and len(points) >= 3:
                self.roi_polygon = Polygon(points)
                logging.info(f"✅ ROI updated in real-time for {self.channel_name}: {len(points)} points")
                return True
            else:
                logging.warning(f"Invalid ROI data for {self.channel_name}")
                return False
        except Exception as e:
            logging.error(f"Error updating ROI for {self.channel_name}: {e}")
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
        now = datetime.now(self.timezone)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        current_day = days[now.weekday()]
        current_hour = f"{now.hour}:00"
        
        # Check if schedule exists for this time
        if current_hour in self.schedule and current_day in self.schedule[current_hour]:
            return True, current_hour, current_day, self.schedule[current_hour][current_day]
        return False, current_hour, current_day, 0
    
    def _draw_roi_overlay(self, frame):
        """Draw ROI polygon overlay on frame"""
        if self.roi_polygon is None:
            return frame
        
        try:
            frame_height, frame_width = frame.shape[:2]
            roi_points = np.array([
                [int(x * frame_width), int(y * frame_height)] 
                for x, y in self.roi_polygon.exterior.coords
            ], dtype=np.int32)
            
            # Draw ROI polygon
            cv2.polylines(frame, [roi_points], True, (255, 255, 0), 2)
            
            # Add ROI label
            if len(roi_points) > 0:
                cv2.putText(frame, "ROI", (roi_points[0][0], roi_points[0][1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        except Exception as e:
            logging.error(f"Error drawing ROI overlay: {e}")
        
        return frame
    
    def _detect_people(self, frame):
        """Enhanced YOLO detection with CUDA support, maximum accuracy, and ROI filtering"""
        try:
            frame_height, frame_width = frame.shape[:2]
            
            # Enhanced detection with very low confidence for maximum recall
            results = self.model(
                frame, 
                conf=0.15,           # VERY LOW threshold for maximum detection
                iou=0.40,            # Lowered IOU for better NMS
                classes=[0],         # Only detect person class
                verbose=False,
                device=self.device,  # Use CUDA if available
                imgsz=640,           # Image size
                max_det=100,         # Handle up to 100 people
                agnostic_nms=True,   # Class-agnostic NMS
                half=False           # Full precision for accuracy
            )
            person_count = 0
            detections = []
            
            annotated_frame = frame.copy()
            
            # Draw ROI polygon if configured
            if self.roi_polygon is not None:
                try:
                    roi_points = np.array([
                        [int(x * frame_width), int(y * frame_height)] 
                        for x, y in self.roi_polygon.exterior.coords
                    ], dtype=np.int32)
                    cv2.polylines(annotated_frame, [roi_points], True, (255, 255, 0), 2)
                    cv2.putText(annotated_frame, "ROI", (roi_points[0][0], roi_points[0][1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                except Exception as e:
                    logging.error(f"Error drawing ROI: {e}")
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    
                    # Very low threshold - catch everyone!
                    if conf > 0.15:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # ROI filtering - only count people inside ROI
                        if not self._is_in_roi(x1, y1, x2, y2, frame_width, frame_height):
                            # Skip filtered-out detections
                            continue
                        
                        person_count += 1
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
                        
                        # Draw bounding box only (no labels)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
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
        now = datetime.now(self.timezone)
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
                db.add(self.OccupancyLog(
                    channel_id=self.channel_id,
                    timestamp=datetime.now(self.timezone),
                    time_slot=current_hour,
                    day_of_week=current_day,
                    live_count=self.live_count,
                    required_count=self.required_count,
                    status=status
                ))
                db.commit()
        except Exception as e:
            logging.error(f"Error logging occupancy: {e}")
        
        # Generate banner text
        banner_text = ''
        if status == 'BELOW_REQUIREMENT':
            shortage = self.required_count - self.live_count
            banner_text = f"⚠️ ALERT: {shortage} people short! ({self.live_count}/{self.required_count} present)"
        elif status == 'OK':
            banner_text = f"✅ OK ({self.live_count}/{self.required_count} present)"
        elif status == 'PAUSED':
            banner_text = f"✓ Requirement met - Monitoring paused"
        elif status == 'NO_SCHEDULE':
            banner_text = ''
        
        # Emit to dashboard via SocketIO
        self.socketio.emit('occupancy_update', {
            'channel_id': self.channel_id,
            'channel_name': self.channel_name,
            'time_slot': self.current_time_slot,
            'live_count': self.live_count,
            'required_count': self.required_count,
            'status': status,
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
        
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            logging.error(f"Failed to open RTSP stream: {self.rtsp_url}")
            return
        
        # Zero-lag settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
        cap.set(cv2.CAP_PROP_FPS, 10)  # Ultra-low capture FPS
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPEG for faster decode
        
        # Get stream FPS for smooth playback
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0 and fps < 120:
            frame_delay = 1.0 / fps
            logging.info(f"Video FPS: {fps:.1f}, Frame delay: {frame_delay:.3f}s")
        else:
            # RTSP stream - use minimal delay
            frame_delay = 0.01  # 100 FPS max for RTSP (smooth streaming)
            logging.info(f"RTSP stream detected, using minimal frame delay: {frame_delay}s")
        
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        last_schedule_check = 0
        last_detection_time = 0
        detection_cooldown = 1.0  # Run YOLO detection once per second (avoid GPU overload)
        frame_delay = 0.03  # ~30 FPS
        
        # Close the direct capture - we'll use frame_hub instead
        cap.release()
        
        while self.is_running:
            frame_start_time = time.time()
            
            # Get frame from shared frame hub (same as Queue Monitor)
            frame = getattr(self, 'frame_hub', None).get_latest() if hasattr(self, 'frame_hub') else None
            if frame is None:
                time.sleep(0.01)
                continue
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
                
                with self.lock:
                    self.latest_frame = annotated_frame
                
                # Check requirement
                self._check_occupancy_requirement()
                
            elif should_detect:
                # Between YOLO detections - still show smooth video
                # This ensures smooth streaming without frame skip
                display_frame = frame.copy()
                
                # Draw ROI overlay
                display_frame = self._draw_roi_overlay(display_frame)
                
                with self.lock:
                    self.latest_frame = display_frame
                    
            else:
                # PAUSED/NO SCHEDULE - Show clean video frame without overlays
                display_frame = frame.copy()
                
                # Optional: Add tiny status indicator in corner (minimal intrusion)
                if detection_status == "NO_SCHEDULE":
                    # Tiny gray dot indicator
                    cv2.circle(display_frame, (display_frame.shape[1] - 20, 20), 8, (100, 100, 100), -1)
                elif detection_status == "PAUSED_REQ_MET":
                    # Tiny green dot indicator
                    cv2.circle(display_frame, (display_frame.shape[1] - 20, 20), 8, (0, 200, 0), -1)
                
                with self.lock:
                    self.latest_frame = display_frame
            
            # Maintain smooth FPS - NO FRAME SKIPPING
            elapsed = time.time() - frame_start_time
            sleep_time = max(0, frame_delay - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        cap.release()
        logging.info(f"Occupancy Monitor stopped for {self.channel_name}")
    
    def stop(self):
        """Stop the processor"""
        logging.info(f"Stopping Occupancy Monitor for {self.channel_name}...")
        self.is_running = False
    
    def shutdown(self):
        """Shutdown method for compatibility"""
        self.stop()


def run_occupancy_monitor(config, frame_hub):
    """
    Public entry function to run occupancy monitor.
    
    Args:
        config: Configuration dictionary containing:
            - rtsp_url: Camera RTSP stream URL
            - channel_id: Unique channel identifier
            - channel_name: Human-readable channel name
            - model: YOLO model for person detection
            - socketio: SocketIO instance for real-time updates
            - session_factory: SQLAlchemy session factory
            - notification_sender: Function to send notifications
            - OccupancyLog: SQLAlchemy ORM class for occupancy logs table
            - OccupancySchedule: SQLAlchemy ORM class for occupancy schedules table
            - timezone: Timezone object (default: Asia/Kolkata)
            - database_url: Database connection URL
            - device: Device to use ('cpu' or 'cuda')
        frame_hub: FrameHub instance providing camera frames
    
    Returns:
        OccupancyMonitorProcessor: The running processor instance
    """
    # Extract table classes from config
    OccupancyLog = config.get('OccupancyLog')
    OccupancySchedule = config.get('OccupancySchedule')
    
    # Create processor with all dependencies injected
    processor = OccupancyMonitorProcessor(
        rtsp_url=config.get('rtsp_url'),
        channel_id=config.get('channel_id'),
        channel_name=config.get('channel_name'),
        model=config.get('model'),
        socketio=config.get('socketio'),
        SessionLocal=config.get('session_factory'),
        send_notification=config.get('notification_sender'),
        OccupancyLog=OccupancyLog,
        OccupancySchedule=OccupancySchedule,
        timezone=config.get('timezone'),
        database_url=config.get('database_url'),
        device=config.get('device', 'cuda')
    )
    
    # Attach frame hub
    processor.frame_hub = frame_hub
    
    # Start processor thread
    processor.start()
    
    return processor
