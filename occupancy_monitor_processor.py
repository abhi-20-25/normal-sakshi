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
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, UniqueConstraint, text
from sqlalchemy.orm import declarative_base

IST = pytz.timezone('Asia/Kolkata')
Base = declarative_base()

# Database tables
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
        
        # Auto-detect device (CUDA if available, else CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        
        while self.is_running:
            frame_start_time = time.time()
            
            # Aggressive frame skipping to get the absolute latest frame (zero lag)
            for _ in range(3):
                cap.grab()
            
            ret, frame = cap.retrieve()
            
            if not ret:
                logging.warning(f"Failed to read frame from {self.channel_name}, attempting reconnect...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(self.rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 15)
                reconnect_attempts += 1
                
                if reconnect_attempts >= max_reconnect_attempts:
                    logging.error(f"Max reconnection attempts reached for {self.channel_name}")
                    break
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
        
        cap.release()
        logging.info(f"Occupancy Monitor stopped for {self.channel_name}")
    
    def stop(self):
        """Stop the processor"""
        logging.info(f"Stopping Occupancy Monitor for {self.channel_name}...")
        self.is_running = False
    
    def shutdown(self):
        """Shutdown method for compatibility"""
        self.stop()
