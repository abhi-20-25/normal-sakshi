#!/usr/bin/env python3
"""
Idle Time Monitor - Detects staff members staying idle in designated zones
Uses ROI polygon to define idle zones and tracks dwell time
"""

import threading
import time
import logging
import json
import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
import pytz

# Timezone
IST = pytz.timezone('Asia/Kolkata')

# Database setup (will be imported from main app in production)
Base = declarative_base()

class IdleTimeMonitor(threading.Thread):
    """
    Monitors designated idle zones and alerts when staff stays idle for too long
    
    Features:
    - ROI-based detection (polygon zones where idle is not allowed)
    - Person tracking with unique IDs
    - Configurable dwell time threshold (default 5 minutes)
    - Screenshot alerts with cooldown
    - Database logging
    """
    
    def __init__(self, rtsp_url, channel_id, channel_name, model, frame_hub=None, detection_callback=None):
        super().__init__(name=f"IdleTime-{channel_name}")
        self.rtsp_url = rtsp_url
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.model = model
        self.frame_hub = frame_hub
        self.detection_callback = detection_callback
        
        # Threading
        self.is_running = True
        self.lock = threading.Lock()
        self.latest_frame = None
        
        # Idle detection settings
        self.dwell_threshold = 300  # 5 minutes in seconds
        self.cooldown = 600  # 10 minutes between alerts for same person
        self.screenshot_cooldown = 30  # 30 seconds between screenshots
        
        # Person tracking
        self.person_tracks = {}  # track_id: {'entry_time': timestamp, 'bbox': bbox, 'zone': zone_name, 'last_alert': timestamp}
        self.last_screenshot_time = 0
        
        # ROI configuration
        self.normalized_idle_zones = []  # List of zones: [{'name': 'corner', 'points': [[x,y], ...]}, ...]
        self.idle_zone_polygons = []  # Converted to pixel coordinates
        
        # Load ROI from database
        self._load_roi_from_db()
        
        # Stats
        self.total_idle_alerts = 0
        self.current_idle_count = 0
        
        logging.info(f"🚀 IdleTimeMonitor initialized for {channel_name}")
    
    def _load_roi_from_db(self):
        """Load idle zone ROI from database"""
        try:
            # Import after initialization to avoid circular dependency
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            
            # Try importing from edit-004 module (it will be in the same directory)
            try:
                from sqlalchemy.orm import sessionmaker
                from sqlalchemy import create_engine
                DATABASE_URL = "postgresql://postgres:Tneural01@127.0.0.1:5432/sakshi"
                engine = create_engine(DATABASE_URL)
                SessionLocal = sessionmaker(bind=engine)
                
                # Define RoiConfig class locally
                from sqlalchemy import Column, Integer, String, Text
                from sqlalchemy.ext.declarative import declarative_base
                Base = declarative_base()
                
                class RoiConfig(Base):
                    __tablename__ = "roi_configs"
                    id = Column(Integer, primary_key=True)
                    channel_id = Column(String)
                    app_name = Column(String)
                    roi_points = Column(Text)
                    restaurant_id = Column(Integer)
                
            except ImportError:
                logging.warning("Could not import database classes, ROI will not be loaded")
                return
            
            with SessionLocal() as db:
                roi_record = db.query(RoiConfig).filter_by(
                    channel_id=self.channel_id,
                    app_name='IdleTimeMonitor'
                ).first()
                
                if roi_record and roi_record.roi_points:
                    try:
                        points_data = json.loads(roi_record.roi_points)
                        
                        # Support both formats: {"zones": [...]} or {"points": [...]}
                        if "zones" in points_data:
                            self.normalized_idle_zones = points_data["zones"]
                        elif "points" in points_data:
                            # Single zone format
                            self.normalized_idle_zones = [{
                                'name': 'idle_zone_1',
                                'points': points_data["points"]
                            }]
                        
                        logging.info(f"✅ Loaded {len(self.normalized_idle_zones)} idle zone(s) for {self.channel_name} from database")
                        for i, zone in enumerate(self.normalized_idle_zones):
                            logging.info(f"   Zone {i+1} '{zone.get('name', 'unnamed')}': {len(zone['points'])} points")
                        return
                    except (json.JSONDecodeError, TypeError, KeyError) as e:
                        logging.error(f"Failed to parse ROI JSON from DB: {e}")
                else:
                    logging.warning(f"No idle zone ROI in DB for {self.channel_name}")
        except Exception as e:
            logging.error(f"Error loading ROI from database: {e}")
        
        # Initialize with empty zones
        self.normalized_idle_zones = []
        self.idle_zone_polygons = []
    
    def _convert_normalized_to_pixels(self, frame_width, frame_height):
        """Convert normalized ROI coordinates (0-1) to pixel coordinates"""
        self.idle_zone_polygons = []
        
        for zone in self.normalized_idle_zones:
            points = zone.get('points', [])
            if len(points) >= 3:
                pixel_coords = [(int(p[0] * frame_width), int(p[1] * frame_height)) for p in points]
                poly = Polygon(pixel_coords)
                
                # Fix invalid polygons
                if not poly.is_valid:
                    poly = poly.buffer(0)
                
                self.idle_zone_polygons.append({
                    'name': zone.get('name', f'zone_{len(self.idle_zone_polygons)+1}'),
                    'polygon': poly,
                    'pixel_points': pixel_coords
                })
        
        if self.idle_zone_polygons:
            logging.info(f"✅ Converted {len(self.idle_zone_polygons)} idle zone polygon(s) to pixel coordinates")
    
    def _is_point_in_idle_zone(self, point):
        """Check if point is inside any idle zone, return zone info if yes"""
        x, y = point
        pt = Point(x, y)
        
        for zone_data in self.idle_zone_polygons:
            if zone_data['polygon'].contains(pt):
                return True, zone_data['name']
        
        return False, None
    
    def update_roi(self, new_roi_points):
        """Update ROI dynamically (called from API)"""
        with self.lock:
            try:
                # Update normalized zones
                if "zones" in new_roi_points:
                    self.normalized_idle_zones = new_roi_points["zones"]
                elif "points" in new_roi_points:
                    self.normalized_idle_zones = [{
                        'name': 'idle_zone_1',
                        'points': new_roi_points["points"]
                    }]
                
                # Reconvert to pixels if frame is available
                if self.latest_frame is not None:
                    h, w = self.latest_frame.shape[:2]
                    self._convert_normalized_to_pixels(w, h)
                
                logging.info(f"🎯 IdleTimeMonitor {self.channel_name} ROI updated successfully!")
            except Exception as e:
                logging.error(f"Error updating ROI for {self.channel_name}: {e}")
    
    def _send_idle_alert(self, track_id, dwell_time, frame, zone_name):
        """Send alert with screenshot when idle detected"""
        current_time = time.time()
        
        # Check screenshot cooldown
        if current_time - self.last_screenshot_time < self.screenshot_cooldown:
            return
        
        # Check per-person cooldown
        if track_id in self.person_tracks:
            last_alert = self.person_tracks[track_id].get('last_alert', 0)
            if current_time - last_alert < self.cooldown:
                return
        
        # Prepare alert message
        minutes = int(dwell_time / 60)
        seconds = int(dwell_time % 60)
        message = f"Idle Staff Detected: {minutes}m {seconds}s in {zone_name}"
        
        logging.warning(f"⚠️ {self.channel_name}: {message}")
        
        # Send screenshot via callback
        if self.detection_callback:
            self.detection_callback(
                'IdleTimeMonitor',
                self.channel_id,
                [frame],
                message,
                False  # is_gif
            )
        
        # Update tracking
        self.last_screenshot_time = current_time
        if track_id in self.person_tracks:
            self.person_tracks[track_id]['last_alert'] = current_time
        
        self.total_idle_alerts += 1
    
    def get_frame(self):
        """Get latest annotated frame for video streaming"""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def shutdown(self):
        """Gracefully shutdown the monitor"""
        logging.info(f"Shutting down IdleTimeMonitor for {self.channel_name}")
        self.is_running = False
    
    def run(self):
        """Main processing loop"""
        logging.info(f"🚀 IdleTimeMonitor thread starting for {self.channel_name}")
        
        frame_count = 0
        roi_initialized = False
        
        while self.is_running:
            # Get frame from FrameHub
            frame = self.frame_hub.get_latest() if self.frame_hub else None
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            h, w = frame.shape[:2]
            
            # Initialize ROI polygons on first valid frame
            if not roi_initialized:
                self._convert_normalized_to_pixels(w, h)
                roi_initialized = True
                
                if not self.idle_zone_polygons:
                    logging.warning(f"⚠️ {self.channel_name}: No idle zones configured, monitor will not detect idle staff")
            
            # Run person detection with tracking
            try:
                results = self.model.track(
                    frame,
                    persist=True,
                    conf=0.15,
                    classes=[0],  # Person class only
                    verbose=False,
                    device='cpu',
                    half=False
                )
            except Exception as e:
                logging.error(f"Error in person detection for {self.channel_name}: {e}")
                continue
            
            if not results or len(results[0].boxes) == 0:
                # No persons detected - clean up old tracks
                with self.lock:
                    self.person_tracks.clear()
                    self.current_idle_count = 0
                annotated_frame = frame.copy()
            else:
                annotated_frame = frame.copy()
                current_time = time.time()
                current_track_ids = set()
                
                # Process each detected person
                for box in results[0].boxes:
                    if box.id is None:
                        continue
                    
                    track_id = int(box.id[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    
                    # Calculate center point of bounding box
                    center_x = int((bbox[0] + bbox[2]) / 2)
                    center_y = int((bbox[1] + bbox[3]) / 2)
                    center_point = (center_x, center_y)
                    
                    # Check if person is in idle zone
                    in_idle_zone, zone_name = self._is_point_in_idle_zone(center_point)
                    
                    if in_idle_zone:
                        current_track_ids.add(track_id)
                        
                        # New person entering idle zone
                        if track_id not in self.person_tracks:
                            self.person_tracks[track_id] = {
                                'entry_time': current_time,
                                'bbox': bbox,
                                'zone': zone_name,
                                'last_alert': 0
                            }
                            logging.info(f"Person {track_id} entered idle zone '{zone_name}' at {self.channel_name}")
                        
                        # Calculate dwell time
                        entry_time = self.person_tracks[track_id]['entry_time']
                        dwell_time = current_time - entry_time
                        
                        # Check if exceeded threshold
                        if dwell_time > self.dwell_threshold:
                            # Send alert
                            self._send_idle_alert(track_id, dwell_time, annotated_frame, zone_name)
                            
                            # Draw RED bounding box for idle person
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            
                            # Draw label
                            minutes = int(dwell_time / 60)
                            label = f"IDLE {minutes}min {zone_name}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(annotated_frame, (x1, y1-25), (x1+label_size[0]+5, y1), (0, 0, 255), -1)
                            cv2.putText(annotated_frame, label, (x1+2, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        else:
                            # Draw YELLOW bounding box for person in zone but not yet idle
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            
                            # Show countdown
                            remaining = self.dwell_threshold - dwell_time
                            minutes = int(remaining / 60)
                            seconds = int(remaining % 60)
                            label = f"{minutes}:{seconds:02d} {zone_name}"
                            cv2.putText(annotated_frame, label, (x1, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    else:
                        # Person not in idle zone - draw green box
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Remove people who left idle zones
                tracks_to_remove = []
                for track_id in self.person_tracks:
                    if track_id not in current_track_ids:
                        zone = self.person_tracks[track_id]['zone']
                        logging.info(f"Person {track_id} left idle zone '{zone}' at {self.channel_name}")
                        tracks_to_remove.append(track_id)
                
                for track_id in tracks_to_remove:
                    del self.person_tracks[track_id]
                
                # Update current idle count
                with self.lock:
                    self.current_idle_count = len(self.person_tracks)
            
            # Draw idle zones on frame
            for zone_data in self.idle_zone_polygons:
                pts = np.array(zone_data['pixel_points'], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], True, (255, 0, 255), 2)  # Magenta
                
                # Draw zone name
                if len(zone_data['pixel_points']) > 0:
                    x, y = zone_data['pixel_points'][0]
                    cv2.putText(annotated_frame, zone_data['name'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Add info overlay
            info_lines = [
                f"Frame: {frame_count}",
                f"Idle Zones: {len(self.idle_zone_polygons)}",
                f"In Zone: {self.current_idle_count}",
                f"Total Alerts: {self.total_idle_alerts}"
            ]
            
            y_offset = 30
            for line in info_lines:
                cv2.rectangle(annotated_frame, (5, y_offset-20), (300, y_offset+5), (0, 0, 0), -1)
                cv2.putText(annotated_frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 30
            
            # Store frame for streaming
            with self.lock:
                self.latest_frame = annotated_frame
            
            # Log stats every 100 frames
            if frame_count % 100 == 0:
                logging.info(f"IdleTimeMonitor {self.channel_name}: Frame {frame_count}, In Zone: {self.current_idle_count}, Alerts: {self.total_idle_alerts}")
        
        logging.info(f"IdleTimeMonitor thread stopped for {self.channel_name}")
