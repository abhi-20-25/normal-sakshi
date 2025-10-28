# shutter_monitor_processor.py

import cv2
import time as pytime
import threading
import logging
from datetime import datetime, date, time as dt_time, timedelta
import pytz
from sqlalchemy import text, exc
from collections import deque
import imageio
import os

# Use the existing IST timezone from your main file
IST = pytz.timezone('Asia/Kolkata')

class ShutterMonitorProcessor(threading.Thread):
    def __init__(self, rtsp_url, channel_id, channel_name, model, socketio, telegram_sender, SessionLocal):
        super().__init__(name=f"ShutterMonitor-{channel_name}")
        self.rtsp_url = rtsp_url
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.model = model
        self.socketio = socketio
        self.send_telegram_notification = telegram_sender
        self.SessionLocal = SessionLocal
        self.is_running = True
        self.lock = threading.Lock()

        # State management variables
        self.tracking_date = date.min
        self._reset_cycle_stats()
        self._load_state_from_db()

        self.last_telegram_alert_status = None
        self.is_awaiting_new_cycle = True
        self.last_db_save_time = pytime.time()
        
        # --- MODIFIED: Time constraint for close detection ---
        self.close_detection_start_time = dt_time(20, 30) # 8:30 PM

        # For video recording
        self.fps = 10
        self.buffer_size = self.fps * 10
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.last_detection_time = pytime.time()
        self.detection_interval = 2
        self.static_folder = 'static'
        self.detections_subfolder = 'detections'

        logging.info(f"ShutterMonitorProcessor for {self.channel_name} initialized and state loaded.")

    def _reset_cycle_stats(self):
        with self.lock:
            self.tracking_date = datetime.now(IST).date()
            self.current_status = 'Unknown'
            self.last_status_time = datetime.now(IST)
            self.first_open_time_today = None
            self.first_close_time_today = None
            self.total_open_duration = timedelta(0)
            self.is_awaiting_new_cycle = True
            logging.info(f"Cycle stats reset for ShutterMonitor on {self.channel_name}")

    def _load_state_from_db(self):
        with self.lock:
            today = datetime.now(IST).date()
            logging.info(f"Attempting to load previous state for {self.channel_name} for date {today}...")
            try:
                with self.SessionLocal() as db:
                    result = db.execute(text("""
                        SELECT first_open_time, first_close_time, total_open_duration_seconds
                        FROM shutter_logs
                        WHERE channel_id = :channel_id AND report_date = :report_date
                    """), {'channel_id': self.channel_id, 'report_date': today}).first()

                    if result:
                        self.first_open_time_today = result.first_open_time
                        self.first_close_time_today = result.first_close_time
                        self.total_open_duration = timedelta(seconds=result.total_open_duration_seconds or 0)
                        logging.info(f"Successfully loaded state for {self.channel_name}: Open={self.first_open_time_today}, Close={self.first_close_time_today}")
                    else:
                        logging.info(f"No previous state found for {self.channel_name} today. Starting fresh.")
            except exc.SQLAlchemyError as e:
                logging.error(f"DB Error loading state for {self.channel_name}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error loading state for {self.channel_name}: {e}")

    def _save_cycle_to_db(self):
        with self.lock:
            if self.first_open_time_today is None: return

            report_date = self.tracking_date
            self._update_durations(force_update=True)
            
            total_seconds_in_day = 24 * 60 * 60
            open_seconds = int(self.total_open_duration.total_seconds())
            closed_seconds = total_seconds_in_day - open_seconds
            if closed_seconds < 0: closed_seconds = 0
            
            with self.SessionLocal() as db:
                try:
                    stmt = text("""
                        INSERT INTO shutter_logs (channel_id, report_date, first_open_time, first_close_time, total_open_duration_seconds, total_closed_duration_seconds)
                        VALUES (:channel_id, :report_date, :fot, :fct, :tods, :tcds)
                        ON CONFLICT (channel_id, report_date) 
                        DO UPDATE SET
                            first_open_time = COALESCE(shutter_logs.first_open_time, EXCLUDED.first_open_time),
                            first_close_time = COALESCE(shutter_logs.first_close_time, EXCLUDED.first_close_time),
                            total_open_duration_seconds = EXCLUDED.total_open_duration_seconds,
                            total_closed_duration_seconds = EXCLUDED.total_closed_duration_seconds;
                    """)
                    db.execute(stmt, {'channel_id': self.channel_id, 'report_date': report_date, 'fot': self.first_open_time_today, 'fct': self.first_close_time_today, 'tods': open_seconds, 'tcds': closed_seconds})
                    db.commit()
                except Exception as e:
                    logging.error(f"Failed to save shutter log to DB for {self.channel_name}: {e}")
                    db.rollback()

    def _emit_update(self):
        with self.lock:
            first_open_str = self.first_open_time_today.isoformat() if self.first_open_time_today else None
            first_close_str = self.first_close_time_today.isoformat() if self.first_close_time_today else None
            
            total_seconds_in_day = 24 * 60 * 60
            open_seconds = int(self.total_open_duration.total_seconds())
            closed_seconds = total_seconds_in_day - open_seconds
            if closed_seconds < 0: closed_seconds = 0

            payload = {'channel_id': self.channel_id, 'last_status': self.current_status, 'last_status_time': self.last_status_time.isoformat(), 'first_open_time': first_open_str, 'first_close_time': first_close_str, 'total_open_duration_seconds': open_seconds, 'total_closed_duration_seconds': closed_seconds}
        self.socketio.emit('shutter_update', payload)

    def _handle_telegram_alert(self, status_to_alert):
        if status_to_alert == self.last_telegram_alert_status: return
        now_ist = datetime.now(IST)
        current_time = now_ist.time()
        morning_start, morning_end = dt_time(8, 0), dt_time(11, 30)
        evening_start, evening_end = dt_time(21, 0), dt_time(23, 0)
        if (morning_start <= current_time <= morning_end) or (evening_start <= current_time <= evening_end):
            message = f"🚨 **Shutter Alert: {self.channel_name}** 🚨\nStatus changed to: **{status_to_alert.upper()}**"
            self.send_telegram_notification(message)
            self.last_telegram_alert_status = status_to_alert
            logging.info(f"Telegram alert sent for {self.channel_name}: {status_to_alert}")

    def _update_durations(self, force_update=False):
        if self.first_open_time_today is None and not force_update:
            self.last_status_time = datetime.now(IST)
            return
        if self.current_status == 'Unknown' and not force_update: return
        now = datetime.now(IST)
        duration = now - self.last_status_time
        if self.current_status == 'open':
            self.total_open_duration += duration
        self.last_status_time = now

    def shutdown(self):
        logging.info(f"Shutting down ShutterMonitor for {self.channel_name}. Saving final state...")
        self.is_running = False
        pytime.sleep(1.0)
        self._save_cycle_to_db()

    def _save_video_and_update_db(self, frames_to_save, event_type):
        """Saves a video and updates the corresponding record in shutter_logs."""
        now = datetime.now(IST)
        ts_string = now.strftime("%Y%m%d_%H%M%S")
        filename = f"ShutterMonitor_{self.channel_id}_{event_type}_{ts_string}.mp4"
        video_subfolder = os.path.join(self.detections_subfolder, 'shutter_videos')
        video_path_relative = os.path.join(video_subfolder, filename)
        video_path_full = os.path.join(self.static_folder, video_path_relative)

        try:
            logging.info(f"Saving {len(frames_to_save)}-frame shutter video to {video_path_full}")
            rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_to_save]
            imageio.mimwrite(video_path_full, rgb_frames, fps=self.fps, quality=7, codec='libx264')
        except Exception as e:
            logging.error(f"Failed to save shutter video for {self.channel_name}: {e}")
            return

        with self.SessionLocal() as db:
            try:
                update_column = 'first_open_video_path' if event_type == 'open' else 'first_close_video_path'
                stmt = text(f"""
                    UPDATE shutter_logs SET {update_column} = :video_path
                    WHERE channel_id = :channel_id AND report_date = :report_date
                """)
                db.execute(stmt, {'video_path': video_path_relative, 'channel_id': self.channel_id, 'report_date': self.tracking_date})
                db.commit()
                logging.info(f"Updated shutter_logs with video path for {self.channel_name}.")
            except Exception as e:
                logging.error(f"Failed to update shutter_logs with video path for {self.channel_name}: {e}")
                db.rollback()

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            logging.error(f"Could not open ShutterMonitor stream for {self.channel_name}")
            return

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Reconnecting to ShutterMonitor stream {self.channel_name}...")
                pytime.sleep(5)
                cap.release(); cap = cv2.VideoCapture(self.rtsp_url)
                continue
            
            self.frame_buffer.append(frame)

            current_time = pytime.time()
            if current_time - self.last_detection_time < self.detection_interval:
                pytime.sleep(1 / (self.fps * 2))
                continue
            
            self.last_detection_time = current_time
            now_ist = datetime.now(IST)

            if now_ist.date() > self.tracking_date and self.tracking_date != date.min:
                logging.info(f"New day detected for {self.channel_name}. Saving previous day's stats and resetting.")
                self._save_cycle_to_db()
                self._reset_cycle_stats()

            latest_frame = self.frame_buffer[-1]
            results = self.model(latest_frame, conf=0.75, verbose=False)
            
            detected_status = None
            if results and len(results[0].boxes) > 0:
                best_detection = max(results[0].boxes, key=lambda x: x.conf)
                detected_status = 'open' if int(best_detection.cls) == 1 else 'close'

            if detected_status and detected_status != self.current_status:
                now = datetime.now(IST)
                self._update_durations()
                
                logging.info(f"Status change for {self.channel_name}: {self.current_status} -> {detected_status}")
                self.current_status = detected_status
                self.last_status_time = now
                
                should_record = False
                if self.current_status == 'open' and self.first_open_time_today is None:
                    self.first_open_time_today = now
                    self.tracking_date = now.date()
                    should_record = True
                
                # --- MODIFIED: Time-based logic for 'close' event ---
                elif self.current_status == 'close' and self.first_open_time_today is not None and self.first_close_time_today is None:
                    if now.time() >= self.close_detection_start_time:
                        self.first_close_time_today = now
                        should_record = True
                        logging.info(f"First close event for {self.channel_name} detected after cutoff time.")
                    else:
                        logging.info(f"Ignoring pre-cutoff 'close' event for {self.channel_name} at {now.strftime('%H:%M:%S')}.")

                if should_record:
                    # Ensure the database entry exists before trying to update it with a video path
                    self._save_cycle_to_db() 
                    frames_copy = list(self.frame_buffer)
                    threading.Thread(target=self._save_video_and_update_db, args=(frames_copy, self.current_status)).start()

                self._handle_telegram_alert(self.current_status)
                self._emit_update()
                self._save_cycle_to_db()

            if pytime.time() - self.last_db_save_time > 300: # Every 5 minutes
                self._save_cycle_to_db()
                self.last_db_save_time = pytime.time()
        
        cap.release()

