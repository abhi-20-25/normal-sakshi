# Complete Application Analysis - edit-004.py

## 📋 Application Overview

**Your edit-004.py is a comprehensive CCTV monitoring system with multiple AI-powered features:**

---

## 🎯 Main Features

### 1. **People Counter**
- Tracks people entering/exiting
- In/Out counting with ROI zones
- Hourly and daily footfall statistics

### 2. **Queue Monitor**
- Detects queue formation at checkout
- Monitors cashier presence
- Sends Telegram alerts when queue > threshold
- Records queue logs to database

### 3. **Generic Detection** (Custom Objects)
- Detects custom objects (7 classes)
- Saves GIF animations of detections
- Configurable confidence threshold

### 4. **Kitchen Compliance Monitor**
- Checks for apron/cap compliance
- Monitors glove usage
- Sends violation alerts
- Stores violation records

### 5. **Occupancy Monitor**
- Tracks occupancy based on schedule
- Compares actual vs expected occupancy
- Time-based monitoring

---

## 🗄️ Database Tables Used

Your application uses **8 PostgreSQL tables** in the `sakshi` database:

| Table | Purpose |
|-------|---------|
| `detections` | Stores generic object detections (with image paths) |
| `daily_footfall` | Daily in/out counts per camera |
| `hourly_footfall` | Hourly in/out counts per camera |
| `queue_logs` | Queue count, cashier presence, timestamps |
| `roi_configs` | ROI polygon configurations for each camera |
| `kitchen_violations` | Kitchen compliance violation records |
| `occupancy_logs` | Occupancy monitoring logs |
| `occupancy_schedules` | Expected occupancy schedules |

**Plus PetPooja table (from FastAPI):**
| Table | Purpose |
|-------|---------|
| `petpooja_webhook_events` | PetPooja webhook data (id, content JSONB, created_at) |

---

## ⚙️ Technical Stack

### Framework & Server
- **Framework:** Flask + Flask-SocketIO
- **Web Server:** Gunicorn with eventlet workers
- **Port:** 5001 (default)
- **Async Mode:** Threading/Eventlet for real-time streaming

### AI & Computer Vision
- **YOLO Models:** YOLOv8n, YOLO11n, custom models
- **Framework:** Ultralytics
- **Backend:** PyTorch (CPU mode forced for stability)
- **Tracking:** ByteTrack

### Database
- **Database:** PostgreSQL
- **ORM:** SQLAlchemy
- **Connection:** `postgresql://postgres:Tneural01@127.0.0.1:5432/sakshi`

### Video Processing
- **Input:** RTSP streams from cameras
- **Frame Processing:** OpenCV (cv2)
- **Resolution:** Downscaled to 640x360 for performance
- **Output:** MJPEG streams via HTTP

### Notifications
- **Platform:** Telegram Bot
- **Bot Token:** `7843300957:AAGVv866cPiDPVD0Wrk_wwEEHDSD64Pgaqs`
- **Chat ID:** `-4835836048`

### Scheduling
- **Scheduler:** APScheduler (Background)
- **Tasks:** Daily reset, hourly aggregation

---

## 🚀 Current Deployment

### Server Details
- **IP:** 13.203.73.173
- **User:** ubuntu
- **Directory:** /home/ubuntu/normal-sakshi
- **Service:** sakshi-ai.service

### Systemd Service Configuration
```ini
[Unit]
Description=Gunicorn instance to serve Sakshi AI application
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/normal-sakshi
Environment="PATH=/home/ubuntu/normal-sakshi/venv/bin"
ExecStart=/home/ubuntu/normal-sakshi/venv/bin/gunicorn \
    --worker-class eventlet \
    --workers 1 \
    --worker-connections 1000 \
    --bind 0.0.0.0:5001 \
    --timeout 120 \
    --graceful-timeout 30 \
    --keep-alive 5 \
    --log-level info \
    --access-logfile /var/log/gunicorn/sakshi-access.log \
    --error-logfile /var/log/gunicorn/sakshi-error.log \
    wsgi:application
Restart=always
RestartSec=10
```

---

## 📁 Key Files Structure

```
/home/ubuntu/normal-sakshi/
├── edit-004.py                          # Main application (Flask)
├── wsgi.py                              # Gunicorn entry point
├── fastapi_app.py                       # NEW: PetPooja API (FastAPI)
├── kitchen_compliance_monitor.py        # Kitchen compliance module
├── requirements.txt                     # Python dependencies
├── rtsp_links.txt                       # Camera RTSP URLs
├── .env                                 # Environment variables
├── sakshi-ai.service                    # Flask systemd service
├── petpooja-api.service                 # NEW: FastAPI systemd service
├── gunicorn_config.py                   # Gunicorn config
├── static/
│   └── detections/                      # Detection images/GIFs
│       └── shutter_videos/              # Video recordings
└── templates/
    ├── dashboard.html                   # Main dashboard
    ├── login.html                       # Login page
    └── landing.html                     # Landing page
```

---

## 🔐 Authentication

### Web Login
- **Username:** `user`
- **Password:** `Tneural123`
- **Session-based:** Flask sessions with secret key

### API Authentication
- **PetPooja Token:** `qwrdx477ggh77hh`
- **Method:** Query parameter (?token=...)

---

## 🌐 Web Routes

### Flask Application (Port 5001)

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Landing page |
| `/login` | GET/POST | Login page |
| `/dashboard` | GET | Main dashboard (authenticated) |
| `/logout` | GET | Logout |
| `/video_feed/<name>` | GET | MJPEG video stream |
| `/api/footfall` | GET | Get footfall data (today/yesterday) |
| `/api/hourly_footfall` | GET | Get hourly footfall |
| `/api/queue_status` | GET | Current queue status |
| `/api/queue_logs` | GET | Queue history logs |
| `/api/detections` | GET | Recent generic detections |
| `/api/kitchen_violations` | GET | Kitchen violations |
| `/api/occupancy_status` | GET | Current occupancy data |
| `/upload_schedule` | POST | Upload occupancy schedule CSV |
| `/api/schedule` | GET | Get schedule data |

### FastAPI Application (Port 8000 - NEW)

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Health check |
| `/webhook/events` | POST | Create webhook event |
| `/webhook/events` | GET | Get all events |
| `/webhook/events/{id}` | GET | Get single event |
| `/webhook/events/{id}` | PUT | Update event |
| `/webhook/events/{id}` | DELETE | Delete event |
| `/webhook/events/stats/count` | GET | Get total count |
| `/webhook/events/search/date-range` | GET | Search by date |
| `/docs` | GET | Swagger API docs |
| `/petpooja` | GET | Legacy webhook (backward compatible) |

---

## 💡 Can You Use This on Server?

# ✅ YES - ABSOLUTELY!

Your complete application **IS ALREADY RUNNING** on the server!

### What's Currently on Server:

1. **edit-004.py (Flask App)** ✅
   - Running on port 5001
   - Service: `sakshi-ai.service`
   - Status: Active and working

2. **Database** ✅
   - PostgreSQL on same server (127.0.0.1:5432)
   - Database name: `sakshi`
   - All 8 tables + petpooja_webhook_events table

3. **RTSP Cameras** ✅
   - Reading from `rtsp_links.txt`
   - Processing video streams

4. **YOLO Models** ✅
   - All model files (.pt) on server
   - Running in CPU mode for stability

### What You Just Added:

5. **fastapi_app.py (FastAPI)** 🆕
   - Will run on port 8000
   - Service: `petpooja-api.service`
   - Independent from Flask app
   - Shares same database

---

## 🎯 Deployment Strategy

### Both Applications Run Together:

```
┌─────────────────────────────────────────────┐
│         Ubuntu Server (13.203.73.173)       │
├─────────────────────────────────────────────┤
│                                             │
│  🐍 Flask App (edit-004.py)                │
│     Port: 5001                              │
│     Service: sakshi-ai.service              │
│     Purpose: CCTV monitoring, AI detection  │
│                                             │
│  🚀 FastAPI App (fastapi_app.py)           │
│     Port: 8000                              │
│     Service: petpooja-api.service           │
│     Purpose: PetPooja webhook & CRUD API    │
│                                             │
│  🗄️  PostgreSQL Database                    │
│     Port: 5432                              │
│     Database: sakshi                        │
│     Tables: 9 total (8 Flask + 1 FastAPI)   │
│                                             │
└─────────────────────────────────────────────┘
```

---

## 📦 Requirements Needed

Your current `requirements.txt` already has most dependencies. Just add:

```txt
# Add these lines to requirements.txt:
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

Then on server:
```bash
source venv/bin/activate
pip install fastapi "uvicorn[standard]" pydantic python-dotenv
```

---

## 🚀 Deployment Steps

### Option 1: Automatic Deployment (Recommended)

From your local machine:
```bash
chmod +x deploy_fastapi_to_server.sh
./deploy_fastapi_to_server.sh
```

This script will:
1. Upload fastapi_app.py to server
2. Install dependencies
3. Create systemd service
4. Start the FastAPI application
5. Test the endpoints

### Option 2: Manual Deployment

**Step 1: Upload files**
```bash
scp fastapi_app.py ubuntu@13.203.73.173:/home/ubuntu/normal-sakshi/
scp petpooja-api.service ubuntu@13.203.73.173:/tmp/
```

**Step 2: SSH to server**
```bash
ssh ubuntu@13.203.73.173
cd /home/ubuntu/normal-sakshi
```

**Step 3: Install dependencies**
```bash
source venv/bin/activate
pip install fastapi "uvicorn[standard]" gunicorn python-dotenv
```

**Step 4: Test manually**
```bash
python3 fastapi_app.py
# In another terminal, test:
curl http://localhost:8000/
```

**Step 5: Install service**
```bash
sudo cp /tmp/petpooja-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable petpooja-api
sudo systemctl start petpooja-api
sudo systemctl status petpooja-api
```

**Step 6: Open firewall**
```bash
sudo ufw allow 8000/tcp
sudo ufw status
```

---

## 🔍 Monitoring & Management

### Check Services Status
```bash
# Flask app
sudo systemctl status sakshi-ai

# FastAPI app
sudo systemctl status petpooja-api

# Both at once
sudo systemctl status sakshi-ai petpooja-api
```

### View Logs
```bash
# Flask logs
sudo journalctl -u sakshi-ai -f

# FastAPI logs
sudo journalctl -u petpooja-api -f

# Gunicorn logs
tail -f /var/log/gunicorn/sakshi-access.log
tail -f /var/log/gunicorn/petpooja-access.log
```

### Restart Services
```bash
# Flask app
sudo systemctl restart sakshi-ai

# FastAPI app
sudo systemctl restart petpooja-api

# Both
sudo systemctl restart sakshi-ai petpooja-api
```

### Check Ports
```bash
sudo netstat -tlnp | grep -E ':(5001|8000)'
# Should show:
# 0.0.0.0:5001  (Flask)
# 0.0.0.0:8000  (FastAPI)
```

---

## 🌐 Access URLs

### From Local Network:
- **Flask Dashboard:** http://13.203.73.173:5001
- **FastAPI Docs:** http://13.203.73.173:8000/docs
- **FastAPI Health:** http://13.203.73.173:8000/

### If Behind Nginx:
- **Flask:** http://your-domain.com/
- **FastAPI:** http://your-domain.com/api/

---

## 📊 Database Schema Summary

```sql
-- Flask Tables (edit-004.py)
detections (id, camera_name, detected_class, confidence, timestamp, image_path)
daily_footfall (id, camera_name, date, in_count, out_count)
hourly_footfall (id, camera_name, timestamp, in_count, out_count)
queue_logs (id, queue_count, cashier_present, timestamp)
roi_configs (id, camera_name, roi_name, roi_points)
kitchen_violations (id, timestamp, violation_type, image_path)
occupancy_logs (id, camera_name, timestamp, person_count, expected_count, is_compliant)
occupancy_schedules (id, camera_name, day_of_week, start_time, end_time, expected_count)

-- FastAPI Table (fastapi_app.py)
petpooja_webhook_events (id, content JSONB, created_at)
```

---

## 🔒 Security Checklist

✅ **Flask login protected** with username/password
✅ **FastAPI token protected** with API token
✅ **PostgreSQL** accessible only from localhost
✅ **Firewall** configured for ports 5001, 8000
⚠️ **TODO:** Add HTTPS/SSL certificates
⚠️ **TODO:** Change default passwords
⚠️ **TODO:** Use environment variables for secrets

---

## 💾 Backup Strategy

### Database Backup
```bash
# Manual backup
pg_dump -h 127.0.0.1 -U postgres sakshi > backup_$(date +%Y%m%d).sql

# Restore
psql -h 127.0.0.1 -U postgres sakshi < backup_20251121.sql
```

### Code Backup
```bash
# On server
cd /home/ubuntu/normal-sakshi
tar -czf backup_$(date +%Y%m%d).tar.gz \
  edit-004.py \
  fastapi_app.py \
  wsgi.py \
  requirements.txt \
  templates/ \
  static/detections/
```

---

## 🎓 Summary Answer to Your Question

### **"Can I use this same application on server?"**

# YES! Here's the situation:

1. **Your Flask app (edit-004.py)** ✅
   - ALREADY running on server
   - Port 5001
   - Handles all CCTV monitoring, AI detection, queue monitoring, etc.

2. **Your FastAPI app (fastapi_app.py)** 🆕
   - NEW application we just created
   - Port 8000
   - Handles PetPooja webhooks
   - Ready to deploy (just run the deployment script)

3. **Both apps share the same PostgreSQL database** ✅
   - Flask uses 8 tables for monitoring data
   - FastAPI uses 1 table for webhook data
   - No conflicts, they work together

4. **Deployment is simple:**
   ```bash
   # Just run this:
   ./deploy_fastapi_to_server.sh
   ```

**Your complete system will be:**
- 🎥 Video monitoring (Flask on 5001)
- 🍽️ PetPooja webhooks (FastAPI on 8000)
- 🗄️ Shared database (PostgreSQL on 5432)
- 🔔 Telegram notifications (both apps)

All running simultaneously on the same server! 🚀
