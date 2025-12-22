# Sakshi AI - Multi-Restaurant Computer Vision System

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-Proprietary-red)

## 🎯 Overview

Sakshi AI is a comprehensive computer vision system designed for multi-restaurant operations, providing real-time monitoring and analytics for:

- **People Counting** - Accurate footfall tracking with IN/OUT detection
- **Queue Monitoring** - Queue length and waiting time analysis
- **Kitchen Compliance** - PPE detection (aprons, caps, gloves)
- **Security Monitoring** - Unauthorized access detection
- **Shutter Monitoring** - Open/closed status tracking
- **Occupancy Management** - Real-time capacity monitoring

## 🏗️ Architecture

### Tech Stack
- **Backend**: Flask + FastAPI
- **Database**: PostgreSQL with JSONB support
- **Computer Vision**: YOLO11n-seg (Ultralytics)
- **Real-time**: Socket.IO + EventLet
- **Frontend**: Vanilla JavaScript + Chart.js
- **Deployment**: Gunicorn + systemd services

### Key Features
- Multi-restaurant support with centralized management
- Database-driven ROI configuration
- Real-time video processing with RTSP streams
- RESTful API for sales/footfall integration
- Web-based ROI editors
- Automated count resets and scheduling

## 🚀 Quick Start

### Prerequisites
```bash
# System Requirements
- Ubuntu 20.04+ / Debian 11+
- Python 3.10+
- PostgreSQL 14+
- 4GB+ RAM recommended
- CUDA (optional, for GPU acceleration)
```

### Installation

1. **Clone Repository**
```bash
git clone https://github.com/abhi-20-25/normal-sakshi.git
cd normal-sakshi
```

2. **Setup Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Configure Database**
```bash
# Create database
sudo -u postgres createdb sakshi

# Run migrations
psql -U postgres -d sakshi -f phase1_create_schema.sql
python3 migrate_database.py
```

4. **Configure Environment**
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

5. **Start Services**
```bash
# Start Flask Dashboard (Port 5001)
python3 edit-004.py

# Start FastAPI Server (Port 8000)
python3 fastapi_app.py
```

## 📊 System Components

### Flask Application (edit-004.py)
- Main dashboard with real-time video feeds
- ROI editor interfaces
- Restaurant management
- Camera configuration

### FastAPI Server (fastapi_app.py)
- Sales analytics API
- Hourly/daily aggregations
- PetPooja webhook integration
- Conversion analytics

### Database Schema
- `restaurants` - Restaurant locations
- `app_configs` - Camera configurations per restaurant
- `hourly_footfall` - Footfall data aggregated hourly
- `petpooja_webhook_events` - Sales data from POS
- `roi_configs` - ROI polygons and line positions

## 🎨 Features

### 1. People Counter
- Bi-directional counting (IN/OUT)
- Adjustable counting line (30-70%)
- Cooldown system to prevent double-counting
- ROI editor: `/roi_editor_people/<camera_id>`

### 2. Queue Monitor
- Dual ROI (queue area + counter area)
- Queue length tracking
- Waiting time estimation
- ROI editor: `/roi_editor/<camera_id>`

### 3. Kitchen Compliance
- PPE detection (apron, cap, gloves)
- Security vest detection
- Compliance scoring
- Alert system for violations

### 4. Footfall to Sales Conversion
- Real-time conversion rate
- Hourly aggregations
- Average order value tracking
- API endpoints for external integration

## 📡 API Endpoints

### Analytics APIs (FastAPI - Port 8000)

**Daily Sales**
```bash
GET /analytics/sales-daily?start_date=2025-01-01&end_date=2025-01-31
```

**Hourly Sales**
```bash
GET /analytics/sales-hourly?start_date=2025-01-01&end_date=2025-01-31
```

**Conversion Analytics**
```bash
GET /analytics/conversion?restaurant_id=1
```

### Configuration APIs (Flask - Port 5001)

**Add Restaurant**
```bash
POST /api/restaurants
Content-Type: application/json

{
  "name": "Tea Toast",
  "location": "Brigade Road",
  "display_name": "Tea Toast - Brigade Road"
}
```

**Add Camera**
```bash
POST /api/restaurants/{restaurant_id}/cameras
Content-Type: application/json

{
  "channel_id": "cam_12345",
  "camera_name": "Main Entrance",
  "rtsp_url": "rtsp://...",
  "app_name": "PeopleCounter"
}
```

## 🔧 Configuration

### Database Connection
```python
# .env file
DATABASE_URL=postgresql://postgres:password@localhost:5432/sakshi
```

### RTSP Streams
```python
# Fallback configuration in rtsp_links.txt
cam_channel_id | Camera Name | rtsp://user:pass@ip:port/stream | AppName
```

### ROI Configuration
- **People Counter**: Line position (0.0-1.0)
- **Queue Monitor**: Two polygons (main queue + counter area)
- Web editors available at:
  - `/roi_editor/<camera_id>` - Queue Monitor
  - `/roi_editor_people/<camera_id>` - People Counter

## 🌐 Multi-Restaurant Setup

### Phase 1: Database Schema ✅
- Restaurant table
- Camera configurations
- ROI storage

### Phase 2: Backend Integration ✅
- API endpoints
- Restaurant filtering
- Data aggregation

### Phase 3: Frontend Dropdown ✅
- Restaurant selector
- Dynamic filtering
- URL-based state

## 🔒 Security

- PostgreSQL user authentication
- RTSP credentials encrypted in database
- API authentication tokens
- Input validation and sanitization
- CORS configuration for APIs

## 📈 Performance

- Optimized YOLO inference (640x640)
- Batch processing disabled for real-time
- Frame skipping for resource management
- Database connection pooling
- Efficient JSONB queries

## 🐛 Troubleshooting

### Common Issues

**Database Connection Failed**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Restart service
sudo systemctl restart postgresql
```

**Camera Feed Not Loading**
```bash
# Test RTSP stream
ffplay rtsp://user:pass@ip:port/stream

# Check logs
journalctl -u sakshi-ai -f
```

**ROI Not Saving**
```bash
# Verify database connection
python3 -c "from sqlalchemy import create_engine; engine = create_engine('postgresql://...'); print(engine.connect())"
```

## 📚 Documentation

Complete documentation available in `DOCUMENTATION.md` covering:
- Detailed installation steps
- API reference with examples
- Deployment guides (Gunicorn, Nginx, systemd)
- Training custom models on Google Colab
- Troubleshooting guide
- Development guidelines

## 🎓 Training Models

Train custom YOLO models using Google Colab (free GPU):

```python
# See DOCUMENTATION.md Section 19: Training & Model Management
# Full Colab notebook with step-by-step instructions
```

## 📦 Project Structure

```
sakshi/
├── edit-004.py              # Main Flask application
├── fastapi_app.py           # FastAPI server
├── requirements.txt         # Python dependencies
├── DOCUMENTATION.md         # Complete documentation
├── templates/               # HTML templates
│   ├── dashboard.html
│   ├── roi_editor.html
│   └── roi_editor_people.html
├── processors/              # Video processors
│   ├── people_counter.py
│   ├── queue_monitor.py
│   └── kitchen_compliance_monitor.py
├── *.pt                     # YOLO model files
└── phase1_create_schema.sql # Database schema

```

## 🤝 Contributing

This is a proprietary system. For feature requests or bug reports, contact the development team.

## 📝 License

Copyright © 2025 Sakshi AI. All rights reserved.

## 🔗 Links

- **Dashboard**: http://localhost:5001/dashboard
- **API Docs**: http://localhost:8000/docs
- **ROI Editors**: 
  - http://localhost:5001/roi_editor/<camera_id>
  - http://localhost:5001/roi_editor_people/<camera_id>

## 📞 Support

For technical support:
- Email: support@sakshi-ai.com
- Documentation: See DOCUMENTATION.md
- GitHub Issues: (for authorized users only)

---

**Version**: 2.0.0 (Multi-Restaurant Release)  
**Last Updated**: December 2025  
**Status**: Production Ready ✅
