# Deployment Guide

## ✅ Pre-Deployment Checklist

Your code is **READY FOR DEPLOYMENT** with the following verified:

### 1. Path Configuration ✅
- ✅ All model paths use relative paths: `models/*.pt`
- ✅ No hardcoded `/home/athul` paths
- ✅ Database URL uses `127.0.0.1` (will be overridden by server .env)
- ✅ Config files organized in proper folders

### 2. Code Structure ✅
```
sakshi/
├── edit-004.py                    # Main Flask app
├── fastapi_app.py                 # FastAPI server
├── kitchen_compliance_monitor.py  # Processor
├── queue_monitor.py               # Processor
├── occupancy_monitor_processor.py # Processor
├── shutter_monitor_processor006.py# Processor
├── main.py                        # Legacy
├── requirements.txt               # Dependencies
├── rtsp_links.txt                 # Camera config
├── start_app.sh                   # Startup script
├── .env                           # Environment config
├── models/                        # 12 YOLO models
├── docs/                          # Documentation
├── config/                        # Gunicorn, WSGI
├── scripts/                       # Migration & setup
├── templates/                     # 7 HTML templates
└── static/                        # CSS, JS, images
```

### 3. Environment Variables Required on Server
Edit `.env` on server:
```bash
# Database (use server's PostgreSQL)
DATABASE_URL=postgresql://postgres:PASSWORD@localhost:5432/sakshi

# PetPooja API
PETPOOJA_API_TOKEN="Z4N8T2W9L3H6Q1P"
```

### 4. Dependencies ✅
- requirements.txt updated with all current versions
- Virtual environment supported
- No system-specific dependencies

## 📦 Deployment Steps

### On Server (13.202.92.108):

```bash
# 1. Navigate to project directory
cd /var/www/sakshi  # or your server path

# 2. Pull latest changes
git pull origin Tea-toast-new-v1

# 3. Update .env file
nano .env
# Update DATABASE_URL to use server's PostgreSQL

# 4. Install/update dependencies (if needed)
source venv/bin/activate
pip install -r requirements.txt

# 5. Restart services
sudo systemctl restart sakshi-ai       # Flask dashboard
sudo systemctl restart sakshi-fastapi  # FastAPI server

# 6. Verify services
sudo systemctl status sakshi-ai
sudo systemctl status sakshi-fastapi

# 7. Check logs
sudo journalctl -u sakshi-ai -f
sudo journalctl -u sakshi-fastapi -f
```

### Service Files Location
After git pull, update service files if needed:
- `scripts/sakshi-ai.service` - Flask dashboard service
- `scripts/petpooja-api.service` - FastAPI service

Copy to systemd:
```bash
sudo cp scripts/sakshi-ai.service /etc/systemd/system/
sudo cp scripts/petpooja-api.service /etc/systemd/system/sakshi-fastapi.service
sudo systemctl daemon-reload
```

## 🧪 Post-Deployment Testing

### 1. Test Dashboard
```bash
curl http://localhost:5001/
# Should return login page HTML
```

### 2. Test FastAPI
```bash
curl "http://localhost:8000/analytics/sales-hourly?start_date=2025-12-15&end_date=2025-12-22&token=Z4N8T2W9L3H6Q1P"
# Should return JSON with hourly sales data
```

### 3. Test Camera Feeds
- Login to dashboard at: http://13.202.92.108:5001
- Verify all 5 camera feeds are working
- Check detection boxes appear on feeds

### 4. Test Conversion Analytics
- Navigate to: http://13.202.92.108:5001/conversion_analytics
- Verify order counts match sales analytics
- Should show same count in both graphs

## 🔍 Troubleshooting

### Issue: Models not loading
**Cause:** Model files not found
**Fix:** 
```bash
# Verify models exist
ls -la models/*.pt
# Should show 12 .pt files
```

### Issue: Import errors
**Cause:** Processor files not found
**Fix:**
```bash
# Verify processors in root
ls -la *_monitor*.py *_processor*.py
# Should show 4 processor files
```

### Issue: Database connection failed
**Cause:** Wrong DATABASE_URL in .env
**Fix:**
```bash
# Check .env
cat .env
# Update with correct server credentials
nano .env
```

### Issue: FastAPI returns 404
**Cause:** Service not running or wrong port
**Fix:**
```bash
# Check if running
sudo systemctl status sakshi-fastapi

# Check port
sudo netstat -tlnp | grep 8000

# Restart if needed
sudo systemctl restart sakshi-fastapi
```

## 📊 Expected Results After Deployment

### Dashboard (Port 5001)
- ✅ 5 camera feeds with real-time detection
- ✅ Conversion analytics with matching counts
- ✅ Restaurant dropdown working
- ✅ ROI editors accessible

### FastAPI (Port 8000)
- ✅ `/analytics/sales-daily` - Daily aggregation
- ✅ `/analytics/sales-hourly` - Hourly breakdown
- ✅ `/analytics/payment-modes` - Payment stats
- ✅ All endpoints return deduplicated counts

### Database
- ✅ `hourly_footfall` table updated every hour
- ✅ `petpooja_webhook_events` receiving orders
- ✅ `roi_configs` storing ROI settings

## 🎯 Success Criteria

After deployment, verify:
1. ✅ All camera feeds load within 5 seconds
2. ✅ Conversion analytics shows matching order counts
3. ✅ FastAPI `/analytics/sales-hourly` returns 200 OK
4. ✅ No errors in `journalctl -u sakshi-ai -n 50`
5. ✅ Restaurant dropdown shows all locations

## 📝 Commits Being Deployed

```
03dd63e - Refactor: Organize project structure into folders
ae708e1 - Docs: Consolidate 21 MD files into single DOCUMENTATION
b853942 - Clean: Remove 13K+ training datasets from git tracking  
80a1aee - Fix: Align conversion analytics order counts
```

## ⚠️ Important Notes

1. **Virtual Environment:** Must activate venv before running
   ```bash
   source venv/bin/activate
   python3 edit-004.py
   ```

2. **File Permissions:** Ensure start_app.sh is executable
   ```bash
   chmod +x start_app.sh
   ```

3. **Model Files:** All 12 .pt files must be in `models/` folder

4. **Database:** Server must have PostgreSQL 14+ running

5. **Ports:** Ensure 5001 (Flask) and 8000 (FastAPI) are open

---

**Deployment Status:** ✅ READY
**Last Updated:** December 22, 2025
**Branch:** Tea-toast-new-v1
**Commits Ahead:** 3
