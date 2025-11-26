# ROI Configuration Server Deployment Guide

## Problem
QueueMonitor ROI works on local machine but not on server after git pull.

## Root Cause
ROI configurations are stored in **PostgreSQL database**, not in code files. Git only syncs code changes, not database data.

## Solution: Update Server Database

### Step 1: Pull Latest Code on Server
```bash
ssh user@server
cd /path/to/sakshi
git pull origin kitchen-update
```

### Step 2: Update ROI Database on Server
```bash
# Option A: Run the pre-made Python script
python3 update_roi_on_server.py

# Option B: Use psql directly
psql -U postgres -d sakshi -c "
INSERT INTO roi_configs (channel_id, app_name, roi_points) 
VALUES ('cam_f822b0bf4e', 'QueueMonitor', '{\"main\": [[0.5549999952316285, 0.5744444105360244], [0.4456249952316284, 0.5272221883138021], [0.3081249952316284, 0.3105555216471354], [0.08624999523162842, 0.4272221883138021], [0.19249999523162842, 0.7938888549804688]], \"secondary\": [[0.5924999952316284, 0.5355555216471354], [0.49874999523162844, 0.502222188313802], [0.3487499952316284, 0.31333329942491317], [0.38156249523162844, 0.3105555216471354], [0.3940624952316284, 0.2883332994249132], [0.5003124952316285, 0.26888885498046877], [0.6721874952316285, 0.4633332994249132]]}')
ON CONFLICT (channel_id, app_name) 
DO UPDATE SET roi_points = EXCLUDED.roi_points;
"

# For PeopleCounter line position
psql -U postgres -d sakshi -c "
INSERT INTO roi_configs (channel_id, app_name, roi_points) 
VALUES ('cam_3df702bb28', 'PeopleCounter', '{\"line_position\": 0.38}')
ON CONFLICT (channel_id, app_name) 
DO UPDATE SET roi_points = EXCLUDED.roi_points;
"
```

### Step 3: Restart Application on Server
```bash
# If using systemd service
sudo systemctl restart sakshi-ai
sudo journalctl -u sakshi-ai -f

# If running manually
# Stop the running process first, then:
python3 edit-004.py
```

### Step 4: Verify on Server
```bash
# Check database has correct ROI
python3 -c "
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json

DATABASE_URL = 'postgresql://postgres:Tneural01@127.0.0.1:5432/sakshi'
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

with SessionLocal() as db:
    result = db.execute(text('SELECT channel_id, app_name FROM roi_configs'))
    for row in result:
        print(f'✅ {row[1]}: {row[0]}')
"

# Check logs for ROI loading
journalctl -u sakshi-ai | grep -i "ROI"
```

## What Changed in Recent Commits

### Commit b16a3af: QueueMonitor ROI Editor
- ✅ Created `templates/roi_editor.html` - Web interface to draw ROIs
- ✅ Modified `edit-004.py` - Added ROI loading from database
- ✅ Made ROI boxes transparent (invisible on video)
- ✅ Person bounding boxes still visible (blue=queue, green=counter)

### Commit 6ac29f2: PeopleCounter Line Position
- ✅ Changed line position from 45% to 38%
- ✅ Created `templates/roi_editor_people.html` - Slider to adjust line
- ✅ Made counting line transparent (invisible on video)
- ✅ Person boxes also transparent

### Commit 0a9a211: Server Deployment Scripts
- ✅ Added `update_roi_on_server.py` - Run this on server
- ✅ Added `export_roi_to_server.py` - Generate export commands
- ✅ Added `roi_server_update.txt` - Pre-generated SQL

## Files to Deploy on Server

### Must Deploy (via git pull):
1. `edit-004.py` - Updated processor code
2. `templates/roi_editor.html` - QueueMonitor ROI editor
3. `templates/roi_editor_people.html` - PeopleCounter line editor
4. `update_roi_on_server.py` - Database update script

### Optional (helper scripts):
- `export_roi_to_server.py` - Generate new exports
- `check_people_roi.py` - Verify database state
- `update_line_position.py` - Update line position

## Current ROI Configuration

### QueueMonitor (cam_f822b0bf4e)
- **Main ROI (Queue Area)**: 5-point polygon
- **Secondary ROI (Counter Area)**: 7-point polygon
- **Visualization**: Transparent (hidden), only person boxes visible

### PeopleCounter (cam_3df702bb28)
- **Line Position**: 38% from left (62% on right side)
- **Visualization**: Transparent (hidden)
- **Direction**: Left→Right = IN, Right→Left = OUT

## Troubleshooting

### Issue: ROI still wrong after git pull
**Cause**: Database not updated
**Fix**: Run `python3 update_roi_on_server.py` on server

### Issue: No person detections visible
**Cause**: Code working correctly (transparent mode enabled)
**Fix**: This is intentional - boxes are now hidden for clean UX

### Issue: Counting not working
**Cause**: ROI not loaded from database
**Fix**: Check logs for "✅ Loaded ROI from database" message

### Issue: Changes not reflected after restart
**Cause**: Using old code version
**Fix**: `git pull origin kitchen-update` and restart service

## Quick Deployment Checklist

On **SERVER**, run these commands:
```bash
# 1. Pull latest code
cd /path/to/sakshi
git pull origin kitchen-update

# 2. Update database
python3 update_roi_on_server.py

# 3. Restart service
sudo systemctl restart sakshi-ai

# 4. Verify
journalctl -u sakshi-ai -n 50 | grep -E "(ROI|line position)"
```

Expected output:
```
✅ Loaded ROI from database for Queue Monitor (5 main points, 7 secondary points)
✅ Loaded counting line position: 38% for Main Entrance
```
