# Idle People Violation - Implementation Guide for LLM

## Objective
Implement an idle person detection system that monitors specific areas (ROI) and alerts when people remain stationary for a configurable duration.

---

## Prerequisites
- Flask application with YOLO-based detection
- PostgreSQL database
- Existing camera infrastructure
- Python 3.10+

---

## Step 1: Install Dependencies

Add to `requirements.txt`:
```
shapely==2.0.2
```

Install:
```bash
pip install shapely==2.0.2
```

---

## Step 2: Create Processor File

Create `idle_people_violation.py` with:
- YOLO11n model for person detection (class 0)
- Frame-based tracking with 15-frame threshold
- ROI polygon filtering using Shapely
- Database integration for violations
- Telegram alerts
- Visual overlay (green boxes for tracking, red for violations)

**Key Features:**
- Load ROI from database on startup
- Check if person center point is within ROI polygon
- Track person IDs and increment frame counter
- Trigger violation when counter reaches 15 frames
- Draw yellow polygon overlay on video feed
- 60-second cooldown between alerts for same person

**Database Table:** `idle_people_violations`
- Columns: id, channel_id, channel_name, timestamp, person_id, frame_count, details, media_path

---

## Step 3: Create ROI Editor Interface

Create `templates/roi_editor_idle_people.html`:
- Canvas-based polygon drawing
- Click to add points (minimum 3 required)
- Load existing ROI from database
- Save/Clear/Undo controls
- Real-time video frame display
- Yellow polygon visualization
- Auto-refresh every 5 seconds (preserves drawn points)
- Flag to prevent auto-reload after manual clear

**API Endpoints Used:**
- GET `/api/get_roi?channel_id={id}&app_name=IdlePeopleViolation`
- POST `/api/set_roi` with JSON: `{channel_id, app_name, roi_points: {points: [[x,y],...]}}`

---

## Step 4: Create Monitoring UI

Create `templates/idle_people_violation.html`:
- Live video feed with ROI overlay
- Info cards (threshold, cameras, status, violations)
- Detection legend (green/red boxes)
- Violation history grid with date filtering
- Lightbox for image viewing
- Restaurant filtering support

**Features:**
- Load history via `/history/IdlePeopleViolation?restaurant_id={id}`
- Filter by date range (last 7 days default)
- Pagination support
- Display violation images from database

**Video Display:**
- Max width: 800px
- Max height: 600px
- Centered with object-fit: contain

---

## Step 5: Integrate with Main Application

Update `edit-004.py`:

**1. Import:**
```python
from idle_people_violation import IdlePeopleViolationProcessor
```

**2. Add to APP_TASKS_CONFIG (around line 104):**
```python
'IdlePeopleViolation': {
    'model_path': 'models/yolo11n.pt',
    'processor_class': IdlePeopleViolationProcessor,
    'confidence': 0.3
}
```

**3. Initialize table (around line 2555):**
```python
IdlePeopleViolationProcessor.initialize_tables(engine)
logging.info("Idle People Violation tables initialized")
```

**4. Add routes:**

```python
@app.route('/idle-people-violation')
@login_required
def idle_people_violation():
    restaurant_id = request.args.get('restaurant_id', type=int)
    app_configs = get_app_configs()
    idle_channels = app_configs.get('IdlePeopleViolation', {}).get('channels', [])
    if restaurant_id:
        idle_channels = [ch for ch in idle_channels if ch.get('restaurant_id') == restaurant_id]
    return render_template('idle_people_violation.html', channels=idle_channels, restaurant_id=restaurant_id)

@app.route('/roi_editor_idle_people')
@login_required
def roi_editor_idle_people():
    channel_id = request.args.get('channel_id')
    if not channel_id:
        app_configs = get_app_configs()
        idle_channels = app_configs.get('IdlePeopleViolation', {}).get('channels', [])
        if idle_channels:
            channel_id = idle_channels[0]['id'] if isinstance(idle_channels[0], dict) else idle_channels[0]
        else:
            return "No channels configured", 404
    return render_template('roi_editor_idle_people.html', channel_id=channel_id)
```

**5. Add video feed route (around line 2560):**
```python
@app.route('/video_feed/IdlePeopleViolation/<channel_id>')
def idle_people_video_feed(channel_id):
    processor = active_processors.get(('IdlePeopleViolation', channel_id))
    if not processor:
        return Response("Processor not found", status=404)
    def generate():
        while True:
            frame = processor.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
```

**6. Update history endpoint to filter by restaurant:**
```python
@app.route('/history/<app_name>')
@login_required
def get_history(app_name):
    # Add restaurant_id parameter
    restaurant_id = request.args.get('restaurant_id', type=int)
    
    # Join with cameras table to filter by restaurant
    query = db.query(Detection).join(Camera, Detection.channel_id == Camera.channel_id).filter(Detection.app_name == app_name)
    
    if restaurant_id:
        query = query.filter(Camera.restaurant_id == restaurant_id)
    
    # ... rest of existing code
```

---

## Step 6: Update Dashboard Navigation

Update `templates/dashboard.html`:

Add navigation button:
```html
<a href="/idle-people-violation{% if selected_restaurant %}?restaurant_id={{ selected_restaurant.id }}{% endif %}" class="nav-link-external">
  <div class="icon">🚶</div><div>Idle People Violation</div>
</a>
```

Add CSS for external links:
```css
.nav-link-external {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    border-radius: 12px;
    color: #d7d8da;
    text-decoration: none;
    cursor: pointer;
    transition: all 0.2s;
}
.nav-link-external:hover {
    background: rgba(255,255,255,0.05);
}
```

---

## Step 7: Database Setup

**1. Add camera to database:**

```sql
-- Generate channel_id using MD5 hash of RTSP URL
INSERT INTO cameras (channel_id, channel_name, rtsp_url, restaurant_id, is_active)
VALUES (
    'cam_bc9c1f31',  -- MD5 hash of RTSP URL
    'Front Office',
    'rtsp://admin:cctv%231234@182.65.205.121:554/cam/realmonitor?channel=5&subtype=1',
    2,  -- Restaurant ID
    true
);

-- Link camera to app
INSERT INTO camera_apps (camera_id, app_name)
SELECT id, 'IdlePeopleViolation'
FROM cameras
WHERE channel_id = 'cam_bc9c1f31';
```

**2. ROI Configuration:**

ROI can be configured in two ways:

**Option A: Use ROI Editor (Recommended)**
- Access `http://localhost:5001/roi_editor_idle_people`
- Draw polygon by clicking on video frame
- Click "Save ROI" button
- ROI automatically saved to database

**Option B: Manual Database Insert**

If you need to add ROI points directly to the database, use this tested configuration:

```sql
-- Insert ROI points for a camera (Example from Tea Toast - Main store)
-- These coordinates define a 6-point polygon for Front Office area monitoring
-- Replace channel_id with your camera's ID and adjust points as needed
INSERT INTO roi_configs (channel_id, app_name, roi_points)
VALUES (
    'cam_bc9c1f31',  -- Replace with your camera's channel_id
    'IdlePeopleViolation',
    '{"points": [[51, 155], [154, 77], [196, 38], [120, 0], [23, 85], [20, 168]]}'::jsonb
)
ON CONFLICT (channel_id, app_name) 
DO UPDATE SET roi_points = EXCLUDED.roi_points;
```

**Verified Working ROI Points (Tea Toast - Front Office):**
```json
{
  "points": [
    [51, 155],
    [154, 77],
    [196, 38],
    [120, 0],
    [23, 85],
    [20, 168]
  ]
}
```

**Verify ROI in database:**
```sql
SELECT channel_id, app_name, roi_points 
FROM roi_configs 
WHERE app_name = 'IdlePeopleViolation';
```

**Important:** After adding/updating ROI, restart the application for processor to load new ROI.

---

## Step 8: Helper Scripts (Optional)

Create `scripts/add_idle_people_camera.py`:
- Auto-generate channel_id from RTSP URL using MD5 hash
- Insert camera and camera_apps records
- Verify configuration

Create `scripts/test_roi_idle_people.py`:
- Check ROI configuration in database
- Verify camera setup
- Test Shapely installation
- Validate polygon creation

Create `scripts/roi_quick_reference.sh`:
- Display quick setup guide
- Show URLs and commands
- Run test script automatically

---

## Step 9: Configuration Details

**ROI Configuration:**
- Stored in `roi_configs` table
- Format: `{"points": [[x1,y1], [x2,y2], ...]}`
- Loaded on processor startup
- Uses Shapely for point-in-polygon checks

**Detection Logic:**
1. YOLO detects person → Get bounding box
2. Calculate center point: `(x1+x2)/2, (y1+y2)/2`
3. Check if center is in ROI polygon → If NO, skip
4. If YES → Increment frame counter for that person ID
5. If counter >= 15 → Trigger violation (red box, alert, save)
6. Draw yellow polygon overlay on all frames

**Frame Threshold:**
- Default: 15 frames
- Configurable via `IDLE_FRAME_THRESHOLD` in idle_people_violation.py
- Adjustable based on camera FPS and requirements

---

## Step 10: Testing Checklist

1. ✅ Application starts without errors
2. ✅ Processor logs: "✅ Loaded ROI for {channel}: X points" OR "No ROI configured"
3. ✅ Access `/roi_editor_idle_people` - video frame loads
4. ✅ Draw polygon (3+ points), click Save - success message appears
5. ✅ Database check: `SELECT * FROM roi_configs WHERE app_name='IdlePeopleViolation'`
6. ✅ Restart application - ROI loads automatically
7. ✅ Access `/idle-people-violation` - yellow polygon overlay visible
8. ✅ Person enters ROI → Green box appears
9. ✅ Person stays idle → Counter increments → Red box at 15 frames
10. ✅ Violation saved to database with screenshot
11. ✅ History page shows violations with images
12. ✅ Restaurant filtering works correctly

---

## URLs Reference

- **ROI Editor**: `http://localhost:5001/roi_editor_idle_people`
- **Live Monitor**: `http://localhost:5001/idle-people-violation`
- **Dashboard**: `http://localhost:5001/dashboard`
- **With Restaurant Filter**: `http://localhost:5001/idle-people-violation?restaurant_id=2`

---

## Key Configuration Values

```python
# In idle_people_violation.py
MODEL_PATH = 'models/yolo11n.pt'
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.3
IDLE_FRAME_THRESHOLD = 15  # Frames before violation
FRAME_SKIP_RATE = 2  # Process every 2nd frame
ALERT_COOLDOWN_SECONDS = 60
```

---

## Troubleshooting

**Video frame not loading in ROI editor:**
- Check processor is running: Look for "✅ Connected to {channel}" in logs
- Verify channel_id is correct in URL
- Check video feed endpoint: `/video_feed/IdlePeopleViolation/cam_bc9c1f31`

**ROI not appearing on live feed:**
- Restart application after saving ROI
- Check logs for "✅ Loaded ROI" message
- Verify roi_points in database is valid JSON

**Detection not working:**
- Verify person is inside ROI polygon (check yellow overlay)
- Check confidence threshold (default 0.3)
- Ensure YOLO model exists at specified path

**History not showing:**
- Check detections table: `SELECT COUNT(*) FROM detections WHERE app_name='IdlePeopleViolation'`
- Verify images exist in `static/detections/` folder
- Check restaurant filtering is applied correctly

---

## Expected Behavior

1. **Person passing through** (moving) → Green box, counter increments but resets when they leave
2. **Person stops in ROI** → Green box, counter keeps incrementing
3. **Person idle 15+ frames** → Red box, violation triggered, screenshot saved, Telegram alert
4. **Person outside ROI** → Completely ignored, no tracking

---

## Database Schema

**roi_configs:**
```sql
channel_id VARCHAR
app_name VARCHAR
roi_points JSONB
UNIQUE(channel_id, app_name)
```

**idle_people_violations:**
```sql
id SERIAL PRIMARY KEY
channel_id VARCHAR
channel_name VARCHAR
timestamp TIMESTAMP
person_id INTEGER
frame_count INTEGER
details VARCHAR
media_path VARCHAR UNIQUE
```

**detections:** (existing table, used for history)
```sql
id SERIAL PRIMARY KEY
app_name VARCHAR
channel_id VARCHAR
timestamp TIMESTAMP
message VARCHAR
media_path VARCHAR
```

---

## Success Indicators

✅ Processor initializes and loads ROI  
✅ Yellow polygon overlay visible on live feed  
✅ Green boxes track people moving in ROI  
✅ Red boxes appear when person is idle 15+ frames  
✅ Violations saved with screenshots  
✅ History page shows violations filtered by restaurant  
✅ ROI editor allows drawing and saving polygons  
✅ Restaurant switching filters channels and history correctly  

---

## Performance Notes

- Uses CPU-only mode for stability
- Frame skip rate of 2 for efficiency
- Lightweight Shapely point-in-polygon checks
- Thread pool for async alerts
- 60-second alert cooldown prevents spam

---

## Customization Options

1. **Change idle threshold:** Edit `IDLE_FRAME_THRESHOLD` in idle_people_violation.py
2. **Adjust sensitivity:** Modify `CONFIDENCE_THRESHOLD` (lower = more detections)
3. **Multiple ROI zones:** Store multiple polygons and check each
4. **Time-based instead of frame-based:** Replace frame counter with timestamp tracking
5. **Different visual colors:** Modify polygon color in `_draw_roi()` method

---

## End Result

A fully functional idle person detection system with:
- Visual ROI editor for defining monitoring zones
- Real-time person tracking with visual indicators
- Automated violation detection and alerting
- Restaurant-filtered detection history with images
- Seamless dashboard integration
- Easy configuration via web interface

Copy this entire guide to an LLM and ask: "Please implement this idle people violation monitoring system following these steps."
