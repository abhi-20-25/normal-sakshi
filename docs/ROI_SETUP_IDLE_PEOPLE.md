# ROI Setup for Idle People Violation Monitor

## Overview
This document describes the Region of Interest (ROI) functionality added to the Idle People Violation monitoring system. ROI allows you to define specific areas within the camera view where idle person detection should occur, ignoring people outside the designated zone.

---

## What is ROI?

**ROI (Region of Interest)** is a user-defined polygon area on the video frame where monitoring takes place. Only people detected within this polygon will be tracked for idle violations.

### Benefits:
- **Reduce false positives**: Ignore areas like hallways, doorways, or non-critical zones
- **Focus monitoring**: Target specific areas like waiting zones or service counters
- **Improve performance**: Process fewer detections by filtering out irrelevant areas
- **Flexible configuration**: Easily adjust monitoring zones without code changes

---

## Implementation Summary

### 1. Files Created

#### `/templates/roi_editor_idle_people.html`
- **Purpose**: Web-based ROI editor interface
- **Features**:
  - Canvas-based polygon drawing
  - Real-time video frame display
  - Point-by-point polygon creation
  - Undo/Clear/Save functionality
  - Visual feedback with yellow polygon overlay
  - Loads existing ROI from database
  - Auto-refreshes video frame every 5 seconds

#### `/scripts/test_roi_idle_people.py`
- **Purpose**: Verify ROI setup and configuration
- **Checks**:
  - Database ROI configuration
  - Camera configuration
  - Shapely library installation
  - Polygon creation test
- **Usage**: `python scripts/test_roi_idle_people.py`

### 2. Files Modified

#### `/idle_people_violation.py`
**Added:**
- Import: `from shapely.geometry import Point, Polygon`
- Method `_load_roi()`: Loads ROI polygon from database on processor startup
- Method `_is_in_roi(x, y)`: Checks if a point (person center) is within ROI
- Method `_draw_roi(frame)`: Draws yellow polygon overlay on video frame
- ROI filtering in detection loop: Skips detections outside ROI

**Changes in `__init__`:**
```python
# ROI (Region of Interest) configuration
self.roi_polygon = None
self._load_roi()
```

**Changes in `run()` loop:**
```python
# Calculate center point of bounding box
center_x = (x1 + x2) // 2
center_y = (y1 + y2) // 2

# Check if person is within ROI
if not self._is_in_roi(center_x, center_y):
    continue  # Skip people outside ROI
```

#### `/edit-004.py`
**Added route:**
```python
@app.route('/roi_editor_idle_people')
@login_required
def roi_editor_idle_people():
    """ROI Editor for Idle People Violation"""
    channel_id = request.args.get('channel_id')
    if not channel_id:
        # Default to first available channel
        app_configs = get_app_configs()
        idle_channels = app_configs.get('IdlePeopleViolation', {}).get('channels', [])
        if idle_channels:
            channel_id = idle_channels[0]
        else:
            return "No channels configured for Idle People Violation", 404
    
    return render_template('roi_editor_idle_people.html', channel_id=channel_id)
```

#### `/templates/idle_people_violation.html`
**Added button in header:**
```html
<a href="/roi_editor_idle_people?channel_id={{ channels[0].id }}" 
   class="back-btn" 
   style="background: linear-gradient(90deg, #1640ff, #2b3bff); border: none;">
  🎯 Edit ROI
</a>
```

#### `/requirements.txt`
**Added dependency:**
```
shapely==2.0.2
```

---

## Database Schema

### ROI Storage
ROI configurations are stored in the existing `roi_configs` table:

```sql
CREATE TABLE roi_configs (
    id SERIAL PRIMARY KEY,
    channel_id VARCHAR NOT NULL,
    app_name VARCHAR NOT NULL,
    roi_points JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(channel_id, app_name)
);
```

### Example ROI Data
```json
{
  "points": [
    [100, 150],
    [500, 150],
    [500, 400],
    [100, 400]
  ]
}
```

### Query ROI
```sql
SELECT roi_points 
FROM roi_configs 
WHERE channel_id = 'cam_bc9c1f31' 
  AND app_name = 'IdlePeopleViolation';
```

---

## How to Use ROI Editor

### Step 1: Access ROI Editor
1. Navigate to Idle People Violation page: `http://localhost:5003/idle-people-violation`
2. Click the **"🎯 Edit ROI"** button in the top-right corner
3. Or directly visit: `http://localhost:5003/roi_editor_idle_people?channel_id=cam_bc9c1f31`

### Step 2: Draw ROI Polygon
1. **Click on the video frame** to add points (minimum 3 points required)
2. Each click adds a point to the polygon
3. The polygon automatically closes after 3+ points
4. Points are numbered sequentially (1, 2, 3, ...)
5. **Yellow polygon** shows the monitoring zone
6. **Point list** displays coordinates below the frame

### Step 3: Adjust Polygon
- **Undo Last Point**: Remove the most recently added point
- **Clear ROI**: Remove all points and start over
- **Re-click**: Add more points to refine the shape

### Step 4: Save ROI
1. Click **"💾 Save ROI"** button
2. Success message confirms save
3. ROI is immediately active (processor reloads on next startup)

### Step 5: Verify ROI
1. Go back to Idle People Violation page
2. **Yellow polygon overlay** should be visible on the video feed
3. Only people inside the polygon will be tracked
4. Run test script: `python scripts/test_roi_idle_people.py`

---

## Technical Details

### Polygon Point Detection
The system uses the **Shapely** library for geometric operations:

```python
from shapely.geometry import Point, Polygon

# Create polygon from ROI points
polygon = Polygon([(100, 150), (500, 150), (500, 400), (100, 400)])

# Check if person center is inside
person_center = Point(300, 250)
is_inside = polygon.contains(person_center)  # Returns True
```

### ROI Loading Process
1. **Processor Initialization**: `_load_roi()` called in `__init__`
2. **Database Query**: Fetch ROI from `roi_configs` table
3. **Polygon Creation**: Convert JSON points to Shapely Polygon
4. **Validation**: Ensure minimum 3 points
5. **Logging**: Confirm ROI loaded or log "monitoring entire frame"

### Detection Filtering
```python
# Calculate person bounding box center
center_x = (x1 + x2) // 2
center_y = (y1 + y2) // 2

# Skip detection if outside ROI
if not self._is_in_roi(center_x, center_y):
    continue  # Person outside monitoring zone
```

### Visual Overlay
```python
def _draw_roi(self, frame):
    """Draw ROI polygon on frame"""
    if self.roi_polygon is None:
        return
    
    points = np.array(self.roi_polygon.exterior.coords, dtype=np.int32)
    
    # Draw polygon outline (yellow)
    cv2.polylines(frame, [points], isClosed=True, color=(255, 255, 0), thickness=2)
    
    # Fill with semi-transparent overlay (10% opacity)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [points], color=(255, 255, 0))
    cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
```

---

## API Endpoints

### Save ROI
**POST** `/api/set_roi`

**Request Body:**
```json
{
  "channel_id": "cam_bc9c1f31",
  "app_name": "IdlePeopleViolation",
  "roi_points": {
    "points": [
      [100, 150],
      [500, 150],
      [500, 400],
      [100, 400]
    ]
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "ROI saved successfully"
}
```

### Get ROI
**GET** `/api/get_roi?channel_id=cam_bc9c1f31&app_name=IdlePeopleViolation`

**Response:**
```json
{
  "channel_id": "cam_bc9c1f31",
  "app_name": "IdlePeopleViolation",
  "roi_points": {
    "points": [
      [100, 150],
      [500, 150],
      [500, 400],
      [100, 400]
    ]
  }
}
```

---

## Troubleshooting

### Issue: ROI Not Displaying
**Solution:**
1. Check browser console for errors
2. Verify video feed is working: `/video_feed/IdlePeopleViolation/cam_bc9c1f31`
3. Ensure ROI saved successfully (check database)
4. Restart the application to reload ROI

### Issue: People Outside ROI Still Detected
**Solution:**
1. Verify ROI loaded in logs: `✅ Loaded ROI for Front Office: X points`
2. Check polygon contains expected area (run test script)
3. Ensure processor restarted after ROI save
4. Check Shapely installation: `pip show shapely`

### Issue: Can't Save ROI
**Solution:**
1. Ensure minimum 3 points added
2. Check `/api/set_roi` endpoint is accessible
3. Verify database connection
4. Check browser network tab for API errors

### Issue: Polygon Not Closing
**Solution:**
- Add at least 3 points (polygon auto-closes after 3rd point)
- Check JavaScript console for errors
- Refresh page and try again

---

## Configuration Options

### Customize ROI Color
Edit `idle_people_violation.py`, method `_draw_roi()`:

```python
# Change outline color (BGR format)
cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 255), thickness=2)  # Cyan

# Change fill color
cv2.fillPoly(overlay, [points], color=(0, 255, 255))  # Cyan fill
```

### Adjust Overlay Opacity
```python
# Increase opacity (0.3 = 30% visible)
cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

# Decrease opacity (0.05 = 5% visible)
cv2.addWeighted(overlay, 0.05, frame, 0.95, 0, frame)
```

### No ROI (Monitor Entire Frame)
- Don't save any ROI (default behavior)
- Or delete ROI from database:
  ```sql
  DELETE FROM roi_configs 
  WHERE channel_id = 'cam_bc9c1f31' 
    AND app_name = 'IdlePeopleViolation';
  ```

---

## Testing & Validation

### Manual Testing Steps
1. ✅ Access ROI editor (`/roi_editor_idle_people`)
2. ✅ Draw a 4-point rectangle polygon
3. ✅ Save ROI (verify success message)
4. ✅ Check database for ROI entry
5. ✅ Restart application
6. ✅ Verify yellow overlay on live feed
7. ✅ Place person inside ROI → should be detected
8. ✅ Place person outside ROI → should be ignored

### Automated Testing
```bash
# Run ROI test script
python scripts/test_roi_idle_people.py

# Expected output:
# ✅ ROI found for channel: cam_bc9c1f31
# ✅ Number of points: 4
# ✅ Shapely is installed and working
# ✅ Test polygon created successfully
```

### Database Verification
```sql
-- Check ROI exists
SELECT * FROM roi_configs WHERE app_name = 'IdlePeopleViolation';

-- View ROI points
SELECT 
    channel_id,
    app_name,
    roi_points->'points' as polygon_points,
    created_at
FROM roi_configs
WHERE app_name = 'IdlePeopleViolation';
```

---

## Performance Impact

### With ROI:
- **✅ Faster processing**: Fewer detections to track
- **✅ Reduced false positives**: Ignores irrelevant areas
- **✅ Lower CPU usage**: Less tracking overhead

### Without ROI:
- **⚠️ Full frame monitoring**: All detections processed
- **⚠️ More false positives**: Tracks people in hallways, etc.
- **⚠️ Higher CPU load**: More tracking operations

---

## Best Practices

1. **Define Clear Boundaries**
   - Draw ROI around waiting areas, service counters, or checkout zones
   - Avoid including doorways or transitional areas
   - Use 4-8 points for most cases (more points = more complex polygon)

2. **Test Thoroughly**
   - Walk through the monitored area
   - Verify detections inside ROI are tracked
   - Confirm detections outside ROI are ignored
   - Check polygon overlay is visible

3. **Adjust as Needed**
   - Monitor for false positives/negatives
   - Refine polygon if needed
   - Update ROI as layout changes

4. **Document Changes**
   - Note ROI changes in deployment logs
   - Keep screenshots of ROI configuration
   - Update team on monitoring zone changes

---

## Summary

The ROI functionality provides:
- ✅ **Flexible zone definition** via web UI
- ✅ **Database persistence** across restarts
- ✅ **Real-time filtering** in detection loop
- ✅ **Visual feedback** with polygon overlay
- ✅ **Easy modification** without code changes
- ✅ **Performance optimization** by reducing processed detections

### Quick Reference
- **ROI Editor**: `http://localhost:5003/roi_editor_idle_people`
- **Live Monitor**: `http://localhost:5003/idle-people-violation`
- **Test Script**: `python scripts/test_roi_idle_people.py`
- **Database Table**: `roi_configs`
- **App Name**: `IdlePeopleViolation`
- **Channel ID**: `cam_bc9c1f31`

---

**Last Updated**: January 2026  
**Version**: 1.0  
**Status**: ✅ Production Ready
