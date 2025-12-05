# Adding a New Restaurant - Complete Guide

## Overview
This guide walks you through adding a new restaurant to the Sakshi AI monitoring system. Each restaurant can have multiple cameras with different use cases (People Counter, Queue Monitor, Kitchen Compliance, etc.).

---

## Prerequisites
- PostgreSQL database running (sakshi database)
- Access to DVR/RTSP streams for the new restaurant
- Restaurant details (name, location, DVR credentials)
- Admin access to the system

---

## Step-by-Step Process

### **STEP 1: Add Restaurant to Database**

Connect to PostgreSQL:
```bash
PGPASSWORD=root psql -h 127.0.0.1 -U postgres -d sakshi
```

Insert the new restaurant:
```sql
INSERT INTO restaurants (
    restaurant_code,
    restaurant_name,
    location,
    dvr_ip,
    dvr_username,
    dvr_password,
    telegram_chat_id,
    is_active
) VALUES (
    'unique_code',              -- Example: 'tea_toast_koramangala'
    'Restaurant Display Name',   -- Example: 'Tea Toast - Koramangala'
    'Full Address',             -- Example: '123 Main Road, Koramangala, Bangalore'
    '192.168.1.100',           -- DVR IP address
    'admin',                    -- DVR username
    'password123',              -- DVR password
    '-1001234567890',          -- Telegram chat ID for alerts (optional)
    true                        -- Active status
);
```

**Get the restaurant ID** (you'll need this):
```sql
SELECT id, restaurant_code, restaurant_name FROM restaurants ORDER BY id DESC LIMIT 1;
```

---

### **STEP 2: Add Cameras for the Restaurant**

For each camera/RTSP stream, insert a record:

```sql
INSERT INTO cameras (
    channel_id,
    channel_name,
    rtsp_url,
    restaurant_id,
    is_active
) VALUES (
    'cam_unique_identifier',    -- Auto-generated or manual (e.g., 'cam_abc123def456')
    'Camera Location Name',      -- Example: 'Main Entrance', 'Kitchen Area'
    'rtsp://username:password@192.168.1.100:554/Streaming/Channels/101',
    1,                          -- Replace with restaurant_id from STEP 1
    true                        -- Active status
);
```

**Example - Adding multiple cameras:**
```sql
-- Camera 1: Main Entrance
INSERT INTO cameras (channel_id, channel_name, rtsp_url, restaurant_id, is_active)
VALUES ('cam_entrance_001', 'Main Entrance', 'rtsp://admin:pass@192.168.1.100:554/Streaming/Channels/101', 2, true);

-- Camera 2: Checkout Queue
INSERT INTO cameras (channel_id, channel_name, rtsp_url, restaurant_id, is_active)
VALUES ('cam_checkout_001', 'Checkout Queue', 'rtsp://admin:pass@192.168.1.100:554/Streaming/Channels/201', 2, true);

-- Camera 3: Kitchen
INSERT INTO cameras (channel_id, channel_name, rtsp_url, restaurant_id, is_active)
VALUES ('cam_kitchen_001', 'Kitchen Area', 'rtsp://admin:pass@192.168.1.100:554/Streaming/Channels/301', 2, true);
```

**Verify cameras added:**
```sql
SELECT id, channel_id, channel_name, restaurant_id FROM cameras WHERE restaurant_id = 2;
```

---

### **STEP 3: Map Cameras to Use Cases (Apps)**

Link each camera to its monitoring application:

```sql
INSERT INTO camera_apps (
    camera_id,
    app_name,
    is_active
) VALUES (
    1,                  -- Replace with camera.id from STEP 2
    'PeopleCounter',    -- Use case: PeopleCounter, QueueMonitor, KitchenCompliance, Generic, OccupancyMonitor
    true
);
```

**Example - Mapping cameras to apps:**
```sql
-- Main Entrance → People Counter
INSERT INTO camera_apps (camera_id, app_name, is_active)
SELECT id, 'PeopleCounter', true FROM cameras WHERE channel_id = 'cam_entrance_001';

-- Checkout Queue → Queue Monitor
INSERT INTO camera_apps (camera_id, app_name, is_active)
SELECT id, 'QueueMonitor', true FROM cameras WHERE channel_id = 'cam_checkout_001';

-- Kitchen → Kitchen Compliance
INSERT INTO camera_apps (camera_id, app_name, is_active)
SELECT id, 'KitchenCompliance', true FROM cameras WHERE channel_id = 'cam_kitchen_001';
```

**Available App Names:**
- `PeopleCounter` - Count people entering/exiting
- `QueueMonitor` - Monitor queue length and wait times
- `KitchenCompliance` - Check uniform, gloves, caps
- `Generic` - General monitoring (security, violations)
- `OccupancyMonitor` - Track occupancy levels

**Verify mappings:**
```sql
SELECT ca.id, c.channel_name, ca.app_name, ca.is_active
FROM camera_apps ca
JOIN cameras c ON ca.camera_id = c.id
WHERE c.restaurant_id = 2;
```

---

### **STEP 4: Configure ROI (Region of Interest) Points**

**IMPORTANT:** ROI points define the monitoring areas in each camera view. You need to draw these polygons using the ROI tool.

#### **Option A: Use the ROI Finder Tool (Recommended)**

1. Start the application:
```bash
cd "/home/rasheeque/VS CODE FOLDER/TEA TOAST"
source .venv/bin/activate
python3 roi-finder.py
```

2. Access the ROI tool: `http://127.0.0.1:5002`

3. Select the camera and use case

4. Draw ROI polygon by clicking points on the video

5. Save - this automatically updates `roi_configs` table

#### **Option B: Manual SQL Insert (Advanced)**

If you have pre-calculated ROI coordinates:

```sql
INSERT INTO roi_configs (
    channel_id,
    app_name,
    roi_points,
    restaurant_id
) VALUES (
    'cam_checkout_001',          -- Camera channel_id
    'QueueMonitor',              -- Use case
    '{"main_roi": [[100,200], [300,200], [300,400], [100,400]], "secondary_roi": [[310,150], [500,150], [500,400], [310,400]]}',
    2                            -- Restaurant ID
);
```

**ROI Points Format:**
- JSON string with coordinate arrays
- `main_roi`: Primary monitoring zone (e.g., queue area, entrance zone)
- `secondary_roi`: Secondary zone (optional - e.g., checkout counter, exit door)
- Coordinates are `[x, y]` pixel positions

**Example ROI configs for different apps:**

```sql
-- People Counter ROI (entrance line)
INSERT INTO roi_configs (channel_id, app_name, roi_points, restaurant_id)
VALUES ('cam_entrance_001', 'PeopleCounter', 
'{"main_roi": [[200,400], [600,400], [600,420], [200,420]]}', 2);

-- Queue Monitor ROI (queue area + counter)
INSERT INTO roi_configs (channel_id, app_name, roi_points, restaurant_id)
VALUES ('cam_checkout_001', 'QueueMonitor',
'{"main_roi": [[50,300], [400,300], [400,600], [50,600]], "secondary_roi": [[410,250], [700,250], [700,600], [410,600]]}', 2);

-- Kitchen Compliance ROI (kitchen work area)
INSERT INTO roi_configs (channel_id, app_name, roi_points, restaurant_id)
VALUES ('cam_kitchen_001', 'KitchenCompliance',
'{"main_roi": [[100,200], [900,200], [900,700], [100,700]]}', 2);
```

**Verify ROI configs:**
```sql
SELECT rc.id, c.channel_name, rc.app_name, 
       LENGTH(rc.roi_points) as roi_data_length
FROM roi_configs rc
JOIN cameras c ON rc.channel_id = c.channel_id
WHERE rc.restaurant_id = 2;
```

---

### **STEP 5: Restart the Application**

After adding all configuration:

```bash
# Stop the running application
pkill -f "python3.*edit-004.py"

# Start with virtual environment
cd "/home/rasheeque/VS CODE FOLDER/TEA TOAST"
source .venv/bin/activate
python3 edit-004.py
```

---

### **STEP 6: Verify in Dashboard**

1. Open dashboard: `http://127.0.0.1:5001`

2. Check restaurant dropdown (top-right corner) - new restaurant should appear

3. Select the new restaurant from dropdown

4. Verify all cameras load correctly

5. Check that ROI lines are displayed on video feeds

---

## Complete Example: Adding "Tea Toast - Koramangala"

```sql
-- STEP 1: Add Restaurant
INSERT INTO restaurants (restaurant_code, restaurant_name, location, dvr_ip, dvr_username, dvr_password, is_active)
VALUES ('tea_toast_koramangala', 'Tea Toast - Koramangala', '80 Feet Road, Koramangala, Bangalore', '192.168.2.50', 'admin', 'TeaToast@2024', true);

-- Get restaurant_id (assume it returns id = 2)
SELECT id FROM restaurants WHERE restaurant_code = 'tea_toast_koramangala';

-- STEP 2: Add 3 Cameras
INSERT INTO cameras (channel_id, channel_name, rtsp_url, restaurant_id, is_active) VALUES
('cam_kmg_entrance', 'Main Entrance', 'rtsp://admin:TeaToast@2024@192.168.2.50:554/Streaming/Channels/101', 2, true),
('cam_kmg_queue', 'Checkout Queue', 'rtsp://admin:TeaToast@2024@192.168.2.50:554/Streaming/Channels/201', 2, true),
('cam_kmg_kitchen', 'Kitchen Area', 'rtsp://admin:TeaToast@2024@192.168.2.50:554/Streaming/Channels/301', 2, true);

-- STEP 3: Map to Apps
INSERT INTO camera_apps (camera_id, app_name, is_active)
SELECT id, 'PeopleCounter', true FROM cameras WHERE channel_id = 'cam_kmg_entrance'
UNION ALL
SELECT id, 'QueueMonitor', true FROM cameras WHERE channel_id = 'cam_kmg_queue'
UNION ALL
SELECT id, 'KitchenCompliance', true FROM cameras WHERE channel_id = 'cam_kmg_kitchen';

-- STEP 4: Add ROI (use roi-finder.py tool to draw these interactively)
-- Or manually if coordinates are known:
INSERT INTO roi_configs (channel_id, app_name, roi_points, restaurant_id) VALUES
('cam_kmg_entrance', 'PeopleCounter', '{"main_roi": [[300,500], [700,500], [700,520], [300,520]]}', 2),
('cam_kmg_queue', 'QueueMonitor', '{"main_roi": [[100,400], [500,400], [500,700], [100,700]], "secondary_roi": [[510,350], [800,350], [800,700], [510,700]]}', 2),
('cam_kmg_kitchen', 'KitchenCompliance', '{"main_roi": [[150,250], [950,250], [950,650], [150,650]]}', 2);
```

---

## Troubleshooting

### **Restaurant not appearing in dropdown**
```sql
-- Check if restaurant exists and is active
SELECT * FROM restaurants WHERE is_active = true;
```

### **Camera feeds not loading**
```sql
-- Verify camera and app mappings
SELECT c.channel_id, c.channel_name, c.rtsp_url, ca.app_name, ca.is_active
FROM cameras c
LEFT JOIN camera_apps ca ON c.id = ca.camera_id
WHERE c.restaurant_id = 2;

-- Test RTSP stream manually
ffplay "rtsp://username:password@ip:port/path"
```

### **ROI not displaying correctly**
```sql
-- Check ROI configuration
SELECT channel_id, app_name, roi_points FROM roi_configs WHERE restaurant_id = 2;

-- Verify JSON format is valid
SELECT channel_id, 
       jsonb_pretty(roi_points::jsonb) as formatted_roi 
FROM roi_configs 
WHERE restaurant_id = 2;
```

### **Application errors on startup**
- Check logs: Application prints detailed startup logs
- Verify all foreign keys are correct (restaurant_id, camera_id)
- Ensure RTSP URLs are accessible from server
- Confirm DVR credentials are correct

---

## Database Schema Reference

### **restaurants**
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key (auto-increment) |
| restaurant_code | VARCHAR | Unique code (e.g., 'tea_toast_brigade') |
| restaurant_name | VARCHAR | Display name |
| location | VARCHAR | Full address |
| dvr_ip | VARCHAR | DVR IP address |
| dvr_username | VARCHAR | DVR login username |
| dvr_password | VARCHAR | DVR login password |
| telegram_chat_id | VARCHAR | Telegram notification chat ID |
| is_active | BOOLEAN | Active status |

### **cameras**
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| channel_id | VARCHAR | Unique camera identifier |
| channel_name | VARCHAR | Display name |
| rtsp_url | VARCHAR | Full RTSP stream URL |
| restaurant_id | INTEGER | FK to restaurants.id |
| is_active | BOOLEAN | Active status |

### **camera_apps**
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| camera_id | INTEGER | FK to cameras.id |
| app_name | VARCHAR | Use case name |
| is_active | BOOLEAN | Active status |

### **roi_configs**
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| channel_id | VARCHAR | FK to cameras.channel_id |
| app_name | VARCHAR | Use case name |
| roi_points | TEXT | JSON string with coordinates |
| restaurant_id | INTEGER | FK to restaurants.id |

**Unique Constraint:** `(channel_id, app_name)` - Each camera can have only one ROI per app

---

## Quick Reference Commands

### **View all restaurants**
```sql
SELECT id, restaurant_code, restaurant_name, location, is_active FROM restaurants;
```

### **View all cameras for a restaurant**
```sql
SELECT c.id, c.channel_id, c.channel_name, ca.app_name
FROM cameras c
LEFT JOIN camera_apps ca ON c.id = ca.camera_id
WHERE c.restaurant_id = 1
ORDER BY c.id;
```

### **View complete configuration for a restaurant**
```sql
SELECT 
    r.restaurant_name,
    c.channel_name,
    ca.app_name,
    CASE WHEN rc.id IS NOT NULL THEN 'Yes' ELSE 'No' END as has_roi
FROM restaurants r
JOIN cameras c ON r.id = c.restaurant_id
LEFT JOIN camera_apps ca ON c.id = ca.camera_id
LEFT JOIN roi_configs rc ON c.channel_id = rc.channel_id AND ca.app_name = rc.app_name
WHERE r.id = 1
ORDER BY c.id, ca.app_name;
```

### **Delete a restaurant (and all related data)**
```sql
-- ⚠️ WARNING: This will cascade delete all cameras, apps, and ROI configs
DELETE FROM restaurants WHERE id = 2;
```

### **Deactivate a restaurant (soft delete)**
```sql
-- Recommended: Just deactivate instead of deleting
UPDATE restaurants SET is_active = false WHERE id = 2;
UPDATE cameras SET is_active = false WHERE restaurant_id = 2;
```

---

## Notes

- **Channel IDs**: Must be unique across all restaurants. Use format like `cam_<location>_<number>` or let the system auto-generate from RTSP URL hash.

- **RTSP URLs**: Must be accessible from the server. Test connectivity before adding to database.

- **ROI Configuration**: Critical for accurate monitoring. Use the `roi-finder.py` tool for interactive drawing rather than manual coordinate entry.

- **Telegram Alerts**: Optional but recommended. Get chat_id by messaging [@userinfobot](https://t.me/userinfobot) on Telegram.

- **Performance**: Each camera spawns a separate processing thread. Monitor server resources when adding multiple restaurants.

- **Backup**: Always backup the database before making bulk changes:
  ```bash
  pg_dump -h 127.0.0.1 -U postgres -d sakshi > backup_$(date +%Y%m%d_%H%M%S).sql
  ```

---

## Support

For issues or questions:
1. Check application logs: `tail -f` the terminal running `edit-004.py`
2. Verify database connections: `PGPASSWORD=root psql -h 127.0.0.1 -U postgres -d sakshi`
3. Test RTSP streams: Use VLC or `ffplay` to verify stream accessibility
4. Review this guide's troubleshooting section

---

**Document Version:** 1.0  
**Last Updated:** December 5, 2025  
**Application:** Sakshi AI - Restaurant Monitoring System
