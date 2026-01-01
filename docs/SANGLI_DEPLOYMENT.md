# 🏪 Sangli Store Deployment Guide

## 📋 Store Details

**Restaurant:** Tea Toast - Sangli  
**Location:** Sangli, Maharashtra  
**DVR IP:** 110.227.214.137  
**DVR Username:** admin  
**DVR Password:** Cctv@4321

---

## 📹 Camera Configuration

| Channel | Location | Use Cases | Model |
|---------|----------|-----------|-------|
| **1** | Main Entrance | PeopleCounter | yolo11n.pt |
| **4** | Front Office | OccupancyMonitor + Generic (Compliance) | yolo11n.pt + kitchen_violation_30_12_2025.pt |
| **5** | Checkout & Kitchen | QueueMonitor + KitchenCompliance | yolo11n.pt + kitchen_violation_30_12_2025.pt |
| **7** | Kitchen Compliance Zone | KitchenCompliance | kitchen_violation_30_12_2025.pt |

**RTSP Base URL:** `rtsp://admin:Cctv%404321@110.227.214.137:554/cam/realmonitor?channel={N}&subtype=0`

---

## 🚀 Deployment Steps

### **STEP 1: Onboard Sangli Store to Database**

Run the automated onboarding script:

```bash
cd /home/sanika/teatoast/normal-sakshi
python3 scripts/onboard_sangli_store.py
```

**Expected Output:**
```
✅ Restaurant created: Tea Toast - Sangli
✅ Channel 1: Main Entrance
✅ Channel 4: Front Office
✅ Channel 5: Checkout & Kitchen Area
✅ Channel 7: Kitchen Compliance Zone
✅ Total Active Cameras: 4
✅ Total Apps Configured: 6
```

---

### **STEP 2: Verify Database Entry**

Check if Sangli store was added:

```bash
PGPASSWORD=Tneural01 psql -U postgres -h 127.0.0.1 -d sakshi -c "
SELECT r.id, r.restaurant_name, r.location, COUNT(c.id) as camera_count
FROM restaurants r
LEFT JOIN cameras c ON r.id = c.restaurant_id
WHERE r.restaurant_code = 'sangli_store'
GROUP BY r.id, r.restaurant_name, r.location;
"
```

View all cameras and apps for Sangli:

```bash
PGPASSWORD=Tneural01 psql -U postgres -h 127.0.0.1 -d sakshi -c "
SELECT 
    c.channel_number,
    c.channel_name,
    STRING_AGG(ca.app_name, ', ') as apps
FROM cameras c
JOIN camera_apps ca ON c.id = ca.camera_id
JOIN restaurants r ON c.restaurant_id = r.id
WHERE r.restaurant_code = 'sangli_store'
GROUP BY c.channel_number, c.channel_name
ORDER BY c.channel_number;
"
```

---

### **STEP 3: Restart Application**

Restart the service to load new cameras:

```bash
sudo systemctl restart sakshi-ai.service
```

Check status:

```bash
sudo systemctl status sakshi-ai.service
```

View logs:

```bash
sudo journalctl -u sakshi-ai.service -f
```

---

### **STEP 4: Configure ROI (Region of Interest)**

**⚠️ IMPORTANT:** You must configure ROI zones for proper detection.

#### **4.1 PeopleCounter (Channel 1 - Main Entrance)**
1. Open: http://YOUR_SERVER_IP:5001/roi_editor_people?channel_id=cam_XXXXX
2. Draw a horizontal line across the entrance
3. Save configuration

#### **4.2 QueueMonitor (Channel 5 - Checkout Area)**
1. Open: http://YOUR_SERVER_IP:5001/roi_editor?channel_id=cam_XXXXX
2. Draw polygon around queue waiting area (main_roi)
3. Draw polygon around checkout counter (secondary_roi)
4. Save configuration

#### **4.3 OccupancyMonitor (Channel 4 - Front Office)**
1. Open: http://YOUR_SERVER_IP:5001/roi_editor?channel_id=cam_XXXXX
2. Draw polygon around the area to monitor occupancy
3. Upload occupancy schedule (if needed)
4. Save configuration

> **Note:** Kitchen Compliance channels (5, 7) and Generic (4) don't require ROI - they scan the entire frame.

---

### **STEP 5: Verify Dashboard Access**

#### **5.1 Test Main Dashboard**
```
http://YOUR_SERVER_IP:5001/dashboard
```

- Check if "Tea Toast - Sangli" appears in restaurant dropdown
- Select Sangli from dropdown
- Verify 4 camera feeds are visible

#### **5.2 Test Restaurant-Specific URL**
```
http://YOUR_SERVER_IP:5001/dashboard?restaurant_id=2
```
(Replace `2` with actual Sangli restaurant ID from database)

---

### **STEP 6: Test Camera Streams**

For each camera, verify:
- ✅ Video feed loads without errors
- ✅ Detection boxes appear (people, violations)
- ✅ Counts increment correctly (PeopleCounter, Queue)
- ✅ Alerts trigger when thresholds met

---

## 🔧 AWS Server Deployment

### **Prerequisites**
- AWS EC2 instance running Ubuntu/Linux
- PostgreSQL database accessible
- Port 5001 open for dashboard access
- Models folder uploaded with all .pt files

### **Deployment Checklist**

- [ ] **Clone Repository**
  ```bash
  git clone https://github.com/abhi-20-25/normal-sakshi.git
  cd normal-sakshi
  git checkout Tea-toast-new-v1
  ```

- [ ] **Install Dependencies**
  ```bash
  python3 -m venv ttenv
  source ttenv/bin/activate
  pip install -r requirements.txt
  ```

- [ ] **Upload Models**
  ```bash
  # Ensure these models exist in models/ folder:
  ls -lh models/yolo11n.pt
  ls -lh models/kitchen_violation_30_12_2025.pt
  ```

- [ ] **Configure Database**
  ```bash
  # Update DATABASE_URL in edit-004.py if needed
  # Run onboarding script
  python3 scripts/onboard_sangli_store.py
  ```

- [ ] **Setup Systemd Service**
  ```bash
  sudo cp scripts/sakshi-ai.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable sakshi-ai.service
  sudo systemctl start sakshi-ai.service
  ```

- [ ] **Configure Firewall**
  ```bash
  sudo ufw allow 5001/tcp
  sudo ufw allow 5002/tcp  # FastAPI (when available)
  ```

- [ ] **Test RTSP Connectivity**
  ```bash
  # Test if DVR is reachable from AWS server
  ffprobe -rtsp_transport tcp "rtsp://admin:Cctv%404321@110.227.214.137:554/cam/realmonitor?channel=1&subtype=0"
  ```

---

## 📊 Post-Deployment Verification

### **Check Application Logs**
```bash
tail -f /var/log/sakshi-ai/app.log
```

Look for:
```
✅ Found 4 cameras for restaurant: Tea Toast - Sangli
📹 Starting PeopleCounter for Main Entrance
📹 Starting QueueMonitor for Checkout & Kitchen Area
📹 Starting KitchenCompliance for Kitchen Compliance Zone
📹 Starting OccupancyMonitor for Front Office
```

### **Test Each Use Case**

1. **PeopleCounter (Ch1)**
   - Walk across entrance line
   - Verify count increments: http://SERVER_IP:5001/api/peak_analytics/cam_XXXXX

2. **QueueMonitor (Ch5)**
   - Simulate queue with multiple people
   - Verify alerts trigger when threshold reached

3. **KitchenCompliance (Ch5, Ch7)**
   - Check kitchen staff for violations
   - Verify alerts for missing gloves/caps/aprons

4. **OccupancyMonitor (Ch4)**
   - Check occupancy count updates
   - Verify schedule-based detection (if configured)

5. **Generic/Front Office (Ch4)**
   - Monitor for general violations
   - Check detection logs

---

## 🔗 Integration Notes

### **PetPooja Integration (Pending)**

When Sangli's PetPooja API becomes available:

1. **Update fastapi_app.py** with Sangli credentials:
   ```python
   RESTAURANT_CONFIGS = {
       'tea_toast': {...},
       'sangli_store': {
           'api_key': 'YOUR_SANGLI_API_KEY',
           'api_secret': 'YOUR_SANGLI_SECRET',
           'restaurant_id': 'SANGLI_REST_ID'
       }
   }
   ```

2. **Restart FastAPI service:**
   ```bash
   sudo systemctl restart petpooja-api.service
   ```

3. **Test webhook endpoint:**
   ```bash
   curl http://localhost:5002/webhook/petpooja/sangli_store
   ```

---

## 🐛 Troubleshooting

### **Camera Not Streaming**
```bash
# Test RTSP connection
ffplay "rtsp://admin:Cctv%404321@110.227.214.137:554/cam/realmonitor?channel=1&subtype=0"

# Check if DVR is accessible
ping 110.227.214.137

# Verify network routing (if on AWS)
traceroute 110.227.214.137
```

### **No Detection Happening**
- Check if models exist: `ls -lh models/*.pt`
- Verify ROI is configured (for PeopleCounter, Queue, Occupancy)
- Check logs for YOLO errors

### **Restaurant Not Showing in Dropdown**
```bash
# Check if restaurant is active
PGPASSWORD=Tneural01 psql -U postgres -h 127.0.0.1 -d sakshi -c "
SELECT * FROM restaurants WHERE restaurant_code = 'sangli_store';
"

# Restart application
sudo systemctl restart sakshi-ai.service
```

---

## 📞 Support Contacts

**Technical Issues:**
- Check logs: `sudo journalctl -u sakshi-ai.service -f`
- Review DOCUMENTATION.md for detailed guides

**DVR Access Issues:**
- Verify DVR IP: 110.227.214.137
- Check credentials: admin / Cctv@4321
- Ensure port 554 is open

---

## ✅ Deployment Verification Checklist

- [ ] Sangli store added to database
- [ ] All 4 cameras configured (Ch1, Ch4, Ch5, Ch7)
- [ ] All 6 app instances linked (1 PeopleCounter, 1 Queue, 2 Kitchen, 1 Occupancy, 1 Generic)
- [ ] Application restarted successfully
- [ ] Camera streams visible on dashboard
- [ ] ROI configured for PeopleCounter, Queue, Occupancy
- [ ] Detection models loading correctly
- [ ] Alerts triggering for violations
- [ ] Dashboard accessible at http://SERVER_IP:5001
- [ ] Sangli appears in restaurant dropdown
- [ ] Filtering by restaurant works
- [ ] Logs show no critical errors

---

## 🎯 Success Criteria

✅ **Fully Operational When:**
1. All 4 camera feeds streaming smoothly
2. People counting working on Channel 1
3. Queue monitoring active on Channel 5
4. Kitchen compliance detecting violations on Ch5 & Ch7
5. Occupancy tracking on Channel 4
6. Front office compliance monitoring on Channel 4
7. Dashboard shows Sangli data separately
8. Alerts sending to Telegram (using Tea Toast chat for now)

---

**Last Updated:** January 1, 2026  
**Script Version:** 1.0  
**Author:** Onboarding Automation System
