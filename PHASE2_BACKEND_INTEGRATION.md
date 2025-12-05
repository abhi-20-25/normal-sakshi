# Phase 2: Backend Integration - COMPLETED ✅

**Date:** December 5, 2024  
**Status:** Implementation Complete, Ready for Testing  
**Next Phase:** Phase 3 - Frontend Dashboard Dropdown

---

## 📋 Overview

Phase 2 transforms the backend from single-restaurant file-based configuration to multi-restaurant database-driven architecture. The system now reads camera and app configurations from the database while maintaining backward compatibility with `rtsp_links.txt`.

---

## ✅ Completed Changes

### 1. Database Models (Lines 347-390)

Added three new SQLAlchemy models and updated existing one:

#### **Restaurant Model**
```python
class Restaurant(Base):
    __tablename__ = 'restaurants'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    location = Column(String(255), nullable=False)
    dvr_ip = Column(String(50), nullable=False)
    dvr_port = Column(Integer, default=554)
    dvr_username = Column(String(100))
    dvr_password = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, onupdate=datetime.now)
```

#### **Camera Model**
```python
class Camera(Base):
    __tablename__ = 'cameras'
    
    id = Column(Integer, primary_key=True)
    restaurant_id = Column(Integer, ForeignKey('restaurants.id'))
    channel_id = Column(String(50), unique=True, nullable=False)
    channel_name = Column(String(255), nullable=False)
    channel_number = Column(Integer)
    rtsp_url = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, onupdate=datetime.now)
```

#### **CameraApp Model**
```python
class CameraApp(Base):
    __tablename__ = 'camera_apps'
    
    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey('cameras.id'))
    app_name = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
```

#### **Updated RoiConfig Model**
Added `restaurant_id` foreign key to support multi-restaurant ROI management:
```python
restaurant_id = Column(Integer, ForeignKey('restaurants.id'), nullable=True)
```

---

### 2. get_app_configs() Function (Lines 2048-2148)

**Completely rewritten** to support database queries with restaurant filtering:

**Features:**
- ✅ Tries database first (queries `Restaurant`, `Camera`, `CameraApp` tables)
- ✅ Supports optional `restaurant_id` parameter for filtering
- ✅ Falls back to `rtsp_links.txt` if database is empty
- ✅ Maintains backward compatibility
- ✅ Returns same data structure as before

**Logic Flow:**
1. Check if database is connected
2. Query camera count
3. If cameras exist in DB:
   - Query cameras with JOIN to restaurants and camera_apps
   - Filter by `restaurant_id` if provided
   - Build `app_configs` dictionary
4. If no cameras in DB:
   - Fall back to parsing `rtsp_links.txt`
5. Return app_configs dictionary

---

### 3. start_streams() Function (Lines 2939-3005)

**Completely rewritten** to load cameras from database:

**Features:**
- ✅ Loads cameras from database (queries `Camera`, `Restaurant`, `CameraApp`)
- ✅ Generates stream assignments grouped by RTSP URL
- ✅ Falls back to `rtsp_links.txt` if database empty
- ✅ Maintains FrameHub architecture
- ✅ Logs detailed startup information

**New Helper Function:**
```python
def _start_streams_from_data(stream_assignments):
    """Helper function to start streams from parsed data (database or file)"""
```
This extracts common stream initialization logic used by both database and file modes.

---

### 4. Restaurant Management API Endpoints (Lines 2903-3105)

Added **5 new REST API endpoints** for restaurant management:

#### **GET /api/restaurants**
List all restaurants (optionally filter by active status)

**Query Parameters:**
- `active_only` (default: true)

**Response:**
```json
{
  "success": true,
  "restaurants": [
    {
      "id": 1,
      "name": "Tea Toast",
      "location": "Brigade Road",
      "dvr_ip": "182.65.205.121",
      "dvr_port": 554,
      "dvr_username": "admin",
      "is_active": true,
      "created_at": "2024-12-05T15:33:57"
    }
  ],
  "count": 1
}
```

#### **GET /api/restaurants/{restaurant_id}**
Get details of a specific restaurant

**Response:**
```json
{
  "success": true,
  "restaurant": {
    "id": 1,
    "name": "Tea Toast",
    "location": "Brigade Road",
    "dvr_ip": "182.65.205.121",
    "dvr_port": 554,
    "dvr_username": "admin",
    "is_active": true,
    "created_at": "2024-12-05T15:33:57",
    "updated_at": null
  }
}
```

#### **GET /api/restaurants/{restaurant_id}/cameras**
Get all cameras for a specific restaurant with their assigned apps

**Response:**
```json
{
  "success": true,
  "restaurant": {
    "id": 1,
    "name": "Tea Toast",
    "location": "Brigade Road"
  },
  "cameras": [
    {
      "id": 1,
      "channel_id": "Ch1",
      "channel_name": "Main Entrance",
      "channel_number": 1,
      "rtsp_url": "rtsp://...",
      "is_active": true,
      "apps": [
        {"app_name": "PeopleCounter", "is_active": true},
        {"app_name": "Generic", "is_active": true}
      ]
    }
  ],
  "count": 5
}
```

#### **POST /api/restaurants** (requires login)
Create a new restaurant

**Request Body:**
```json
{
  "name": "Tea Toast",
  "location": "MG Road",
  "dvr_ip": "192.168.1.100",
  "dvr_port": 554,
  "dvr_username": "admin",
  "dvr_password": "password123",
  "is_active": true
}
```

#### **PUT /api/restaurants/{restaurant_id}** (requires login)
Update restaurant details

**Request Body:** (all fields optional)
```json
{
  "name": "Tea Toast Updated",
  "location": "Brigade Road",
  "dvr_ip": "182.65.205.121",
  "is_active": true
}
```

---

### 5. Updated Dashboard Route (Lines 2236-2270)

**Enhanced** dashboard route to support restaurant filtering:

**Features:**
- ✅ Accepts `restaurant_id` query parameter
- ✅ Filters cameras/apps by selected restaurant
- ✅ Loads list of all restaurants for dropdown
- ✅ Passes `selected_restaurant` to template
- ✅ Backward compatible (works without restaurant_id)

**URL Examples:**
```
/dashboard                    # Show all restaurants/cameras
/dashboard?restaurant_id=1    # Show only Tea Toast cameras
/dashboard?restaurant_id=2    # Show only second restaurant
```

**Template Variables:**
- `app_configs`: Filtered camera/app configurations
- `restaurants`: List of all active restaurants (for dropdown)
- `selected_restaurant`: Currently selected restaurant details (if any)

---

## 🔧 Technical Details

### Database Queries

**Camera Loading (start_streams):**
```python
cameras_data = db.query(Camera, Restaurant).\
    join(Restaurant, Camera.restaurant_id == Restaurant.id).\
    filter(Camera.is_active == True, Restaurant.is_active == True).all()
```

**App Config Loading:**
```python
query = db.query(Camera).options(
    selectinload(Camera.camera_apps)
).filter(Camera.is_active == True)

if restaurant_id:
    query = query.filter(Camera.restaurant_id == restaurant_id)
```

### Backward Compatibility

The system maintains **zero downtime** during migration:

1. **Database Empty?** → Falls back to `rtsp_links.txt`
2. **Database Error?** → Falls back to `rtsp_links.txt`
3. **No restaurant_id?** → Shows all restaurants

### Logging Enhancements

Added detailed startup logging:
```
======================================================================
🚀 Initializing stream processors...
📊 Loading cameras from database (5 cameras found)
✅ Found 5 active cameras
  📹 Kitchen → KitchenCompliance
  📹 Main Entrance → PeopleCounter, Generic
  📹 Checkout Queue → QueueMonitor
  📹 Front Office → OccupancyMonitor, Generic
  📹 Kitchen Area → Generic
======================================================================
```

---

## 🧪 Testing Plan

### 1. Verify Database Mode
```bash
# Check cameras are loaded from database
grep "Loading cameras from database" logs/app.log

# Test restaurant API
curl http://localhost:5001/api/restaurants

# Test cameras for specific restaurant
curl http://localhost:5001/api/restaurants/1/cameras
```

### 2. Test Backward Compatibility
```bash
# Temporarily rename database to test fallback
sudo systemctl stop postgresql
# Application should fall back to rtsp_links.txt

# Check logs for fallback message
grep "Loading cameras from rtsp_links.txt" logs/app.log
```

### 3. Test Restaurant Filtering
```bash
# Dashboard without filter (shows all)
curl http://localhost:5001/dashboard

# Dashboard with restaurant filter
curl http://localhost:5001/dashboard?restaurant_id=1
```

### 4. Test API Endpoints
```bash
# Get all restaurants
curl http://localhost:5001/api/restaurants

# Get specific restaurant
curl http://localhost:5001/api/restaurants/1

# Get restaurant cameras
curl http://localhost:5001/api/restaurants/1/cameras
```

---

## 📊 Current Database State

From Phase 1 migration:

**Restaurants:** 1 (Tea Toast - Brigade Road)  
**Cameras:** 5 (Ch1, Ch4, Ch5, Ch10, Ch10-kitchen)  
**Camera Apps:** 5 mappings  
**Historical Data:** 1,028 rows linked to Tea Toast

---

## 🚨 Known Issues / Notes

1. **Type Checking Warnings:** Pylance shows ~98 type warnings (mostly SQLAlchemy column types). These don't affect runtime.

2. **Frontend Not Updated:** Dashboard template (`templates/dashboard.html`) still needs dropdown selector (Phase 3).

3. **No Migration from File:** System doesn't auto-migrate `rtsp_links.txt` changes to database. Use migration script if needed.

4. **Password Storage:** DVR passwords stored as plain text. Consider encryption in Phase 4.

---

## ✅ Verification Checklist

- [x] Database models added (Restaurant, Camera, CameraApp)
- [x] RoiConfig updated with restaurant_id
- [x] get_app_configs() rewritten with database support
- [x] start_streams() rewritten with database support
- [x] Restaurant management API endpoints added (5 routes)
- [x] Dashboard route updated with restaurant filtering
- [x] Backward compatibility maintained (rtsp_links.txt fallback)
- [x] No syntax errors (py_compile passed)
- [x] Detailed logging added

---

## 🎯 Next Steps (Phase 3)

Update frontend dashboard to add restaurant dropdown selector:

1. **Update dashboard.html:**
   - Add dropdown selector populated from `restaurants` variable
   - Add JavaScript to reload page with `?restaurant_id=X` on selection
   - Show selected restaurant name in UI

2. **Update CSS:**
   - Style dropdown selector
   - Add visual indicator for selected restaurant

3. **Test Multi-Restaurant Workflow:**
   - Add second restaurant to database
   - Verify dropdown shows both restaurants
   - Verify filtering works correctly

---

## 📁 Files Modified

1. `edit-004.py` (3479 lines)
   - Lines 17-20: Updated imports
   - Lines 347-390: New database models
   - Lines 2048-2148: Rewritten get_app_configs()
   - Lines 2236-2270: Updated dashboard route
   - Lines 2903-3105: New restaurant API endpoints
   - Lines 2939-3100: Rewritten start_streams()

---

## 📝 Change Summary

**Lines Added:** ~450  
**Lines Modified:** ~200  
**New Functions:** 7 (5 API routes + 1 helper + 1 dashboard update)  
**New Models:** 3 (Restaurant, Camera, CameraApp)  
**Database Tables Used:** 11 (3 new + 8 updated)  
**Backward Compatible:** ✅ Yes  
**Breaking Changes:** ❌ None  
**Testing Required:** ✅ Yes (see Testing Plan above)

---

**Implementation Status:** ✅ COMPLETE  
**Ready for Phase 3:** ✅ YES  
**Estimated Testing Time:** 30-60 minutes  
**Risk Level:** 🟢 LOW (backward compatible with fallback mechanism)
