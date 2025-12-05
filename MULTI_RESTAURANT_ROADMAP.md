# 🏢 Multi-Restaurant Support Implementation Roadmap

## 📋 **Executive Summary**

Transform the current single-restaurant SAKSHI AI system into a multi-tenant platform where users can select different restaurants from a dropdown, automatically loading the correct RTSP streams and ROI configurations for each location.

---

## 🎯 **Current Architecture Analysis**

### **Current Setup:**
- ✅ Single restaurant: Tea Toast
- ✅ RTSP links in `rtsp_links.txt` (flat file)
- ✅ ROI configs in `roi_configs` table (PostgreSQL)
- ✅ Channel ID derived from RTSP URL hash
- ✅ All cameras share the same DVR IP: `182.65.205.121`

### **Limitations:**
- ❌ No restaurant/location identifier
- ❌ RTSP links hardcoded in file
- ❌ No restaurant-specific settings
- ❌ ROI configs not tied to restaurant
- ❌ Dashboard shows all cameras regardless of location

---

## 🏗️ **Phase 1: Database Schema Design** (Priority: HIGH)

### **1.1 Create `restaurants` Table**

```sql
CREATE TABLE restaurants (
    id SERIAL PRIMARY KEY,
    restaurant_code VARCHAR(50) UNIQUE NOT NULL,  -- e.g., 'tea_toast', 'cafe_mocha'
    restaurant_name VARCHAR(200) NOT NULL,         -- Display name
    location VARCHAR(200),                         -- Address/area
    dvr_ip VARCHAR(50),                           -- DVR IP address
    dvr_username VARCHAR(100),                    -- DVR credentials
    dvr_password VARCHAR(100),                    -- Encrypted password
    telegram_chat_id VARCHAR(50),                 -- Restaurant-specific alerts
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_restaurants_code ON restaurants(restaurant_code);
CREATE INDEX idx_restaurants_active ON restaurants(is_active);
```

**Sample Data:**
```sql
INSERT INTO restaurants (restaurant_code, restaurant_name, location, dvr_ip, dvr_username, dvr_password, telegram_chat_id)
VALUES 
    ('tea_toast', 'Tea Toast - Brigade Road', 'Brigade Road, Bangalore', '182.65.205.121', 'admin', 'cctv#1234', '-4835836048'),
    ('cafe_mocha', 'Cafe Mocha - MG Road', 'MG Road, Bangalore', '192.168.1.100', 'admin', 'pass1234', '-4835836049'),
    ('bistro_bay', 'Bistro Bay - Indiranagar', 'Indiranagar, Bangalore', '192.168.1.200', 'admin', 'secure123', '-4835836050');
```

---

### **1.2 Create `cameras` Table**

Replace the flat `rtsp_links.txt` file with a database table:

```sql
CREATE TABLE cameras (
    id SERIAL PRIMARY KEY,
    restaurant_id INTEGER REFERENCES restaurants(id) ON DELETE CASCADE,
    channel_number INTEGER NOT NULL,              -- DVR channel (1-32)
    channel_name VARCHAR(100) NOT NULL,           -- "Main Entrance", "Checkout Queue"
    subtype INTEGER DEFAULT 0,                    -- 0=main stream, 1=sub stream
    rtsp_url TEXT,                                -- Full RTSP URL (auto-generated)
    channel_id VARCHAR(100) UNIQUE,               -- Hash ID for internal use
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(restaurant_id, channel_number, subtype)
);

-- Indexes
CREATE INDEX idx_cameras_restaurant ON cameras(restaurant_id);
CREATE INDEX idx_cameras_channel_id ON cameras(channel_id);
CREATE INDEX idx_cameras_active ON cameras(is_active);
```

**Sample Data:**
```sql
-- Tea Toast cameras
INSERT INTO cameras (restaurant_id, channel_number, channel_name, subtype, is_active)
VALUES 
    (1, 10, 'Kitchen Camera', 1, true),
    (1, 1, 'Main Entrance', 1, true),
    (1, 4, 'Checkout Queue', 0, true),
    (1, 5, 'Front Office Violation', 0, true),
    (1, 10, 'Kitchen Area', 0, true);

-- Cafe Mocha cameras
INSERT INTO cameras (restaurant_id, channel_number, channel_name, subtype, is_active)
VALUES 
    (2, 1, 'Front Door', 1, true),
    (2, 2, 'Counter Area', 0, true),
    (2, 3, 'Kitchen', 1, true);
```

---

### **1.3 Create `camera_apps` Table** (Many-to-Many)

Link cameras to AI apps (PeopleCounter, QueueMonitor, etc.):

```sql
CREATE TABLE camera_apps (
    id SERIAL PRIMARY KEY,
    camera_id INTEGER REFERENCES cameras(id) ON DELETE CASCADE,
    app_name VARCHAR(50) NOT NULL,                -- 'PeopleCounter', 'QueueMonitor', etc.
    is_active BOOLEAN DEFAULT TRUE,
    config JSONB,                                 -- App-specific settings
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_camera_apps_camera ON camera_apps(camera_id);
CREATE INDEX idx_camera_apps_app ON camera_apps(app_name);

-- Unique constraint
ALTER TABLE camera_apps ADD CONSTRAINT unique_camera_app 
    UNIQUE(camera_id, app_name);
```

**Sample Data:**
```sql
-- Tea Toast: Main Entrance → PeopleCounter
INSERT INTO camera_apps (camera_id, app_name, config)
VALUES (2, 'PeopleCounter', '{"confidence": 0.15}'::jsonb);

-- Tea Toast: Checkout Queue → QueueMonitor
INSERT INTO camera_apps (camera_id, app_name, config)
VALUES (3, 'QueueMonitor', '{"confidence": 0.15, "alert_threshold": 3}'::jsonb);

-- Tea Toast: Kitchen Camera → KitchenCompliance
INSERT INTO camera_apps (camera_id, app_name, config)
VALUES (1, 'KitchenCompliance', '{"confidence": 0.35}'::jsonb);
```

---

### **1.4 Update `roi_configs` Table**

Add restaurant_id to link ROI configs to specific restaurants:

```sql
-- Add restaurant_id column
ALTER TABLE roi_configs ADD COLUMN restaurant_id INTEGER REFERENCES restaurants(id);

-- Update existing data (set to Tea Toast)
UPDATE roi_configs SET restaurant_id = 1;

-- Make it required for new entries
ALTER TABLE roi_configs ALTER COLUMN restaurant_id SET NOT NULL;

-- Update unique constraint to include restaurant
ALTER TABLE roi_configs DROP CONSTRAINT IF EXISTS _roi_uc;
ALTER TABLE roi_configs ADD CONSTRAINT _roi_uc 
    UNIQUE(restaurant_id, channel_id, app_name);

-- Index
CREATE INDEX idx_roi_configs_restaurant ON roi_configs(restaurant_id);
```

---

### **1.5 Update Other Tables**

Add restaurant_id to all detection/logging tables:

```sql
-- detections table
ALTER TABLE detections ADD COLUMN restaurant_id INTEGER REFERENCES restaurants(id);
CREATE INDEX idx_detections_restaurant ON detections(restaurant_id);

-- daily_footfall table
ALTER TABLE daily_footfall ADD COLUMN restaurant_id INTEGER REFERENCES restaurants(id);
CREATE INDEX idx_daily_footfall_restaurant ON daily_footfall(restaurant_id);

-- hourly_footfall table
ALTER TABLE hourly_footfall ADD COLUMN restaurant_id INTEGER REFERENCES restaurants(id);
CREATE INDEX idx_hourly_footfall_restaurant ON hourly_footfall(restaurant_id);

-- queue_logs table
ALTER TABLE queue_logs ADD COLUMN restaurant_id INTEGER REFERENCES restaurants(id);
CREATE INDEX idx_queue_logs_restaurant ON queue_logs(restaurant_id);

-- kitchen_violations table
ALTER TABLE kitchen_violations ADD COLUMN restaurant_id INTEGER REFERENCES restaurants(id);
CREATE INDEX idx_kitchen_violations_restaurant ON kitchen_violations(restaurant_id);

-- occupancy_logs table
ALTER TABLE occupancy_logs ADD COLUMN restaurant_id INTEGER REFERENCES restaurants(id);
CREATE INDEX idx_occupancy_logs_restaurant ON occupancy_logs(restaurant_id);

-- occupancy_schedules table
ALTER TABLE occupancy_schedules ADD COLUMN restaurant_id INTEGER REFERENCES restaurants(id);
CREATE INDEX idx_occupancy_schedules_restaurant ON occupancy_schedules(restaurant_id);
```

---

## 🔧 **Phase 2: Backend Refactoring** (Priority: HIGH)

### **2.1 Create Database Models**

Add new SQLAlchemy models in `edit-004.py` or separate `models.py`:

```python
class Restaurant(Base):
    __tablename__ = "restaurants"
    id = Column(Integer, primary_key=True)
    restaurant_code = Column(String(50), unique=True, nullable=False)
    restaurant_name = Column(String(200), nullable=False)
    location = Column(String(200))
    dvr_ip = Column(String(50))
    dvr_username = Column(String(100))
    dvr_password = Column(String(100))
    telegram_chat_id = Column(String(50))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))
    updated_at = Column(DateTime, default=lambda: datetime.now(IST))

class Camera(Base):
    __tablename__ = "cameras"
    id = Column(Integer, primary_key=True)
    restaurant_id = Column(Integer, ForeignKey('restaurants.id'))
    channel_number = Column(Integer, nullable=False)
    channel_name = Column(String(100), nullable=False)
    subtype = Column(Integer, default=0)
    rtsp_url = Column(Text)
    channel_id = Column(String(100), unique=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))
    updated_at = Column(DateTime, default=lambda: datetime.now(IST))

class CameraApp(Base):
    __tablename__ = "camera_apps"
    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey('cameras.id'))
    app_name = Column(String(50), nullable=False)
    is_active = Column(Boolean, default=True)
    config = Column(JSONB)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))
```

---

### **2.2 Replace `get_app_configs()` Function**

Refactor to load from database instead of `rtsp_links.txt`:

```python
def get_app_configs(restaurant_id=None):
    """Get application configs, optionally filtered by restaurant"""
    app_configs = defaultdict(lambda: {'channels': [], 'online_count': 0})
    
    if not db_connected:
        return {}
    
    with SessionLocal() as db:
        # Build query
        query = db.query(Camera, CameraApp, Restaurant).\\
            join(CameraApp, Camera.id == CameraApp.camera_id).\\
            join(Restaurant, Camera.restaurant_id == Restaurant.id).\\
            filter(Camera.is_active == True, CameraApp.is_active == True)
        
        # Filter by restaurant if specified
        if restaurant_id:
            query = query.filter(Camera.restaurant_id == restaurant_id)
        
        results = query.all()
        
        # Build app configs structure
        for camera, camera_app, restaurant in results:
            app_name = camera_app.app_name
            
            # Check if channel is online
            processors = stream_processors.get(camera.channel_id, [])
            is_alive = any(p.is_alive() for p in processors) if processors else False
            
            # Add channel to app config
            if not any(d['id'] == camera.channel_id for d in app_configs[app_name]['channels']):
                app_configs[app_name]['channels'].append({
                    'id': camera.channel_id,
                    'name': camera.channel_name,
                    'restaurant_id': restaurant.id,
                    'restaurant_name': restaurant.restaurant_name,
                    'is_alive': is_alive
                })
        
        # Calculate online counts
        for app_name, config in app_configs.items():
            online_count = sum(1 for ch in config['channels'] if ch.get('is_alive', False))
            config['online_count'] = online_count
    
    return dict(app_configs)
```

---

### **2.3 Create Restaurant Management API**

Add REST API endpoints for restaurant management:

```python
@app.route('/api/restaurants', methods=['GET'])
@login_required
def get_restaurants():
    """Get list of all restaurants"""
    with SessionLocal() as db:
        restaurants = db.query(Restaurant).filter_by(is_active=True).all()
        return jsonify([{
            'id': r.id,
            'code': r.restaurant_code,
            'name': r.restaurant_name,
            'location': r.location
        } for r in restaurants])

@app.route('/api/restaurants/<int:restaurant_id>/cameras', methods=['GET'])
@login_required
def get_restaurant_cameras(restaurant_id):
    """Get all cameras for a specific restaurant"""
    with SessionLocal() as db:
        cameras = db.query(Camera, CameraApp).\\
            join(CameraApp, Camera.id == CameraApp.camera_id).\\
            filter(Camera.restaurant_id == restaurant_id).\\
            filter(Camera.is_active == True).all()
        
        return jsonify([{
            'id': cam.id,
            'name': cam.channel_name,
            'channel_id': cam.channel_id,
            'apps': [app.app_name for _, app in cameras if _.id == cam.id]
        } for cam, _ in cameras])
```

---

### **2.4 Update `start_streams()` Function**

Modify to load cameras from database instead of file:

```python
def start_streams():
    """Initialize all video stream processors from database"""
    logging.info("Initializing stream processors from database...")
    
    if not db_connected:
        logging.error("Database not connected. Cannot start streams.")
        return
    
    with SessionLocal() as db:
        # Get all active cameras with their apps
        cameras = db.query(Camera, Restaurant).\\
            join(Restaurant, Camera.restaurant_id == Restaurant.id).\\
            filter(Camera.is_active == True, Restaurant.is_active == True).all()
        
        for camera, restaurant in cameras:
            # Generate RTSP URL
            rtsp_url = f"rtsp://{restaurant.dvr_username}:{restaurant.dvr_password}@" \\
                      f"{restaurant.dvr_ip}:554/cam/realmonitor?" \\
                      f"channel={camera.channel_number}&subtype={camera.subtype}"
            
            # Update camera with generated URL
            camera.rtsp_url = rtsp_url
            if not camera.channel_id:
                camera.channel_id = get_stable_channel_id(rtsp_url)
            db.commit()
            
            # Start FrameHub
            hub = FrameHub(rtsp_url, camera.channel_name)
            frame_hubs[camera.channel_id] = hub
            hub.start()
            
            # Get apps for this camera
            apps = db.query(CameraApp).\\
                filter_by(camera_id=camera.id, is_active=True).all()
            
            # Start processors for each app
            for camera_app in apps:
                app_name = camera_app.app_name
                # ... (existing processor initialization logic)
```

---

## 🎨 **Phase 3: Frontend Implementation** (Priority: HIGH)

### **3.1 Update Dashboard HTML**

Add restaurant dropdown to `templates/dashboard.html`:

```html
<!-- Restaurant Selector -->
<div class="restaurant-selector">
    <label for="restaurant-select">🏢 Select Restaurant:</label>
    <select id="restaurant-select" class="form-control">
        <option value="all">All Restaurants</option>
        <!-- Populated via AJAX -->
    </select>
</div>

<script>
// Load restaurants on page load
async function loadRestaurants() {
    const response = await fetch('/api/restaurants');
    const restaurants = await response.json();
    
    const select = document.getElementById('restaurant-select');
    restaurants.forEach(restaurant => {
        const option = document.createElement('option');
        option.value = restaurant.id;
        option.textContent = `${restaurant.name} - ${restaurant.location}`;
        select.appendChild(option);
    });
    
    // Load saved selection from localStorage
    const saved = localStorage.getItem('selected_restaurant');
    if (saved) {
        select.value = saved;
        filterByRestaurant(saved);
    }
}

// Filter dashboard by restaurant
function filterByRestaurant(restaurantId) {
    // Save selection
    localStorage.setItem('selected_restaurant', restaurantId);
    
    // Reload dashboard with filter
    if (restaurantId === 'all') {
        location.href = '/dashboard';
    } else {
        location.href = `/dashboard?restaurant_id=${restaurantId}`;
    }
}

// Event listener
document.getElementById('restaurant-select').addEventListener('change', (e) => {
    filterByRestaurant(e.target.value);
});

// Initialize
loadRestaurants();
</script>
```

---

### **3.2 Update Dashboard Route**

Modify Flask route to handle restaurant filtering:

```python
@app.route('/dashboard')
@login_required
def dashboard():
    restaurant_id = request.args.get('restaurant_id', type=int)
    
    # Get restaurant info
    restaurant_info = None
    if restaurant_id:
        with SessionLocal() as db:
            restaurant = db.query(Restaurant).filter_by(id=restaurant_id).first()
            if restaurant:
                restaurant_info = {
                    'id': restaurant.id,
                    'name': restaurant.restaurant_name,
                    'location': restaurant.location
                }
    
    # Get app configs filtered by restaurant
    app_configs = get_app_configs(restaurant_id=restaurant_id)
    
    return render_template('dashboard.html', 
                         app_configs=app_configs,
                         selected_restaurant=restaurant_info)
```

---

## 🔐 **Phase 4: Security & Configuration** (Priority: MEDIUM)

### **4.1 Encrypt Sensitive Data**

Store DVR passwords encrypted:

```python
from cryptography.fernet import Fernet
import os

# Generate encryption key (store securely, e.g., environment variable)
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', Fernet.generate_key())
cipher = Fernet(ENCRYPTION_KEY)

def encrypt_password(password: str) -> str:
    return cipher.encrypt(password.encode()).decode()

def decrypt_password(encrypted: str) -> str:
    return cipher.decrypt(encrypted.encode()).decode()
```

---

### **4.2 Create Migration Script**

Migrate existing data from `rtsp_links.txt` to database:

```python
# migrate_to_database.py
def migrate_rtsp_links_to_db():
    """Migrate rtsp_links.txt to database"""
    
    # Create Tea Toast restaurant (existing)
    with SessionLocal() as db:
        restaurant = Restaurant(
            restaurant_code='tea_toast',
            restaurant_name='Tea Toast - Brigade Road',
            location='Brigade Road, Bangalore',
            dvr_ip='182.65.205.121',
            dvr_username='admin',
            dvr_password='cctv#1234',
            telegram_chat_id='-4835836048'
        )
        db.add(restaurant)
        db.commit()
        
        # Parse rtsp_links.txt
        with open('rtsp_links.txt', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = [p.strip() for p in line.split(',')]
                    # ... (parse and create Camera + CameraApp entries)
```

---

## 📊 **Phase 5: Testing & Deployment** (Priority: MEDIUM)

### **5.1 Testing Checklist**

- [ ] Create test restaurants in database
- [ ] Add test cameras with different DVR IPs
- [ ] Verify RTSP URL generation
- [ ] Test restaurant dropdown switching
- [ ] Verify ROI configs load correctly per restaurant
- [ ] Test multi-tenant data isolation
- [ ] Performance test with 3+ restaurants
- [ ] Test Telegram alerts per restaurant

---

### **5.2 Deployment Steps**

1. **Backup current system**
   ```bash
   pg_dump sakshi > sakshi_backup_$(date +%Y%m%d).sql
   cp rtsp_links.txt rtsp_links.txt.backup
   ```

2. **Run database migrations**
   ```bash
   python3 migrate_schema.py
   python3 migrate_data.py
   ```

3. **Test with single restaurant** (backward compatibility)

4. **Add second restaurant** (gradual rollout)

5. **Monitor logs and performance**

---

## 📈 **Phase 6: Advanced Features** (Priority: LOW)

### **6.1 Restaurant Dashboard**
- Per-restaurant analytics
- Restaurant comparison view
- Multi-restaurant reports

### **6.2 Restaurant-Specific Settings**
- Custom alert thresholds per restaurant
- Telegram channels per location
- Business hours configuration

### **6.3 Role-Based Access**
- Restaurant managers (single location access)
- Regional managers (multiple locations)
- Super admin (all locations)

---

## ⏱️ **Implementation Timeline**

| Phase | Estimated Time | Dependencies |
|-------|---------------|--------------|
| Phase 1: Database Schema | 1-2 days | None |
| Phase 2: Backend Refactoring | 3-4 days | Phase 1 |
| Phase 3: Frontend Implementation | 2-3 days | Phase 2 |
| Phase 4: Security & Config | 1-2 days | Phase 2 |
| Phase 5: Testing & Deployment | 2-3 days | Phase 1-4 |
| Phase 6: Advanced Features | 3-5 days | Phase 5 |
| **Total** | **12-19 days** | |

---

## 🎯 **Success Criteria**

✅ **Must Have:**
- [ ] Restaurant dropdown in dashboard
- [ ] Cameras load based on selected restaurant
- [ ] ROI configs tied to restaurant
- [ ] All existing features work per restaurant
- [ ] Zero downtime migration

✅ **Should Have:**
- [ ] Restaurant management UI
- [ ] Per-restaurant Telegram alerts
- [ ] Data isolation between restaurants

✅ **Nice to Have:**
- [ ] Restaurant comparison analytics
- [ ] Role-based access control
- [ ] Mobile app support

---

## 🚧 **Risk Mitigation**

| Risk | Mitigation |
|------|------------|
| Data loss during migration | Full backup + test migration on copy |
| Performance degradation | Database indexing + query optimization |
| Breaking existing integrations | Backward compatibility layer |
| Complex RTSP URL variations | URL builder with validation |
| ROI config conflicts | Restaurant-scoped unique constraints |

---

## 📝 **Next Steps**

**Immediate Actions:**
1. Review and approve roadmap
2. Set up development database
3. Create Phase 1 migration scripts
4. Begin schema implementation

**Quick Win (MVP - 3 days):**
- Implement Phase 1 (database schema)
- Create basic migration from rtsp_links.txt
- Add simple restaurant dropdown (hardcoded options)
- Test with 2 restaurants

This approach ensures a **smooth, incremental rollout** with minimal disruption to the existing Tea Toast operations! 🚀
