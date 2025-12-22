-- ============================================================================
-- PHASE 1: Multi-Restaurant Database Schema
-- ============================================================================
-- This script creates the necessary tables for multi-restaurant support
-- Run this on the 'sakshi' database

BEGIN;

-- ============================================================================
-- 1. CREATE RESTAURANTS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS restaurants (
    id SERIAL PRIMARY KEY,
    restaurant_code VARCHAR(50) UNIQUE NOT NULL,
    restaurant_name VARCHAR(200) NOT NULL,
    location VARCHAR(200),
    dvr_ip VARCHAR(50),
    dvr_username VARCHAR(100),
    dvr_password VARCHAR(100),
    telegram_chat_id VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_restaurants_code ON restaurants(restaurant_code);
CREATE INDEX IF NOT EXISTS idx_restaurants_active ON restaurants(is_active);

COMMENT ON TABLE restaurants IS 'Stores information about each restaurant location';
COMMENT ON COLUMN restaurants.restaurant_code IS 'Unique code identifier (e.g., tea_toast)';
COMMENT ON COLUMN restaurants.dvr_ip IS 'DVR IP address for RTSP streams';

-- ============================================================================
-- 2. CREATE CAMERAS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS cameras (
    id SERIAL PRIMARY KEY,
    restaurant_id INTEGER REFERENCES restaurants(id) ON DELETE CASCADE,
    channel_number INTEGER NOT NULL,
    channel_name VARCHAR(100) NOT NULL,
    subtype INTEGER DEFAULT 0,
    rtsp_url TEXT,
    channel_id VARCHAR(100) UNIQUE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT unique_restaurant_channel UNIQUE(restaurant_id, channel_number, subtype)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_cameras_restaurant ON cameras(restaurant_id);
CREATE INDEX IF NOT EXISTS idx_cameras_channel_id ON cameras(channel_id);
CREATE INDEX IF NOT EXISTS idx_cameras_active ON cameras(is_active);

COMMENT ON TABLE cameras IS 'Stores camera configurations for each restaurant';
COMMENT ON COLUMN cameras.channel_number IS 'DVR channel number (1-32)';
COMMENT ON COLUMN cameras.subtype IS '0=main stream, 1=sub stream';
COMMENT ON COLUMN cameras.channel_id IS 'Hash-based unique identifier for internal use';

-- ============================================================================
-- 3. CREATE CAMERA_APPS TABLE (Many-to-Many)
-- ============================================================================
CREATE TABLE IF NOT EXISTS camera_apps (
    id SERIAL PRIMARY KEY,
    camera_id INTEGER REFERENCES cameras(id) ON DELETE CASCADE,
    app_name VARCHAR(50) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    config JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT unique_camera_app UNIQUE(camera_id, app_name)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_camera_apps_camera ON camera_apps(camera_id);
CREATE INDEX IF NOT EXISTS idx_camera_apps_app ON camera_apps(app_name);

COMMENT ON TABLE camera_apps IS 'Links cameras to AI applications (PeopleCounter, QueueMonitor, etc)';
COMMENT ON COLUMN camera_apps.config IS 'App-specific configuration as JSON';

-- ============================================================================
-- 4. UPDATE EXISTING TABLES - ADD RESTAURANT_ID
-- ============================================================================

-- Add restaurant_id to roi_configs
ALTER TABLE roi_configs ADD COLUMN IF NOT EXISTS restaurant_id INTEGER REFERENCES restaurants(id);
CREATE INDEX IF NOT EXISTS idx_roi_configs_restaurant ON roi_configs(restaurant_id);

-- Add restaurant_id to detections
ALTER TABLE detections ADD COLUMN IF NOT EXISTS restaurant_id INTEGER REFERENCES restaurants(id);
CREATE INDEX IF NOT EXISTS idx_detections_restaurant ON detections(restaurant_id);

-- Add restaurant_id to daily_footfall
ALTER TABLE daily_footfall ADD COLUMN IF NOT EXISTS restaurant_id INTEGER REFERENCES restaurants(id);
CREATE INDEX IF NOT EXISTS idx_daily_footfall_restaurant ON daily_footfall(restaurant_id);

-- Add restaurant_id to hourly_footfall
ALTER TABLE hourly_footfall ADD COLUMN IF NOT EXISTS restaurant_id INTEGER REFERENCES restaurants(id);
CREATE INDEX IF NOT EXISTS idx_hourly_footfall_restaurant ON hourly_footfall(restaurant_id);

-- Add restaurant_id to queue_logs
ALTER TABLE queue_logs ADD COLUMN IF NOT EXISTS restaurant_id INTEGER REFERENCES restaurants(id);
CREATE INDEX IF NOT EXISTS idx_queue_logs_restaurant ON queue_logs(restaurant_id);

-- Add restaurant_id to kitchen_violations
ALTER TABLE kitchen_violations ADD COLUMN IF NOT EXISTS restaurant_id INTEGER REFERENCES restaurants(id);
CREATE INDEX IF NOT EXISTS idx_kitchen_violations_restaurant ON kitchen_violations(restaurant_id);

-- Add restaurant_id to occupancy_logs
ALTER TABLE occupancy_logs ADD COLUMN IF NOT EXISTS restaurant_id INTEGER REFERENCES restaurants(id);
CREATE INDEX IF NOT EXISTS idx_occupancy_logs_restaurant ON occupancy_logs(restaurant_id);

-- Add restaurant_id to occupancy_schedules
ALTER TABLE occupancy_schedules ADD COLUMN IF NOT EXISTS restaurant_id INTEGER REFERENCES restaurants(id);
CREATE INDEX IF NOT EXISTS idx_occupancy_schedules_restaurant ON occupancy_schedules(restaurant_id);

-- ============================================================================
-- 5. INSERT DEFAULT RESTAURANT (Tea Toast)
-- ============================================================================
INSERT INTO restaurants (
    restaurant_code, 
    restaurant_name, 
    location, 
    dvr_ip, 
    dvr_username, 
    dvr_password, 
    telegram_chat_id,
    is_active
)
VALUES (
    'tea_toast',
    'Tea Toast - Brigade Road',
    'Brigade Road, Bangalore',
    '182.65.205.121',
    'admin',
    'cctv#1234',
    '-4835836048',
    TRUE
)
ON CONFLICT (restaurant_code) DO NOTHING;

-- ============================================================================
-- 6. UPDATE EXISTING DATA TO REFERENCE TEA TOAST
-- ============================================================================

-- Update roi_configs to reference Tea Toast
UPDATE roi_configs 
SET restaurant_id = (SELECT id FROM restaurants WHERE restaurant_code = 'tea_toast')
WHERE restaurant_id IS NULL;

-- Update detections to reference Tea Toast
UPDATE detections 
SET restaurant_id = (SELECT id FROM restaurants WHERE restaurant_code = 'tea_toast')
WHERE restaurant_id IS NULL;

-- Update daily_footfall to reference Tea Toast
UPDATE daily_footfall 
SET restaurant_id = (SELECT id FROM restaurants WHERE restaurant_code = 'tea_toast')
WHERE restaurant_id IS NULL;

-- Update hourly_footfall to reference Tea Toast
UPDATE hourly_footfall 
SET restaurant_id = (SELECT id FROM restaurants WHERE restaurant_code = 'tea_toast')
WHERE restaurant_id IS NULL;

-- Update queue_logs to reference Tea Toast
UPDATE queue_logs 
SET restaurant_id = (SELECT id FROM restaurants WHERE restaurant_code = 'tea_toast')
WHERE restaurant_id IS NULL;

-- Update kitchen_violations to reference Tea Toast
UPDATE kitchen_violations 
SET restaurant_id = (SELECT id FROM restaurants WHERE restaurant_code = 'tea_toast')
WHERE restaurant_id IS NULL;

-- Update occupancy_logs to reference Tea Toast
UPDATE occupancy_logs 
SET restaurant_id = (SELECT id FROM restaurants WHERE restaurant_code = 'tea_toast')
WHERE restaurant_id IS NULL;

-- Update occupancy_schedules to reference Tea Toast
UPDATE occupancy_schedules 
SET restaurant_id = (SELECT id FROM restaurants WHERE restaurant_code = 'tea_toast')
WHERE restaurant_id IS NULL;

-- ============================================================================
-- 7. VERIFICATION QUERIES
-- ============================================================================

-- Show created tables
SELECT 
    'Restaurants' as table_name, 
    COUNT(*) as row_count 
FROM restaurants
UNION ALL
SELECT 'Cameras', COUNT(*) FROM cameras
UNION ALL
SELECT 'Camera Apps', COUNT(*) FROM camera_apps;

COMMIT;

-- Display success message
DO $$
BEGIN
    RAISE NOTICE '✅ Phase 1 Schema Created Successfully!';
    RAISE NOTICE '📊 Tables created: restaurants, cameras, camera_apps';
    RAISE NOTICE '🔗 Existing tables updated with restaurant_id';
    RAISE NOTICE '🏢 Default restaurant (Tea Toast) inserted';
    RAISE NOTICE '';
    RAISE NOTICE '⚠️  Next Steps:';
    RAISE NOTICE '   1. Run phase1_migrate_data.py to migrate rtsp_links.txt';
    RAISE NOTICE '   2. Verify data with: SELECT * FROM restaurants;';
    RAISE NOTICE '   3. Backup database before proceeding to Phase 2';
END $$;
