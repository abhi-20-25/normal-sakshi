-- Setup Idle Time Monitor for Restaurant 2
-- Add IdleTimeMonitor app to camera_apps for Front Office Violation camera

-- First, add the app to camera_apps
INSERT INTO camera_apps (camera_id, app_name, is_active, config)
VALUES (
    7,  -- Front Office Violation camera
    'IdleTimeMonitor',
    true,
    '{"dwell_threshold": 300, "cooldown": 600}'::jsonb
)
ON CONFLICT DO NOTHING;

-- Insert idle zone ROI for Front Office camera
-- This creates a sample zone - you'll need to adjust coordinates via the web UI
INSERT INTO roi_configs (channel_id, app_name, roi_points, restaurant_id)
VALUES (
    'cam_8345d279a3ad',  -- Front Office Violation camera channel_id
    'IdleTimeMonitor',
    '{"zones": [{"name": "counter_area", "points": [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]]}]}',
    2
)
ON CONFLICT (channel_id, app_name) 
DO UPDATE SET 
    roi_points = EXCLUDED.roi_points,
    restaurant_id = EXCLUDED.restaurant_id;

-- Verify the setup
SELECT 
    c.id, 
    c.channel_id, 
    c.channel_name, 
    ca.app_name,
    rc.app_name as roi_app,
    rc.roi_points
FROM cameras c
LEFT JOIN camera_apps ca ON c.id = ca.camera_id
LEFT JOIN roi_configs rc ON c.channel_id = rc.channel_id
WHERE c.restaurant_id = 2 AND (ca.app_name = 'IdleTimeMonitor' OR rc.app_name = 'IdleTimeMonitor');
