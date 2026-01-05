-- Sync AWS roi_configs table with local database
-- This script will update existing rows and insert missing ones

-- Update existing rows with restaurant_id
UPDATE roi_configs SET restaurant_id = 2 WHERE id = 1;
UPDATE roi_configs SET restaurant_id = 2 WHERE id = 2;

-- Insert missing ROI configs
INSERT INTO roi_configs (id, channel_id, app_name, roi_points, restaurant_id) 
VALUES (
    5, 
    'cam_e13961bdb0c0', 
    'PeopleCounter', 
    '{"line_position": 0.44}',
    1
) ON CONFLICT (id) DO UPDATE SET 
    channel_id = EXCLUDED.channel_id,
    app_name = EXCLUDED.app_name,
    roi_points = EXCLUDED.roi_points,
    restaurant_id = EXCLUDED.restaurant_id;

INSERT INTO roi_configs (id, channel_id, app_name, roi_points, restaurant_id) 
VALUES (
    8, 
    'cam_17f5ffe2df24', 
    'QueueMonitor', 
    '{"main": [[0.1862499952316284, 0.13388883802625867], [0.6674999952316284, 0.5005555046929253], [0.4112499952316284, 0.8699999491373698], [0.06124999523162842, 0.3061110602484809]], "secondary": [[0.18156249523162843, 0.007777743869357639], [0.8018749952316284, 0.016111077202690973], [0.6487499952316285, 0.44666663275824653], [0.1784374952316284, 0.07722218831380208]]}',
    1
) ON CONFLICT (id) DO UPDATE SET 
    channel_id = EXCLUDED.channel_id,
    app_name = EXCLUDED.app_name,
    roi_points = EXCLUDED.roi_points,
    restaurant_id = EXCLUDED.restaurant_id;

-- Reset sequence to prevent ID conflicts in future inserts
SELECT setval('roi_configs_id_seq', (SELECT MAX(id) FROM roi_configs));
