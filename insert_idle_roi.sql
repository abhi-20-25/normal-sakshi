-- Insert idle zone ROI for QueueMonitor camera
INSERT INTO roi_configs (channel_id, app_name, roi_points, restaurant_id)
VALUES (
    'cam_f822b0bf4e',  -- QueueMonitor camera channel_id
    'IdleTimeMonitor',
    '{"points": [[0.8243749976158142, 0.28444442749023435], [0.7259374976158142, 0.5649999830457899], [0.8399999976158142, 0.7344444394111633], [0.9384374976158142, 0.4777777712290287]]}',
    1
)
ON CONFLICT (channel_id, app_name) 
DO UPDATE SET roi_points = EXCLUDED.roi_points;
