-- View current Queue Monitor ROI configuration
SELECT 
    id,
    channel_id,
    app_name,
    roi_points
FROM roi_configs
WHERE app_name = 'QueueMonitor';

-- To update the ROI, uncomment and modify the coordinates below:
/*
UPDATE roi_configs 
SET roi_points = '{
    "main": [
        [0.35, 0.40],
        [0.65, 0.40],
        [0.70, 0.80],
        [0.30, 0.80]
    ],
    "secondary": [
        [0.60, 0.30],
        [0.80, 0.30],
        [0.80, 0.60],
        [0.60, 0.60]
    ]
}'
WHERE channel_id = 'cam_f822b0bf4e' AND app_name = 'QueueMonitor';
*/

-- Verify the update
SELECT 
    channel_id,
    app_name,
    roi_points
FROM roi_configs
WHERE channel_id = 'cam_f822b0bf4e' AND app_name = 'QueueMonitor';
