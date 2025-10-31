# Gunicorn Troubleshooting Guide

## Issue: App loads in browser but nothing comes / video feed doesn't work

### Quick Checks

1. **Check if Gunicorn is running:**
   ```bash
   sudo systemctl status sakshi-ai
   ```

2. **Check logs for initialization:**
   ```bash
   sudo journalctl -u sakshi-ai -n 100 | grep -i "initialized\|processor\|error"
   ```
   
   You should see:
   - `Starting application initialization...`
   - `✓ Application initialized successfully`
   - `Total processors started: X across Y channels`

3. **Check if processors are actually running:**
   ```bash
   ps aux | grep -E "PeopleCounter|QueueMonitor|FrameHub" | grep -v grep
   ```
   
   You should see thread processes running.

4. **Run diagnostic script:**
   ```bash
   python3 check_gunicorn.py
   ```

### Common Issues

#### Issue 1: Initialization Not Happening

**Symptoms:** App loads but no video feed, no processors running

**Check:**
```bash
sudo journalctl -u sakshi-ai | grep "Application initialized"
```

**If not found:**
- Check for errors in logs: `sudo journalctl -u sakshi-ai | grep -i error`
- Verify RTSP links file exists: `ls -la /home/ubuntu/normal-sakshi/rtsp_links.txt`
- Check database connection

**Fix:** Make sure `wsgi.py` properly imports and initializes the module.

#### Issue 2: Processors Not Starting

**Symptoms:** App loads, initialization logged, but no video feed

**Check:**
```bash
# Check if FrameHub threads are running
ps aux | grep FrameHub

# Check logs for processor start messages
sudo journalctl -u sakshi-ai | grep "Started.*for"
```

**Common causes:**
- RTSP streams not accessible from server
- Model files not found
- CUDA errors preventing processor start

**Fix:** 
- Test RTSP from server: `ffmpeg -i "rtsp://..." -frames:v 1 test.jpg`
- Check model files exist
- Check CUDA status: `curl http://localhost:5001/api/cuda_status`

#### Issue 3: Video Feed Returns Nothing

**Symptoms:** Browser shows loading but no video

**Check:**
```bash
# Test video feed directly
curl -I http://localhost:5001/video_feed/PeopleCounter/cam_3df702bb28

# Should return 302 (redirect to login) or 200 if authenticated
```

**Common causes:**
- Processors not initialized
- Processor threads not alive
- FrameHub not getting frames
- Login required (check if authenticated)

**Fix:**
- Ensure logged in
- Check processor is alive: Look for "Streaming video feed" in logs
- Verify channel_id matches: Check `stream_processors` dict

#### Issue 4: RTSP Not Accessible

**Question:** Will RTSP be accessible over gunicorn service?

**Answer:** YES, but:
- RTSP streams are accessed by the **server**, not the browser
- Processors run in background threads on the server
- They connect to RTSP cameras and process frames
- Browser receives processed video over HTTP (not RTSP)

**To verify RTSP accessibility:**
```bash
# From your server, test RTSP connection
ffmpeg -i "rtsp://admin:cctv%231234@182.65.205.121:554/cam/realmonitor?channel=1&subtype=1" -frames:v 1 -y test.jpg

# If this works, RTSP is accessible
# If it fails, check network/firewall
```

#### Issue 5: Workers Not Handling Async Properly

**Symptoms:** Everything seems OK but video feed hangs

**Check worker type:**
```bash
ps aux | grep gunicorn | grep -E "eventlet|gevent"
```

**Must use async workers (eventlet or gevent), not sync workers.**

**Fix:** Ensure service file uses:
```ini
ExecStart=... --worker-class eventlet ...
```

### Debugging Steps

1. **Stop the service:**
   ```bash
   sudo systemctl stop sakshi-ai
   ```

2. **Run manually to see output:**
   ```bash
   cd /home/ubuntu/normal-sakshi
   source venv/bin/activate
   python3 -c "from wsgi import application; print('WSGI OK')"
   ```

3. **Check initialization:**
   ```bash
   python3 -c "import sys; sys.path.insert(0, '.'); from edit_004 import stream_processors; print('Processors:', stream_processors)"
   ```

4. **Test video feed locally:**
   ```bash
   gunicorn --worker-class eventlet --workers 1 --bind 127.0.0.1:5001 wsgi:application
   ```

5. **Check logs in real-time:**
   ```bash
   sudo journalctl -u sakshi-ai -f
   ```

### Expected Log Messages

When working correctly, you should see:

```
Loading application from: /home/ubuntu/normal-sakshi/edit-004.py
Module loaded successfully
Starting application initialization...
CUDA recovery scheduler started - will attempt to re-enable CUDA every 5 minutes
Started PeopleCounter for cam_xxx (Main Entrance).
Started QueueMonitor for cam_xxx (Checkout Queue).
✓ Application initialized successfully - processors and scheduler started
Total processors started: 3 across 2 channels
WSGI application ready
```

### Still Not Working?

1. **Check all logs:**
   ```bash
   sudo journalctl -u sakshi-ai > /tmp/gunicorn_logs.txt
   cat /tmp/gunicorn_logs.txt
   ```

2. **Verify RTSP accessibility:**
   ```bash
   # Test each RTSP URL from your server
   while IFS= read -r line; do
     if [[ ! $line =~ ^# ]] && [[ -n $line ]]; then
       rtsp=$(echo $line | cut -d',' -f1)
       echo "Testing: $rtsp"
       timeout 5 ffmpeg -i "$rtsp" -frames:v 1 -y /tmp/test.jpg 2>&1 | head -5
     fi
   done < rtsp_links.txt
   ```

3. **Check system resources:**
   ```bash
   # Check memory
   free -h
   
   # Check CPU
   top
   
   # Check disk space
   df -h
   ```

### Quick Test Commands

```bash
# 1. Service status
sudo systemctl status sakshi-ai

# 2. Recent logs
sudo journalctl -u sakshi-ai -n 50

# 3. Test endpoint (after login)
curl http://localhost:5001/api/cuda_status

# 4. Check processes
ps aux | grep -E "gunicorn|python.*edit-004"

# 5. Check port
netstat -tuln | grep 5001
```

