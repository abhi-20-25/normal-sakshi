# Gunicorn Setup Guide for Sakshi AI

## Issue with Your Current Service File

Your current service file uses `edit-004:app` which doesn't work properly with Flask-SocketIO. The issues are:

1. **Direct app reference**: Using `edit-004:app` bypasses the SocketIO WSGI wrapper needed for async operations
2. **Worker class**: `gevent` works but `eventlet` is better for Flask-SocketIO
3. **File name with hyphen**: `edit-004.py` has a hyphen which can cause import issues

## Solution

### 1. Updated Service File

Use the provided `sakshi-ai.service` file which:
- Uses `wsgi:application` (proper WSGI entry point)
- Uses `eventlet` worker class (better for SocketIO)
- Includes proper logging configuration
- Has correct timeout settings

### 2. Installation Steps

1. **Copy the service file** to systemd:
   ```bash
   sudo cp sakshi-ai.service /etc/systemd/system/
   ```

2. **Create log directories**:
   ```bash
   sudo mkdir -p /var/log/gunicorn
   sudo chown ubuntu:www-data /var/log/gunicorn
   ```

3. **Make sure eventlet is installed**:
   ```bash
   /home/ubuntu/normal-sakshi/venv/bin/pip install eventlet
   ```

4. **Reload systemd**:
   ```bash
   sudo systemctl daemon-reload
   ```

5. **Start the service**:
   ```bash
   sudo systemctl start sakshi-ai
   ```

6. **Enable on boot**:
   ```bash
   sudo systemctl enable sakshi-ai
   ```

7. **Check status**:
   ```bash
   sudo systemctl status sakshi-ai
   ```

8. **View logs**:
   ```bash
   sudo journalctl -u sakshi-ai -f
   # OR
   tail -f /var/log/gunicorn/sakshi-error.log
   ```

### 3. Key Changes from Your Original Service File

| Original | Updated | Reason |
|----------|---------|--------|
| `edit-004:app` | `wsgi:application` | Proper WSGI wrapper for SocketIO |
| `--worker-class gevent` | `--worker-class eventlet` | Better SocketIO support |
| No worker-connections | `--worker-connections 1000` | Better for video streaming |
| No log files | `--access-logfile` and `--error-logfile` | Better debugging |
| No graceful-timeout | `--graceful-timeout 30` | Cleaner shutdowns |

### 4. Testing the Setup

After starting the service, test the video feed:
```bash
curl http://localhost:5001/video_feed/PeopleCounter/cam_3df702bb28
```

You should see the multipart video stream.

### 5. Troubleshooting

**If video feed still doesn't work:**

1. **Check if processors are initialized**:
   ```bash
   sudo journalctl -u sakshi-ai | grep "Application initialized"
   ```
   You should see "Application initialized - processors and scheduler started"

2. **Check if workers are starting**:
   ```bash
   ps aux | grep gunicorn
   ```
   You should see gunicorn master and worker processes

3. **Check SocketIO compatibility**:
   ```bash
   /home/ubuntu/normal-sakshi/venv/bin/pip show eventlet
   ```
   Should show eventlet is installed

4. **Test WSGI entry point directly**:
   ```bash
   cd /home/ubuntu/normal-sakshi
   source venv/bin/activate
   python -c "from wsgi import application; print('WSGI OK')"
   ```

### 6. Alternative: If Eventlet Doesn't Work

If you prefer to use gevent, update the service file:
```ini
ExecStart=... --worker-class gevent --workers 1 ...
```

But make sure to install gevent:
```bash
/home/ubuntu/normal-sakshi/venv/bin/pip install gevent
```

### 7. Important Notes

- The `wsgi.py` file handles the hyphen in `edit-004.py` filename
- Initialization happens when the module is imported (not just in `__main__`)
- Video feed requires async workers (eventlet/gevent), not sync workers
- SocketIO websockets need proper async support

