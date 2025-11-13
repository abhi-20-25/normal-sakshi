# Server Deployment Guide - GPU Issues Fix

## Problem: Detection Not Working on Server

The server is experiencing GPU/CUDA issues causing detection to fail. This is common when:
- Multiple processes compete for GPU memory
- CUDA version mismatch between server and local
- GPU drivers are outdated or missing
- Server GPU has insufficient VRAM

## Solution: Force CPU Mode on Server

### Step 1: Update Server Code

Copy the latest `edit-004.py` to the server:
```bash
scp edit-004.py ubuntu@YOUR_SERVER_IP:/home/ubuntu/normal-sakshi/
```

### Step 2: Modify Service File to Force CPU Mode

On the server, edit the systemd service file:
```bash
sudo nano /etc/systemd/system/sakshi-ai.service
```

Add `FORCE_CPU=true` to the Environment section:
```ini
[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/normal-sakshi
Environment="PATH=/home/ubuntu/normal-sakshi/venv/bin"
Environment="FORCE_CPU=true"    # <--- ADD THIS LINE
ExecStart=/home/ubuntu/normal-sakshi/venv/bin/gunicorn \
    --worker-class eventlet \
    --workers 1 \
    --worker-connections 1000 \
    --bind 0.0.0.0:5001 \
    --timeout 120 \
    --graceful-timeout 30 \
    --keep-alive 5 \
    --log-level info \
    --access-logfile /var/log/gunicorn/sakshi-access.log \
    --error-logfile /var/log/gunicorn/sakshi-error.log \
    wsgi:application
```

### Step 3: Reload and Restart Service

```bash
sudo systemctl daemon-reload
sudo systemctl restart sakshi-ai.service
sudo systemctl status sakshi-ai.service
```

### Step 4: Verify CPU Mode is Active

Check the logs to confirm CPU mode:
```bash
sudo tail -f /var/log/gunicorn/sakshi-error.log | grep -i "cpu\|gpu\|cuda"
```

You should see:
```
💻 CPU mode FORCED via FORCE_CPU environment variable
```

### Step 5: Monitor Detection

Check if detection is working:
```bash
sudo tail -f /var/log/gunicorn/sakshi-error.log | grep "PeopleCounter"
```

You should see detection and counting logs.

## Alternative: Check CUDA Status via API

While the server is running, check CUDA status:
```bash
curl http://localhost:5001/api/cuda_status
```

If you see many errors, CPU mode is the best solution.

## Alternative: Reset CUDA Errors

If you want to try GPU again after fixing driver issues:
```bash
curl -X POST http://localhost:5001/api/reset_cuda
```

## Performance Comparison

**CPU Mode (Recommended for Server):**
- ✅ Stable and reliable
- ✅ No CUDA errors
- ✅ Lower memory usage
- ⚠️ Slower inference (~5-10 FPS)
- ✅ Good enough for 1-5 cameras

**GPU Mode:**
- ✅ Fast inference (~20-30 FPS)
- ⚠️ Requires proper CUDA setup
- ⚠️ Can crash with memory issues
- ⚠️ Needs GPU with 2GB+ VRAM per camera

## Troubleshooting

### If CPU mode is still slow:
1. Reduce frame resolution (already at 640x360)
2. Reduce number of cameras
3. Increase frame skip interval

### If detection still fails:
1. Check model files exist: `ls -lh /home/ubuntu/normal-sakshi/*.pt`
2. Check venv has correct packages: `source venv/bin/activate && pip list | grep torch`
3. Check logs for specific errors: `sudo tail -100 /var/log/gunicorn/sakshi-error.log`

### Common Server Issues:

**Issue: "CUDA out of memory"**
Solution: Use `FORCE_CPU=true`

**Issue: "torch._C._cuda_init()"**
Solution: CUDA drivers missing, use `FORCE_CPU=true`

**Issue: Models not loading**
Solution: Copy `.pt` files to server:
```bash
scp *.pt ubuntu@YOUR_SERVER_IP:/home/ubuntu/normal-sakshi/
```

## Monitoring Server Health

Check service status:
```bash
sudo systemctl status sakshi-ai.service
```

Check error logs:
```bash
sudo tail -50 /var/log/gunicorn/sakshi-error.log
```

Check access logs:
```bash
sudo tail -50 /var/log/gunicorn/sakshi-access.log
```

Restart if needed:
```bash
sudo systemctl restart sakshi-ai.service
```
