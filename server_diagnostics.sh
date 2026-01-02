#!/bin/bash
# Server Diagnostics Script for normal-sakshi

echo "=============================================="
echo "🔍 SERVER DIAGNOSTICS REPORT"
echo "=============================================="
echo "Date: $(date)"
echo ""

# 1. DISK USAGE ANALYSIS
echo "📊 DISK USAGE ANALYSIS"
echo "----------------------------------------------"
df -h /
echo ""

echo "🔍 Top 10 Largest Directories:"
du -h --max-depth=2 ~/normal-sakshi 2>/dev/null | sort -rh | head -10
echo ""

echo "🔍 Large Log Files:"
find ~/normal-sakshi -name "*.log" -type f -size +10M -exec ls -lh {} \; 2>/dev/null | head -10
echo ""

echo "🔍 Video/Media Files:"
find ~/normal-sakshi -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.jpg" -o -name "*.png" \) -size +10M 2>/dev/null | wc -l
echo "files found larger than 10MB"
echo ""

# 2. RUNNING SERVICES
echo "🔧 RUNNING SERVICES"
echo "----------------------------------------------"
echo "Sakshi AI Service:"
sudo systemctl status sakshi-ai.service --no-pager | head -15
echo ""

echo "PetPooja Service:"
sudo systemctl status petpooja-api.service --no-pager | head -15
echo ""

# 3. PROCESS MONITORING
echo "⚡ PROCESS INFORMATION"
echo "----------------------------------------------"
echo "Python processes:"
ps aux | grep python | grep -v grep | head -10
echo ""

echo "Memory Usage:"
free -h
echo ""

echo "CPU Load:"
uptime
echo ""

# 4. GUNICORN STATUS
echo "🌐 GUNICORN STATUS"
echo "----------------------------------------------"
if pgrep -f gunicorn > /dev/null; then
    echo "✅ Gunicorn is running"
    ps aux | grep gunicorn | grep -v grep | head -5
else
    echo "❌ Gunicorn is NOT running"
fi
echo ""

# 5. LOG FILE SIZES
echo "📝 LOG FILE SIZES"
echo "----------------------------------------------"
if [ -d ~/normal-sakshi/logs ]; then
    ls -lh ~/normal-sakshi/logs/ | head -20
else
    echo "No logs directory found"
fi
echo ""

# 6. DETECTION VIDEOS
echo "🎥 DETECTION VIDEOS/IMAGES"
echo "----------------------------------------------"
if [ -d ~/normal-sakshi/static/detections ]; then
    echo "Total files in detections:"
    find ~/normal-sakshi/static/detections -type f | wc -l
    echo ""
    echo "Total size of detections:"
    du -sh ~/normal-sakshi/static/detections
fi
echo ""

# 7. DATABASE CONNECTION
echo "🗄️  DATABASE STATUS"
echo "----------------------------------------------"
if command -v psql &> /dev/null; then
    echo "PostgreSQL client available"
else
    echo "PostgreSQL client not found"
fi
echo ""

# 8. RECENT ERRORS
echo "❌ RECENT ERRORS (Last 20 lines)"
echo "----------------------------------------------"
if [ -f ~/normal-sakshi/fastapi_app.log ]; then
    echo "FastAPI errors:"
    tail -20 ~/normal-sakshi/fastapi_app.log
fi
echo ""

# 9. SYSTEM UPDATES
echo "🔄 SYSTEM STATUS"
echo "----------------------------------------------"
echo "Pending updates:"
apt list --upgradable 2>/dev/null | head -10
echo ""

echo "Restart required:"
if [ -f /var/run/reboot-required ]; then
    echo "⚠️  YES - System restart is required"
    cat /var/run/reboot-required.pkgs 2>/dev/null | head -10
else
    echo "✅ No restart required"
fi
echo ""

echo "=============================================="
echo "✅ DIAGNOSTICS COMPLETE"
echo "=============================================="
