#!/bin/bash
# AWS Deployment Verification Script

echo "=================================="
echo "AWS Deployment Verification"
echo "=================================="
echo ""

# Check Python version
echo "1. Checking Python version..."
python3 --version
echo ""

# Check if virtual environment exists
echo "2. Checking virtual environment..."
if [ -d "ttenv" ]; then
    echo "✅ Virtual environment found"
else
    echo "❌ Virtual environment NOT found - run: python3 -m venv ttenv"
    exit 1
fi
echo ""

# Activate venv and check dependencies
echo "3. Checking dependencies..."
source ttenv/bin/activate
python3 -c "import flask, cv2, torch, ultralytics, sqlalchemy, requests" 2>/dev/null && echo "✅ Core dependencies OK" || echo "❌ Missing dependencies - run: pip install -r requirements.txt"
echo ""

# Check models
echo "4. Checking model files..."
if [ -f "models/yolo11n.pt" ] && [ -f "models/kitchen_violation_30_12_2025.pt" ]; then
    echo "✅ Critical models present"
    ls -lh models/*.pt | wc -l | xargs echo "   Total models:"
else
    echo "❌ Missing critical models"
    exit 1
fi
echo ""

# Check database connectivity
echo "5. Checking database connection..."
PGPASSWORD=Tneural01 psql -U postgres -h 127.0.0.1 -d sakshi -c "SELECT 'DB_OK';" -t 2>/dev/null | grep -q DB_OK && echo "✅ Database accessible" || echo "⚠️  Database not accessible (update DATABASE_URL for AWS)"
echo ""

# Check RTSP connectivity (optional - may fail locally)
echo "6. Checking RTSP camera sample..."
timeout 3 ffprobe -v quiet "rtsp://admin:Admin123@110.227.215.246:554/cam/realmonitor?channel=3&subtype=0" 2>/dev/null && echo "✅ RTSP camera reachable" || echo "⚠️  RTSP may not be accessible from AWS (check firewall)"
echo ""

# Check PetPooja API
echo "7. Checking PetPooja external API..."
curl -s -m 5 "http://13.202.92.108:8000/health" >/dev/null 2>&1 && echo "✅ PetPooja API reachable" || echo "⚠️  PetPooja API unreachable (analytics will use local DB)"
echo ""

echo "=================================="
echo "Verification Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Stage changes:    git add -A"
echo "2. Commit:           git commit -m 'Your message'"
echo "3. Push:             git push origin Tea-toast-new-v1"
echo "4. Deploy on AWS:    sudo systemctl restart sakshi-ai.service"
echo ""
