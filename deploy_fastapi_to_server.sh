#!/bin/bash
# Deploy FastAPI PetPooja application to server
# Run from local machine

SERVER="ubuntu@13.203.73.173"
REMOTE_DIR="/home/ubuntu/normal-sakshi"
LOCAL_DIR="/home/athul/sakshi/normal-sakshi"

echo "==================================="
echo "FastAPI Deployment to Server"
echo "==================================="

# Step 1: Upload files
echo ""
echo "Step 1: Uploading files to server..."
scp "${LOCAL_DIR}/fastapi_app.py" "${SERVER}:${REMOTE_DIR}/"
scp "${LOCAL_DIR}/petpooja-api.service" "${SERVER}:/tmp/"
scp "${LOCAL_DIR}/API_DOCUMENTATION.md" "${SERVER}:${REMOTE_DIR}/"
scp "${LOCAL_DIR}/FASTAPI_SERVER_DEPLOYMENT.md" "${SERVER}:${REMOTE_DIR}/"

echo "✓ Files uploaded"

# Step 2: Run installation on server
echo ""
echo "Step 2: Installing on server..."
ssh "${SERVER}" << 'ENDSSH'
cd /home/ubuntu/normal-sakshi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing FastAPI dependencies..."
pip install fastapi "uvicorn[standard]" gunicorn --quiet

# Stop service if running
echo "Stopping existing service..."
sudo systemctl stop petpooja-api 2>/dev/null || true

# Install service file
echo "Installing systemd service..."
sudo cp /tmp/petpooja-api.service /etc/systemd/system/

# Create log directory if not exists
sudo mkdir -p /var/log/gunicorn
sudo chown ubuntu:www-data /var/log/gunicorn

# Reload systemd
sudo systemctl daemon-reload

# Enable and start service
echo "Starting service..."
sudo systemctl enable petpooja-api
sudo systemctl start petpooja-api

# Wait a moment for service to start
sleep 3

# Check status
echo ""
echo "==================================="
echo "Service Status:"
echo "==================================="
sudo systemctl status petpooja-api --no-pager

echo ""
echo "==================================="
echo "Testing API..."
echo "==================================="

# Test health endpoint
if curl -s http://localhost:8000/ | grep -q "success"; then
    echo "✓ Health check passed"
else
    echo "✗ Health check failed"
fi

# Test with token
if curl -s "http://localhost:8000/webhook/events/stats/count?token=qwrdx477ggh77hh" | grep -q "total_events"; then
    echo "✓ Token authentication working"
else
    echo "✗ Token authentication failed"
fi

ENDSSH

echo ""
echo "==================================="
echo "Deployment Complete!"
echo "==================================="
echo ""
echo "API is now running on:"
echo "  http://13.203.73.173:8000"
echo ""
echo "API Documentation:"
echo "  http://13.203.73.173:8000/docs"
echo ""
echo "Test from local machine:"
echo "  curl http://13.203.73.173:8000/"
echo ""
echo "Useful commands on server:"
echo "  sudo systemctl status petpooja-api"
echo "  sudo systemctl restart petpooja-api"
echo "  sudo journalctl -u petpooja-api -f"
echo "  tail -f /var/log/gunicorn/petpooja-access.log"
echo ""
