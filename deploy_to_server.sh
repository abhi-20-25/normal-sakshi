#!/bin/bash
# Deploy updated edit-004.py to server and restart

SERVER_USER="ubuntu"
SERVER_HOST="your-server-ip"  # UPDATE THIS!
SERVER_PATH="/home/ubuntu/normal-sakshi"

echo "📦 Deploying to server..."

# Copy the updated file
scp edit-004.py ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/

# Restart the service on server
ssh ${SERVER_USER}@${SERVER_HOST} << 'EOF'
cd /home/ubuntu/normal-sakshi
echo "🔄 Stopping existing service..."
sudo systemctl stop sakshi  # or: pkill -f edit-004.py

echo "🚀 Starting service..."
source venv/bin/activate
nohup python3 edit-004.py > server.log 2>&1 &

echo "✅ Service restarted. Checking logs..."
sleep 3
tail -20 server.log
EOF

echo "✅ Deployment complete!"
