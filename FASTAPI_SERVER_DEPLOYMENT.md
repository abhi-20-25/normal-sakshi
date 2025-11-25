# FastAPI PetPooja Webhook - Server Deployment Guide

## Overview
Deploy the PetPooja FastAPI application on Ubuntu server (13.203.73.173) alongside your existing Sakshi AI application.

---

## Current Server Setup
- **Flask App:** Running on port 5001 (edit-004.py)
- **New FastAPI App:** Will run on port 8000 (fastapi_app.py)
- **Database:** Same PostgreSQL database (sakshi)
- **User:** ubuntu
- **Working Directory:** /home/ubuntu/normal-sakshi

---

## 1. Update Requirements

Add FastAPI dependencies to your existing requirements.txt:

```bash
# On server
cd /home/ubuntu/normal-sakshi
source venv/bin/activate

# Install FastAPI dependencies
pip install fastapi "uvicorn[standard]" gunicorn
```

Or add to requirements.txt:
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
```

Then:
```bash
pip install -r requirements.txt
```

---

## 2. Upload Files to Server

From your local machine:

```bash
# Upload fastapi_app.py
scp /home/athul/sakshi/normal-sakshi/fastapi_app.py ubuntu@13.203.73.173:/home/ubuntu/normal-sakshi/

# Upload service file
scp /home/athul/sakshi/normal-sakshi/petpooja-api.service ubuntu@13.203.73.173:/tmp/

# Upload .env if needed (ensure DATABASE_URL points to server DB)
# scp /home/athul/sakshi/normal-sakshi/.env ubuntu@13.203.73.173:/home/ubuntu/normal-sakshi/
```

---

## 3. Update .env on Server

SSH to server and update DATABASE_URL:

```bash
ssh ubuntu@13.203.73.173
cd /home/ubuntu/normal-sakshi
nano .env
```

Update to server's PostgreSQL:
```env
# If PostgreSQL is on same server
DATABASE_URL=postgresql://postgres:Tneural01@127.0.0.1:5432/sakshi
PETPOOJA_API_TOKEN="qwrdx477ggh77hh"
```

**OR** if PostgreSQL is remote:
```env
DATABASE_URL=postgresql://postgres:Tneural01@<DB_HOST>:5432/sakshi
PETPOOJA_API_TOKEN="qwrdx477ggh77hh"
```

---

## 4. Test the Application

Before creating systemd service, test manually:

```bash
cd /home/ubuntu/normal-sakshi
source venv/bin/activate

# Test direct run
python3 fastapi_app.py

# OR test with gunicorn
gunicorn fastapi_app:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 2 \
    --bind 0.0.0.0:8000 \
    --log-level info
```

**Test from another terminal:**
```bash
# Health check
curl http://localhost:8000/

# Test with token
curl "http://localhost:8000/webhook/events/stats/count?token=qwrdx477ggh77hh"
```

If successful, proceed to systemd setup.

---

## 5. Setup Systemd Service

```bash
# Copy service file
sudo cp /tmp/petpooja-api.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable petpooja-api

# Start the service
sudo systemctl start petpooja-api

# Check status
sudo systemctl status petpooja-api
```

---

## 6. Configure Firewall

Allow port 8000:

```bash
# If using UFW
sudo ufw allow 8000/tcp
sudo ufw status

# If using iptables
sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
sudo iptables-save
```

---

## 7. Test from External

From your local machine:

```bash
# Health check
curl http://13.203.73.173:8000/

# Get count
curl "http://13.203.73.173:8000/webhook/events/stats/count?token=qwrdx477ggh77hh"

# Get all events
curl "http://13.203.73.173:8000/webhook/events?token=qwrdx477ggh77hh&limit=10"

# Create event
curl -X POST "http://13.203.73.173:8000/webhook/events?token=qwrdx477ggh77hh" \
  -H "Content-Type: application/json" \
  -d '{
    "event": "test",
    "order_id": "12345"
  }'
```

---

## 8. Setup Nginx Reverse Proxy (Optional but Recommended)

Create nginx config for both services:

```bash
sudo nano /etc/nginx/sites-available/sakshi
```

```nginx
# Flask App (existing)
server {
    listen 80;
    server_name your-domain.com;

    # Flask App
    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # FastAPI App
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable and restart:
```bash
sudo ln -s /etc/nginx/sites-available/sakshi /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

Then access via:
- Flask App: http://your-domain.com/
- FastAPI App: http://your-domain.com/api/

---

## 9. Monitoring & Logs

### View Logs:
```bash
# FastAPI service logs
sudo journalctl -u petpooja-api -f

# Gunicorn access logs
tail -f /var/log/gunicorn/petpooja-access.log

# Gunicorn error logs
tail -f /var/log/gunicorn/petpooja-error.log
```

### Service Management:
```bash
# Stop service
sudo systemctl stop petpooja-api

# Restart service
sudo systemctl restart petpooja-api

# Check status
sudo systemctl status petpooja-api

# Disable service
sudo systemctl disable petpooja-api
```

---

## 10. Security Considerations

### A. Change API Token
Update token in .env to something more secure:
```env
PETPOOJA_API_TOKEN="your-very-secure-random-token-here"
```

### B. Enable SSL (HTTPS)
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

### C. Rate Limiting (Optional)
Add to nginx config:
```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

location /api/ {
    limit_req zone=api_limit burst=20 nodelay;
    # ... rest of config
}
```

---

## 11. Database Access

The FastAPI app uses the **same database** as your Flask app:

**Table:** `petpooja_webhook_events`
- **id** (integer, primary key)
- **content** (jsonb)
- **created_at** (timestamp with timezone)

Both applications can access the same data!

---

## 12. Configure PetPooja Webhook

In PetPooja dashboard, set webhook URL to:

**Option 1 - Direct:** 
```
http://13.203.73.173:8000/webhook/events?token=qwrdx477ggh77hh
```

**Option 2 - With Nginx:**
```
http://your-domain.com/api/webhook/events?token=qwrdx477ggh77hh
```

**Option 3 - Legacy endpoint (backward compatible):**
```
http://13.203.73.173:8000/petpooja?token=qwrdx477ggh77hh
```

---

## 13. Running Both Services

Your server will now run:

1. **Flask App (Sakshi AI)** - Port 5001
   - Service: `sakshi-ai.service`
   - Edit-004.py with video monitoring

2. **FastAPI App (PetPooja)** - Port 8000
   - Service: `petpooja-api.service`
   - Webhook receiver and CRUD API

Both share the same PostgreSQL database!

---

## 14. Quick Deployment Script

Create deployment script:

```bash
nano deploy_fastapi.sh
```

```bash
#!/bin/bash
# FastAPI Deployment Script

echo "=== PetPooja FastAPI Deployment ==="

# Activate virtual environment
cd /home/ubuntu/normal-sakshi
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install fastapi "uvicorn[standard]" gunicorn

# Stop service if running
echo "Stopping service..."
sudo systemctl stop petpooja-api 2>/dev/null || true

# Copy service file
echo "Installing systemd service..."
sudo cp petpooja-api.service /etc/systemd/system/

# Reload and start
echo "Starting service..."
sudo systemctl daemon-reload
sudo systemctl enable petpooja-api
sudo systemctl start petpooja-api

# Check status
echo "Service status:"
sudo systemctl status petpooja-api --no-pager

echo "=== Deployment Complete ==="
echo "API running on http://localhost:8000"
echo "Documentation: http://localhost:8000/docs"
```

Make executable and run:
```bash
chmod +x deploy_fastapi.sh
./deploy_fastapi.sh
```

---

## Troubleshooting

### Service won't start:
```bash
sudo journalctl -u petpooja-api -n 50
```

### Port already in use:
```bash
sudo lsof -i :8000
sudo kill -9 <PID>
```

### Database connection error:
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql -h 127.0.0.1 -U postgres -d sakshi
```

### Check open ports:
```bash
sudo netstat -tlnp | grep -E ':(5001|8000)'
```

---

## API Endpoints Summary

Once deployed, access via:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/webhook/events` | POST | Create webhook event |
| `/webhook/events` | GET | Get all events (paginated) |
| `/webhook/events/{id}` | GET | Get single event |
| `/webhook/events/{id}` | PUT | Update event |
| `/webhook/events/{id}` | DELETE | Delete event |
| `/webhook/events/stats/count` | GET | Get total count |
| `/webhook/events/search/date-range` | GET | Search by date |
| `/docs` | GET | Interactive API docs |
| `/redoc` | GET | Alternative API docs |
| `/petpooja` | GET | Legacy endpoint |

All endpoints (except `/` and `/docs`) require `?token=qwrdx477ggh77hh`

---

## Success Criteria

✅ Service running: `sudo systemctl status petpooja-api`
✅ Health check: `curl http://localhost:8000/`
✅ Can create events via API
✅ Data appears in PostgreSQL table
✅ Accessible from external IP
✅ Logs are being written
✅ Auto-restarts on failure

---

## Support

For issues, check:
1. Service logs: `sudo journalctl -u petpooja-api -f`
2. Gunicorn logs: `/var/log/gunicorn/petpooja-*.log`
3. PostgreSQL logs: `/var/log/postgresql/`
4. Application is reading .env correctly
5. Database connectivity
