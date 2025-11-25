# PetPooja Webhook FastAPI Documentation

## Overview
This FastAPI application provides a complete CRUD (Create, Read, Update, Delete) interface for managing PetPooja webhook events stored in a PostgreSQL database.

## Database Schema

### Table: `petpooja_webhook_events`
- **id**: Integer (Primary Key, Auto-increment)
- **content**: JSONB (Stores webhook payload)
- **created_at**: DateTime (Timezone-aware, Auto-generated)

## Setup

### 1. Install Dependencies
```bash
pip install fastapi "uvicorn[standard]" sqlalchemy psycopg2-binary python-dotenv
```

### 2. Environment Variables
Create a `.env` file with:
```
DATABASE_URL=postgresql://user:password@host:port/database
PETPOOJA_API_TOKEN=your_secret_token
```

### 3. Run the Application
```bash
# Development
python fastapi_app.py

# Or with uvicorn directly
uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

All endpoints (except root) require authentication via `token` query parameter.

### Authentication
Add `?token=YOUR_TOKEN` to all requests (except root endpoint).

---

## Endpoints

### 1. Health Check
**GET** `/`

Check if the API is running.

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "status": "success",
  "message": "PetPooja Webhook API is running",
  "version": "1.0.0"
}
```

---

### 2. Create Webhook Event
**POST** `/webhook/events?token=YOUR_TOKEN`

Create a new webhook event.

**Request Body:**
```json
{
  "content": {
    "order_id": "12345",
    "customer": "John Doe",
    "total": 1500
  }
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/webhook/events?token=YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": {
      "order_id": "12345",
      "customer": "John Doe",
      "total": 1500
    }
  }'
```

**Response (201):**
```json
{
  "id": 1,
  "content": {
    "order_id": "12345",
    "customer": "John Doe",
    "total": 1500
  },
  "created_at": "2025-11-21T10:30:00+00:00"
}
```

---

### 3. Get All Webhook Events
**GET** `/webhook/events?token=YOUR_TOKEN`

Retrieve all webhook events with pagination.

**Query Parameters:**
- `skip`: Number of records to skip (default: 0)
- `limit`: Maximum records to return (default: 100, max: 1000)

**Example:**
```bash
curl "http://localhost:8000/webhook/events?token=YOUR_TOKEN&skip=0&limit=10"
```

**Response (200):**
```json
[
  {
    "id": 1,
    "content": {
      "order_id": "12345",
      "customer": "John Doe",
      "total": 1500
    },
    "created_at": "2025-11-21T10:30:00+00:00"
  },
  {
    "id": 2,
    "content": {
      "order_id": "67890",
      "customer": "Jane Smith",
      "total": 2500
    },
    "created_at": "2025-11-21T11:00:00+00:00"
  }
]
```

---

### 4. Get Single Webhook Event
**GET** `/webhook/events/{event_id}?token=YOUR_TOKEN`

Retrieve a specific webhook event by ID.

**Example:**
```bash
curl "http://localhost:8000/webhook/events/1?token=YOUR_TOKEN"
```

**Response (200):**
```json
{
  "id": 1,
  "content": {
    "order_id": "12345",
    "customer": "John Doe",
    "total": 1500
  },
  "created_at": "2025-11-21T10:30:00+00:00"
}
```

**Error Response (404):**
```json
{
  "status": "error",
  "message": "Event with ID 999 not found"
}
```

---

### 5. Update Webhook Event
**PUT** `/webhook/events/{event_id}?token=YOUR_TOKEN`

Update an existing webhook event's content.

**Request Body:**
```json
{
  "content": {
    "order_id": "12345",
    "customer": "John Doe Updated",
    "total": 1800,
    "status": "completed"
  }
}
```

**Example:**
```bash
curl -X PUT "http://localhost:8000/webhook/events/1?token=YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": {
      "order_id": "12345",
      "customer": "John Doe Updated",
      "total": 1800,
      "status": "completed"
    }
  }'
```

**Response (200):**
```json
{
  "id": 1,
  "content": {
    "order_id": "12345",
    "customer": "John Doe Updated",
    "total": 1800,
    "status": "completed"
  },
  "created_at": "2025-11-21T10:30:00+00:00"
}
```

---

### 6. Delete Webhook Event
**DELETE** `/webhook/events/{event_id}?token=YOUR_TOKEN`

Delete a webhook event by ID.

**Example:**
```bash
curl -X DELETE "http://localhost:8000/webhook/events/1?token=YOUR_TOKEN"
```

**Response (200):**
```json
{
  "status": "success",
  "message": "Event with ID 1 deleted successfully"
}
```

---

### 7. Get Events Count
**GET** `/webhook/events/stats/count?token=YOUR_TOKEN`

Get total count of webhook events.

**Example:**
```bash
curl "http://localhost:8000/webhook/events/stats/count?token=YOUR_TOKEN"
```

**Response (200):**
```json
{
  "status": "success",
  "total_events": 150
}
```

---

### 8. Search by Date Range
**GET** `/webhook/events/search/date-range?token=YOUR_TOKEN`

Search webhook events within a date range.

**Query Parameters:**
- `start_date`: ISO format datetime (optional)
- `end_date`: ISO format datetime (optional)

**Example:**
```bash
curl "http://localhost:8000/webhook/events/search/date-range?token=YOUR_TOKEN&start_date=2025-11-01T00:00:00&end_date=2025-11-21T23:59:59"
```

**Response (200):**
```json
[
  {
    "id": 1,
    "content": {
      "order_id": "12345"
    },
    "created_at": "2025-11-21T10:30:00+00:00"
  }
]
```

---

### 9. Legacy Endpoint (Backward Compatibility)
**GET** `/petpooja?token=YOUR_TOKEN`

Original endpoint maintained for backward compatibility with existing webhooks.

**Query Parameters:**
- `payload`: JSON object (form-encoded)

---

## Interactive Documentation

Once the server is running, access:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive API documentation where you can test all endpoints.

---

## Error Responses

### 401 Unauthorized
```json
{
  "status": "error",
  "message": "Invalid authentication token"
}
```

### 404 Not Found
```json
{
  "status": "error",
  "message": "Event with ID X not found"
}
```

### 500 Internal Server Error
```json
{
  "status": "error",
  "message": "Failed to create event: <error details>"
}
```

---

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"
TOKEN = "your_secret_token"

# Create event
response = requests.post(
    f"{BASE_URL}/webhook/events?token={TOKEN}",
    json={"content": {"order_id": "123", "amount": 1500}}
)
print(response.json())

# Get all events
response = requests.get(f"{BASE_URL}/webhook/events?token={TOKEN}&limit=10")
events = response.json()
print(f"Total events retrieved: {len(events)}")

# Get single event
event_id = 1
response = requests.get(f"{BASE_URL}/webhook/events/{event_id}?token={TOKEN}")
print(response.json())

# Update event
response = requests.put(
    f"{BASE_URL}/webhook/events/{event_id}?token={TOKEN}",
    json={"content": {"order_id": "123", "status": "completed"}}
)
print(response.json())

# Delete event
response = requests.delete(f"{BASE_URL}/webhook/events/{event_id}?token={TOKEN}")
print(response.json())

# Get count
response = requests.get(f"{BASE_URL}/webhook/events/stats/count?token={TOKEN}")
print(response.json())
```

---

## Production Deployment

### Using Gunicorn
```bash
pip install gunicorn
gunicorn fastapi_app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Systemd Service
Create `/etc/systemd/system/petpooja-api.service`:
```ini
[Unit]
Description=PetPooja FastAPI Service
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/your/app
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/gunicorn fastapi_app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable petpooja-api
sudo systemctl start petpooja-api
```

---

## Security Notes

1. **Keep your API token secret** - Never commit it to version control
2. **Use HTTPS in production** - Configure SSL/TLS certificates
3. **Rate limiting** - Consider adding rate limiting middleware
4. **CORS** - Configure CORS if accessed from web browsers
5. **Database connection pooling** - Already handled by SQLAlchemy

---

## Testing

```bash
# Install pytest
pip install pytest httpx

# Run tests (create test_api.py first)
pytest test_api.py -v
```
