#!/bin/bash
# Test FastAPI server endpoints

SERVER="http://13.203.73.173:8000"
TOKEN="qwrdx477ggh77hh"

echo "======================================"
echo "Testing PetPooja FastAPI Server"
echo "======================================"
echo ""

echo "1. Health Check:"
curl -s $SERVER/ | jq '.' || curl -s $SERVER/
echo -e "\n"

echo "2. Get Events Count:"
curl -s "$SERVER/webhook/events/stats/count?token=$TOKEN" | jq '.' || curl -s "$SERVER/webhook/events/stats/count?token=$TOKEN"
echo -e "\n"

echo "3. Get All Events (limit 5):"
curl -s "$SERVER/webhook/events?token=$TOKEN&limit=5" | jq '.' || curl -s "$SERVER/webhook/events?token=$TOKEN&limit=5"
echo -e "\n"

echo "4. Create Test Event:"
curl -s -X POST "$SERVER/webhook/events?token=$TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "event": "test_event",
    "message": "Testing from script",
    "timestamp": "'$(date -Iseconds)'"
  }' | jq '.' || curl -s -X POST "$SERVER/webhook/events?token=$TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"event": "test_event", "message": "Testing from script"}'
echo -e "\n"

echo "======================================"
echo "✅ API Access URLs:"
echo "======================================"
echo "Swagger UI:  $SERVER/docs"
echo "ReDoc:       $SERVER/redoc"
echo "Health:      $SERVER/"
echo "======================================"
