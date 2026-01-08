# Sales Analytics Multi-Store Fix

## Problem Identified

Sales Analytics is showing combined data (7 orders) for both stores instead of filtering correctly:
- Sangli should show: 1 order
- Tilakwadi should show: 6 orders
- Combined it's showing: 7 orders in BOTH dashboards

## Root Cause

1. **Conversion Analytics** ✅ Works - Uses local endpoint `/api/analytics/footfall-conversion` which filters correctly
2. **Sales Analytics** ❌ Broken - Uses remote FastAPI `http://13.202.92.108:8000/analytics/sales-daily` which is:
   - Either not running
   - Running old code without proper filtering
   - Not receiving the restaurant_id parameter correctly

## Evidence

In `templates/dashboard.html` line 3180:
```javascript
const salesRes = await fetch(`${FASTAPI_URL}/analytics/sales-daily?days=7&token=${API_TOKEN}${restaurantParam}`);
```

The `restaurantParam` includes the PetPooja restaurant ID (e.g., `&restaurant_id=38vpyhwq19`), but the remote server is not filtering.

## Solutions

### Option 1: Restart Remote FastAPI Server (Recommended)
The remote FastAPI server at `13.202.92.108:8000` needs to be restarted with the current `fastapi_app.py` code which has proper restaurant filtering in all endpoints.

```bash
# On the remote server (13.202.92.108)
cd /path/to/normal-sakshi
source ttenv/bin/activate
pkill -f "uvicorn.*fastapi_app"
nohup python -m uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 &
```

### Option 2: Create Local Endpoints (Temporary Workaround)
Add local endpoints in `edit-004.py` that mirror the FastAPI endpoints but use local database queries with proper restaurant filtering.

This would require creating:
- `/api/sales/daily` - Daily sales with restaurant filtering
- `/api/sales/payment-modes` - Payment modes with restaurant filtering  
- `/api/sales/order-types` - Order types with restaurant filtering
- `/api/sales/top-items` - Top items with restaurant filtering

### Option 3: Use Raw Events Processing (Current Fallback)
The `petpooja_integration.py` already has fallback processing that can work from raw webhook events, but it needs to be exposed through local endpoints.

## Recommended Action

**Restart the remote FastAPI server** with the current code. This is the cleanest solution as:
1. The code is already correct in `fastapi_app.py`
2. All endpoints support restaurant filtering
3. No code changes needed
4. Maintains separation of concerns

## Testing After Fix

After restarting the remote FastAPI server:

1. Switch to Sangli store (restaurant_id=1)
   - Should show 1 order in Sales Analytics
   - PetPooja ID: 38vpyhwq19

2. Switch to Tilakwadi store (restaurant_id=2)  
   - Should show 6 orders in Sales Analytics
   - PetPooja ID: mc96bfd0

3. Select "All Restaurants"
   - Should show 7 orders combined
