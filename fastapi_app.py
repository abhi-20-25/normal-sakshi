import os
import sys
import logging
from typing import Optional, List, Any
from datetime import date, datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException, status, Query, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, DateTime, func, desc, Float, Text, cast, Date, Index, text
from sqlalchemy.orm import aliased
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from pydantic import BaseModel, Field
from sqlalchemy import Date
from fastapi.middleware.cors import CORSMiddleware 


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fastapi_app.log')
    ]
)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
PETPOOJA_API_TOKEN = os.getenv("PETPOOJA_API_TOKEN")

if not DATABASE_URL or not PETPOOJA_API_TOKEN:
    print("FATAL ERROR: Make sure DATABASE_URL and PETPOOJA_API_TOKEN are set.")
    sys.exit(1)

Base = declarative_base()

# SQLAlchemy Model
class PetpoojaWebhookEvent(Base):
    __tablename__ = "petpooja_webhook_events"
    id = Column(Integer, primary_key=True)
    content = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class SalesStats(BaseModel):
    date: str
    total_sales: float
    order_count: int

class HourlySalesStats(BaseModel):
    date: str
    hour: int
    orders: int
    revenue: float

class PaymentStats(BaseModel):
    method: str
    amount: float
    count: int

class OrderTypeStats(BaseModel):
    order_source: str # e.g., POS, Zomato, Swiggy
    order_type: str   # e.g., Dine In, Delivery
    count: int
    total_sales: float

class TopItem(BaseModel):
    name: str
    quantity_sold: int
    total_revenue: float

class RestaurantInfo(BaseModel):
    rest_id: str
    name: str
    address: str
    contact: str

class DashboardResponse(BaseModel):
    sales_timeline: List[SalesStats]
    payment_modes: List[PaymentStats]
    order_types: List[OrderTypeStats]
    top_items: List[TopItem]

# Create engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)
logging.info("✅ Database tables created/verified successfully")

# Pydantic Models for API
class WebhookEventCreate(BaseModel):
    # Make content optional and allow any dict to be passed
    content: Optional[dict] = Field(None, description="JSON content of the webhook event")
    
    class Config:
        extra = "allow"  # Allow additional fields
    
    def get_content(self):
        """Return content if provided, otherwise return the entire model as dict"""
        if self.content is not None:
            return self.content
        # Return all fields except 'content' as the content
        return {k: v for k, v in self.model_dump().items() if k != 'content' and v is not None}

class WebhookEventUpdate(BaseModel):
    content: dict = Field(..., description="Updated JSON content")

class WebhookEventResponse(BaseModel):
    id: int
    content: dict
    created_at: datetime

    class Config:
        from_attributes = True

# FastAPI App
app = FastAPI(
    title="PetPooja Webhook API",
    description="FastAPI application for managing PetPooja webhook events",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logging.info("=" * 60)
    logging.info("🚀 PetPooja FastAPI Application Starting")
    logging.info("=" * 60)
    logging.info(f"Database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'configured'}")
    logging.info(f"API Token: {'*' * (len(PETPOOJA_API_TOKEN) - 4) + PETPOOJA_API_TOKEN[-4:]}")
    logging.info(f"Log file: fastapi_app.log")
    logging.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    logging.info("=" * 60)
    logging.info("🛑 PetPooja FastAPI Application Shutting Down")
    logging.info("=" * 60)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Middleware for token authentication
def verify_token(token: str = Query(..., description="API authentication token")):
    if token != PETPOOJA_API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"status": "error", "message": "Invalid authentication token"}
        )
    return token

# Health check endpoint
@app.get("/")
def root():
    return {
        "status": "success",
        "message": "PetPooja Webhook API is running",
        "version": "1.0.0"
    }

# Get list of all restaurants
@app.get("/restaurants", response_model=List[RestaurantInfo])
def get_restaurants(token: str = Depends(verify_token), db: Session = Depends(get_db)):
    """
    Get list of all unique restaurants from webhook data.
    Returns restaurant ID, name, address, and contact information.
    """
    try:
        sql = text("""
            SELECT DISTINCT
                content->'properties'->'Restaurant'->>'restID' as rest_id,
                content->'properties'->'Restaurant'->>'res_name' as name,
                content->'properties'->'Restaurant'->>'address' as address,
                content->'properties'->'Restaurant'->>'contact_information' as contact
            FROM petpooja_webhook_events
            WHERE content->>'event' = 'orderdetails'
            AND content->'properties'->'Restaurant'->>'restID' IS NOT NULL
            ORDER BY name
        """)
        
        results = db.execute(sql).fetchall()
        
        return [
            RestaurantInfo(
                rest_id=r[0] or "unknown",
                name=r[1] or "Unknown Restaurant",
                address=r[2] or "",
                contact=r[3] or ""
            ) for r in results
        ]
    except Exception as e:
        logging.error(f"Error fetching restaurants: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "error", "message": f"Failed to fetch restaurants: {str(e)}"}
        )

# CREATE - Add new webhook event
@app.post(
    "/webhook/events",
    response_model=WebhookEventResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Webhook Events"]
)
def create_webhook_event(
    event: WebhookEventCreate,
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Create a new webhook event in the database.
    Requires valid API token for authentication.
    Accepts either {"content": {...}} or directly {...}
    """
    start_time = datetime.now()
    logging.info(f"[CREATE] Starting webhook event creation at {start_time}")
    
    try:
        # Get the content - either from content field or entire payload
        content_data = event.get_content()
        logging.info(f"[CREATE] Received event data with {len(str(content_data))} bytes")
        
        db_event = PetpoojaWebhookEvent(content=content_data)
        db.add(db_event)
        
        commit_start = datetime.now()
        db.commit()
        commit_end = datetime.now()
        commit_duration = (commit_end - commit_start).total_seconds()
        
        db.refresh(db_event)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        logging.info(f"[CREATE] ✅ Event created successfully | ID: {db_event.id} | "
                    f"DB Commit Time: {commit_duration:.3f}s | Total Time: {total_duration:.3f}s | "
                    f"Timestamp: {db_event.created_at}")
        
        return db_event
    except Exception as e:
        db.rollback()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logging.error(f"[CREATE] ❌ Failed to create event | Duration: {duration:.3f}s | Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "error", "message": f"Failed to create event: {str(e)}"}
        )

# READ - Get all webhook events (with pagination)
@app.get(
    "/webhook/events",
    response_model=List[WebhookEventResponse],
    tags=["Webhook Events"]
)
def get_all_webhook_events(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Retrieve all webhook events with pagination.
    Results are ordered by creation date (newest first).
    """
    start_time = datetime.now()
    logging.info(f"[READ ALL] Fetching events | skip={skip}, limit={limit}")
    
    try:
        query_start = datetime.now()
        events = db.query(PetpoojaWebhookEvent)\
            .order_by(desc(PetpoojaWebhookEvent.created_at))\
            .offset(skip)\
            .limit(limit)\
            .all()
        query_end = datetime.now()
        query_duration = (query_end - query_start).total_seconds()
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        logging.info(f"[READ ALL] ✅ Retrieved {len(events)} events | "
                    f"DB Query Time: {query_duration:.3f}s | Total Time: {total_duration:.3f}s")
        
        return events
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logging.error(f"[READ ALL] ❌ Failed to retrieve events | Duration: {duration:.3f}s | Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "error", "message": f"Failed to retrieve events: {str(e)}"}
        )

# READ - Get single webhook event by ID
@app.get(
    "/webhook/events/{event_id}",
    response_model=WebhookEventResponse,
    tags=["Webhook Events"]
)
def get_webhook_event(
    event_id: int,
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Retrieve a specific webhook event by its ID.
    """
    start_time = datetime.now()
    logging.info(f"[READ ONE] Fetching event ID: {event_id}")
    
    query_start = datetime.now()
    event = db.query(PetpoojaWebhookEvent).filter(PetpoojaWebhookEvent.id == event_id).first()
    query_end = datetime.now()
    query_duration = (query_end - query_start).total_seconds()
    
    if not event:
        logging.warning(f"[READ ONE] ⚠️ Event ID {event_id} not found | Query Time: {query_duration:.3f}s")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"status": "error", "message": f"Event with ID {event_id} not found"}
        )
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    logging.info(f"[READ ONE] ✅ Event retrieved | ID: {event_id} | "
                f"Created: {event.created_at} | "
                f"DB Query Time: {query_duration:.3f}s | Total Time: {total_duration:.3f}s")
    
    return event

# UPDATE - Update webhook event by ID
@app.put(
    "/webhook/events/{event_id}",
    response_model=WebhookEventResponse,
    tags=["Webhook Events"]
)
def update_webhook_event(
    event_id: int,
    event_update: WebhookEventUpdate,
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Update an existing webhook event's content.
    """
    start_time = datetime.now()
    logging.info(f"[UPDATE] Starting update for event ID: {event_id} at {start_time}")
    
    db_event = db.query(PetpoojaWebhookEvent).filter(PetpoojaWebhookEvent.id == event_id).first()
    
    if not db_event:
        logging.warning(f"[UPDATE] ⚠️ Event ID {event_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"status": "error", "message": f"Event with ID {event_id} not found"}
        )
    
    try:
        old_content_size = len(str(db_event.content))
        new_content_size = len(str(event_update.content))
        
        db_event.content = event_update.content
        
        commit_start = datetime.now()
        db.commit()
        commit_end = datetime.now()
        commit_duration = (commit_end - commit_start).total_seconds()
        
        db.refresh(db_event)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        logging.info(f"[UPDATE] ✅ Event updated successfully | ID: {event_id} | "
                    f"Content Size: {old_content_size}→{new_content_size} bytes | "
                    f"DB Commit Time: {commit_duration:.3f}s | Total Time: {total_duration:.3f}s")
        
        return db_event
    except Exception as e:
        db.rollback()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logging.error(f"[UPDATE] ❌ Failed to update event ID {event_id} | Duration: {duration:.3f}s | Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "error", "message": f"Failed to update event: {str(e)}"}
        )

# DELETE - Delete webhook event by ID
@app.delete(
    "/webhook/events/{event_id}",
    status_code=status.HTTP_200_OK,
    tags=["Webhook Events"]
)
def delete_webhook_event(
    event_id: int,
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Delete a webhook event by its ID.
    """
    start_time = datetime.now()
    logging.info(f"[DELETE] Starting deletion for event ID: {event_id} at {start_time}")
    
    db_event = db.query(PetpoojaWebhookEvent).filter(PetpoojaWebhookEvent.id == event_id).first()
    
    if not db_event:
        logging.warning(f"[DELETE] ⚠️ Event ID {event_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"status": "error", "message": f"Event with ID {event_id} not found"}
        )
    
    try:
        event_created_at = db_event.created_at
        content_size = len(str(db_event.content))
        
        db.delete(db_event)
        
        commit_start = datetime.now()
        db.commit()
        commit_end = datetime.now()
        commit_duration = (commit_end - commit_start).total_seconds()
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        logging.info(f"[DELETE] ✅ Event deleted successfully | ID: {event_id} | "
                    f"Created: {event_created_at} | Content Size: {content_size} bytes | "
                    f"DB Commit Time: {commit_duration:.3f}s | Total Time: {total_duration:.3f}s")
        
        return {
            "status": "success",
            "message": f"Event with ID {event_id} deleted successfully"
        }
    except Exception as e:
        db.rollback()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logging.error(f"[DELETE] ❌ Failed to delete event ID {event_id} | Duration: {duration:.3f}s | Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "error", "message": f"Failed to delete event: {str(e)}"}
        )

# STATISTICS - Get count of events
@app.get(
    "/webhook/events/stats/count",
    tags=["Statistics"]
)
def get_events_count(
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Get the total count of webhook events in the database.
    """
    try:
        count = db.query(PetpoojaWebhookEvent).count()
        return {
            "status": "success",
            "total_events": count
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "error", "message": f"Failed to count events: {str(e)}"}
        )

# SEARCH - Search events by date range
@app.get(
    "/webhook/events/search/date-range",
    response_model=List[WebhookEventResponse],
    tags=["Search"]
)
def search_events_by_date(
    start_date: Optional[datetime] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[datetime] = Query(None, description="End date (ISO format)"),
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Search webhook events within a specific date range.
    """
    try:
        query = db.query(PetpoojaWebhookEvent)
        
        if start_date:
            query = query.filter(PetpoojaWebhookEvent.created_at >= start_date)
        if end_date:
            query = query.filter(PetpoojaWebhookEvent.created_at <= end_date)
        
        events = query.order_by(desc(PetpoojaWebhookEvent.created_at)).all()
        return events
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "error", "message": f"Search failed: {str(e)}"}
        )

# Legacy endpoint - Keep for backward compatibility with existing webhook
@app.get("/petpooja", tags=["Legacy"])
def create_petpooja_event_legacy(
    payload: dict,
    token: str = Query(...)
):
    """
    Legacy endpoint for PetPooja webhooks (for backward compatibility).
    """
    start_time = datetime.now()
    logging.info(f"[LEGACY] Received webhook at legacy endpoint /petpooja at {start_time}")
    
    if token != PETPOOJA_API_TOKEN:
        logging.warning(f"[LEGACY] ❌ Authentication failed - invalid token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"status": "error", "message": "Authentication required"},
        )

    db = SessionLocal()
    try:
        logging.info(f"[LEGACY] Processing payload with {len(str(payload))} bytes")
        
        db_event = PetpoojaWebhookEvent(content=payload)
        db.add(db_event)
        
        commit_start = datetime.now()
        db.commit()
        commit_end = datetime.now()
        commit_duration = (commit_end - commit_start).total_seconds()
        
        db.refresh(db_event)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        logging.info(f"[LEGACY] ✅ Event saved successfully | ID: {db_event.id} | "
                    f"DB Commit Time: {commit_duration:.3f}s | Total Time: {total_duration:.3f}s | "
                    f"Timestamp: {db_event.created_at}")
        
        return {
            "status": "success",
            "message": "Data saved successfully",
            "id": db_event.id
        }
    except Exception as e:
        db.rollback()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logging.error(f"[LEGACY] ❌ Failed to save event | Duration: {duration:.3f}s | Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "error", "message": f"Server error: {e}"},
        )
    finally:
        db.close()


# --- ANALYTICS ENDPOINTS (UPDATED FOR DEDUPLICATION & MONTHLY FILTER) ---

def get_current_month_start():
    today = date.today()
    return datetime(today.year, today.month, 1)

@app.get("/analytics/sales-daily", response_model=List[SalesStats])
def get_daily_sales(
    days: int = None, 
    start_date: str = None, 
    end_date: str = None,
    restaurant_id: str = Query(None, description="Filter by restaurant ID (restID from webhook)"),
    token: str = Depends(verify_token), 
    db: Session = Depends(get_db)
):
    """Get daily sales stats. Either use 'days' param (last N days) or 'start_date'/'end_date' for specific range."""
    
    order_id_path = func.jsonb_extract_path_text(PetpoojaWebhookEvent.content, 'properties', 'Order', 'orderID')
    
    subquery_filters = [PetpoojaWebhookEvent.content['event'].astext == 'orderdetails']
    if restaurant_id:
        subquery_filters.append(func.jsonb_extract_path_text(PetpoojaWebhookEvent.content, 'properties', 'Restaurant', 'restID') == restaurant_id)
    
    subquery = db.query(func.max(PetpoojaWebhookEvent.id))\
        .filter(*subquery_filters)\
        .group_by(order_id_path)\
        .subquery()
    
    date_col = cast(func.jsonb_extract_path_text(PetpoojaWebhookEvent.content, 'properties', 'Order', 'created_on'), DateTime).cast(Date)
    total_col = cast(func.jsonb_extract_path_text(PetpoojaWebhookEvent.content, 'properties', 'Order', 'total'), Float)
    
    query = db.query(
        date_col.label('date'),
        func.sum(total_col).label('total_sales'),
        func.count(PetpoojaWebhookEvent.id).label('order_count')
    ).filter(
        PetpoojaWebhookEvent.id.in_(subquery)  # Only count the latest version of the order
    )
    
    # Filter by date range if provided, otherwise use days parameter
    if start_date and end_date:
        query = query.filter(date_col >= start_date, date_col <= end_date)
    elif days:
        # Calculate date range from days parameter
        from datetime import datetime, timedelta
        end_dt = datetime.now().date()
        start_dt = end_dt - timedelta(days=days - 1)
        query = query.filter(date_col >= start_dt, date_col <= end_dt)
    
    # Order by date ascending (oldest first) for consistency with conversion analytics
    results = query.group_by(date_col).order_by(date_col).all()
    
    return [SalesStats(date=str(r.date), total_sales=r.total_sales or 0, order_count=r.order_count) for r in results]

@app.get("/analytics/sales-hourly", response_model=List[HourlySalesStats])
def get_hourly_sales(
    start_date: str = Query(..., description="Start date YYYY-MM-DD"), 
    end_date: str = Query(..., description="End date YYYY-MM-DD"),
    restaurant_id: str = Query(None, description="Filter by restaurant ID (restID from webhook)"),
    token: str = Depends(verify_token), 
    db: Session = Depends(get_db)
):
    """Get hourly sales breakdown for conversion analytics - uses same deduplication logic as sales-daily"""
    
    # Step 1: Get unique orders only (deduplicate by orderID, keep latest version)
    order_id_path = func.jsonb_extract_path_text(PetpoojaWebhookEvent.content, 'properties', 'Order', 'orderID')
    
    subquery_filters = [PetpoojaWebhookEvent.content['event'].astext == 'orderdetails']
    if restaurant_id:
        subquery_filters.append(func.jsonb_extract_path_text(PetpoojaWebhookEvent.content, 'properties', 'Restaurant', 'restID') == restaurant_id)
    
    subquery = db.query(func.max(PetpoojaWebhookEvent.id))\
        .filter(*subquery_filters)\
        .group_by(order_id_path)\
        .subquery()
    
    # Step 2: Extract date and hour from Order.created_on (when order was placed)
    created_on_path = func.jsonb_extract_path_text(PetpoojaWebhookEvent.content, 'properties', 'Order', 'created_on')
    date_col = cast(created_on_path, DateTime).cast(Date)
    hour_col = func.extract('hour', cast(created_on_path, DateTime))
    total_col = cast(func.jsonb_extract_path_text(PetpoojaWebhookEvent.content, 'properties', 'Order', 'total'), Float)
    
    # Step 3: Query with deduplication
    results = db.query(
        date_col.label('sale_date'),
        hour_col.label('sale_hour'),
        func.count(PetpoojaWebhookEvent.id).label('orders'),
        func.sum(total_col).label('revenue')
    ).filter(
        PetpoojaWebhookEvent.id.in_(subquery),  # Only unique orders (latest version)
        date_col >= start_date,
        date_col <= end_date
    ).group_by(date_col, hour_col).order_by(date_col, hour_col).all()
    
    return [
        HourlySalesStats(
            date=str(r.sale_date),
            hour=int(r.sale_hour),
            orders=r.orders or 0,
            revenue=float(r.revenue) if r.revenue else 0.0
        ) for r in results
    ]

@app.get("/analytics/payment-modes", response_model=List[PaymentStats])
def get_payment_stats(
    restaurant_id: str = Query(None, description="Filter by restaurant ID (restID from webhook)"),
    token: str = Depends(verify_token), 
    db: Session = Depends(get_db)
):
    start_date = get_current_month_start()
    
    restaurant_filter = ""
    if restaurant_id:
        restaurant_filter = "AND content->'properties'->'Restaurant'->>'restID' = :restaurant_id"
    
    sql = text(f"""
        WITH unique_orders AS (
            SELECT content
            FROM petpooja_webhook_events
            WHERE id IN (
                SELECT MAX(id)
                FROM petpooja_webhook_events
                WHERE created_at >= :start_date 
                AND content->>'event' = 'orderdetails'
                {restaurant_filter}
                GROUP BY content->'properties'->'Order'->>'orderID'
            )
        )
        SELECT 
            content->'properties'->'Order'->>'payment_type' as method,
            SUM(CAST(content->'properties'->'Order'->>'total' AS FLOAT)) as amount,
            COUNT(*) as count
        FROM unique_orders
        GROUP BY method
    """)
    
    params = {"start_date": start_date}
    if restaurant_id:
        params["restaurant_id"] = restaurant_id
    results = db.execute(sql, params).fetchall()
    
    # FIX: Access by index (0, 1, 2) instead of name
    return [
        PaymentStats(
            method=r[0] or "Unknown", 
            amount=r[1] or 0, 
            count=r[2]
        ) for r in results
    ]

@app.get("/analytics/order-types", response_model=List[OrderTypeStats])
def get_order_types(
    restaurant_id: str = Query(None, description="Filter by restaurant ID (restID from webhook)"),
    token: str = Depends(verify_token), 
    db: Session = Depends(get_db)
):
    start_date = get_current_month_start()
    
    restaurant_filter = ""
    if restaurant_id:
        restaurant_filter = "AND content->'properties'->'Restaurant'->>'restID' = :restaurant_id"
    
    sql = text(f"""
        WITH unique_orders AS (
            SELECT content
            FROM petpooja_webhook_events
            WHERE id IN (
                SELECT MAX(id)
                FROM petpooja_webhook_events
                WHERE created_at >= :start_date 
                AND content->>'event' = 'orderdetails'
                {restaurant_filter}
                GROUP BY content->'properties'->'Order'->>'orderID'
            )
        )
        SELECT 
            content->'properties'->'Order'->>'order_from' as s,
            content->'properties'->'Order'->>'order_type' as t,
            COUNT(*) as c,
            SUM(CAST(content->'properties'->'Order'->>'total' AS FLOAT)) as tot
        FROM unique_orders
        GROUP BY s, t
    """)
    
    params = {"start_date": start_date}
    if restaurant_id:
        params["restaurant_id"] = restaurant_id
    results = db.execute(sql, params).fetchall()
    
    # FIX: Access by index (0=source, 1=type, 2=count, 3=total)
    return [
        OrderTypeStats(
            order_source=r[0] or "Unk", 
            order_type=r[1] or "Unk", 
            count=r[2], 
            total_sales=r[3] or 0
        ) for r in results
    ]

@app.get("/analytics/top-items", response_model=List[TopItem])
def get_top_items(
    limit: int = 5,
    restaurant_id: str = Query(None, description="Filter by restaurant ID (restID from webhook)"),
    token: str = Depends(verify_token), 
    db: Session = Depends(get_db)
):
    start_date = get_current_month_start()
    
    restaurant_filter = ""
    if restaurant_id:
        restaurant_filter = "AND content->'properties'->'Restaurant'->>'restID' = :restaurant_id"
    
    sql_query = text(f"""
        WITH unique_orders AS (
            SELECT content
            FROM petpooja_webhook_events
            WHERE id IN (
                SELECT MAX(id)
                FROM petpooja_webhook_events
                WHERE created_at >= :start_date 
                AND content->>'event' = 'orderdetails'
                {restaurant_filter}
                GROUP BY content->'properties'->'Order'->>'orderID'
            )
        )
        SELECT 
            item->>'name' as name,
            SUM(CAST(COALESCE(NULLIF(item->>'quantity', ''), '0') AS NUMERIC)) as qty,
            SUM(CAST(COALESCE(NULLIF(item->>'total', ''), '0') AS NUMERIC)) as rev
        FROM 
            unique_orders,
            jsonb_array_elements(content->'properties'->'OrderItem') item
        WHERE 
            jsonb_typeof(content->'properties'->'OrderItem') = 'array'
            AND item->>'name' IS NOT NULL 
            AND item->>'name' != ''
        GROUP BY 
            item->>'name'
        ORDER BY 
            qty DESC
        LIMIT :limit
    """)
    
    params = {"start_date": start_date, "limit": limit}
    if restaurant_id:
        params["restaurant_id"] = restaurant_id
    try:
        results = db.execute(sql_query, params).fetchall()
        
        # FIX: Access by index (0=name, 1=qty, 2=rev)
        return [
            TopItem(
                name=r[0], 
                quantity_sold=int(r[1]) if r[1] else 0, 
                total_revenue=float(r[2]) if r[2] else 0.0
            ) for r in results
        ]
    except Exception as e:
        logging.error(f"Error in top-items: {e}")
        return []

@app.get("/analytics/items-by-hour")
def get_items_by_hour(
    days: int = 30,
    restaurant_id: str = Query(None, description="Filter by restaurant ID (restID from webhook)"),
    token: str = Depends(verify_token), 
    db: Session = Depends(get_db)
):
    """
    Get item-level sales data grouped by hour for menu time popularity analysis.
    Returns: [{'item_name': str, 'quantity': int, 'revenue': float, 'hour': int}]
    """
    try:
        # Deduplicate orders by orderID, keep latest version
        order_id_path = func.jsonb_extract_path_text(PetpoojaWebhookEvent.content, 'properties', 'Order', 'orderID')
        
        subquery_filters = [PetpoojaWebhookEvent.content['event'].astext == 'orderdetails']
        if restaurant_id:
            subquery_filters.append(func.jsonb_extract_path_text(PetpoojaWebhookEvent.content, 'properties', 'Restaurant', 'restID') == restaurant_id)
        
        subquery = db.query(func.max(PetpoojaWebhookEvent.id))\
            .filter(*subquery_filters)\
            .group_by(order_id_path)\
            .subquery()
        
        # Extract item-level data with hour
        created_on_path = func.jsonb_extract_path_text(PetpoojaWebhookEvent.content, 'properties', 'Order', 'created_on')
        hour_col = func.extract('hour', cast(created_on_path, DateTime))
        
        restaurant_filter = ""
        if restaurant_id:
            restaurant_filter = "AND content->'properties'->'Restaurant'->>'restID' = :restaurant_id"
        
        sql_query = text(f"""
            WITH unique_orders AS (
                SELECT content
                FROM petpooja_webhook_events
                WHERE id IN (
                    SELECT MAX(id)
                    FROM petpooja_webhook_events
                    WHERE content->>'event' = 'orderdetails'
                    {restaurant_filter}
                    GROUP BY content->'properties'->'Order'->>'orderID'
                )
                AND created_at >= CURRENT_DATE - INTERVAL ':days days'
            )
            SELECT 
                item->>'name' as item_name,
                SUM(CAST(COALESCE(NULLIF(item->>'quantity', ''), '0') AS NUMERIC)) as quantity,
                SUM(CAST(COALESCE(NULLIF(item->>'total', ''), '0') AS NUMERIC)) as revenue,
                EXTRACT(HOUR FROM CAST(content->'properties'->'Order'->>'created_on' AS TIMESTAMP))::INTEGER as hour
            FROM 
                unique_orders,
                jsonb_array_elements(content->'properties'->'OrderItem') item
            WHERE 
                jsonb_typeof(content->'properties'->'OrderItem') = 'array'
                AND item->>'name' IS NOT NULL 
                AND item->>'name' != ''
            GROUP BY 
                item->>'name', hour
            ORDER BY 
                item_name, hour
        """)
        
        params = {"days": days}
        if restaurant_id:
            params["restaurant_id"] = restaurant_id
        results = db.execute(sql_query, params).fetchall()
        
        return [
            {
                "item_name": r[0],
                "quantity": float(r[1]) if r[1] else 0,
                "revenue": float(r[2]) if r[2] else 0.0,
                "hour": int(r[3]) if r[3] else 0
            } for r in results
        ]
    except Exception as e:
        logging.error(f"Error in items-by-hour: {e}")
        return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
