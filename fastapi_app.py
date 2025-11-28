import os
import sys
import logging
from typing import Optional, List, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException, status, Query, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, DateTime, func, desc, Float, Text, cast
from sqlalchemy.orm import aliased
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from pydantic import BaseModel, Field
from sqlalchemy import Date

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


# ANALYTICS - Total Sales Per Day
@app.get("/analytics/sales-daily", response_model=List[SalesStats], tags=["Analytics"])
def get_daily_sales(
    days: int = Query(30, description="Number of past days to analyze"),
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    date_col = cast(
        func.jsonb_extract_path_text(PetpoojaWebhookEvent.content, 'properties', 'Order', 'created_on'), 
        DateTime
    ).cast(Date) 

    total_col = cast(
        func.jsonb_extract_path_text(PetpoojaWebhookEvent.content, 'properties', 'Order', 'total'),
        Float
    )

    results = db.query(
        date_col.label('date'),
        func.sum(total_col).label('total_sales'),
        func.count(PetpoojaWebhookEvent.id).label('order_count')
    ).filter(
        PetpoojaWebhookEvent.content['event'].astext == 'orderdetails'
    ).group_by(
        date_col
    ).order_by(
        desc('date')
    ).limit(days).all()

    return [
        SalesStats(
            date=str(row.date), 
            total_sales=row.total_sales or 0, 
            order_count=row.order_count
        ) for row in results
    ]

# ANALYTICS - Payment Modes
@app.get("/analytics/payment-modes", response_model=List[PaymentStats], tags=["Analytics"])
def get_payment_stats(
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    
    payment_type_col = func.jsonb_extract_path_text(PetpoojaWebhookEvent.content, 'properties', 'Order', 'payment_type')
    total_col = cast(func.jsonb_extract_path_text(PetpoojaWebhookEvent.content, 'properties', 'Order', 'total'), Float)

    results = db.query(
        payment_type_col.label('method'),
        func.sum(total_col).label('amount'),
        func.count(PetpoojaWebhookEvent.id).label('count')
    ).filter(
        PetpoojaWebhookEvent.content['event'].astext == 'orderdetails'
    ).group_by(
        payment_type_col
    ).all()

    return [
        PaymentStats(
            method=row.method if row.method else "Unknown",
            amount=row.amount or 0,
            count=row.count
        ) for row in results
    ]

# ANALYTICS - Order Types (Online vs Offline)
@app.get("/analytics/order-types", response_model=List[OrderTypeStats], tags=["Analytics"])
def get_order_type_stats(
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    
    # order_from usually indicates POS (Offline) vs Zomato/Swiggy (Online)
    source_col = func.jsonb_extract_path_text(PetpoojaWebhookEvent.content, 'properties', 'Order', 'order_from')
    type_col = func.jsonb_extract_path_text(PetpoojaWebhookEvent.content, 'properties', 'Order', 'order_type')
    total_col = cast(func.jsonb_extract_path_text(PetpoojaWebhookEvent.content, 'properties', 'Order', 'total'), Float)

    results = db.query(
        source_col.label('order_source'),
        type_col.label('order_type'),
        func.count(PetpoojaWebhookEvent.id).label('count'),
        func.sum(total_col).label('total_sales')
    ).filter(
        PetpoojaWebhookEvent.content['event'].astext == 'orderdetails'
    ).group_by(
        source_col, type_col
    ).all()

    return [
        OrderTypeStats(
            order_source=row.order_source if row.order_source else "Unknown",
            order_type=row.order_type if row.order_type else "Unknown",
            count=row.count,
            total_sales=row.total_sales or 0
        ) for row in results
    ]

# ANALYTICS - Top Selling Items
@app.get("/analytics/top-items", response_model=List[TopItem], tags=["Analytics"])
def get_top_items(
    limit: int = 5,
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    
    items = func.jsonb_array_elements(PetpoojaWebhookEvent.content['properties']['OrderItem']).alias('item')
    
    item_name = items.column.op('->>')('name')
    item_qty = cast(items.column.op('->>')('quantity'), Integer)
    item_total = cast(items.column.op('->>')('total'), Float)

    results = db.query(
        item_name.label('name'),
        func.sum(item_qty).label('quantity_sold'),
        func.sum(item_total).label('total_revenue')
    ).select_from(
        PetpoojaWebhookEvent
    ).join(
        items, 
        PetpoojaWebhookEvent.content['event'].astext == 'orderdetails',
        isouter=True
    ).group_by(
        item_name
    ).order_by(
        desc('quantity_sold')
    ).limit(limit).all()

    return [
        TopItem(
            name=row.name if row.name else "Unknown",
            quantity_sold=row.quantity_sold or 0,
            total_revenue=row.total_revenue or 0
        ) for row in results
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
