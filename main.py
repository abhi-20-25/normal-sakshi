import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# pip install fastapi "uvicorn[standard]" sqlalchemy psycopg2-binary
from fastapi import FastAPI, Request, HTTPException, status, Query
from sqlalchemy import create_engine, Column, Integer, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL")
PETPOOJA_API_TOKEN = os.getenv("PETPOOJA_API_TOKEN")

if not DATABASE_URL or not PETPOOJA_API_TOKEN:
    print("FATAL ERROR: Make sure DATABASE_URL and PETPOOJA_API_TOKEN are set.")
    sys.exit(1)

Base = declarative_base()

class PetpoojaWebhookEvent(Base):
    __tablename__ = "petpooja_webhook_events"
    id = Column(Integer, primary_key=True)
    content = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


app = FastAPI(title="PetPooja Webhook Receiver")


@app.get("/petpooja")
def create_petpooja_event(
    payload: dict,
    token: str = Query(...)
):
   
    if token != PETPOOJA_API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"status": "error", "message": "Authentication required"},
        )

    db = SessionLocal()
    try:
        db_event = PetpoojaWebhookEvent(content=payload)
        
        db.add(db_event)
        db.commit()
        db.refresh(db_event)
        
        return {
            "status": "success",
            "message": "Data saved successfully",
            "id": db_event.id
        }
    except Exception as e:
        db.rollback() 
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "error", "message": f"Server error: {e}"},
        )
    finally:
        db.close() 