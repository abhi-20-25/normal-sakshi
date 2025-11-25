#!/usr/bin/env python3
"""Update PeopleCounter line position to 38% (0.38)"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json

DATABASE_URL = 'postgresql://postgres:Tneural01@127.0.0.1:5432/sakshi'
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

print("Updating PeopleCounter line position to 38%...")

with SessionLocal() as db:
    # Update the line position
    roi_data = {"line_position": 0.38}
    
    stmt = text("""
        INSERT INTO roi_configs (channel_id, app_name, roi_points) 
        VALUES (:cid, :an, :rp)
        ON CONFLICT (channel_id, app_name) 
        DO UPDATE SET roi_points = EXCLUDED.roi_points;
    """)
    
    db.execute(stmt, {
        'cid': 'cam_3df702bb28',
        'an': 'PeopleCounter',
        'rp': json.dumps(roi_data)
    })
    db.commit()
    
    print("✅ Line position updated to 38% in database")
    
    # Verify
    result = db.execute(text("""
        SELECT roi_points FROM roi_configs 
        WHERE channel_id = 'cam_3df702bb28' AND app_name = 'PeopleCounter'
    """))
    row = result.fetchone()
    if row:
        data = json.loads(row[0])
        print(f"Verified: {data}")
        print(f"Line position: {data.get('line_position', 'N/A') * 100:.0f}%")
