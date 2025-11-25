#!/usr/bin/env python3
"""
Export ROI configuration from local database to apply on server
Run this on LOCAL machine to get SQL commands to run on SERVER
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json

DATABASE_URL = 'postgresql://postgres:Tneural01@127.0.0.1:5432/sakshi'
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

print("=" * 80)
print("ROI CONFIGURATION EXPORT - Copy these commands and run on SERVER")
print("=" * 80)
print()

with SessionLocal() as db:
    result = db.execute(text("""
        SELECT channel_id, app_name, roi_points 
        FROM roi_configs 
        ORDER BY app_name, channel_id
    """))
    rows = result.fetchall()
    
    if rows:
        print("-- SQL commands to run on SERVER database:")
        print()
        
        for row in rows:
            channel_id = row[0]
            app_name = row[1]
            roi_points = row[2]  # Already JSON string
            
            # Escape single quotes in JSON
            roi_json = roi_points.replace("'", "''")
            
            sql = f"""
-- Update {app_name} for {channel_id}
INSERT INTO roi_configs (channel_id, app_name, roi_points) 
VALUES ('{channel_id}', '{app_name}', '{roi_json}')
ON CONFLICT (channel_id, app_name) 
DO UPDATE SET roi_points = EXCLUDED.roi_points;
"""
            print(sql)
        
        print()
        print("=" * 80)
        print("PYTHON SCRIPT to run on SERVER:")
        print("=" * 80)
        print()
        print("""
# Create this file as update_roi_server.py on SERVER and run it:

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json

DATABASE_URL = 'postgresql://postgres:Tneural01@127.0.0.1:5432/sakshi'
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

roi_configs = [
""")
        
        for row in rows:
            channel_id = row[0]
            app_name = row[1]
            roi_data = json.loads(row[2])
            
            print(f"    {{")
            print(f"        'channel_id': '{channel_id}',")
            print(f"        'app_name': '{app_name}',")
            print(f"        'roi_points': {json.dumps(roi_data)}")
            print(f"    }},")
        
        print("""]

with SessionLocal() as db:
    for config in roi_configs:
        stmt = text(\"\"\"
            INSERT INTO roi_configs (channel_id, app_name, roi_points) 
            VALUES (:cid, :an, :rp)
            ON CONFLICT (channel_id, app_name) 
            DO UPDATE SET roi_points = EXCLUDED.roi_points;
        \"\"\")
        db.execute(stmt, {
            'cid': config['channel_id'],
            'an': config['app_name'],
            'rp': json.dumps(config['roi_points'])
        })
        print(f"✅ Updated {config['app_name']} for {config['channel_id']}")
    
    db.commit()
    print("\\n✅ All ROI configurations updated on server!")
""")
    else:
        print("❌ No ROI configurations found in local database!")
