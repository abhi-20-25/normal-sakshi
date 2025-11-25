from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json

DATABASE_URL = 'postgresql://postgres:Tneural01@127.0.0.1:5432/sakshi'
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

print("Checking PeopleCounter ROI...")
with SessionLocal() as db:
    result = db.execute(text("SELECT channel_id, app_name, roi_points FROM roi_configs WHERE app_name = 'PeopleCounter'"))
    rows = result.fetchall()
    
    if rows:
        for row in rows:
            print(f"\nChannel: {row[0]}")
            roi_data = json.loads(row[2])
            if 'line_position' in roi_data:
                print(f"✅ Line Position: {roi_data['line_position'] * 100:.1f}%")
            else:
                print("❌ No line_position field found")
            print(f"Full ROI data: {roi_data}")
    else:
        print("❌ No PeopleCounter ROI found")
