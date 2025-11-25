#!/usr/bin/env python3
"""
Run this script on SERVER to update ROI configurations from local machine
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json

DATABASE_URL = 'postgresql://postgres:Tneural01@127.0.0.1:5432/sakshi'
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# ROI configurations exported from local machine
roi_configs = [
    {
        'channel_id': 'cam_3df702bb28',
        'app_name': 'PeopleCounter',
        'roi_points': {"line_position": 0.38}
    },
    {
        'channel_id': 'cam_f822b0bf4e',
        'app_name': 'QueueMonitor',
        'roi_points': {
            "main": [
                [0.5549999952316285, 0.5744444105360244],
                [0.4456249952316284, 0.5272221883138021],
                [0.3081249952316284, 0.3105555216471354],
                [0.08624999523162842, 0.4272221883138021],
                [0.19249999523162842, 0.7938888549804688]
            ],
            "secondary": [
                [0.5924999952316284, 0.5355555216471354],
                [0.49874999523162844, 0.502222188313802],
                [0.3487499952316284, 0.31333329942491317],
                [0.38156249523162844, 0.3105555216471354],
                [0.3940624952316284, 0.2883332994249132],
                [0.5003124952316285, 0.26888885498046877],
                [0.6721874952316285, 0.4633332994249132]
            ]
        }
    }
]

print("Updating ROI configurations on server...")
print()

with SessionLocal() as db:
    for config in roi_configs:
        stmt = text("""
            INSERT INTO roi_configs (channel_id, app_name, roi_points) 
            VALUES (:cid, :an, :rp)
            ON CONFLICT (channel_id, app_name) 
            DO UPDATE SET roi_points = EXCLUDED.roi_points;
        """)
        db.execute(stmt, {
            'cid': config['channel_id'],
            'an': config['app_name'],
            'rp': json.dumps(config['roi_points'])
        })
        print(f"✅ Updated {config['app_name']} for {config['channel_id']}")
    
    db.commit()
    print("\n✅ All ROI configurations updated successfully!")
    print("\nNext steps:")
    print("1. Restart edit-004.py on server: sudo systemctl restart sakshi-ai")
    print("2. Check logs: journalctl -u sakshi-ai -f")
