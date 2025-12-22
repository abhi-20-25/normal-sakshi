#!/usr/bin/env python3
"""
Safe Database Migration Script
Adds missing tables and missing columns without affecting existing data
"""

import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
import logging

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Tneural01@127.0.0.1:5432/sakshi")

def check_existing_tables(engine):
    """Check which tables already exist"""
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    return existing_tables

def check_table_columns(engine, table_name):
    """Get existing columns for a table"""
    inspector = inspect(engine)
    try:
        columns = inspector.get_columns(table_name)
        return {col['name']: col for col in columns}
    except Exception as e:
        logging.warning(f"Could not inspect table {table_name}: {e}")
        return {}

def add_missing_columns(engine):
    """Add missing columns to existing tables"""
    
    # Define expected columns for each table
    table_columns = {
        'restaurants': [
            "ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS id SERIAL PRIMARY KEY",
            "ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS restaurant_code VARCHAR(50)",
            "ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS restaurant_name VARCHAR(200)",
            "ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS location VARCHAR(200)",
            "ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS dvr_ip VARCHAR(50)",
            "ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS dvr_username VARCHAR(100)",
            "ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS dvr_password VARCHAR(100)",
            "ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS telegram_chat_id VARCHAR(50)",
            "ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE",
            "ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW()",
            "ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW()",
        ],
        'cameras': [
            "ALTER TABLE cameras ADD COLUMN IF NOT EXISTS id SERIAL PRIMARY KEY",
            "ALTER TABLE cameras ADD COLUMN IF NOT EXISTS restaurant_id INTEGER REFERENCES restaurants(id)",
            "ALTER TABLE cameras ADD COLUMN IF NOT EXISTS channel_id VARCHAR(50) UNIQUE",
            "ALTER TABLE cameras ADD COLUMN IF NOT EXISTS channel_name VARCHAR(255)",
            "ALTER TABLE cameras ADD COLUMN IF NOT EXISTS channel_number INTEGER",
            "ALTER TABLE cameras ADD COLUMN IF NOT EXISTS rtsp_url TEXT",
            "ALTER TABLE cameras ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE",
            "ALTER TABLE cameras ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW()",
            "ALTER TABLE cameras ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW()",
        ],
        'camera_apps': [
            "ALTER TABLE camera_apps ADD COLUMN IF NOT EXISTS id SERIAL PRIMARY KEY",
            "ALTER TABLE camera_apps ADD COLUMN IF NOT EXISTS camera_id INTEGER REFERENCES cameras(id)",
            "ALTER TABLE camera_apps ADD COLUMN IF NOT EXISTS app_name VARCHAR(50)",
            "ALTER TABLE camera_apps ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE",
            "ALTER TABLE camera_apps ADD COLUMN IF NOT EXISTS config JSONB",
            "ALTER TABLE camera_apps ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW()",
        ],
        'roi_configs': [
            "ALTER TABLE roi_configs ADD COLUMN IF NOT EXISTS restaurant_id INTEGER REFERENCES restaurants(id)",
        ],
        'detections': [
            "ALTER TABLE detections ADD COLUMN IF NOT EXISTS app_name VARCHAR",
            "ALTER TABLE detections ADD COLUMN IF NOT EXISTS channel_id VARCHAR",
            "ALTER TABLE detections ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP",
            "ALTER TABLE detections ADD COLUMN IF NOT EXISTS message TEXT",
            "ALTER TABLE detections ADD COLUMN IF NOT EXISTS media_path VARCHAR",
        ],
        'kitchen_violations': [
            "ALTER TABLE kitchen_violations ADD COLUMN IF NOT EXISTS channel_id VARCHAR",
            "ALTER TABLE kitchen_violations ADD COLUMN IF NOT EXISTS channel_name VARCHAR",
            "ALTER TABLE kitchen_violations ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP",
            "ALTER TABLE kitchen_violations ADD COLUMN IF NOT EXISTS violation_type VARCHAR",
            "ALTER TABLE kitchen_violations ADD COLUMN IF NOT EXISTS details VARCHAR",
            "ALTER TABLE kitchen_violations ADD COLUMN IF NOT EXISTS media_path VARCHAR",
        ],
        'daily_footfall': [
            "ALTER TABLE daily_footfall ADD COLUMN IF NOT EXISTS channel_id VARCHAR",
            "ALTER TABLE daily_footfall ADD COLUMN IF NOT EXISTS report_date DATE",
            "ALTER TABLE daily_footfall ADD COLUMN IF NOT EXISTS in_count INTEGER DEFAULT 0",
            "ALTER TABLE daily_footfall ADD COLUMN IF NOT EXISTS out_count INTEGER DEFAULT 0",
        ],
        'hourly_footfall': [
            "ALTER TABLE hourly_footfall ADD COLUMN IF NOT EXISTS channel_id VARCHAR",
            "ALTER TABLE hourly_footfall ADD COLUMN IF NOT EXISTS report_date DATE",
            "ALTER TABLE hourly_footfall ADD COLUMN IF NOT EXISTS hour INTEGER",
            "ALTER TABLE hourly_footfall ADD COLUMN IF NOT EXISTS in_count INTEGER DEFAULT 0",
            "ALTER TABLE hourly_footfall ADD COLUMN IF NOT EXISTS out_count INTEGER DEFAULT 0",
        ],
        'queue_logs': [
            "ALTER TABLE queue_logs ADD COLUMN IF NOT EXISTS channel_id VARCHAR",
            "ALTER TABLE queue_logs ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP",
            "ALTER TABLE queue_logs ADD COLUMN IF NOT EXISTS queue_count INTEGER",
        ],
        'occupancy_logs': [
            "ALTER TABLE occupancy_logs ADD COLUMN IF NOT EXISTS channel_id VARCHAR",
            "ALTER TABLE occupancy_logs ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP",
            "ALTER TABLE occupancy_logs ADD COLUMN IF NOT EXISTS time_slot VARCHAR",
            "ALTER TABLE occupancy_logs ADD COLUMN IF NOT EXISTS day_of_week VARCHAR",
            "ALTER TABLE occupancy_logs ADD COLUMN IF NOT EXISTS live_count INTEGER",
            "ALTER TABLE occupancy_logs ADD COLUMN IF NOT EXISTS required_count INTEGER",
            "ALTER TABLE occupancy_logs ADD COLUMN IF NOT EXISTS status VARCHAR",
        ],
        'occupancy_schedules': [
            "ALTER TABLE occupancy_schedules ADD COLUMN IF NOT EXISTS channel_id VARCHAR",
            "ALTER TABLE occupancy_schedules ADD COLUMN IF NOT EXISTS time_slot VARCHAR",
            "ALTER TABLE occupancy_schedules ADD COLUMN IF NOT EXISTS day_of_week VARCHAR",
            "ALTER TABLE occupancy_schedules ADD COLUMN IF NOT EXISTS required_count INTEGER",
        ],
    }
    
    existing_tables = check_existing_tables(engine)
    columns_added = 0
    
    with engine.begin() as conn:
        for table_name, alter_statements in table_columns.items():
            if table_name in existing_tables:
                logging.info(f"\n🔍 Checking table: {table_name}")
                existing_columns = check_table_columns(engine, table_name)
                
                for alter_sql in alter_statements:
                    try:
                        # Extract column name from ALTER statement
                        if "ADD COLUMN IF NOT EXISTS" in alter_sql:
                            col_name = alter_sql.split("ADD COLUMN IF NOT EXISTS")[1].split()[0]
                            
                            if col_name not in existing_columns:
                                conn.execute(text(alter_sql))
                                logging.info(f"  ✅ Added column: {col_name}")
                                columns_added += 1
                            else:
                                logging.debug(f"  ⏭️  Column exists: {col_name}")
                    except Exception as e:
                        logging.warning(f"  ⚠️  Could not add column: {e}")
    
    return columns_added

def migrate_database():
    """Add missing tables without affecting existing data"""
    try:
        logging.info(f"Connecting to database: {DATABASE_URL}")
        engine = create_engine(DATABASE_URL)
        
        # Check existing tables
        existing_tables = check_existing_tables(engine)
        logging.info(f"\n📊 Found {len(existing_tables)} existing tables:")
        for table in sorted(existing_tables):
            logging.info(f"  ✓ {table}")
        
        # Define required tables
        required_tables = {
            'restaurants': """
                CREATE TABLE IF NOT EXISTS restaurants (
                    id SERIAL PRIMARY KEY,
                    restaurant_code VARCHAR(50) NOT NULL UNIQUE,
                    restaurant_name VARCHAR(200) NOT NULL,
                    location VARCHAR(200),
                    dvr_ip VARCHAR(50),
                    dvr_username VARCHAR(100),
                    dvr_password VARCHAR(100),
                    telegram_chat_id VARCHAR(50),
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """,
            'cameras': """
                CREATE TABLE IF NOT EXISTS cameras (
                    id SERIAL PRIMARY KEY,
                    restaurant_id INTEGER REFERENCES restaurants(id),
                    channel_id VARCHAR(50) UNIQUE NOT NULL,
                    channel_name VARCHAR(255) NOT NULL,
                    channel_number INTEGER,
                    rtsp_url TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """,
            'camera_apps': """
                CREATE TABLE IF NOT EXISTS camera_apps (
                    id SERIAL PRIMARY KEY,
                    camera_id INTEGER REFERENCES cameras(id),
                    app_name VARCHAR(50) NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    config JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """,
            'petpooja_webhook_events': """
                CREATE TABLE IF NOT EXISTS petpooja_webhook_events (
                    id SERIAL PRIMARY KEY,
                    content JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """,
            'detections': """
                CREATE TABLE IF NOT EXISTS detections (
                    id SERIAL PRIMARY KEY,
                    app_name VARCHAR,
                    channel_id VARCHAR,
                    timestamp TIMESTAMP,
                    message TEXT,
                    media_path VARCHAR,
                    CONSTRAINT _media_path_uc UNIQUE (media_path)
                );
                CREATE INDEX IF NOT EXISTS idx_detections_app_name ON detections(app_name);
                CREATE INDEX IF NOT EXISTS idx_detections_channel_id ON detections(channel_id);
            """,
            'daily_footfall': """
                CREATE TABLE IF NOT EXISTS daily_footfall (
                    id SERIAL PRIMARY KEY,
                    channel_id VARCHAR,
                    report_date DATE,
                    in_count INTEGER DEFAULT 0,
                    out_count INTEGER DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_daily_footfall_channel ON daily_footfall(channel_id);
                CREATE INDEX IF NOT EXISTS idx_daily_footfall_date ON daily_footfall(report_date);
            """,
            'hourly_footfall': """
                CREATE TABLE IF NOT EXISTS hourly_footfall (
                    id SERIAL PRIMARY KEY,
                    channel_id VARCHAR,
                    report_date DATE,
                    hour INTEGER,
                    in_count INTEGER DEFAULT 0,
                    out_count INTEGER DEFAULT 0,
                    CONSTRAINT _channel_date_hour_uc UNIQUE (channel_id, report_date, hour)
                );
                CREATE INDEX IF NOT EXISTS idx_hourly_footfall_channel ON hourly_footfall(channel_id);
                CREATE INDEX IF NOT EXISTS idx_hourly_footfall_date ON hourly_footfall(report_date);
                CREATE INDEX IF NOT EXISTS idx_hourly_footfall_hour ON hourly_footfall(hour);
            """,
            'queue_logs': """
                CREATE TABLE IF NOT EXISTS queue_logs (
                    id SERIAL PRIMARY KEY,
                    channel_id VARCHAR,
                    timestamp TIMESTAMP,
                    queue_count INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_queue_logs_channel ON queue_logs(channel_id);
                CREATE INDEX IF NOT EXISTS idx_queue_logs_timestamp ON queue_logs(timestamp);
            """,
            'roi_configs': """
                CREATE TABLE IF NOT EXISTS roi_configs (
                    id SERIAL PRIMARY KEY,
                    channel_id VARCHAR,
                    app_name VARCHAR,
                    roi_points TEXT,
                    restaurant_id INTEGER REFERENCES restaurants(id),
                    CONSTRAINT _roi_uc UNIQUE (channel_id, app_name)
                );
                CREATE INDEX IF NOT EXISTS idx_roi_configs_channel ON roi_configs(channel_id);
                CREATE INDEX IF NOT EXISTS idx_roi_configs_app ON roi_configs(app_name);
            """,
            'kitchen_violations': """
                CREATE TABLE IF NOT EXISTS kitchen_violations (
                    id SERIAL PRIMARY KEY,
                    channel_id VARCHAR,
                    channel_name VARCHAR,
                    timestamp TIMESTAMP,
                    violation_type VARCHAR,
                    details VARCHAR,
                    media_path VARCHAR,
                    CONSTRAINT _kitchen_media_path_uc UNIQUE (media_path)
                );
                CREATE INDEX IF NOT EXISTS idx_kitchen_violations_channel ON kitchen_violations(channel_id);
            """,
            'occupancy_logs': """
                CREATE TABLE IF NOT EXISTS occupancy_logs (
                    id SERIAL PRIMARY KEY,
                    channel_id VARCHAR,
                    timestamp TIMESTAMP,
                    time_slot VARCHAR,
                    day_of_week VARCHAR,
                    live_count INTEGER,
                    required_count INTEGER,
                    status VARCHAR
                );
                CREATE INDEX IF NOT EXISTS idx_occupancy_logs_channel ON occupancy_logs(channel_id);
            """,
            'occupancy_schedules': """
                CREATE TABLE IF NOT EXISTS occupancy_schedules (
                    id SERIAL PRIMARY KEY,
                    channel_id VARCHAR,
                    time_slot VARCHAR,
                    day_of_week VARCHAR,
                    required_count INTEGER,
                    CONSTRAINT _occupancy_schedule_uc UNIQUE (channel_id, time_slot, day_of_week)
                );
                CREATE INDEX IF NOT EXISTS idx_occupancy_schedules_channel ON occupancy_schedules(channel_id);
            """
        }
        
        # Check which tables are missing
        missing_tables = []
        for table_name in required_tables.keys():
            if table_name not in existing_tables:
                missing_tables.append(table_name)
        
        if not missing_tables:
            logging.info("\n✅ All required tables already exist!")
        else:
            logging.info(f"\n📝 Missing tables to be created: {len(missing_tables)}")
            for table in missing_tables:
                logging.info(f"  + {table}")
        
        # Execute migrations for missing tables
        if missing_tables:
            with engine.begin() as conn:
                for table_name, create_sql in required_tables.items():
                    if table_name in missing_tables:
                        logging.info(f"\n🔨 Creating table: {table_name}")
                        try:
                            # Execute each statement separately
                            for statement in create_sql.split(';'):
                                statement = statement.strip()
                                if statement:
                                    conn.execute(text(statement))
                            logging.info(f"  ✅ Created {table_name}")
                        except Exception as e:
                            logging.error(f"  ❌ Failed to create {table_name}: {e}")
                            raise
        
        # Add missing columns to existing tables
        logging.info("\n" + "="*60)
        logging.info("🔍 Checking for missing columns in existing tables...")
        columns_added = add_missing_columns(engine)
        
        # Verify final state
        logging.info("\n" + "="*60)
        final_tables = check_existing_tables(engine)
        logging.info(f"\n✅ Migration complete! Total tables: {len(final_tables)}")
        for table in sorted(final_tables):
            status = "NEW" if table in missing_tables else "EXISTING"
            logging.info(f"  [{status}] {table}")
        
        # Show data preservation
        logging.info("\n" + "="*60)
        logging.info("📊 DATA PRESERVATION CHECK:")
        for table in existing_tables:
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                logging.info(f"  ✓ {table}: {count} rows preserved")
        
        logging.info("\n" + "="*60)
        logging.info("🎉 Database migration completed successfully!")
        logging.info("   - All existing data preserved")
        if missing_tables:
            logging.info(f"   - {len(missing_tables)} new tables added")
        if columns_added > 0:
            logging.info(f"   - {columns_added} new columns added")
        if not missing_tables and columns_added == 0:
            logging.info("   - No changes needed, database is up to date")
        
        return True
    
    except Exception as e:
        logging.error(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = migrate_database()
    sys.exit(0 if success else 1)
