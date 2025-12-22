#!/usr/bin/env python3
"""
Interactive Queue Monitor ROI Reconfiguration Tool
Allows you to easily update the ROI polygons for your Queue Monitor
"""

import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Database configuration
DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/sakshi"
CHANNEL_ID = "cam_f822b0bf4e"

def view_current_roi():
    """View the current ROI configuration"""
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        query = text("""
            SELECT channel_id, app_name, roi_points 
            FROM roi_configs 
            WHERE channel_id = :channel_id AND app_name = 'QueueMonitor'
        """)
        
        result = session.execute(query, {'channel_id': CHANNEL_ID}).fetchone()
        
        if result:
            print(f"\n📊 Current ROI Configuration for {CHANNEL_ID}:")
            print(f"   App Name: {result[1]}")
            roi_data = json.loads(result[2])
            
            print(f"\n   🔵 Main ROI (Queue Area): {len(roi_data.get('main', []))} points")
            for i, point in enumerate(roi_data.get('main', []), 1):
                print(f"      Point {i}: [{point[0]:.4f}, {point[1]:.4f}]")
            
            print(f"\n   🟢 Secondary ROI (Counter Area): {len(roi_data.get('secondary', []))} points")
            for i, point in enumerate(roi_data.get('secondary', []), 1):
                print(f"      Point {i}: [{point[0]:.4f}, {point[1]:.4f}]")
            
            return roi_data
        else:
            print(f"❌ No ROI configuration found for {CHANNEL_ID}")
            return None
            
    except Exception as e:
        print(f"❌ Error viewing ROI: {e}")
        return None
    finally:
        session.close()

def update_roi(new_roi_config):
    """Update the Queue Monitor ROI in the database"""
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Convert to JSON string
        roi_json = json.dumps(new_roi_config)
        
        # Update the database
        query = text("""
            UPDATE roi_configs 
            SET roi_points = :roi_points
            WHERE channel_id = :channel_id AND app_name = 'QueueMonitor'
        """)
        
        result = session.execute(query, {
            'roi_points': roi_json,
            'channel_id': CHANNEL_ID
        })
        session.commit()
        
        if result.rowcount > 0:
            print(f"\n✅ Successfully updated Queue Monitor ROI for channel {CHANNEL_ID}")
            print(f"   🔵 Main ROI points: {len(new_roi_config['main'])}")
            print(f"   🟢 Secondary ROI points: {len(new_roi_config['secondary'])}")
            return True
        else:
            print(f"❌ No ROI configuration found for channel {CHANNEL_ID}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating ROI: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def get_polygon_input(roi_name, color_emoji):
    """Get polygon points from user input"""
    print(f"\n{color_emoji} Enter {roi_name} polygon points:")
    print("   Coordinates should be between 0.0 (left/top) and 1.0 (right/bottom)")
    print("   Format: x,y (e.g., 0.5,0.3)")
    print("   Type 'done' when finished entering points")
    
    points = []
    point_num = 1
    
    while True:
        try:
            user_input = input(f"   Point {point_num} (or 'done'): ").strip()
            
            if user_input.lower() == 'done':
                if len(points) < 3:
                    print(f"   ⚠️  Need at least 3 points for a polygon. You have {len(points)}.")
                    continue
                break
            
            # Parse x,y
            x_str, y_str = user_input.split(',')
            x, y = float(x_str.strip()), float(y_str.strip())
            
            # Validate range
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                print(f"   ⚠️  Coordinates must be between 0.0 and 1.0")
                continue
            
            points.append([x, y])
            print(f"   ✓ Added point {point_num}: [{x:.4f}, {y:.4f}]")
            point_num += 1
            
        except ValueError:
            print(f"   ⚠️  Invalid format. Use: x,y (e.g., 0.5,0.3)")
        except KeyboardInterrupt:
            print("\n   ⚠️  Input cancelled")
            return None
    
    return points

def preset_simple_rectangle():
    """Preset: Simple rectangular regions"""
    return {
        "main": [
            [0.20, 0.40],  # Top-left
            [0.60, 0.40],  # Top-right
            [0.60, 0.85],  # Bottom-right
            [0.20, 0.85]   # Bottom-left
        ],
        "secondary": [
            [0.65, 0.30],  # Top-left
            [0.85, 0.30],  # Top-right
            [0.85, 0.65],  # Bottom-right
            [0.65, 0.65]   # Bottom-left
        ]
    }

def preset_large_queue():
    """Preset: Large queue area covering most of frame"""
    return {
        "main": [
            [0.10, 0.30],
            [0.70, 0.30],
            [0.70, 0.90],
            [0.10, 0.90]
        ],
        "secondary": [
            [0.75, 0.25],
            [0.95, 0.25],
            [0.95, 0.70],
            [0.75, 0.70]
        ]
    }

def preset_custom_from_current():
    """Use current ROI as template for editing"""
    current = view_current_roi()
    if current:
        return current
    return None

def interactive_mode():
    """Interactive configuration mode"""
    print("\n" + "=" * 70)
    print("🎯 Queue Monitor ROI Configuration Tool - Interactive Mode")
    print("=" * 70)
    
    while True:
        print("\n📋 Choose an option:")
        print("   1. View current ROI configuration")
        print("   2. Enter custom ROI manually")
        print("   3. Use preset: Simple rectangles (default)")
        print("   4. Use preset: Large queue area")
        print("   5. Keep current ROI (no changes)")
        print("   0. Exit")
        
        choice = input("\nYour choice: ").strip()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
        
        elif choice == '1':
            view_current_roi()
            continue
        
        elif choice == '2':
            # Manual input
            print("\n📝 Manual ROI Configuration")
            main_points = get_polygon_input("Main ROI (Queue Area)", "🔵")
            if main_points is None:
                continue
            
            secondary_points = get_polygon_input("Secondary ROI (Counter Area)", "🟢")
            if secondary_points is None:
                continue
            
            new_roi = {
                "main": main_points,
                "secondary": secondary_points
            }
            
        elif choice == '3':
            print("\n📦 Using preset: Simple rectangles")
            new_roi = preset_simple_rectangle()
            print("\n   🔵 Main ROI (Queue): Left side rectangle")
            print("   🟢 Secondary ROI (Counter): Right side rectangle")
            
        elif choice == '4':
            print("\n📦 Using preset: Large queue area")
            new_roi = preset_large_queue()
            print("\n   🔵 Main ROI (Queue): Large area covering most of frame")
            print("   🟢 Secondary ROI (Counter): Right side area")
            
        elif choice == '5':
            print("\n✅ Keeping current configuration. No changes made.")
            break
        
        else:
            print("⚠️  Invalid choice. Please try again.")
            continue
        
        # Preview the new configuration
        print("\n" + "=" * 70)
        print("📋 Preview of New ROI Configuration:")
        print("=" * 70)
        print(f"\n🔵 Main ROI (Queue Area): {len(new_roi['main'])} points")
        for i, point in enumerate(new_roi['main'], 1):
            print(f"   Point {i}: [{point[0]:.4f}, {point[1]:.4f}]")
        
        print(f"\n🟢 Secondary ROI (Counter Area): {len(new_roi['secondary'])} points")
        for i, point in enumerate(new_roi['secondary'], 1):
            print(f"   Point {i}: [{point[0]:.4f}, {point[1]:.4f}]")
        
        # Confirm update
        print("\n" + "=" * 70)
        confirm = input("\n💾 Save this configuration to database? (yes/no): ").strip().lower()
        
        if confirm in ['yes', 'y']:
            if update_roi(new_roi):
                print("\n" + "=" * 70)
                print("⚠️  IMPORTANT: Restart the application for changes to take effect!")
                print("   Run: sudo systemctl restart sakshi-ai.service")
                print("   Or stop and restart edit-004.py manually")
                print("=" * 70)
                break
            else:
                print("\n❌ Update failed. Please try again.")
        else:
            print("\n❌ Update cancelled. Going back to menu...")

if __name__ == "__main__":
    try:
        interactive_mode()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
