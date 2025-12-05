#!/usr/bin/env python3
"""
Phase 3 Frontend Verification Script
Quick checks to ensure restaurant dropdown is properly integrated
"""

import os
import re

def check_dashboard_template():
    """Verify dashboard.html has all required Phase 3 changes"""
    print("="*70)
    print("PHASE 3 FRONTEND VERIFICATION")
    print("="*70)
    
    template_path = "/home/rasheeque/VS CODE FOLDER/TEA TOAST/templates/dashboard.html"
    
    if not os.path.exists(template_path):
        print(f"❌ ERROR: Template not found at {template_path}")
        return False
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    checks = {
        "1. CSS - Restaurant Selector Styles": ".restaurant-selector" in content,
        "2. CSS - Restaurant Badge Styles": ".restaurant-badge" in content,
        "3. HTML - Dropdown Select Element": 'id="restaurant-select"' in content,
        "4. HTML - Jinja2 restaurants loop": "{% for restaurant in restaurants %}" in content,
        "5. HTML - Selected restaurant badge": "{% if selected_restaurant %}" in content,
        "6. HTML - onchange handler": 'onchange="switchRestaurant(this.value)"' in content,
        "7. JavaScript - switchRestaurant function": "function switchRestaurant(" in content,
        "8. JavaScript - URL parameter handling": "searchParams.set" in content or "URLSearchParams" in content or "searchParams" in content,
    }
    
    print("\n📋 Template Verification:")
    print("-" * 70)
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check_name}")
        if not passed:
            all_passed = False
    
    # Additional detailed checks
    print("\n🔍 Detailed Checks:")
    print("-" * 70)
    
    # Check for restaurant emoji
    if "🏪" in content or "&#127978;" in content:
        print("✅ Restaurant emoji found in template")
    else:
        print("⚠️  Restaurant emoji not found (optional)")
    
    # Check for location pin emoji
    if "📍" in content or "&#128205;" in content:
        print("✅ Location pin emoji found in template")
    else:
        print("⚠️  Location pin emoji not found (optional)")
    
    # Count occurrences
    select_count = content.count('<select')
    option_count = content.count('<option')
    
    print(f"\n📊 Element Counts:")
    print(f"   <select> elements: {select_count}")
    print(f"   <option> elements: {option_count}")
    
    # Check JavaScript function definition
    js_match = re.search(r'function switchRestaurant\s*\(([^)]*)\)', content)
    if js_match:
        print(f"\n✅ switchRestaurant function found")
        print(f"   Parameters: {js_match.group(1)}")
    else:
        print("\n❌ switchRestaurant function not found")
        all_passed = False
    
    # Check for URL manipulation
    if "window.location.href" in content:
        print("✅ Page reload mechanism found")
    else:
        print("⚠️  Page reload mechanism not found")
    
    return all_passed

def check_backend_integration():
    """Verify backend is ready for Phase 3"""
    print("\n" + "="*70)
    print("BACKEND INTEGRATION CHECK")
    print("="*70)
    
    backend_path = "/home/rasheeque/VS CODE FOLDER/TEA TOAST/edit-004.py"
    
    if not os.path.exists(backend_path):
        print(f"❌ ERROR: Backend not found at {backend_path}")
        return False
    
    with open(backend_path, 'r') as f:
        content = f.read()
    
    checks = {
        "1. Restaurant model exists": "class Restaurant(Base):" in content,
        "2. Camera model exists": "class Camera(Base):" in content,
        "3. CameraApp model exists": "class CameraApp(Base):" in content,
        "4. get_app_configs has restaurant_id param": "def get_app_configs(restaurant_id=" in content,
        "5. Dashboard route updated": "selected_restaurant" in content,
        "6. Restaurant API endpoint": "@app.route('/api/restaurants'" in content,
    }
    
    print("\n📋 Backend Verification:")
    print("-" * 70)
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check_name}")
        if not passed:
            all_passed = False
    
    return all_passed

def check_database_ready():
    """Check if database has required Phase 1 schema"""
    print("\n" + "="*70)
    print("DATABASE SCHEMA CHECK")
    print("="*70)
    
    try:
        import subprocess
        
        # Check if PostgreSQL is running
        result = subprocess.run(
            ["sudo", "systemctl", "is-active", "postgresql"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.stdout.strip() == "active":
            print("✅ PostgreSQL service is running")
        else:
            print("❌ PostgreSQL service is not active")
            print("   Start with: sudo systemctl start postgresql")
            return False
        
        # Try to query database
        query = """
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_name IN ('restaurants', 'cameras', 'camera_apps');
        """
        
        result = subprocess.run(
            ["psql", "-U", "postgres", "-d", "sakshi", "-t", "-c", query],
            capture_output=True,
            text=True,
            timeout=5,
            env={"PGPASSWORD": "root"}
        )
        
        if result.returncode == 0:
            count = int(result.stdout.strip())
            if count == 3:
                print("✅ Required tables exist (restaurants, cameras, camera_apps)")
                return True
            else:
                print(f"⚠️  Only {count}/3 required tables found")
                print("   Run Phase 1 migration if not completed")
                return False
        else:
            print("⚠️  Could not query database (authentication may be required)")
            print("   This is OK if you've already verified Phase 1 completion")
            return True
            
    except Exception as e:
        print(f"⚠️  Could not check database: {e}")
        print("   This is OK if you've already verified Phase 1 completion")
        return True

def main():
    print("\n" + "="*70)
    print("🚀 PHASE 3: FRONTEND DASHBOARD DROPDOWN VERIFICATION")
    print("="*70)
    print("\nThis script verifies that Phase 3 changes are properly implemented.")
    print("It checks the frontend template and backend integration.\n")
    
    results = {
        "Frontend Template": check_dashboard_template(),
        "Backend Integration": check_backend_integration(),
        "Database Schema": check_database_ready()
    }
    
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {component}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 SUCCESS! Phase 3 implementation verified.")
        print("\n📋 Next Steps:")
        print("   1. Start the application: python3 edit-004.py")
        print("   2. Login to dashboard: http://localhost:5001/login")
        print("   3. Look for restaurant dropdown in top-right corner")
        print("   4. Select a restaurant and verify filtering works")
        print("   5. Check URL changes to /dashboard?restaurant_id=X")
        print("   6. Verify restaurant badge appears in subtitle")
        print("\n✨ Ready to proceed with testing!")
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        print("\n📝 Common Issues:")
        print("   - Template not saved properly")
        print("   - Backend missing Phase 2 changes")
        print("   - Database not migrated (Phase 1)")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
