# 🚀 Phase 1 Implementation - Quick Start Guide

## ✅ What's Been Created

I've created all the necessary files for Phase 1 (Database Schema & Migration):

### **📄 Files Created:**

1. **`phase1_create_schema.sql`** - SQL script to create new tables
2. **`phase1_migrate_data.py`** - Python script to migrate rtsp_links.txt data
3. **`verify_migration.py`** - Verification script to check migration success
4. **`test_phase1.py`** - Quick tests for Phase 1
5. **`run_phase1_migration.sh`** - Automated execution script (with backup)
6. **`MULTI_RESTAURANT_ROADMAP.md`** - Complete roadmap document

---

## 🎯 Quick Execution (3 Easy Steps)

### **Option 1: Automated (RECOMMENDED)**

```bash
# Run the all-in-one script (includes backup)
./run_phase1_migration.sh
```

This will:
- ✅ Backup your current database automatically
- ✅ Create all new tables (restaurants, cameras, camera_apps)
- ✅ Migrate data from rtsp_links.txt
- ✅ Run verification checks
- ✅ Show you a complete summary

---

### **Option 2: Manual Step-by-Step**

If you prefer to run each step manually:

```bash
# Step 1: Backup database (IMPORTANT!)
pg_dump -U postgres sakshi > backups/sakshi_backup_$(date +%Y%m%d).sql

# Step 2: Create schema
psql -U postgres -d sakshi -f phase1_create_schema.sql

# Step 3: Migrate data
python3 phase1_migrate_data.py

# Step 4: Verify migration
python3 verify_migration.py

# Step 5: Quick test
python3 test_phase1.py
```

---

## 📊 What Gets Created

### **New Database Tables:**

1. **`restaurants`** - Stores restaurant info (Tea Toast, future locations)
   ```sql
   id | restaurant_code | restaurant_name | location | dvr_ip | ...
   ```

2. **`cameras`** - Replaces rtsp_links.txt
   ```sql
   id | restaurant_id | channel_number | channel_name | rtsp_url | ...
   ```

3. **`camera_apps`** - Links cameras to AI apps (many-to-many)
   ```sql
   id | camera_id | app_name | config | ...
   ```

### **Updated Tables:**
- All existing tables get `restaurant_id` column:
  - `roi_configs`
  - `detections`
  - `daily_footfall`
  - `hourly_footfall`
  - `queue_logs`
  - `kitchen_violations`
  - `occupancy_logs`
  - `occupancy_schedules`

---

## 🔍 Verification Queries

After migration, you can check the data:

```sql
-- See all restaurants
SELECT * FROM restaurants;

-- See all cameras
SELECT c.*, r.restaurant_name 
FROM cameras c 
JOIN restaurants r ON c.restaurant_id = r.id;

-- See camera-app linkages
SELECT c.channel_name, ca.app_name, ca.config
FROM cameras c
JOIN camera_apps ca ON c.id = ca.camera_id
ORDER BY c.channel_name;

-- Check existing data is linked
SELECT COUNT(*), 
       COUNT(restaurant_id) as linked 
FROM roi_configs;
```

---

## ⚠️ Important Notes

### **Safety:**
- ✅ **Automatic backup** created before migration
- ✅ **rtsp_links.txt preserved** - don't delete it yet!
- ✅ **Rollback available** if needed

### **Rollback (if needed):**
```bash
# Restore from backup
psql -U postgres -d sakshi < backups/sakshi_backup_YYYYMMDD.sql
```

### **No Impact on Current System:**
- ✅ Your current `edit-004.py` will **continue to work**
- ✅ It still reads from `rtsp_links.txt` 
- ✅ Migration only **adds** tables, doesn't remove anything
- ✅ Phase 2 will update the code to use the database

---

## 🎯 Expected Output

After successful migration, you should see:

```
✅ PHASE 1 MIGRATION COMPLETED SUCCESSFULLY!

📋 What was done:
   1. ✅ Database backed up
   2. ✅ Schema created (restaurants, cameras, camera_apps tables)
   3. ✅ Existing tables updated with restaurant_id
   4. ✅ Data migrated from rtsp_links.txt
   5. ✅ Verification completed

📊 Summary:
   • Cameras added: 5
   • Apps linked: 5
   • Restaurant: Tea Toast - Brigade Road
```

---

## 🐛 Troubleshooting

### **Issue: "pg_dump: command not found"**
```bash
# Install PostgreSQL client tools
sudo apt-get install postgresql-client
```

### **Issue: "Permission denied"**
```bash
# Make script executable
chmod +x run_phase1_migration.sh
```

### **Issue: "Password authentication failed"**
```bash
# Update DATABASE_URL in the Python scripts if your password is different
DATABASE_URL = "postgresql://postgres:YOUR_PASSWORD@127.0.0.1:5432/sakshi"
```

### **Issue: "tabulate module not found"**
```bash
# Install required Python package
pip install tabulate
```

---

## 📈 Next Steps After Phase 1

Once Phase 1 is complete and verified:

1. ✅ **Review the verification output**
2. ✅ **Test queries on new tables**
3. ✅ **Keep backup safe**
4. 🚀 **Proceed to Phase 2** - Backend Integration
   - Update `edit-004.py` to read from database
   - Add restaurant dropdown to dashboard
   - Test with Tea Toast first
   - Add second restaurant

---

## 💡 Quick Test

To quickly verify everything works:

```bash
# This should show "Tea Toast - Brigade Road"
psql -U postgres -d sakshi -c "SELECT restaurant_name FROM restaurants;"

# This should show your 5 cameras
psql -U postgres -d sakshi -c "SELECT channel_name FROM cameras;"

# Run automated test
python3 test_phase1.py
```

---

## 🆘 Need Help?

If you encounter any issues:

1. **Check the logs** - Scripts print detailed error messages
2. **Run verification** - `python3 verify_migration.py`
3. **Check backup** - Ensure backup file exists in `backups/`
4. **Rollback if needed** - Use the backup to restore

---

## ✅ Success Checklist

Before moving to Phase 2, verify:

- [ ] All scripts executed without errors
- [ ] Verification shows correct counts
- [ ] `restaurants` table has Tea Toast entry
- [ ] `cameras` table has all 5 cameras
- [ ] `camera_apps` table has all app linkages
- [ ] Existing data has `restaurant_id` set
- [ ] Backup file exists and is not empty
- [ ] `test_phase1.py` passes all tests

---

**Ready to execute? Run: `./run_phase1_migration.sh`** 🚀
