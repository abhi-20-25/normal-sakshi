#!/bin/bash
# ============================================================================
# Phase 1 Execution Script - Multi-Restaurant Migration
# ============================================================================
# This script safely executes the Phase 1 migration with backups

set -e  # Exit on error

echo "============================================================================"
echo "🚀 PHASE 1: Multi-Restaurant Database Migration"
echo "============================================================================"
echo ""

# Configuration
DB_NAME="sakshi"
DB_USER="postgres"
BACKUP_DIR="backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ============================================================================
# Step 1: Create backup directory
# ============================================================================
echo -e "${YELLOW}📁 Creating backup directory...${NC}"
mkdir -p "$BACKUP_DIR"

# ============================================================================
# Step 2: Backup current database
# ============================================================================
echo -e "${YELLOW}💾 Backing up current database...${NC}"
BACKUP_FILE="$BACKUP_DIR/sakshi_pre_migration_$TIMESTAMP.sql"

pg_dump -U "$DB_USER" "$DB_NAME" > "$BACKUP_FILE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Database backed up to: $BACKUP_FILE${NC}"
    BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    echo -e "   Size: $BACKUP_SIZE"
else
    echo -e "${RED}❌ Backup failed! Aborting migration.${NC}"
    exit 1
fi

# ============================================================================
# Step 3: Backup rtsp_links.txt
# ============================================================================
echo ""
echo -e "${YELLOW}📄 Backing up rtsp_links.txt...${NC}"
if [ -f "rtsp_links.txt" ]; then
    cp rtsp_links.txt "$BACKUP_DIR/rtsp_links_$TIMESTAMP.txt"
    echo -e "${GREEN}✅ rtsp_links.txt backed up${NC}"
else
    echo -e "${YELLOW}⚠️  rtsp_links.txt not found (skipping)${NC}"
fi

# ============================================================================
# Step 4: Run SQL schema migration
# ============================================================================
echo ""
echo -e "${YELLOW}🔧 Creating new database schema...${NC}"
psql -U "$DB_USER" -d "$DB_NAME" -f phase1_create_schema.sql

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Schema created successfully${NC}"
else
    echo -e "${RED}❌ Schema creation failed!${NC}"
    echo -e "${YELLOW}ℹ️  To restore: psql -U $DB_USER -d $DB_NAME < $BACKUP_FILE${NC}"
    exit 1
fi

# ============================================================================
# Step 5: Run Python data migration
# ============================================================================
echo ""
echo -e "${YELLOW}📊 Migrating data from rtsp_links.txt...${NC}"
python3 phase1_migrate_data.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Data migration completed successfully${NC}"
else
    echo -e "${RED}❌ Data migration failed!${NC}"
    echo -e "${YELLOW}ℹ️  To restore: psql -U $DB_USER -d $DB_NAME < $BACKUP_FILE${NC}"
    exit 1
fi

# ============================================================================
# Step 6: Run verification
# ============================================================================
echo ""
echo -e "${YELLOW}🔍 Running verification checks...${NC}"
python3 verify_migration.py

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================================"
echo -e "${GREEN}✅ PHASE 1 MIGRATION COMPLETED SUCCESSFULLY!${NC}"
echo "============================================================================"
echo ""
echo "📋 What was done:"
echo "   1. ✅ Database backed up to: $BACKUP_FILE"
echo "   2. ✅ Schema created (restaurants, cameras, camera_apps tables)"
echo "   3. ✅ Existing tables updated with restaurant_id"
echo "   4. ✅ Data migrated from rtsp_links.txt"
echo "   5. ✅ Verification completed"
echo ""
echo "⚠️  IMPORTANT NOTES:"
echo "   • Keep rtsp_links.txt as backup (don't delete)"
echo "   • Test the system before proceeding to Phase 2"
echo "   • To rollback: psql -U $DB_USER -d $DB_NAME < $BACKUP_FILE"
echo ""
echo "🎯 Next Steps:"
echo "   1. Review verification output above"
echo "   2. Test queries: SELECT * FROM restaurants;"
echo "   3. Proceed to Phase 2 (Backend Integration)"
echo ""
echo "============================================================================"
