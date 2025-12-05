#!/bin/bash
# Quick command reference for Phase 1

echo "🚀 PHASE 1 - MULTI-RESTAURANT MIGRATION"
echo "========================================"
echo ""
echo "Choose an option:"
echo ""
echo "1. 🎯 Run Full Migration (Automated - RECOMMENDED)"
echo "   ./run_phase1_migration.sh"
echo ""
echo "2. 📊 View Current Database"
echo "   psql -U postgres -d sakshi -c \"SELECT * FROM restaurants;\""
echo ""
echo "3. 🔍 Verify Migration"
echo "   python3 verify_migration.py"
echo ""
echo "4. 🧪 Quick Test"
echo "   python3 test_phase1.py"
echo ""
echo "5. 💾 Manual Backup"
echo "   pg_dump -U postgres sakshi > backups/manual_backup_\$(date +%Y%m%d_%H%M%S).sql"
echo ""
echo "6. ⏪ Rollback (if needed)"
echo "   psql -U postgres -d sakshi < backups/sakshi_backup_YYYYMMDD.sql"
echo ""
echo "7. 📖 Read Documentation"
echo "   cat PHASE1_QUICKSTART.md"
echo ""
echo "========================================"
echo ""
read -p "Enter option number (or press Enter to exit): " choice
echo ""

case $choice in
    1)
        echo "🚀 Running full migration..."
        ./run_phase1_migration.sh
        ;;
    2)
        echo "📊 Querying database..."
        psql -U postgres -d sakshi -c "SELECT * FROM restaurants;"
        ;;
    3)
        echo "🔍 Running verification..."
        python3 verify_migration.py
        ;;
    4)
        echo "🧪 Running tests..."
        python3 test_phase1.py
        ;;
    5)
        echo "💾 Creating backup..."
        BACKUP_FILE="backups/manual_backup_$(date +%Y%m%d_%H%M%S).sql"
        pg_dump -U postgres sakshi > "$BACKUP_FILE"
        echo "✅ Backup saved to: $BACKUP_FILE"
        ;;
    6)
        echo "⚠️  Rollback requires backup file path"
        ls -lht backups/*.sql | head -5
        echo ""
        read -p "Enter backup file path: " backup_path
        if [ -f "$backup_path" ]; then
            read -p "⚠️  This will restore the database. Continue? (yes/no): " confirm
            if [ "$confirm" = "yes" ]; then
                psql -U postgres -d sakshi < "$backup_path"
                echo "✅ Database restored"
            else
                echo "❌ Rollback cancelled"
            fi
        else
            echo "❌ File not found: $backup_path"
        fi
        ;;
    7)
        echo "📖 Opening documentation..."
        cat PHASE1_QUICKSTART.md | less
        ;;
    *)
        echo "👋 Goodbye!"
        ;;
esac
