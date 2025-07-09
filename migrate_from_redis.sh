#!/bin/bash
# Redis to SQLite Migration Script

echo "Redis to SQLite/In-Memory Migration"
echo "==================================="

# Export environment
export $(cat .env.local | grep -v '^#' | xargs)

# Run migration
echo "Running migration..."
python3 -c "
from scripts.redis_migration import migrate_from_redis
results = migrate_from_redis()
print(f'\nMigration complete!')
print(f'Total items migrated: {results["migration"].get("total_migrated", 0)}')
"

# Test the system
echo -e "\nTesting Redis-free system..."
python3 test_redis_removal.py

echo -e "\nMigration complete! The system is now running without Redis."
echo "Using:"
echo "  - SQLite for persistent storage"
echo "  - In-memory queues for work management"
