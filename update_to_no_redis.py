#!/usr/bin/env python3
"""
Update system to remove Redis dependency

This script updates the codebase to use SQLite/In-memory instead of Redis.
"""

import os
import shutil
from datetime import datetime


def create_backup(file_path):
    """Create backup of a file before modifying."""
    if os.path.exists(file_path):
        backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(file_path, backup_path)
        print(f"Created backup: {backup_path}")
        return backup_path
    return None


def update_hybrid_orchestrator():
    """Update hybrid orchestrator to use Redis-free components."""
    file_path = "scripts/hybrid_continuous_orchestrator.py"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Create backup
    create_backup(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace imports
    replacements = [
        # Replace continuous_orchestrator import
        ('from scripts.continuous_orchestrator import ContinuousOrchestrator',
         'from scripts.continuous_orchestrator_no_redis import ContinuousOrchestratorNoRedis as ContinuousOrchestrator'),
        
        # Add note about Redis removal
        ('"""', '"""\nNOTE: This version uses SQLite/In-memory instead of Redis.\n'),
    ]
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"Updated import: {old[:50]}...")
    
    # Write updated content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Updated {file_path}")


def update_env_file():
    """Update .env.local to reflect Redis removal."""
    env_file = ".env.local"
    
    if not os.path.exists(env_file):
        print(f"Environment file not found: {env_file}")
        return
    
    create_backup(env_file)
    
    with open(env_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if line.startswith('USE_MCP_REDIS='):
            new_lines.append('USE_MCP_REDIS=false  # Redis removed - using SQLite/In-memory\n')
        elif 'REDIS_' in line and not line.startswith('#'):
            new_lines.append(f'# {line}')  # Comment out Redis settings
        else:
            new_lines.append(line)
    
    with open(env_file, 'w') as f:
        f.writelines(new_lines)
    
    print(f"✓ Updated {env_file} - Redis disabled")


def create_migration_script():
    """Create a migration script for easy execution."""
    script_content = '''#!/bin/bash
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
print(f'\\nMigration complete!')
print(f'Total items migrated: {results[\"migration\"].get(\"total_migrated\", 0)}')
"

# Test the system
echo -e "\\nTesting Redis-free system..."
python3 test_redis_removal.py

echo -e "\\nMigration complete! The system is now running without Redis."
echo "Using:"
echo "  - SQLite for persistent storage"
echo "  - In-memory queues for work management"
'''
    
    with open('migrate_from_redis.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('migrate_from_redis.sh', 0o755)
    print("✓ Created migrate_from_redis.sh")


def update_requirements():
    """Update requirements.txt to remove Redis dependency."""
    req_file = "requirements.txt"
    
    if not os.path.exists(req_file):
        print(f"Requirements file not found: {req_file}")
        return
    
    create_backup(req_file)
    
    with open(req_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    redis_removed = False
    
    for line in lines:
        if line.strip().startswith('redis') or 'redis-py' in line:
            new_lines.append(f'# {line}')  # Comment out Redis
            redis_removed = True
        else:
            new_lines.append(line)
    
    with open(req_file, 'w') as f:
        f.writelines(new_lines)
    
    if redis_removed:
        print(f"✓ Updated {req_file} - Redis dependency commented out")
    else:
        print(f"No Redis dependency found in {req_file}")


def main():
    """Run all updates."""
    print("Updating system to remove Redis dependency")
    print("==========================================\n")
    
    # Update files
    update_hybrid_orchestrator()
    update_env_file()
    update_requirements()
    create_migration_script()
    
    print("\n✅ System updated!")
    print("\nNext steps:")
    print("1. Run ./migrate_from_redis.sh to migrate any existing Redis data")
    print("2. Test with: python3 test_redis_removal.py")
    print("3. Run the system normally - it will use SQLite/In-memory automatically")
    print("\nThe system is now Redis-free and uses:")
    print("- SQLite for persistent state storage (no connection limits)")
    print("- In-memory queues for high-performance work management")
    print("- Full Redis API compatibility for easy migration")


if __name__ == "__main__":
    main()