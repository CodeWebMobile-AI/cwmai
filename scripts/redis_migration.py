"""
Redis to SQLite/Hybrid Migration Utilities

Provides tools to migrate from Redis to the new hybrid state manager.
Includes data migration, connection replacement, and verification tools.
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timezone
import traceback

from scripts.hybrid_state_manager import HybridStateManager, create_redis_replacement


class RedisMigration:
    """Utilities for migrating from Redis to SQLite/Hybrid system."""
    
    def __init__(self):
        """Initialize migration utilities."""
        self.logger = logging.getLogger(__name__)
        self.redis_client = None
        self.hybrid_manager = None
        
    def check_redis_available(self) -> bool:
        """Check if Redis is available and has data.
        
        Returns:
            True if Redis is available
        """
        try:
            import redis
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            return True
        except Exception as e:
            self.logger.info(f"Redis not available: {e}")
            return False
    
    def migrate_data(self, target_db: str = "cwmai_state.db") -> Dict[str, Any]:
        """Migrate all data from Redis to SQLite.
        
        Args:
            target_db: Target SQLite database path
            
        Returns:
            Migration statistics
        """
        stats = {
            'started_at': datetime.now(timezone.utc).isoformat(),
            'keys_migrated': 0,
            'hashes_migrated': 0,
            'sets_migrated': 0,
            'lists_migrated': 0,
            'errors': [],
            'success': False
        }
        
        if not self.check_redis_available():
            stats['errors'].append("Redis not available")
            return stats
        
        try:
            # Initialize hybrid manager
            self.hybrid_manager = HybridStateManager(db_path=target_db)
            
            # Get all keys from Redis
            all_keys = self.redis_client.keys('*')
            self.logger.info(f"Found {len(all_keys)} keys in Redis")
            
            for key in all_keys:
                try:
                    # Determine key type
                    key_type = self.redis_client.type(key)
                    
                    if key_type == 'string':
                        # Migrate string value
                        value = self.redis_client.get(key)
                        ttl = self.redis_client.ttl(key)
                        
                        if ttl > 0:
                            self.hybrid_manager.set(key, value, ex=ttl)
                        else:
                            self.hybrid_manager.set(key, value)
                        
                        stats['keys_migrated'] += 1
                        
                    elif key_type == 'hash':
                        # Migrate hash
                        hash_data = self.redis_client.hgetall(key)
                        for field, value in hash_data.items():
                            self.hybrid_manager.hset(key, field, value)
                        stats['hashes_migrated'] += 1
                        
                    elif key_type == 'set':
                        # Migrate set
                        members = self.redis_client.smembers(key)
                        if members:
                            self.hybrid_manager.sadd(key, *members)
                        stats['sets_migrated'] += 1
                        
                    elif key_type == 'list':
                        # Migrate list
                        list_data = self.redis_client.lrange(key, 0, -1)
                        if list_data:
                            self.hybrid_manager.rpush(key, *list_data)
                        stats['lists_migrated'] += 1
                        
                    else:
                        self.logger.warning(f"Unknown key type {key_type} for key {key}")
                        
                except Exception as e:
                    error_msg = f"Error migrating key {key}: {e}"
                    self.logger.error(error_msg)
                    stats['errors'].append(error_msg)
            
            # Calculate total
            stats['total_migrated'] = (
                stats['keys_migrated'] + 
                stats['hashes_migrated'] + 
                stats['sets_migrated'] + 
                stats['lists_migrated']
            )
            
            stats['completed_at'] = datetime.now(timezone.utc).isoformat()
            stats['success'] = len(stats['errors']) == 0
            
            self.logger.info(f"Migration completed: {stats['total_migrated']} items migrated")
            
        except Exception as e:
            stats['errors'].append(f"Migration failed: {e}")
            self.logger.error(f"Migration failed: {e}")
            traceback.print_exc()
        
        return stats
    
    def verify_migration(self, sample_size: int = 100) -> Dict[str, Any]:
        """Verify migration by comparing Redis and SQLite data.
        
        Args:
            sample_size: Number of keys to sample for verification
            
        Returns:
            Verification results
        """
        results = {
            'verified': 0,
            'mismatches': [],
            'missing': [],
            'success': False
        }
        
        if not self.redis_client or not self.hybrid_manager:
            results['error'] = "Redis or hybrid manager not initialized"
            return results
        
        try:
            # Get sample of keys
            all_keys = self.redis_client.keys('*')[:sample_size]
            
            for key in all_keys:
                try:
                    key_type = self.redis_client.type(key)
                    
                    if key_type == 'string':
                        redis_val = self.redis_client.get(key)
                        hybrid_val = self.hybrid_manager.get(key)
                        
                        if redis_val != hybrid_val:
                            results['mismatches'].append({
                                'key': key,
                                'type': 'string',
                                'redis': redis_val,
                                'hybrid': hybrid_val
                            })
                        else:
                            results['verified'] += 1
                            
                    elif key_type == 'hash':
                        redis_hash = self.redis_client.hgetall(key)
                        hybrid_hash = self.hybrid_manager.hgetall(key)
                        
                        if redis_hash != hybrid_hash:
                            results['mismatches'].append({
                                'key': key,
                                'type': 'hash',
                                'redis_fields': len(redis_hash),
                                'hybrid_fields': len(hybrid_hash)
                            })
                        else:
                            results['verified'] += 1
                            
                    # Add more type checks as needed
                    
                except Exception as e:
                    results['missing'].append({
                        'key': key,
                        'error': str(e)
                    })
            
            results['success'] = len(results['mismatches']) == 0 and len(results['missing']) == 0
            results['total_checked'] = len(all_keys)
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def create_compatibility_wrapper(self) -> None:
        """Create Redis compatibility wrapper to replace imports.
        
        This creates a module that can be imported instead of redis.
        """
        wrapper_content = '''"""
Redis Compatibility Wrapper

Import this instead of redis to use the hybrid state manager.

Example:
    # Before:
    import redis
    client = redis.Redis()
    
    # After:
    import redis_compat as redis
    client = redis.Redis()
"""

from scripts.hybrid_state_manager import create_redis_replacement as Redis
from scripts.hybrid_state_manager import RedisReplacement as StrictRedis

# Common Redis exceptions (no-ops for compatibility)
class RedisError(Exception):
    pass

class ConnectionError(RedisError):
    pass

class TimeoutError(RedisError):
    pass

# Export common names
__all__ = ['Redis', 'StrictRedis', 'RedisError', 'ConnectionError', 'TimeoutError']
'''
        
        with open('redis_compat.py', 'w') as f:
            f.write(wrapper_content)
        
        self.logger.info("Created redis_compat.py wrapper")
    
    def update_imports(self, file_path: str, dry_run: bool = True) -> List[str]:
        """Update Redis imports in a Python file.
        
        Args:
            file_path: Path to Python file
            dry_run: If True, only show changes without modifying
            
        Returns:
            List of changes made
        """
        changes = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')
            
            new_lines = []
            
            for line in lines:
                if 'import redis' in line and 'redis_compat' not in line:
                    # Replace redis imports
                    if line.strip() == 'import redis':
                        new_line = 'import redis_compat as redis'
                        changes.append(f"Line: '{line}' -> '{new_line}'")
                        new_lines.append(new_line)
                    elif 'from redis import' in line:
                        new_line = line.replace('from redis import', 'from redis_compat import')
                        changes.append(f"Line: '{line}' -> '{new_line}'")
                        new_lines.append(new_line)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            
            if changes and not dry_run:
                with open(file_path, 'w') as f:
                    f.write('\n'.join(new_lines))
                self.logger.info(f"Updated {len(changes)} imports in {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error updating imports in {file_path}: {e}")
        
        return changes
    
    def find_redis_usage(self, directory: str = '.') -> Dict[str, List[str]]:
        """Find all files using Redis in a directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            Dict mapping file paths to Redis-related lines
        """
        redis_usage = {}
        
        for root, dirs, files in os.walk(directory):
            # Skip virtual environments and cache
            dirs[:] = [d for d in dirs if d not in {'venv', '__pycache__', '.git'}]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                        
                        redis_lines = []
                        for i, line in enumerate(lines):
                            if 'redis' in line.lower() and 'redis_compat' not in line:
                                redis_lines.append(f"Line {i+1}: {line.strip()}")
                        
                        if redis_lines:
                            redis_usage[file_path] = redis_lines
                            
                    except Exception as e:
                        self.logger.debug(f"Error reading {file_path}: {e}")
        
        return redis_usage


def migrate_from_redis(target_db: str = "cwmai_state.db", 
                      update_imports: bool = False) -> Dict[str, Any]:
    """
    Complete migration from Redis to SQLite/Hybrid system.
    
    Args:
        target_db: Target SQLite database
        update_imports: Whether to update Python imports
        
    Returns:
        Migration results
    """
    migration = RedisMigration()
    results = {
        'migration': {},
        'verification': {},
        'imports_updated': [],
        'redis_usage': {}
    }
    
    # Step 1: Migrate data
    print("Step 1: Migrating data from Redis...")
    results['migration'] = migration.migrate_data(target_db)
    
    if results['migration']['success']:
        # Step 2: Verify migration
        print("Step 2: Verifying migration...")
        results['verification'] = migration.verify_migration()
        
        # Step 3: Create compatibility wrapper
        print("Step 3: Creating compatibility wrapper...")
        migration.create_compatibility_wrapper()
        
        # Step 4: Find Redis usage
        print("Step 4: Finding Redis usage in codebase...")
        results['redis_usage'] = migration.find_redis_usage('scripts')
        
        # Step 5: Update imports if requested
        if update_imports:
            print("Step 5: Updating imports...")
            for file_path in results['redis_usage'].keys():
                changes = migration.update_imports(file_path, dry_run=False)
                if changes:
                    results['imports_updated'].append({
                        'file': file_path,
                        'changes': changes
                    })
    
    return results


if __name__ == "__main__":
    # Run migration
    print("Redis to SQLite/Hybrid Migration Tool")
    print("=====================================")
    
    results = migrate_from_redis(update_imports=False)
    
    print("\nMigration Results:")
    print(f"- Keys migrated: {results['migration'].get('keys_migrated', 0)}")
    print(f"- Hashes migrated: {results['migration'].get('hashes_migrated', 0)}")
    print(f"- Sets migrated: {results['migration'].get('sets_migrated', 0)}")
    print(f"- Lists migrated: {results['migration'].get('lists_migrated', 0)}")
    print(f"- Total migrated: {results['migration'].get('total_migrated', 0)}")
    
    if results['migration'].get('errors'):
        print(f"\nErrors: {len(results['migration']['errors'])}")
        for error in results['migration']['errors'][:5]:
            print(f"  - {error}")
    
    if results.get('verification'):
        print(f"\nVerification: {results['verification'].get('verified', 0)} items verified")
        if results['verification'].get('mismatches'):
            print(f"  - Mismatches: {len(results['verification']['mismatches'])}")
    
    print(f"\nFiles using Redis: {len(results.get('redis_usage', {}))}")
    
    print("\nNext steps:")
    print("1. Review the migration results")
    print("2. Test your application with redis_compat")
    print("3. Run with update_imports=True to update all imports")
    print("4. Remove Redis dependency from requirements.txt")