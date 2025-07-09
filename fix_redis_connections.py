#!/usr/bin/env python3
"""
Fix Redis connection leaks by implementing proper cleanup
"""

import os
import sys
from pathlib import Path

def fix_redis_integration():
    """Add proper connection cleanup to redis_integration.py"""
    
    redis_integration_path = Path("scripts/redis_integration/__init__.py")
    
    if not redis_integration_path.exists():
        print("❌ Could not find scripts/redis_integration/__init__.py")
        return False
    
    # Read the current file
    content = redis_integration_path.read_text()
    
    # Check if cleanup is already implemented
    if "cleanup_connections" in content:
        print("✓ Connection cleanup already implemented")
        return True
    
    # Add cleanup function
    cleanup_code = '''

# Connection cleanup functionality
async def cleanup_idle_connections():
    """Clean up idle Redis connections"""
    try:
        client = await get_redis_client()
        
        # Get all client connections
        client_list = await client.client_list()
        
        # Find and kill idle connections (>300 seconds)
        killed = 0
        for client_info in client_list:
            idle_time = int(client_info.get('idle', 0))
            if idle_time > 300:  # 5 minutes
                client_id = client_info.get('id')
                if client_id:
                    try:
                        await client.client_kill(client_id)
                        killed += 1
                    except Exception:
                        pass
        
        if killed > 0:
            logger.info(f"Cleaned up {killed} idle Redis connections")
            
        return killed
    except Exception as e:
        logger.error(f"Error cleaning up connections: {e}")
        return 0

# Auto-cleanup task
_cleanup_task = None

async def start_connection_cleanup():
    """Start automatic connection cleanup task"""
    global _cleanup_task
    
    async def cleanup_loop():
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await cleanup_idle_connections()
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    if _cleanup_task is None:
        _cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("Started Redis connection cleanup task")

# Proper connection context manager
class RedisConnection:
    """Context manager for Redis connections with guaranteed cleanup"""
    
    def __init__(self):
        self.client = None
        
    async def __aenter__(self):
        self.client = await get_redis_client()
        return self.client
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            try:
                # Ensure connection is returned to pool
                await self.client.connection_pool.disconnect()
            except Exception:
                pass

# Export cleanup functions
__all__ = [
    'get_redis_client',
    'get_connection_pool', 
    'RedisError',
    'cleanup_idle_connections',
    'start_connection_cleanup',
    'RedisConnection'
]
'''
    
    # Insert before the final __all__ export if it exists
    if "__all__" in content:
        # Replace the existing __all__ with our extended version
        import re
        content = re.sub(r'__all__\s*=\s*\[.*?\]', '', content, flags=re.DOTALL)
    
    # Add cleanup code at the end
    content += cleanup_code
    
    # Write back
    redis_integration_path.write_text(content)
    print("✓ Added connection cleanup to redis_integration/__init__.py")
    
    return True

def fix_redis_client():
    """Fix redis_client.py to properly close connections"""
    
    redis_client_path = Path("scripts/redis_integration/redis_client.py")
    
    if not redis_client_path.exists():
        print("❌ Could not find scripts/redis_integration/redis_client.py")
        return False
    
    content = redis_client_path.read_text()
    
    # Look for the subscribe method and ensure proper cleanup
    if "async def subscribe" in content and "finally:" not in content:
        # Need to add proper cleanup
        print("⚠️  PubSub cleanup needed in redis_client.py")
        
        # Add this after imports
        import_addition = '''
import weakref
import atexit

# Track all pubsub connections for cleanup
_active_pubsubs = weakref.WeakSet()

def _cleanup_pubsubs():
    """Cleanup function called on exit"""
    for pubsub in list(_active_pubsubs):
        try:
            pubsub.close()
        except:
            pass

atexit.register(_cleanup_pubsubs)
'''
        
        # Find where to insert (after imports)
        import_end = content.rfind("import") 
        if import_end != -1:
            # Find the end of the import line
            newline_pos = content.find("\n", import_end)
            if newline_pos != -1:
                content = content[:newline_pos+1] + import_addition + content[newline_pos+1:]
        
        # Update subscribe method to track pubsub
        content = content.replace(
            "pubsub = self.client.pubsub()",
            "pubsub = self.client.pubsub()\n        _active_pubsubs.add(pubsub)"
        )
    
    redis_client_path.write_text(content)
    print("✓ Updated redis_client.py with cleanup tracking")
    
    return True

def create_cleanup_script():
    """Create a script to clean up existing idle connections"""
    
    cleanup_script = '''#!/usr/bin/env python3
"""
Clean up idle Redis connections immediately
"""

import asyncio
import redis.asyncio as redis

async def cleanup_now():
    """Clean up all idle connections right now"""
    print("Cleaning up idle Redis connections...")
    
    client = redis.Redis(
        host='localhost',
        port=6379,
        decode_responses=True
    )
    
    try:
        # Get all clients
        client_list = await client.client_list()
        print(f"Found {len(client_list)} total connections")
        
        # Kill idle connections
        killed = 0
        for client_info in client_list:
            idle_time = int(client_info.get('idle', 0))
            if idle_time > 60:  # Kill anything idle >1 minute
                client_id = client_info.get('id')
                client_name = client_info.get('name', 'unnamed')
                if client_id:
                    try:
                        await client.client_kill_filter(ID=client_id)
                        killed += 1
                        print(f"  Killed connection {client_id} (idle: {idle_time}s, name: {client_name})")
                    except Exception as e:
                        print(f"  Failed to kill {client_id}: {e}")
        
        print(f"\\n✓ Cleaned up {killed} idle connections")
        
        # Show remaining
        remaining = await client.client_list()
        print(f"Remaining connections: {len(remaining)}")
        
    finally:
        await client.aclose()

if __name__ == "__main__":
    asyncio.run(cleanup_now())
'''
    
    with open("cleanup_redis_now.py", "w") as f:
        f.write(cleanup_script)
    
    print("✓ Created cleanup_redis_now.py")
    
    # Make it executable
    os.chmod("cleanup_redis_now.py", 0o755)
    
    return True

def main():
    """Apply all Redis connection fixes"""
    print("Applying Redis connection fixes...")
    print("="*60)
    
    # Fix redis integration
    if not fix_redis_integration():
        print("Failed to fix redis_integration")
        return False
    
    # Fix redis client
    if not fix_redis_client():
        print("Failed to fix redis_client")
        return False
    
    # Create cleanup script
    if not create_cleanup_script():
        print("Failed to create cleanup script")
        return False
    
    print("\n" + "="*60)
    print("✓ All fixes applied!")
    print("\nNext steps:")
    print("1. Run the cleanup script: python3 cleanup_redis_now.py")
    print("2. Restart your CWMAI system to use the new connection handling")
    print("3. Monitor with: python3 monitor_redis_health.py")
    
    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1)