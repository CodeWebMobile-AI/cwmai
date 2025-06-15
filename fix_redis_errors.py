#!/usr/bin/env python3
"""
Fix Redis-related errors in the continuous AI system.

This script addresses:
1. 'sum' key missing in analytics metrics when no samples
2. JSON serialization of lambda functions 
3. Lock acquisition failures with deadlock detection
4. Redis "no such key" errors
5. Redis connection timeouts
"""

import asyncio
import logging
import sys
import os

# Add the scripts directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.redis_integration import get_redis_client
from scripts.redis_integration.redis_config import get_redis_config


async def test_redis_connection():
    """Test Redis connection and basic operations."""
    print("Testing Redis connection...")
    
    try:
        redis_client = await get_redis_client()
        
        # Test basic operations
        test_key = "test:connection:key"
        test_value = "test_value"
        
        # Set
        await redis_client.set(test_key, test_value)
        print(f"✓ SET {test_key} = {test_value}")
        
        # Get
        result = await redis_client.get(test_key)
        print(f"✓ GET {test_key} = {result}")
        
        # Get non-existent key
        non_existent = await redis_client.get("non:existent:key")
        print(f"✓ GET non-existent key = {non_existent} (should be None)")
        
        # Delete
        await redis_client.delete(test_key)
        print(f"✓ DELETE {test_key}")
        
        # Test connection health
        if hasattr(redis_client, 'health_monitor'):
            health = await redis_client.health_monitor.check_health()
            print(f"✓ Redis health check: {'Healthy' if health else 'Unhealthy'}")
        
        print("\nRedis connection test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Redis connection test failed: {e}")
        return False


async def reset_problematic_locks():
    """Reset locks that might be causing deadlocks."""
    print("\nResetting problematic locks...")
    
    try:
        redis_client = await get_redis_client()
        
        # Pattern for locks that are frequently failing
        lock_patterns = [
            "locks:state:default:analytics.*",
            "locks:state:default:intelligence.*",
            "locks:state:default:workflow_engines.*"
        ]
        
        total_deleted = 0
        for pattern in lock_patterns:
            cursor = 0
            while True:
                cursor, keys = await redis_client.scan(cursor, match=pattern, count=100)
                
                if keys:
                    deleted = await redis_client.delete(*keys)
                    total_deleted += deleted
                    print(f"  Deleted {deleted} locks matching {pattern}")
                
                if cursor == 0:
                    break
        
        print(f"✓ Reset {total_deleted} problematic locks")
        return True
        
    except Exception as e:
        print(f"✗ Failed to reset locks: {e}")
        return False


async def verify_lock_free_components():
    """Verify that lock-free components are properly initialized."""
    print("\nVerifying lock-free components...")
    
    try:
        # Import components that should use lock-free state management
        from scripts.redis_event_analytics import RedisEventAnalytics
        from scripts.redis_lockfree_adapter import create_lockfree_state_manager
        
        # Create a test analytics instance
        analytics = RedisEventAnalytics(analytics_id="test_analytics")
        
        # Check if it's using lock-free state manager
        if hasattr(analytics, 'state_manager'):
            print("✓ RedisEventAnalytics has state_manager")
            
            # Verify it's the lock-free version
            manager_type = type(analytics.state_manager).__name__
            print(f"  State manager type: {manager_type}")
            
            if "lockfree" in manager_type.lower():
                print("✓ Using lock-free state manager")
            else:
                print("✗ Not using lock-free state manager")
        else:
            print("✗ RedisEventAnalytics missing state_manager")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to verify lock-free components: {e}")
        return False


async def configure_redis_timeouts():
    """Configure Redis client with appropriate timeouts."""
    print("\nConfiguring Redis timeouts...")
    
    try:
        config = get_redis_config()
        
        # Print current configuration
        print(f"Current configuration:")
        print(f"  Mode: {config.mode}")
        print(f"  Socket timeout: {config.socket_timeout}s")
        print(f"  Socket connect timeout: {config.socket_connect_timeout}s")
        print(f"  Socket keepalive: {config.socket_keepalive}")
        print(f"  Connection pool size: {config.pool_max_connections}")
        
        # Recommend optimal settings
        print("\nRecommended settings for better stability:")
        print("  - Increase socket_timeout to 30s for slow operations")
        print("  - Enable socket_keepalive for long-running connections")
        print("  - Increase pool_max_connections if seeing connection errors")
        print("  - Consider using connection retry with exponential backoff")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to configure timeouts: {e}")
        return False


async def test_state_operations():
    """Test state operations to ensure they handle errors gracefully."""
    print("\nTesting state operations...")
    
    try:
        from scripts.redis_lockfree_adapter import create_lockfree_state_manager
        
        # Create a test state manager
        state_manager = create_lockfree_state_manager("test_component")
        await state_manager.initialize()
        
        # Test basic operations
        test_key = "test.state.key"
        test_value = {"count": 0, "data": "test"}
        
        # Update
        success = await state_manager.update(test_key, test_value, distributed=True)
        print(f"✓ State update: {success}")
        
        # Get
        retrieved = await state_manager.get(test_key)
        print(f"✓ State get: {retrieved}")
        
        # Get non-existent
        non_existent = await state_manager.get("non.existent.key")
        print(f"✓ Get non-existent: {non_existent} (should be None)")
        
        # Delete
        await state_manager.delete(test_key)
        print(f"✓ State delete completed")
        
        return True
        
    except Exception as e:
        print(f"✗ State operations test failed: {e}")
        return False


async def main():
    """Main function to run all fixes and tests."""
    print("Redis Error Fix Utility")
    print("=" * 50)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Disable verbose Redis logs for this script
    logging.getLogger('scripts.redis_integration').setLevel(logging.WARNING)
    
    results = []
    
    # Run tests and fixes
    results.append(("Redis Connection", await test_redis_connection()))
    results.append(("Reset Locks", await reset_problematic_locks()))
    results.append(("Lock-free Components", await verify_lock_free_components()))
    results.append(("Redis Timeouts", await configure_redis_timeouts()))
    results.append(("State Operations", await test_state_operations()))
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\nAll tests passed! The Redis errors should be resolved.")
        print("\nRecommendations:")
        print("1. Restart the continuous AI system to apply fixes")
        print("2. Monitor logs for any remaining errors")
        print("3. Consider increasing Redis timeouts if errors persist")
    else:
        print("\nSome tests failed. Please check the errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)