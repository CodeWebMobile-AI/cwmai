#!/usr/bin/env python3
"""Simple Redis connection test."""

import asyncio
import sys
sys.path.insert(0, '/workspaces/cwmai/scripts')

from redis_integration.redis_client import RedisClient

async def test_redis():
    """Test basic Redis connection."""
    print("Testing Redis connection...")
    
    redis_client = RedisClient()
    await redis_client.connect()
    print("✅ Redis connected successfully!")
    
    # Test basic operations
    await redis_client.set("test_key", "test_value")
    value = await redis_client.get("test_key")
    print(f"✅ Basic set/get works: {value}")
    
    # Test streams
    stream_name = "cwmai:work_queue:critical"
    try:
        info = await redis_client.xinfo_stream(stream_name)
        print(f"✅ Stream info retrieved: {info.get('length', 0)} items in {stream_name}")
    except Exception as e:
        print(f"❌ Stream info error: {e}")
    
    # Get connection stats
    stats = redis_client.get_connection_stats()
    print(f"\nConnection Stats:")
    print(f"  Connected: {stats['connected']}")
    print(f"  Connection ID: {stats['connection_id']}")
    print(f"  Redis Mode: {stats['config']['mode']}")
    
    await redis_client.disconnect()
    print("\n✅ Redis test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_redis())