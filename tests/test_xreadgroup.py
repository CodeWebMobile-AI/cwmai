#!/usr/bin/env python3
"""Test xreadgroup directly."""

import asyncio
import sys
sys.path.insert(0, '/workspaces/cwmai/scripts')

from redis_integration.redis_client import RedisClient

async def test_xreadgroup():
    """Test xreadgroup directly."""
    redis_client = RedisClient()
    await redis_client.connect()
    
    print("=== TESTING XREADGROUP DIRECTLY ===\n")
    
    # Add a test message
    stream = "test:stream"
    await redis_client.xadd(stream, {"test": "message"})
    
    # Create consumer group
    try:
        await redis_client.xgroup_create(stream, "test_group", id='0')
        print("Created consumer group")
    except:
        print("Consumer group already exists")
    
    # Try to read
    print("\nTrying xreadgroup...")
    messages = await redis_client.xreadgroup(
        "test_group",
        "test_consumer", 
        {stream: '>'},
        count=1,
        block=None
    )
    
    print(f"Result: {messages}")
    
    # Try with block=1000 (1 second)
    print("\nTrying xreadgroup with block=1000...")
    messages2 = await redis_client.xreadgroup(
        "test_group",
        "test_consumer", 
        {stream: '>'},
        count=1,
        block=1000
    )
    
    print(f"Result: {messages2}")
    
    # Clean up
    await redis_client.delete(stream)

if __name__ == "__main__":
    asyncio.run(test_xreadgroup())