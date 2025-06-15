#!/usr/bin/env python3
"""Test script to verify Redis async/await fixes."""

import asyncio
import sys
sys.path.append('/workspaces/cwmai')

from scripts.redis_integration.redis_client import get_redis_client
from scripts.redis_integration.redis_streams_manager import RedisStreamsManager


async def test_redis_operations():
    """Test various Redis operations to ensure async/await is working correctly."""
    print("Testing Redis async/await fixes...")
    
    # Test 1: Redis Client Stream Operations
    print("\n1. Testing Redis client stream operations...")
    try:
        redis_client = await get_redis_client()
        
        # Test xadd
        stream_id = await redis_client.xadd("test:stream", {"data": "test"})
        print(f"  ✓ xadd successful: {stream_id}")
        
        # Test xgroup_create 
        await redis_client.xgroup_create("test:stream", "test-group", id='0')
        print("  ✓ xgroup_create successful")
        
        # Test xreadgroup
        messages = await redis_client.xreadgroup("test-group", "test-consumer", {"test:stream": ">"})
        print(f"  ✓ xreadgroup successful: {len(messages) if messages else 0} messages")
        
        # Test xclaim
        if messages:
            # Get message ID from first message
            stream_name, stream_messages = messages[0]
            if stream_messages:
                msg_id, _ = stream_messages[0]
                # Try to claim the message
                claimed = await redis_client.xclaim("test:stream", "test-group", "test-consumer2", 0, [msg_id])
                print(f"  ✓ xclaim successful: {len(claimed)} messages claimed")
        
        # Test eval
        result = await redis_client.eval("return 'Hello from Lua'", 0)
        print(f"  ✓ eval successful: {result}")
        
        # Test scan
        cursor, keys = await redis_client.scan(0, match="test:*", count=10)
        print(f"  ✓ scan successful: cursor={cursor}, keys={len(keys)}")
        
        # Clean up
        await redis_client.delete("test:stream")
        
    except Exception as e:
        print(f"  ✗ Redis client test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Redis Streams Manager
    print("\n2. Testing Redis Streams Manager...")
    try:
        streams_manager = RedisStreamsManager(redis_client)
        await streams_manager.start()
        
        # Test produce
        msg_id = await streams_manager.produce("test-events", {"event": "test"})
        print(f"  ✓ Stream produce successful: {msg_id}")
        
        # Test read
        messages = await streams_manager.read_messages("test-events", "test-readers", count=1)
        print(f"  ✓ Stream read successful: {len(messages)} messages")
        
        await streams_manager.stop()
        
    except Exception as e:
        print(f"  ✗ Streams manager test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Close Redis connection
    await redis_client.close()
    print("\n✓ All tests completed!")


if __name__ == "__main__":
    asyncio.run(test_redis_operations())