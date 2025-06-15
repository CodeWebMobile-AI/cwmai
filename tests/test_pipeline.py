#!/usr/bin/env python3
"""Test Redis pipeline functionality."""

import asyncio
import sys
sys.path.insert(0, '/workspaces/cwmai/scripts')

from redis_integration.redis_client import RedisClient

async def test_pipeline():
    """Test pipeline operations."""
    print("Testing Redis pipeline...")
    
    client = RedisClient()
    await client.connect()
    
    # Test 1: Simple pipeline
    print("\nTest 1: Simple pipeline with xadd")
    async with client.pipeline() as pipe:
        await pipe.xadd("test_stream", {"field1": "value1"})
        await pipe.xadd("test_stream", {"field2": "value2"})
        results = await pipe.execute()
        print(f"Pipeline results: {results}")
    
    # Test 2: Check stream contents
    print("\nTest 2: Reading stream contents")
    info = await client.xinfo_stream("test_stream")
    print(f"Stream info - Length: {info.get('length', 0)}")
    
    # Test 3: Read entries
    entries = await client.xrevrange("test_stream", count=10)
    print(f"Stream entries: {len(entries)}")
    for entry_id, data in entries:
        print(f"  {entry_id}: {data}")
    
    # Clean up
    await client.delete("test_stream")
    await client.disconnect()
    print("\nPipeline test completed!")

if __name__ == "__main__":
    asyncio.run(test_pipeline())