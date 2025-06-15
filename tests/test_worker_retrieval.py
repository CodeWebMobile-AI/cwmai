#!/usr/bin/env python3
"""Test worker retrieval specifically."""

import asyncio
import sys
from datetime import datetime, timezone

sys.path.insert(0, '/workspaces/cwmai/scripts')

from work_item_types import WorkItem, TaskPriority
from redis_work_queue import RedisWorkQueue
from redis_integration.redis_client import RedisClient

async def test_worker_retrieval():
    """Test worker retrieval issue."""
    print("=== WORKER RETRIEVAL TEST ===\n")
    
    # Initialize
    redis_client = RedisClient()
    await redis_client.connect()
    
    work_queue = RedisWorkQueue(redis_client=redis_client)
    await work_queue.initialize()
    
    # Clear and recreate
    stream = "cwmai:work_queue:high"
    await redis_client.delete(stream)
    await redis_client.xgroup_create(stream, "cwmai_workers", mkstream=True)
    
    # Add one item directly
    print("1. Adding item directly to stream...")
    data = {
        'id': 'direct_test_1',
        'task_type': 'TEST',
        'title': 'Direct test item',
        'description': 'Testing direct add',
        'priority': 'HIGH',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'estimated_cycles': '1',
        'dependencies': '[]',
        'metadata': '{}'
    }
    
    msg_id = await redis_client.xadd(stream, data)
    print(f"  Added with ID: {msg_id}")
    
    # Check stream
    info = await redis_client.xinfo_stream(stream)
    print(f"\n2. Stream info:")
    print(f"  Length: {info.get('length', 0)}")
    print(f"  Groups: {info.get('groups', 0)}")
    
    # Try xreadgroup directly
    print(f"\n3. Testing xreadgroup directly...")
    messages = await redis_client.xreadgroup(
        "cwmai_workers",
        "test_consumer",
        {stream: '>'},
        count=10
    )
    print(f"  Read {len(messages)} message groups")
    for stream_name, stream_messages in messages:
        print(f"  Stream {stream_name}: {len(stream_messages)} messages")
        for msg_id, data in stream_messages[:1]:  # Show first message
            print(f"    Message {msg_id}: {list(data.keys())}")
    
    # Now test through work queue
    print(f"\n4. Testing through work queue...")
    retrieved = await work_queue.get_work_for_worker("test_worker", "general", count=5)
    print(f"  Retrieved {len(retrieved)} items through work queue")
    
    # Check pending
    print(f"\n5. Checking pending messages...")
    pending = await redis_client.xpending(stream, "cwmai_workers")
    print(f"  Pending info: {pending}")
    
    await redis_client.disconnect()
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_worker_retrieval())