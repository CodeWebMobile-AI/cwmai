#!/usr/bin/env python3
"""Test complete work flow."""

import asyncio
import sys
from datetime import datetime, timezone

sys.path.insert(0, '/workspaces/cwmai/scripts')

from work_item_types import WorkItem, TaskPriority
from redis_work_queue import RedisWorkQueue
from redis_integration.redis_client import RedisClient

async def test_complete_flow():
    """Test complete work item flow."""
    print("=== Testing Complete Work Flow ===\n")
    
    # Initialize components
    redis_client = RedisClient()
    await redis_client.connect()
    
    work_queue = RedisWorkQueue(redis_client=redis_client)
    await work_queue.initialize()
    
    # Step 1: Add work items
    print("1. Adding work items...")
    test_item = WorkItem(
        id="test_flow_1",
        task_type="TEST",
        title="Test flow item",
        description="Testing complete flow",
        priority=TaskPriority.HIGH,
        repository=None,
        estimated_cycles=1,
        created_at=datetime.now(timezone.utc)
    )
    
    await work_queue.add_work(test_item)
    await work_queue._flush_buffer()  # Force flush
    print("✅ Work item added and flushed")
    
    # Step 2: Check stream directly
    print("\n2. Checking Redis stream directly...")
    stream_name = "cwmai:work_queue:high"
    entries = await redis_client.xrevrange(stream_name, count=5)
    print(f"Stream entries: {len(entries)}")
    for entry_id, data in entries[:2]:  # Show first 2
        print(f"  Entry {entry_id}: {list(data.keys())}")
    
    # Step 3: Get work for worker
    print("\n3. Retrieving work for worker...")
    retrieved = await work_queue.get_work_for_worker("test_worker", "general", count=1)
    print(f"Retrieved {len(retrieved)} items")
    if retrieved:
        item = retrieved[0]
        print(f"  ID: {item.id}")
        print(f"  Title: {item.title}")
        print(f"  Assigned to: {item.assigned_worker}")
    
    # Step 4: Check queue stats
    print("\n4. Final queue stats...")
    stats = await work_queue.get_queue_stats()
    print(f"Total queued: {stats['total_queued']}")
    print(f"HIGH queue length: {stats['priority_queues']['HIGH']['length']}")
    
    # Clean up
    await work_queue.cleanup()
    await redis_client.disconnect()
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_complete_flow())