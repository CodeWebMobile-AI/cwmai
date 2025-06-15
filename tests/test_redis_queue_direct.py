#!/usr/bin/env python3
"""Test Redis work queue directly."""

import asyncio
import sys
import logging
from datetime import datetime, timezone

sys.path.insert(0, '/workspaces/cwmai/scripts')

from redis_work_queue import RedisWorkQueue
from work_item_types import WorkItem, TaskPriority

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_redis_queue():
    """Test Redis work queue operations."""
    print("=== TESTING REDIS WORK QUEUE DIRECTLY ===\n")
    
    # Create work queue
    queue = RedisWorkQueue()
    await queue.initialize()
    
    # Check initial stats
    print("1. Initial queue stats:")
    stats = await queue.get_queue_stats()
    print(f"   Total queued: {stats['total_queued']}")
    print(f"   Buffer size: {stats['buffer_size']}")
    for priority, pstats in stats['priority_queues'].items():
        print(f"   {priority}: {pstats.get('length', 0)} items")
    
    # Add some test work items
    print("\n2. Adding test work items...")
    test_items = [
        WorkItem(
            id="test_1",
            task_type="SYSTEM_IMPROVEMENT",
            title="Test System Improvement",
            description="Test description",
            priority=TaskPriority.HIGH,
            repository=None,
            estimated_cycles=1
        ),
        WorkItem(
            id="test_2", 
            task_type="NEW_PROJECT",
            title="Test New Project",
            description="Test project description",
            priority=TaskPriority.MEDIUM,
            repository=None,
            estimated_cycles=2
        )
    ]
    
    await queue.add_work_batch(test_items)
    
    # Force flush
    print("\n3. Forcing buffer flush...")
    await queue._flush_buffer()
    
    # Check stats after adding
    print("\n4. Queue stats after adding:")
    stats = await queue.get_queue_stats()
    print(f"   Total queued: {stats['total_queued']}")
    print(f"   Buffer size: {stats['buffer_size']}")
    for priority, pstats in stats['priority_queues'].items():
        print(f"   {priority}: {pstats.get('length', 0)} items")
    
    # Try to get work as a worker
    print("\n5. Getting work for worker 'test_worker'...")
    work_items = await queue.get_work_for_worker(
        worker_id="test_worker",
        specialization="general",
        count=5
    )
    
    print(f"   Retrieved {len(work_items)} items:")
    for item in work_items:
        print(f"   - {item.task_type}: {item.title} (priority: {item.priority.name})")
    
    # Check stats after retrieval
    print("\n6. Queue stats after retrieval:")
    stats = await queue.get_queue_stats()
    print(f"   Total queued: {stats['total_queued']}")
    print(f"   Total pending: {stats['total_pending']}")
    
    # Cleanup
    await queue.cleanup()
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_redis_queue())