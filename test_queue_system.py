#!/usr/bin/env python3
"""
Test script to verify the queue system is working properly.
"""

import asyncio
import sys
sys.path.append('/workspaces/cwmai')

from scripts.work_item_types import TaskPriority, WorkItem
from scripts.redis_work_queue import RedisWorkQueue
from scripts.redis_integration.redis_client import get_redis_client
import uuid
from datetime import datetime, timezone


async def test_priority_enum():
    """Test priority enum handling."""
    print("Testing TaskPriority enum...")
    
    # Test enum properties
    for priority in TaskPriority:
        print(f"  {priority.name}: value={priority.value}, type={type(priority)}")
        print(f"    isinstance check: {isinstance(priority, TaskPriority)}")
        print(f"    has name: {hasattr(priority, 'name')}, has value: {hasattr(priority, 'value')}")
    
    print("\nTesting enum comparison...")
    high1 = TaskPriority.HIGH
    high2 = TaskPriority.HIGH
    print(f"  HIGH == HIGH: {high1 == high2}")
    print(f"  HIGH is HIGH: {high1 is high2}")
    print(f"  HIGH in dict: {high1 in {TaskPriority.HIGH: 'test'}}")


async def test_queue_operations():
    """Test queue operations."""
    print("\nTesting queue operations...")
    
    # Initialize Redis client
    redis_client = await get_redis_client()
    queue = RedisWorkQueue(redis_client)
    await queue.initialize()
    
    # Create test work items
    work_items = []
    for i, priority in enumerate([TaskPriority.HIGH, TaskPriority.MEDIUM, TaskPriority.LOW]):
        item = WorkItem(
            id=f"TEST-{uuid.uuid4().hex[:8]}",
            task_type="testing",
            title=f"Test task {i+1}",
            description=f"Testing priority {priority.name}",
            priority=priority,
            repository=None,
            estimated_cycles=1,
            created_at=datetime.now(timezone.utc)
        )
        work_items.append(item)
        print(f"  Created: {item.title} with priority {item.priority.name}")
    
    # Add items to queue
    print("\nAdding items to queue...")
    await queue.add_work_batch(work_items)
    
    # Force flush
    await queue._flush_buffer()
    
    # Check queue stats
    print("\nChecking queue stats...")
    stats = await queue.get_queue_stats()
    print(f"  Total queued: {stats.get('total_queued', 0)}")
    print(f"  Buffer size: {stats.get('buffer_size', 0)}")
    for priority_name, priority_stats in stats.get('priority_queues', {}).items():
        print(f"  {priority_name}: {priority_stats.get('length', 0)} items")
    
    # Try to get work
    print("\nTrying to get work for test_worker...")
    retrieved = await queue.get_work_for_worker("test_worker", "general", count=5)
    print(f"  Retrieved {len(retrieved)} items")
    for item in retrieved:
        print(f"    - {item.title} (priority: {item.priority.name})")
    
    # Clean up
    await queue.cleanup()
    await redis_client.disconnect()
    print("\nTest completed!")


async def main():
    """Run all tests."""
    await test_priority_enum()
    await test_queue_operations()


if __name__ == "__main__":
    asyncio.run(main())