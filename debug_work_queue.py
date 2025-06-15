#!/usr/bin/env python3
"""Debug work queue issues."""

import asyncio
import logging
import sys
from datetime import datetime, timezone

sys.path.insert(0, '/workspaces/cwmai/scripts')

from work_item_types import WorkItem, TaskPriority
from redis_work_queue import RedisWorkQueue
from redis_integration.redis_client import RedisClient

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_work_queue():
    """Test work queue operations."""
    logger.info("=== Testing Work Queue ===")
    
    # Initialize work queue
    work_queue = RedisWorkQueue()
    await work_queue.initialize()
    
    # Get initial stats
    stats = await work_queue.get_queue_stats()
    logger.info(f"Initial queue stats: {stats}")
    
    # Create test work items
    test_items = [
        WorkItem(
            id=f"test_{i}",
            task_type="TEST",
            title=f"Test task {i}",
            description=f"Test description {i}",
            priority=TaskPriority.HIGH,
            repository=None,
            estimated_cycles=1,
            created_at=datetime.now(timezone.utc)
        )
        for i in range(3)
    ]
    
    # Add items one by one
    logger.info(f"Adding {len(test_items)} test items...")
    for item in test_items:
        await work_queue.add_work(item)
        logger.debug(f"Added item: {item.id}")
    
    # Force flush
    logger.info("Forcing buffer flush...")
    await work_queue._flush_buffer()
    
    # Wait a bit
    await asyncio.sleep(1)
    
    # Check stats again
    stats_after = await work_queue.get_queue_stats()
    logger.info(f"Queue stats after adding: {stats_after}")
    
    # Try to retrieve items
    logger.info("Attempting to retrieve work items...")
    retrieved = await work_queue.get_work_for_worker("test_worker", "general", count=5)
    logger.info(f"Retrieved {len(retrieved)} items")
    for item in retrieved:
        logger.info(f"  - {item.id}: {item.title}")
    
    # Check Redis directly
    redis_client = work_queue.redis_client
    stream_name = "cwmai:work_queue:high"
    
    # Get stream info
    try:
        info = await redis_client.xinfo_stream(stream_name)
        logger.info(f"\nDirect Redis check - Stream {stream_name}:")
        logger.info(f"  Length: {info.get('length', 0)}")
        logger.info(f"  Groups: {info.get('groups', 0)}")
        logger.info(f"  First entry: {info.get('first-entry')}")
        logger.info(f"  Last entry: {info.get('last-entry')}")
    except Exception as e:
        logger.error(f"Error checking stream: {e}")
    
    # Clean up
    await work_queue.cleanup()
    await redis_client.disconnect()

if __name__ == "__main__":
    asyncio.run(test_work_queue())