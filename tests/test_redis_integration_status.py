#!/usr/bin/env python3
"""Test Redis integration status."""

import asyncio
import logging
import sys
sys.path.insert(0, '/workspaces/cwmai')
from scripts.redis_integration.redis_client import RedisClient
from scripts.redis_work_queue import RedisWorkQueue
from scripts.redis_task_persistence import RedisTaskPersistence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_redis_integration():
    """Test all Redis integration components."""
    logger.info("Testing Redis Integration Status...")
    
    # Test Redis connection
    redis_client = RedisClient()
    await redis_client.connect()
    logger.info("✅ Redis connection: OK")
    
    # Test work queue
    work_queue = RedisWorkQueue()
    await work_queue.initialize()
    stats = await work_queue.get_queue_stats()
    logger.info(f"✅ Redis work queue: OK - Stats: {stats}")
    
    # Test task persistence
    task_persistence = RedisTaskPersistence()
    await task_persistence.initialize()
    completion_stats = await task_persistence.get_completion_stats()
    logger.info(f"✅ Redis task persistence: OK - Stats: {completion_stats}")
    
    # Check for stuck tasks
    all_queues = [
        "cwmai:work_queue:critical",
        "cwmai:work_queue:high",
        "cwmai:work_queue:medium", 
        "cwmai:work_queue:low",
        "cwmai:work_queue:background"
    ]
    
    total_items = 0
    for queue in all_queues:
        try:
            info = await redis_client.xinfo_stream(queue)
            length = info.get('length', 0)
            total_items += length
            if length > 0:
                logger.info(f"  Queue {queue}: {length} items")
        except:
            pass
    
    logger.info(f"\nTotal items in all queues: {total_items}")
    
    # Clean up
    await redis_client.disconnect()
    
    return {
        'redis_connected': True,
        'work_queue_active': True,
        'task_persistence_active': True,
        'total_queued_items': total_items,
        'queue_stats': stats,
        'completion_stats': completion_stats
    }

if __name__ == "__main__":
    result = asyncio.run(test_redis_integration())
    print(f"\nRedis Integration Summary:")
    print(f"  Redis Connected: {result['redis_connected']}")
    print(f"  Work Queue Active: {result['work_queue_active']}")
    print(f"  Task Persistence Active: {result['task_persistence_active']}")
    print(f"  Total Queued Items: {result['total_queued_items']}")
    print(f"  Queue Stats: {result['queue_stats']}")
    print(f"  Completion Stats: {result['completion_stats']}")