#!/usr/bin/env python3
"""Simple debug of work queue status"""

import asyncio
import sys
sys.path.append('/workspaces/cwmai')

from scripts.redis_work_queue import RedisWorkQueue
from scripts.redis_integration.redis_client import get_redis_client

async def main():
    print("=== Checking Work Queue Status ===\n")
    
    # Initialize Redis components
    redis_client = await get_redis_client()
    work_queue = RedisWorkQueue(redis_client=redis_client)
    await work_queue.initialize()
    
    # Get queue stats
    queue_stats = await work_queue.get_queue_stats()
    print("Queue Statistics:")
    print(f"  Buffer size: {queue_stats['buffer_size']}")
    print(f"  Total queued: {queue_stats['total_queued']}")
    print(f"  Total pending: {queue_stats['total_pending']}")
    
    print("\nPriority Queues:")
    total_items = 0
    total_pending = 0
    for priority, stats in queue_stats['priority_queues'].items():
        length = stats.get('length', 0)
        pending = stats.get('pending', 0)
        total_items += length
        total_pending += pending
        print(f"  {priority}: {length} items in stream, {pending} pending")
    
    print(f"\nTotal items across all streams: {total_items}")
    print(f"Total pending items: {total_pending}")
    
    # Check each stream directly
    print("\nDirect stream checks:")
    streams = {
        'CRITICAL': 'cwmai:work_queue:critical',
        'HIGH': 'cwmai:work_queue:high', 
        'MEDIUM': 'cwmai:work_queue:medium',
        'LOW': 'cwmai:work_queue:low',
        'BACKGROUND': 'cwmai:work_queue:background'
    }
    
    for priority, stream_name in streams.items():
        try:
            # Get stream length
            info = await redis_client.xlen(stream_name)
            print(f"  {stream_name}: {info} messages")
            
            # Get last few entries to see what's in there
            if info > 0:
                entries = await redis_client.xrange(stream_name, count=2)
                if entries:
                    print(f"    Sample entry: {entries[0][0]}")
        except Exception as e:
            print(f"  {stream_name}: Error - {e}")
    
    # Check for stuck pending messages
    print("\nChecking for stuck messages...")
    requeued = await work_queue.requeue_stuck_work(timeout_minutes=1)
    print(f"  Requeued {requeued} stuck messages")
    
    # Get a sample of work for a test worker
    print("\nTrying to get work for test worker...")
    test_work = await work_queue.get_work_for_worker("test_worker", specialization="general", count=3)
    print(f"  Got {len(test_work)} work items")
    for item in test_work:
        print(f"    - {item.title} ({item.task_type})")

if __name__ == "__main__":
    asyncio.run(main())