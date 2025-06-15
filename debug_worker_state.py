#!/usr/bin/env python3
"""Debug worker state and queue status"""

import asyncio
import sys
sys.path.append('/workspaces/cwmai')

from scripts.redis_work_queue import RedisWorkQueue
from scripts.redis_integration.redis_client import get_redis_client
from scripts.continuous_orchestrator import ContinuousOrchestrator
from scripts.redis_lockfree_state_manager import RedisLockFreeStateManager

async def main():
    print("=== Debugging Worker State ===\n")
    
    # Initialize Redis components
    redis_client = await get_redis_client()
    work_queue = RedisWorkQueue(redis_client=redis_client)
    await work_queue.initialize()
    
    state_manager = RedisLockFreeStateManager()
    await state_manager.initialize()
    
    # Get queue stats
    queue_stats = await work_queue.get_queue_stats()
    print("Queue Statistics:")
    print(f"  Buffer size: {queue_stats['buffer_size']}")
    print(f"  Total queued: {queue_stats['total_queued']}")
    print(f"  Total pending: {queue_stats['total_pending']}")
    
    print("\nPriority Queues:")
    for priority, stats in queue_stats['priority_queues'].items():
        print(f"  {priority}: {stats.get('length', 0)} items, {stats.get('pending', 0)} pending")
    
    # Get system state
    system_state = await state_manager.get_state()
    print(f"\nSystem State:")
    print(f"  Status: {system_state.get('status', 'unknown')}")
    print(f"  Start time: {system_state.get('start_time', 'unknown')}")
    
    workers_state = system_state.get('workers', {})
    print(f"\nWorkers ({len(workers_state)}):")
    
    active_count = 0
    for worker_id, worker_info in workers_state.items():
        status = worker_info.get('status', 'unknown')
        current_work = worker_info.get('current_work')
        if current_work:
            active_count += 1
            print(f"  {worker_id}: {status} - Working on: {current_work.get('title', 'unknown')}")
        else:
            print(f"  {worker_id}: {status} - No current work")
    
    metrics = system_state.get('metrics', {})
    print(f"\nMetrics:")
    print(f"  Total work created: {metrics.get('total_work_created', 0)}")
    print(f"  Total work completed: {metrics.get('total_work_completed', 0)}")
    print(f"  Active workers: {active_count}/{len(workers_state)}")
    
    # Check for stuck pending messages
    print("\nChecking for stuck messages...")
    requeued = await work_queue.requeue_stuck_work(timeout_minutes=1)  # Aggressive 1 minute timeout
    print(f"  Requeued {requeued} stuck messages")
    
    # Check Redis connectivity
    try:
        await redis_client.ping()
        print("\n✓ Redis connection is healthy")
    except Exception as e:
        print(f"\n✗ Redis connection error: {e}")

if __name__ == "__main__":
    asyncio.run(main())