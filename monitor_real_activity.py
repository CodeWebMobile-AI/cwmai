#!/usr/bin/env python3
"""Monitor real worker activity and queue throughput"""

import asyncio
import sys
from datetime import datetime, timezone
sys.path.append('/workspaces/cwmai')

from scripts.redis_work_queue import RedisWorkQueue
from scripts.redis_integration.redis_client import get_redis_client
from scripts.redis_lockfree_state_manager import RedisLockFreeStateManager

async def main():
    print("=== Real-Time Worker Activity Monitor ===\n")
    
    # Initialize components
    redis_client = await get_redis_client()
    work_queue = RedisWorkQueue(redis_client=redis_client)
    await work_queue.initialize()
    
    state_manager = RedisLockFreeStateManager()
    await state_manager.initialize()
    
    # Get initial state
    initial_state = await state_manager.get_state('system_state')
    initial_completed = initial_state.get('metrics', {}).get('total_work_completed', 0) if initial_state else 0
    
    print(f"Starting metrics at {datetime.now()}")
    print(f"Initial completed tasks: {initial_completed}")
    
    # Monitor for 30 seconds
    print("\nMonitoring for 30 seconds...\n")
    
    for i in range(6):  # 6 iterations of 5 seconds each
        await asyncio.sleep(5)
        
        # Get current stats
        queue_stats = await work_queue.get_queue_stats()
        state = await state_manager.get_state('system_state')
        if not state:
            state = {}
        metrics = state.get('metrics', {})
        workers = state.get('workers', {})
        
        # Calculate throughput
        current_completed = metrics.get('total_work_completed', 0)
        completed_in_interval = current_completed - initial_completed
        
        # Count worker states
        worker_states = {}
        for worker_id, worker_info in workers.items():
            status = worker_info.get('status', 'unknown')
            worker_states[status] = worker_states.get(status, 0) + 1
        
        # Get queue distribution
        queue_dist = []
        total_available = 0
        for priority, stats in queue_stats['priority_queues'].items():
            length = stats.get('length', 0)
            pending = stats.get('pending', 0)
            available = length - pending
            total_available += available
            if length > 0:
                queue_dist.append(f"{priority}:{length}({pending} pending)")
        
        print(f"[{i*5}s] Completed: {completed_in_interval} | Queue: {total_available} available | Workers: {worker_states} | Queues: {' '.join(queue_dist)}")
        
        # Show sample of active work
        active_work = []
        for worker_id, worker_info in workers.items():
            if worker_info.get('current_work'):
                work = worker_info['current_work']
                active_work.append(f"{worker_id}: {work.get('title', 'unknown')[:30]}...")
        
        if active_work:
            print(f"       Active: {' | '.join(active_work[:3])}")
    
    # Final summary
    final_state = await state_manager.get_state('system_state')
    final_completed = final_state.get('metrics', {}).get('total_work_completed', 0) if final_state else 0
    total_completed = final_completed - initial_completed
    
    print(f"\n=== Summary ===")
    print(f"Total tasks completed in 30s: {total_completed}")
    print(f"Average throughput: {total_completed/30:.2f} tasks/second")
    print(f"Total work created: {final_state.get('metrics', {}).get('total_work_created', 0)}")

if __name__ == "__main__":
    asyncio.run(main())