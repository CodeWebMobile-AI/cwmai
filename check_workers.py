#!/usr/bin/env python3
"""
Quick Worker Status Check

Simple script to check what workers are doing right now.
"""

import asyncio
import json
import os
from datetime import datetime
from tabulate import tabulate

# Try to get status from the continuous_orchestrator_state.json file
def check_workers_from_file():
    """Check worker status from state file."""
    state_file = '/workspaces/cwmai/continuous_orchestrator_state.json'
    
    if not os.path.exists(state_file):
        print("No orchestrator state file found. The system may not be running.")
        return
    
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        print("\n" + "="*80)
        print(f"WORKER STATUS (from state file)")
        print(f"Last Updated: {state.get('last_updated', 'Unknown')}")
        print("="*80)
        
        # Display metrics
        metrics = state.get('metrics', {})
        print(f"\nSystem Metrics:")
        print(f"  Total Work Completed: {metrics.get('total_work_completed', 0)}")
        print(f"  Total Work Created: {metrics.get('total_work_created', 0)}")
        print(f"  Total Errors: {metrics.get('total_errors', 0)}")
        print(f"  Worker Utilization: {metrics.get('worker_utilization', 0)*100:.1f}%")
        print(f"  Work Per Hour: {metrics.get('work_per_hour', 0):.2f}")
        
        # Display workers
        workers = state.get('workers', {})
        if workers:
            print(f"\n\nWorkers ({len(workers)} total):")
            headers = ['Worker ID', 'Status', 'Specialization', 'Completed', 'Errors']
            rows = []
            
            for worker_id, worker_info in workers.items():
                rows.append([
                    worker_id,
                    worker_info.get('status', 'unknown'),
                    worker_info.get('specialization', 'general'),
                    worker_info.get('total_completed', 0),
                    worker_info.get('total_errors', 0)
                ])
            
            print(tabulate(rows, headers=headers, tablefmt='grid'))
        else:
            print("\nNo workers found in state file.")
        
        # Display work queue
        work_queue = state.get('work_queue', [])
        print(f"\n\nWork Queue ({len(work_queue)} items):")
        if work_queue:
            headers = ['Title', 'Type', 'Priority', 'Repository']
            rows = []
            
            for i, item in enumerate(work_queue[:10]):  # Show first 10
                rows.append([
                    item.get('title', '')[:50] + '...' if len(item.get('title', '')) > 50 else item.get('title', ''),
                    item.get('task_type', 'unknown'),
                    item.get('priority', 'medium'),
                    item.get('repository', 'N/A')
                ])
            
            print(tabulate(rows, headers=headers, tablefmt='grid'))
            
            if len(work_queue) > 10:
                print(f"\n... and {len(work_queue) - 10} more items in queue")
        else:
            print("Queue is empty.")
        
        # Display completed work (recent)
        completed = state.get('completed_work', [])
        if completed:
            print(f"\n\nRecent Completed Work (last 5):")
            headers = ['Title', 'Type', 'Repository', 'Completed At']
            rows = []
            
            for item in completed[-5:]:
                completed_at = item.get('completed_at', 'Unknown')
                if completed_at != 'Unknown':
                    try:
                        dt = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
                        completed_at = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        pass
                
                rows.append([
                    item.get('title', '')[:40] + '...' if len(item.get('title', '')) > 40 else item.get('title', ''),
                    item.get('task_type', 'unknown'),
                    item.get('repository', 'N/A')[:20],
                    completed_at
                ])
            
            print(tabulate(rows, headers=headers, tablefmt='grid'))
            
    except Exception as e:
        print(f"Error reading state file: {e}")


# Try to get live status from Redis
async def check_workers_live():
    """Check live worker status from Redis."""
    try:
        from scripts.redis_work_queue import RedisWorkQueue
        from scripts.redis_lockfree_state_manager import RedisLockFreeStateManager
        
        print("\n" + "="*80)
        print("LIVE WORKER STATUS (from Redis)")
        print("="*80)
        
        # Initialize Redis components
        redis_queue = RedisWorkQueue()
        redis_state = RedisLockFreeStateManager()
        await redis_state.initialize()
        
        # Get queue stats
        queue_stats = await redis_queue.get_queue_stats()
        print(f"\nQueue Status:")
        print(f"  Total Queued: {queue_stats.get('total_queued', 0)}")
        print(f"  Processing: {queue_stats.get('total_processing', 0)}")
        print(f"  Completed: {queue_stats.get('total_completed', 0)}")
        print(f"  Failed: {queue_stats.get('total_failed', 0)}")
        
        # Get active workers
        active_workers = await redis_state.get_set_members("active_workers")
        print(f"\nActive Workers: {len(active_workers)}")
        
        if active_workers:
            headers = ['Worker ID', 'Status', 'Current Task', 'Completed', 'Errors']
            rows = []
            
            for worker_id in active_workers:
                # Get worker state
                worker_state = await redis_state.get_state(f"workers:{worker_id}")
                completed = await redis_state.get_counter(f"worker:{worker_id}:completed")
                errors = await redis_state.get_counter(f"worker:{worker_id}:errors")
                
                current_task = 'Idle'
                if worker_state and worker_state.get('current_task'):
                    current_task = worker_state['current_task'].get('title', 'Working...')[:40] + '...'
                
                rows.append([
                    worker_id[:20],
                    worker_state.get('status', 'unknown') if worker_state else 'unknown',
                    current_task,
                    completed,
                    errors
                ])
            
            print(tabulate(rows, headers=headers, tablefmt='grid'))
        
        await redis_state.close()
        
    except ImportError:
        print("\nRedis components not available. Showing file-based status only.")
    except Exception as e:
        print(f"\nError getting live status: {e}")
        print("Falling back to file-based status.")


async def main():
    """Main entry point."""
    print("Checking worker status...")
    
    # First check file-based status
    check_workers_from_file()
    
    # Then try to get live status
    await check_workers_live()
    
    print("\n" + "="*80)
    print("Status check complete.")
    print("\nTo monitor continuously, run:")
    print("  python scripts/worker_status_monitor.py --continuous")


if __name__ == "__main__":
    asyncio.run(main())