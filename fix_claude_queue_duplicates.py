#!/usr/bin/env python3
"""
Fix Claude Queue Duplicates

This script fixes the issue where multiple executors are working on the same
issue by moving duplicates back to pending.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Set

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.claude_execution_queue import ClaudeExecutionQueue, ClaudeTask, ClaudeTaskStatus


async def fix_duplicate_processing():
    """Fix duplicate processing tasks by keeping only the oldest one."""
    queue = ClaudeExecutionQueue()
    
    print("🔍 Fixing Claude Queue Duplicate Processing Tasks")
    print("=" * 60)
    
    # Load queue
    queue._load_queue()
    
    # Track which issues are being processed
    processing_by_issue: Dict[str, List[ClaudeTask]] = {}
    
    # Group processing tasks by repository#issue
    for task_id, task in queue._processing.items():
        key = f"{task.repository}#{task.issue_number}"
        if key not in processing_by_issue:
            processing_by_issue[key] = []
        processing_by_issue[key].append(task)
    
    # Find duplicates
    tasks_to_requeue = []
    for key, tasks in processing_by_issue.items():
        if len(tasks) > 1:
            print(f"\n⚠️  Found {len(tasks)} executors working on {key}:")
            
            # Sort by started_at to keep the oldest
            tasks.sort(key=lambda t: t.started_at or datetime.now(timezone.utc))
            
            # Keep the first one, requeue the rest
            keeper = tasks[0]
            print(f"   ✅ Keeping: {keeper.executor_id} (started {keeper.started_at})")
            
            for task in tasks[1:]:
                print(f"   ♻️  Requeuing: {task.executor_id} (started {task.started_at})")
                tasks_to_requeue.append(task)
    
    if tasks_to_requeue:
        print(f"\n📋 Moving {len(tasks_to_requeue)} duplicate tasks back to pending...")
        
        # Move tasks back to pending
        for task in tasks_to_requeue:
            # Remove from processing
            if task.id in queue._processing:
                del queue._processing[task.id]
            
            # Reset task state
            task.status = ClaudeTaskStatus.PENDING
            task.started_at = None
            task.executor_id = None
            
            # Add back to pending queue
            queue._queue.append(task)
        
        # Save updated queue
        queue._save_queue()
        
        print(f"✅ Moved {len(tasks_to_requeue)} tasks back to pending")
    else:
        print("\n✅ No duplicate processing tasks found")
    
    # Show final stats
    print("\n📊 Final Queue Status:")
    stats = await queue.get_stats()
    print(f"   Pending: {stats['pending']}")
    print(f"   Processing: {stats['processing']}")
    print(f"   Completed: {stats['completed']}")
    
    # Show what's processing now
    if stats['processing'] > 0:
        print(f"\n🔄 Currently Processing:")
        processing = queue.get_processing_tasks()
        for task in processing:
            print(f"   • {task['repository']} #{task['issue_number']} by {task.get('executor_id', 'unknown')}")


async def check_executor_pool_status():
    """Check if the executor pool is actually running."""
    print("\n🔍 Checking Claude Executor Pool Status")
    print("=" * 60)
    
    # Check for pool stats file
    pool_stats_file = "dynamic_claude_pool_stats.json"
    if os.path.exists(pool_stats_file):
        try:
            with open(pool_stats_file, 'r') as f:
                stats = json.load(f)
            
            print(f"✅ Pool stats file found")
            print(f"   Executors: {stats.get('num_executors', 0)}")
            print(f"   Total Completed: {stats.get('total_completed', 0)}")
            
            # Check if stats are recent
            if 'scaling_metrics' in stats:
                events = stats['scaling_metrics'].get('recent_scaling_events', [])
                if events:
                    last_event = events[-1]
                    print(f"   Last scaling event: {last_event.get('timestamp', 'Unknown')}")
        except Exception as e:
            print(f"⚠️  Error reading pool stats: {e}")
    else:
        print("❌ No pool stats file found - executor pool may not be running")
        print("\nTo start the executor pool, run:")
        print("  python3 run_hybrid_continuous.py")


async def main():
    """Main function."""
    await fix_duplicate_processing()
    await check_executor_pool_status()


if __name__ == "__main__":
    asyncio.run(main())