#!/usr/bin/env python3
"""
Check Claude Execution Queue Status

Shows detailed information about the Claude execution queue and
can recover stuck tasks.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.claude_execution_queue import ClaudeExecutionQueue, ClaudeTaskStatus


async def check_queue_status():
    """Check and display queue status."""
    queue = ClaudeExecutionQueue()
    
    print("🔍 Claude Execution Queue Status")
    print("=" * 60)
    
    # Get stats
    stats = await queue.get_stats()
    print(f"\n📊 Overview:")
    print(f"   Pending: {stats['pending']}")
    print(f"   Processing: {stats['processing']}")
    print(f"   Completed: {stats['completed']}")
    print(f"   Failed: {stats.get('failed', 0)}")
    
    # Show pending tasks
    if stats['pending'] > 0:
        print(f"\n📋 Pending Tasks ({stats['pending']}):")
        pending = queue.get_pending_tasks()
        for i, task in enumerate(pending[:10], 1):  # Show first 10
            created = datetime.fromisoformat(task['created_at'].replace('Z', '+00:00'))
            age = datetime.now(timezone.utc) - created
            print(f"   {i}. {task['repository']} #{task['issue_number']}")
            print(f"      Created: {age.total_seconds()/60:.1f} minutes ago")
            print(f"      ID: {task['id']}")
        if len(pending) > 10:
            print(f"   ... and {len(pending) - 10} more")
    
    # Show processing tasks
    if stats['processing'] > 0:
        print(f"\n⚙️  Processing Tasks ({stats['processing']}):")
        processing = queue.get_processing_tasks()
        for i, task in enumerate(processing, 1):
            started = datetime.fromisoformat(task['started_at'].replace('Z', '+00:00'))
            duration = datetime.now(timezone.utc) - started
            print(f"   {i}. {task['repository']} #{task['issue_number']}")
            print(f"      Started: {duration.total_seconds()/60:.1f} minutes ago")
            print(f"      Executor: {task.get('executor_id', 'unknown')}")
            print(f"      ID: {task['id']}")
            
            # Flag if running too long (>30 minutes)
            if duration.total_seconds() > 1800:
                print(f"      ⚠️  WARNING: Running for over 30 minutes!")
    
    # Check for stuck tasks
    print("\n🔧 Checking for stuck tasks...")
    stuck_count = 0
    for task in processing:
        started = datetime.fromisoformat(task['started_at'].replace('Z', '+00:00'))
        duration = datetime.now(timezone.utc) - started
        if duration.total_seconds() > 3600:  # 1 hour
            stuck_count += 1
    
    if stuck_count > 0:
        print(f"⚠️  Found {stuck_count} potentially stuck tasks (running > 1 hour)")
        print("\nTo recover stuck tasks, run:")
        print("  python3 check_claude_queue.py --recover")
    else:
        print("✅ No stuck tasks found")
    
    # Check for duplicates
    print("\n🔍 Checking for duplicate tasks...")
    seen = {}
    duplicates = 0
    for task in pending:
        key = f"{task['repository']}#{task['issue_number']}"
        if key in seen:
            duplicates += 1
            print(f"   Duplicate: {key}")
        else:
            seen[key] = task['id']
    
    if duplicates > 0:
        print(f"⚠️  Found {duplicates} duplicate tasks")
    else:
        print("✅ No duplicate tasks found")
    
    return stats, stuck_count


async def recover_stuck_tasks():
    """Recover stuck tasks by moving them back to pending."""
    queue = ClaudeExecutionQueue()
    
    print("\n♻️  Recovering stuck tasks...")
    # Recover tasks stuck for more than 1 hour
    recovered = await queue.recover_stale_tasks(timeout_seconds=3600)
    
    print(f"✅ Recovered {len(recovered)} stuck tasks")
    for task in recovered:
        print(f"   - {task['repository']} #{task['issue_number']}")


async def clear_duplicates():
    """Remove duplicate tasks from the queue."""
    queue = ClaudeExecutionQueue()
    
    print("\n🧹 Removing duplicate tasks...")
    
    # Load queue data
    queue._load_queue()
    
    seen = {}
    cleaned_queue = []
    removed = 0
    
    # Keep first occurrence of each repository#issue
    for task in queue._queue:
        key = f"{task.repository}#{task.issue_number}"
        if key not in seen:
            seen[key] = True
            cleaned_queue.append(task)
        else:
            removed += 1
            print(f"   Removing duplicate: {key} (ID: {task.id})")
    
    if removed > 0:
        queue._queue = cleaned_queue
        queue._save_queue()
        print(f"✅ Removed {removed} duplicate tasks")
    else:
        print("✅ No duplicates to remove")


async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check Claude Execution Queue Status")
    parser.add_argument('--recover', action='store_true', help='Recover stuck tasks')
    parser.add_argument('--clean-duplicates', action='store_true', help='Remove duplicate tasks')
    parser.add_argument('--clear-all', action='store_true', help='Clear all pending tasks (use with caution!)')
    
    args = parser.parse_args()
    
    # Always show status first
    stats, stuck_count = await check_queue_status()
    
    if args.recover:
        await recover_stuck_tasks()
    
    if args.clean_duplicates:
        await clear_duplicates()
    
    if args.clear_all:
        response = input("\n⚠️  Are you sure you want to clear ALL pending tasks? (yes/no): ")
        if response.lower() == 'yes':
            queue = ClaudeExecutionQueue()
            queue._load_queue()
            cleared = len(queue._queue)
            queue._queue = []
            queue._save_queue()
            print(f"✅ Cleared {cleared} pending tasks")
        else:
            print("❌ Clear operation cancelled")
    
    # Show final status if any action was taken
    if args.recover or args.clean_duplicates or args.clear_all:
        print("\n📊 Final Status:")
        final_stats = await ClaudeExecutionQueue().get_stats()
        print(f"   Pending: {final_stats['pending']}")
        print(f"   Processing: {final_stats['processing']}")
        print(f"   Completed: {final_stats['completed']}")
        print(f"   Failed: {final_stats.get('failed', 0)}")


if __name__ == "__main__":
    asyncio.run(main())