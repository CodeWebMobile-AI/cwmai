#!/usr/bin/env python3
"""Recover stuck Claude tasks and unblock the queue."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from scripts.claude_execution_queue import ClaudeExecutionQueue, ClaudeTaskStatus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def recover_stuck_tasks(dry_run=True):
    """Recover tasks that have been stuck in processing for too long."""
    queue = ClaudeExecutionQueue()
    
    print("=== CLAUDE TASK RECOVERY ===")
    
    # Get current stats
    stats = await queue.get_stats()
    print(f"\nCurrent Queue Status:")
    print(f"  Pending: {stats['pending']}")
    print(f"  Processing: {stats['processing']}")
    print(f"  Completed: {stats['completed']}")
    
    # Check for stuck tasks
    processing_tasks = queue.get_processing_tasks()
    stuck_tasks = []
    
    if processing_tasks:
        current_time = datetime.now(timezone.utc)
        
        for task in processing_tasks:
            if task.get('started_at'):
                try:
                    dt = datetime.fromisoformat(task['started_at'].replace('Z', '+00:00'))
                    duration = (current_time - dt).total_seconds()
                    
                    # Consider tasks stuck if running for more than 2 hours
                    if duration > 7200:  # 2 hours
                        stuck_tasks.append({
                            'task': task,
                            'duration_hours': duration / 3600
                        })
                except Exception as e:
                    print(f"Error parsing date: {e}")
    
    if not stuck_tasks:
        print("\n✅ No stuck tasks found!")
        return
    
    print(f"\n⚠️ Found {len(stuck_tasks)} stuck tasks:")
    for item in stuck_tasks:
        task = item['task']
        duration = item['duration_hours']
        print(f"  - {task['repository']} #{task['issue_number']} "
              f"(executor: {task.get('executor_id', 'unknown')}, "
              f"running for {duration:.1f} hours)")
    
    if dry_run:
        print("\n🔍 DRY RUN MODE - No changes will be made")
        print("Run with --execute to actually recover these tasks")
        return
    
    print("\n🔧 Recovering stuck tasks...")
    
    # Manually recover each stuck task
    recovered = 0
    failed = 0
    
    for item in stuck_tasks:
        task = item['task']
        task_id = task['id']
        
        try:
            # Move task back to pending state
            await queue.fail_task(
                task_id, 
                f"Task timed out after {item['duration_hours']:.1f} hours - recovering",
                retry=True
            )
            
            print(f"  ✅ Recovered: {task['repository']} #{task['issue_number']}")
            recovered += 1
            
        except Exception as e:
            print(f"  ❌ Failed to recover {task['repository']} #{task['issue_number']}: {e}")
            failed += 1
    
    print(f"\n📊 Recovery Summary:")
    print(f"  Successfully recovered: {recovered}")
    print(f"  Failed to recover: {failed}")
    
    # Check new stats
    new_stats = await queue.get_stats()
    print(f"\nNew Queue Status:")
    print(f"  Pending: {new_stats['pending']} (was {stats['pending']})")
    print(f"  Processing: {new_stats['processing']} (was {stats['processing']})")
    
    # Check repository status
    repo_status = queue.get_repository_status()
    available_repos = sum(1 for status in repo_status.values() if not status['active'])
    print(f"\n🗂️ Repository Availability:")
    print(f"  Total repositories: {len(repo_status)}")
    print(f"  Available for new work: {available_repos}")
    
    if available_repos > 0:
        print("\n✅ Repositories are now unblocked and ready for new work!")
    else:
        print("\n⚠️ All repositories still have active tasks")


async def check_claude_processes():
    """Check if Claude processes are actually running."""
    print("\n🔍 Checking for Claude processes...")
    
    try:
        # Check for any claude processes
        result = await asyncio.create_subprocess_exec(
            'ps', 'aux',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await result.communicate()
        
        lines = stdout.decode('utf-8').split('\n')
        claude_processes = [line for line in lines if 'claude' in line.lower() and 'python' not in line]
        
        if claude_processes:
            print(f"Found {len(claude_processes)} Claude processes:")
            for proc in claude_processes[:5]:
                print(f"  {proc[:120]}...")
        else:
            print("❌ No Claude processes found running")
            print("This explains why tasks are stuck - the executors aren't actually running Claude")
            
    except Exception as e:
        print(f"Error checking processes: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Recover stuck Claude tasks")
    parser.add_argument('--execute', action='store_true', 
                        help='Actually recover tasks (default is dry run)')
    parser.add_argument('--check-processes', action='store_true',
                        help='Also check for running Claude processes')
    
    args = parser.parse_args()
    
    async def main():
        await recover_stuck_tasks(dry_run=not args.execute)
        
        if args.check_processes:
            await check_claude_processes()
    
    asyncio.run(main())