#!/usr/bin/env python3
"""Debug why Claude executors are not picking up tasks."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from scripts.claude_execution_queue import ClaudeExecutionQueue
from scripts.dynamic_claude_executor_pool import DynamicClaudeExecutorPool

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def debug_queue_and_pool():
    """Debug the queue and pool state."""
    print("=== CLAUDE POOL DEBUG ===")
    
    # Check queue state
    queue = ClaudeExecutionQueue()
    stats = await queue.get_stats()
    print(f"\n📦 Queue Statistics:")
    print(f"  Pending: {stats['pending']}")
    print(f"  Processing: {stats['processing']}")
    print(f"  Completed: {stats['completed']}")
    
    # Check repository status
    repo_status = queue.get_repository_status()
    print(f"\n🗂️ Repository Status:")
    for repo, status in repo_status.items():
        print(f"  {repo}:")
        print(f"    Pending: {status['pending']}")
        print(f"    Processing: {status['processing']}")
        print(f"    Active: {status['active']}")
    
    # Check pending tasks
    pending_tasks = queue.get_pending_tasks()
    print(f"\n📋 Pending Tasks Details:")
    if pending_tasks:
        # Group by repository
        repos_with_pending = {}
        for task in pending_tasks:
            repo = task['repository']
            if repo not in repos_with_pending:
                repos_with_pending[repo] = []
            repos_with_pending[repo].append(task)
        
        for repo, tasks in repos_with_pending.items():
            print(f"\n  Repository: {repo}")
            # Check if this repo has active tasks
            has_active = queue.has_active_task_for_repository(repo)
            print(f"  Has Active Task: {has_active}")
            print(f"  Pending Count: {len(tasks)}")
            for task in tasks[:3]:  # Show first 3
                created = task.get('created_at', 'Unknown')
                if created != 'Unknown':
                    try:
                        dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                        age = (datetime.now(timezone.utc) - dt).total_seconds()
                        created = f"{age/3600:.1f} hours ago"
                    except:
                        pass
                print(f"    - Issue #{task.get('issue_number')} (created {created})")
    
    # Check processing tasks
    processing_tasks = queue.get_processing_tasks()
    print(f"\n🔄 Processing Tasks Details:")
    if processing_tasks:
        for task in processing_tasks:
            repo = task['repository']
            issue = task['issue_number']
            executor = task.get('executor_id', 'Unknown')
            started = task.get('started_at', 'Unknown')
            if started != 'Unknown':
                try:
                    dt = datetime.fromisoformat(started.replace('Z', '+00:00'))
                    duration = (datetime.now(timezone.utc) - dt).total_seconds()
                    started = f"{duration/3600:.1f} hours ago"
                except:
                    pass
            print(f"  - {repo} #{issue} by {executor} (started {started})")
    
    # Test get_next_task logic
    print(f"\n🧪 Testing get_next_task Logic:")
    
    # Simulate an executor trying to get work
    test_executor_id = "test_executor_999"
    
    # Check which repos are blocked
    blocked_repos = []
    available_repos = []
    
    for repo in repos_with_pending.keys():
        if queue.has_active_task_for_repository(repo):
            blocked_repos.append(repo)
        else:
            available_repos.append(repo)
    
    print(f"  Blocked repositories (have active tasks): {len(blocked_repos)}")
    for repo in blocked_repos[:5]:
        print(f"    - {repo}")
    
    print(f"\n  Available repositories (no active tasks): {len(available_repos)}")
    for repo in available_repos[:5]:
        print(f"    - {repo}")
    
    # The issue seems to be that repositories are marked as having "active" tasks
    # even though the executors show no current_task. Let's investigate further.
    
    print(f"\n🔍 Investigating Repository Lock Issue:")
    
    # Check if processing tasks are actually being worked on
    if processing_tasks:
        print(f"  Found {len(processing_tasks)} tasks marked as 'running'")
        print(f"  These tasks are blocking their repositories from getting new work")
        
        # Check if these tasks are stuck
        stuck_tasks = []
        for task in processing_tasks:
            if task.get('started_at'):
                try:
                    dt = datetime.fromisoformat(task['started_at'].replace('Z', '+00:00'))
                    duration = (datetime.now(timezone.utc) - dt).total_seconds()
                    if duration > 3600:  # More than 1 hour
                        stuck_tasks.append((task, duration))
                except:
                    pass
        
        if stuck_tasks:
            print(f"\n  ⚠️ Found {len(stuck_tasks)} potentially stuck tasks:")
            for task, duration in stuck_tasks[:5]:
                print(f"    - {task['repository']} #{task['issue_number']} "
                      f"(running for {duration/3600:.1f} hours)")
    
    print("\n💡 DIAGNOSIS:")
    print("The issue is that tasks are stuck in 'running' state, blocking their repositories.")
    print("The executors show 0% utilization because they're not actually processing these tasks.")
    print("This suggests the executors might have crashed or lost connection to these tasks.")
    
    print("\n🔧 RECOMMENDATIONS:")
    print("1. Clear stuck tasks from processing state")
    print("2. Implement heartbeat monitoring for executors")
    print("3. Add automatic recovery for orphaned tasks")
    print("4. Verify Claude CLI is actually executing")

if __name__ == "__main__":
    asyncio.run(debug_queue_and_pool())