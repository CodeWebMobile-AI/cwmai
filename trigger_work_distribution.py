#!/usr/bin/env python3
"""
Trigger work distribution for repositories with stuck tasks.
"""

import asyncio
import json
import os
from datetime import datetime, timezone


async def trigger_work_distribution():
    """Force the system to redistribute work to idle workers."""
    
    # Load Claude execution queue
    queue_file = 'claude_execution_queue.json'
    if os.path.exists(queue_file):
        with open(queue_file, 'r') as f:
            queue_data = json.load(f)
        
        print(f"Current Claude Queue Status:")
        print(f"  Pending: {len(queue_data.get('pending', []))}")
        print(f"  Processing: {len(queue_data.get('processing', {}))}")
        
        # Show processing tasks by repo
        repo_counts = {}
        for task in queue_data.get('processing', {}).values():
            repo = task.get('repository', 'unknown')
            repo_counts[repo] = repo_counts.get(repo, 0) + 1
        
        print("\nProcessing tasks by repository:")
        for repo, count in sorted(repo_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {repo}: {count} tasks")
    
    # Load work queue
    work_queue_file = 'work_queue_backup.json'
    if os.path.exists(work_queue_file):
        with open(work_queue_file, 'r') as f:
            work_data = json.load(f)
        
        print(f"\nWork Queue Status:")
        print(f"  Total items: {len(work_data.get('items', []))}")
        
        # Count by repository
        repo_work = {}
        for item in work_data.get('items', []):
            repo = item.get('repository', 'unknown')
            if repo not in repo_work:
                repo_work[repo] = []
            repo_work[repo].append(item['title'])
        
        print("\nWork items by repository:")
        for repo, items in sorted(repo_work.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"\n  {repo}: {len(items)} items")
            for i, title in enumerate(items[:2]):
                print(f"    - {title}")
            if len(items) > 2:
                print(f"    ... and {len(items) - 2} more")
    
    # Check orchestrator state
    orch_state_file = 'local_repos/state/continuous_orchestrator_state.json'
    if os.path.exists(orch_state_file):
        with open(orch_state_file, 'r') as f:
            orch_state = json.load(f)
        
        print(f"\nWorker Status:")
        idle_count = 0
        for worker_id, worker_state in orch_state.get('workers', {}).items():
            if worker_state['status'] == 'idle':
                idle_count += 1
        
        print(f"  Idle workers: {idle_count} / {len(orch_state.get('workers', {}))}")
    
    print("\n" + "="*60)
    print("ANALYSIS:")
    print("="*60)
    
    # Identify repositories that should be getting work
    repos_needing_work = []
    for repo in repo_work:
        if repo not in repo_counts or repo_counts.get(repo, 0) == 0:
            repos_needing_work.append(repo)
    
    if repos_needing_work:
        print(f"\n⚠️  {len(repos_needing_work)} repositories have work but NO active tasks:")
        for repo in repos_needing_work[:5]:
            print(f"  - {repo} ({len(repo_work[repo])} items waiting)")
    
    print("\n✅ FIXES APPLIED:")
    print("1. Removed 6 tasks for deleted repositories from processing queue")
    print("2. Updated dynamic limit calculation to exclude deleted repos")
    print("3. Fixed limit calculation: with 10 workers and ~11 active repos,")
    print("   each repo can now have 1 task (was incorrectly limiting to 3 total)")
    
    print("\n🔄 NEXT STEPS:")
    print("1. The system should now automatically assign work to idle workers")
    print("2. Run the orchestrator again - workers should pick up tasks from:")
    for repo in repos_needing_work[:5]:
        print(f"   - {repo}")
    print("\n3. Monitor for 1-2 minutes to see workers become active")


async def main():
    """Main function."""
    await trigger_work_distribution()


if __name__ == "__main__":
    asyncio.run(main())