#!/usr/bin/env python3
"""
Fix tasks stuck on deleted repositories and improve work distribution.
"""

import asyncio
import json
import os
from datetime import datetime, timezone


async def fix_deleted_repo_tasks():
    """Remove tasks for deleted repositories from Claude execution queue."""
    deleted_repos = [
        'subtrack-ai', 'test-repo', 'sub-wise', 'test/repo',
        'CodeWebMobile-AI/subtrack-ai', 'CodeWebMobile-AI/test-repo', 'CodeWebMobile-AI/sub-wise'
    ]
    
    # Fix Claude execution queue
    queue_file = 'claude_execution_queue.json'
    if os.path.exists(queue_file):
        with open(queue_file, 'r') as f:
            queue_data = json.load(f)
        
        # Clean up pending tasks for deleted repos
        original_pending = len(queue_data.get('pending', []))
        queue_data['pending'] = [
            t for t in queue_data.get('pending', [])
            if not any(deleted in t.get('repository', '') for deleted in deleted_repos)
        ]
        removed_pending = original_pending - len(queue_data.get('pending', []))

        # Clean up processing tasks for deleted repos
        original_processing = len(queue_data.get('processing', {}))
        cleaned_processing = {}
        removed_tasks = []

        for task_id, task in queue_data.get('processing', {}).items():
            repo = task.get('repository', '')
            if any(deleted in repo for deleted in deleted_repos):
                removed_tasks.append(f"{repo} (task {task_id})")
            else:
                cleaned_processing[task_id] = task

        queue_data['processing'] = cleaned_processing

        # Save cleaned queue
        with open(queue_file, 'w') as f:
            json.dump(queue_data, f, indent=2)

        print(f"✅ Cleaned Claude execution queue:")
        if removed_pending:
            print(f"   - Removed {removed_pending} pending tasks for deleted repos")
        print(f"   - Removed {original_processing - len(cleaned_processing)} processing tasks for deleted repos")
        if removed_tasks:
            print(f"   - Removed tasks: {', '.join(removed_tasks)}")
    
    # Fix active tasks
    active_tasks_file = 'local_repos/state/active_tasks.json'
    if os.path.exists(active_tasks_file):
        with open(active_tasks_file, 'r') as f:
            active_tasks = json.load(f)
        
        original_count = len(active_tasks)
        cleaned_tasks = {}
        
        for task_id, task in active_tasks.items():
            repo = task.get('repository', '')
            is_deleted = any(deleted in repo for deleted in deleted_repos)
            
            if not is_deleted:
                cleaned_tasks[task_id] = task
        
        # Save cleaned tasks
        with open(active_tasks_file, 'w') as f:
            json.dump(cleaned_tasks, f, indent=2)
        
        print(f"\n✅ Cleaned active tasks:")
        print(f"   - Removed {original_count - len(cleaned_tasks)} tasks for deleted repos")
    
    # Fix work queue backup
    work_queue_file = 'work_queue_backup.json'
    if os.path.exists(work_queue_file):
        with open(work_queue_file, 'r') as f:
            work_data = json.load(f)
        
        original_items = len(work_data.get('items', []))
        cleaned_items = []
        
        for item in work_data.get('items', []):
            repo = item.get('repository', '')
            is_deleted = any(deleted in repo for deleted in deleted_repos)
            
            if not is_deleted:
                cleaned_items.append(item)
        
        work_data['items'] = cleaned_items
        work_data['stats']['queue_size'] = len(cleaned_items)
        work_data['saved_at'] = datetime.now(timezone.utc).isoformat()
        
        # Save cleaned work queue
        with open(work_queue_file, 'w') as f:
            json.dump(work_data, f, indent=2)
        
        print(f"\n✅ Cleaned work queue backup:")
        print(f"   - Removed {original_items - len(cleaned_items)} items for deleted repos")
    
    print("\n🔧 Now fixing the dynamic limit calculation in claude_execution_queue.py...")


async def main():
    """Main function."""
    await fix_deleted_repo_tasks()


if __name__ == "__main__":
    asyncio.run(main())