#!/usr/bin/env python3
"""
Comprehensive fix to remove all tasks for deleted repositories from the system.
This script will:
1. Remove tasks from claude_execution_queue.json
2. Remove tasks from work_queue_backup.json
3. Remove tasks from active_tasks.json
4. Remove tasks from task_hashes.json
5. Update continuous_orchestrator_state.json to remove completed work references
"""

import json
import os
from datetime import datetime

# Deleted repositories to remove
DELETED_REPOS = ['subtrack-ai', 'test-repo', 'sub-wise']

def clean_file(file_path, deleted_repos):
    """Clean a JSON file of deleted repository references."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return 0
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        original_count = 0
        removed_count = 0
        
        # Handle different file structures
        if file_path.endswith('claude_execution_queue.json'):
            # Clean pending tasks
            if 'pending' in data:
                original_count = len(data['pending'])
                data['pending'] = [task for task in data['pending'] 
                                 if not any(repo in str(task.get('repository', '')) for repo in deleted_repos)]
                removed_count = original_count - len(data['pending'])
            
            # Clean processing tasks
            if 'processing' in data:
                to_remove = []
                for task_id, task in data['processing'].items():
                    if any(repo in str(task.get('repository', '')) for repo in deleted_repos):
                        to_remove.append(task_id)
                        removed_count += 1
                
                for task_id in to_remove:
                    del data['processing'][task_id]
        
        elif file_path.endswith('work_queue_backup.json'):
            # Clean work queue items
            if 'items' in data:
                original_count = len(data['items'])
                data['items'] = [item for item in data['items'] 
                               if not any(repo in str(item.get('repository', '')) for repo in deleted_repos)]
                removed_count = original_count - len(data['items'])
        
        elif file_path.endswith('active_tasks.json'):
            # Remove tasks for deleted repos
            to_remove = []
            for task_id, task in data.items():
                if any(repo in str(task.get('repository', '')) for repo in deleted_repos):
                    to_remove.append(task_id)
                    removed_count += 1
            
            for task_id in to_remove:
                del data[task_id]
        
        elif file_path.endswith('task_hashes.json'):
            # Remove hashes for deleted repo tasks
            # First, we need to check which task IDs are for deleted repos
            active_tasks_path = file_path.replace('task_hashes.json', 'active_tasks.json')
            deleted_task_ids = set()
            
            if os.path.exists(active_tasks_path):
                with open(active_tasks_path, 'r') as f:
                    active_tasks = json.load(f)
                
                for task_id, task in active_tasks.items():
                    if any(repo in str(task.get('repository', '')) for repo in deleted_repos):
                        deleted_task_ids.add(task_id)
            
            # Remove hashes that point to deleted tasks
            to_remove = []
            for hash_val, task_id in data.items():
                if task_id in deleted_task_ids or any(repo in str(task_id) for repo in deleted_repos):
                    to_remove.append(hash_val)
                    removed_count += 1
            
            for hash_val in to_remove:
                del data[hash_val]
        
        elif file_path.endswith('continuous_orchestrator_state.json'):
            # Clean completed work references
            if 'completed_work' in data:
                original_count = len(data['completed_work'])
                data['completed_work'] = [work for work in data['completed_work'] 
                                        if not any(repo in str(work) for repo in deleted_repos)]
                removed_count = original_count - len(data['completed_work'])
        
        # Save cleaned data
        if removed_count > 0:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Cleaned {file_path}: Removed {removed_count} references to deleted repositories")
        else:
            print(f"✅ {file_path}: No deleted repository references found")
        
        return removed_count
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return 0

def main():
    print("🧹 Comprehensive cleanup of deleted repository tasks")
    print(f"Deleted repositories: {', '.join(DELETED_REPOS)}")
    print("="*60)
    
    total_removed = 0
    
    # Files to clean
    files_to_clean = [
        'claude_execution_queue.json',
        'work_queue_backup.json',
        'local_repos/state/active_tasks.json',
        'local_repos/state/task_hashes.json',
        'local_repos/state/continuous_orchestrator_state.json'
    ]
    
    for file_path in files_to_clean:
        removed = clean_file(file_path, DELETED_REPOS)
        total_removed += removed
    
    # Also check for any task persistence files
    task_files = ['in_progress_tasks.json', 'task_state.json', 'task_history.json']
    for file_path in task_files:
        if os.path.exists(file_path):
            removed = clean_file(file_path, DELETED_REPOS)
            total_removed += removed
    
    print("="*60)
    print(f"🎯 Total references removed: {total_removed}")
    
    # Additional check: Look for any stray references in the claude queue
    if os.path.exists('claude_execution_queue.json'):
        with open('claude_execution_queue.json', 'r') as f:
            queue_data = json.load(f)
        
        # Ensure the queue doesn't try to recreate these tasks
        if 'repository_limits' not in queue_data:
            queue_data['repository_limits'] = {}
        
        # Set limits to 0 for deleted repos to prevent new tasks
        for repo in DELETED_REPOS:
            queue_data['repository_limits'][repo] = 0
        
        with open('claude_execution_queue.json', 'w') as f:
            json.dump(queue_data, f, indent=2)
        
        print("\n✅ Set repository limits to 0 for deleted repositories to prevent new tasks")
    
    print("\n✨ Cleanup complete! The system should no longer process tasks for deleted repositories.")

if __name__ == "__main__":
    main()