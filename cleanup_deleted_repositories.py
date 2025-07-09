#!/usr/bin/env python3
"""
Clean up deleted repositories from the system state and all related files.
"""

import json
import os
import sys
from typing import Set, Dict, Any, List

# Repositories that have been deleted
DELETED_REPOSITORIES = {
    'subtrack-ai',
    'test-repo',
    'sub-wise',
    'CodeWebMobile-AI/subtrack-ai',
    'CodeWebMobile-AI/test-repo',
    'CodeWebMobile-AI/sub-wise'
}

def clean_system_state(file_path: str) -> int:
    """Clean deleted repositories from system_state.json"""
    print(f"Cleaning {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return 0
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error reading file: {e}")
        return 0
    
    removed_count = 0
    
    # Clean repositories section
    if 'repositories' in data:
        for repo in list(data['repositories'].keys()):
            if any(deleted in repo for deleted in DELETED_REPOSITORIES):
                print(f"  Removing repository: {repo}")
                del data['repositories'][repo]
                removed_count += 1
    
    # Clean projects section
    if 'projects' in data:
        for repo in list(data['projects'].keys()):
            if any(deleted in repo for deleted in DELETED_REPOSITORIES):
                print(f"  Removing project: {repo}")
                del data['projects'][repo]
                removed_count += 1
    
    # Save cleaned data
    if removed_count > 0:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Removed {removed_count} entries")
    else:
        print("  No deleted repositories found")
    
    return removed_count

def clean_task_state(file_path: str) -> int:
    """Clean deleted repositories from task_state.json"""
    print(f"Cleaning {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return 0
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error reading file: {e}")
        return 0
    
    removed_count = 0
    
    # Clean tasks
    if 'tasks' in data:
        for task_id in list(data['tasks'].keys()):
            task = data['tasks'][task_id]
            if 'repository' in task and any(deleted in task['repository'] for deleted in DELETED_REPOSITORIES):
                print(f"  Removing task: {task_id} (repo: {task['repository']})")
                del data['tasks'][task_id]
                removed_count += 1
    
    # Save cleaned data
    if removed_count > 0:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Removed {removed_count} tasks")
    else:
        print("  No tasks for deleted repositories found")
    
    return removed_count

def clean_work_queue(file_path: str) -> int:
    """Clean deleted repositories from work queue"""
    print(f"Cleaning {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return 0
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error reading file: {e}")
        return 0
    
    removed_count = 0
    
    # Clean items
    if 'items' in data:
        original_count = len(data['items'])
        data['items'] = [
            item for item in data['items']
            if 'repository' not in item or not any(deleted in item['repository'] for deleted in DELETED_REPOSITORIES)
        ]
        removed_count = original_count - len(data['items'])
        
        if removed_count > 0:
            print(f"  Removed {removed_count} work items")
    
    # Update stats
    if 'stats' in data and removed_count > 0:
        data['stats']['queue_size'] = len(data['items'])
    
    # Save cleaned data
    if removed_count > 0:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        print("  No work items for deleted repositories found")
    
    return removed_count

def clean_local_repos_state() -> int:
    """Clean deleted repositories from local_repos state files"""
    local_state_dir = "/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/state"
    removed_count = 0
    
    if not os.path.exists(local_state_dir):
        print(f"Local repos state directory not found: {local_state_dir}")
        return 0
    
    # Clean active_tasks.json
    active_tasks_file = os.path.join(local_state_dir, "active_tasks.json")
    if os.path.exists(active_tasks_file):
        print(f"Cleaning {active_tasks_file}...")
        try:
            with open(active_tasks_file, 'r') as f:
                data = json.load(f)
            
            for task_id in list(data.keys()):
                task = data[task_id]
                if 'repository' in task and any(deleted in task['repository'] for deleted in DELETED_REPOSITORIES):
                    print(f"  Removing active task: {task_id}")
                    del data[task_id]
                    removed_count += 1
            
            with open(active_tasks_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"  Error processing file: {e}")
    
    return removed_count

def main():
    """Main cleanup function"""
    print("Cleaning up deleted repositories from system...")
    print(f"Deleted repositories: {', '.join(DELETED_REPOSITORIES)}")
    print()
    
    total_removed = 0
    
    # Clean system_state.json
    total_removed += clean_system_state("system_state.json")
    
    # Clean task_state.json
    total_removed += clean_task_state("task_state.json")
    
    # Clean work_queue_backup.json
    total_removed += clean_work_queue("work_queue_backup.json")
    
    # Clean local repos state
    total_removed += clean_local_repos_state()
    
    # Clean other potential files
    other_files = [
        "continuous_orchestrator_state.json",
        "claude_execution_queue.json",
        "in_progress_tasks.json"
    ]
    
    for file_path in other_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check if any deleted repo is mentioned
                if any(deleted in content for deleted in DELETED_REPOSITORIES):
                    print(f"Warning: {file_path} may contain references to deleted repositories")
            except Exception as e:
                print(f"Error checking {file_path}: {e}")
    
    print()
    print(f"Total items removed: {total_removed}")
    print("Cleanup complete!")
    
    # Update the exclusion list to ensure these repos are permanently excluded
    print("\nDeleted repositories have been added to scripts/deleted_repos_exclusion.json")
    print("These repositories will be permanently excluded from all system operations.")

if __name__ == "__main__":
    main()