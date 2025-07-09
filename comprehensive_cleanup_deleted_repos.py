#!/usr/bin/env python3
"""
Comprehensive cleanup of deleted repositories from all system files.
"""

import json
import os
import shutil
from datetime import datetime
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

def contains_deleted_repo(text: str) -> bool:
    """Check if text contains any deleted repository reference"""
    if isinstance(text, str):
        for repo in DELETED_REPOSITORIES:
            if repo in text:
                return True
    return False

def clean_dict_recursive(data: Dict, path: str = "") -> int:
    """Recursively clean dictionary of deleted repository references"""
    removed = 0
    if not isinstance(data, dict):
        return removed
    
    keys_to_remove = []
    
    for key, value in data.items():
        current_path = f"{path}.{key}" if path else key
        
        # Check if key contains deleted repo
        if contains_deleted_repo(str(key)):
            keys_to_remove.append(key)
            print(f"  Removing key: {current_path}")
            removed += 1
            continue
        
        # Check value
        if isinstance(value, dict):
            # Recursively clean nested dictionaries
            removed += clean_dict_recursive(value, current_path)
        elif isinstance(value, list):
            # Clean lists
            original_len = len(value)
            value[:] = [item for item in value if not (isinstance(item, str) and contains_deleted_repo(item))]
            if len(value) < original_len:
                print(f"  Cleaned list at {current_path}: removed {original_len - len(value)} items")
                removed += original_len - len(value)
        elif isinstance(value, str) and contains_deleted_repo(value):
            keys_to_remove.append(key)
            print(f"  Removing {current_path} (contains deleted repo)")
            removed += 1
    
    # Remove marked keys
    for key in keys_to_remove:
        del data[key]
    
    return removed

def clean_system_state_comprehensive(file_path: str) -> int:
    """Comprehensively clean system_state.json"""
    print(f"\nCleaning {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return 0
    
    # Backup original
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(file_path, backup_path)
    print(f"  Created backup: {backup_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error reading file: {e}")
        return 0
    
    removed = clean_dict_recursive(data)
    
    # Save cleaned data
    if removed > 0:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Total removed: {removed} entries")
    else:
        print("  No deleted repositories found")
    
    return removed

def clean_json_file(file_path: str, clean_func=None) -> int:
    """Generic JSON file cleaner"""
    print(f"\nCleaning {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return 0
    
    # Backup original
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(file_path, backup_path)
    print(f"  Created backup: {backup_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error reading file: {e}")
        return 0
    
    if clean_func:
        removed = clean_func(data)
    else:
        removed = clean_dict_recursive(data)
    
    # Save cleaned data
    if removed > 0:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Total removed: {removed} entries")
    else:
        print("  No deleted repositories found")
    
    return removed

def clean_work_queue_data(data: Dict) -> int:
    """Clean work queue specific structure"""
    removed = 0
    
    if 'items' in data and isinstance(data['items'], list):
        original_count = len(data['items'])
        data['items'] = [
            item for item in data['items']
            if not (isinstance(item, dict) and 'repository' in item and 
                   contains_deleted_repo(str(item['repository'])))
        ]
        removed = original_count - len(data['items'])
        if removed > 0:
            print(f"  Removed {removed} work items")
    
    # Update stats
    if 'stats' in data and removed > 0:
        data['stats']['queue_size'] = len(data.get('items', []))
    
    return removed

def clean_active_tasks_data(data: Dict) -> int:
    """Clean active tasks structure"""
    removed = 0
    keys_to_remove = []
    
    for task_id, task in data.items():
        if isinstance(task, dict) and 'repository' in task:
            if contains_deleted_repo(str(task['repository'])):
                keys_to_remove.append(task_id)
                print(f"  Removing task: {task_id} (repo: {task['repository']})")
    
    for key in keys_to_remove:
        del data[key]
        removed += 1
    
    return removed

def main():
    """Main comprehensive cleanup function"""
    print("=" * 60)
    print("Comprehensive Cleanup of Deleted Repositories")
    print("=" * 60)
    print(f"Deleted repositories: {', '.join(sorted(DELETED_REPOSITORIES))}")
    
    total_removed = 0
    
    # Clean system_state.json
    total_removed += clean_system_state_comprehensive("system_state.json")
    
    # Clean work_queue_backup.json
    total_removed += clean_json_file("work_queue_backup.json", clean_work_queue_data)
    
    # Clean local_repos/state files
    local_state_files = {
        "local_repos/state/active_tasks.json": clean_active_tasks_data,
        "local_repos/state/continuous_orchestrator_state.json": clean_dict_recursive,
    }
    
    for file_path, clean_func in local_state_files.items():
        if os.path.exists(file_path):
            total_removed += clean_json_file(file_path, clean_func)
    
    # Clean other JSON files
    other_files = [
        "in_progress_tasks.json",
        "continuous_orchestrator_state.json",
    ]
    
    for file_path in other_files:
        if os.path.exists(file_path):
            total_removed += clean_json_file(file_path)
    
    print("\n" + "=" * 60)
    print(f"Total items removed across all files: {total_removed}")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Run clean_claude_execution_queue.py to clean Claude queue")
    print("2. Run verify_deleted_repos_blocked.py to verify cleanup")
    print("3. Monitor system to ensure no new tasks are created for deleted repos")
    
    return total_removed

if __name__ == "__main__":
    main()