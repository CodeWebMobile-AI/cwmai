#!/usr/bin/env python3
"""
Final cleanup for task_state.json to remove deleted repository references
"""

import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DELETED_REPOS = ['subtrack-ai', 'test-repo', 'sub-wise', 'subwise']

def contains_deleted_repo(text):
    """Check if text contains any deleted repository name"""
    if not text:
        return False
    text_lower = str(text).lower()
    return any(deleted in text_lower for deleted in DELETED_REPOS)

def main():
    """Clean task_state.json specifically"""
    task_state_file = Path('/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/state/task_state.json')
    
    if not task_state_file.exists():
        logger.error("task_state.json not found")
        return
    
    logger.info("Final cleanup of task_state.json")
    
    # Read the file
    with open(task_state_file, 'r') as f:
        data = json.load(f)
    
    # Backup original
    backup_file = task_state_file.with_suffix('.json.backup_final_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    with open(backup_file, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Created backup: {backup_file.name}")
    
    # Clean completed_tasks array
    if 'completed_tasks' in data and isinstance(data['completed_tasks'], list):
        original_count = len(data['completed_tasks'])
        cleaned_tasks = []
        removed_count = 0
        
        for task in data['completed_tasks']:
            repo = task.get('repository', '')
            task_id = task.get('id', '')
            title = task.get('title', '')
            
            if not contains_deleted_repo(repo) and not contains_deleted_repo(task_id) and not contains_deleted_repo(title):
                cleaned_tasks.append(task)
            else:
                removed_count += 1
                logger.info(f"  Removing completed task: {task_id} - {repo}")
        
        data['completed_tasks'] = cleaned_tasks
        logger.info(f"  Cleaned completed_tasks: {original_count} -> {len(cleaned_tasks)} (removed {removed_count})")
    
    # Also check for any other arrays or sections
    sections_cleaned = 0
    for key in list(data.keys()):
        if key == 'completed_tasks':
            continue
            
        value = data[key]
        
        # If it's a list, clean it
        if isinstance(value, list):
            original_count = len(value)
            cleaned_list = []
            
            for item in value:
                if isinstance(item, dict):
                    repo = item.get('repository', '')
                    if not contains_deleted_repo(repo):
                        cleaned_list.append(item)
                    else:
                        logger.info(f"  Removing from {key}: {item.get('id', 'unknown')} - {repo}")
                else:
                    cleaned_list.append(item)
            
            if len(cleaned_list) < original_count:
                data[key] = cleaned_list
                sections_cleaned += 1
                logger.info(f"  Cleaned {key}: {original_count} -> {len(cleaned_list)}")
        
        # If it's a dict, check for repo references
        elif isinstance(value, dict):
            if contains_deleted_repo(str(value)):
                logger.info(f"  Found reference in {key}, but complex structure - skipping")
    
    # Save cleaned data
    with open(task_state_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"\nFinal cleanup complete! Cleaned completed_tasks and {sections_cleaned} other sections.")
    
    # Verify cleanup
    with open(task_state_file, 'r') as f:
        content = f.read()
    
    if any(deleted in content.lower() for deleted in DELETED_REPOS):
        logger.warning("WARNING: Some references may still exist in complex nested structures")
    else:
        logger.info("SUCCESS: No references to deleted repositories found!")

if __name__ == "__main__":
    main()