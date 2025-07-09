#!/usr/bin/env python3
"""
Deep clean all references to deleted repositories from all state files
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

def clean_json_file(file_path, cleaner_func):
    """Generic function to clean a JSON file"""
    if not file_path.exists():
        logger.info(f"{file_path.name} not found")
        return False
    
    logger.info(f"Cleaning {file_path.name}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"  Error reading {file_path.name}: {e}")
        return False
    
    # Backup original
    backup_file = file_path.with_suffix('.json.backup_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    with open(backup_file, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"  Created backup: {backup_file.name}")
    
    # Clean the data
    cleaned_data = cleaner_func(data)
    
    # Save cleaned data
    with open(file_path, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    return True

def clean_task_hashes(data):
    """Clean task_hashes.json data"""
    cleaned_data = {}
    removed_count = 0
    
    for key, value in data.items():
        if not contains_deleted_repo(key) and not contains_deleted_repo(value):
            cleaned_data[key] = value
        else:
            removed_count += 1
            logger.info(f"  Removing hash: {key} -> {value}")
    
    logger.info(f"  Removed {removed_count} entries from task_hashes")
    return cleaned_data

def clean_task_state(data):
    """Clean task_state.json data"""
    # Clean completed_work
    if 'completed_work' in data and isinstance(data['completed_work'], list):
        original_count = len(data['completed_work'])
        data['completed_work'] = [
            work for work in data['completed_work']
            if not contains_deleted_repo(work.get('repository', ''))
        ]
        removed = original_count - len(data['completed_work'])
        if removed > 0:
            logger.info(f"  Removed {removed} items from completed_work")
    
    # Clean task_history
    if 'task_history' in data and isinstance(data['task_history'], dict):
        cleaned_history = {}
        removed_count = 0
        
        for repo, history in data['task_history'].items():
            if not contains_deleted_repo(repo):
                cleaned_history[repo] = history
            else:
                removed_count += 1
                logger.info(f"  Removing task history for: {repo}")
        
        data['task_history'] = cleaned_history
        if removed_count > 0:
            logger.info(f"  Removed {removed_count} repositories from task_history")
    
    # Clean any lists of tasks
    for section in ['active_tasks', 'pending_tasks', 'failed_tasks', 'tasks']:
        if section in data and isinstance(data[section], list):
            original_count = len(data[section])
            data[section] = [
                task for task in data[section]
                if not contains_deleted_repo(task.get('repository', ''))
                   and not contains_deleted_repo(task.get('title', ''))
                   and not contains_deleted_repo(task.get('id', ''))
            ]
            removed = original_count - len(data[section])
            if removed > 0:
                logger.info(f"  Removed {removed} items from {section}")
    
    # Clean any dict of tasks
    if 'tasks' in data and isinstance(data['tasks'], dict):
        cleaned_tasks = {}
        removed_count = 0
        
        for task_id, task in data['tasks'].items():
            if not contains_deleted_repo(task_id) and \
               not contains_deleted_repo(task.get('repository', '')) and \
               not contains_deleted_repo(task.get('title', '')):
                cleaned_tasks[task_id] = task
            else:
                removed_count += 1
                logger.info(f"  Removing task: {task_id}")
        
        data['tasks'] = cleaned_tasks
        if removed_count > 0:
            logger.info(f"  Removed {removed_count} tasks")
    
    return data

def clean_completed_tasks(data):
    """Clean completed_tasks.json data"""
    if isinstance(data, list):
        original_count = len(data)
        cleaned_data = [
            task for task in data
            if not contains_deleted_repo(task.get('repository', ''))
               and not contains_deleted_repo(task.get('title', ''))
               and not contains_deleted_repo(task.get('id', ''))
        ]
        removed = original_count - len(cleaned_data)
        if removed > 0:
            logger.info(f"  Removed {removed} completed tasks")
        return cleaned_data
    elif isinstance(data, dict):
        # If it's a dict with a 'tasks' key
        if 'tasks' in data:
            data['tasks'] = clean_completed_tasks(data['tasks'])
        else:
            # Clean dict entries
            cleaned_data = {}
            removed_count = 0
            for key, value in data.items():
                if not contains_deleted_repo(key) and \
                   (not isinstance(value, dict) or \
                    (not contains_deleted_repo(value.get('repository', '')) and \
                     not contains_deleted_repo(value.get('title', '')))):
                    cleaned_data[key] = value
                else:
                    removed_count += 1
                    logger.info(f"  Removing: {key}")
            if removed_count > 0:
                logger.info(f"  Removed {removed_count} entries")
            return cleaned_data
    return data

def clean_task_history(data):
    """Clean task_history.json data"""
    if isinstance(data, dict):
        cleaned_data = {}
        removed_count = 0
        
        for key, value in data.items():
            if not contains_deleted_repo(key):
                # Also clean the value if it's a list of tasks
                if isinstance(value, list):
                    cleaned_value = [
                        task for task in value
                        if not contains_deleted_repo(task.get('repository', ''))
                           and not contains_deleted_repo(task.get('title', ''))
                    ]
                    if len(cleaned_value) < len(value):
                        logger.info(f"  Cleaned {len(value) - len(cleaned_value)} tasks from {key}")
                    cleaned_data[key] = cleaned_value
                else:
                    cleaned_data[key] = value
            else:
                removed_count += 1
                logger.info(f"  Removing history for: {key}")
        
        if removed_count > 0:
            logger.info(f"  Removed {removed_count} repository histories")
        return cleaned_data
    return data

def main():
    """Main cleanup function"""
    logger.info("Deep cleaning all state files to remove deleted repository references")
    logger.info(f"Deleted repositories: {', '.join(DELETED_REPOS)}")
    logger.info("")
    
    state_dir = Path('/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/state')
    
    # Clean each state file
    files_to_clean = [
        ('task_hashes.json', clean_task_hashes),
        ('task_state.json', clean_task_state),
        ('completed_tasks.json', clean_completed_tasks),
        ('task_history.json', clean_task_history),
    ]
    
    cleaned_count = 0
    for filename, cleaner_func in files_to_clean:
        file_path = state_dir / filename
        if clean_json_file(file_path, cleaner_func):
            cleaned_count += 1
    
    # Also check active_tasks.json
    active_tasks_path = state_dir / 'active_tasks.json'
    if active_tasks_path.exists():
        logger.info("Checking active_tasks.json")
        with open(active_tasks_path, 'r') as f:
            data = json.load(f)
        
        has_deleted_refs = False
        for task_id, task in data.items():
            if contains_deleted_repo(task.get('repository', '')) or \
               contains_deleted_repo(task.get('title', '')) or \
               contains_deleted_repo(task_id):
                logger.warning(f"  Found reference in active_tasks: {task_id} -> {task.get('repository', '')}")
                has_deleted_refs = True
        
        if has_deleted_refs:
            logger.warning("  Active tasks still contain references to deleted repos!")
            logger.warning("  These need to be handled by the task management system")
    
    logger.info(f"\nDeep cleanup complete! Cleaned {cleaned_count} files.")
    logger.info("The system should now be free of deleted repository references.")

if __name__ == "__main__":
    main()