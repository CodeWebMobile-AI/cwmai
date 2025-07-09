#!/usr/bin/env python3
"""
Clean up remaining state files that contain references to deleted repositories
"""

import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_task_hashes():
    """Clean task_hashes.json"""
    deleted_repos = ['subtrack-ai', 'test-repo', 'sub-wise', 'subwise']
    task_hashes_file = Path('/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/state/task_hashes.json')
    
    if not task_hashes_file.exists():
        logger.info("task_hashes.json not found")
        return
    
    logger.info("Cleaning task_hashes.json")
    with open(task_hashes_file, 'r') as f:
        data = json.load(f)
    
    # Backup original
    backup_file = task_hashes_file.with_suffix('.json.backup_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    with open(backup_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Clean the data
    cleaned_data = {}
    removed_count = 0
    
    for key, value in data.items():
        # Check if key contains any deleted repo
        if not any(deleted in key.lower() for deleted in deleted_repos):
            cleaned_data[key] = value
        else:
            removed_count += 1
            logger.info(f"  Removing hash for: {key}")
    
    # Save cleaned data
    with open(task_hashes_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    logger.info(f"  Removed {removed_count} entries from task_hashes.json")

def cleanup_task_state():
    """Clean task_state.json"""
    deleted_repos = ['subtrack-ai', 'test-repo', 'sub-wise', 'subwise']
    task_state_file = Path('/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/state/task_state.json')
    
    if not task_state_file.exists():
        logger.info("task_state.json not found")
        return
    
    logger.info("Cleaning task_state.json")
    with open(task_state_file, 'r') as f:
        data = json.load(f)
    
    # Backup original
    backup_file = task_state_file.with_suffix('.json.backup_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    with open(backup_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Clean completed_work
    if 'completed_work' in data:
        original_count = len(data['completed_work'])
        cleaned_work = []
        
        for work in data['completed_work']:
            repo = work.get('repository', '')
            if not any(deleted in repo.lower() for deleted in deleted_repos):
                cleaned_work.append(work)
            else:
                logger.info(f"  Removing completed work for: {repo}")
        
        data['completed_work'] = cleaned_work
        logger.info(f"  Cleaned completed_work: {original_count} -> {len(cleaned_work)}")
    
    # Clean task_history
    if 'task_history' in data:
        cleaned_history = {}
        removed_count = 0
        
        for repo, history in data['task_history'].items():
            if not any(deleted in repo.lower() for deleted in deleted_repos):
                cleaned_history[repo] = history
            else:
                removed_count += 1
                logger.info(f"  Removing task history for: {repo}")
        
        data['task_history'] = cleaned_history
        logger.info(f"  Removed {removed_count} repositories from task_history")
    
    # Clean any other sections that might contain repository references
    for section in ['active_tasks', 'pending_tasks', 'failed_tasks']:
        if section in data and isinstance(data[section], list):
            original_count = len(data[section])
            cleaned_tasks = []
            
            for task in data[section]:
                repo = task.get('repository', '')
                if not any(deleted in repo.lower() for deleted in deleted_repos):
                    cleaned_tasks.append(task)
                else:
                    logger.info(f"  Removing {section} task for: {repo}")
            
            data[section] = cleaned_tasks
            if original_count != len(cleaned_tasks):
                logger.info(f"  Cleaned {section}: {original_count} -> {len(cleaned_tasks)}")
    
    # Save cleaned data
    with open(task_state_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info("  task_state.json cleaned successfully")

def main():
    """Main cleanup function"""
    logger.info("Cleaning remaining state files with deleted repository references")
    
    # Clean task_hashes.json
    cleanup_task_hashes()
    
    # Clean task_state.json
    cleanup_task_state()
    
    logger.info("\nCleanup complete! The system should now be free of deleted repository references.")

if __name__ == "__main__":
    main()