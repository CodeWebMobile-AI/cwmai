#!/usr/bin/env python3
"""
Clean up deleted repositories from claude_execution_queue.json
"""

import json
import os
from typing import Set, Dict, Any, List
import shutil
from datetime import datetime

# Repositories that have been deleted
DELETED_REPOSITORIES = {
    'subtrack-ai',
    'test-repo',
    'sub-wise',
    'CodeWebMobile-AI/subtrack-ai',
    'CodeWebMobile-AI/test-repo',
    'CodeWebMobile-AI/sub-wise'
}

def should_remove_item(item: Dict[str, Any]) -> bool:
    """Check if an item should be removed based on repository"""
    # Check repository field
    if 'repository' in item:
        for deleted in DELETED_REPOSITORIES:
            if deleted in str(item['repository']):
                return True
    
    # Check issue_path field
    if 'issue_path' in item:
        for deleted in DELETED_REPOSITORIES:
            if deleted in str(item['issue_path']):
                return True
    
    # Check prompt field (may contain repository references)
    if 'prompt' in item:
        for deleted in DELETED_REPOSITORIES:
            if deleted in str(item['prompt']):
                return True
    
    return False

def clean_claude_execution_queue():
    """Clean deleted repositories from claude_execution_queue.json"""
    file_path = "claude_execution_queue.json"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Create backup
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    removed_count = 0
    
    # Clean pending
    if 'pending' in data and isinstance(data['pending'], list):
        original_count = len(data['pending'])
        data['pending'] = [item for item in data['pending'] if not should_remove_item(item)]
        removed_from_pending = original_count - len(data['pending'])
        removed_count += removed_from_pending
        print(f"Removed {removed_from_pending} items from pending")
    
    # Clean queue (legacy name)
    if 'queue' in data and isinstance(data['queue'], list):
        original_count = len(data['queue'])
        data['queue'] = [item for item in data['queue'] if not should_remove_item(item)]
        removed_from_queue = original_count - len(data['queue'])
        removed_count += removed_from_queue
        print(f"Removed {removed_from_queue} items from queue")
    
    # Clean processing (might be a dict or list)
    if 'processing' in data:
        if isinstance(data['processing'], list):
            original_count = len(data['processing'])
            data['processing'] = [item for item in data['processing'] if not should_remove_item(item)]
            removed_from_processing = original_count - len(data['processing'])
            removed_count += removed_from_processing
            print(f"Removed {removed_from_processing} items from processing")
        elif isinstance(data['processing'], dict):
            original_count = len(data['processing'])
            keys_to_remove = []
            for key, item in data['processing'].items():
                if should_remove_item(item):
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del data['processing'][key]
            removed_from_processing = len(keys_to_remove)
            removed_count += removed_from_processing
            if removed_from_processing > 0:
                print(f"Removed {removed_from_processing} items from processing")
    
    # Clean completed
    if 'completed' in data and isinstance(data['completed'], list):
        original_count = len(data['completed'])
        # Count items to be removed
        items_to_remove = sum(1 for item in data['completed'] if should_remove_item(item))
        if items_to_remove > 0:
            print(f"Found {items_to_remove} items with deleted repos in completed section")
            data['completed'] = [item for item in data['completed'] if not should_remove_item(item)]
            removed_from_completed = original_count - len(data['completed'])
            removed_count += removed_from_completed
            print(f"Removed {removed_from_completed} items from completed")
        else:
            print(f"No items to remove from completed section")
    
    # Clean failed
    if 'failed' in data and isinstance(data['failed'], list):
        original_count = len(data['failed'])
        data['failed'] = [item for item in data['failed'] if not should_remove_item(item)]
        removed_from_failed = original_count - len(data['failed'])
        removed_count += removed_from_failed
        print(f"Removed {removed_from_failed} items from failed")
    
    # Update stats
    if 'stats' in data:
        if 'queue' in data:
            data['stats']['queue_size'] = len(data['queue'])
        if 'processing' in data:
            data['stats']['processing_size'] = len(data['processing'])
        if 'completed' in data:
            data['stats']['completed_size'] = len(data['completed'])
        if 'failed' in data:
            data['stats']['failed_size'] = len(data['failed'])
    
    # Save cleaned data
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nTotal items removed: {removed_count}")
    print(f"Cleaned file saved to: {file_path}")
    print(f"Backup saved to: {backup_path}")

def main():
    """Main cleanup function"""
    print("Cleaning claude_execution_queue.json...")
    print(f"Deleted repositories: {', '.join(DELETED_REPOSITORIES)}")
    print()
    
    clean_claude_execution_queue()
    
    print("\nCleanup complete!")
    print("\nNote: The deleted repositories have been added to scripts/deleted_repos_exclusion.json")
    print("and will be permanently excluded from all system operations.")

if __name__ == "__main__":
    main()