#!/usr/bin/env python3
"""
Clear Old Work Queue Items

Removes old fixed task types (maintenance_dependencies, testing_unit_tests) 
from the work queue backup file.
"""

import json
import os
from datetime import datetime

def clear_old_work_queue():
    """Clear old task types from work queue backup."""
    queue_file = "local_repos/state/work_queue_backup.json"
    
    if not os.path.exists(queue_file):
        print(f"❌ Work queue file not found: {queue_file}")
        return
    
    # Load current queue
    try:
        with open(queue_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading work queue: {e}")
        return
    
    # Get current work items
    work_items = data.get('work_items', [])
    initial_count = len(work_items)
    
    # Filter out old task types
    old_task_types = ['maintenance_dependencies', 'testing_unit_tests']
    filtered_items = []
    removed_items = []
    
    for item in work_items:
        task_type = item.get('task_type', '')
        if task_type in old_task_types:
            removed_items.append(item)
        else:
            filtered_items.append(item)
    
    # Update data
    data['work_items'] = filtered_items
    
    # Save cleaned queue
    try:
        with open(queue_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Cleaned work queue")
        print(f"   - Initial items: {initial_count}")
        print(f"   - Removed items: {len(removed_items)}")
        print(f"   - Remaining items: {len(filtered_items)}")
        
        if removed_items:
            print("\n📋 Removed old task types:")
            for item in removed_items:
                print(f"   - {item.get('task_type')}: {item.get('title', 'Unknown')}")
    except Exception as e:
        print(f"❌ Error saving cleaned queue: {e}")

if __name__ == "__main__":
    clear_old_work_queue()