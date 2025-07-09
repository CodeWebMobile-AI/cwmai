#!/usr/bin/env python3
"""
Clear duplicate tracking to allow work discovery to proceed.

The system is stuck because all work is being marked as duplicates.
This script clears the duplicate tracking state to allow fresh discovery.
"""

import json
import os
from pathlib import Path

def clear_duplicate_tracking():
    """Clear all duplicate tracking state."""
    
    # 1. Clear task persistence duplicate tracking
    task_persistence_file = Path("work_finder_completed_tasks.json")
    if task_persistence_file.exists():
        print(f"Found task persistence file: {task_persistence_file}")
        with open(task_persistence_file, 'r') as f:
            data = json.load(f)
        
        print(f"  - Completed tasks: {len(data.get('completed_tasks', {}))}")
        print(f"  - In-progress tasks: {len(data.get('in_progress_tasks', {}))}")
        print(f"  - Title hashes: {len(data.get('title_hashes', []))}")
        print(f"  - Description hashes: {len(data.get('description_hashes', []))}")
        
        # Clear the hashes but keep completed tasks for reference
        data['title_hashes'] = []
        data['description_hashes'] = []
        data['skip_stats'] = {}
        data['title_cooldown'] = {}
        data['problematic_tasks'] = {}
        
        with open(task_persistence_file, 'w') as f:
            json.dump(data, f, indent=2)
        print("✅ Cleared duplicate tracking hashes")
    
    # 2. Clear orchestrator's discovered work IDs
    orchestrator_state_file = Path("local_repos/state/continuous_orchestrator_state.json")
    if orchestrator_state_file.exists():
        print(f"\nFound orchestrator state file: {orchestrator_state_file}")
        with open(orchestrator_state_file, 'r') as f:
            data = json.load(f)
        
        if 'completed_work' in data:
            print(f"  - Completed work items: {len(data['completed_work'])}")
            # Don't clear completed work, just ensure it's properly tracked
        
        with open(orchestrator_state_file, 'w') as f:
            json.dump(data, f, indent=2)
        print("✅ Orchestrator state preserved")
    
    # 3. Clear work queue processing history
    work_queue_file = Path("work_queue_backup.json")
    if work_queue_file.exists():
        print(f"\nFound work queue file: {work_queue_file}")
        with open(work_queue_file, 'r') as f:
            data = json.load(f)
        
        # Keep items but clear processing to allow retry
        if 'processing' in data:
            for item in data['processing']:
                if 'item' in item:
                    data['items'].append(item['item'])
            data['processing'] = []
        
        # Reset stats
        data['stats']['processing'] = 0
        
        with open(work_queue_file, 'w') as f:
            json.dump(data, f, indent=2)
        print("✅ Cleared work queue processing state")
    
    print("\n✅ Duplicate tracking cleared - work discovery should proceed normally")

if __name__ == "__main__":
    clear_duplicate_tracking()