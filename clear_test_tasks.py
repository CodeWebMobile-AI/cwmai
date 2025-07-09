#!/usr/bin/env python3
"""
Clear all test tasks from the Claude execution queue.
"""

import json
from pathlib import Path

def clear_test_tasks():
    """Remove all test tasks from the queue."""
    queue_file = Path("claude_execution_queue.json")
    
    if not queue_file.exists():
        print("No Claude execution queue file found.")
        return
    
    # Load the queue
    with open(queue_file, 'r') as f:
        queue_data = json.load(f)
    
    print(f"Current queue state:")
    print(f"  Pending: {len(queue_data.get('pending', []))}")
    print(f"  Processing: {len(queue_data.get('processing', []))}")
    print(f"  Completed: {len(queue_data.get('completed', []))}")
    
    # Filter out test tasks from completed
    original_completed = queue_data.get('completed', [])
    filtered_completed = []
    
    for task in original_completed:
        # Keep only non-test tasks
        if "test" not in task.get('prompt', '').lower() and "dummy/path" not in task.get('issue_path', ''):
            filtered_completed.append(task)
        else:
            print(f"  Removing test task: {task.get('id')}")
    
    # Update the queue
    queue_data['completed'] = filtered_completed
    
    # Clear any stuck processing tasks (move back to pending)
    processing = queue_data.get('processing', [])
    pending = queue_data.get('pending', [])
    
    if processing:
        print(f"\nMoving {len(processing)} processing tasks back to pending...")
        for task in processing:
            task['status'] = 'pending'
            task['started_at'] = None
            task['executor_id'] = None
            pending.append(task)
        queue_data['processing'] = []
    
    queue_data['pending'] = pending
    
    # Save the cleaned queue
    with open(queue_file, 'w') as f:
        json.dump(queue_data, f, indent=2)
    
    print(f"\n✅ Queue cleaned:")
    print(f"  Pending: {len(queue_data.get('pending', []))}")
    print(f"  Processing: {len(queue_data.get('processing', []))}")
    print(f"  Completed: {len(queue_data.get('completed', []))}")
    print(f"  Removed {len(original_completed) - len(filtered_completed)} test tasks")

if __name__ == "__main__":
    clear_test_tasks()