#!/usr/bin/env python3
"""
Clear stuck Claude tasks from the execution queue.
"""

import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path

async def clear_stuck_tasks():
    """Clear tasks stuck in processing state."""
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
    
    # Move stuck processing tasks back to pending
    processing = queue_data.get('processing', [])
    pending = queue_data.get('pending', [])
    
    stuck_count = 0
    stuck_tasks = []
    
    # Processing is a list in this case
    for task in processing[:]:  # Create a copy to iterate over
        if task.get('started_at'):
            # Check if task has been running for more than 30 minutes
            started = datetime.fromisoformat(task['started_at'].replace('Z', '+00:00'))
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            
            if elapsed > 1800:  # 30 minutes
                print(f"\nStuck task found:")
                print(f"  ID: {task['id']}")
                print(f"  Repository: {task['repository']}")
                print(f"  Issue: #{task['issue_number']}")
                print(f"  Running for: {elapsed/60:.1f} minutes")
                
                # Reset the task
                task['status'] = 'pending'
                task['started_at'] = None
                task['executor_id'] = None
                task['error'] = None
                
                # Move to pending queue
                pending.insert(0, task)
                stuck_tasks.append(task)
                stuck_count += 1
    
    # Remove stuck tasks from processing
    for task in stuck_tasks:
        processing.remove(task)
    
    if stuck_count > 0:
        # Update the queue
        queue_data['pending'] = pending
        queue_data['processing'] = processing
        
        # Save the updated queue
        with open(queue_file, 'w') as f:
            json.dump(queue_data, f, indent=2)
        
        print(f"\n✅ Moved {stuck_count} stuck tasks back to pending queue")
    else:
        print("\n✅ No stuck tasks found")
    
    print(f"\nUpdated queue state:")
    print(f"  Pending: {len(queue_data.get('pending', []))}")
    print(f"  Processing: {len(queue_data.get('processing', []))}")
    print(f"  Completed: {len(queue_data.get('completed', []))}")

if __name__ == "__main__":
    asyncio.run(clear_stuck_tasks())