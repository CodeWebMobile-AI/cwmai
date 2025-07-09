#!/usr/bin/env python3
"""
Clear stuck work items to allow workers to process new work.

The workers are stuck on work items that they're trying to execute
directly instead of queueing for Claude execution.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

def clear_stuck_work():
    """Clear stuck work items from processing state."""
    
    # 1. Clear work queue processing items
    work_queue_file = Path("work_queue_backup.json")
    if work_queue_file.exists():
        print("Clearing stuck work queue items...")
        with open(work_queue_file, 'r') as f:
            data = json.load(f)
        
        # Move processing items back to queue
        processing_items = data.get('processing', [])
        if processing_items:
            print(f"  Found {len(processing_items)} stuck items in processing")
            for item_wrapper in processing_items:
                if 'item' in item_wrapper:
                    # Reset the item
                    item = item_wrapper['item']
                    data['items'].append(item)
                    print(f"    Moved {item['id']} back to queue")
            
            # Clear processing
            data['processing'] = []
            data['stats']['processing'] = 0
            
            # Save
            with open(work_queue_file, 'w') as f:
                json.dump(data, f, indent=2)
            print("✅ Cleared work queue processing state")
    
    # 2. Reset orchestrator worker states
    orch_state_file = Path("local_repos/state/continuous_orchestrator_state.json")
    if orch_state_file.exists():
        print("\nResetting worker states...")
        with open(orch_state_file, 'r') as f:
            data = json.load(f)
        
        # Reset all workers to idle
        for worker_id, worker_data in data.get('workers', {}).items():
            if worker_data['status'] == 'working':
                print(f"  Resetting {worker_id} from working to idle")
                worker_data['status'] = 'idle'
                worker_data['current_work'] = None
        
        # Update timestamp
        data['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        # Save
        with open(orch_state_file, 'w') as f:
            json.dump(data, f, indent=2)
        print("✅ Reset worker states")
    
    print("\n✅ Stuck work cleared - workers should now pick up new work")
    print("\nNote: The root issue is that workers are trying to execute work")
    print("directly instead of queueing it for Claude execution. This needs")
    print("to be fixed in the execute_work method.")

if __name__ == "__main__":
    clear_stuck_work()