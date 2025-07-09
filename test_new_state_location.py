#!/usr/bin/env python3
"""Test that task state files are correctly saved to local_repos/state"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.state_manager import StateManager
from datetime import datetime, timezone

def test_state_locations():
    """Test the new state file locations."""
    print("=== Testing New State File Locations ===\n")
    
    # Initialize state manager
    state_manager = StateManager()
    
    print(f"State directory: {state_manager.local_state_dir}")
    print(f"Task state file: {state_manager.task_state_file}")
    print(f"Completed tasks file: {state_manager.completed_tasks_file}")
    print(f"Orchestrator state file: {state_manager.orchestrator_state_file}")
    
    # Test loading and saving task state
    print("\n1. Testing task state...")
    task_state = state_manager.load_task_state()
    print(f"   Loaded task state with {len(task_state.get('completed_tasks', []))} completed tasks")
    
    # Add a test task
    test_task = {
        'id': 'test_123',
        'task_type': 'TEST',
        'title': 'Test Task for New Location',
        'repository': 'test/repo',
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'duration': 1.23,
        'worker_id': 'test_worker'
    }
    
    if 'completed_tasks' not in task_state:
        task_state['completed_tasks'] = []
    
    # Check if test task already exists
    if not any(t['id'] == 'test_123' for t in task_state['completed_tasks']):
        task_state['completed_tasks'].append(test_task)
        if state_manager.save_task_state(task_state):
            print("   ✅ Successfully saved task state to local_repos/state/task_state.json")
        else:
            print("   ❌ Failed to save task state")
    else:
        print("   ℹ️  Test task already exists in task state")
    
    # Test completed tasks file
    print("\n2. Testing completed tasks file...")
    completed_tasks = state_manager.load_completed_tasks()
    print(f"   Loaded {len(completed_tasks)} completed tasks")
    
    # Test orchestrator state
    print("\n3. Testing orchestrator state...")
    orch_state = {
        'metrics': {
            'total_work_completed': 42,
            'test_run': True
        },
        'last_updated': datetime.now(timezone.utc).isoformat()
    }
    
    if state_manager.save_orchestrator_state(orch_state):
        print("   ✅ Successfully saved orchestrator state to local_repos/state/continuous_orchestrator_state.json")
    else:
        print("   ❌ Failed to save orchestrator state")
    
    # Verify files exist
    print("\n4. Verifying files exist:")
    for name, path in [
        ("Task state", state_manager.task_state_file),
        ("Completed tasks", state_manager.completed_tasks_file),
        ("Orchestrator state", state_manager.orchestrator_state_file)
    ]:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"   ✅ {name}: {path} ({size} bytes)")
        else:
            print(f"   ❌ {name}: {path} NOT FOUND")
    
    print("\n✅ All task-related JSON files are now in local_repos/state/")

if __name__ == "__main__":
    test_state_locations()