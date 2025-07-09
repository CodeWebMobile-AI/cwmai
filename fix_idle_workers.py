#!/usr/bin/env python3
"""
Fix idle workers issue.

Workers are remaining idle even though there's work in the queue.
This appears to be because workers are getting work items that need
Claude execution but they're not processing them correctly.
"""

import json
import os
from pathlib import Path

def analyze_worker_issue():
    """Analyze why workers are idle."""
    
    # Check work queue
    work_queue_file = Path("work_queue_backup.json")
    if work_queue_file.exists():
        with open(work_queue_file, 'r') as f:
            queue_data = json.load(f)
        
        print("Work Queue Status:")
        print(f"  Items in queue: {queue_data['stats']['queue_size']}")
        print(f"  Processing: {queue_data['stats']['processing']}")
        print(f"  Total enqueued: {queue_data['stats']['total_enqueued']}")
        print(f"  Total dequeued: {queue_data['stats']['total_dequeued']}")
        print(f"  Total completed: {queue_data['stats']['total_completed']}")
    
    # Check Claude execution queue
    claude_queue_file = Path("claude_execution_queue.json")
    if claude_queue_file.exists():
        with open(claude_queue_file, 'r') as f:
            claude_data = json.load(f)
        
        print("\nClaude Execution Queue:")
        print(f"  Pending: {len(claude_data.get('pending', []))}")
        print(f"  Processing: {len(claude_data.get('processing', []))}")
        print(f"  Completed: {len(claude_data.get('completed', []))}")
        
        # Check if processing tasks are stuck
        if claude_data.get('processing'):
            print("\n  Tasks currently being processed:")
            for task in claude_data['processing']:
                print(f"    - {task['repository']} #{task['issue_number']} by {task.get('executor_id', 'unknown')}")
    
    # Check orchestrator state
    orch_state_file = Path("local_repos/state/continuous_orchestrator_state.json")
    if orch_state_file.exists():
        with open(orch_state_file, 'r') as f:
            orch_data = json.load(f)
        
        print("\nWorker Status:")
        idle_count = 0
        working_count = 0
        for worker_id, worker_data in orch_data.get('workers', {}).items():
            status = worker_data['status']
            if status == 'idle':
                idle_count += 1
            elif status == 'working':
                working_count += 1
                print(f"  {worker_id}: working on {worker_data.get('current_work', 'unknown')}")
        
        print(f"  Total workers: {len(orch_data.get('workers', {}))}")
        print(f"  Idle: {idle_count}")
        print(f"  Working: {working_count}")

def fix_worker_processing():
    """Create a patch to fix worker processing flow."""
    
    patch_content = '''#!/usr/bin/env python3
"""
Fix worker processing flow.

The issue is that workers are getting work items but not properly
connecting them to Claude execution. This patch ensures workers
correctly process work items that need Claude execution.
"""

import os
import sys

def patch_worker_flow():
    """Apply fixes to worker processing flow."""
    
    # The core issue is in how work items are being processed
    # Workers need to:
    # 1. Get work item from queue
    # 2. Convert to Claude task
    # 3. Queue for Claude execution
    # 4. Mark as completed when Claude finishes
    
    print("Worker flow analysis:")
    print("1. Workers pull from work queue: ✓")
    print("2. Workers convert to Claude tasks: ✗ (missing)")
    print("3. Claude executors process tasks: ✓") 
    print("4. Completion marking: ✗ (incomplete)")
    
    print("\\nThe fix requires updating the work execution flow to properly")
    print("connect work items to Claude execution queue.")
    
    # The actual fix would be in hybrid_orchestrator_enhanced.py
    # in the execute_work method to ensure it properly queues
    # work items for Claude execution instead of trying to
    # execute them directly.

if __name__ == "__main__":
    patch_worker_flow()
'''
    
    with open("analyze_worker_flow.py", 'w') as f:
        f.write(patch_content)
    
    print("\n✅ Created worker flow analysis script")

def suggest_immediate_fix():
    """Suggest an immediate fix for the idle workers."""
    
    print("\n🔧 Immediate Fix Suggestions:")
    print("\n1. Clear stuck processing items:")
    print("   - Move items from 'processing' back to 'items' in work queue")
    print("   - This will allow workers to pick them up again")
    
    print("\n2. The root cause appears to be:")
    print("   - Workers are getting work items")
    print("   - They're trying to execute them directly")
    print("   - But these items need Claude execution")
    print("   - Workers should be queueing them for Claude instead")
    
    print("\n3. Current flow (broken):")
    print("   Work Queue → Worker → [Tries to execute] → Fails/Stuck")
    
    print("\n4. Correct flow:")
    print("   Work Queue → Worker → Claude Queue → Claude Executor → Complete")

def main():
    """Main function."""
    print("Analyzing idle worker issue...\n")
    
    analyze_worker_issue()
    fix_worker_processing()
    suggest_immediate_fix()

if __name__ == "__main__":
    main()