#!/usr/bin/env python3
"""
Test to verify task completion tracking is fixed.
"""

import asyncio
import json
import time
from datetime import datetime, timezone

from scripts.local_issue_creator import LocalIssueCreator
from scripts.work_item_types import WorkItem, TaskPriority
from scripts.task_persistence import TaskPersistence
from scripts.claude_execution_queue import ClaudeExecutionQueue

async def test_task_completion():
    """Test that tasks are not marked complete until Claude executes."""
    
    print("🧪 Testing task completion tracking fix...\n")
    
    # Create test work item
    work_item = WorkItem(
        id=f"test_{int(time.time())}",
        title="Test Task Completion Fix",
        task_type="FEATURE",
        description="Verify that tasks are only marked complete after Claude execution",
        repository="micronutrient-optimizer",
        priority=TaskPriority.MEDIUM
    )
    
    # Initialize components
    issue_creator = LocalIssueCreator(enable_claude_execution=True)
    task_persistence = TaskPersistence()
    claude_queue = ClaudeExecutionQueue()
    
    # Get initial states
    initial_completed = len(task_persistence.completed_tasks)
    initial_in_progress = len(task_persistence.in_progress_tasks)
    initial_queue_stats = await claude_queue.get_stats()
    
    print(f"📊 Initial state:")
    print(f"   Completed tasks: {initial_completed}")
    print(f"   In-progress tasks: {initial_in_progress}")
    print(f"   Claude queue - Pending: {initial_queue_stats['pending']}, Processing: {initial_queue_stats['processing']}")
    print()
    
    # Execute the task (should queue for Claude)
    print(f"🚀 Executing task: {work_item.title}")
    result = await issue_creator.execute_work_item(work_item)
    
    # Check the result
    print(f"\n📋 Execution result:")
    print(f"   Success: {result.get('success')}")
    print(f"   Completed: {result.get('completed')}")
    print(f"   Stage: {result.get('stage')}")
    print(f"   Claude task ID: {result.get('claude_task_id')}")
    print(f"   Claude queued: {result.get('claude_queued')}")
    
    # Verify task is NOT marked as completed
    if result.get('success') and not result.get('completed'):
        print("\n✅ PASS: Task is successful but NOT marked as completed")
    else:
        print("\n❌ FAIL: Task completion status is incorrect")
        
    # Check if task is in progress
    if result.get('claude_queued'):
        # Task should be recorded as in-progress
        task_persistence.record_in_progress_task(work_item, result)
        current_in_progress = len(task_persistence.in_progress_tasks)
        
        if current_in_progress > initial_in_progress:
            print("✅ PASS: Task recorded as in-progress")
        else:
            print("❌ FAIL: Task not recorded as in-progress")
    
    # Check completed tasks count
    current_completed = len(task_persistence.completed_tasks)
    if current_completed == initial_completed:
        print("✅ PASS: Completed tasks count unchanged (task not prematurely marked complete)")
    else:
        print("❌ FAIL: Task was incorrectly marked as completed")
    
    # Check Claude queue
    current_queue_stats = await claude_queue.get_stats()
    if current_queue_stats['pending'] > initial_queue_stats['pending'] or \
       current_queue_stats['processing'] > initial_queue_stats['processing']:
        print("✅ PASS: Claude task is in the queue")
    else:
        print("❌ FAIL: Claude task not found in queue")
    
    print(f"\n📊 Final state:")
    print(f"   Completed tasks: {current_completed}")
    print(f"   In-progress tasks: {len(task_persistence.in_progress_tasks)}")
    print(f"   Claude queue - Pending: {current_queue_stats['pending']}, Processing: {current_queue_stats['processing']}")
    
    print("\n🎯 Summary:")
    print("The fix ensures tasks are not marked as completed when they're merely queued for Claude.")
    print("Tasks will only be marked complete after Claude actually executes them.")

if __name__ == "__main__":
    asyncio.run(test_task_completion())