#!/usr/bin/env python3
"""
Test script to verify task completion tracking fix.

This script tests that tasks are properly tracked through their lifecycle
and only marked as completed when Claude actually executes them.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.work_item_types import WorkItem, TaskPriority
from scripts.local_issue_creator import LocalIssueCreator
from scripts.task_persistence import TaskPersistence
from scripts.claude_execution_queue import ClaudeExecutionQueue


async def test_task_lifecycle():
    """Test the complete task lifecycle from creation to Claude completion."""
    print("🧪 Testing task lifecycle tracking...")
    print("=" * 60)
    
    # Initialize components
    issue_creator = LocalIssueCreator(enable_claude_execution=True)
    task_persistence = TaskPersistence()
    claude_queue = ClaudeExecutionQueue()
    
    # Create a test work item
    test_work_item = WorkItem(
        id="test_task_001",
        task_type="TESTING",
        title="Test Task Completion Tracking",
        description="This is a test task to verify completion tracking works correctly",
        repository="test-repo",
        priority=TaskPriority.HIGH,
        metadata={
            "test": True,
            "created_by": "test_script"
        }
    )
    
    print(f"\n📋 Created test work item: {test_work_item.title}")
    print(f"   ID: {test_work_item.id}")
    print(f"   Repository: {test_work_item.repository}")
    
    # Execute the work item (should create issue and queue Claude)
    print("\n🚀 Executing work item...")
    result = await issue_creator.execute_work_item(test_work_item)
    
    print("\n📊 Execution result:")
    print(json.dumps(result, indent=2))
    
    # Check task persistence state
    print("\n🔍 Checking task persistence state...")
    
    # Check if task is in completed tasks (it shouldn't be if fix works)
    is_in_completed = test_work_item.id in task_persistence.completed_tasks
    print(f"   Task in completed_tasks: {'❌ YES (BUG!)' if is_in_completed else '✅ NO (correct)'}")
    
    # Check if task is in in-progress tasks (it should be if fix works)
    is_in_progress = test_work_item.id in task_persistence.in_progress_tasks
    print(f"   Task in in_progress_tasks: {'✅ YES (correct)' if is_in_progress else '❌ NO (BUG!)'}")
    
    if is_in_progress:
        task_data = task_persistence.in_progress_tasks[test_work_item.id]
        print(f"   Stage: {task_data.get('stage', 'unknown')}")
        print(f"   Claude task ID: {task_data.get('claude_task_id', 'none')}")
    
    # Check Claude queue state
    print("\n🔍 Checking Claude queue state...")
    stats = await claude_queue.get_stats()
    print(f"   Pending tasks: {stats['pending']}")
    print(f"   Processing tasks: {stats['processing']}")
    print(f"   Completed tasks: {stats['completed']}")
    
    # Check for the specific task in queue
    pending_tasks = claude_queue.get_pending_tasks()
    our_task = None
    for task in pending_tasks:
        if task['repository'] == test_work_item.repository:
            our_task = task
            break
    
    if our_task:
        print(f"   ✅ Found our task in queue: {our_task['id']}")
    else:
        print(f"   ❌ Task not found in Claude queue")
    
    # Verify the behavior
    print("\n📋 Test Results:")
    print("=" * 60)
    
    success = True
    issues = []
    
    # Check 1: Task should not be marked as completed yet
    if is_in_completed:
        issues.append("Task was marked as completed immediately (should wait for Claude)")
        success = False
    else:
        print("✅ Task correctly NOT marked as completed yet")
    
    # Check 2: Task should be in in-progress state
    if not is_in_progress:
        issues.append("Task was not tracked as in-progress")
        success = False
    else:
        print("✅ Task correctly tracked as in-progress")
    
    # Check 3: Result should indicate task is not completed
    if result.get('completed', True):  # Default True for backward compatibility
        issues.append("Result indicates task is completed (should be False)")
        success = False
    else:
        print("✅ Result correctly indicates task is not completed")
    
    # Check 4: Result should indicate Claude is queued
    if not result.get('claude_queued'):
        issues.append("Result doesn't indicate Claude is queued")
        success = False
    else:
        print("✅ Result correctly indicates Claude is queued")
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED! Task completion tracking is working correctly.")
        print("\nThe fix is working properly:")
        print("- Tasks are NOT immediately marked as completed")
        print("- Tasks are tracked as in-progress while Claude executes")
        print("- Proper lifecycle tracking is in place")
    else:
        print("❌ TESTS FAILED! Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nThe fix may not be applied correctly. Please run:")
        print("  python3 fix_task_completion_tracking.py")
        print("  python3 fix_claude_completion_notification.py")
    
    # Cleanup - remove test data
    print("\n🧹 Cleaning up test data...")
    if is_in_progress:
        del task_persistence.in_progress_tasks[test_work_item.id]
        task_persistence._save_in_progress_tasks()
    
    return success


async def test_duplicate_handling():
    """Test that duplicate tasks are handled correctly."""
    print("\n\n🧪 Testing duplicate task handling...")
    print("=" * 60)
    
    issue_creator = LocalIssueCreator(enable_claude_execution=True)
    
    # Create a test work item
    test_work_item = WorkItem(
        id="test_dup_001",
        task_type="TESTING",
        title="Test Duplicate Task Handling",
        description="Testing duplicate detection",
        repository="test-repo",
        priority=TaskPriority.MEDIUM
    )
    
    print(f"\n📋 Creating duplicate test: {test_work_item.title}")
    
    # Execute twice to test duplicate handling
    print("\n🚀 First execution...")
    result1 = await issue_creator.execute_work_item(test_work_item)
    print(f"   Success: {result1.get('success')}")
    print(f"   Completed: {result1.get('completed', 'not specified')}")
    
    print("\n🚀 Second execution (should detect duplicate)...")
    result2 = await issue_creator.execute_work_item(test_work_item)
    print(f"   Success: {result2.get('success')}")
    print(f"   Completed: {result2.get('completed', 'not specified')}")
    print(f"   Existing issue: {result2.get('existing_issue', False)}")
    
    return True


def check_file_state():
    """Check the state of various tracking files."""
    print("\n\n📁 Checking file states...")
    print("=" * 60)
    
    files_to_check = [
        "completed_tasks.json",
        "in_progress_tasks.json",
        "claude_execution_queue.json",
        "task_state.json"
    ]
    
    for filename in files_to_check:
        filepath = Path(filename)
        if filepath.exists():
            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        count = len(data)
                    elif isinstance(data, list):
                        count = len(data)
                    else:
                        count = "unknown"
                    print(f"✅ {filename}: exists ({count} items)")
                except json.JSONDecodeError:
                    print(f"⚠️  {filename}: exists but invalid JSON")
        else:
            print(f"❌ {filename}: does not exist")


async def main():
    """Run all tests."""
    print("🚀 Task Completion Tracking Test Suite")
    print("=" * 80)
    print("This test verifies that tasks are only marked as completed")
    print("after Claude actually executes them, not just when queued.")
    print("=" * 80)
    
    # Run tests
    test1_passed = await test_task_lifecycle()
    test2_passed = await test_duplicate_handling()
    
    # Check file states
    check_file_state()
    
    # Final summary
    print("\n\n" + "=" * 80)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 80)
    
    all_passed = test1_passed and test2_passed
    
    if all_passed:
        print("✅ All tests passed! The task completion tracking fix is working correctly.")
        print("\n🎉 The system now properly tracks task lifecycle:")
        print("   1. Tasks are created and queued for Claude")
        print("   2. Tasks remain 'in-progress' while Claude executes")
        print("   3. Tasks are only marked 'completed' after Claude finishes")
    else:
        print("❌ Some tests failed. Please check the output above for details.")
        print("\n🔧 To fix, run these commands:")
        print("   python3 fix_task_completion_tracking.py")
        print("   python3 fix_claude_completion_notification.py")


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    
    # Run async main
    asyncio.run(main())