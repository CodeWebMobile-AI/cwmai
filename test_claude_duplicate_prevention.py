#!/usr/bin/env python3
"""
Test Claude Duplicate Prevention

This script tests that the Claude execution queue properly prevents
multiple executors from working on the same issue.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.claude_execution_queue import ClaudeExecutionQueue, ClaudeTask, ClaudeTaskStatus


async def test_duplicate_prevention():
    """Test that duplicate prevention works correctly."""
    print("🧪 Testing Claude Duplicate Prevention")
    print("=" * 60)
    
    # Create a test queue file
    test_queue_file = "test_claude_queue.json"
    
    # Clean up any existing test file
    if os.path.exists(test_queue_file):
        os.remove(test_queue_file)
    
    # Create queue instance
    queue = ClaudeExecutionQueue(test_queue_file)
    
    # Test 1: Adding initial task
    print("\n✅ Test 1: Adding initial task")
    task1 = await queue.add_task(
        issue_path="/test/repo/issue-1.json",
        repository="test-org/test-repo",
        issue_number=1,
        prompt="Test prompt for issue 1"
    )
    print(f"   Added task {task1.id} for issue #1")
    
    # Test 2: Check if pending task exists
    print("\n✅ Test 2: Checking has_pending_task_for_issue")
    has_pending = queue.has_pending_task_for_issue("test-org/test-repo", 1)
    print(f"   has_pending_task_for_issue: {has_pending} (should be True)")
    assert has_pending == True, "Should detect pending task"
    
    # Test 3: Claim task for processing
    print("\n✅ Test 3: Claiming task for processing")
    claimed_task = await queue.get_next_task("executor_001")
    print(f"   Executor claimed task: {claimed_task.id}")
    
    # Test 4: Check if processing task is detected
    print("\n✅ Test 4: Checking has_pending_task after claiming")
    has_pending = queue.has_pending_task_for_issue("test-org/test-repo", 1)
    print(f"   has_pending_task_for_issue: {has_pending} (should be True - includes processing)")
    assert has_pending == True, "Should detect processing task"
    
    # Test 5: Try to add duplicate while processing
    print("\n✅ Test 5: Testing duplicate prevention while processing")
    if queue.has_pending_task_for_issue("test-org/test-repo", 1):
        print("   ✓ Correctly prevented duplicate - task already processing")
    else:
        print("   ✗ ERROR: Should have prevented duplicate!")
        assert False, "Failed to prevent duplicate"
    
    # Test 6: Complete the task
    print("\n✅ Test 6: Completing task")
    await queue.complete_task(claimed_task.id, {"status": "completed"})
    print(f"   Task {claimed_task.id} completed")
    
    # Test 7: Check after completion
    print("\n✅ Test 7: Checking has_pending_task after completion")
    has_pending = queue.has_pending_task_for_issue("test-org/test-repo", 1)
    print(f"   has_pending_task_for_issue: {has_pending} (should be False)")
    assert has_pending == False, "Should not detect completed task as pending"
    
    # Test 8: Check completed task detection
    print("\n✅ Test 8: Checking has_completed_task")
    has_completed = queue.has_completed_task_for_issue("test-org/test-repo", 1)
    print(f"   has_completed_task_for_issue: {has_completed} (should be True)")
    assert has_completed == True, "Should detect completed task"
    
    # Test 9: Multiple executors scenario
    print("\n✅ Test 9: Testing multiple executor scenario")
    # Add multiple tasks for different issues
    task2 = await queue.add_task(
        issue_path="/test/repo/issue-2.json",
        repository="test-org/test-repo",
        issue_number=2,
        prompt="Test prompt for issue 2"
    )
    task3 = await queue.add_task(
        issue_path="/test/repo/issue-3.json",
        repository="test-org/test-repo",
        issue_number=3,
        prompt="Test prompt for issue 3"
    )
    print(f"   Added tasks for issues #2 and #3")
    
    # Multiple executors claim tasks
    claimed2 = await queue.get_next_task("executor_002")
    claimed3 = await queue.get_next_task("executor_003")
    print(f"   Executor 002 claimed issue #{claimed2.issue_number}")
    print(f"   Executor 003 claimed issue #{claimed3.issue_number}")
    
    # Try to add duplicate for issue being processed
    if queue.has_pending_task_for_issue("test-org/test-repo", 2):
        print("   ✓ Correctly prevented duplicate for issue #2")
    else:
        print("   ✗ ERROR: Failed to prevent duplicate for issue #2")
        assert False
    
    # Clean up test file
    if os.path.exists(test_queue_file):
        os.remove(test_queue_file)
    
    print("\n✅ All tests passed! Duplicate prevention is working correctly.")
    
    # Show stats
    stats = await queue.get_stats()
    print(f"\nFinal queue stats:")
    print(f"   Pending: {stats['pending']}")
    print(f"   Processing: {stats['processing']}")
    print(f"   Completed: {stats['completed']}")


async def test_real_scenario():
    """Test with the actual queue file to see current state."""
    print("\n\n🔍 Checking Real Queue Status")
    print("=" * 60)
    
    queue = ClaudeExecutionQueue()  # Use default queue file
    
    # Check for specific issues
    test_cases = [
        ("CodeWebMobile-AI/subtrack-ai", 2),
        ("CodeWebMobile-AI/subtrack-ai", 3),
        ("CodeWebMobile-AI/subtrack-ai", 4),
    ]
    
    for repo, issue_num in test_cases:
        has_pending = queue.has_pending_task_for_issue(repo, issue_num)
        has_completed = queue.has_completed_task_for_issue(repo, issue_num)
        
        print(f"\n{repo} issue #{issue_num}:")
        print(f"   Pending/Processing: {has_pending}")
        print(f"   Completed: {has_completed}")
        
        if has_pending:
            # Find which state it's in
            in_queue = any(t.repository == repo and t.issue_number == issue_num for t in queue._queue)
            in_processing = any(t.repository == repo and t.issue_number == issue_num for t in queue._processing.values())
            
            if in_queue:
                print(f"   → Status: In queue (pending)")
            elif in_processing:
                # Find executor
                for task_id, task in queue._processing.items():
                    if task.repository == repo and task.issue_number == issue_num:
                        print(f"   → Status: Being processed by {task.executor_id}")
                        if task.started_at:
                            duration = (datetime.now(timezone.utc) - task.started_at).total_seconds()
                            print(f"   → Duration: {duration:.0f}s")
    
    # Show overall stats
    stats = await queue.get_stats()
    print(f"\n📊 Overall Queue Stats:")
    print(f"   Pending: {stats['pending']}")
    print(f"   Processing: {stats['processing']}")
    print(f"   Completed: {stats['completed']}")
    print(f"   Active Executors: {len(stats['executors'])}")
    if stats['executors']:
        print(f"   Executors: {', '.join(stats['executors'])}")


async def main():
    """Run all tests."""
    await test_duplicate_prevention()
    await test_real_scenario()


if __name__ == "__main__":
    asyncio.run(main())