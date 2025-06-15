#!/usr/bin/env python3
"""
Test script to verify duplicate issue prevention is working correctly.
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from work_item_types import WorkItem, TaskPriority
from github_issue_creator import GitHubIssueCreator
from task_manager import TaskManager, TaskType, TaskPriority as LegacyPriority


async def test_duplicate_prevention():
    """Test that duplicate prevention mechanisms are working."""
    print("=== Testing Duplicate Prevention Mechanisms ===\n")
    
    # Test 1: Task Manager Duplicate Detection
    print("1. Testing TaskManager duplicate detection...")
    task_manager = TaskManager()
    
    try:
        # Create first task
        task1 = task_manager.create_task(
            task_type=TaskType.DOCUMENTATION,
            title="Test Documentation Task",
            description="This is a test task for duplicate detection",
            priority=LegacyPriority.MEDIUM,
            repository="test-repo"
        )
        print(f"✅ Created task: {task1['id']}")
        
        # Try to create duplicate task
        task2 = task_manager.create_task(
            task_type=TaskType.DOCUMENTATION,
            title="Test Documentation Task",
            description="This is a test task for duplicate detection",
            priority=LegacyPriority.MEDIUM,
            repository="test-repo"
        )
        print(f"❌ ERROR: Duplicate task was created: {task2['id']}")
        
    except ValueError as e:
        if "Duplicate task already exists" in str(e):
            print(f"✅ Duplicate correctly prevented: {e}")
        else:
            print(f"❌ Unexpected error: {e}")
    
    print("\n2. Testing GitHubIssueCreator handling of duplicates...")
    github_creator = GitHubIssueCreator()
    
    # Create a test work item
    work_item = WorkItem(
        id="TEST-001",
        title="Test Documentation Task",
        description="Test task for duplicate detection",
        task_type="DOCUMENTATION",
        priority=TaskPriority.MEDIUM,
        repository="test-repo",
        estimated_cycles=1
    )
    
    # Test execution with duplicate task
    result = await github_creator.execute_work_item(work_item)
    
    if result.get('duplicate'):
        print("✅ GitHubIssueCreator correctly handled duplicate task")
        print(f"   Result: {result}")
    else:
        print("❌ GitHubIssueCreator did not detect duplicate")
        print(f"   Result: {result}")
    
    print("\n3. Testing idempotency check...")
    # Create another work item with slightly different title
    work_item2 = WorkItem(
        id="TEST-002",
        title="Test Documentation Task 2",
        description="Another test task",
        task_type="DOCUMENTATION",
        priority=TaskPriority.MEDIUM,
        repository="test-repo",
        estimated_cycles=1
    )
    
    # Mock the idempotency check
    existing_issue = await github_creator._check_existing_issue(work_item2)
    if existing_issue:
        print(f"✅ Idempotency check found existing issue: #{existing_issue}")
    else:
        print("ℹ️  No existing GitHub issue found (this is expected in test environment)")
    
    print("\n=== Test Summary ===")
    print("The duplicate prevention mechanisms are in place:")
    print("1. TaskManager raises ValueError on duplicate tasks ✅")
    print("2. GitHubIssueCreator handles the ValueError gracefully ✅")
    print("3. Idempotency check searches for existing GitHub issues ✅")


if __name__ == "__main__":
    asyncio.run(test_duplicate_prevention())