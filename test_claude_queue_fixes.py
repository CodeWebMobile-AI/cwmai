#!/usr/bin/env python3
"""
Test Claude Queue Fixes

Tests the fixes for:
1. Claude task queue deduplication 
2. GitHub issue queue filtering of completed tasks
"""

import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.claude_execution_queue import ClaudeExecutionQueue, ClaudeTaskStatus
from scripts.github_issue_queue_no_redis import GitHubIssueQueueNoRedis
from scripts.ai_brain import IntelligentAIBrain


async def test_claude_queue_deduplication():
    """Test that Claude queue properly deduplicates tasks."""
    print("\n=== Testing Claude Queue Deduplication ===")
    
    # Create a temporary queue file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        queue_file = f.name
    
    try:
        # Initialize queue
        queue = ClaudeExecutionQueue(queue_file=queue_file)
        
        # Test 1: Add a new task
        print("\n1. Adding first task...")
        task1 = await queue.add_task(
            issue_path="/path/to/issue1.json",
            repository="test-repo",
            issue_number=123,
            prompt="Fix bug in authentication"
        )
        assert task1 is not None, "First task should be created"
        print(f"✓ Created task {task1.id}")
        
        # Test 2: Try to add duplicate - should return None
        print("\n2. Attempting to add duplicate task...")
        task2 = await queue.add_task(
            issue_path="/path/to/issue1.json",
            repository="test-repo",
            issue_number=123,
            prompt="Fix bug in authentication (duplicate)"
        )
        assert task2 is None, "Duplicate task should return None"
        print("✓ Duplicate rejected as expected")
        
        # Test 3: Add different issue - should succeed
        print("\n3. Adding different issue...")
        task3 = await queue.add_task(
            issue_path="/path/to/issue2.json",
            repository="test-repo",
            issue_number=124,
            prompt="Add new feature"
        )
        assert task3 is not None, "Different issue should be created"
        print(f"✓ Created task {task3.id}")
        
        # Test 4: Complete first task and try to add again
        print("\n4. Completing task and checking duplicate detection...")
        await queue.complete_task(task1.id, {"success": True})
        
        task4 = await queue.add_task(
            issue_path="/path/to/issue1.json",
            repository="test-repo",
            issue_number=123,
            prompt="Try to add completed issue again"
        )
        assert task4 is None, "Completed task should not be re-added"
        print("✓ Completed task properly rejected")
        
        # Get stats
        stats = await queue.get_stats()
        print(f"\nQueue stats: {json.dumps(stats, indent=2)}")
        
        print("\n✅ Claude queue deduplication working correctly!")
        
    finally:
        # Cleanup
        if os.path.exists(queue_file):
            os.unlink(queue_file)


async def test_github_issue_filtering():
    """Test that GitHub issue queue filters completed tasks."""
    print("\n=== Testing GitHub Issue Filtering ===")
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        claude_queue_file = f.name
    
    try:
        # Initialize AI brain (mock)
        ai_brain = IntelligentAIBrain()
        
        # Initialize GitHub issue queue
        issue_queue = GitHubIssueQueueNoRedis(ai_brain=ai_brain)
        
        # Pre-populate Claude queue with completed tasks
        claude_queue = issue_queue.claude_queue
        
        # Add some completed tasks
        print("\n1. Pre-populating Claude queue with completed tasks...")
        task1 = await claude_queue.add_task(
            issue_path="/path/to/completed.json",
            repository="CodeWebMobile-AI/test-repo",
            issue_number=100,
            prompt="Already completed"
        )
        if task1:
            await claude_queue.complete_task(task1.id, {"success": True})
            print("✓ Added completed task for issue #100")
        
        # Add a pending task
        task2 = await claude_queue.add_task(
            issue_path="/path/to/pending.json",
            repository="CodeWebMobile-AI/test-repo",
            issue_number=101,
            prompt="Currently processing"
        )
        if task2:
            print("✓ Added pending task for issue #101")
        
        # Mock GitHub issues
        mock_issues = [
            {
                'number': 100,
                'title': 'Completed Issue',
                'body': 'This was already completed',
                'html_url': 'https://github.com/test/repo/issues/100',
                'created_at': '2024-01-01T00:00:00Z',
                'labels': [],
                'repository': {'full_name': 'CodeWebMobile-AI/test-repo'}
            },
            {
                'number': 101,
                'title': 'Pending Issue',
                'body': 'This is being processed',
                'html_url': 'https://github.com/test/repo/issues/101',
                'created_at': '2024-01-01T00:00:00Z',
                'labels': [],
                'repository': {'full_name': 'CodeWebMobile-AI/test-repo'}
            },
            {
                'number': 102,
                'title': 'New Issue',
                'body': 'This is a new issue',
                'html_url': 'https://github.com/test/repo/issues/102',
                'created_at': '2024-01-01T00:00:00Z',
                'labels': [],
                'repository': {'full_name': 'CodeWebMobile-AI/test-repo'}
            }
        ]
        
        # Mock the get_open_issues method
        async def mock_get_open_issues():
            return mock_issues
        
        issue_queue.github_creator.get_open_issues = mock_get_open_issues
        
        # Test filtering
        print("\n2. Testing issue filtering...")
        work_items = await issue_queue.get_work_items()
        
        print(f"\nReceived {len(work_items)} work items:")
        for item in work_items:
            print(f"  - {item.title}")
        
        # Verify only the new issue is returned
        assert len(work_items) == 1, f"Expected 1 work item, got {len(work_items)}"
        assert work_items[0].metadata['issue_number'] == 102, "Should only get the new issue"
        
        print("\n✅ GitHub issue filtering working correctly!")
        
    finally:
        # Cleanup
        if os.path.exists(claude_queue_file):
            os.unlink(claude_queue_file)


async def main():
    """Run all tests."""
    print("Testing Claude Queue and GitHub Issue Fixes")
    print("=" * 50)
    
    try:
        await test_claude_queue_deduplication()
        await test_github_issue_filtering()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Load environment
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    
    asyncio.run(main())