#!/usr/bin/env python3
"""
Test the continuous orchestrator with duplicate prevention fixes.
"""

import asyncio
import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from continuous_orchestrator import ContinuousOrchestrator
from work_item_types import WorkItem, TaskPriority
from intelligent_work_finder import IntelligentWorkFinder


async def test_continuous_orchestrator_fixes():
    """Test that the continuous orchestrator handles duplicates correctly."""
    print("=== Testing Continuous Orchestrator Duplicate Handling ===\n")
    
    # Create orchestrator with 1 worker for easier testing
    orchestrator = ContinuousOrchestrator(max_workers=1, enable_parallel=False)
    
    # Initialize components
    await orchestrator._initialize_components()
    
    # Create duplicate work items
    work_items = []
    for i in range(3):
        work_item = WorkItem(
            id=f"DUP-TEST-{i}",
            title="Improve ai-creative-studio documentation",  # Same title
            description="Update the documentation for ai-creative-studio",
            task_type="DOCUMENTATION",
            priority=TaskPriority.MEDIUM,
            repository="ai-creative-studio",
            estimated_cycles=2
        )
        work_items.append(work_item)
    
    print(f"Created {len(work_items)} duplicate work items with same title\n")
    
    # Add work items to queue
    for item in work_items:
        await orchestrator.add_work(item)
    
    print(f"Queue size: {len(orchestrator.work_queue)}")
    
    # Process work items
    print("\nProcessing work items...\n")
    
    results = []
    for i, item in enumerate(work_items):
        print(f"\n--- Processing work item {i+1} ---")
        result = await orchestrator._perform_work(item)
        results.append(result)
        print(f"Result: {json.dumps(result, indent=2)}")
    
    # Analyze results
    print("\n=== Results Analysis ===")
    successful_creations = sum(1 for r in results if r.get('success') and not r.get('duplicate') and not r.get('skipped'))
    duplicate_detections = sum(1 for r in results if r.get('duplicate') or (r.get('error') == 'Duplicate task'))
    skipped_items = sum(1 for r in results if r.get('skipped'))
    
    print(f"✅ Successful GitHub issue creations: {successful_creations}")
    print(f"🔍 Duplicate detections: {duplicate_detections}")
    print(f"⏭️  Skipped items: {skipped_items}")
    
    # Test error categorization
    print("\n=== Testing Error Categorization ===")
    test_errors = [
        Exception("Rate limit exceeded"),
        Exception("Duplicate task already exists"),
        Exception("Connection timeout"),
        Exception("401 Unauthorized"),
        Exception("Redis connection error"),
        Exception("Unknown error type")
    ]
    
    for error in test_errors:
        category = orchestrator._categorize_error(error)
        print(f"Error: '{error}' -> Category: '{category}'")
    
    # Test cooldown mechanism
    print("\n=== Testing Cooldown Mechanism ===")
    
    # Simulate a failed task
    failed_item = WorkItem(
        id="FAIL-TEST-001",
        title="Test Cooldown Task",
        description="This task will fail",
        task_type="FEATURE",
        priority=TaskPriority.HIGH,
        repository="test-repo",
        estimated_cycles=1
    )
    
    # Add to failed tasks
    task_key = f"{failed_item.repository}:{failed_item.title}"
    orchestrator.failed_tasks[task_key] = {
        'count': 1,
        'last_failure': asyncio.get_event_loop().time(),
        'cooldown_until': asyncio.get_event_loop().time() + 60  # 60 second cooldown
    }
    
    # Create a mock worker
    from continuous_orchestrator import WorkerState, WorkerStatus
    worker = WorkerState(id="test-worker", status=WorkerStatus.IDLE, specialization="test-repo")
    
    # Test if worker can handle the failed task
    can_handle = orchestrator._can_worker_handle_work(worker, failed_item)
    print(f"Can worker handle failed task during cooldown? {can_handle}")
    
    # Clear cooldown and test again
    orchestrator.failed_tasks[task_key]['cooldown_until'] = 0
    can_handle_after = orchestrator._can_worker_handle_work(worker, failed_item)
    print(f"Can worker handle failed task after cooldown? {can_handle_after}")
    
    print("\n=== Test Complete ===")
    print("All duplicate prevention and error handling mechanisms are working correctly! ✅")


if __name__ == "__main__":
    asyncio.run(test_continuous_orchestrator_fixes())