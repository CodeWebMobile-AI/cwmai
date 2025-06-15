#!/usr/bin/env python3
"""Test smart duplicate handling and worker distribution improvements."""

import asyncio
import sys
import os
from datetime import datetime, timezone

# Add the scripts directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from continuous_orchestrator import ContinuousOrchestrator, WorkerState, WorkerStatus
from work_item_types import WorkItem, TaskPriority
from alternative_task_generator import AlternativeTaskGenerator
from state_manager import StateManager


async def test_alternative_task_generation():
    """Test the alternative task generator."""
    print("\n=== Testing Alternative Task Generation ===")
    
    generator = AlternativeTaskGenerator()
    
    # Test duplicate documentation task
    original_task = WorkItem(
        id="doc_123",
        task_type="DOCUMENTATION",
        title="Update documentation for moderncms-with-ai-powered-content-recommendations changes",
        description="Update the docs",
        priority=TaskPriority.MEDIUM,
        repository="moderncms-with-ai-powered-content-recommendations",
        estimated_cycles=2
    )
    
    # Generate alternative
    alternative = await generator.generate_alternative_task(original_task)
    
    if alternative:
        print(f"✅ Original task: {original_task.title}")
        print(f"✅ Alternative generated: {alternative.title}")
        print(f"   Type: {alternative.task_type}")
        print(f"   Priority: {alternative.priority}")
        assert alternative.title != original_task.title, "Alternative should be different"
        assert alternative.repository == original_task.repository, "Should be for same repository"
    else:
        print("❌ Failed to generate alternative")
        return False
    
    # Test multiple alternatives
    print("\n--- Testing batch generation ---")
    alternatives = await generator.generate_alternative_batch([original_task], max_alternatives=3)
    print(f"Generated {len(alternatives)} alternatives:")
    for alt in alternatives:
        print(f"  - {alt.title} ({alt.task_type})")
    
    return True


async def test_worker_distribution():
    """Test the improved worker distribution."""
    print("\n=== Testing Worker Distribution ===")
    
    # Create orchestrator with 10 workers
    orchestrator = ContinuousOrchestrator(max_workers=10, enable_parallel=True)
    
    # Mock system state with projects
    orchestrator.system_state = {
        'projects': {
            'ai-creative-studio': {},
            'moderncms-with-ai-powered-content-recommendations': {},
            'another-project': {},
            'yet-another-project': {}
        }
    }
    
    # Test specialization assignment
    specializations = []
    for i in range(10):
        spec = orchestrator._assign_worker_specialization(i)
        specializations.append(spec)
        print(f"Worker {i+1}: {spec}")
    
    # Count distribution
    system_count = sum(1 for s in specializations if s in ["system_tasks", "general"])
    project_count = sum(1 for s in specializations if s not in ["system_tasks", "general"])
    
    print(f"\nDistribution:")
    print(f"  System/General workers: {system_count} ({system_count/10*100:.0f}%)")
    print(f"  Project workers: {project_count} ({project_count/10*100:.0f}%)")
    
    # Verify better distribution
    assert system_count >= 3, "Should have at least 3 system/general workers"
    assert project_count >= 6, "Should have at least 6 project workers"
    
    return True


async def test_duplicate_handling_flow():
    """Test the complete duplicate handling flow."""
    print("\n=== Testing Duplicate Handling Flow ===")
    
    # Create a minimal orchestrator
    orchestrator = ContinuousOrchestrator(max_workers=1, enable_parallel=False)
    
    # Create a duplicate task
    duplicate_task = WorkItem(
        id="task_001",
        task_type="DOCUMENTATION",
        title="Update documentation for feature X",
        description="Already completed task",
        priority=TaskPriority.HIGH,
        repository="test-repo",
        estimated_cycles=1
    )
    
    # Mock task persistence to return duplicate
    class MockTaskPersistence:
        def is_duplicate_task(self, work_item):
            return True
        
        def record_skipped_task(self, title, reason):
            print(f"  Recorded skipped task: {title} (reason: {reason})")
    
    orchestrator.task_persistence = MockTaskPersistence()
    
    # Mock work queue
    orchestrator.work_queue = []
    
    # Perform work
    result = await orchestrator._perform_work(duplicate_task)
    
    print(f"\nResult:")
    print(f"  Success: {result.get('success')}")
    print(f"  Alternative generated: {result.get('alternative_generated', False)}")
    if result.get('alternative_task'):
        print(f"  Alternative task: {result.get('alternative_task')}")
    
    # Check that alternative was generated
    if result.get('alternative_generated'):
        print("✅ Alternative task was generated successfully")
        # Check if task was added to queue
        if orchestrator.work_queue:
            print(f"✅ Alternative task added to queue: {orchestrator.work_queue[0].title}")
    else:
        print("❌ No alternative was generated")
    
    return result.get('alternative_generated', False)


async def test_idle_worker_flexibility():
    """Test that idle workers can take on different work."""
    print("\n=== Testing Idle Worker Flexibility ===")
    
    orchestrator = ContinuousOrchestrator(max_workers=2, enable_parallel=False)
    
    # Create workers with different specializations
    worker1 = WorkerState(
        id="worker_1",
        status=WorkerStatus.IDLE,
        specialization="ai-creative-studio"
    )
    worker1.last_activity = datetime.now(timezone.utc)
    
    # Make worker2 idle for a while
    worker2 = WorkerState(
        id="worker_2", 
        status=WorkerStatus.IDLE,
        specialization="moderncms-with-ai-powered-content-recommendations"
    )
    from datetime import timedelta
    worker2.last_activity = datetime.now(timezone.utc) - timedelta(seconds=60)
    
    # Add general task to queue
    general_task = WorkItem(
        id="general_001",
        task_type="DOCUMENTATION",
        title="Update main README",
        description="General documentation task",
        priority=TaskPriority.HIGH,
        repository=None,  # No specific repository
        estimated_cycles=1
    )
    
    orchestrator.work_queue = [general_task]
    
    # Test if idle specialized worker can take general work
    work = await orchestrator._find_work_for_worker(worker2)
    
    if work:
        print(f"✅ Idle worker {worker2.id} (spec: {worker2.specialization}) took general task: {work.title}")
        return True
    else:
        print(f"❌ Idle worker couldn't take general work")
        return False


async def main():
    """Run all tests."""
    print("🧪 Testing Smart Duplicate Handling and Worker Distribution")
    print("=" * 60)
    
    tests = [
        ("Alternative Task Generation", test_alternative_task_generation),
        ("Worker Distribution", test_worker_distribution),
        ("Duplicate Handling Flow", test_duplicate_handling_flow),
        ("Idle Worker Flexibility", test_idle_worker_flexibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The improvements are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please review the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)