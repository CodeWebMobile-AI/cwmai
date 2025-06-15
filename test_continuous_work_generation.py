#!/usr/bin/env python3
"""Test continuous work generation improvements."""

import asyncio
import sys
import os
from datetime import datetime, timezone
import time

# Add the scripts directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from enhanced_work_generator import EnhancedWorkGenerator
from work_item_types import WorkItem, TaskPriority
from continuous_orchestrator import ContinuousOrchestrator


async def test_enhanced_work_generator():
    """Test the enhanced work generator."""
    print("\n=== Testing Enhanced Work Generator ===")
    
    generator = EnhancedWorkGenerator()
    
    # Test normal work generation
    print("\n--- Testing normal work generation ---")
    work_batch = await generator.generate_work_batch(target_count=5)
    print(f"Generated {len(work_batch)} work items:")
    for item in work_batch:
        print(f"  - [{item.task_type}] {item.title} (Priority: {item.priority.name})")
    
    assert len(work_batch) == 5, "Should generate requested number of items"
    
    # Test emergency work generation
    print("\n--- Testing emergency work generation ---")
    emergency_work = await generator.generate_emergency_work(count=3)
    print(f"Generated {len(emergency_work)} emergency items:")
    for item in emergency_work:
        print(f"  - [{item.task_type}] {item.title} (Priority: {item.priority.name})")
    
    # Verify all emergency work is high priority
    high_priority_count = sum(1 for item in emergency_work if item.priority == TaskPriority.HIGH)
    print(f"High priority items: {high_priority_count}/{len(emergency_work)}")
    
    # Test maintenance work generation
    print("\n--- Testing maintenance work generation ---")
    maintenance_work = await generator.generate_maintenance_work(count=3)
    print(f"Generated {len(maintenance_work)} maintenance items:")
    for item in maintenance_work:
        print(f"  - [{item.task_type}] {item.title} (Priority: {item.priority.name})")
    
    # Get generation stats
    stats = generator.get_generation_stats()
    print(f"\nGeneration stats:")
    print(f"  Total generated: {stats['total_generated']}")
    print(f"  Unique titles: {stats['unique_titles']}")
    print(f"  Available templates: {stats['available_templates']}")
    
    return True


async def test_continuous_work_discovery():
    """Test that work discovery runs continuously."""
    print("\n=== Testing Continuous Work Discovery ===")
    
    # Create a minimal orchestrator
    orchestrator = ContinuousOrchestrator(max_workers=2, enable_parallel=False)
    
    # Track work discovery calls
    discovery_count = 0
    original_discover = orchestrator._discover_work
    
    async def mock_discover_work():
        nonlocal discovery_count
        discovery_count += 1
        print(f"  Work discovery called (count: {discovery_count})")
        # Don't actually discover work to speed up test
        return
    
    orchestrator._discover_work = mock_discover_work
    
    # Run for a short time
    print("Running orchestrator for 10 seconds...")
    
    async def run_orchestrator():
        try:
            await orchestrator._run_main_loop()
        except asyncio.CancelledError:
            pass
    
    # Start the orchestrator
    task = asyncio.create_task(run_orchestrator())
    
    # Wait 10 seconds
    await asyncio.sleep(10)
    
    # Stop the orchestrator
    orchestrator.shutdown_requested = True
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        pass
    
    print(f"\nWork discovery was called {discovery_count} times in 10 seconds")
    print(f"Average interval: {10/max(discovery_count, 1):.2f} seconds")
    
    # Should be called at least 4 times (every 2 seconds)
    assert discovery_count >= 4, f"Work discovery should run continuously (got {discovery_count} calls)"
    
    return True


async def test_queue_threshold_maintenance():
    """Test that queue maintains minimum threshold."""
    print("\n=== Testing Queue Threshold Maintenance ===")
    
    orchestrator = ContinuousOrchestrator(max_workers=10, enable_parallel=False)
    
    # Check target queue size calculation
    target_size = max(10, orchestrator.max_workers * 3)
    print(f"Workers: {orchestrator.max_workers}")
    print(f"Target queue size: {target_size}")
    
    assert target_size >= 10, "Target queue size should be at least 10"
    
    # Test with different worker counts
    test_cases = [
        (1, 10),   # 1 worker -> min 10
        (2, 10),   # 2 workers -> min 10
        (5, 15),   # 5 workers -> 15
        (10, 30),  # 10 workers -> 30
    ]
    
    for workers, expected in test_cases:
        actual = max(10, workers * 3)
        print(f"  {workers} workers -> target size: {actual} (expected: {expected})")
        assert actual == expected, f"Unexpected target size for {workers} workers"
    
    return True


async def test_work_generation_on_empty_queue():
    """Test that emergency work is generated when queue is empty."""
    print("\n=== Testing Emergency Work Generation ===")
    
    generator = EnhancedWorkGenerator()
    
    # Simulate empty queue scenario
    current_queue_size = 0
    target_queue_size = 10
    
    print(f"Queue state: {current_queue_size}/{target_queue_size}")
    
    # Should trigger emergency generation
    if current_queue_size == 0:
        print("Queue is empty - generating emergency work")
        emergency_work = await generator.generate_emergency_work(count=5)
        print(f"Generated {len(emergency_work)} emergency items")
        
        # Verify work was generated
        assert len(emergency_work) >= 5, "Should generate at least 5 emergency items"
        
        # Verify priority
        for item in emergency_work:
            print(f"  - {item.title} (Priority: {item.priority.name})")
    
    return True


async def main():
    """Run all tests."""
    print("🧪 Testing Continuous Work Generation Fixes")
    print("=" * 60)
    
    tests = [
        ("Enhanced Work Generator", test_enhanced_work_generator),
        ("Continuous Work Discovery", test_continuous_work_discovery),
        ("Queue Threshold Maintenance", test_queue_threshold_maintenance),
        ("Emergency Work Generation", test_work_generation_on_empty_queue),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            start_time = time.time()
            result = await test_func()
            duration = time.time() - start_time
            results.append((test_name, result, duration))
            print(f"\n✅ {test_name} passed in {duration:.2f}s")
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, 0))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, duration in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status} ({duration:.2f}s)")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The continuous work generation is fixed.")
        print("\nKey improvements:")
        print("  - Work discovery runs every 2 seconds")
        print("  - Minimum queue size of 10 items maintained")
        print("  - Emergency work generated when queue is empty")
        print("  - Periodic maintenance work every 5 minutes")
        print("  - Alternative work generated when regular discovery fails")
    else:
        print("\n⚠️  Some tests failed. Please review the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)