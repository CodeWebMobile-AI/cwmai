#!/usr/bin/env python3
"""
Test the Task Monitoring System
"""

import asyncio
import os
import sys

sys.path.insert(0, '/Users/fortressarigo/PhpstormProjects/cwmai')
os.environ['STORAGE_MODE'] = 'local'

from scripts.task_monitoring_system import get_monitoring_system
from scripts.monitoring_integration import MonitoringIntegration, MonitoredAIWorker
from scripts.work_item_types import WorkItem, TaskPriority
import time


async def test_monitoring_system():
    """Test various monitoring features."""
    print("Testing Task Monitoring System...")
    print("=" * 80)
    
    # Get monitoring system
    monitor = get_monitoring_system()
    integration = MonitoringIntegration()
    
    # Test 1: Basic task monitoring
    print("\n1. Testing basic task monitoring:")
    
    task_id = "test_task_001"
    metrics = monitor.start_task_monitoring(
        task_id=task_id,
        task_type="feature_api_endpoint",
        repository="test-repo",
        estimated_duration=3600  # 1 hour
    )
    
    print(f"   ✓ Started monitoring task {task_id}")
    
    # Simulate some work and resource usage
    for i in range(3):
        monitor.record_resource_usage(task_id)
        monitor.record_api_call(task_id, "openai")
        await asyncio.sleep(1)
    
    # Complete the task
    completed_metrics = monitor.complete_task_monitoring(task_id, success=True)
    print(f"   ✓ Completed task monitoring")
    print(f"   - Actual duration: {completed_metrics.actual_duration:.1f}s")
    print(f"   - API calls: {completed_metrics.total_api_calls}")
    
    # Test 2: Execution steps
    print("\n2. Testing execution steps:")
    
    task_id = "test_task_002"
    monitor.start_task_monitoring(
        task_id=task_id,
        task_type="testing_unit_tests",
        repository="test-repo"
    )
    
    with integration.track_execution_step(task_id, "setup"):
        print("   ✓ Setup step")
        await asyncio.sleep(0.5)
    
    with integration.track_execution_step(task_id, "test_execution"):
        print("   ✓ Test execution step")
        await asyncio.sleep(0.5)
    
    with integration.track_execution_step(task_id, "cleanup"):
        print("   ✓ Cleanup step")
        await asyncio.sleep(0.5)
    
    monitor.complete_task_monitoring(task_id, success=True)
    
    # Test 3: Task dependencies
    print("\n3. Testing task dependencies:")
    
    parent_id = "test_task_003"
    child_id = "test_task_004"
    
    monitor.add_dependency(parent_id, child_id, "blocks")
    blocking = monitor.get_blocking_tasks(child_id)
    print(f"   ✓ Task {child_id} is blocked by: {blocking}")
    
    # Test 4: Worker performance
    print("\n4. Testing worker performance:")
    
    worker_id = "test_worker_001"
    for i in range(5):
        task_type = "feature_api_endpoint" if i % 2 == 0 else "testing_unit_tests"
        monitor.update_worker_performance(
            worker_id=worker_id,
            task_id=f"task_{i}",
            task_type=task_type,
            success=i != 2,  # One failure
            duration=3600 + (i * 300)
        )
    
    best_worker = monitor.get_best_worker_for_task("feature_api_endpoint")
    print(f"   ✓ Best worker for API endpoints: {best_worker}")
    
    # Test 5: Predictive analytics
    print("\n5. Testing predictive analytics:")
    
    predicted_duration = monitor.predict_task_duration("feature_api_endpoint", "test-repo")
    print(f"   ✓ Predicted duration: {predicted_duration/3600:.1f} hours")
    
    failure_risk = monitor.calculate_failure_risk("testing_unit_tests")
    print(f"   ✓ Failure risk: {failure_risk:.1%}")
    
    optimal_time = monitor.get_optimal_scheduling_time("feature_api_endpoint")
    print(f"   ✓ Optimal scheduling: {optimal_time}")
    
    # Test 6: Quality metrics
    print("\n6. Testing quality metrics:")
    
    task_id = "test_task_005"
    quality = await monitor.measure_quality_metrics(task_id, "/tmp/test-repo")
    print(f"   ✓ Quality metrics captured")
    
    # Test 7: Lifecycle events
    print("\n7. Testing lifecycle events:")
    
    task_id = "test_task_006"
    monitor.start_task_monitoring(task_id, "maintenance_bug_fix", "test-repo")
    
    monitor.record_lifecycle_event(task_id, "paused", {"reason": "waiting for review"})
    print("   ✓ Task paused")
    
    await asyncio.sleep(1)
    
    monitor.record_lifecycle_event(task_id, "resumed", {"reviewer": "user123"})
    print("   ✓ Task resumed")
    
    monitor.complete_task_monitoring(task_id, success=True)
    
    # Test 8: Using monitored worker
    print("\n8. Testing monitored AI worker:")
    
    worker = MonitoredAIWorker("test_worker_002")
    work_item = WorkItem(
        id="test_work_001",
        title="Test Work Item",
        description="Test description",
        task_type="feature_api_endpoint",
        repository="test-repo",
        priority=TaskPriority.HIGH
    )
    
    try:
        result = await worker.execute_work_item(work_item)
        print(f"   ✓ Work item executed successfully")
    except Exception as e:
        print(f"   ✗ Work item failed: {e}")
    
    # Test 9: Generate reports
    print("\n9. Testing reporting:")
    
    # Task report
    report = monitor.generate_task_report("test_task_001")
    print(f"   ✓ Generated task report with {len(report)} sections")
    
    # System insights
    insights = monitor.get_system_insights()
    print(f"   ✓ Generated system insights:")
    print(f"     - Task types tracked: {len(insights['task_types'])}")
    print(f"     - Workers monitored: {len(insights['worker_performance'])}")
    
    # Test 10: Learning patterns
    print("\n10. Testing learning patterns:")
    
    # Simulate some failures to create patterns
    for i in range(3):
        task_id = f"fail_task_{i}"
        monitor.start_task_monitoring(task_id, "integration_api", "test-repo")
        monitor.update_task_metrics(task_id, {
            "error_count": 1,
            "error_types": ["ConnectionTimeout"]
        })
        monitor.complete_task_monitoring(task_id, success=False, error="ConnectionTimeout")
    
    patterns = monitor.get_common_failure_patterns("integration_api")
    print(f"   ✓ Found {len(patterns)} failure patterns")
    for pattern in patterns:
        print(f"     - {pattern['error_type']}: {pattern['frequency']} occurrences")
    
    print("\n" + "=" * 80)
    print("All monitoring tests completed successfully!")
    
    # Save monitoring data
    monitor._save_monitoring_data()
    print("\nMonitoring data saved to task_monitoring/ directory")


if __name__ == "__main__":
    asyncio.run(test_monitoring_system())