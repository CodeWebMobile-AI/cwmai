#!/usr/bin/env python3
"""
Test the integrated monitoring system in the orchestrator
"""
import asyncio
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, '/Users/fortressarigo/PhpstormProjects/cwmai')
os.environ['STORAGE_MODE'] = 'local'

from scripts.enhanced_hybrid_orchestrator import EnhancedHybridOrchestrator
from scripts.work_item_types import WorkItem, TaskPriority
from scripts.task_monitoring_system import get_monitoring_system
import time


async def test_integrated_monitoring():
    """Test monitoring integration in the orchestrator."""
    print("Testing Integrated Monitoring System")
    print("=" * 80)
    
    # Initialize orchestrator (which includes monitoring)
    orchestrator = EnhancedHybridOrchestrator(
        max_workers=2,
        enable_parallel=True,
        enable_research=False  # Disable research for faster test
    )
    
    # Get monitoring system
    monitor = get_monitoring_system()
    
    print("\n1. Creating test work items...")
    
    # Create test work items
    test_items = [
        WorkItem(
            id="test_monitor_001",
            title="Test Feature Implementation",
            description="Test feature for monitoring",
            task_type="feature_api_endpoint",
            repository="test-monitoring-repo",
            priority=TaskPriority.HIGH
        ),
        WorkItem(
            id="test_monitor_002",
            title="Test Unit Tests",
            description="Test unit tests for monitoring",
            task_type="testing_unit_tests",
            repository="test-monitoring-repo",
            priority=TaskPriority.MEDIUM
        )
    ]
    
    # Add work items to queue using hybrid state
    for item in test_items:
        orchestrator.hybrid_state.enqueue_work(item)
    
    print(f"   ✓ Added {len(test_items)} work items to queue")
    
    # Run orchestrator for a short time
    print("\n2. Running orchestrator with monitoring...")
    
    # Create a task to run the orchestrator
    orchestrator_task = asyncio.create_task(run_orchestrator_briefly(orchestrator, duration=30))
    
    # Monitor the progress
    start_time = time.time()
    while time.time() - start_time < 30:
        await asyncio.sleep(5)
        
        # Check monitoring data
        active_tasks = len(monitor.active_tasks)
        completed_count = len([t for t in monitor.task_history.get("feature_api_endpoint", [])])
        
        print(f"\r   Active tasks: {active_tasks}, Completed: {completed_count}", end="", flush=True)
    
    print("\n")
    
    # Stop orchestrator
    await orchestrator_task
    
    # Generate reports
    print("\n3. Analyzing monitoring data...")
    
    # Get system insights
    insights = monitor.get_system_insights()
    
    print("\n   Task Type Performance:")
    for task_type, stats in insights.get("task_types", {}).items():
        print(f"   - {task_type}:")
        print(f"     Total: {stats.get('total_tasks', 0)}")
        print(f"     Success rate: {stats.get('recent_success_rate', 0):.1%}")
        print(f"     Avg duration: {stats.get('avg_duration_seconds', 0):.1f}s")
    
    print("\n   Worker Performance:")
    for worker_id, perf in insights.get("worker_performance", {}).items():
        print(f"   - {worker_id}:")
        print(f"     Completed: {perf.get('tasks_completed', 0)}")
        print(f"     Failed: {perf.get('tasks_failed', 0)}")
        print(f"     Efficiency: {perf.get('avg_efficiency', 0):.1f}")
    
    # Check monitoring files
    print("\n4. Checking monitoring data persistence...")
    
    monitoring_dir = monitor.monitoring_dir
    if monitoring_dir.exists():
        files = list(monitoring_dir.glob("*.json"))
        print(f"   ✓ Found {len(files)} monitoring files")
        
        # Show some files
        for file in files[:5]:
            print(f"     - {file.name}")
    
    # Test predictive features
    print("\n5. Testing predictive analytics...")
    
    # Predict duration for a task
    predicted = monitor.predict_task_duration("feature_api_endpoint", "test-repo")
    print(f"   - Predicted duration for API endpoint: {predicted/3600:.1f} hours")
    
    # Get failure risk
    risk = monitor.calculate_failure_risk("testing_unit_tests")
    print(f"   - Failure risk for unit tests: {risk:.1%}")
    
    # Get best worker
    best = monitor.get_best_worker_for_task("feature_api_endpoint")
    print(f"   - Best worker for API endpoints: {best or 'No data yet'}")
    
    # Get optimal scheduling
    optimal = monitor.get_optimal_scheduling_time("feature_api_endpoint")
    print(f"   - Optimal scheduling: Hour {optimal['hour']} ({optimal['reason']})")
    
    print("\n" + "=" * 80)
    print("Monitoring integration test completed!")
    
    # Save final monitoring data
    monitor._save_monitoring_data()
    print("\n✓ Monitoring data saved")


async def run_orchestrator_briefly(orchestrator, duration=30):
    """Run orchestrator for a brief period."""
    try:
        # Start workers
        await orchestrator._start_workers()
        
        # Run for specified duration
        start_time = time.time()
        while time.time() - start_time < duration:
            # Process work queue
            if orchestrator.work_queue:
                work_item = orchestrator.work_queue.pop(0)
                
                # Find idle worker
                idle_worker = None
                for worker in orchestrator.workers.values():
                    if worker.status.value == "idle":
                        idle_worker = worker
                        break
                
                if idle_worker:
                    # Execute work
                    await orchestrator._process_work_item(idle_worker.id, work_item)
            
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"\nError in orchestrator: {e}")
    finally:
        # Stop workers
        for worker_id in list(orchestrator.workers.keys()):
            await orchestrator._stop_worker(worker_id)


if __name__ == "__main__":
    asyncio.run(test_integrated_monitoring())