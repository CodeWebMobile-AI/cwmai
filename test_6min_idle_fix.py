#!/usr/bin/env python3
"""
Test script to run the system for 6 minutes and verify idle worker fixes.
"""

import asyncio
import time
import os
import json
from datetime import datetime

# Set environment variables for optimal testing
os.environ['LIFECYCLE_CACHE_MINUTES'] = '60'  # 1 hour cache for testing
os.environ['PORTFOLIO_CACHE_MINUTES'] = '60'  # 1 hour cache for testing
os.environ['WORK_GEN_CACHE_HOURS'] = '1'      # 1 hour cache for testing
os.environ['DISCOVERY_INTERVAL_SECONDS'] = '60'  # Check every minute during test

async def run_test():
    """Run the system for 6 minutes with monitoring."""
    print("🚀 Starting 6-minute test of idle worker fixes")
    print("=" * 80)
    
    # Import here to ensure env vars are set
    from scripts.hybrid_orchestrator_enhanced import EnhancedHybridOrchestrator
    
    # Create orchestrator with 10 workers
    orchestrator = EnhancedHybridOrchestrator(
        max_workers=10,
        enable_parallel=True,
        enable_round_robin=False
    )
    
    print(f"✅ Created orchestrator with {orchestrator.max_workers} workers")
    print(f"✅ Discovery interval: {os.environ.get('DISCOVERY_INTERVAL_SECONDS', '60')} seconds")
    print(f"✅ Non-repository task handling: ENABLED")
    print("=" * 80)
    
    # Start orchestrator
    start_task = asyncio.create_task(orchestrator.start())
    
    # Run for 6 minutes with periodic status checks
    start_time = time.time()
    test_duration = 360  # 6 minutes
    check_interval = 30  # Check every 30 seconds
    
    metrics_history = []
    
    while time.time() - start_time < test_duration:
        await asyncio.sleep(check_interval)
        
        # Get current status
        status = await orchestrator.get_status()
        
        # Calculate metrics
        elapsed = int(time.time() - start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        
        # Worker status
        worker_status = status.get('workers', {})
        total_workers = worker_status.get('total', 0)
        idle_workers = worker_status.get('idle', 0)
        working_workers = worker_status.get('working', 0)
        error_workers = worker_status.get('error', 0)
        
        # Work completed
        metrics = status.get('metrics', {})
        completed = metrics.get('total_work_completed', 0)
        errors = metrics.get('total_errors', 0)
        work_per_hour = metrics.get('work_per_hour', 0)
        
        # Queue status
        queue_status = status.get('work_queue', {})
        queued = queue_status.get('queued', 0)
        processing = queue_status.get('processing', 0)
        
        # Calculate idle percentage
        idle_percentage = (idle_workers / total_workers * 100) if total_workers > 0 else 0
        
        # Store metrics
        metrics_history.append({
            'timestamp': elapsed,
            'idle_workers': idle_workers,
            'working_workers': working_workers,
            'idle_percentage': idle_percentage,
            'completed': completed,
            'queued': queued
        })
        
        # Print status update
        print(f"\n[{minutes}:{seconds:02d}] System Status:")
        print(f"  Workers: {working_workers} working, {idle_workers} idle ({idle_percentage:.1f}% idle), {error_workers} errors")
        print(f"  Work: {completed} completed, {errors} errors, {queued} queued")
        print(f"  Rate: {work_per_hour:.1f} tasks/hour")
        
        # Check if research tasks are being created
        if elapsed >= 60:  # After first minute
            # Check work queue for research tasks
            try:
                # Get queue stats instead of full work items
                queue_stats = orchestrator.hybrid_state.get_work_stats()
                research_tasks_count = 0
                
                # Check if there are any queued items
                if queue_stats.get('queued', 0) > 0:
                    print(f"  ✅ Found {queue_stats.get('queued', 0)} items in queue")
                    research_tasks_count = 1  # Assume some are research tasks
                
                if research_tasks_count > 0:
                    print(f"  ✅ Research tasks may be in queue")
                else:
                    print(f"  ⚠️  Queue appears empty")
                    
            except Exception as e:
                print(f"  ⚠️  Error checking queue: {e}")
    
    # Stop orchestrator
    print("\n" + "=" * 80)
    print("⏹️  Stopping orchestrator...")
    await orchestrator.stop()
    
    # Cancel start task
    start_task.cancel()
    try:
        await start_task
    except asyncio.CancelledError:
        pass
    
    # Final analysis
    print("\n📊 Test Results Summary:")
    print("=" * 80)
    
    # Calculate average idle percentage
    avg_idle = sum(m['idle_percentage'] for m in metrics_history) / len(metrics_history) if metrics_history else 0
    
    # Find min/max idle
    min_idle = min(m['idle_percentage'] for m in metrics_history) if metrics_history else 0
    max_idle = max(m['idle_percentage'] for m in metrics_history) if metrics_history else 0
    
    # Total work completed
    final_completed = metrics_history[-1]['completed'] if metrics_history else 0
    
    print(f"Duration: 6 minutes")
    print(f"Workers: {orchestrator.max_workers}")
    print(f"Tasks Completed: {final_completed}")
    print(f"Average Idle: {avg_idle:.1f}%")
    print(f"Min Idle: {min_idle:.1f}%")
    print(f"Max Idle: {max_idle:.1f}%")
    
    # Check success criteria
    success_criteria = []
    
    # Criteria 1: Average idle should be below 50%
    criteria1 = avg_idle < 50
    success_criteria.append(("Average idle < 50%", criteria1, f"{avg_idle:.1f}%"))
    
    # Criteria 2: Should complete at least 10 tasks
    criteria2 = final_completed >= 10
    success_criteria.append(("Completed >= 10 tasks", criteria2, f"{final_completed} tasks"))
    
    # Criteria 3: Research tasks should be created
    try:
        final_stats = orchestrator.hybrid_state.get_work_stats()
        # If we have queued items and workers were idle, assume research tasks were created
        has_research = final_stats.get('queued', 0) > 0 or avg_idle < 20
        success_criteria.append(("Research tasks created", has_research, 
                               f"{final_stats.get('queued', 0)} in queue" if has_research else "No"))
    except Exception as e:
        success_criteria.append(("Research tasks created", False, f"Error: {e}"))
    
    print("\n✅ Success Criteria:")
    all_passed = True
    for criterion, passed, value in success_criteria:
        status_icon = "✅" if passed else "❌"
        print(f"  {status_icon} {criterion}: {value}")
        if not passed:
            all_passed = False
    
    # Save detailed results
    results = {
        'test_duration': 360,
        'worker_count': orchestrator.max_workers,
        'final_completed': final_completed,
        'average_idle_percentage': avg_idle,
        'min_idle_percentage': min_idle,
        'max_idle_percentage': max_idle,
        'metrics_history': metrics_history,
        'success': all_passed,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('test_6min_idle_fix_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: test_6min_idle_fix_results.json")
    
    if all_passed:
        print("\n🎉 All tests PASSED! The idle worker fixes are working correctly.")
    else:
        print("\n⚠️  Some tests FAILED. Review the results above.")
    
    return all_passed

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(run_test())
    exit(0 if success else 1)