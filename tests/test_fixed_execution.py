#!/usr/bin/env python3
"""Test the fixed work execution."""

import asyncio
import sys
import logging
from datetime import datetime, timezone

sys.path.insert(0, '/workspaces/cwmai/scripts')

from continuous_orchestrator import ContinuousOrchestrator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_fixed_execution():
    """Test the fixed execution with different worker counts."""
    print("=== TESTING FIXED WORK EXECUTION ===\n")
    
    # Test 1: Single worker (should be general)
    print("Test 1: Single worker configuration")
    orchestrator1 = ContinuousOrchestrator(max_workers=1)
    spec1 = orchestrator1._assign_worker_specialization(0)
    print(f"  Worker 0 specialization: {spec1}")
    print(f"  Expected: general")
    print(f"  ✅ Correct!" if spec1 == "general" else "  ❌ Incorrect!")
    
    # Test 2: Two workers
    print("\nTest 2: Two worker configuration")
    orchestrator2 = ContinuousOrchestrator(max_workers=2)
    spec2_0 = orchestrator2._assign_worker_specialization(0)
    spec2_1 = orchestrator2._assign_worker_specialization(1)
    print(f"  Worker 0 specialization: {spec2_0}")
    print(f"  Worker 1 specialization: {spec2_1}")
    print(f"  Expected: system_tasks and repository name")
    
    # Test 3: Run actual orchestrator for 20 seconds
    print("\nTest 3: Running orchestrator with 1 worker for 20 seconds...")
    orchestrator = ContinuousOrchestrator(max_workers=1)
    
    # Run in background
    run_task = asyncio.create_task(orchestrator.start())
    
    # Wait 20 seconds
    await asyncio.sleep(20)
    
    # Get status
    status = orchestrator.get_status()
    print(f"\nOrchestrator Status:")
    print(f"  Running: {status['running']}")
    print(f"  Workers: {len(status['workers'])}")
    print(f"  Work completed: {status['metrics']['total_work_completed']}")
    print(f"  Work created: {status['metrics']['total_work_created']}")
    print(f"  Errors: {status['metrics']['total_errors']}")
    print(f"  Queue size: {status['work_queue_size']}")
    
    # Worker details
    for worker_id, worker_info in status['workers'].items():
        print(f"\n  Worker {worker_id}:")
        print(f"    Status: {worker_info['status']}")
        print(f"    Specialization: {worker_info['specialization']}")
        print(f"    Completed: {worker_info['total_completed']}")
        print(f"    Errors: {worker_info['total_errors']}")
        print(f"    Current work: {worker_info['current_work']}")
    
    # Stop orchestrator
    await orchestrator.stop()
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_fixed_execution())