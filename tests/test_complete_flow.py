#!/usr/bin/env python3
"""Test complete work flow from discovery to execution."""

import asyncio
import sys
import logging
from datetime import datetime, timezone

sys.path.insert(0, '/workspaces/cwmai/scripts')

from continuous_orchestrator import ContinuousOrchestrator

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

async def test_complete_flow():
    """Test the complete flow."""
    print("=== TESTING COMPLETE WORK FLOW ===\n")
    
    # Create orchestrator with 1 worker
    orchestrator = ContinuousOrchestrator(max_workers=1)
    
    # Run orchestrator for 30 seconds
    print("Running orchestrator for 30 seconds...")
    run_task = asyncio.create_task(orchestrator.start())
    
    # Monitor progress
    for i in range(6):  # Check every 5 seconds for 30 seconds
        await asyncio.sleep(5)
        
        status = orchestrator.get_status()
        print(f"\n[{i*5+5}s] Status check:")
        print(f"  Workers: {len(status['workers'])}")
        print(f"  Queue size: {status['work_queue_size']}")
        print(f"  Work completed: {status['metrics']['total_work_completed']}")
        print(f"  Work created: {status['metrics']['total_work_created']}")
        
        # Check worker details
        for worker_id, worker_info in status['workers'].items():
            print(f"  {worker_id}: {worker_info['status']} - "
                  f"completed: {worker_info['total_completed']}, "
                  f"current: {worker_info['current_work']}")
    
    # Stop orchestrator
    print("\nStopping orchestrator...")
    await orchestrator.stop()
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass
    
    # Final status
    status = orchestrator.get_status()
    print("\nFinal Status:")
    print(f"  Total work completed: {status['metrics']['total_work_completed']}")
    print(f"  Total work created: {status['metrics']['total_work_created']}")
    print(f"  Total errors: {status['metrics']['total_errors']}")
    print(f"  Redis integration score: {status['redis_integration_score']}%")
    
    # Task persistence stats
    if 'task_completion_stats' in status:
        stats = status['task_completion_stats']
        print(f"\nTask Persistence Stats:")
        print(f"  Total completed: {stats.get('total_completed', 0)}")
        print(f"  Total skipped: {stats.get('total_skipped', 0)}")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_complete_flow())