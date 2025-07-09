#!/usr/bin/env python3
"""
Simple test to check if work is being queued and picked up.
"""

import asyncio
import os
import time

# Set quick intervals for testing
os.environ['DISCOVERY_INTERVAL_SECONDS'] = '10'

async def test_work_queue():
    """Test work queue functionality."""
    from scripts.hybrid_orchestrator_enhanced import EnhancedHybridOrchestrator
    
    print("Testing work queue system...")
    print("=" * 60)
    
    # Create orchestrator with 3 workers
    orchestrator = EnhancedHybridOrchestrator(
        max_workers=3,
        enable_parallel=True
    )
    
    # Start orchestrator
    start_task = asyncio.create_task(orchestrator.start())
    
    # Run for 30 seconds
    print("\nRunning for 30 seconds...")
    
    for i in range(6):  # 6 x 5 seconds = 30 seconds
        await asyncio.sleep(5)
        
        # Get status
        status = await orchestrator.get_status()
        
        workers = status.get('workers', {})
        metrics = status.get('metrics', {})
        queue = status.get('work_queue', {})
        
        print(f"\n[{i*5}s] Status:")
        print(f"  Workers: {workers.get('working', 0)}/{workers.get('total', 0)} working")
        print(f"  Queue: {queue.get('queued', 0)} queued, {queue.get('processing', 0)} processing")
        print(f"  Completed: {metrics.get('total_work_completed', 0)}")
        
        # Check if work discovery happened
        if hasattr(orchestrator, 'work_finder') and hasattr(orchestrator.work_finder, 'discovered_work'):
            print(f"  Discovered work: {len(orchestrator.work_finder.discovered_work)} items")
    
    # Stop
    await orchestrator.stop()
    start_task.cancel()
    try:
        await start_task
    except asyncio.CancelledError:
        pass
    
    # Final summary
    final_status = await orchestrator.get_status()
    print("\n" + "=" * 60)
    print("Final Summary:")
    print(f"  Total completed: {final_status['metrics']['total_work_completed']}")
    print(f"  Workers that were active: {final_status['workers']['working']}")
    
    return final_status['metrics']['total_work_completed'] > 0

if __name__ == "__main__":
    success = asyncio.run(test_work_queue())
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: Work {'was' if success else 'was NOT'} processed")
    exit(0 if success else 1)