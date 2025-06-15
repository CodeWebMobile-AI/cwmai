#!/usr/bin/env python3
"""Test complete execution flow."""

import asyncio
import sys
import logging

sys.path.insert(0, '/workspaces/cwmai/scripts')

from continuous_orchestrator import ContinuousOrchestrator

# Set up logging  
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set specific loggers to DEBUG for worker and queue
logging.getLogger('continuous_orchestrator').setLevel(logging.DEBUG) 
logging.getLogger('redis_work_queue').setLevel(logging.DEBUG)

async def test_execution():
    """Test work execution."""
    print("=== TESTING COMPLETE EXECUTION FLOW ===\n")
    
    # Clear Redis queues first
    from redis_integration.redis_client import RedisClient
    redis_client = RedisClient()
    await redis_client.connect()
    
    streams = [
        "cwmai:work_queue:critical",
        "cwmai:work_queue:high", 
        "cwmai:work_queue:medium",
        "cwmai:work_queue:low",
        "cwmai:work_queue:background"
    ]
    
    for stream in streams:
        await redis_client.delete(stream)
    
    print("Cleared Redis queues\n")
    
    # Create orchestrator
    orchestrator = ContinuousOrchestrator(max_workers=1)
    
    # Run for 30 seconds
    print("Running orchestrator for 30 seconds...")
    run_task = asyncio.create_task(orchestrator.start())
    
    # Monitor execution
    for i in range(6):  # 30 seconds total
        await asyncio.sleep(5)
        
        status = orchestrator.get_status()
        print(f"\n[{(i+1)*5}s] Status:")
        print(f"  Work created: {status['metrics']['total_work_created']}")
        print(f"  Work completed: {status['metrics']['total_work_completed']}")
        print(f"  Queue size: {status['work_queue_size']}")
        
        # Check worker status
        for worker_id, worker_info in status['workers'].items():
            print(f"  {worker_id}: {worker_info['status']} - "
                  f"completed: {worker_info['total_completed']}, "
                  f"current: {worker_info['current_work']}")
    
    # Stop
    await orchestrator.stop()
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass
    
    # Final status
    status = orchestrator.get_status()
    print(f"\n\nFinal Results:")
    print(f"  Total work created: {status['metrics']['total_work_created']}")
    print(f"  Total work completed: {status['metrics']['total_work_completed']}")
    print(f"  Total errors: {status['metrics']['total_errors']}")
    
    if status['metrics']['total_work_completed'] > 0:
        print("\n✅ SUCCESS: Work was executed!")
    else:
        print("\n❌ FAILURE: No work was executed")
        
        # Debug: Check Redis directly
        print("\nDebug - checking Redis streams:")
        for stream in streams:
            try:
                info = await redis_client.xinfo_stream(stream)
                print(f"  {stream}: {info.get('length', 0)} messages")
            except:
                pass

if __name__ == "__main__":
    asyncio.run(test_execution())