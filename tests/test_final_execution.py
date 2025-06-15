#!/usr/bin/env python3
"""Final test of work execution."""

import asyncio
import sys
import logging

sys.path.insert(0, '/workspaces/cwmai/scripts')

from continuous_orchestrator import ContinuousOrchestrator
from redis_integration.redis_client import RedisClient

# Set up logging  
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_execution():
    """Test work execution."""
    print("=== FINAL EXECUTION TEST ===\n")
    
    # Clear Redis queues
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
    
    # Run for 60 seconds
    print("Running orchestrator for 60 seconds...")
    run_task = asyncio.create_task(orchestrator.start())
    
    # Wait for work to be discovered (usually happens around 15s)
    await asyncio.sleep(20)
    
    # Now check work queue
    print("\nChecking Redis work queues after 20s:")
    for stream in streams:
        try:
            info = await redis_client.xinfo_stream(stream)
            length = info.get('length', 0)
            if length > 0:
                print(f"  {stream}: {length} messages")
                # Check pending
                pending = await redis_client.xpending(stream, "cwmai_workers")
                if pending:
                    print(f"    Pending: {pending}")
        except:
            pass
    
    # Check orchestrator status
    status = orchestrator.get_status()
    print(f"\nOrchestrator status at 20s:")
    print(f"  Work created: {status['metrics']['total_work_created']}")
    print(f"  Work completed: {status['metrics']['total_work_completed']}")
    print(f"  Queue size: {status['work_queue_size']}")
    
    # Wait another 20 seconds
    await asyncio.sleep(20)
    
    # Check again
    print(f"\nOrchestrator status at 40s:")
    status = orchestrator.get_status()
    print(f"  Work created: {status['metrics']['total_work_created']}")
    print(f"  Work completed: {status['metrics']['total_work_completed']}")
    print(f"  Queue size: {status['work_queue_size']}")
    
    # Worker status
    for worker_id, worker_info in status['workers'].items():
        print(f"  {worker_id}: {worker_info['status']} - "
              f"completed: {worker_info['total_completed']}, "
              f"errors: {worker_info['total_errors']}")
    
    # Stop
    await orchestrator.stop()
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass
    
    # Final check
    print(f"\n\nFinal Results:")
    status = orchestrator.get_status()
    print(f"  Total work created: {status['metrics']['total_work_created']}")
    print(f"  Total work completed: {status['metrics']['total_work_completed']}")
    print(f"  Total errors: {status['metrics']['total_errors']}")
    print(f"  Redis integration score: {status['redis_integration_score']}%")
    
    # Check Redis features
    print(f"\nRedis Features Status:")
    redis_components = status.get('redis_components', {})
    for feature, enabled in redis_components.items():
        print(f"  {feature}: {'✅' if enabled else '❌'}")
    
    if status['metrics']['total_work_completed'] > 0:
        print("\n✅ SUCCESS: Work was executed!")
    else:
        print("\n❌ FAILURE: No work was executed")
        
        # Final Redis check
        print("\nFinal Redis queue status:")
        for stream in streams:
            try:
                info = await redis_client.xinfo_stream(stream)
                print(f"  {stream}: {info.get('length', 0)} messages")
            except:
                pass

if __name__ == "__main__":
    asyncio.run(test_execution())