#!/usr/bin/env python3
"""Simple test to verify work execution."""

import asyncio
import sys
import logging

sys.path.insert(0, '/workspaces/cwmai/scripts')

from continuous_orchestrator import ContinuousOrchestrator

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set specific loggers to DEBUG
logging.getLogger('continuous_orchestrator').setLevel(logging.DEBUG)
logging.getLogger('redis_work_queue').setLevel(logging.DEBUG)

async def test_execution():
    """Test work execution."""
    print("=== TESTING WORK EXECUTION ===\n")
    
    # Create orchestrator
    orchestrator = ContinuousOrchestrator(max_workers=1)
    
    # Run for 15 seconds
    print("Running orchestrator for 15 seconds...")
    run_task = asyncio.create_task(orchestrator.start())
    
    await asyncio.sleep(15)
    
    # Stop and get final status
    await orchestrator.stop()
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass
    
    # Final status
    status = orchestrator.get_status()
    print(f"\nFinal Status:")
    print(f"  Work completed: {status['metrics']['total_work_completed']}")
    print(f"  Work created: {status['metrics']['total_work_created']}")
    print(f"  Errors: {status['metrics']['total_errors']}")
    
    if status['metrics']['total_work_completed'] > 0:
        print("\n✅ SUCCESS: Work was executed!")
    else:
        print("\n❌ FAILURE: No work was executed")

if __name__ == "__main__":
    asyncio.run(test_execution())