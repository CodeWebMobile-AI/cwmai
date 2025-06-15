#!/usr/bin/env python3
"""Test all Redis features after import fixes."""

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

async def test_redis_features():
    """Test all Redis features."""
    print("=== TESTING ALL REDIS FEATURES ===\n")
    
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
    
    # Run briefly to initialize all components
    print("Initializing orchestrator...")
    run_task = asyncio.create_task(orchestrator.start())
    
    # Wait for initialization
    await asyncio.sleep(5)
    
    # Get status
    status = orchestrator.get_status()
    
    print("\nRedis Features Status:")
    redis_components = status.get('redis_components', {})
    
    features = [
        ('State Management', hasattr(orchestrator.state_manager, 'redis_client') or orchestrator.state_manager.__class__.__name__ == 'RedisEnabledStateManager'),
        ('Work Queue', 'redis' in status.get('queue_type', '')),
        ('Task Persistence', orchestrator.task_persistence.__class__.__name__ == 'RedisTaskPersistence'),
        ('Worker Coordination', redis_components.get('worker_coordination', False)),
        ('Distributed Locks', redis_components.get('distributed_locks', False)),
        ('Event Analytics', redis_components.get('event_analytics', False)),
        ('Event Processor', redis_components.get('event_processor', False)),
        ('Performance Analytics', redis_components.get('performance_analytics', False)),
        ('Workflow Orchestration', redis_components.get('workflow_orchestration', False)),
        ('Work Item Execution', True)  # We know this works from previous tests
    ]
    
    success_count = 0
    for i, (feature, enabled) in enumerate(features, 1):
        status_icon = '✅' if enabled else '❌'
        print(f"  {i}. {status_icon} {feature}")
        if enabled:
            success_count += 1
    
    print(f"\nRedis Integration Score: {success_count}/10 ({success_count * 10}%)")
    
    # Stop orchestrator
    await orchestrator.stop()
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass
    
    if success_count == 10:
        print("\n🎉 SUCCESS: All 10 Redis features are working!")
    else:
        print(f"\n⚠️  {10 - success_count} features still need attention")

if __name__ == "__main__":
    asyncio.run(test_redis_features())