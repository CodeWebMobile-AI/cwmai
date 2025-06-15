#!/usr/bin/env python3
"""Detailed test of work item execution flow."""

import asyncio
import logging
import sys
from datetime import datetime, timezone

sys.path.insert(0, '/workspaces/cwmai/scripts')

from work_item_types import WorkItem, TaskPriority
from redis_work_queue import RedisWorkQueue
from redis_integration.redis_client import RedisClient
from continuous_orchestrator import ContinuousOrchestrator

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_execution_flow():
    """Test the complete work item execution flow with detailed logging."""
    print("=== DETAILED WORK EXECUTION TEST ===\n")
    
    # Step 1: Initialize Redis and Work Queue
    print("Step 1: Initializing Redis and Work Queue...")
    redis_client = RedisClient()
    await redis_client.connect()
    print("✅ Redis connected")
    
    work_queue = RedisWorkQueue(redis_client=redis_client)
    await work_queue.initialize()
    print("✅ Work queue initialized")
    
    # Step 2: Clear any existing work
    print("\nStep 2: Clearing existing work items...")
    for stream in work_queue.priority_streams.values():
        try:
            # Delete the stream to clear all items
            await redis_client.delete(stream)
            print(f"  Cleared stream: {stream}")
        except:
            pass
    
    # Re-create consumer groups after clearing
    print("  Re-creating consumer groups...")
    for priority, stream in work_queue.priority_streams.items():
        await work_queue._ensure_stream_and_group(stream)
        print(f"  Created group for {priority.name}")
    
    # Step 3: Add test work items
    print("\nStep 3: Adding test work items...")
    test_items = [
        WorkItem(
            id=f"exec_test_{i}",
            task_type="TEST_EXECUTION",
            title=f"Test execution item {i}",
            description=f"Testing execution flow for item {i}",
            priority=TaskPriority.HIGH,
            repository=None,
            estimated_cycles=1,
            created_at=datetime.now(timezone.utc)
        )
        for i in range(3)
    ]
    
    for item in test_items:
        await work_queue.add_work(item)
        print(f"  Added: {item.id} - {item.title}")
    
    # Force flush
    await work_queue._flush_buffer()
    print("✅ Work items flushed to Redis")
    
    # Step 4: Verify items are in Redis
    print("\nStep 4: Verifying items in Redis streams...")
    stream_name = work_queue.priority_streams[TaskPriority.HIGH]
    info = await redis_client.xinfo_stream(stream_name)
    print(f"  Stream {stream_name}:")
    print(f"    Length: {info.get('length', 0)}")
    print(f"    Groups: {info.get('groups', 0)}")
    first_entry = info.get('first-entry')
    if isinstance(first_entry, (list, tuple)) and len(first_entry) > 0:
        print(f"    First entry ID: {first_entry[0]}")
    else:
        print(f"    First entry: {first_entry}")
    
    # Step 5: Test worker retrieval
    print("\nStep 5: Testing worker retrieval...")
    print("  Attempting to get work for worker 'test_worker_1'...")
    
    # Get work with detailed logging
    retrieved = await work_queue.get_work_for_worker(
        worker_id="test_worker_1",
        specialization="general",
        count=2
    )
    
    print(f"  Retrieved {len(retrieved)} items:")
    for item in retrieved:
        print(f"    - {item.id}: {item.title}")
        print(f"      Assigned to: {item.assigned_worker}")
        print(f"      Started at: {item.started_at}")
    
    # Step 6: Check what's left in the queue
    print("\nStep 6: Checking remaining items in queue...")
    info_after = await redis_client.xinfo_stream(stream_name)
    print(f"  Stream length after retrieval: {info_after.get('length', 0)}")
    
    # Step 7: Check pending messages
    print("\nStep 7: Checking pending messages...")
    pending_info = await redis_client.xpending(stream_name, "cwmai_workers")
    print(f"  Pending messages info: {pending_info}")
    
    # Step 8: Test the orchestrator's _find_work_for_worker method
    print("\nStep 8: Testing orchestrator's work finding...")
    orchestrator = ContinuousOrchestrator(max_workers=1)
    orchestrator.redis_work_queue = work_queue
    orchestrator.use_redis_queue = True
    
    # Create a mock worker
    from continuous_orchestrator import WorkerState, WorkerStatus
    test_worker = WorkerState(
        id="test_worker_2",
        status=WorkerStatus.IDLE,
        specialization="general"
    )
    
    work_for_worker = await orchestrator._find_work_for_worker(test_worker)
    if work_for_worker:
        print(f"  Orchestrator found work: {work_for_worker.id} - {work_for_worker.title}")
    else:
        print("  Orchestrator found no work!")
    
    # Step 9: Test the complete execution
    print("\nStep 9: Testing complete work execution...")
    if work_for_worker:
        print(f"  Executing work item: {work_for_worker.id}")
        try:
            result = await orchestrator._perform_work(work_for_worker)
            print(f"  Execution result: {result}")
        except Exception as e:
            print(f"  Execution error: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 10: Final queue status
    print("\nStep 10: Final queue status...")
    final_stats = await work_queue.get_queue_stats()
    print(f"  Total queued: {final_stats['total_queued']}")
    print(f"  Total pending: {final_stats['total_pending']}")
    for priority, info in final_stats['priority_queues'].items():
        if info.get('length', 0) > 0 or info.get('pending', 0) > 0:
            print(f"  {priority}: length={info.get('length', 0)}, pending={info.get('pending', 0)}")
    
    # Clean up
    await redis_client.disconnect()
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_execution_flow())