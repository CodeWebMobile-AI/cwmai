#!/usr/bin/env python3
"""Test script to verify worker monitor fix."""

import asyncio
import sys
import os
import logging
from datetime import datetime, timezone

# Add scripts directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.worker_status_monitor import WorkerStatusMonitor
from scripts.redis_lockfree_state_manager import RedisLockFreeStateManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def simulate_worker_for_test():
    """Create a test worker in Redis to verify the fix."""
    manager = RedisLockFreeStateManager()
    await manager.initialize()
    
    worker_id = "test-monitor-worker"
    
    try:
        # Add worker to active set
        await manager.add_to_set("active_workers", worker_id)
        
        # Create worker state
        worker_state = {
            'status': 'working',
            'specialization': 'general',
            'started_at': datetime.now(timezone.utc).isoformat(),
            'last_activity': datetime.now(timezone.utc).isoformat(),
            'total_completed': 10,
            'total_errors': 2,
            'current_task': {
                'id': 'test-123',
                'title': 'Test task for monitor verification',
                'task_type': 'FEATURE',
                'repository': 'test/repo'
            }
        }
        await manager.update_worker_state(worker_id, worker_state)
        
        # Set counters
        await manager.increment_counter(f"worker:{worker_id}:completed", 10)
        await manager.increment_counter(f"worker:{worker_id}:errors", 2)
        
        logger.info(f"Created test worker: {worker_id}")
        return worker_id
        
    except Exception as e:
        logger.error(f"Error creating test worker: {e}")
        raise
    finally:
        await manager.close()


async def test_worker_monitor():
    """Test the worker monitor to ensure it reads from correct keys."""
    print("Testing Worker Monitor Fix...")
    
    monitor = WorkerStatusMonitor()
    
    try:
        # Initialize the monitor
        print("1. Initializing monitor...")
        await monitor.initialize()
        print("   ✓ Monitor initialized successfully")
        
        # Create a test worker
        print("\n2. Creating test worker...")
        test_worker_id = await simulate_worker_for_test()
        print(f"   ✓ Test worker created: {test_worker_id}")
        
        # Get worker status
        print("\n3. Getting worker status...")
        status = await monitor.get_worker_status()
        print(f"   ✓ Status retrieved: {status is not None}")
        
        # Test data structure
        print("\n4. Testing data structure...")
        assert isinstance(status, dict), "Status should be a dict"
        assert 'workers' in status, "Status should have 'workers' key"
        assert 'queue_status' in status, "Status should have 'queue_status' key"
        assert 'system_health' in status, "Status should have 'system_health' key"
        assert 'active_tasks' in status, "Status should have 'active_tasks' key"
        print("   ✓ Data structure is correct")
        
        # Check if test worker is found
        print("\n5. Verifying test worker is found...")
        workers = status.get('workers', {})
        if test_worker_id in workers:
            print(f"   ✓ Test worker found in monitor!")
            worker_data = workers[test_worker_id]
            print(f"     - Status: {worker_data.get('status')}")
            print(f"     - Current task: {worker_data.get('current_task', {}).get('title', 'None')}")
            print(f"     - Completed: {worker_data.get('total_completed')}")
            print(f"     - Errors: {worker_data.get('total_errors')}")
            
            # Verify data matches what we set
            assert worker_data.get('status') == 'working', f"Expected status 'working', got {worker_data.get('status')}"
            assert worker_data.get('total_completed') == 10, f"Expected 10 completed, got {worker_data.get('total_completed')}"
            assert worker_data.get('current_task') is not None, "Current task should not be None"
            print("   ✓ Worker data matches expected values!")
        else:
            print(f"   ❌ Test worker NOT found! Available workers: {list(workers.keys())}")
            raise AssertionError("Test worker not found in monitor output")
        
        # Test queue_status structure
        print("\n6. Testing queue_status structure...")
        queue_status = status.get('queue_status', {})
        if queue_status:
            # Check for correct keys
            expected_keys = ['total_queued', 'total_processing', 'total_completed', 'total_failed']
            for key in expected_keys:
                if key in queue_status:
                    print(f"   ✓ Found key: {key} = {queue_status[key]}")
            
            # Check priority breakdown
            if 'by_priority' in queue_status:
                print(f"   ✓ Priority breakdown: {queue_status['by_priority']}")
        
        # Test workers structure
        print("\n5. Testing workers structure...")
        workers = status.get('workers', {})
        print(f"   Found {len(workers)} workers")
        
        for worker_id, worker_data in workers.items():
            if worker_data:
                print(f"   Worker {worker_id}:")
                print(f"     - status: {worker_data.get('status', 'unknown')}")
                print(f"     - total_completed: {worker_data.get('total_completed', 0)}")
                print(f"     - specialization: {worker_data.get('specialization', 'general')}")
        
        # Test active_tasks structure
        print("\n6. Testing active_tasks structure...")
        active_tasks = status.get('active_tasks', [])
        print(f"   ✓ active_tasks is a list: {isinstance(active_tasks, list)}")
        print(f"   Found {len(active_tasks)} active tasks")
        
        print("\n✅ All tests passed! Worker monitor is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        await monitor.close()
        
        # Clean up test worker
        manager = RedisLockFreeStateManager()
        await manager.initialize()
        await manager.remove_from_set("active_workers", "test-monitor-worker")
        await manager.close()
        print("\n   ✓ Cleanup completed")
    
    return True


async def main():
    """Main test runner."""
    success = await test_worker_monitor()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Use a simple event loop to avoid nested loop issues
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()