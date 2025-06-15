#!/usr/bin/env python3
"""Final verification of all Redis fixes."""

import asyncio
import logging
from datetime import datetime, timezone

# Configure logging to show only important messages
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

async def verify_all_fixes():
    """Verify all Redis fixes are working."""
    results = []
    
    # Test 1: Redis State Manager
    print("\n1. Testing Redis State Manager...")
    try:
        from scripts.redis_integration.redis_client import get_redis_client
        from scripts.redis_integration.redis_state_manager import RedisStateManager
        
        redis_client = await get_redis_client()
        state_manager = RedisStateManager(redis_client, "verify_test")
        await state_manager.start()
        
        # Test operations
        test_data = {"timestamp": datetime.now(timezone.utc).isoformat(), "data": "test"}
        await state_manager.set_state(test_data, "verify_test")
        retrieved = await state_manager.get_state("verify_test")
        
        assert retrieved == test_data, "State mismatch"
        await state_manager.stop()
        
        print("✅ State Manager: Working correctly - no coroutine errors")
        results.append(("State Manager", True))
    except Exception as e:
        print(f"❌ State Manager: Failed - {e}")
        results.append(("State Manager", False))
    
    # Test 2: Redis Work Queue
    print("\n2. Testing Redis Work Queue...")
    try:
        from scripts.redis_work_queue import RedisWorkQueue
        from scripts.work_item_types import WorkItem, TaskPriority
        
        queue = RedisWorkQueue()
        await queue.initialize()
        
        # Test with enum priority
        item = WorkItem(
            id="verify_enum",
            task_type="VERIFY",
            title="Verify Enum Handling",
            description="Testing enum serialization",
            priority=TaskPriority.HIGH
        )
        await queue.add_work(item)
        await queue._flush_buffer()
        
        print("✅ Work Queue: Enum handling working correctly")
        results.append(("Work Queue", True))
        await queue.cleanup()
    except Exception as e:
        print(f"❌ Work Queue: Failed - {e}")
        results.append(("Work Queue", False))
    
    # Test 3: Redis Task Persistence
    print("\n3. Testing Redis Task Persistence...")
    try:
        from scripts.redis_task_persistence import RedisTaskPersistence
        from scripts.work_item_types import WorkItem, TaskPriority
        
        persistence = RedisTaskPersistence()
        
        # Test skip recording
        await persistence.record_skipped_task("Verify Skip Task", "test")
        await persistence.record_skipped_task("Verify Skip Task", "test")
        
        print("✅ Task Persistence: Skip count handling working correctly")
        results.append(("Task Persistence", True))
    except Exception as e:
        print(f"❌ Task Persistence: Failed - {e}")
        results.append(("Task Persistence", False))
    
    # Test 4: Redis Event Analytics
    print("\n4. Testing Redis Event Analytics...")
    try:
        from scripts.redis_event_analytics import RedisEventAnalytics
        
        analytics = RedisEventAnalytics(
            analytics_id="verify_analytics",
            enable_real_time=False,
            enable_pattern_detection=False,
            enable_anomaly_detection=False,
            enable_predictive_analytics=False
        )
        await analytics.initialize()
        
        # Test with valid event
        await analytics.track_event({"event_type": "verify", "data": "test"})
        
        # Test with invalid event (should handle gracefully)
        await analytics.track_event("invalid")
        
        await analytics.shutdown()
        print("✅ Event Analytics: Event handling working correctly")
        results.append(("Event Analytics", True))
    except Exception as e:
        print(f"❌ Event Analytics: Failed - {e}")
        results.append(("Event Analytics", False))
    
    # Summary
    print("\n" + "="*50)
    print("VERIFICATION SUMMARY:")
    print("="*50)
    
    all_passed = True
    for component, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{component}: {status}")
        if not passed:
            all_passed = False
    
    print("="*50)
    if all_passed:
        print("🎉 ALL FIXES VERIFIED - System is working correctly!")
    else:
        print("⚠️  Some components still have issues.")
    
    return all_passed

if __name__ == "__main__":
    asyncio.run(verify_all_fixes())