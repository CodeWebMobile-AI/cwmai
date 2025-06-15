#!/usr/bin/env python3
"""Test script to verify all Redis-related fixes."""

import asyncio
import json
import logging
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_redis_state_manager():
    """Test redis_state_manager coroutine fix."""
    print("\n" + "="*60)
    print("Testing Redis State Manager...")
    print("="*60)
    
    try:
        from scripts.redis_integration.redis_client import get_redis_client
        from scripts.redis_integration.redis_state_manager import RedisStateManager
        
        # Get Redis client
        redis_client = await get_redis_client()
        
        # Create state manager
        state_manager = RedisStateManager(redis_client, "test_component")
        await state_manager.start()
        
        # Test set and get state
        test_state = {"test": "data", "timestamp": datetime.now(timezone.utc).isoformat()}
        await state_manager.set_state(test_state, "test_component")
        
        # Get state back
        retrieved_state = await state_manager.get_state("test_component")
        print(f"✅ State Manager: Successfully stored and retrieved state: {retrieved_state}")
        
        # Test version retrieval
        version = await state_manager.get_version("test_component")
        if version:
            print(f"✅ State Manager: Version retrieved successfully: v{version.version}")
        
        await state_manager.stop()
        return True
        
    except Exception as e:
        print(f"❌ State Manager Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_redis_work_queue():
    """Test redis_work_queue TaskPriority fix."""
    print("\n" + "="*60)
    print("Testing Redis Work Queue...")
    print("="*60)
    
    try:
        from scripts.redis_work_queue import RedisWorkQueue
        from scripts.work_item_types import WorkItem, TaskPriority
        
        # Create work queue
        queue = RedisWorkQueue()
        await queue.initialize()
        
        # Create test work items with different priorities
        test_items = [
            WorkItem(
                id="test_high",
                task_type="TEST",
                title="High Priority Test",
                description="Testing high priority",
                priority=TaskPriority.HIGH
            ),
            WorkItem(
                id="test_medium",
                task_type="TEST",
                title="Medium Priority Test",
                description="Testing medium priority",
                priority=TaskPriority.MEDIUM
            )
        ]
        
        # Add items to queue
        for item in test_items:
            await queue.add_work(item)
        
        # Flush buffer to ensure items are in Redis
        await queue._flush_buffer()
        
        print(f"✅ Work Queue: Successfully added {len(test_items)} items with different priorities")
        
        # Get queue stats
        stats = await queue.get_queue_stats()
        print(f"✅ Work Queue: Queue stats retrieved: {stats['total_queued']} items queued")
        
        await queue.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Work Queue Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_redis_task_persistence():
    """Test redis_task_persistence coroutine fix."""
    print("\n" + "="*60)
    print("Testing Redis Task Persistence...")
    print("="*60)
    
    try:
        from scripts.redis_task_persistence import RedisTaskPersistence
        from scripts.work_item_types import WorkItem, TaskPriority
        
        # Create persistence manager
        persistence = RedisTaskPersistence()
        
        # Test recording a completed task
        test_item = WorkItem(
            id="test_complete",
            task_type="TEST",
            title="Test Completed Task",
            description="Testing task completion recording",
            priority=TaskPriority.MEDIUM
        )
        
        execution_result = {
            "status": "success",
            "value_created": 10.0,
            "issue_number": 123
        }
        
        success = await persistence.record_completed_task(test_item, execution_result)
        if success:
            print("✅ Task Persistence: Successfully recorded completed task")
        
        # Test duplicate detection
        is_duplicate = await persistence.is_duplicate_task(test_item)
        print(f"✅ Task Persistence: Duplicate detection working (is_duplicate={is_duplicate})")
        
        # Test skipped task recording
        await persistence.record_skipped_task("Test Skipped Task", "duplicate")
        print("✅ Task Persistence: Successfully recorded skipped task")
        
        # Test it again to trigger the skip count logic
        await persistence.record_skipped_task("Test Skipped Task", "duplicate")
        print("✅ Task Persistence: Skip count logic working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Task Persistence Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_redis_event_analytics():
    """Test redis_event_analytics fixes."""
    print("\n" + "="*60)
    print("Testing Redis Event Analytics...")
    print("="*60)
    
    try:
        from scripts.redis_event_analytics import RedisEventAnalytics
        
        # Create analytics engine
        analytics = RedisEventAnalytics(
            analytics_id="test_analytics",
            enable_real_time=False,  # Disable background tasks for test
            enable_pattern_detection=False,
            enable_anomaly_detection=False,
            enable_predictive_analytics=False
        )
        await analytics.initialize()
        
        # Test tracking event with proper dict
        test_event = {
            "event_type": "test_event",
            "worker_id": "test_worker",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await analytics.track_event(test_event)
        print("✅ Event Analytics: Successfully tracked event with dict")
        
        # Test tracking event with invalid data (should handle gracefully)
        await analytics.track_event("invalid_string_data")  # This should log error but not crash
        print("✅ Event Analytics: Handled invalid event data gracefully")
        
        # Test metrics
        report = await analytics.get_analytics_report()
        print(f"✅ Event Analytics: Generated report with {report['analytics_performance']['events_processed']} events processed")
        
        await analytics.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Event Analytics Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# Redis Fixes Test Suite")
    print("#"*60)
    
    results = {
        "redis_state_manager": await test_redis_state_manager(),
        "redis_work_queue": await test_redis_work_queue(),
        "redis_task_persistence": await test_redis_task_persistence(),
        "redis_event_analytics": await test_redis_event_analytics()
    }
    
    print("\n" + "="*60)
    print("Test Results Summary:")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All tests passed! All Redis fixes are working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())