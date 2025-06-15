"""
Run the integration test separately to see the full output
"""
import asyncio
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from smart_redis_integration import SmartRedisIntegration
from work_item_types import WorkItem, TaskPriority


async def run_integration_test():
    """Run a full integration test"""
    print("\n=== Running Full Integration Test ===\n")
    
    # Create smart system
    integration = SmartRedisIntegration(None, num_workers=3)
    
    # Start system
    await integration.start(["system_tasks", "ai-creative-studio", None])
    
    print("✓ System started with 3 workers")
    
    # Create test work items
    test_items = [
        WorkItem(
            id=f"test_item_{i}",
            task_type="SYSTEM_IMPROVEMENT" if i % 2 == 0 else "NEW_PROJECT",
            title=f"Test task {i}",
            description=f"This is test task number {i}",
            priority=TaskPriority.HIGH if i < 3 else TaskPriority.MEDIUM,
            repository="ai-creative-studio" if i % 3 == 0 else None
        )
        for i in range(5)
    ]
    
    print(f"✓ Created {len(test_items)} test work items")
    
    # Process work items
    for item in test_items:
        await integration._process_work_item(item)
    
    # Wait for processing
    await asyncio.sleep(10)
    
    # Get final stats
    stats = integration.get_integration_stats()
    
    print("\n=== Final Statistics ===")
    print(f"Tasks processed: {stats['integration']['tasks_processed']}")
    print(f"Tasks succeeded: {stats['integration']['tasks_succeeded']}")
    print(f"Success rate: {stats['integration']['success_rate']:.2%}")
    
    if 'marketplace' in stats['orchestrator']:
        print(f"Active workers: {stats['orchestrator']['marketplace']['active_workers']}")
        print(f"Total auctions: {stats['orchestrator']['marketplace']['total_tasks_processed']}")
    
    # Print worker performance
    print("\n=== Worker Performance ===")
    if 'marketplace' in stats['orchestrator'] and 'worker_performance' in stats['orchestrator']['marketplace']:
        for worker_id, perf in stats['orchestrator']['marketplace']['worker_performance'].items():
            print(f"\n{worker_id}:")
            print(f"  - Specialization: {perf['specialization']}")
            print(f"  - Tasks completed: {perf['total_completed']}")
            print(f"  - Success rate: {perf['overall_success_rate']:.2%}")
            print(f"  - Current load: {perf['current_load']:.2%}")
    
    # Print detailed worker capabilities
    print("\n=== Worker Capabilities ===")
    if 'workers' in stats['orchestrator']:
        for worker_id, metrics in stats['orchestrator']['workers'].items():
            if metrics['capabilities']:
                print(f"\n{worker_id} capabilities:")
                for cap_key, cap_data in metrics['capabilities'].items():
                    print(f"  - {cap_key}: {cap_data['success_rate']:.2%} success, {cap_data['total_tasks']} tasks")
    
    # Stop system
    await integration.stop()
    
    print("\n✓ System stopped successfully")
    
    # Verify results
    assert stats['integration']['tasks_processed'] > 0
    print("\n✓ All integration tests passed!")


if __name__ == "__main__":
    asyncio.run(run_integration_test())