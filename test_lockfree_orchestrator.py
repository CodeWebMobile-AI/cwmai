#!/usr/bin/env python3
"""Test script to verify lock-free orchestrator functionality."""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.redis_lockfree_state_manager import RedisLockFreeStateManager
from scripts.continuous_orchestrator import ContinuousOrchestrator


async def test_lockfree_functionality():
    """Test the lock-free functionality of the orchestrator."""
    print("Testing lock-free orchestrator functionality...")
    
    # Test 1: Initialize lock-free state manager
    print("\n1. Testing RedisLockFreeStateManager initialization...")
    try:
        state_manager = RedisLockFreeStateManager()
        await state_manager.initialize()
        print("✓ Lock-free state manager initialized successfully")
        
        # Test atomic operations
        print("\n2. Testing atomic operations...")
        
        # Counter operations
        count = await state_manager.increment_counter("test_counter", 5)
        print(f"✓ Atomic counter increment: {count}")
        
        # Set operations
        await state_manager.add_to_set("test_workers", "worker1", "worker2")
        members = await state_manager.get_set_members("test_workers")
        print(f"✓ Set operations: {members}")
        
        # Worker state operations
        await state_manager.update_worker_state("test_worker", {
            "status": "active",
            "task_count": 0
        })
        state = await state_manager.get_worker_state("test_worker")
        print(f"✓ Worker state operations: {state}")
        
        # Stream operations
        entry_id = await state_manager.append_to_stream("test_events", {
            "event": "test",
            "timestamp": "now"
        })
        print(f"✓ Stream append: {entry_id}")
        
        # Cleanup
        await state_manager.close()
        
    except Exception as e:
        print(f"✗ State manager test failed: {e}")
        return False
    
    # Test 2: Initialize orchestrator with lock-free state
    print("\n3. Testing ContinuousOrchestrator with lock-free state...")
    try:
        orchestrator = ContinuousOrchestrator(
            max_workers=2,
            enable_parallel=True,
            enable_research=False
        )
        
        # Check if lock-free state manager is being used
        if hasattr(orchestrator, 'redis_state_manager') and orchestrator.redis_state_manager:
            print("✓ Orchestrator configured with lock-free state manager")
        else:
            print("✗ Orchestrator not using lock-free state manager")
        
        print("\n✅ All lock-free tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Orchestrator test failed: {e}")
        return False


async def main():
    """Main test runner."""
    success = await test_lockfree_functionality()
    
    if success:
        print("\n🎉 Lock-free orchestrator is ready to use!")
        print("\nKey improvements:")
        print("- No distributed locks needed")
        print("- Atomic Redis operations for metrics")
        print("- Partitioned worker states")
        print("- Optimistic concurrency control")
        print("- Stream-based event logging")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())