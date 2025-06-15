#!/usr/bin/env python3
"""Test script to reproduce and fix the coroutine JSON serialization issue."""

import asyncio
import json
import logging
from scripts.redis_integration.redis_client import RedisClient, get_redis_client
from scripts.redis_integration.redis_state_manager import RedisStateManager
from scripts.redis_state_adapter import RedisStateAdapter
from scripts.state_manager import StateManager

logging.basicConfig(level=logging.DEBUG)

async def test_redis_state_manager():
    """Test Redis state manager for coroutine issues."""
    print("Testing Redis state manager...")
    
    try:
        # Get Redis client
        redis_client = await get_redis_client()
        
        # Create state manager
        redis_state = RedisStateManager(redis_client, "test_component")
        await redis_state.start()
        
        # Test get_state
        print("\nTesting get_state...")
        state = await redis_state.get_state("test_component")
        print(f"State: {state}")
        
        # Test get_version
        print("\nTesting get_version...")
        version = await redis_state.get_version("test_component")
        print(f"Version: {version}")
        
        # Test set_state
        print("\nTesting set_state...")
        test_data = {"test": "data", "timestamp": "2025-06-12T12:00:00"}
        result = await redis_state.set_state(test_data, "test_component")
        print(f"Set state result: {result}")
        
        # Test get_state again
        print("\nTesting get_state after set...")
        state = await redis_state.get_state("test_component")
        print(f"State after set: {state}")
        
        await redis_state.stop()
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()

async def test_redis_state_adapter():
    """Test Redis state adapter for coroutine issues."""
    print("\n\nTesting Redis state adapter...")
    
    try:
        # Create local state manager
        state_manager = StateManager("test_state.json")
        
        # Create Redis adapter
        adapter = RedisStateAdapter(state_manager, component_id="cwmai_orchestrator")
        await adapter.initialize()
        
        # Test get version
        print("\nTesting adapter operations...")
        local_state = adapter.get_state()
        print(f"Local state: {local_state}")
        
        # Clean up
        await adapter.cleanup()
        print("\nAdapter test completed successfully!")
        
    except Exception as e:
        print(f"\nError during adapter test: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main test function."""
    await test_redis_state_manager()
    await test_redis_state_adapter()

if __name__ == "__main__":
    asyncio.run(main())