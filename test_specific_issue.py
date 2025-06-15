#!/usr/bin/env python3
"""Test specific Redis state manager issue."""

import asyncio
import logging

# Set up more detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_state_manager_issue():
    """Test the specific coroutine issue in state manager."""
    try:
        from scripts.redis_integration.redis_client import get_redis_client
        from scripts.redis_integration.redis_state_manager import RedisStateManager
        
        print("Getting Redis client...")
        redis_client = await get_redis_client()
        
        print("Creating state manager...")
        state_manager = RedisStateManager(redis_client, "test_issue")
        
        # Don't start background tasks to isolate the issue
        state_manager.auto_sync = False
        state_manager.enable_change_tracking = False
        
        print("Testing get_state (should return None for non-existent)...")
        state = await state_manager.get_state("test_issue")
        print(f"Initial state: {state}")
        
        print("Setting state...")
        test_data = {"test": "data"}
        success = await state_manager.set_state(test_data, "test_issue")
        print(f"Set state result: {success}")
        
        if success:
            print("Getting state back...")
            retrieved = await state_manager.get_state("test_issue")
            print(f"Retrieved state: {retrieved}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_state_manager_issue())