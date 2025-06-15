#!/usr/bin/env python3
"""Test the patched Redis client."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.redis_integration import get_redis_client


async def test_patches():
    """Test that patches work correctly."""
    print("Testing Redis client patches...")
    
    redis_client = await get_redis_client()
    
    # Test 1: Get non-existent key should return None without error
    result = await redis_client.get("definitely:does:not:exist:key:12345")
    assert result is None, f"Expected None, got {result}"
    print("✓ Non-existent key returns None")
    
    # Test 2: Set and get a value
    test_key = "patch:test:key"
    await redis_client.set(test_key, "test_value")
    result = await redis_client.get(test_key)
    assert result == "test_value", f"Expected 'test_value', got {result}"
    print("✓ Set/Get works correctly")
    
    # Clean up
    await redis_client.delete(test_key)
    
    print("\nAll patch tests passed!")


if __name__ == "__main__":
    asyncio.run(test_patches())
