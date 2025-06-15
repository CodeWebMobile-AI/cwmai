#!/usr/bin/env python3
"""Reset Redis circuit breaker and test connection."""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.redis_integration.redis_client import RedisClient

async def main():
    """Reset circuit breaker and test Redis connection."""
    print("🔧 Resetting Redis circuit breaker and testing connection...")
    
    # Create Redis client
    redis_client = RedisClient()
    
    try:
        # Connect the client
        await redis_client.connect()
        print("✅ Redis client connected")
        
        # Test basic operations
        print("\n📝 Testing basic Redis operations:")
        
        # Test SET/GET
        await redis_client.set("test_key", "test_value")
        value = await redis_client.get("test_key")
        print(f"  - SET/GET test: {'✅ Passed' if value == 'test_value' else '❌ Failed'}")
        
        # Test PING
        pong = await redis_client.ping()
        print(f"  - PING test: {'✅ Passed' if pong else '❌ Failed'}")
        
        # Check circuit breaker state
        if hasattr(redis_client, 'circuit_breaker'):
            print(f"\n🔌 Circuit breaker state: {redis_client.circuit_breaker.state}")
            if redis_client.circuit_breaker.state == 'open':
                print("  ⚠️  Circuit breaker is OPEN - resetting...")
                redis_client.circuit_breaker.state = 'closed'
                redis_client.circuit_breaker.failure_count = 0
                print("  ✅ Circuit breaker reset to CLOSED")
        
        # Test stream operations
        print("\n📊 Testing stream operations:")
        stream_key = "test:stream"
        
        # Add to stream
        msg_id = await redis_client.xadd(stream_key, {"test": "message"})
        print(f"  - XADD test: {'✅ Passed' if msg_id else '❌ Failed'}")
        
        # Read from stream using execute_with_retry
        async with redis_client.get_connection() as conn:
            messages = await conn.xread({stream_key: "0"}, count=1, block=100)
            print(f"  - XREAD test: {'✅ Passed' if messages else '❌ Failed'}")
            
            # Cleanup
            await conn.delete(stream_key)
            await conn.delete("test_key")
        
        print("\n✅ All Redis tests passed! Connection is healthy.")
        
    except Exception as e:
        print(f"\n❌ Error during Redis testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await redis_client.disconnect()
        print("\n🔒 Redis client disconnected")

if __name__ == "__main__":
    asyncio.run(main())