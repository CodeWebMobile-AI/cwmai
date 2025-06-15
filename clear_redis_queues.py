#!/usr/bin/env python3
"""Clear Redis work queues."""

import asyncio
import sys
sys.path.insert(0, '/workspaces/cwmai/scripts')

from redis_integration.redis_client import RedisClient

async def clear_queues():
    """Clear all work queues."""
    redis_client = RedisClient()
    await redis_client.connect()
    
    print("=== CLEARING REDIS WORK QUEUES ===\n")
    
    # Delete all work queue streams
    streams = [
        "cwmai:work_queue:critical",
        "cwmai:work_queue:high", 
        "cwmai:work_queue:medium",
        "cwmai:work_queue:low",
        "cwmai:work_queue:background"
    ]
    
    for stream in streams:
        try:
            result = await redis_client.delete(stream)
            print(f"Deleted {stream}: {result}")
        except Exception as e:
            print(f"Error deleting {stream}: {e}")
    
    print("\n✅ Queues cleared!")

if __name__ == "__main__":
    asyncio.run(clear_queues())