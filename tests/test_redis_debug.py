#!/usr/bin/env python3
"""Debug Redis streams directly."""

import asyncio
import sys
sys.path.insert(0, '/workspaces/cwmai/scripts')

from redis_integration.redis_client import RedisClient

async def debug_redis():
    """Debug Redis streams."""
    redis_client = RedisClient()
    await redis_client.connect()
    
    print("=== REDIS STREAM DEBUG ===\n")
    
    # Check all priority streams
    streams = [
        "cwmai:work_queue:critical",
        "cwmai:work_queue:high", 
        "cwmai:work_queue:medium",
        "cwmai:work_queue:low",
        "cwmai:work_queue:background"
    ]
    
    for stream in streams:
        print(f"\nStream: {stream}")
        try:
            # Get stream info
            info = await redis_client.xinfo_stream(stream)
            print(f"  Length: {info.get('length', 0)}")
            print(f"  Groups: {info.get('groups', 0)}")
            
            # Get last few entries
            entries = await redis_client.xrevrange(stream, count=3)
            if entries:
                print(f"  Last {len(entries)} entries:")
                for entry_id, data in entries:
                    # Decode data
                    decoded = {}
                    for k, v in data.items():
                        key = k.decode() if isinstance(k, bytes) else k
                        value = v.decode() if isinstance(v, bytes) else v
                        decoded[key] = value
                    print(f"    {entry_id}: {decoded.get('title', 'Unknown')}")
            else:
                print("  No entries")
                
            # Check pending messages
            try:
                pending = await redis_client.xpending(stream, "cwmai_workers")
                if pending and isinstance(pending, dict):
                    print(f"  Pending: {pending.get('pending', 0)}")
                elif pending and isinstance(pending, (list, tuple)) and len(pending) > 0:
                    print(f"  Pending: {pending[0]}")
            except:
                print("  No consumer group")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    # Check if work is being tracked elsewhere
    print("\n\nChecking other keys:")
    keys = await redis_client.keys("cwmai:*")
    for key in keys[:20]:  # First 20 keys
        key_str = key.decode() if isinstance(key, bytes) else key
        if "work" in key_str.lower():
            print(f"  {key_str}")

if __name__ == "__main__":
    asyncio.run(debug_redis())