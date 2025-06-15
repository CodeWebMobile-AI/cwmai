#!/usr/bin/env python3
"""Check Redis state for stored repositories."""

import asyncio
import json
import sys
sys.path.insert(0, '/workspaces/cwmai/scripts')

from redis_integration.redis_client import RedisClient

async def check_redis_state():
    """Check what state is stored in Redis."""
    redis_client = RedisClient()
    await redis_client.connect()
    
    print("=== CHECKING REDIS STATE ===\n")
    
    # Check for system state
    state_keys = [
        "cwmai:state:cwmai_orchestrator",
        "cwmai:state:system",
        "cwmai:state:projects",
        "cwmai:state:repositories"
    ]
    
    for key in state_keys:
        try:
            value = await redis_client.get(key)
            if value:
                print(f"\n{key}:")
                try:
                    data = json.loads(value)
                    if isinstance(data, dict) and 'projects' in data:
                        print(f"  Found {len(data['projects'])} projects:")
                        for proj_name in list(data['projects'].keys())[:5]:
                            print(f"    - {proj_name}")
                        if len(data['projects']) > 5:
                            print(f"    ... and {len(data['projects']) - 5} more")
                    else:
                        print(f"  {json.dumps(data, indent=2)[:500]}")
                except:
                    print(f"  {value[:200]}")
            else:
                print(f"{key}: Not found")
        except Exception as e:
            print(f"{key}: Error - {e}")
    
    # Check for any keys with 'ai-creative-studio' or 'moderncms'
    print("\n=== SEARCHING FOR SPECIFIC REPOSITORIES ===")
    
    # Scan for keys containing these repos
    cursor = 0
    found_keys = []
    pattern = "*"
    
    while True:
        cursor, keys = await redis_client.scan(cursor, match=pattern, count=100)
        for key in keys:
            try:
                value = await redis_client.get(key)
                if value and ('ai-creative-studio' in value or 'moderncms-with-ai-powered' in value):
                    found_keys.append(key)
            except:
                pass
        
        if cursor == 0:
            break
    
    if found_keys:
        print(f"\nFound {len(found_keys)} keys containing target repositories:")
        for key in found_keys[:10]:
            print(f"  - {key}")
    else:
        print("\nNo keys found containing target repositories")
    
    await redis_client.close()

if __name__ == "__main__":
    asyncio.run(check_redis_state())