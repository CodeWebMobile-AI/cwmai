#!/usr/bin/env python3
"""
Fix for Worker Monitor Key Mismatch Issue

The worker status monitor is looking for keys in the wrong location.
This script fixes the key mismatch between:
- Redis lockfree state manager: worker:state:{worker_id}
- Worker status monitor: workers:{worker_id}
"""

import asyncio
import logging
from scripts.redis_lockfree_state_manager import RedisLockFreeStateManager

# Key mapping analysis
KEY_MAPPING = {
    "Worker State Keys": {
        "Lockfree Manager": "worker:state:{worker_id}",
        "Monitor Looking For": "workers:{worker_id}",
        "Fix": "Update monitor to use correct prefix"
    },
    "Active Workers Set": {
        "Lockfree Manager": "set:active_workers",
        "Monitor Looking For": "active_workers",
        "Fix": "Already handled by get_set_members method"
    },
    "Worker Counters": {
        "Lockfree Manager": "counter:worker:{worker_id}:completed",
        "Monitor Looking For": "worker:{worker_id}:completed",
        "Fix": "Already handled by get_counter method"
    }
}

def print_analysis():
    """Print the key mismatch analysis."""
    print("=" * 80)
    print("WORKER MONITOR KEY MISMATCH ANALYSIS")
    print("=" * 80)
    
    for category, info in KEY_MAPPING.items():
        print(f"\n{category}:")
        print(f"  Lockfree Manager uses: {info['Lockfree Manager']}")
        print(f"  Monitor looks for: {info['Monitor Looking For']}")
        print(f"  Fix: {info['Fix']}")
    
    print("\n" + "=" * 80)
    print("SOLUTION:")
    print("=" * 80)
    print("""
The worker status monitor needs to be updated to use the correct get_state method
from the Redis lockfree state manager. The fix is to change line 132 in 
worker_status_monitor.py from:

    worker_data = await self.redis_state_manager.get_state(f"workers:{worker_id}")

To:

    worker_data = await self.redis_state_manager.get_worker_state(worker_id)

This will use the correct prefix and method from the lockfree state manager.
""")

async def verify_redis_keys():
    """Verify the actual keys in Redis."""
    print("\n" + "=" * 80)
    print("VERIFYING REDIS KEYS")
    print("=" * 80)
    
    manager = RedisLockFreeStateManager()
    await manager.initialize()
    
    try:
        # Check active workers
        active_workers = await manager.get_set_members("active_workers")
        print(f"\nActive workers in set: {len(active_workers)}")
        for worker_id in list(active_workers)[:5]:  # Show first 5
            print(f"  - {worker_id}")
        
        # Check worker states
        print("\nChecking worker states:")
        for worker_id in list(active_workers)[:3]:  # Check first 3
            # Try the correct method
            state = await manager.get_worker_state(worker_id)
            if state:
                print(f"  ✓ Found state for {worker_id}: status={state.get('status', 'unknown')}")
            else:
                print(f"  ✗ No state found for {worker_id}")
            
            # Try the incorrect key that monitor uses
            wrong_state = await manager.get_state(f"workers:{worker_id}")
            if wrong_state:
                print(f"    ! Found data at wrong key workers:{worker_id}")
            else:
                print(f"    ✓ No data at wrong key workers:{worker_id} (expected)")
    
    finally:
        await manager.close()

if __name__ == "__main__":
    print_analysis()
    asyncio.run(verify_redis_keys())