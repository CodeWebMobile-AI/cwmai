#!/usr/bin/env python3
"""Test worker monitor functionality."""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.worker_status_monitor_fixed import WorkerMonitorFixed

async def test_worker_monitor():
    """Test the worker monitor."""
    print("Testing Worker Monitor...")
    
    try:
        # Initialize monitor
        monitor = WorkerMonitorFixed()
        await monitor.initialize()
        print("✓ Monitor initialized")
        
        # Get worker status
        status = await monitor.get_worker_status()
        print(f"\n📊 Worker Status Report:")
        print(f"Timestamp: {status.get('timestamp', 'N/A')}")
        
        # Workers
        workers = status.get('workers', {})
        print(f"\nWorkers ({len(workers)} total):")
        for worker_id, worker_info in workers.items():
            if worker_info:
                print(f"  {worker_id}: {worker_info.get('status', 'unknown')} - {worker_info.get('current_task', 'idle')}")
        
        # Queue status
        queue = status.get('queue_status', {})
        print(f"\nQueue Status:")
        print(f"  Total queued: {queue.get('total_queued', 0)}")
        print(f"  By priority: {queue.get('by_priority', {})}")
        
        # System health
        health = status.get('system_health', {})
        print(f"\nSystem Health:")
        print(f"  Overall: {health.get('system_health', 0):.1f}%")
        print(f"  Worker Health: {health.get('worker_health', 0):.1f}%")
        print(f"  Queue Health: {health.get('queue_health', 0):.1f}%")
        
        # Cleanup
        await monitor.cleanup()
        print("\n✓ Test completed successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_worker_monitor())