#!/usr/bin/env python3
"""Test continuous AI system with worker monitoring enabled."""

import asyncio
import sys
import signal

# Add to path
sys.path.insert(0, '.')

from run_continuous_ai import ContinuousAIRunner


async def test_with_monitor():
    """Test the continuous AI system with worker monitoring."""
    print("Starting Continuous AI System with Worker Monitoring...")
    print("=" * 60)
    print("This will run for 15 seconds to test the monitoring.")
    print("Watch for worker status updates every 5 seconds.")
    print("=" * 60)
    
    # Create runner with monitoring enabled
    runner = ContinuousAIRunner(
        max_workers=10,
        enable_parallel=True,
        mode="test",  # Use test mode for quick testing
        enable_research=False,  # Disable research for simple test
        enable_round_robin=False,
        enable_worker_monitor=True  # Enable worker monitoring
    )
    
    # Run for 15 seconds then stop
    async def run_with_timeout():
        # Start the system
        task = asyncio.create_task(runner.start())
        
        # Wait for 15 seconds
        await asyncio.sleep(15)
        
        # Shutdown
        print("\n" + "=" * 60)
        print("Test complete. Shutting down...")
        await runner.shutdown()
        
        # Cancel the task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    await run_with_timeout()
    print("✅ Test completed successfully!")


def main():
    """Main entry point."""
    try:
        asyncio.run(test_with_monitor())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()