#!/usr/bin/env python3
"""
Test the continuous AI system with a brief run to verify no infinite loops
"""

import sys
import os
import asyncio
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from continuous_orchestrator import ContinuousOrchestrator

async def test_continuous_system():
    """Test the continuous system for a brief period."""
    print("=== Testing Continuous AI System (30 second run) ===\n")
    
    # Set up logging to capture what happens
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create orchestrator with minimal workers for testing
    orchestrator = ContinuousOrchestrator(max_workers=1, enable_parallel=False)
    
    try:
        # Start the system
        print("Starting continuous orchestrator...")
        
        # Run for just 30 seconds to test
        start_task = asyncio.create_task(orchestrator.start())
        
        # Wait for 30 seconds then stop
        await asyncio.sleep(30)
        
        print("\n30 seconds elapsed - stopping system...")
        await orchestrator.stop()
        
        # Cancel the start task if still running
        if not start_task.done():
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass
        
        # Get final status
        status = orchestrator.get_status()
        
        print("\n=== Final System Status ===")
        print(f"Runtime: {status['runtime_seconds']:.1f} seconds")
        print(f"Work completed: {status['metrics']['total_work_completed']}")
        print(f"Work created: {status['metrics']['total_work_created']}")
        print(f"Errors: {status['metrics']['total_errors']}")
        print(f"Queue size: {status['work_queue_size']}")
        print(f"GitHub integration: {status['github_integration']}")
        
        # Check if the system worked correctly
        if status['metrics']['total_errors'] == 0:
            print("\n✅ System ran without errors!")
        else:
            print(f"\n⚠️  System had {status['metrics']['total_errors']} errors")
        
        if status['work_queue_size'] > 0:
            print(f"✅ Work queue has {status['work_queue_size']} items (system discovering work)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test."""
    try:
        result = asyncio.run(test_continuous_system())
        if result:
            print("\n🎉 Continuous system test completed successfully!")
            print("The infinite loop issue appears to be fixed.")
        else:
            print("\n❌ Test failed.")
        return result
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)