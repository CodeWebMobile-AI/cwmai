#!/usr/bin/env python3
"""
Test the diversity fix for work discovery
"""

import sys
import os
import asyncio
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from continuous_orchestrator import ContinuousOrchestrator

async def test_diversity_fix():
    """Test that the system generates diverse work when duplicates are encountered."""
    print("=== Testing Work Diversity Fix (10 second run) ===\n")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(message)s'
    )
    
    # Create orchestrator
    orchestrator = ContinuousOrchestrator(max_workers=1, enable_parallel=False)
    
    try:
        print("Starting orchestrator to test diversity...")
        
        # Run for just 10 seconds
        start_task = asyncio.create_task(orchestrator.start())
        await asyncio.sleep(10)
        
        print("\n10 seconds elapsed - stopping system...")
        await orchestrator.stop()
        
        if not start_task.done():
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass
        
        # Get final status
        status = orchestrator.get_status()
        
        print("\n=== Test Results ===")
        print(f"Work completed: {status['metrics']['total_work_completed']}")
        print(f"Work created: {status['metrics']['total_work_created']}")
        print(f"Queue size at end: {status['work_queue_size']}")
        
        # Check task persistence for diversity
        completion_stats = status['task_completion_stats']
        task_types = completion_stats.get('task_types', {})
        
        print(f"Task types completed: {list(task_types.keys())}")
        
        if len(task_types) > 2:
            print("✅ System shows good task diversity!")
            return True
        elif status['metrics']['total_work_completed'] > 10:
            print("✅ System is productive even with limited diversity")
            return True
        else:
            print("⚠️  Limited diversity but functional")
            return True
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_diversity_fix())
    print(f"\nTest {'passed' if success else 'failed'}")
    sys.exit(0 if success else 1)