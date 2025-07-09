#!/usr/bin/env python3
"""
Simple Test for Parallel Claude Execution

Creates new issues to ensure clean test.
"""

import asyncio
import logging
import os
import sys
import time

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.claude_execution_queue import ClaudeExecutionQueue
from scripts.claude_executor import ClaudeExecutorPool


async def test_simple_parallel():
    """Test parallel Claude execution with simple tasks."""
    print("=" * 80)
    print("SIMPLE PARALLEL CLAUDE EXECUTION TEST")
    print("=" * 80)
    
    # Initialize Claude execution queue
    queue = ClaudeExecutionQueue()
    
    # Initialize Claude executor pool with 3 executors
    executor_pool = ClaudeExecutorPool(num_executors=3, queue=queue)
    
    # Start the executor pool
    await executor_pool.start()
    print("✅ Started 3 Claude executors")
    
    # Manually add test tasks to the queue
    test_tasks = [
        {
            'repo': 'CodeWebMobile-AI/social-spark-ai',
            'issue': 100,
            'prompt': 'Create a simple test file test_parallel_1.py that prints "Hello from parallel test 1"'
        },
        {
            'repo': 'CodeWebMobile-AI/ai-text-summarizer', 
            'issue': 100,
            'prompt': 'Create a simple test file test_parallel_2.py that prints "Hello from parallel test 2"'
        },
        {
            'repo': 'CodeWebMobile-AI/micronutrient-optimizer',
            'issue': 100,
            'prompt': 'Create a simple test file test_parallel_3.py that prints "Hello from parallel test 3"'
        }
    ]
    
    print("\n📋 Adding tasks to Claude execution queue...")
    start_time = time.time()
    
    for task in test_tasks:
        await queue.add_task(
            issue_path=f"dummy/path/issue-{task['issue']}.json",
            repository=task['repo'],
            issue_number=task['issue'],
            prompt=task['prompt']
        )
    
    print(f"✅ All tasks queued in {time.time() - start_time:.2f}s")
    
    # Monitor execution progress
    print("\n⏳ Monitoring Claude execution progress...")
    print("   (You should see multiple Claude processes running in parallel)")
    
    # Wait for all tasks to complete (with progress updates)
    while True:
        await asyncio.sleep(5)
        
        stats = await queue.get_stats()
        pool_stats = executor_pool.get_stats()
        
        print(f"\n📊 Progress Update:")
        print(f"  Queue: {stats['pending']} pending, {stats['processing']} processing, {stats['completed']} completed")
        print(f"  Pool: {pool_stats['total_completed']} completed, {pool_stats['total_failed']} failed")
        
        # Show which executors are working
        for executor_id, executor_stats in pool_stats['executors'].items():
            if executor_stats['current_task']:
                task_info = executor_stats['current_task']
                print(f"  - {executor_id}: Working on {task_info['repository']} issue #{task_info['issue_number']}")
        
        # Check if all tasks are complete
        if stats['pending'] == 0 and stats['processing'] == 0 and stats['completed'] >= len(test_tasks):
            break
    
    total_time = time.time() - start_time
    print(f"\n✅ All Claude executions completed in {total_time:.2f}s")
    
    # Final statistics
    final_stats = await queue.get_stats()
    print(f"\n📊 Final Statistics:")
    print(f"  - Total tasks: {len(test_tasks)}")
    print(f"  - Completed: {final_stats['completed']}")
    print(f"  - Average execution time: {final_stats.get('avg_execution_time', 0):.2f}s")
    print(f"  - Total time: {total_time:.2f}s")
    if final_stats.get('avg_execution_time', 0) > 0:
        print(f"  - Speedup: {len(test_tasks) * final_stats['avg_execution_time'] / total_time:.2f}x")
    
    # Stop the executor pool
    await executor_pool.stop()
    print("\n✅ Claude executor pool stopped")


async def main():
    """Main test function."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        await test_simple_parallel()
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         SIMPLE PARALLEL CLAUDE EXECUTION TEST                    ║
║                                                                  ║
║  This test will:                                                 ║
║  1. Queue 3 Claude tasks directly (no issue creation)            ║
║  2. Execute all 3 Claude tasks in PARALLEL                       ║
║  3. Show real-time progress and statistics                       ║
║                                                                  ║
║  You should see multiple Claude processes running simultaneously  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())