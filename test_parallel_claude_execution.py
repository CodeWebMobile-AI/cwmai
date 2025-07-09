#!/usr/bin/env python3
"""
Test Parallel Claude Execution

Demonstrates how the new parallel Claude execution system works.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.claude_execution_queue import ClaudeExecutionQueue
from scripts.claude_executor import ClaudeExecutor, ClaudeExecutorPool
from scripts.local_issue_creator import LocalIssueCreator
from scripts.work_item_types import WorkItem, TaskPriority


async def test_parallel_claude():
    """Test parallel Claude execution."""
    print("=" * 80)
    print("TESTING PARALLEL CLAUDE EXECUTION")
    print("=" * 80)
    
    # Initialize Claude execution queue
    queue = ClaudeExecutionQueue()
    
    # Initialize Claude executor pool with 3 executors
    executor_pool = ClaudeExecutorPool(num_executors=3, queue=queue)
    
    # Start the executor pool
    await executor_pool.start()
    print("✅ Started 3 Claude executors")
    
    # Initialize local issue creator
    issue_creator = LocalIssueCreator(enable_claude_execution=True)
    
    # Create 3 test work items for different repositories
    test_repos = [
        "CodeWebMobile-AI/social-spark-ai",
        "CodeWebMobile-AI/ai-text-summarizer", 
        "CodeWebMobile-AI/micronutrient-optimizer"
    ]
    
    work_items = []
    for i, repo in enumerate(test_repos):
        work_item = WorkItem(
            id=f"test_{i+1}",
            title=f"Test Parallel Claude Execution {i+1}",
            description=f"Create a simple test file test_parallel_{i+1}.py that prints 'Hello from parallel test {i+1}'",
            task_type="FEATURE",
            priority=TaskPriority.HIGH,
            repository=repo
        )
        work_items.append(work_item)
    
    # Create issues and queue Claude execution
    print("\n📋 Creating issues and queuing Claude tasks...")
    start_time = asyncio.get_event_loop().time()
    
    tasks = []
    for work_item in work_items:
        task = asyncio.create_task(issue_creator.execute_work_item(work_item))
        tasks.append(task)
    
    # Wait for all issues to be created and queued
    results = await asyncio.gather(*tasks)
    
    queue_time = asyncio.get_event_loop().time() - start_time
    print(f"\n✅ All issues created and Claude tasks queued in {queue_time:.2f}s")
    
    # Show queue status
    queue_stats = await queue.get_stats()
    print(f"\n📊 Queue Status:")
    print(f"  - Pending: {queue_stats['pending']}")
    print(f"  - Processing: {queue_stats['processing']}")
    print(f"  - Completed: {queue_stats['completed']}")
    
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
        if stats['pending'] == 0 and stats['processing'] == 0 and stats['completed'] >= len(work_items):
            break
    
    total_time = asyncio.get_event_loop().time() - start_time
    print(f"\n✅ All Claude executions completed in {total_time:.2f}s")
    
    # Final statistics
    final_stats = await queue.get_stats()
    print(f"\n📊 Final Statistics:")
    print(f"  - Total tasks: {len(work_items)}")
    print(f"  - Completed: {final_stats['completed']}")
    print(f"  - Average execution time: {final_stats['avg_execution_time']:.2f}s")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Speedup: {len(work_items) * final_stats['avg_execution_time'] / total_time:.2f}x")
    
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
        await test_parallel_claude()
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║            PARALLEL CLAUDE EXECUTION TEST                        ║
║                                                                  ║
║  This test will:                                                 ║
║  1. Create 3 issues in different repositories                    ║
║  2. Queue Claude execution for each issue                        ║
║  3. Execute all 3 Claude tasks in PARALLEL                       ║
║  4. Show real-time progress and statistics                       ║
║                                                                  ║
║  You should see multiple Claude processes running simultaneously  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())