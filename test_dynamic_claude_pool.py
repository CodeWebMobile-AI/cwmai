#!/usr/bin/env python3
"""
Test Dynamic Claude Executor Pool

This script demonstrates the dynamic scaling capabilities of the DynamicClaudeExecutorPool
by simulating various workload scenarios.
"""

import asyncio
import logging
import os
import json
import time
from datetime import datetime

# Add current directory to Python path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.dynamic_claude_executor_pool import DynamicClaudeExecutorPool
from scripts.claude_execution_queue import ClaudeExecutionQueue


async def simulate_workload():
    """Simulate different workload patterns to test dynamic scaling."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create queue and dynamic pool
    queue = ClaudeExecutionQueue("test_claude_queue.json")
    pool = DynamicClaudeExecutorPool(
        min_executors=1,
        max_executors=5,
        queue=queue,
        scale_up_threshold=3.0,  # Scale up when > 3 tasks per executor
        scale_down_threshold=0.3,  # Scale down when < 30% utilization
        scale_check_interval=10,  # Check every 10 seconds for demo
        idle_timeout=60  # Scale down after 60 seconds idle
    )
    
    try:
        # Start the pool
        await pool.start()
        logger.info("🚀 Dynamic Claude executor pool started")
        
        # Monitor task to print stats periodically
        async def monitor_stats():
            while True:
                stats = pool.get_stats()
                logger.info(f"\n📊 Pool Stats:")
                logger.info(f"  Executors: {stats['num_executors']} (min={stats['min_executors']}, max={stats['max_executors']})")
                logger.info(f"  Utilization: {stats['scaling_metrics']['current_utilization']:.1%}")
                logger.info(f"  Avg Queue Depth (5min): {stats['scaling_metrics']['avg_queue_depth_5min']:.1f}")
                logger.info(f"  Avg Wait Time (5min): {stats['scaling_metrics']['avg_wait_time_5min']:.1f}s")
                
                queue_stats = await queue.get_stats()
                logger.info(f"  Queue: {queue_stats['pending']} pending, {queue_stats['processing']} processing")
                
                if stats['scaling_metrics']['recent_scaling_events']:
                    logger.info(f"  Recent scaling events:")
                    for event in stats['scaling_metrics']['recent_scaling_events'][-3:]:
                        logger.info(f"    - {event['type']}: {event['from_count']}→{event['to_count']} ({event['reason']})")
                
                await asyncio.sleep(15)
        
        monitor_task = asyncio.create_task(monitor_stats())
        
        # Scenario 1: Low workload (should stay at minimum)
        logger.info("\n🔵 Scenario 1: Low workload (adding 2 tasks)")
        for i in range(2):
            await queue.add_task(
                issue_path=f"test_issues/issue_{i}.json",
                repository="test-repo",
                issue_number=i+1,
                prompt=f"Test task {i+1}: Simple documentation update"
            )
        
        await asyncio.sleep(30)
        
        # Scenario 2: Burst of tasks (should scale up)
        logger.info("\n🔴 Scenario 2: Burst workload (adding 15 tasks)")
        for i in range(15):
            await queue.add_task(
                issue_path=f"test_issues/burst_{i}.json",
                repository="test-repo",
                issue_number=100+i,
                prompt=f"Burst task {i+1}: Complex feature implementation"
            )
        
        await asyncio.sleep(45)
        
        # Scenario 3: Sustained high load (should maintain scaled executors)
        logger.info("\n🟡 Scenario 3: Sustained load (adding 5 tasks every 20s)")
        for batch in range(3):
            for i in range(5):
                await queue.add_task(
                    issue_path=f"test_issues/sustained_{batch}_{i}.json",
                    repository="test-repo",
                    issue_number=200+batch*10+i,
                    prompt=f"Sustained task batch {batch+1}, item {i+1}"
                )
            await asyncio.sleep(20)
        
        # Scenario 4: Gradual decrease (should scale down)
        logger.info("\n🟢 Scenario 4: Gradual decrease (waiting for idle)")
        await asyncio.sleep(90)  # Wait for idle timeout
        
        # Final stats
        await asyncio.sleep(10)
        final_stats = pool.get_stats()
        logger.info("\n📈 Final Statistics:")
        logger.info(f"  Total completed: {final_stats['total_completed']}")
        logger.info(f"  Total failed: {final_stats['total_failed']}")
        logger.info(f"  Avg execution time: {final_stats['avg_execution_time']:.1f}s")
        logger.info(f"  Final executor count: {final_stats['num_executors']}")
        
        # Cancel monitor
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
    finally:
        # Clean up
        await pool.stop()
        logger.info("✅ Test completed")


async def test_real_workload():
    """Test with real Claude tasks (requires Claude CLI)."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Check if Claude is available
    import subprocess
    try:
        result = subprocess.run(['claude', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Claude CLI not found. Please install Claude first.")
            return
    except FileNotFoundError:
        logger.error("Claude CLI not found. Please install Claude first.")
        return
    
    # Create queue and pool with production-like settings
    queue = ClaudeExecutionQueue()
    pool = DynamicClaudeExecutorPool(
        min_executors=1,
        max_executors=5,
        queue=queue,
        scale_up_threshold=2.0,  # More aggressive scaling for real tasks
        scale_down_threshold=0.5,
        scale_check_interval=20,
        idle_timeout=180
    )
    
    try:
        await pool.start()
        logger.info("🚀 Dynamic Claude executor pool started for real workload")
        
        # Add some real tasks
        tasks = [
            {
                "repository": "test-project",
                "issue_number": 1,
                "prompt": "Create a simple Python function that calculates the factorial of a number with proper error handling and docstring."
            },
            {
                "repository": "test-project",
                "issue_number": 2,
                "prompt": "Write unit tests for a factorial function using pytest, including edge cases."
            },
            {
                "repository": "test-project",
                "issue_number": 3,
                "prompt": "Create a README.md file explaining how to use a factorial function with examples."
            }
        ]
        
        # Add tasks
        for task in tasks:
            await queue.add_task(
                issue_path=f"issues/{task['issue_number']}.json",
                repository=task['repository'],
                issue_number=task['issue_number'],
                prompt=task['prompt']
            )
        
        logger.info(f"Added {len(tasks)} real tasks to the queue")
        
        # Monitor until complete
        while True:
            stats = await queue.get_stats()
            pool_stats = pool.get_stats()
            
            logger.info(f"\nQueue: {stats['pending']} pending, {stats['processing']} processing, {stats['completed']} completed")
            logger.info(f"Pool: {pool_stats['num_executors']} executors, {pool_stats['scaling_metrics']['current_utilization']:.1%} utilization")
            
            if stats['pending'] == 0 and stats['processing'] == 0:
                logger.info("All tasks completed!")
                break
            
            await asyncio.sleep(10)
        
    finally:
        await pool.stop()
        logger.info("✅ Real workload test completed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Dynamic Claude Executor Pool")
    parser.add_argument('--real', action='store_true', help='Test with real Claude tasks')
    args = parser.parse_args()
    
    if args.real:
        asyncio.run(test_real_workload())
    else:
        asyncio.run(simulate_workload())