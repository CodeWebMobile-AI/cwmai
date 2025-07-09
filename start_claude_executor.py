#!/usr/bin/env python3
"""
Start Claude Executor to Process Queued Tasks

Processes Claude execution tasks from the queue in parallel.
"""

import asyncio
import logging
import os
import sys
import argparse
import signal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.claude_executor import ClaudeExecutorPool
from scripts.claude_execution_queue import ClaudeExecutionQueue


async def main(num_executors: int = 3):
    """Run the Claude executor pool."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('claude_executor.log', mode='a')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Create queue
    queue = ClaudeExecutionQueue()
    
    # Get current stats
    stats = await queue.get_stats()
    logger.info(f"📊 Queue Status:")
    logger.info(f"   Pending: {stats['pending']}")
    logger.info(f"   Processing: {stats['processing']}")
    logger.info(f"   Completed: {stats['completed']}")
    logger.info(f"   Failed: {stats.get('failed', 0)}")
    
    if stats['pending'] == 0 and stats['processing'] == 0:
        logger.info("✅ No tasks to process")
        return
    
    # Create executor pool
    pool = ClaudeExecutorPool(num_executors=num_executors, queue=queue)
    
    # Setup signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received...")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start executors
        await pool.start()
        logger.info(f"🚀 Started {num_executors} Claude executors")
        
        # Wait for shutdown signal or all tasks to complete
        while not shutdown_event.is_set():
            await asyncio.sleep(5)
            
            # Check if all tasks are complete
            stats = await queue.get_stats()
            if stats['pending'] == 0 and stats['processing'] == 0:
                logger.info("✅ All tasks completed")
                break
            
            # Log progress
            logger.info(f"📊 Progress: {stats['completed']} completed, "
                       f"{stats['processing']} processing, {stats['pending']} pending")
        
    finally:
        # Stop executors
        logger.info("Stopping executors...")
        await pool.stop()
        
        # Final stats
        final_stats = await queue.get_stats()
        pool_stats = pool.get_stats()
        
        logger.info("\n📊 Final Statistics:")
        logger.info(f"   Total Completed: {final_stats['completed']}")
        logger.info(f"   Total Failed: {final_stats.get('failed', 0)}")
        logger.info(f"   Still Pending: {final_stats['pending']}")
        logger.info(f"   Average Execution Time: {pool_stats.get('avg_execution_time', 0):.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Claude Executor Pool")
    parser.add_argument('--executors', type=int, default=3,
                       help='Number of parallel executors (default: 3)')
    
    args = parser.parse_args()
    
    # Ensure we're in local mode
    os.environ['STORAGE_MODE'] = 'LOCAL'
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║              CLAUDE EXECUTOR POOL                             ║
║                                                               ║
║  Processing queued Claude tasks in parallel                   ║
║  Executors: {args.executors}                                         ║
║                                                               ║
║  Press Ctrl+C to shutdown gracefully                          ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main(args.executors))