"""
Parallel Claude Orchestrator

Integrates the Claude executor pool with the continuous orchestrator
to enable parallel Claude execution.
"""

import asyncio
import logging
import os
from typing import Optional

from scripts.claude_execution_queue import ClaudeExecutionQueue
from scripts.claude_executor import ClaudeExecutorPool
from scripts.continuous_orchestrator_no_redis import ContinuousOrchestratorNoRedis


class ParallelClaudeOrchestrator(ContinuousOrchestratorNoRedis):
    """Enhanced orchestrator with parallel Claude execution support."""
    
    def __init__(self, max_workers: int = 10, enable_parallel: bool = True,
                 enable_round_robin: bool = False, num_claude_executors: int = 3):
        """Initialize the orchestrator with Claude executor pool.
        
        Args:
            max_workers: Maximum number of parallel workers
            enable_parallel: Whether to enable parallel processing
            enable_round_robin: Whether to enable round-robin AI provider selection
            num_claude_executors: Number of parallel Claude executors
        """
        super().__init__(max_workers, enable_parallel, enable_round_robin)
        
        self.num_claude_executors = num_claude_executors
        self.claude_queue: Optional[ClaudeExecutionQueue] = None
        self.claude_executor_pool: Optional[ClaudeExecutorPool] = None
        self.claude_pool_task: Optional[asyncio.Task] = None
        
        self.logger.info(f"Parallel Claude Orchestrator initialized with {num_claude_executors} Claude executors")
    
    async def start(self):
        """Start the orchestrator with Claude executor pool."""
        self.logger.info("Starting Parallel Claude Orchestrator...")
        
        # Initialize Claude execution queue (shared across all workers and executors)
        self.claude_queue = ClaudeExecutionQueue()
        
        # Initialize Claude executor pool
        self.claude_executor_pool = ClaudeExecutorPool(
            num_executors=self.num_claude_executors,
            queue=self.claude_queue
        )
        
        # Start Claude executor pool
        await self.claude_executor_pool.start()
        self.logger.info(f"Started {self.num_claude_executors} Claude executors")
        
        # Start monitoring task for Claude queue
        self.claude_monitor_task = asyncio.create_task(self._monitor_claude_queue())
        
        # Call parent start method
        await super().start()
    
    async def stop(self):
        """Stop the orchestrator and Claude executor pool."""
        self.logger.info("Stopping Parallel Claude Orchestrator...")
        
        # Stop Claude monitor task
        if self.claude_monitor_task and not self.claude_monitor_task.done():
            self.claude_monitor_task.cancel()
            try:
                await self.claude_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop Claude executor pool
        if self.claude_executor_pool:
            await self.claude_executor_pool.stop()
        
        # Call parent stop method
        await super().stop()
    
    async def _monitor_claude_queue(self):
        """Monitor Claude execution queue and log statistics."""
        while not self.shutdown_requested:
            try:
                # Get queue statistics
                queue_stats = await self.claude_queue.get_stats()
                pool_stats = self.claude_executor_pool.get_stats() if self.claude_executor_pool else {}
                
                # Log statistics
                self.logger.info(
                    f"📊 Claude Queue: {queue_stats['pending']} pending, "
                    f"{queue_stats['processing']} processing, "
                    f"{queue_stats['completed']} completed | "
                    f"Pool: {pool_stats.get('total_completed', 0)} tasks completed, "
                    f"avg time: {pool_stats.get('avg_execution_time', 0):.1f}s"
                )
                
                # Recover stale tasks (but don't timeout by default)
                await self.claude_queue.recover_stale_tasks(timeout_seconds=None)
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in Claude queue monitor: {e}")
                await asyncio.sleep(60)
    
    async def get_status(self) -> dict:
        """Get orchestrator status including Claude execution metrics."""
        status = await super().get_status()
        
        # Add Claude execution metrics
        if self.claude_queue:
            claude_stats = await self.claude_queue.get_stats()
            status['claude_execution'] = {
                'queue_stats': claude_stats,
                'num_executors': self.num_claude_executors
            }
            
            if self.claude_executor_pool:
                status['claude_execution']['pool_stats'] = self.claude_executor_pool.get_stats()
        
        return status


async def main():
    """Example usage of the Parallel Claude Orchestrator."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create orchestrator with parallel Claude execution
    orchestrator = ParallelClaudeOrchestrator(
        max_workers=10,                # 10 workers for creating issues
        enable_parallel=True,
        enable_round_robin=False,
        num_claude_executors=3         # 3 parallel Claude executors
    )
    
    try:
        # Start the orchestrator
        await orchestrator.start()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())