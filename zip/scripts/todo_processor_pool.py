"""
Todo Processor Pool

Dynamic pool of workers that process todo messages concurrently,
similar to Claude Code's executor pool pattern.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone
from dataclasses import dataclass
import time

from natural_language_todo_queue import ConcurrentTodoQueue, TodoMessage, MessageStatus


@dataclass
class ProcessorStats:
    """Statistics for a single processor."""
    processor_id: str
    messages_processed: int = 0
    errors_encountered: int = 0
    total_processing_time: float = 0.0
    last_activity: Optional[datetime] = None
    current_message: Optional[str] = None
    is_idle: bool = True


class TodoProcessor:
    """Individual processor that handles todo messages."""
    
    def __init__(
        self,
        processor_id: str,
        queue: ConcurrentTodoQueue,
        process_func: Callable,
        stats: ProcessorStats
    ):
        """Initialize a todo processor."""
        self.processor_id = processor_id
        self.queue = queue
        self.process_func = process_func
        self.stats = stats
        self.logger = logging.getLogger(f"{__name__}.{processor_id}")
        self.running = True
        
    async def run(self):
        """Main processor loop."""
        self.logger.info(f"Processor {self.processor_id} started")
        
        while self.running:
            try:
                # Get next message from queue
                message = self.queue.dequeue(timeout=1.0)
                
                if message:
                    await self._process_message(message)
                else:
                    # No messages, mark as idle
                    self.stats.is_idle = True
                    self.stats.current_message = None
                    await asyncio.sleep(0.1)  # Brief pause when idle
                    
            except Exception as e:
                self.logger.error(f"Processor error: {e}")
                await asyncio.sleep(1)  # Back off on error
                
        self.logger.info(f"Processor {self.processor_id} stopped")
        
    async def _process_message(self, message: TodoMessage):
        """Process a single message."""
        self.stats.is_idle = False
        self.stats.current_message = message.id
        self.stats.last_activity = datetime.now(timezone.utc)
        
        start_time = time.time()
        
        try:
            self.logger.debug(f"Processing message {message.id}: {message.content[:50]}...")
            
            # Call the actual processing function
            result = await self.process_func(message)
            
            # Mark as completed
            self.queue.complete(message.id, result=result)
            
            # Update stats
            self.stats.messages_processed += 1
            self.stats.total_processing_time += (time.time() - start_time)
            
            self.logger.info(f"Completed message {message.id} in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to process message {message.id}: {e}")
            
            # Mark as failed
            self.queue.complete(message.id, error=str(e))
            
            # Update stats
            self.stats.errors_encountered += 1
            
    def stop(self):
        """Stop this processor."""
        self.running = False


class DynamicProcessorPool:
    """
    Manages a dynamic pool of processors that scale based on load.
    Similar to Claude Code's dynamic executor pool.
    """
    
    def __init__(
        self,
        queue: ConcurrentTodoQueue,
        process_func: Callable,
        min_processors: int = 2,
        max_processors: int = 10
    ):
        """Initialize the processor pool."""
        self.logger = logging.getLogger(__name__)
        self.queue = queue
        self.process_func = process_func
        self.min_processors = min_processors
        self.max_processors = max_processors
        
        # Processor management
        self.processors: Dict[str, TodoProcessor] = {}
        self.processor_tasks: Dict[str, asyncio.Task] = {}
        self.processor_stats: Dict[str, ProcessorStats] = {}
        
        # Scaling parameters
        self.scale_up_threshold = 5    # Messages per processor
        self.scale_down_threshold = 0.5 # Utilization ratio
        self.idle_timeout = 60         # Seconds before scaling down
        
        # Pool state
        self.running = False
        self.monitor_task = None
        
    async def start(self):
        """Start the processor pool."""
        self.running = True
        self.logger.info(f"Starting processor pool (min={self.min_processors}, max={self.max_processors})")
        
        # Start minimum processors
        for i in range(self.min_processors):
            await self._add_processor(f"processor-{i}")
            
        # Start monitoring task
        self.monitor_task = asyncio.create_task(self._monitor_and_scale())
        
    async def stop(self):
        """Stop the processor pool."""
        self.running = False
        self.logger.info("Stopping processor pool")
        
        # Stop all processors
        for processor in self.processors.values():
            processor.stop()
            
        # Wait for tasks to complete
        if self.processor_tasks:
            await asyncio.gather(*self.processor_tasks.values(), return_exceptions=True)
            
        # Cancel monitor
        if self.monitor_task:
            self.monitor_task.cancel()
            
        self.logger.info("Processor pool stopped")
        
    async def _add_processor(self, processor_id: str):
        """Add a new processor to the pool."""
        if processor_id in self.processors:
            return
            
        # Create stats
        stats = ProcessorStats(processor_id=processor_id)
        self.processor_stats[processor_id] = stats
        
        # Create processor
        processor = TodoProcessor(processor_id, self.queue, self.process_func, stats)
        self.processors[processor_id] = processor
        
        # Start processor task
        task = asyncio.create_task(processor.run())
        self.processor_tasks[processor_id] = task
        
        self.logger.info(f"Added processor {processor_id} (total: {len(self.processors)})")
        
    async def _remove_processor(self, processor_id: str):
        """Remove a processor from the pool."""
        if processor_id not in self.processors:
            return
            
        # Stop processor
        processor = self.processors[processor_id]
        processor.stop()
        
        # Wait for task
        task = self.processor_tasks.get(processor_id)
        if task:
            await task
            
        # Clean up
        del self.processors[processor_id]
        del self.processor_tasks[processor_id]
        del self.processor_stats[processor_id]
        
        self.logger.info(f"Removed processor {processor_id} (total: {len(self.processors)})")
        
    async def _monitor_and_scale(self):
        """Monitor pool and scale as needed."""
        while self.running:
            try:
                await self._check_scaling()
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                await asyncio.sleep(5)
                
    async def _check_scaling(self):
        """Check if scaling is needed."""
        status = self.queue.get_status()
        pending = status["pending"]
        current_size = len(self.processors)
        
        # Calculate metrics
        messages_per_processor = pending / max(current_size, 1)
        utilization = self._calculate_utilization()
        
        self.logger.debug(
            f"Scaling check - Pending: {pending}, Processors: {current_size}, "
            f"MPP: {messages_per_processor:.1f}, Utilization: {utilization:.1%}"
        )
        
        # Scale up if needed
        if messages_per_processor > self.scale_up_threshold and current_size < self.max_processors:
            new_count = min(
                current_size + 1,  # Add one at a time
                self.max_processors
            )
            for i in range(current_size, new_count):
                await self._add_processor(f"processor-{i}")
                
        # Scale down if underutilized
        elif utilization < self.scale_down_threshold and current_size > self.min_processors:
            # Find idle processors
            idle_processors = self._find_idle_processors()
            
            if idle_processors:
                # Remove one idle processor
                processor_id = idle_processors[0]
                await self._remove_processor(processor_id)
                
    def _calculate_utilization(self) -> float:
        """Calculate overall processor utilization."""
        if not self.processor_stats:
            return 0.0
            
        active_count = sum(1 for stats in self.processor_stats.values() if not stats.is_idle)
        return active_count / len(self.processor_stats)
        
    def _find_idle_processors(self) -> List[str]:
        """Find processors that have been idle for too long."""
        idle_processors = []
        now = datetime.now(timezone.utc)
        
        for processor_id, stats in self.processor_stats.items():
            if stats.is_idle and stats.last_activity:
                idle_time = (now - stats.last_activity).total_seconds()
                if idle_time > self.idle_timeout:
                    idle_processors.append(processor_id)
                    
        return idle_processors
        
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        total_processed = sum(s.messages_processed for s in self.processor_stats.values())
        total_errors = sum(s.errors_encountered for s in self.processor_stats.values())
        total_time = sum(s.total_processing_time for s in self.processor_stats.values())
        
        active_processors = [
            pid for pid, stats in self.processor_stats.items()
            if not stats.is_idle
        ]
        
        return {
            "pool_size": len(self.processors),
            "active_processors": len(active_processors),
            "utilization": self._calculate_utilization(),
            "total_processed": total_processed,
            "total_errors": total_errors,
            "avg_processing_time": total_time / max(total_processed, 1),
            "processor_details": {
                pid: {
                    "processed": stats.messages_processed,
                    "errors": stats.errors_encountered,
                    "idle": stats.is_idle,
                    "current": stats.current_message
                }
                for pid, stats in self.processor_stats.items()
            }
        }


class ProcessorPoolMonitor:
    """Monitors processor pool health and performance."""
    
    def __init__(self, pool: DynamicProcessorPool):
        self.pool = pool
        self.logger = logging.getLogger(__name__)
        
    async def monitor_loop(self, interval: float = 30):
        """Run monitoring loop."""
        while True:
            try:
                stats = self.pool.get_pool_stats()
                
                self.logger.info(
                    f"Pool stats - Size: {stats['pool_size']}, "
                    f"Active: {stats['active_processors']}, "
                    f"Processed: {stats['total_processed']}, "
                    f"Errors: {stats['total_errors']}"
                )
                
                # Alert on high error rate
                if stats['total_processed'] > 0:
                    error_rate = stats['total_errors'] / stats['total_processed']
                    if error_rate > 0.1:  # 10% error rate
                        self.logger.warning(f"High error rate: {error_rate:.1%}")
                        
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                
            await asyncio.sleep(interval)


# Example processor function
async def example_todo_processor(message: TodoMessage) -> Dict[str, Any]:
    """Example processor that simulates todo processing."""
    # Simulate processing time
    await asyncio.sleep(0.5)
    
    # Parse message content (simplified)
    content_lower = message.content.lower()
    
    if "add" in content_lower or "create" in content_lower:
        return {"action": "add", "task": message.content}
    elif "complete" in content_lower or "done" in content_lower:
        return {"action": "complete", "task": message.content}
    elif "list" in content_lower or "show" in content_lower:
        return {"action": "list"}
    else:
        return {"action": "unknown", "content": message.content}


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        # Create queue and pool
        queue = ConcurrentTodoQueue()
        pool = DynamicProcessorPool(queue, example_todo_processor, min_processors=2, max_processors=5)
        
        # Start pool
        await pool.start()
        
        # Add some test messages
        for i in range(10):
            queue.enqueue(f"Add task {i} to the list")
            
        # Let it process
        await asyncio.sleep(5)
        
        # Check stats
        print(f"Pool stats: {pool.get_pool_stats()}")
        print(f"Queue status: {queue.get_status()}")
        
        # Stop pool
        await pool.stop()
        
    asyncio.run(main())