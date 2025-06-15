"""
Smart Redis Integration - Bridges the intelligent system with existing Redis work queue
"""
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from scripts.redis_work_queue import RedisWorkQueue
from scripts.smart_orchestrator import SmartOrchestrator
from scripts.work_item_types import WorkItem


class SmartRedisIntegration:
    """Integration layer between smart orchestrator and Redis work queue"""
    
    def __init__(self, redis_queue: RedisWorkQueue, num_workers: int = 5):
        self.logger = logging.getLogger("SmartRedisIntegration")
        self.redis_queue = redis_queue
        self.orchestrator = SmartOrchestrator(num_workers, redis_queue)
        
        # Processing state
        self.is_running = False
        self.processing_tasks = {}
        
        # Performance tracking
        self.tasks_processed = 0
        self.tasks_succeeded = 0
        
        self.logger.info("Smart Redis Integration initialized")
    
    async def start(self, specializations: Optional[list] = None) -> None:
        """Start the smart processing system"""
        self.logger.info("Starting Smart Redis Integration...")
        
        # Initialize orchestrator
        await self.orchestrator.initialize(specializations)
        
        # Start processing loop
        self.is_running = True
        asyncio.create_task(self._process_queue_loop())
        
        self.logger.info("Smart Redis Integration started")
    
    async def _process_queue_loop(self) -> None:
        """Main loop to process items from Redis queue"""
        while self.is_running:
            try:
                # Get work items for processing
                work_items = await self._get_work_items_batch()
                
                if work_items:
                    # Process each work item
                    for work_item in work_items:
                        asyncio.create_task(self._process_work_item(work_item))
                    
                    self.logger.info(f"Processing {len(work_items)} work items")
                else:
                    # No work available, wait a bit
                    await asyncio.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(5)
    
    async def _get_work_items_batch(self, batch_size: int = 10) -> list[WorkItem]:
        """Get a batch of work items from Redis queue"""
        if not self.redis_queue:
            return []  # No Redis queue available
            
        work_items = []
        
        # Use a temporary worker ID for fetching
        temp_worker_id = "smart_orchestrator_fetch"
        
        # Get available work
        items = await self.redis_queue.get_work_for_worker(
            temp_worker_id,
            specialization="general",
            count=batch_size
        )
        
        return items
    
    async def _process_work_item(self, work_item: WorkItem) -> None:
        """Process a single work item through the smart system"""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Track processing
            self.processing_tasks[work_item.id] = {
                "work_item": work_item,
                "start_time": start_time,
                "status": "bidding"
            }
            
            # Submit to orchestrator for intelligent processing
            winner_worker_id = await self.orchestrator.process_work_item(work_item)
            
            if winner_worker_id:
                self.processing_tasks[work_item.id]["status"] = "assigned"
                self.processing_tasks[work_item.id]["worker_id"] = winner_worker_id
                
                # Simulate work execution (in real system, this would be actual work)
                success, duration, quality = await self._simulate_work_execution(work_item)
                
                # Report completion
                await self.orchestrator.report_task_completion(
                    work_item, winner_worker_id, success, duration, quality
                )
                
                # Update metrics
                self.tasks_processed += 1
                if success:
                    self.tasks_succeeded += 1
                
                # Mark as completed in Redis if available
                if self.redis_queue:
                    await self.redis_queue.mark_completed(work_item.id)
                
                self.logger.info(
                    f"Completed {work_item.id}: success={success}, "
                    f"duration={duration:.2f}h, quality={quality:.2f}"
                )
            else:
                # No worker could handle this task
                self.logger.warning(f"No worker available for task {work_item.id}")
                
                # Requeue for later if Redis available
                if self.redis_queue:
                    await self.redis_queue.add_work(work_item)
            
        except Exception as e:
            self.logger.error(f"Error processing work item {work_item.id}: {e}")
        finally:
            # Clean up tracking
            if work_item.id in self.processing_tasks:
                del self.processing_tasks[work_item.id]
    
    async def _simulate_work_execution(self, work_item: WorkItem) -> tuple[bool, float, float]:
        """
        Simulate work execution for testing
        In production, this would be replaced with actual work execution
        
        Returns:
            Tuple of (success, duration_hours, quality_score)
        """
        import random
        
        # Simulate different outcomes based on task type
        if work_item.task_type == "SYSTEM_IMPROVEMENT":
            # System improvements are usually successful but take time
            await asyncio.sleep(random.uniform(2, 5))
            success = random.random() > 0.1  # 90% success rate
            duration = random.uniform(0.5, 2.0)
            quality = random.uniform(0.7, 1.0) if success else 0.0
            
        elif work_item.task_type == "NEW_PROJECT":
            # New projects are complex
            await asyncio.sleep(random.uniform(3, 8))
            success = random.random() > 0.3  # 70% success rate
            duration = random.uniform(2.0, 5.0)
            quality = random.uniform(0.6, 0.95) if success else 0.0
            
        else:
            # Default case
            await asyncio.sleep(random.uniform(1, 3))
            success = random.random() > 0.2  # 80% success rate
            duration = random.uniform(0.3, 1.5)
            quality = random.uniform(0.7, 1.0) if success else 0.0
        
        return success, duration, quality
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get statistics about the integration"""
        orchestrator_stats = self.orchestrator.get_system_stats()
        
        return {
            "integration": {
                "is_running": self.is_running,
                "tasks_processed": self.tasks_processed,
                "tasks_succeeded": self.tasks_succeeded,
                "success_rate": self.tasks_succeeded / self.tasks_processed if self.tasks_processed > 0 else 0,
                "currently_processing": len(self.processing_tasks)
            },
            "orchestrator": orchestrator_stats
        }
    
    async def stop(self) -> None:
        """Stop the integration"""
        self.logger.info("Stopping Smart Redis Integration...")
        
        self.is_running = False
        
        # Wait for current tasks to complete
        max_wait = 30  # seconds
        start_wait = datetime.now(timezone.utc)
        
        while self.processing_tasks and (datetime.now(timezone.utc) - start_wait).total_seconds() < max_wait:
            self.logger.info(f"Waiting for {len(self.processing_tasks)} tasks to complete...")
            await asyncio.sleep(1)
        
        # Shutdown orchestrator
        await self.orchestrator.shutdown()
        
        self.logger.info("Smart Redis Integration stopped")


# Modified Redis work queue integration for compatibility
async def create_smart_redis_queue(redis_client=None) -> SmartRedisIntegration:
    """Create a smart Redis work queue with AI capabilities"""
    # Initialize Redis work queue
    if redis_client:
        redis_queue = RedisWorkQueue(redis_client)
        await redis_queue.initialize()
    else:
        redis_queue = None
    
    # Create smart integration
    integration = SmartRedisIntegration(redis_queue)
    
    return integration