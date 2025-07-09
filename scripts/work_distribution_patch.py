"""
Patch for Enhanced Hybrid Orchestrator to use Enhanced Work Queue Manager

This patch integrates the enhanced work queue manager to fix:
1. Duplicate task assignment
2. Simultaneous work discovery
3. Work distribution issues
"""

import os
import sys
from typing import List, Optional

# Import the enhanced work queue manager
from scripts.enhanced_work_queue_manager import EnhancedWorkQueueManager
from scripts.work_item_types import WorkItem


class WorkDistributionPatch:
    """Patches the orchestrator to use enhanced work distribution."""
    
    @staticmethod
    def apply_to_orchestrator(orchestrator):
        """Apply the work distribution patch to an orchestrator instance."""
        
        print("🔧 Applying work distribution patch...")
        
        # Replace work queue with enhanced version
        orchestrator._work_queue_manager = EnhancedWorkQueueManager()
        
        # Store original methods
        orchestrator._original_find_work_for_worker = orchestrator._find_work_for_worker
        orchestrator._original_find_work = orchestrator.find_work if hasattr(orchestrator, 'find_work') else None
        
        # Patch _find_work_for_worker method
        async def patched_find_work_for_worker(worker):
            """Enhanced work finder that uses the work queue manager."""
            
            # First check if we need to discover new work
            if orchestrator._work_queue_manager.should_discover_work():
                try:
                    # Only one worker does discovery at a time
                    orchestrator.logger.info(f"Worker {worker.id} triggering work discovery")
                    
                    # Get current worker count
                    active_workers = len([w for w in orchestrator.workers.values() 
                                        if w.status != 'SHUTTING_DOWN'])
                    
                    # Discover work
                    if orchestrator._original_find_work:
                        new_work = await orchestrator._original_find_work(
                            max_items=active_workers * 2  # Get 2x tasks to have buffer
                        )
                    else:
                        # Fallback if no find_work method
                        new_work = []
                    
                    # Add to queue
                    if new_work:
                        added = await orchestrator._work_queue_manager.add_work(new_work)
                        orchestrator.logger.info(f"Work discovery added {added} new tasks")
                        
                finally:
                    await orchestrator._work_queue_manager.end_discovery()
            
            # Clean up stale assignments periodically
            if hasattr(orchestrator, '_last_cleanup'):
                if time.time() - orchestrator._last_cleanup > 300:  # Every 5 minutes
                    stale_count = await orchestrator._work_queue_manager.cleanup_stale_assignments()
                    if stale_count > 0:
                        orchestrator.logger.warning(f"Cleaned up {stale_count} stale assignments")
                    orchestrator._last_cleanup = time.time()
            else:
                orchestrator._last_cleanup = time.time()
            
            # Get work for this specific worker
            work_item = await orchestrator._work_queue_manager.get_work_for_worker(
                worker.id,
                worker.specialization
            )
            
            return work_item
        
        # Replace the method
        orchestrator._find_work_for_worker = patched_find_work_for_worker
        
        # Add work completion tracking
        original_execute_work = orchestrator._execute_work
        
        async def patched_execute_work(worker, work_item):
            """Enhanced work execution with completion tracking."""
            try:
                result = await original_execute_work(worker, work_item)
                
                # Mark work as completed
                success = result.get('success', False) if isinstance(result, dict) else True
                await orchestrator._work_queue_manager.complete_work(
                    worker.id,
                    work_item.id,
                    success
                )
                
                return result
            except Exception as e:
                # Mark as failed on exception
                await orchestrator._work_queue_manager.complete_work(
                    worker.id,
                    work_item.id,
                    False
                )
                raise
        
        orchestrator._execute_work = patched_execute_work
        
        # Add status method
        def get_work_distribution_status():
            """Get current work distribution status."""
            return orchestrator._work_queue_manager.get_queue_status()
        
        orchestrator.get_work_distribution_status = get_work_distribution_status
        
        print("✅ Work distribution patch applied successfully")
        
        return orchestrator


# Add import for time
import time
