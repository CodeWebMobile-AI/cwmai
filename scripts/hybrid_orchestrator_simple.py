"""
Simplified Hybrid Orchestrator

A lightweight version of the enhanced hybrid orchestrator that:
- Uses simple work generation to avoid timeouts
- Still integrates SmartTaskType system
- Provides fast work discovery
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from scripts.continuous_orchestrator_no_redis import ContinuousOrchestratorNoRedis as ContinuousOrchestrator
from scripts.work_item_types import WorkItem, TaskPriority
from scripts.simple_work_generator import SimpleWorkGenerator
from scripts.enhanced_smart_task_system import create_enhanced_task_system
from scripts.ai_brain_factory import AIBrainFactory
from scripts.state_manager import StateManager
from scripts.task_types import SmartTaskType


class SimplifiedHybridOrchestrator(ContinuousOrchestrator):
    """Simplified hybrid orchestrator for fast work generation."""
    
    def __init__(self, max_workers: int = 5, enable_parallel: bool = True,
                 enable_research: bool = False, enable_round_robin: bool = False):
        """Initialize simplified orchestrator."""
        # Initialize base orchestrator
        super().__init__(max_workers, enable_parallel, enable_research, enable_round_robin)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize simple work generator
        self.simple_work_generator = SimpleWorkGenerator(self.ai_brain)
        
        # Initialize smart task system (for metadata only, no heavy operations)
        self.task_system = create_enhanced_task_system()
        
        self.logger.info("Simplified hybrid orchestrator initialized")
    
    async def find_work(self) -> List[WorkItem]:
        """Find work using simple generation."""
        all_work = []
        
        try:
            # Skip GitHub issues in local mode
            self.logger.info("Skipping GitHub issues (local mode)")
            
            # Generate work using simple generator
            if len(all_work) < self.max_workers:
                needed = self.max_workers - len(all_work)
                
                # Get projects from system state
                projects = self.system_state.get('projects', {})
                
                # Generate simple work items
                simple_work = self.simple_work_generator.generate_work_items(
                    projects,
                    max_items=needed
                )
                
                all_work.extend(simple_work)
                self.logger.info(f"Simple generator created {len(simple_work)} work items")
                
                # Log work item details
                for work in simple_work[:3]:  # First 3
                    self.logger.debug(f"  - [{work.task_type}] {work.title} ({work.repository})")
        
        except Exception as e:
            self.logger.error(f"Error finding work: {e}")
            import traceback
            traceback.print_exc()
        
        return all_work[:self.max_workers]
    
    async def find_single_work_item(self) -> Optional[WorkItem]:
        """Find a single work item."""
        work_items = await self.find_work()
        return work_items[0] if work_items else None
    
    async def execute_work(self, work_item: WorkItem, worker_id: str) -> Dict[str, Any]:
        """Execute work with basic tracking."""
        # Add smart task metadata if available
        if hasattr(SmartTaskType, work_item.task_type):
            try:
                smart_type = SmartTaskType[work_item.task_type]
                metadata = self.task_system.get_task_metadata(smart_type.value)
                if metadata:
                    work_item.metadata['smart_task'] = True
                    work_item.metadata['complexity'] = metadata.complexity.name
                    work_item.metadata['category'] = metadata.category.value
            except:
                pass  # Ignore if not a smart task type
        
        # Execute using parent method
        return await super().execute_work(work_item, worker_id)
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get basic analytics."""
        base_analytics = await super().get_analytics()
        
        # Add simplified metrics
        base_analytics['orchestrator_type'] = 'simplified'
        base_analytics['work_generation_method'] = 'simple'
        
        return base_analytics
    
    async def run(self):
        """Alias for start()."""
        await self.start()
    
    async def shutdown(self):
        """Graceful shutdown."""
        self.logger.info("Shutting down simplified hybrid orchestrator...")
        self.shutdown_requested = True
        
        # Wait for workers to finish
        if hasattr(self, 'workers'):
            for worker in self.workers.values():
                worker.status = 'shutting_down'


def create_simplified_hybrid_orchestrator(**kwargs) -> SimplifiedHybridOrchestrator:
    """Create a simplified hybrid orchestrator."""
    return SimplifiedHybridOrchestrator(**kwargs)