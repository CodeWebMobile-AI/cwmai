"""
Coordinated Work Discovery System

Integrates the WorkDiscoveryCoordinator and ResearchAwareScheduler with
the existing IntelligentWorkFinder to dramatically reduce API calls.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path

from scripts.work_discovery_coordinator import WorkDiscoveryCoordinator, EventDrivenDiscovery
from scripts.research_aware_scheduler import ResearchAwareScheduler, SmartWorkQueue
from scripts.intelligent_work_finder import IntelligentWorkFinder
from scripts.gemini_deep_research import GeminiDeepResearch
from scripts.ai_brain import IntelligentAIBrain


class CoordinatedWorkDiscovery:
    """
    Coordinates work discovery across the entire system.
    
    Replaces the pattern of 10 workers checking every 30 seconds with:
    - ONE centralized coordinator
    - Research-aware scheduling
    - Event-driven discovery
    - Smart queue management
    """
    
    def __init__(self, 
                 ai_brain: IntelligentAIBrain,
                 system_state: Dict[str, Any],
                 worker_count: int = 10):
        """
        Initialize the coordinated discovery system.
        
        Args:
            ai_brain: AI brain for generating work
            system_state: System state dictionary
            worker_count: Number of workers in the system
        """
        self.ai_brain = ai_brain
        self.system_state = system_state
        self.worker_count = worker_count
        
        # Initialize components
        self.work_finder = IntelligentWorkFinder(ai_brain, system_state)
        self.research_scheduler = ResearchAwareScheduler()
        self.work_queue = SmartWorkQueue(worker_count)
        
        # Initialize coordinator with discovery callback
        self.coordinator = WorkDiscoveryCoordinator(
            work_finder_callback=self._discover_work,
            worker_count=worker_count,
            base_interval=60,  # 1 minute base
            max_interval=600   # 10 minutes max
        )
        
        # Initialize event-driven discovery
        self.event_discovery = EventDrivenDiscovery(self.coordinator)
        
        # Wire up events
        self._setup_event_handlers()
        
        # Track active research
        self.active_research_id: Optional[str] = None
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _setup_event_handlers(self):
        """Set up event handlers between components."""
        # Research scheduler notifies coordinator
        self.research_scheduler.on_research_complete(self._on_research_complete)
        
        # Work queue notifies when generation needed
        self.work_queue.on_work_generated(self._on_work_generation_needed)
        
        # Coordinator event handlers
        self.coordinator.register_event_listener('research_completed', 
                                               self.event_discovery.on_research_completed)
        self.coordinator.register_event_listener('project_completed',
                                               self.event_discovery.on_project_completed)
        self.coordinator.register_event_listener('queue_low',
                                               self.event_discovery.on_queue_low)
        
    async def _discover_work(self) -> List[Dict[str, Any]]:
        """
        Main work discovery callback used by coordinator.
        
        Returns:
            List of work items discovered
        """
        # Check if we should wait for research
        if self.research_scheduler.should_wait_for_research():
            wait_time = self.research_scheduler.get_wait_time()
            self.logger.info(f"Deferring work discovery - research in progress ({wait_time:.0f}s remaining)")
            return []
            
        try:
            # Use the work finder to discover opportunities
            self.logger.info("🔍 Starting coordinated work discovery...")
            
            # Check if queue needs work
            if not self.work_queue.should_generate_work():
                self.logger.info("Queue is healthy - skipping discovery")
                return []
            
            # Get current queue size
            current_queue_size = len(self.system_state.get('work_queue', []))
            
            # Determine how many items to generate
            target_count = self.work_queue.get_generation_size()
            self.logger.info(f"Generating {target_count} work items to maintain queue")
            
            # Discover work
            opportunities = await self.work_finder.discover_work(
                max_items=target_count,
                current_workload=current_queue_size
            )
            
            # Update queue size
            if opportunities:
                current_queue_size = len(self.system_state.get('work_queue', []))
                new_size = current_queue_size + len(opportunities)
                self.work_queue.update_size(new_size)
                self.coordinator._last_queue_update = new_size
                
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error in coordinated work discovery: {e}")
            return []
            
    def _on_research_complete(self, research_phase):
        """Handle research completion."""
        self.logger.info(
            f"Research '{research_phase.phase_id}' completed with "
            f"{research_phase.problems_found} problems"
        )
        
        # Notify coordinator
        self.coordinator.complete_research(research_phase.problems_found)
        
        # Clear active research
        if self.active_research_id == research_phase.phase_id:
            self.active_research_id = None
            
    async def _on_work_generation_needed(self, count: int):
        """Handle work generation request from queue."""
        self.logger.info(f"Queue requesting {count} new work items")
        await self.coordinator.trigger_discovery(f"queue_request: {count} items")
        
    def start_research(self, research_type: str = "market_discovery"):
        """
        Signal that research is starting.
        
        Args:
            research_type: Type of research being performed
        """
        # Generate unique research ID
        research_id = f"{research_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_research_id = research_id
        
        # Notify both scheduler and coordinator
        self.research_scheduler.start_research(research_id)
        self.coordinator.start_research()
        
        self.logger.info(f"📚 Research phase started: {research_id}")
        
    def complete_research(self, problems_found: int = 0):
        """
        Signal that research has completed.
        
        Args:
            problems_found: Number of problems discovered
        """
        if not self.active_research_id:
            self.logger.warning("No active research to complete")
            return
            
        # Complete in scheduler (will trigger callbacks)
        self.research_scheduler.complete_research(problems_found)
        
        self.logger.info(f"✅ Research completed: {problems_found} problems found")
        
    async def on_project_completed(self, project_name: str):
        """
        Handle project completion event.
        
        Args:
            project_name: Name of completed project
        """
        await self.event_discovery.on_project_completed(project_name)
        
    async def on_workers_idle(self, idle_count: int):
        """
        Handle idle workers event.
        
        Args:
            idle_count: Number of idle workers
        """
        await self.event_discovery.on_worker_idle_threshold(idle_count, self.worker_count)
        
    def update_queue_size(self, size: int):
        """
        Update the current work queue size.
        
        Args:
            size: Current number of items in queue
        """
        self.work_queue.update_size(size)
        self.coordinator.update_queue_size(size)
        
    def start(self):
        """Start the coordinated discovery system."""
        self.coordinator.start()
        self.logger.info(
            "✨ Coordinated work discovery started - "
            f"reducing discovery attempts from ~1,200/hour to ~20-30/hour"
        )
        
    async def stop(self):
        """Stop the coordinated discovery system."""
        await self.coordinator.stop()
        self.logger.info("Coordinated work discovery stopped")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'coordinator': self.coordinator.get_stats(),
            'research_scheduler': self.research_scheduler.get_stats(),
            'work_queue': self.work_queue.get_stats(),
            'active_research': self.active_research_id
        }
        
    def log_status(self):
        """Log current status of all components."""
        self.logger.info("=== Coordinated Work Discovery Status ===")
        
        # Coordinator status
        coord_stats = self.coordinator.get_stats()
        self.logger.info(
            f"Coordinator: {coord_stats['total_attempts']} attempts, "
            f"{coord_stats['success_rate']} success, "
            f"next check in {coord_stats['next_check_interval']}"
        )
        
        # Research status
        self.research_scheduler.log_summary()
        
        # Queue status
        queue_stats = self.work_queue.get_stats()
        self.logger.info(
            f"Work Queue: {queue_stats['current_size']}/{queue_stats['optimal_size']} items, "
            f"consumption: {queue_stats['consumption_rate']}, "
            f"health: {queue_stats['health']}"
        )
        
        
# Convenience function for integration
async def create_coordinated_discovery(ai_brain: IntelligentAIBrain,
                                     system_state: Dict[str, Any],
                                     worker_count: int = 10) -> CoordinatedWorkDiscovery:
    """
    Create and configure a coordinated discovery system.
    
    Args:
        ai_brain: AI brain instance
        system_state: System state dictionary
        worker_count: Number of workers
        
    Returns:
        Configured CoordinatedWorkDiscovery instance
    """
    discovery = CoordinatedWorkDiscovery(ai_brain, system_state, worker_count)
    
    # Load any saved stats if available
    stats_file = Path("work_discovery_stats.json")
    if stats_file.exists():
        try:
            with open(stats_file) as f:
                saved_stats = json.load(f)
                # Could restore some state here if needed
                logging.info(f"Loaded discovery stats from {stats_file}")
        except Exception as e:
            logging.warning(f"Could not load stats: {e}")
            
    return discovery