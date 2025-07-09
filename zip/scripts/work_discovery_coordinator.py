"""
Centralized Work Discovery Coordinator

Reduces work discovery attempts from 1,200/hour to ~20-30/hour by:
1. Using ONE coordinator instead of 10 workers checking independently
2. Implementing exponential backoff when no work found
3. Coordinating with research phase to avoid checking during research
4. Event-driven discovery triggers
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class DiscoveryStats:
    """Track discovery efficiency metrics."""
    total_attempts: int = 0
    successful_attempts: int = 0
    work_items_found: int = 0
    total_time_spent: float = 0.0
    last_attempt_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_empty: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of discovery attempts."""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts
    
    @property
    def average_time_per_attempt(self) -> float:
        """Average time spent per discovery attempt."""
        if self.total_attempts == 0:
            return 0.0
        return self.total_time_spent / self.total_attempts


class WorkDiscoveryCoordinator:
    """
    Centralized coordinator for work discovery.
    
    Instead of having 10 workers independently check for work every 30 seconds,
    this coordinator intelligently schedules discovery attempts based on:
    - Research phase status
    - Previous discovery success
    - Current queue depth
    - Event triggers
    """
    
    def __init__(self, 
                 work_finder_callback: Callable,
                 worker_count: int = 10,
                 base_interval: int = 60,
                 max_interval: int = 600):
        """
        Initialize the coordinator.
        
        Args:
            work_finder_callback: Async function to discover work
            worker_count: Number of workers to coordinate for
            base_interval: Base interval between checks (seconds)
            max_interval: Maximum interval between checks (seconds)
        """
        self.work_finder_callback = work_finder_callback
        self.worker_count = worker_count
        self.base_interval = base_interval
        self.max_interval = max_interval
        
        # State tracking
        self.research_in_progress = False
        self.research_start_time: Optional[datetime] = None
        self.expected_research_duration = 240  # 4 minutes typical
        
        # Discovery stats
        self.stats = DiscoveryStats()
        
        # Work queue tracking
        self.current_queue_size = 0
        self.min_queue_size = worker_count * 2
        self.optimal_queue_size = worker_count * 5
        
        # Event listeners
        self.event_listeners: Dict[str, List[Callable]] = {
            'research_started': [],
            'research_completed': [],
            'project_completed': [],
            'queue_low': [],
            'workers_idle': []
        }
        
        # Backoff configuration
        self.backoff_multiplier = 2
        self.backoff_factor = 1.0  # Can be adjusted based on success rate
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._running = False
        self._discovery_task: Optional[asyncio.Task] = None
        
    def register_event_listener(self, event: str, callback: Callable):
        """Register a callback for specific events."""
        if event in self.event_listeners:
            self.event_listeners[event].append(callback)
            
    async def emit_event(self, event: str, data: Any = None):
        """Emit an event to all registered listeners."""
        if event in self.event_listeners:
            for callback in self.event_listeners[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"Error in event listener for {event}: {e}")
    
    def start_research(self):
        """Signal that research phase has started."""
        self.research_in_progress = True
        self.research_start_time = datetime.now()
        self.logger.info("Research phase started - pausing work discovery")
        asyncio.create_task(self.emit_event('research_started'))
        
    def complete_research(self, problems_found: int = 0):
        """Signal that research phase has completed."""
        self.research_in_progress = False
        duration = (datetime.now() - self.research_start_time).total_seconds() if self.research_start_time else 0
        self.logger.info(f"Research phase completed in {duration:.1f}s - found {problems_found} problems")
        
        # Update expected duration based on actual
        if duration > 0:
            self.expected_research_duration = int(duration * 1.1)  # Add 10% buffer
            
        # Reset consecutive empty since we have new problems
        if problems_found > 0:
            self.stats.consecutive_empty = 0
            
        asyncio.create_task(self.emit_event('research_completed', problems_found))
        
    def update_queue_size(self, size: int):
        """Update current work queue size."""
        old_size = self.current_queue_size
        self.current_queue_size = size
        
        # Check if queue is getting low
        if size < self.min_queue_size and old_size >= self.min_queue_size:
            asyncio.create_task(self.emit_event('queue_low', size))
            
    def get_next_check_interval(self) -> float:
        """
        Calculate the next check interval using exponential backoff.
        
        Returns:
            Seconds until next check
        """
        # If research is in progress, wait for it to complete
        if self.research_in_progress:
            if self.research_start_time:
                elapsed = (datetime.now() - self.research_start_time).total_seconds()
                remaining = max(self.expected_research_duration - elapsed, 30)
                return remaining
            return self.expected_research_duration
            
        # If queue is critically low, check soon
        if self.current_queue_size < self.worker_count:
            return self.base_interval * 0.5  # Half the base interval
            
        # Exponential backoff based on consecutive empty discoveries
        if self.stats.consecutive_empty == 0:
            interval = self.base_interval
        else:
            # Calculate backoff: base * (multiplier ^ consecutive_empty) * factor
            interval = self.base_interval * (self.backoff_multiplier ** self.stats.consecutive_empty) * self.backoff_factor
            
        # Cap at maximum interval
        return min(interval, self.max_interval)
        
    def adjust_backoff_factor(self):
        """Adjust backoff factor based on success rate."""
        success_rate = self.stats.success_rate
        
        if success_rate < 0.05:  # Less than 5% success
            self.backoff_factor = min(self.backoff_factor * 1.2, 2.0)
            self.logger.info(f"Low success rate ({success_rate:.1%}) - increased backoff factor to {self.backoff_factor}")
        elif success_rate > 0.3:  # More than 30% success
            self.backoff_factor = max(self.backoff_factor * 0.8, 0.5)
            self.logger.info(f"High success rate ({success_rate:.1%}) - decreased backoff factor to {self.backoff_factor}")
            
    async def discover_work_once(self) -> int:
        """
        Perform a single work discovery attempt.
        
        Returns:
            Number of work items discovered
        """
        start_time = time.time()
        self.stats.total_attempts += 1
        self.stats.last_attempt_time = datetime.now()
        
        try:
            # Call the work finder
            work_items = await self.work_finder_callback()
            
            # Update stats
            elapsed = time.time() - start_time
            self.stats.total_time_spent += elapsed
            
            items_found = len(work_items) if work_items else 0
            
            if items_found > 0:
                self.stats.successful_attempts += 1
                self.stats.work_items_found += items_found
                self.stats.last_success_time = datetime.now()
                self.stats.consecutive_empty = 0
                self.logger.info(f"✅ Discovery successful: found {items_found} work items in {elapsed:.1f}s")
            else:
                self.stats.consecutive_empty += 1
                self.logger.debug(f"No work found (consecutive empty: {self.stats.consecutive_empty})")
                
            # Adjust backoff based on performance
            if self.stats.total_attempts % 10 == 0:  # Every 10 attempts
                self.adjust_backoff_factor()
                
            return items_found
            
        except Exception as e:
            self.logger.error(f"Error during work discovery: {e}")
            self.stats.consecutive_empty += 1
            return 0
            
    async def run_discovery_loop(self):
        """Main discovery loop with intelligent scheduling."""
        self.logger.info("Starting centralized work discovery coordinator")
        
        while self._running:
            try:
                # Check if we should discover work now
                if self.should_discover_now():
                    items_found = await self.discover_work_once()
                    
                    # Update queue size if callback provided it
                    if hasattr(self, '_last_queue_update'):
                        self.update_queue_size(self._last_queue_update)
                
                # Calculate next check interval
                interval = self.get_next_check_interval()
                
                # Log scheduling decision
                if interval > self.base_interval:
                    self.logger.info(
                        f"📅 Next discovery in {interval:.0f}s "
                        f"(backoff due to {self.stats.consecutive_empty} empty attempts)"
                    )
                else:
                    self.logger.debug(f"Next discovery in {interval:.0f}s")
                
                # Wait for next check or event
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(self.base_interval)
                
    def should_discover_now(self) -> bool:
        """
        Determine if we should run discovery now.
        
        Returns:
            True if discovery should run
        """
        # Don't discover during research
        if self.research_in_progress:
            elapsed = (datetime.now() - self.research_start_time).total_seconds() if self.research_start_time else 0
            if elapsed < self.expected_research_duration:
                self.logger.debug(f"Skipping discovery - research in progress ({elapsed:.0f}s elapsed)")
                return False
                
        # Always discover if queue is critically low
        if self.current_queue_size < self.worker_count:
            self.logger.info("Queue critically low - forcing discovery")
            return True
            
        return True
        
    async def trigger_discovery(self, reason: str = "manual"):
        """Manually trigger a discovery attempt."""
        self.logger.info(f"Manual discovery triggered: {reason}")
        items_count = await self.discover_work_once()
        return items_count
        
    def start(self):
        """Start the coordinator."""
        if self._running:
            return
            
        self._running = True
        self._discovery_task = asyncio.create_task(self.run_discovery_loop())
        self.logger.info("Work discovery coordinator started")
        
    async def stop(self):
        """Stop the coordinator."""
        self._running = False
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Work discovery coordinator stopped")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        stats = {
            'total_attempts': self.stats.total_attempts,
            'successful_attempts': self.stats.successful_attempts,
            'success_rate': f"{self.stats.success_rate:.1%}",
            'work_items_found': self.stats.work_items_found,
            'average_time_per_attempt': f"{self.stats.average_time_per_attempt:.1f}s",
            'consecutive_empty': self.stats.consecutive_empty,
            'current_backoff_factor': self.backoff_factor,
            'next_check_interval': f"{self.get_next_check_interval():.0f}s",
            'research_in_progress': self.research_in_progress,
            'current_queue_size': self.current_queue_size,
            'last_attempt': self.stats.last_attempt_time.isoformat() if self.stats.last_attempt_time else None,
            'last_success': self.stats.last_success_time.isoformat() if self.stats.last_success_time else None
        }
        return stats
        
    def save_stats(self, filepath: Path):
        """Save statistics to file."""
        stats = self.get_stats()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
            
    def log_stats_summary(self):
        """Log a summary of discovery statistics."""
        stats = self.get_stats()
        self.logger.info(
            f"Discovery Stats: "
            f"{stats['total_attempts']} attempts, "
            f"{stats['success_rate']} success rate, "
            f"{stats['work_items_found']} items found, "
            f"next check in {stats['next_check_interval']}"
        )


# Event-driven discovery triggers
class EventDrivenDiscovery:
    """Helper class for event-driven discovery triggers."""
    
    def __init__(self, coordinator: WorkDiscoveryCoordinator):
        self.coordinator = coordinator
        self.logger = logging.getLogger(self.__class__.__name__)
        
    async def on_project_completed(self, project_name: str):
        """Trigger discovery when a project is completed."""
        self.logger.info(f"Project completed: {project_name}")
        # Immediately look for follow-up work
        await self.coordinator.trigger_discovery(f"project_completed: {project_name}")
        
    async def on_research_completed(self, problems_found: int):
        """Trigger discovery when research completes."""
        if problems_found > 0:
            self.logger.info(f"Research found {problems_found} problems - triggering discovery")
            await self.coordinator.trigger_discovery(f"research_completed: {problems_found} problems")
            
    async def on_worker_idle_threshold(self, idle_count: int, total_count: int):
        """Trigger discovery when too many workers are idle."""
        idle_ratio = idle_count / total_count if total_count > 0 else 0
        if idle_ratio > 0.5:
            self.logger.info(f"{idle_count}/{total_count} workers idle - triggering discovery")
            await self.coordinator.trigger_discovery(f"high_idle_ratio: {idle_ratio:.1%}")
            
    async def on_queue_low(self, queue_size: int):
        """Trigger discovery when queue is low."""
        self.logger.info(f"Queue low ({queue_size} items) - triggering discovery")
        await self.coordinator.trigger_discovery(f"queue_low: {queue_size}")