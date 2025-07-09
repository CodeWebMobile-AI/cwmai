"""
Research-Aware Scheduling System

Coordinates work discovery with research phases to avoid wasted API calls.
When research is running (3-4 minutes), the system waits for completion
before checking for new work.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ResearchPhase:
    """Track a research phase."""
    phase_id: str
    start_time: datetime
    expected_duration: float  # seconds
    actual_duration: Optional[float] = None
    problems_found: int = 0
    completed: bool = False
    
    @property
    def elapsed_time(self) -> float:
        """Time elapsed since research started."""
        return (datetime.now() - self.start_time).total_seconds()
        
    @property
    def remaining_time(self) -> float:
        """Estimated time remaining in research."""
        if self.completed:
            return 0
        return max(0, self.expected_duration - self.elapsed_time)
        
    @property
    def is_overdue(self) -> bool:
        """Check if research is taking longer than expected."""
        return not self.completed and self.elapsed_time > self.expected_duration * 1.2


class ResearchAwareScheduler:
    """
    Scheduler that coordinates work discovery with research phases.
    
    Key features:
    - Tracks ongoing research phases
    - Prevents work discovery during active research
    - Learns typical research durations
    - Triggers immediate discovery after research completes
    """
    
    def __init__(self, default_research_duration: float = 240.0):
        """
        Initialize the scheduler.
        
        Args:
            default_research_duration: Expected research duration in seconds (default 4 minutes)
        """
        self.default_research_duration = default_research_duration
        self.current_research: Optional[ResearchPhase] = None
        self.research_history: List[ResearchPhase] = []
        self.duration_history: List[float] = []
        
        # Callbacks
        self.on_research_complete_callbacks: List[Callable] = []
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def get_expected_duration(self) -> float:
        """
        Get expected research duration based on history.
        
        Returns:
            Expected duration in seconds
        """
        if len(self.duration_history) >= 3:
            # Use average of last 5 research phases
            recent = self.duration_history[-5:]
            avg_duration = sum(recent) / len(recent)
            # Add 10% buffer
            return avg_duration * 1.1
        return self.default_research_duration
        
    def start_research(self, phase_id: str) -> ResearchPhase:
        """
        Signal that a new research phase has started.
        
        Args:
            phase_id: Unique identifier for this research phase
            
        Returns:
            The new ResearchPhase object
        """
        if self.current_research and not self.current_research.completed:
            self.logger.warning(f"Starting new research while {self.current_research.phase_id} is still active")
            self.complete_research(0, forced=True)
            
        expected_duration = self.get_expected_duration()
        self.current_research = ResearchPhase(
            phase_id=phase_id,
            start_time=datetime.now(),
            expected_duration=expected_duration
        )
        
        self.logger.info(
            f"📚 Research phase '{phase_id}' started - "
            f"expected duration: {expected_duration:.0f}s"
        )
        
        # Schedule a check for overdue research
        asyncio.create_task(self._monitor_research_timeout())
        
        return self.current_research
        
    def complete_research(self, problems_found: int, forced: bool = False) -> Optional[ResearchPhase]:
        """
        Signal that the current research phase has completed.
        
        Args:
            problems_found: Number of problems discovered
            forced: Whether this is a forced completion (timeout/error)
            
        Returns:
            The completed ResearchPhase object
        """
        if not self.current_research:
            self.logger.warning("No active research to complete")
            return None
            
        if self.current_research.completed:
            self.logger.warning(f"Research {self.current_research.phase_id} already completed")
            return self.current_research
            
        # Calculate actual duration
        actual_duration = self.current_research.elapsed_time
        self.current_research.actual_duration = actual_duration
        self.current_research.problems_found = problems_found
        self.current_research.completed = True
        
        # Update history
        self.research_history.append(self.current_research)
        if not forced and actual_duration > 30:  # Only track normal completions
            self.duration_history.append(actual_duration)
            # Keep only last 10 durations
            self.duration_history = self.duration_history[-10:]
        
        # Log completion
        status = "forced" if forced else "completed"
        self.logger.info(
            f"✅ Research phase '{self.current_research.phase_id}' {status} - "
            f"duration: {actual_duration:.0f}s (expected: {self.current_research.expected_duration:.0f}s), "
            f"problems found: {problems_found}"
        )
        
        # Notify callbacks
        asyncio.create_task(self._notify_research_complete(self.current_research))
        
        completed = self.current_research
        self.current_research = None
        return completed
        
    async def _monitor_research_timeout(self):
        """Monitor for research taking too long."""
        if not self.current_research:
            return
            
        phase_id = self.current_research.phase_id
        timeout = self.current_research.expected_duration * 1.5  # 50% over expected
        
        await asyncio.sleep(timeout)
        
        # Check if this research is still active
        if (self.current_research and 
            self.current_research.phase_id == phase_id and 
            not self.current_research.completed):
            
            self.logger.warning(
                f"⚠️ Research phase '{phase_id}' timeout - "
                f"elapsed: {self.current_research.elapsed_time:.0f}s"
            )
            self.complete_research(0, forced=True)
            
    async def _notify_research_complete(self, research: ResearchPhase):
        """Notify all callbacks that research has completed."""
        for callback in self.on_research_complete_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(research)
                else:
                    callback(research)
            except Exception as e:
                self.logger.error(f"Error in research complete callback: {e}")
                
    def on_research_complete(self, callback: Callable):
        """Register a callback for when research completes."""
        self.on_research_complete_callbacks.append(callback)
        
    def is_research_active(self) -> bool:
        """Check if research is currently active."""
        return self.current_research is not None and not self.current_research.completed
        
    def should_wait_for_research(self) -> bool:
        """
        Determine if work discovery should wait for research.
        
        Returns:
            True if should wait, False otherwise
        """
        if not self.is_research_active():
            return False
            
        # Wait if research has significant time remaining
        remaining = self.current_research.remaining_time
        if remaining > 30:  # More than 30 seconds left
            self.logger.debug(
                f"Waiting for research '{self.current_research.phase_id}' - "
                f"{remaining:.0f}s remaining"
            )
            return True
            
        # Don't wait if research is almost done
        return False
        
    def get_wait_time(self) -> float:
        """
        Get how long to wait for research to complete.
        
        Returns:
            Seconds to wait (0 if no active research)
        """
        if not self.is_research_active():
            return 0
            
        # Wait for expected completion plus small buffer
        return self.current_research.remaining_time + 5
        
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        stats = {
            'research_active': self.is_research_active(),
            'total_research_phases': len(self.research_history),
            'average_duration': sum(self.duration_history) / len(self.duration_history) if self.duration_history else 0,
            'current_research': None
        }
        
        if self.current_research:
            stats['current_research'] = {
                'phase_id': self.current_research.phase_id,
                'elapsed_time': f"{self.current_research.elapsed_time:.0f}s",
                'remaining_time': f"{self.current_research.remaining_time:.0f}s",
                'is_overdue': self.current_research.is_overdue
            }
            
        # Recent research summary
        if self.research_history:
            recent = self.research_history[-5:]
            stats['recent_research'] = [
                {
                    'phase_id': r.phase_id,
                    'duration': f"{r.actual_duration:.0f}s" if r.actual_duration else 'unknown',
                    'problems_found': r.problems_found
                }
                for r in recent
            ]
            
        return stats
        
    def log_summary(self):
        """Log a summary of research scheduling."""
        if self.is_research_active():
            self.logger.info(
                f"Research Scheduler: Active research '{self.current_research.phase_id}' - "
                f"{self.current_research.remaining_time:.0f}s remaining"
            )
        else:
            avg_duration = sum(self.duration_history) / len(self.duration_history) if self.duration_history else self.default_research_duration
            self.logger.info(
                f"Research Scheduler: No active research - "
                f"average duration: {avg_duration:.0f}s from {len(self.research_history)} phases"
            )


class SmartWorkQueue:
    """
    Intelligent work queue management that coordinates with research.
    
    Maintains optimal queue depth based on worker count and predicts
    when to generate more work.
    """
    
    def __init__(self, worker_count: int = 10):
        """
        Initialize the smart queue.
        
        Args:
            worker_count: Number of workers consuming from queue
        """
        self.worker_count = worker_count
        self.min_queue_size = worker_count * 2
        self.optimal_queue_size = worker_count * 5
        self.max_queue_size = worker_count * 10
        
        # Queue metrics
        self.current_size = 0
        self.consumption_rate = 0.0  # items per minute
        self.generation_callbacks: List[Callable] = []
        
        # History for rate calculation
        self.consumption_history: List[tuple] = []  # (timestamp, items_consumed)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def update_size(self, new_size: int):
        """Update current queue size."""
        old_size = self.current_size
        self.current_size = new_size
        
        if old_size > new_size:
            consumed = old_size - new_size
            self.consumption_history.append((datetime.now(), consumed))
            # Keep only last hour
            cutoff = datetime.now() - timedelta(hours=1)
            self.consumption_history = [
                (ts, items) for ts, items in self.consumption_history 
                if ts > cutoff
            ]
            
        # Calculate consumption rate
        if len(self.consumption_history) >= 2:
            total_consumed = sum(items for _, items in self.consumption_history)
            time_span = (self.consumption_history[-1][0] - self.consumption_history[0][0]).total_seconds() / 60
            if time_span > 0:
                self.consumption_rate = total_consumed / time_span
                
    def needs_work(self) -> bool:
        """Check if queue needs more work."""
        return self.current_size < self.min_queue_size
        
    def predict_depletion_time(self) -> Optional[float]:
        """
        Predict when queue will be depleted.
        
        Returns:
            Minutes until depletion (None if rate unknown)
        """
        if self.consumption_rate <= 0 or self.current_size == 0:
            return None
            
        return self.current_size / self.consumption_rate
        
    def should_generate_work(self) -> bool:
        """
        Determine if work should be generated proactively.
        
        Returns:
            True if work generation is recommended
        """
        # Always generate if below minimum
        if self.needs_work():
            return True
            
        # Check predicted depletion
        depletion_time = self.predict_depletion_time()
        if depletion_time and depletion_time < 10:  # Less than 10 minutes
            self.logger.info(f"Queue predicted to deplete in {depletion_time:.1f} minutes")
            return True
            
        # Generate if below optimal and consumption is high
        if self.current_size < self.optimal_queue_size and self.consumption_rate > self.worker_count:
            return True
            
        return False
        
    def get_generation_size(self) -> int:
        """Calculate how many work items to generate."""
        if self.current_size < self.min_queue_size:
            # Emergency generation - get to optimal quickly
            return self.optimal_queue_size - self.current_size
        else:
            # Maintenance generation - top up to optimal
            return max(self.worker_count * 2, self.optimal_queue_size - self.current_size)
            
    def on_work_generated(self, callback: Callable):
        """Register callback for work generation events."""
        self.generation_callbacks.append(callback)
        
    async def notify_generation_needed(self, count: int):
        """Notify that work generation is needed."""
        for callback in self.generation_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(count)
                else:
                    callback(count)
            except Exception as e:
                self.logger.error(f"Error in generation callback: {e}")
                
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        depletion = self.predict_depletion_time()
        return {
            'current_size': self.current_size,
            'min_size': self.min_queue_size,
            'optimal_size': self.optimal_queue_size,
            'consumption_rate': f"{self.consumption_rate:.1f} items/min",
            'needs_work': self.needs_work(),
            'depletion_time': f"{depletion:.1f} min" if depletion else "unknown",
            'health': self._get_health_status()
        }
        
    def _get_health_status(self) -> str:
        """Get queue health status."""
        ratio = self.current_size / self.optimal_queue_size
        if ratio < 0.2:
            return "critical"
        elif ratio < 0.5:
            return "low"
        elif ratio < 1.5:
            return "healthy"
        else:
            return "overfull"