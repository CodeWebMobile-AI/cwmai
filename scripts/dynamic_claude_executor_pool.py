"""
Dynamic Claude Executor Pool

An enhanced version of ClaudeExecutorPool that automatically scales the number of executors
based on workload metrics and queue depth.

Key features:
- Dynamic scaling based on queue depth and wait times
- Configurable min/max executors
- Automatic scale down when idle
- Detailed scaling metrics and logging
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import statistics

from scripts.claude_execution_queue import ClaudeExecutionQueue, ClaudeTask
from scripts.claude_executor import ClaudeExecutor
from scripts.local_github_simulator import LocalGitHubSimulator


@dataclass
class ScalingMetrics:
    """Tracks metrics used for scaling decisions."""
    queue_depth_history: List[Tuple[datetime, int]] = field(default_factory=list)
    wait_time_history: List[Tuple[datetime, float]] = field(default_factory=list)
    executor_utilization: Dict[str, List[Tuple[datetime, float]]] = field(default_factory=dict)
    scaling_events: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_queue_depth(self, depth: int):
        """Record current queue depth."""
        now = datetime.now(timezone.utc)
        self.queue_depth_history.append((now, depth))
        # Keep only last hour
        cutoff = now - timedelta(hours=1)
        self.queue_depth_history = [(t, d) for t, d in self.queue_depth_history if t > cutoff]
    
    def add_wait_time(self, wait_seconds: float):
        """Record task wait time."""
        now = datetime.now(timezone.utc)
        self.wait_time_history.append((now, wait_seconds))
        # Keep only last hour
        cutoff = now - timedelta(hours=1)
        self.wait_time_history = [(t, w) for t, w in self.wait_time_history if t > cutoff]
    
    def add_utilization(self, executor_id: str, utilization: float):
        """Record executor utilization."""
        now = datetime.now(timezone.utc)
        if executor_id not in self.executor_utilization:
            self.executor_utilization[executor_id] = []
        self.executor_utilization[executor_id].append((now, utilization))
        # Keep only last hour
        cutoff = now - timedelta(hours=1)
        self.executor_utilization[executor_id] = [
            (t, u) for t, u in self.executor_utilization[executor_id] if t > cutoff
        ]
    
    def add_scaling_event(self, event_type: str, from_count: int, to_count: int, reason: str):
        """Record a scaling event."""
        self.scaling_events.append({
            'timestamp': datetime.now(timezone.utc),
            'type': event_type,
            'from_count': from_count,
            'to_count': to_count,
            'reason': reason
        })
        # Keep only last 100 events
        self.scaling_events = self.scaling_events[-100:]
    
    def get_avg_queue_depth(self, minutes: int = 5) -> float:
        """Get average queue depth over the last N minutes."""
        if not self.queue_depth_history:
            return 0
        
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        recent = [(t, d) for t, d in self.queue_depth_history if t > cutoff]
        
        if not recent:
            return 0
        
        return statistics.mean(d for _, d in recent)
    
    def get_avg_wait_time(self, minutes: int = 5) -> float:
        """Get average wait time over the last N minutes."""
        if not self.wait_time_history:
            return 0
        
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        recent = [(t, w) for t, w in self.wait_time_history if t > cutoff]
        
        if not recent:
            return 0
        
        return statistics.mean(w for _, w in recent)
    
    def get_avg_utilization(self, minutes: int = 5) -> float:
        """Get average utilization across all executors."""
        if not self.executor_utilization:
            return 0
        
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        all_utils = []
        
        for executor_utils in self.executor_utilization.values():
            recent = [(t, u) for t, u in executor_utils if t > cutoff]
            if recent:
                all_utils.extend(u for _, u in recent)
        
        if not all_utils:
            return 0
        
        return statistics.mean(all_utils)


class DynamicClaudeExecutorPool:
    """A Claude executor pool that dynamically scales based on workload."""
    
    def __init__(self, 
                 min_executors: int = 3,
                 max_executors: int = 15,
                 queue: Optional[ClaudeExecutionQueue] = None,
                 scale_up_threshold: float = 5.0,  # Queue items per executor
                 scale_down_threshold: float = 0.5,  # Utilization threshold
                 scale_check_interval: int = 30,  # Seconds between scaling checks
                 idle_timeout: int = 300):  # Seconds before scaling down idle executors
        """Initialize the dynamic executor pool.
        
        Args:
            min_executors: Minimum number of executors to maintain
            max_executors: Maximum number of executors allowed
            queue: Shared execution queue (created if not provided)
            scale_up_threshold: Queue depth per executor to trigger scale up
            scale_down_threshold: Utilization threshold to trigger scale down
            scale_check_interval: Seconds between scaling checks
            idle_timeout: Seconds of idle time before scaling down
        """
        self.min_executors = max(1, min_executors)
        self.max_executors = max(self.min_executors, max_executors)
        self.queue = queue or ClaudeExecutionQueue()
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_check_interval = scale_check_interval
        self.idle_timeout = idle_timeout
        
        self.executors: Dict[str, ClaudeExecutor] = {}
        self.executor_tasks: Dict[str, asyncio.Task] = {}
        self.executor_start_times: Dict[str, datetime] = {}
        self.executor_last_active: Dict[str, datetime] = {}
        
        self.metrics = ScalingMetrics()
        self.logger = logging.getLogger(__name__)
        self.scaling_task: Optional[asyncio.Task] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        self.shutdown_requested = False
        
        # Track current target size
        self.current_executors = 0
        self.target_executors = self.min_executors
    
    async def start(self):
        """Start the executor pool with minimum executors."""
        self.logger.info(f"Starting dynamic Claude executor pool (min={self.min_executors}, max={self.max_executors})")
        
        # Start with minimum executors
        await self._scale_to(self.min_executors)
        
        # Start scaling and monitoring tasks
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info(f"Dynamic pool started with {self.current_executors} executors")
    
    async def stop(self):
        """Stop all executors and scaling tasks."""
        self.logger.info("Stopping dynamic Claude executor pool")
        self.shutdown_requested = True
        
        # Cancel scaling and monitoring tasks
        if self.scaling_task:
            self.scaling_task.cancel()
            try:
                await self.scaling_task
            except asyncio.CancelledError:
                pass
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Stop all executors
        await self._scale_to(0)
        
        self.logger.info("Dynamic Claude executor pool stopped")
    
    async def _scale_to(self, target_count: int):
        """Scale to a specific number of executors.
        
        Args:
            target_count: Target number of executors
        """
        current_count = len(self.executors)
        
        if target_count == current_count:
            return
        
        if target_count > current_count:
            # Scale up
            to_add = target_count - current_count
            self.logger.info(f"Scaling up: adding {to_add} executors (current: {current_count}, target: {target_count})")
            
            for i in range(to_add):
                executor_id = f"claude_executor_{self.current_executors:03d}"
                await self._start_executor(executor_id)
                self.current_executors += 1
            
            self.metrics.add_scaling_event("scale_up", current_count, target_count, 
                                         f"Added {to_add} executors")
        else:
            # Scale down
            to_remove = current_count - target_count
            self.logger.info(f"Scaling down: removing {to_remove} executors (current: {current_count}, target: {target_count})")
            
            # Remove the most recently added executors first (LIFO)
            executors_to_remove = sorted(self.executors.keys(), reverse=True)[:to_remove]
            
            for executor_id in executors_to_remove:
                await self._stop_executor(executor_id)
            
            self.metrics.add_scaling_event("scale_down", current_count, target_count,
                                         f"Removed {to_remove} executors")
    
    async def _start_executor(self, executor_id: str):
        """Start a single executor.
        
        Args:
            executor_id: ID for the new executor
        """
        executor = ClaudeExecutor(executor_id, self.queue)
        self.executors[executor_id] = executor
        self.executor_start_times[executor_id] = datetime.now(timezone.utc)
        self.executor_last_active[executor_id] = datetime.now(timezone.utc)
        
        # Start executor task
        self.executor_tasks[executor_id] = asyncio.create_task(executor.run())
        
        self.logger.info(f"Started executor {executor_id}")
    
    async def _stop_executor(self, executor_id: str):
        """Stop a single executor.
        
        Args:
            executor_id: ID of the executor to stop
        """
        if executor_id not in self.executors:
            return
        
        executor = self.executors[executor_id]
        
        # Don't stop if executor is currently processing a task
        if executor.current_task:
            self.logger.warning(f"Refusing to stop executor {executor_id} - currently processing task")
            return
        
        # Request shutdown
        await executor.shutdown()
        
        # Wait for task to complete
        if executor_id in self.executor_tasks:
            task = self.executor_tasks[executor_id]
            try:
                await asyncio.wait_for(task, timeout=30)
            except asyncio.TimeoutError:
                self.logger.warning(f"Executor {executor_id} did not stop gracefully, cancelling")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Clean up
        self.executors.pop(executor_id, None)
        self.executor_tasks.pop(executor_id, None)
        self.executor_start_times.pop(executor_id, None)
        self.executor_last_active.pop(executor_id, None)
        
        self.logger.info(f"Stopped executor {executor_id}")
    
    async def _scaling_loop(self):
        """Main scaling loop that checks metrics and adjusts executor count."""
        while not self.shutdown_requested:
            try:
                await asyncio.sleep(self.scale_check_interval)
                
                if self.shutdown_requested:
                    break
                
                # Get current metrics
                queue_stats = await self.queue.get_stats()
                pending_tasks = queue_stats['pending']
                processing_tasks = queue_stats['processing']
                avg_execution_time = queue_stats.get('avg_execution_time', 180)  # Default 3 minutes
                
                current_count = len(self.executors)
                
                # Record metrics
                self.metrics.add_queue_depth(pending_tasks)
                
                # Calculate metrics for scaling decisions
                avg_queue_depth = self.metrics.get_avg_queue_depth(minutes=5)
                # Get wait time from queue stats instead of separate tracking
                avg_wait_time = queue_stats.get('avg_wait_time', 0)
                avg_utilization = self._calculate_current_utilization()
                
                # Log current state
                self.logger.debug(f"Scaling check: executors={current_count}, pending={pending_tasks}, "
                                f"processing={processing_tasks}, avg_queue={avg_queue_depth:.1f}, "
                                f"avg_wait={avg_wait_time:.1f}s, utilization={avg_utilization:.1%}")
                
                # Determine if we should scale
                scale_decision = self._make_scaling_decision(
                    current_count, pending_tasks, processing_tasks,
                    avg_queue_depth, avg_wait_time, avg_utilization, avg_execution_time
                )
                
                if scale_decision != current_count:
                    await self._scale_to(scale_decision)
                
            except Exception as e:
                self.logger.error(f"Error in scaling loop: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
    
    async def _monitoring_loop(self):
        """Monitor executor activity and track metrics."""
        while not self.shutdown_requested:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                if self.shutdown_requested:
                    break
                
                # Update executor activity
                for executor_id, executor in self.executors.items():
                    if executor.current_task:
                        self.executor_last_active[executor_id] = datetime.now(timezone.utc)
                    
                    # Calculate utilization
                    utilization = 1.0 if executor.current_task else 0.0
                    self.metrics.add_utilization(executor_id, utilization)
                
                # Check for tasks that started waiting
                for task in self.queue._queue:
                    if task.created_at:
                        wait_time = (datetime.now(timezone.utc) - task.created_at).total_seconds()
                        if wait_time > 0:
                            self.metrics.add_wait_time(wait_time)
                
                # Save stats to file for external monitoring
                self._save_stats_to_file()
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _save_stats_to_file(self):
        """Save current stats to a file for external monitoring."""
        try:
            stats = self.get_stats()
            # Convert datetime objects to ISO format for JSON serialization
            for event in stats['scaling_metrics']['recent_scaling_events']:
                if isinstance(event.get('timestamp'), datetime):
                    event['timestamp'] = event['timestamp'].isoformat()
            
            with open("dynamic_claude_pool_stats.json", "w") as f:
                json.dump(stats, f, indent=2, default=str)
        except Exception as e:
            self.logger.debug(f"Failed to save stats to file: {e}")
    
    def _calculate_current_utilization(self) -> float:
        """Calculate current utilization across all executors."""
        if not self.executors:
            return 0.0
        
        busy_count = sum(1 for e in self.executors.values() if e.current_task)
        return busy_count / len(self.executors)
    
    def _make_scaling_decision(self, current_count: int, pending_tasks: int, 
                              processing_tasks: int, avg_queue_depth: float,
                              avg_wait_time: float, avg_utilization: float,
                              avg_execution_time: float) -> int:
        """Make a scaling decision based on current metrics.
        
        Returns:
            Target number of executors
        """
        # Don't scale if we're at limits
        if current_count <= 0:
            return self.min_executors
        
        # Scale up conditions
        scale_up_reasons = []
        
        # 1. High queue depth per executor
        queue_per_executor = pending_tasks / current_count if current_count > 0 else pending_tasks
        if queue_per_executor > self.scale_up_threshold:
            scale_up_reasons.append(f"high queue depth ({queue_per_executor:.1f} per executor)")
        
        # 2. Long average wait times (> 5 minutes)
        if avg_wait_time > 300:
            scale_up_reasons.append(f"long wait times ({avg_wait_time:.0f}s)")
        
        # 3. High utilization (> 90%) with pending tasks
        if avg_utilization > 0.9 and pending_tasks > 0:
            scale_up_reasons.append(f"high utilization ({avg_utilization:.1%}) with pending tasks")
        
        # 4. Expected wait time for new tasks is too long
        expected_wait = (processing_tasks * avg_execution_time) / current_count if current_count > 0 else 0
        if expected_wait > 600 and pending_tasks > 0:  # > 10 minutes expected wait
            scale_up_reasons.append(f"long expected wait ({expected_wait:.0f}s)")
        
        if scale_up_reasons and current_count < self.max_executors:
            # Scale up by 1-3 executors based on severity
            if len(scale_up_reasons) >= 3 or queue_per_executor > self.scale_up_threshold * 2:
                increment = min(3, self.max_executors - current_count)
            elif len(scale_up_reasons) >= 2:
                increment = min(2, self.max_executors - current_count)
            else:
                increment = min(1, self.max_executors - current_count)
            
            new_count = current_count + increment
            self.logger.info(f"Scaling up due to: {', '.join(scale_up_reasons)}")
            return new_count
        
        # Scale down conditions
        scale_down_reasons = []
        
        # 1. Low utilization with no pending tasks
        if avg_utilization < self.scale_down_threshold and pending_tasks == 0:
            scale_down_reasons.append(f"low utilization ({avg_utilization:.1%}) with no pending tasks")
        
        # 2. Idle executors beyond minimum (only if no pending tasks)
        if current_count > self.min_executors and pending_tasks == 0:
            idle_executors = self._count_idle_executors()
            if idle_executors > 0:
                scale_down_reasons.append(f"{idle_executors} idle executors")
        
        # 3. Very low queue depth (only if no pending tasks)
        if avg_queue_depth < 0.5 and current_count > self.min_executors and pending_tasks == 0:
            scale_down_reasons.append(f"very low queue depth ({avg_queue_depth:.1f})")
        
        # Only scale down if there are NO pending tasks
        if scale_down_reasons and current_count > self.min_executors and pending_tasks == 0:
            # Scale down by 1 executor at a time
            new_count = max(self.min_executors, current_count - 1)
            self.logger.info(f"Scaling down due to: {', '.join(scale_down_reasons)}")
            return new_count
        
        # No scaling needed
        return current_count
    
    def _count_idle_executors(self) -> int:
        """Count executors that have been idle beyond the timeout."""
        idle_count = 0
        now = datetime.now(timezone.utc)
        
        for executor_id, last_active in self.executor_last_active.items():
            if (now - last_active).total_seconds() > self.idle_timeout:
                executor = self.executors.get(executor_id)
                if executor and not executor.current_task:
                    idle_count += 1
        
        return idle_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics including scaling metrics.
        
        Returns:
            Pool statistics with scaling information
        """
        stats = {
            'num_executors': len(self.executors),
            'min_executors': self.min_executors,
            'max_executors': self.max_executors,
            'executors': {},
            'scaling_metrics': {
                'avg_queue_depth_5min': self.metrics.get_avg_queue_depth(5),
                'avg_wait_time_5min': self.metrics.get_avg_wait_time(5),
                'avg_utilization_5min': self.metrics.get_avg_utilization(5),
                'current_utilization': self._calculate_current_utilization(),
                'recent_scaling_events': self.metrics.scaling_events[-5:]  # Last 5 events
            }
        }
        
        # Individual executor stats
        for executor_id, executor in self.executors.items():
            executor_stats = executor.get_stats()
            executor_stats['idle_time'] = (
                datetime.now(timezone.utc) - self.executor_last_active.get(executor_id, datetime.now(timezone.utc))
            ).total_seconds()
            executor_stats['uptime'] = (
                datetime.now(timezone.utc) - self.executor_start_times.get(executor_id, datetime.now(timezone.utc))
            ).total_seconds()
            stats['executors'][executor_id] = executor_stats
        
        # Aggregate stats
        total_completed = sum(e.stats['tasks_completed'] for e in self.executors.values())
        total_failed = sum(e.stats['tasks_failed'] for e in self.executors.values())
        total_time = sum(e.stats['total_execution_time'] for e in self.executors.values())
        
        stats['total_completed'] = total_completed
        stats['total_failed'] = total_failed
        
        if total_completed > 0:
            stats['avg_execution_time'] = total_time / total_completed
        else:
            stats['avg_execution_time'] = 0
        
        return stats