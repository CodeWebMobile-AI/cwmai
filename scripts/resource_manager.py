"""
Resource Manager for Continuous 24/7 AI System

Manages system resources, rate limiting, and load balancing
for optimal performance in parallel processing scenarios.
"""

import asyncio
import logging
import time
import psutil
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json


class ResourceType(Enum):
    """Types of resources managed by the system."""
    CPU = "cpu"
    MEMORY = "memory"
    API_CALLS = "api_calls"
    GITHUB_API = "github_api"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class ResourceLimit:
    """Resource usage limit definition."""
    resource_type: ResourceType
    max_usage: float  # Maximum allowed usage (percentage or absolute)
    current_usage: float = 0.0
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reset_interval: int = 3600  # Reset interval in seconds (1 hour default)
    
    def is_exceeded(self) -> bool:
        """Check if resource limit is exceeded."""
        return self.current_usage >= self.max_usage
    
    def should_reset(self) -> bool:
        """Check if usage should be reset based on interval."""
        now = datetime.now(timezone.utc)
        return (now - self.last_reset).total_seconds() >= self.reset_interval
    
    def reset_usage(self):
        """Reset usage counter."""
        self.current_usage = 0.0
        self.last_reset = datetime.now(timezone.utc)


@dataclass
class WorkerResourceUsage:
    """Resource usage tracking for a worker."""
    worker_id: str
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    api_calls_per_hour: int = 0
    github_api_calls: int = 0
    active_tasks: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ResourceManager:
    """Manages system resources for continuous AI operation."""
    
    def __init__(self):
        """Initialize the resource manager."""
        self.logger = logging.getLogger(__name__)
        
        # Resource limits
        self.limits: Dict[ResourceType, ResourceLimit] = {
            ResourceType.CPU: ResourceLimit(
                ResourceType.CPU,
                max_usage=80.0,  # 80% CPU usage limit
                reset_interval=300  # Reset every 5 minutes
            ),
            ResourceType.MEMORY: ResourceLimit(
                ResourceType.MEMORY,
                max_usage=85.0,  # 85% memory usage limit
                reset_interval=300  # Reset every 5 minutes
            ),
            ResourceType.API_CALLS: ResourceLimit(
                ResourceType.API_CALLS,
                max_usage=1000,  # 1000 API calls per hour
                reset_interval=3600  # Reset every hour
            ),
            ResourceType.GITHUB_API: ResourceLimit(
                ResourceType.GITHUB_API,
                max_usage=4000,  # GitHub API limit (5000 per hour, with buffer)
                reset_interval=3600  # Reset every hour
            ),
            ResourceType.STORAGE: ResourceLimit(
                ResourceType.STORAGE,
                max_usage=90.0,  # 90% storage usage limit
                reset_interval=3600  # Reset every hour
            )
        }
        
        # Worker resource tracking
        self.worker_usage: Dict[str, WorkerResourceUsage] = {}
        
        # Rate limiting queues
        self.rate_limit_queues: Dict[ResourceType, asyncio.Queue] = {}
        self.rate_limit_tasks: Dict[ResourceType, asyncio.Task] = {}
        
        # Performance metrics
        self.metrics = {
            'resource_checks': 0,
            'rate_limit_hits': 0,
            'resource_warnings': 0,
            'load_balancing_actions': 0,
            'system_throttling_events': 0
        }
        
        # System monitoring
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.metric_update_task: Optional[asyncio.Task] = None
        
        # State manager reference (will be set by orchestrator)
        self.state_manager = None
        
        # Performance tracking for efficiency calculation
        self.task_completion_times = []  # List of (timestamp, duration) tuples
        self.task_success_count = 0
        self.task_failure_count = 0
        self.last_efficiency_update = datetime.now(timezone.utc)
        
    async def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Initialize rate limiting
        await self._initialize_rate_limiting()
        
        # Start metric updates if state manager is available
        if self.state_manager:
            self.metric_update_task = asyncio.create_task(self._metric_update_loop())
        
        self.logger.info("Resource manager monitoring started")
    
    async def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.metric_update_task:
            self.metric_update_task.cancel()
            try:
                await self.metric_update_task
            except asyncio.CancelledError:
                pass
        
        # Stop rate limiting tasks
        for task in self.rate_limit_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Resource manager monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Update system resource usage
                await self._update_system_resources()
                
                # Check for resource limit violations
                await self._check_resource_limits()
                
                # Reset usage counters if needed
                self._reset_expired_counters()
                
                # Update metrics
                self.metrics['resource_checks'] += 1
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(60)  # Longer pause on error
    
    async def _update_system_resources(self):
        """Update current system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.limits[ResourceType.CPU].current_usage = cpu_percent
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.limits[ResourceType.MEMORY].current_usage = memory_percent
            
            # Storage usage
            disk = psutil.disk_usage('/')
            storage_percent = (disk.used / disk.total) * 100
            self.limits[ResourceType.STORAGE].current_usage = storage_percent
            
        except Exception as e:
            self.logger.error(f"Error updating system resources: {e}")
    
    async def _check_resource_limits(self):
        """Check if any resource limits are exceeded."""
        for resource_type, limit in self.limits.items():
            if limit.is_exceeded():
                self.metrics['resource_warnings'] += 1
                
                if resource_type in [ResourceType.CPU, ResourceType.MEMORY]:
                    # System resource limit exceeded
                    self.logger.warning(
                        f"{resource_type.value.upper()} usage at {limit.current_usage:.1f}% "
                        f"(limit: {limit.max_usage:.1f}%)"
                    )
                    
                    # Trigger throttling if needed
                    await self._handle_resource_pressure(resource_type)
                    
                elif resource_type in [ResourceType.API_CALLS, ResourceType.GITHUB_API]:
                    # API rate limit exceeded
                    self.logger.warning(
                        f"{resource_type.value} rate limit hit: {limit.current_usage} "
                        f"(limit: {limit.max_usage})"
                    )
                    self.metrics['rate_limit_hits'] += 1
    
    async def _handle_resource_pressure(self, resource_type: ResourceType):
        """Handle high resource usage by throttling operations."""
        if resource_type == ResourceType.CPU:
            # Introduce small delays to reduce CPU pressure
            self.logger.info("Applying CPU throttling - introducing delays")
            self.metrics['system_throttling_events'] += 1
            await asyncio.sleep(2)  # 2-second pause
            
        elif resource_type == ResourceType.MEMORY:
            # Trigger garbage collection and reduce concurrent operations
            self.logger.info("Applying memory throttling - reducing concurrency")
            import gc
            gc.collect()
            self.metrics['system_throttling_events'] += 1
    
    def _reset_expired_counters(self):
        """Reset usage counters that have expired."""
        for limit in self.limits.values():
            if limit.should_reset():
                old_usage = limit.current_usage
                limit.reset_usage()
                if old_usage > 0:
                    self.logger.debug(
                        f"Reset {limit.resource_type.value} usage counter "
                        f"(was: {old_usage}, limit: {limit.max_usage})"
                    )
    
    async def _initialize_rate_limiting(self):
        """Initialize rate limiting for API calls."""
        # GitHub API rate limiting
        self.rate_limit_queues[ResourceType.GITHUB_API] = asyncio.Queue()
        self.rate_limit_tasks[ResourceType.GITHUB_API] = asyncio.create_task(
            self._github_rate_limiter()
        )
        
        # General API rate limiting
        self.rate_limit_queues[ResourceType.API_CALLS] = asyncio.Queue()
        self.rate_limit_tasks[ResourceType.API_CALLS] = asyncio.create_task(
            self._api_rate_limiter()
        )
    
    async def _github_rate_limiter(self):
        """Rate limiter for GitHub API calls."""
        while self.monitoring_active:
            try:
                # GitHub allows 5000 requests per hour
                # That's roughly 1.4 requests per second
                # We'll be conservative and allow 1 request per second
                await asyncio.sleep(1)
                
                # Process queued GitHub API requests
                queue = self.rate_limit_queues[ResourceType.GITHUB_API]
                if not queue.empty():
                    callback = await queue.get()
                    if callable(callback):
                        await callback()
                    queue.task_done()
                    
            except Exception as e:
                self.logger.error(f"Error in GitHub rate limiter: {e}")
                await asyncio.sleep(5)
    
    async def _api_rate_limiter(self):
        """Rate limiter for general API calls."""
        while self.monitoring_active:
            try:
                # General rate limiting - allow 2 API calls per second
                await asyncio.sleep(0.5)
                
                # Process queued API requests
                queue = self.rate_limit_queues[ResourceType.API_CALLS]
                if not queue.empty():
                    callback = await queue.get()
                    if callable(callback):
                        await callback()
                    queue.task_done()
                    
            except Exception as e:
                self.logger.error(f"Error in API rate limiter: {e}")
                await asyncio.sleep(5)
    
    async def request_resource(self, resource_type: ResourceType, 
                             amount: float = 1.0) -> bool:
        """Request use of a resource.
        
        Args:
            resource_type: Type of resource to request
            amount: Amount of resource to request
            
        Returns:
            True if resource is available, False if rate limited
        """
        if resource_type not in self.limits:
            return True  # Unknown resource type, allow by default
        
        limit = self.limits[resource_type]
        
        # Check if adding this amount would exceed the limit
        if limit.current_usage + amount > limit.max_usage:
            self.logger.debug(
                f"Resource request denied: {resource_type.value} "
                f"({limit.current_usage + amount} > {limit.max_usage})"
            )
            return False
        
        # Grant the resource
        limit.current_usage += amount
        return True
    
    async def queue_rate_limited_operation(self, resource_type: ResourceType, 
                                         operation_callback: callable):
        """Queue an operation to be rate limited.
        
        Args:
            resource_type: Type of resource being rate limited
            operation_callback: Async function to call when rate limit allows
        """
        if resource_type in self.rate_limit_queues:
            await self.rate_limit_queues[resource_type].put(operation_callback)
        else:
            # No rate limiting for this resource type, execute immediately
            await operation_callback()
    
    def update_worker_usage(self, worker_id: str, cpu_percent: float = None,
                          memory_mb: float = None, api_calls: int = None,
                          github_calls: int = None, active_tasks: int = None):
        """Update resource usage for a specific worker.
        
        Args:
            worker_id: ID of the worker
            cpu_percent: CPU usage percentage
            memory_mb: Memory usage in MB
            api_calls: Number of API calls made
            github_calls: Number of GitHub API calls made
            active_tasks: Number of active tasks
        """
        if worker_id not in self.worker_usage:
            self.worker_usage[worker_id] = WorkerResourceUsage(worker_id)
        
        usage = self.worker_usage[worker_id]
        
        if cpu_percent is not None:
            usage.cpu_percent = cpu_percent
        if memory_mb is not None:
            usage.memory_mb = memory_mb
        if api_calls is not None:
            usage.api_calls_per_hour += api_calls
        if github_calls is not None:
            usage.github_api_calls += github_calls
        if active_tasks is not None:
            usage.active_tasks = active_tasks
            
        usage.last_updated = datetime.now(timezone.utc)
    
    def get_optimal_worker(self, workers: List[str], 
                          task_requirements: Dict[str, Any] = None) -> Optional[str]:
        """Get the optimal worker for a task based on current resource usage.
        
        Args:
            workers: List of available worker IDs
            task_requirements: Task resource requirements
            
        Returns:
            ID of the optimal worker, or None if no worker is suitable
        """
        if not workers:
            return None
        
        # Score workers based on current usage
        worker_scores = {}
        
        for worker_id in workers:
            if worker_id not in self.worker_usage:
                # New worker, give it high priority
                worker_scores[worker_id] = 1000
                continue
            
            usage = self.worker_usage[worker_id]
            
            # Calculate score (higher is better)
            # Lower CPU/memory usage = higher score
            # Fewer active tasks = higher score
            score = (
                (100 - usage.cpu_percent) * 2 +  # CPU weight: 2x
                (100 - min(usage.memory_mb / 100, 100)) * 1.5 +  # Memory weight: 1.5x
                (10 - min(usage.active_tasks, 10)) * 3  # Active tasks weight: 3x
            )
            
            worker_scores[worker_id] = score
        
        # Return worker with highest score
        optimal_worker = max(worker_scores.items(), key=lambda x: x[1])[0]
        
        self.metrics['load_balancing_actions'] += 1
        
        return optimal_worker
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics.
        
        Returns:
            Dictionary with system health information
        """
        health_score = 100.0
        
        # Reduce score for high resource usage
        for resource_type, limit in self.limits.items():
            if resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.STORAGE]:
                usage_percent = limit.current_usage / limit.max_usage * 100
                if usage_percent > 80:
                    health_score -= (usage_percent - 80) * 2  # Penalty for high usage
        
        # Determine health status
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 60:
            status = "fair"
        elif health_score >= 40:
            status = "poor"
        else:
            status = "critical"
        
        return {
            'health_score': max(0, health_score),
            'status': status,
            'resource_usage': {
                resource_type.value: {
                    'current': limit.current_usage,
                    'limit': limit.max_usage,
                    'percentage': min(100, (limit.current_usage / limit.max_usage) * 100)
                }
                for resource_type, limit in self.limits.items()
            },
            'worker_count': len(self.worker_usage),
            'metrics': self.metrics.copy(),
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
    
    def get_resource_recommendations(self) -> List[str]:
        """Get recommendations for resource optimization.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check CPU usage
        cpu_usage = self.limits[ResourceType.CPU].current_usage
        if cpu_usage > 90:
            recommendations.append("Consider reducing parallel workers - CPU usage very high")
        elif cpu_usage > 75:
            recommendations.append("Monitor CPU usage - consider throttling operations")
        
        # Check memory usage
        memory_usage = self.limits[ResourceType.MEMORY].current_usage
        if memory_usage > 90:
            recommendations.append("Memory usage critical - consider restarting workers")
        elif memory_usage > 80:
            recommendations.append("Memory usage high - monitor for memory leaks")
        
        # Check API rate limits
        api_usage = self.limits[ResourceType.API_CALLS].current_usage
        api_limit = self.limits[ResourceType.API_CALLS].max_usage
        if api_usage > api_limit * 0.9:
            recommendations.append("API rate limit approaching - reduce API-heavy operations")
        
        # Check GitHub API usage
        github_usage = self.limits[ResourceType.GITHUB_API].current_usage
        github_limit = self.limits[ResourceType.GITHUB_API].max_usage
        if github_usage > github_limit * 0.9:
            recommendations.append("GitHub API rate limit approaching - throttle GitHub operations")
        
        # Check worker distribution
        if len(self.worker_usage) > 1:
            cpu_usages = [usage.cpu_percent for usage in self.worker_usage.values()]
            if max(cpu_usages) - min(cpu_usages) > 30:
                recommendations.append("Uneven worker load distribution - consider rebalancing")
        
        if not recommendations:
            recommendations.append("System resources are well balanced")
        
        return recommendations
    
    def set_state_manager(self, state_manager):
        """Set reference to state manager for metric updates."""
        self.state_manager = state_manager
        
        # Start metric update task if monitoring is active
        if self.monitoring_active and not self.metric_update_task:
            self.metric_update_task = asyncio.create_task(self._metric_update_loop())
    
    async def _metric_update_loop(self):
        """Periodically update system metrics in state."""
        while self.monitoring_active and self.state_manager:
            try:
                # Calculate and update efficiency metrics
                await self._update_efficiency_metrics()
                
                # Wait 5 minutes before next update
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(60)
    
    async def _update_efficiency_metrics(self):
        """Calculate and update resource efficiency metrics."""
        try:
            # Calculate efficiency based on:
            # 1. Task completion rate
            # 2. Resource utilization
            # 3. Error rate
            # 4. Average completion time
            
            now = datetime.now(timezone.utc)
            time_window = 3600  # 1 hour window
            
            # Filter recent task completions
            recent_completions = [
                (ts, duration) for ts, duration in self.task_completion_times
                if (now - datetime.fromtimestamp(ts, timezone.utc)).total_seconds() < time_window
            ]
            
            # Calculate metrics
            total_tasks = self.task_success_count + self.task_failure_count
            success_rate = self.task_success_count / max(total_tasks, 1)
            
            # Average completion time
            avg_completion_time = 0
            if recent_completions:
                avg_completion_time = sum(d for _, d in recent_completions) / len(recent_completions)
            
            # Resource utilization efficiency
            cpu_efficiency = 1.0 - (self.limits[ResourceType.CPU].current_usage / 100.0)
            memory_efficiency = 1.0 - (self.limits[ResourceType.MEMORY].current_usage / 100.0)
            
            # Worker utilization
            active_workers = sum(1 for w in self.worker_usage.values() if w.active_tasks > 0)
            total_workers = len(self.worker_usage)
            worker_efficiency = active_workers / max(total_workers, 1)
            
            # Calculate overall efficiency (weighted average)
            efficiency = (
                success_rate * 0.3 +  # 30% weight on success rate
                worker_efficiency * 0.3 +  # 30% weight on worker utilization
                cpu_efficiency * 0.2 +  # 20% weight on CPU efficiency
                memory_efficiency * 0.1 +  # 10% weight on memory efficiency
                min(1.0, 60.0 / max(avg_completion_time, 1)) * 0.1  # 10% weight on speed
            )
            
            # Ensure efficiency is between 0 and 1
            efficiency = max(0.0, min(1.0, efficiency))
            
            # Update state
            if self.state_manager:
                state = self.state_manager.load_state()
                if 'system_performance' not in state:
                    state['system_performance'] = {}
                if 'learning_metrics' not in state['system_performance']:
                    state['system_performance']['learning_metrics'] = {}
                
                # Update metrics
                state['system_performance']['learning_metrics']['resource_efficiency'] = round(efficiency, 3)
                state['system_performance']['learning_metrics']['task_success_rate'] = round(success_rate, 3)
                state['system_performance']['learning_metrics']['avg_completion_time'] = round(avg_completion_time, 2)
                state['system_performance']['learning_metrics']['worker_utilization'] = round(worker_efficiency, 3)
                state['system_performance']['learning_metrics']['last_updated'] = now.isoformat()
                
                # Add detailed metrics
                state['system_performance']['resource_metrics'] = {
                    'cpu_usage': round(self.limits[ResourceType.CPU].current_usage, 2),
                    'memory_usage': round(self.limits[ResourceType.MEMORY].current_usage, 2),
                    'storage_usage': round(self.limits[ResourceType.STORAGE].current_usage, 2),
                    'active_workers': active_workers,
                    'total_workers': total_workers
                }
                
                # Update state first, then save
                self.state_manager.state = state
                self.state_manager.save_state()
                
                self.logger.info(
                    f"Updated efficiency metrics: {efficiency:.1%} "
                    f"(success: {success_rate:.1%}, workers: {worker_efficiency:.1%})"
                )
                
                self.last_efficiency_update = now
                
        except Exception as e:
            self.logger.error(f"Error calculating efficiency metrics: {e}")
    
    def record_task_completion(self, duration: float, success: bool = True):
        """Record a task completion for metric calculation.
        
        Args:
            duration: Time taken to complete the task in seconds
            success: Whether the task completed successfully
        """
        timestamp = time.time()
        self.task_completion_times.append((timestamp, duration))
        
        # Keep only recent completions (last 24 hours)
        cutoff = timestamp - 86400
        self.task_completion_times = [
            (ts, d) for ts, d in self.task_completion_times if ts > cutoff
        ]
        
        # Update counters
        if success:
            self.task_success_count += 1
        else:
            self.task_failure_count += 1
        
        # Trigger immediate update if efficiency is very low
        if self.state_manager:
            state = self.state_manager.load_state()
            current_efficiency = state.get('system_performance', {}).get('learning_metrics', {}).get('resource_efficiency', 0.5)
            
            if current_efficiency < 0.1:  # Less than 10% efficiency
                asyncio.create_task(self._update_efficiency_metrics())
    
    async def _metric_update_loop(self):
        """Periodically update resource efficiency metrics in state."""
        while self.monitoring_active:
            try:
                await self.update_resource_metrics()
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                self.logger.error(f"Error updating resource metrics: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def update_resource_metrics(self):
        """Dynamically calculate and update resource efficiency in state."""
        if not self.state_manager:
            return
        
        try:
            # Get current resource usage
            cpu_usage = self.limits[ResourceType.CPU].current_usage
            memory_usage = self.limits[ResourceType.MEMORY].current_usage
            
            # Calculate efficiency (inverse of usage with weighting)
            # Lower usage = higher efficiency
            # CPU weighted more heavily than memory
            cpu_efficiency = max(0.0, (100 - cpu_usage) / 100)
            memory_efficiency = max(0.0, (100 - memory_usage) / 100)
            
            # Weighted average: 60% CPU, 40% memory
            overall_efficiency = (cpu_efficiency * 0.6) + (memory_efficiency * 0.4)
            
            # Apply smoothing to avoid sudden jumps
            current_state = self.state_manager.get_state()
            if 'metrics' in current_state:
                old_efficiency = current_state['metrics'].get('resource_efficiency', 0.0)
                # Smooth with 70% old value, 30% new value
                overall_efficiency = (old_efficiency * 0.7) + (overall_efficiency * 0.3)
            
            # Update state with new metrics
            metrics_update = {
                'metrics': {
                    'resource_efficiency': round(overall_efficiency, 3),
                    'cpu_usage': round(cpu_usage, 1),
                    'memory_usage': round(memory_usage, 1),
                    'last_updated': datetime.now(timezone.utc).isoformat(),
                    'health_score': self.get_system_health()['health_score']
                }
            }
            
            self.state_manager.update_state(metrics_update)
            
            self.logger.debug(
                f"Updated resource metrics - Efficiency: {overall_efficiency:.3f}, "
                f"CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update resource metrics: {e}")
    
    def set_state_manager(self, state_manager):
        """Set the state manager reference for metric updates.
        
        Args:
            state_manager: StateManager instance
        """
        self.state_manager = state_manager
        self.logger.info("State manager connected to resource manager")