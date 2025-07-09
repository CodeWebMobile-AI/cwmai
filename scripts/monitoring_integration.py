"""
Integration module for Task Monitoring System

Provides easy integration points for the monitoring system
into existing components like workers and orchestrators.
"""

import asyncio
import time
from typing import Dict, Any, Optional, Callable, List
from functools import wraps

from scripts.task_monitoring_system import get_monitoring_system
from scripts.work_item_types import WorkItem


class MonitoringIntegration:
    """Helper class for integrating monitoring into existing systems."""
    
    def __init__(self):
        """Initialize the integration helper."""
        self.monitor = get_monitoring_system()
    
    def monitor_task_execution(self, func: Callable) -> Callable:
        """Decorator to monitor task execution."""
        @wraps(func)
        async def wrapper(self, work_item: WorkItem, *args, **kwargs):
            # Start monitoring
            task_id = work_item.id
            task_type = work_item.task_type
            repository = work_item.repository
            
            # Start task monitoring
            metrics = self.monitor.start_task_monitoring(
                task_id=task_id,
                task_type=task_type,
                repository=repository,
                estimated_duration=getattr(work_item, 'estimated_duration', 14400)
            )
            
            # Record lifecycle event
            self.monitor.record_lifecycle_event(task_id, "started", {
                "worker_id": getattr(self, 'worker_id', 'unknown'),
                "priority": work_item.priority.value
            })
            
            # Start resource monitoring
            resource_task = asyncio.create_task(self._monitor_resources(task_id))
            
            try:
                # Execute the actual function
                result = await func(self, work_item, *args, **kwargs)
                
                # Task completed successfully
                self.monitor.complete_task_monitoring(task_id, success=True)
                self.monitor.record_lifecycle_event(task_id, "completed", {
                    "result": "success"
                })
                
                # Update worker performance
                if hasattr(self, 'worker_id'):
                    duration = time.time() - time.mktime(
                        time.strptime(metrics.started_at[:19], "%Y-%m-%dT%H:%M:%S")
                    )
                    self.monitor.update_worker_performance(
                        self.worker_id, task_id, task_type, True, duration
                    )
                
                return result
                
            except Exception as e:
                # Task failed
                self.monitor.complete_task_monitoring(task_id, success=False, error=str(e))
                self.monitor.record_lifecycle_event(task_id, "failed", {
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                
                # Update worker performance
                if hasattr(self, 'worker_id'):
                    duration = time.time() - time.mktime(
                        time.strptime(metrics.started_at[:19], "%Y-%m-%dT%H:%M:%S")
                    )
                    self.monitor.update_worker_performance(
                        self.worker_id, task_id, task_type, False, duration
                    )
                
                raise
                
            finally:
                # Stop resource monitoring
                resource_task.cancel()
                try:
                    await resource_task
                except asyncio.CancelledError:
                    pass
        
        return wrapper
    
    async def _monitor_resources(self, task_id: str):
        """Monitor resources during task execution."""
        while True:
            try:
                self.monitor.record_resource_usage(task_id)
                await asyncio.sleep(5)  # Check every 5 seconds
            except asyncio.CancelledError:
                break
    
    def track_execution_step(self, task_id: str, step_name: str):
        """Context manager for tracking execution steps."""
        class StepTracker:
            def __init__(self, monitor, task_id, step_name):
                self.monitor = monitor
                self.task_id = task_id
                self.step_name = step_name
                self.step = None
            
            def __enter__(self):
                self.step = self.monitor.start_execution_step(
                    self.task_id, self.step_name
                )
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    self.monitor.complete_execution_step(
                        self.task_id, self.step_name, success=True
                    )
                else:
                    self.monitor.complete_execution_step(
                        self.task_id, self.step_name, success=False,
                        error=str(exc_val)
                    )
                return False
        
        return StepTracker(self.monitor, task_id, step_name)
    
    def track_api_call(self, task_id: str, provider: str):
        """Track an API call."""
        self.monitor.record_api_call(task_id, provider)
    
    def add_task_dependency(self, parent_id: str, child_id: str, dep_type: str = "blocks"):
        """Add a dependency between tasks."""
        self.monitor.add_dependency(parent_id, child_id, dep_type)
    
    def track_claude_execution(self, task_id: str, success: bool, duration: float):
        """Track Claude Code CLI execution."""
        self.monitor.update_task_metrics(task_id, {
            "claude_execution_attempted": True,
            "claude_execution_success": success,
            "claude_execution_time": duration
        })
    
    def track_retry(self, task_id: str):
        """Track a retry attempt."""
        if task_id in self.monitor.active_tasks:
            self.monitor.active_tasks[task_id].retry_count += 1
    
    def predict_task_duration(self, task_type: str, repository: str) -> float:
        """Get predicted duration for a task."""
        return self.monitor.predict_task_duration(task_type, repository)
    
    def get_failure_risk(self, task_type: str) -> float:
        """Get failure risk assessment."""
        return self.monitor.calculate_failure_risk(task_type)
    
    def get_best_worker(self, task_type: str) -> Optional[str]:
        """Get the best worker for a task type."""
        return self.monitor.get_best_worker_for_task(task_type)
    
    def get_optimal_schedule_time(self, task_type: str) -> Dict[str, Any]:
        """Get optimal scheduling time for a task."""
        return self.monitor.get_optimal_scheduling_time(task_type)
    
    async def measure_quality(self, task_id: str, repo_path: str):
        """Measure code quality after task completion."""
        return await self.monitor.measure_quality_metrics(task_id, repo_path)
    
    def track_lifecycle_event(self, task_id: str, event: str, metadata: Optional[Dict] = None):
        """Track a custom lifecycle event."""
        self.monitor.record_lifecycle_event(task_id, event, metadata)
    
    def get_blocking_tasks(self, task_id: str) -> List[str]:
        """Get tasks that are blocking this task."""
        return self.monitor.get_blocking_tasks(task_id)
    
    
    def get_task_report(self, task_id: str) -> Dict[str, Any]:
        """Get comprehensive report for a task."""
        return self.monitor.generate_task_report(task_id)
    
    def get_system_insights(self) -> Dict[str, Any]:
        """Get system-wide insights."""
        return self.monitor.get_system_insights()


# Example integration for AI Worker
class MonitoredAIWorker:
    """Example of how to integrate monitoring into an AI Worker."""
    
    def __init__(self, worker_id: str):
        """Initialize the monitored worker."""
        self.worker_id = worker_id
        self.monitoring = MonitoringIntegration()
    
    @MonitoringIntegration().monitor_task_execution
    async def execute_work_item(self, work_item: WorkItem) -> Dict[str, Any]:
        """Execute a work item with full monitoring."""
        task_id = work_item.id
        
        # Track execution steps
        with self.monitoring.track_execution_step(task_id, "validation"):
            # Validate work item
            if not work_item.title:
                raise ValueError("Work item missing title")
        
        with self.monitoring.track_execution_step(task_id, "preparation"):
            # Check for blocking tasks
            blocking = self.monitoring.get_blocking_tasks(task_id)
            if blocking:
                self.monitoring.track_lifecycle_event(task_id, "blocked", {
                    "blocking_tasks": blocking
                })
                raise Exception(f"Task blocked by: {blocking}")
        
        with self.monitoring.track_execution_step(task_id, "ai_processing"):
            # Simulate AI processing
            self.monitoring.track_api_call(task_id, "openai")
            await asyncio.sleep(2)  # Simulate processing
        
        with self.monitoring.track_execution_step(task_id, "claude_execution"):
            # Track Claude execution
            start_time = time.time()
            success = True  # Simulate success
            duration = time.time() - start_time
            self.monitoring.track_claude_execution(task_id, success, duration)
        
        # Return result
        return {
            "success": True,
            "task_id": task_id,
            "worker_id": self.worker_id
        }


# Monitoring hooks for existing systems
def integrate_monitoring_into_orchestrator(orchestrator):
    """Add monitoring hooks to continuous orchestrator."""
    monitor = MonitoringIntegration()
    
    # Hook into work assignment
    original_assign = orchestrator._assign_work_to_worker
    
    async def monitored_assign(worker_id: str, work_item: WorkItem):
        # Predict duration
        predicted_duration = monitor.predict_task_duration(
            work_item.task_type, work_item.repository
        )
        work_item.estimated_duration = predicted_duration
        
        # Get best worker suggestion
        best_worker = monitor.get_best_worker(work_item.task_type)
        if best_worker and best_worker != worker_id:
            orchestrator.logger.info(
                f"Note: {best_worker} might be better for {work_item.task_type} tasks"
            )
        
        # Check failure risk
        risk = monitor.get_failure_risk(work_item.task_type)
        if risk > 0.7:
            orchestrator.logger.warning(
                f"High failure risk ({risk:.1%}) for task {work_item.id}"
            )
        
        # Call original
        return await original_assign(worker_id, work_item)
    
    orchestrator._assign_work_to_worker = monitored_assign
    
    return orchestrator


def integrate_monitoring_into_worker(worker):
    """Add monitoring hooks to AI worker."""
    monitor = MonitoringIntegration()
    
    # Replace execute method with monitored version
    worker.execute_work_item = monitor.monitor_task_execution(worker.execute_work_item)
    
    # Add monitoring instance
    worker.monitoring = monitor
    
    return worker