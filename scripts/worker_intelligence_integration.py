"""
Worker Intelligence Integration

Integration layer to enhance existing worker files with the new intelligence
and logging system. Provides seamless integration with minimal code changes
to existing worker implementations.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from contextlib import asynccontextmanager
import functools
import inspect

# Import the new intelligence components
from scripts.worker_logging_config import (
    setup_worker_logger, 
    WorkerOperationContext, 
    LogLevel
)
from scripts.worker_intelligence_hub import (
    WorkerIntelligenceHub, 
    WorkerSpecialization
)
from scripts.worker_metrics_collector import (
    WorkerMetricsCollector, 
    TaskTimer
)
from scripts.error_analyzer import (
    ErrorAnalyzer, 
    ErrorCaptureContext
)
from scripts.work_item_tracker import (
    WorkItemTracker, 
    WorkItemContext, 
    WorkItemPriority
)
from scripts.worker_status_reporter import (
    WorkerStatusReporter
)


@dataclass
class WorkerEnhancementConfig:
    """Configuration for worker enhancements."""
    enable_logging: bool = True
    enable_intelligence: bool = True
    enable_metrics: bool = True
    enable_error_analysis: bool = True
    enable_work_tracking: bool = True
    enable_status_reporting: bool = True
    log_level: LogLevel = LogLevel.INFO
    worker_specialization: WorkerSpecialization = WorkerSpecialization.GENERAL


class IntelligentWorkerMixin:
    """Mixin class to add intelligence capabilities to existing workers."""
    
    def __init__(self, worker_id: str, config: WorkerEnhancementConfig = None):
        """Initialize intelligent worker capabilities.
        
        Args:
            worker_id: Unique identifier for this worker
            config: Enhancement configuration
        """
        self.worker_id = worker_id
        self.config = config or WorkerEnhancementConfig()
        
        # Initialize components
        self._initialize_intelligence_components()
        
        # Track worker state
        self.is_initialized = False
        self.start_time = datetime.now(timezone.utc)
        self.task_count = 0
        self.error_count = 0
    
    def _initialize_intelligence_components(self):
        """Initialize intelligence components."""
        # Logging
        if self.config.enable_logging:
            self.logger = setup_worker_logger(
                self.worker_id.replace('_', '-'),  # Clean worker name
                self.config.log_level
            )
        else:
            import logging
            self.logger = logging.getLogger(self.worker_id)
        
        # Intelligence hub (will be set by coordinator)
        self.intelligence_hub: Optional[WorkerIntelligenceHub] = None
        
        # Metrics collector (will be set by coordinator)
        self.metrics_collector: Optional[WorkerMetricsCollector] = None
        
        # Error analyzer (will be set by coordinator)
        self.error_analyzer: Optional[ErrorAnalyzer] = None
        
        # Work item tracker (will be set by coordinator)
        self.work_item_tracker: Optional[WorkItemTracker] = None
        
        # Status reporter (will be set by coordinator)
        self.status_reporter: Optional[WorkerStatusReporter] = None
    
    def set_intelligence_components(self, 
                                  intelligence_hub: WorkerIntelligenceHub = None,
                                  metrics_collector: WorkerMetricsCollector = None,
                                  error_analyzer: ErrorAnalyzer = None,
                                  work_item_tracker: WorkItemTracker = None,
                                  status_reporter: WorkerStatusReporter = None):
        """Set intelligence component references."""
        if intelligence_hub and self.config.enable_intelligence:
            self.intelligence_hub = intelligence_hub
            # Register this worker
            self.intelligence_hub.register_worker(
                self.worker_id, 
                self.config.worker_specialization
            )
        
        if metrics_collector and self.config.enable_metrics:
            self.metrics_collector = metrics_collector
            # Initialize worker state
            self.metrics_collector.update_worker_state(
                self.worker_id,
                specialization=self.config.worker_specialization.value,
                health_score=1.0,
                current_load=0.0
            )
        
        if error_analyzer and self.config.enable_error_analysis:
            self.error_analyzer = error_analyzer
        
        if work_item_tracker and self.config.enable_work_tracking:
            self.work_item_tracker = work_item_tracker
        
        if status_reporter and self.config.enable_status_reporting:
            self.status_reporter = status_reporter
        
        self.is_initialized = True
        self.logger.info(f"Worker {self.worker_id} intelligence components initialized")
    
    @asynccontextmanager
    async def intelligent_task_execution(self, task_id: str, task_type: str, 
                                       task_metadata: Dict[str, Any] = None):
        """Context manager for intelligent task execution.
        
        Args:
            task_id: Unique task identifier
            task_type: Type of task being executed
            task_metadata: Additional task metadata
        """
        task_metadata = task_metadata or {}
        
        # Start timing and tracking
        start_time = time.time()
        
        # Create work item if tracker is available
        work_item_id = None
        if self.work_item_tracker:
            work_item_id = self.work_item_tracker.create_work_item(
                title=f"Task {task_id}",
                description=f"Executing {task_type} task",
                work_type=task_type,
                priority=WorkItemPriority.MEDIUM,
                created_by=self.worker_id,
                context=task_metadata
            )
            self.work_item_tracker.assign_to_worker(work_item_id, self.worker_id)
        
        # Start metrics collection
        if self.metrics_collector:
            self.metrics_collector.start_task_timer(task_id, self.worker_id)
        
        # Set up operation context
        with WorkerOperationContext(self.worker_id, task_type, task_metadata):
            # Set up error capture
            with ErrorCaptureContext(
                self.error_analyzer if self.error_analyzer else None, 
                self.worker_id, 
                task_type, 
                task_metadata
            ) if self.error_analyzer else nullcontext():
                
                # Set up work item tracking
                if work_item_id and self.work_item_tracker:
                    with WorkItemContext(self.work_item_tracker, work_item_id, self.worker_id):
                        try:
                            self.task_count += 1
                            self._update_worker_load(0.5)  # Moderate load during execution
                            
                            self.logger.info(f"Starting task {task_id} ({task_type})")
                            yield {
                                'task_id': task_id,
                                'work_item_id': work_item_id,
                                'start_time': start_time
                            }
                            
                            # Task completed successfully
                            duration = time.time() - start_time
                            self.logger.info(f"Completed task {task_id} in {duration:.2f}s")
                            
                        except Exception as e:
                            self.error_count += 1
                            self.logger.error(f"Task {task_id} failed: {e}")
                            raise
                        finally:
                            self._update_worker_load(0.0)  # Reset load
                            
                            # End metrics collection
                            if self.metrics_collector:
                                self.metrics_collector.end_task_timer(
                                    task_id, 
                                    self.worker_id, 
                                    True  # Success if no exception
                                )
                else:
                    # Simplified execution without work item tracking
                    try:
                        self.task_count += 1
                        self._update_worker_load(0.5)
                        
                        self.logger.info(f"Starting task {task_id} ({task_type})")
                        yield {
                            'task_id': task_id,
                            'start_time': start_time
                        }
                        
                        duration = time.time() - start_time
                        self.logger.info(f"Completed task {task_id} in {duration:.2f}s")
                        
                    except Exception as e:
                        self.error_count += 1
                        self.logger.error(f"Task {task_id} failed: {e}")
                        raise
                    finally:
                        self._update_worker_load(0.0)
                        
                        if self.metrics_collector:
                            self.metrics_collector.end_task_timer(task_id, self.worker_id, True)
    
    def _update_worker_load(self, load: float):
        """Update worker load in metrics collector."""
        if self.metrics_collector:
            self.metrics_collector.update_worker_state(
                self.worker_id,
                current_load=load
            )
    
    def get_optimal_task(self, available_tasks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get optimal task for this worker using intelligence hub."""
        if not self.intelligence_hub or not available_tasks:
            return available_tasks[0] if available_tasks else None
        
        # Score each task for this worker
        best_task = None
        best_score = -1
        
        for task in available_tasks:
            task_type = task.get('type', 'unknown')
            task_metadata = task.get('metadata', {})
            
            # Get recommendation from intelligence hub
            recommendations = self.intelligence_hub.get_worker_recommendations(
                task_type, task_metadata
            )
            
            if recommendations['recommended_worker'] == self.worker_id:
                # Find this worker's score
                for worker_id, score in recommendations['all_worker_scores']:
                    if worker_id == self.worker_id:
                        if score > best_score:
                            best_score = score
                            best_task = task
                        break
        
        return best_task or available_tasks[0]
    
    def report_performance_metrics(self) -> Dict[str, Any]:
        """Report current performance metrics."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        metrics = {
            'worker_id': self.worker_id,
            'uptime_seconds': uptime,
            'tasks_completed': self.task_count,
            'errors_encountered': self.error_count,
            'error_rate': self.error_count / max(self.task_count, 1),
            'tasks_per_hour': self.task_count / max(uptime / 3600, 0.001),
            'is_initialized': self.is_initialized
        }
        
        # Add intelligence hub metrics if available
        if self.intelligence_hub and self.worker_id in self.intelligence_hub.worker_profiles:
            profile = self.intelligence_hub.worker_profiles[self.worker_id]
            metrics.update({
                'health_score': profile.health_score,
                'performance_score': profile.calculate_current_performance(),
                'specialization': profile.specialization.value,
                'preferred_tasks': list(profile.preferred_task_types),
                'avoided_tasks': list(profile.avoided_task_types)
            })
        
        return metrics


# Null context manager for when components are not available
class nullcontext:
    """Null context manager that does nothing."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def enhance_worker_method(task_type: str = None):
    """Decorator to enhance worker methods with intelligence capabilities.
    
    Args:
        task_type: Type of task this method handles
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Generate task ID
            task_id = f"{self.worker_id}_{func.__name__}_{int(time.time())}"
            
            # Determine task type
            determined_task_type = task_type or func.__name__
            
            # Extract task metadata from function arguments
            task_metadata = {}
            if hasattr(self, '_extract_task_metadata'):
                task_metadata = self._extract_task_metadata(func, args, kwargs)
            
            # Use intelligent task execution if mixin is available
            if hasattr(self, 'intelligent_task_execution'):
                async with self.intelligent_task_execution(
                    task_id, determined_task_type, task_metadata
                ) as context:
                    return await func(self, *args, **kwargs)
            else:
                # Fallback to regular execution
                return await func(self, *args, **kwargs)
        
        return wrapper
    return decorator


def enhance_sync_worker_method(task_type: str = None):
    """Decorator to enhance synchronous worker methods with intelligence capabilities.
    
    Args:
        task_type: Type of task this method handles
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Generate task ID
            task_id = f"{self.worker_id}_{func.__name__}_{int(time.time())}"
            
            # Determine task type
            determined_task_type = task_type or func.__name__
            
            # Extract task metadata
            task_metadata = {}
            if hasattr(self, '_extract_task_metadata'):
                task_metadata = self._extract_task_metadata(func, args, kwargs)
            
            # Use error capture if available
            if hasattr(self, 'error_analyzer') and self.error_analyzer:
                with ErrorCaptureContext(
                    self.error_analyzer, 
                    self.worker_id, 
                    determined_task_type, 
                    task_metadata
                ):
                    return self._execute_with_intelligence(
                        func, task_id, determined_task_type, task_metadata, args, kwargs
                    )
            else:
                return self._execute_with_intelligence(
                    func, task_id, determined_task_type, task_metadata, args, kwargs
                )
        
        return wrapper
    return decorator


class WorkerIntelligenceCoordinator:
    """Coordinates intelligence components for multiple workers."""
    
    def __init__(self):
        """Initialize the coordinator."""
        self.logger = setup_worker_logger("intelligence_coordinator")
        
        # Initialize all intelligence components
        self.intelligence_hub = WorkerIntelligenceHub()
        self.metrics_collector = WorkerMetricsCollector()
        self.error_analyzer = ErrorAnalyzer()
        self.work_item_tracker = WorkItemTracker()
        self.status_reporter = WorkerStatusReporter()
        
        # Connect components
        self.status_reporter.set_external_components(
            intelligence_hub=self.intelligence_hub,
            metrics_collector=self.metrics_collector,
            error_analyzer=self.error_analyzer,
            work_item_tracker=self.work_item_tracker
        )
        
        # Track enhanced workers
        self.enhanced_workers: Dict[str, Any] = {}
        
        self.logger.info("Worker intelligence coordinator initialized")
    
    async def start(self):
        """Start all intelligence components."""
        self.logger.info("Starting intelligence components")
        
        await self.intelligence_hub.start()
        await self.metrics_collector.start_collection()
        await self.error_analyzer.start()
        await self.work_item_tracker.start_monitoring()
        await self.status_reporter.start_monitoring()
        
        self.logger.info("All intelligence components started")
    
    async def stop(self):
        """Stop all intelligence components."""
        self.logger.info("Stopping intelligence components")
        
        await self.status_reporter.stop_monitoring()
        await self.work_item_tracker.stop_monitoring()
        await self.error_analyzer.stop()
        await self.metrics_collector.stop_collection()
        await self.intelligence_hub.stop()
        
        self.logger.info("All intelligence components stopped")
    
    def enhance_worker(self, worker_instance, worker_id: str, 
                      config: WorkerEnhancementConfig = None) -> Any:
        """Enhance an existing worker instance with intelligence capabilities.
        
        Args:
            worker_instance: Existing worker instance
            worker_id: Unique identifier for the worker
            config: Enhancement configuration
            
        Returns:
            Enhanced worker instance
        """
        config = config or WorkerEnhancementConfig()
        
        # Create enhanced worker class dynamically
        class EnhancedWorker(IntelligentWorkerMixin, worker_instance.__class__):
            def __init__(self, original_instance):
                # Copy all attributes from original instance
                for attr_name in dir(original_instance):
                    if not attr_name.startswith('_'):
                        try:
                            setattr(self, attr_name, getattr(original_instance, attr_name))
                        except AttributeError:
                            pass
                
                # Initialize intelligence capabilities
                IntelligentWorkerMixin.__init__(self, worker_id, config)
            
            def _extract_task_metadata(self, func, args, kwargs):
                """Extract task metadata from function call."""
                metadata = {}
                
                # Get function signature
                sig = inspect.signature(func)
                bound_args = sig.bind(self, *args, **kwargs)
                bound_args.apply_defaults()
                
                # Extract relevant parameters
                for param_name, param_value in bound_args.arguments.items():
                    if param_name == 'self':
                        continue
                    
                    # Add to metadata if it's a simple type
                    if isinstance(param_value, (str, int, float, bool, list, dict)):
                        metadata[param_name] = param_value
                
                return metadata
            
            def _execute_with_intelligence(self, func, task_id, task_type, 
                                         task_metadata, args, kwargs):
                """Execute function with intelligence tracking."""
                start_time = time.time()
                
                # Start timing
                if self.metrics_collector:
                    self.metrics_collector.start_task_timer(task_id, self.worker_id)
                
                try:
                    self.task_count += 1
                    self._update_worker_load(0.5)
                    
                    if hasattr(self, 'logger'):
                        self.logger.info(f"Starting task {task_id} ({task_type})")
                    
                    result = func(self, *args, **kwargs)
                    
                    duration = time.time() - start_time
                    if hasattr(self, 'logger'):
                        self.logger.info(f"Completed task {task_id} in {duration:.2f}s")
                    
                    return result
                    
                except Exception as e:
                    self.error_count += 1
                    if hasattr(self, 'logger'):
                        self.logger.error(f"Task {task_id} failed: {e}")
                    raise
                finally:
                    self._update_worker_load(0.0)
                    
                    if self.metrics_collector:
                        self.metrics_collector.end_task_timer(task_id, self.worker_id, True)
        
        # Create enhanced instance
        enhanced_worker = EnhancedWorker(worker_instance)
        
        # Set intelligence components
        enhanced_worker.set_intelligence_components(
            self.intelligence_hub,
            self.metrics_collector,
            self.error_analyzer,
            self.work_item_tracker,
            self.status_reporter
        )
        
        # Track enhanced worker
        self.enhanced_workers[worker_id] = enhanced_worker
        
        self.logger.info(f"Enhanced worker {worker_id}")
        return enhanced_worker
    
    def get_system_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive system dashboard data."""
        return self.status_reporter.get_dashboard_data()
    
    def get_worker_performance(self, worker_id: str) -> Dict[str, Any]:
        """Get performance data for specific worker."""
        if worker_id in self.enhanced_workers:
            worker = self.enhanced_workers[worker_id]
            return worker.report_performance_metrics()
        return {}
    
    def create_alert(self, alert_type, severity, title, description, 
                    affected_workers=None):
        """Create system alert."""
        return self.status_reporter.create_custom_alert(
            alert_type, severity, title, description, affected_workers
        )


# Convenience function for quick worker enhancement
def create_intelligent_worker(worker_class, worker_id: str, 
                            config: WorkerEnhancementConfig = None, 
                            coordinator: WorkerIntelligenceCoordinator = None,
                            *args, **kwargs):
    """Create an intelligent worker instance.
    
    Args:
        worker_class: Worker class to instantiate
        worker_id: Unique worker identifier
        config: Enhancement configuration
        coordinator: Intelligence coordinator (optional)
        *args, **kwargs: Arguments for worker class constructor
        
    Returns:
        Enhanced worker instance
    """
    # Create original worker instance
    original_worker = worker_class(*args, **kwargs)
    
    # Create or get coordinator
    if coordinator is None:
        coordinator = WorkerIntelligenceCoordinator()
    
    # Enhance the worker
    enhanced_worker = coordinator.enhance_worker(original_worker, worker_id, config)
    
    return enhanced_worker, coordinator


# Example usage for existing workers
async def demonstrate_integration():
    """Demonstrate how to integrate with existing workers."""
    
    # Example existing worker class
    class ExampleWorker:
        def __init__(self, name):
            self.name = name
        
        def process_data(self, data):
            """Process some data."""
            return f"Processed {data} by {self.name}"
        
        async def analyze_task(self, task):
            """Analyze a task."""
            await asyncio.sleep(0.1)  # Simulate work
            return f"Analyzed {task} by {self.name}"
    
    # Create intelligence coordinator
    coordinator = WorkerIntelligenceCoordinator()
    await coordinator.start()
    
    try:
        # Enhance existing worker
        original_worker = ExampleWorker("test_worker")
        enhanced_worker = coordinator.enhance_worker(
            original_worker, 
            "example_worker_1",
            WorkerEnhancementConfig(
                worker_specialization=WorkerSpecialization.GENERAL,
                log_level=LogLevel.DEBUG
            )
        )
        
        # Use enhanced worker
        async with enhanced_worker.intelligent_task_execution(
            "task_1", "data_processing", {"data_size": 100}
        ):
            result = enhanced_worker.process_data("test_data")
            print(f"Result: {result}")
        
        async with enhanced_worker.intelligent_task_execution(
            "task_2", "task_analysis", {"complexity": "medium"}
        ):
            result = await enhanced_worker.analyze_task("complex_task")
            print(f"Result: {result}")
        
        # Get performance metrics
        metrics = enhanced_worker.report_performance_metrics()
        print(f"Worker metrics: {metrics}")
        
        # Get system dashboard
        dashboard = coordinator.get_system_dashboard()
        print(f"System health: {dashboard['system_health']['status']}")
        
    finally:
        await coordinator.stop()


if __name__ == "__main__":
    asyncio.run(demonstrate_integration())