"""
Worker Logging Configuration

Unified logging system for parallel workers with correlation IDs, structured JSON format,
and worker-specific context for complete visibility into worker operations.
"""

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum


class WorkerEventType(Enum):
    """Types of worker events for structured logging."""
    WORKER_START = "worker_start"
    WORKER_STOP = "worker_stop"
    TASK_CLAIM = "task_claim"
    TASK_START = "task_start"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETE = "task_complete"
    TASK_FAIL = "task_fail"
    COORDINATION = "coordination"
    INTELLIGENCE = "intelligence"
    PERFORMANCE = "performance"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class WorkerContext:
    """Worker-specific context for logging."""
    worker_id: str
    worker_type: str  # continuous, swarm, production
    specialization: Optional[str] = None
    current_task_id: Optional[str] = None
    correlation_id: Optional[str] = None
    start_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class WorkerMetrics:
    """Performance metrics for worker logging."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_duration: float = 0.0
    avg_task_duration: float = 0.0
    success_rate: float = 0.0
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


class CorrelationIDManager:
    """Manages correlation IDs for tracking work across workers."""
    
    _local = threading.local()
    
    @classmethod
    def generate_id(cls) -> str:
        """Generate a new correlation ID."""
        return str(uuid.uuid4())
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str):
        """Set correlation ID for current thread."""
        cls._local.correlation_id = correlation_id
    
    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        """Get correlation ID for current thread."""
        return getattr(cls._local, 'correlation_id', None)
    
    @classmethod
    @contextmanager
    def with_correlation_id(cls, correlation_id: Optional[str] = None):
        """Context manager for setting correlation ID."""
        if correlation_id is None:
            correlation_id = cls.generate_id()
        
        old_id = cls.get_correlation_id()
        cls.set_correlation_id(correlation_id)
        try:
            yield correlation_id
        finally:
            if old_id:
                cls.set_correlation_id(old_id)
            else:
                cls._local.correlation_id = None


class WorkerJSONFormatter(logging.Formatter):
    """JSON formatter for structured worker logging."""
    
    def __init__(self, include_worker_context: bool = True):
        super().__init__()
        self.include_worker_context = include_worker_context
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add correlation ID if available
        correlation_id = CorrelationIDManager.get_correlation_id()
        if correlation_id:
            log_entry['correlation_id'] = correlation_id
        
        # Add worker context if available and enabled
        if self.include_worker_context and hasattr(record, 'worker_context'):
            log_entry['worker'] = record.worker_context.to_dict()
        
        # Add metrics if available
        if hasattr(record, 'metrics'):
            log_entry['metrics'] = record.metrics.to_dict()
        
        # Add event type if available
        if hasattr(record, 'event_type'):
            log_entry['event_type'] = record.event_type.value
        
        # Add task information if available
        if hasattr(record, 'task_info'):
            log_entry['task'] = record.task_info
        
        # Add performance data if available
        if hasattr(record, 'performance'):
            log_entry['performance'] = record.performance
        
        # Add exception information
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add any extra fields
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'pathname', 'filename', 'module',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'levelno',
                          'levelname', 'message', 'exc_info', 'exc_text', 'stack_info',
                          'worker_context', 'metrics', 'event_type', 'task_info', 'performance']:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry['extra'] = extra_fields
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class WorkerLogger:
    """Enhanced logger for worker operations."""
    
    def __init__(self, worker_context: WorkerContext, logger_name: Optional[str] = None):
        self.worker_context = worker_context
        self.logger_name = logger_name or f"worker.{worker_context.worker_type}.{worker_context.worker_id}"
        self.logger = logging.getLogger(self.logger_name)
        self.metrics = WorkerMetrics()
        self._task_start_times: Dict[str, float] = {}
    
    def _log_with_context(self, level: int, event_type: WorkerEventType, message: str,
                         task_info: Optional[Dict[str, Any]] = None,
                         performance: Optional[Dict[str, Any]] = None,
                         **kwargs):
        """Log with worker context and structured data."""
        extra = {
            'worker_context': self.worker_context,
            'event_type': event_type,
            'metrics': self.metrics
        }
        
        if task_info:
            extra['task_info'] = task_info
        
        if performance:
            extra['performance'] = performance
        
        extra.update(kwargs)
        
        self.logger.log(level, message, extra=extra)
    
    def worker_start(self, specialization: Optional[str] = None, **kwargs):
        """Log worker startup."""
        self.worker_context.specialization = specialization
        self.worker_context.start_time = time.time()
        
        self._log_with_context(
            logging.INFO, 
            WorkerEventType.WORKER_START,
            f"Worker {self.worker_context.worker_id} starting",
            **kwargs
        )
    
    def worker_stop(self, reason: str = "normal", **kwargs):
        """Log worker shutdown."""
        uptime = None
        if self.worker_context.start_time:
            uptime = time.time() - self.worker_context.start_time
        
        performance = {'uptime_seconds': uptime} if uptime else None
        
        self._log_with_context(
            logging.INFO,
            WorkerEventType.WORKER_STOP,
            f"Worker {self.worker_context.worker_id} stopping: {reason}",
            performance=performance,
            **kwargs
        )
    
    def task_claim(self, task_id: str, task_type: str, **kwargs):
        """Log task claiming."""
        self.worker_context.current_task_id = task_id
        
        task_info = {
            'task_id': task_id,
            'task_type': task_type
        }
        
        self._log_with_context(
            logging.INFO,
            WorkerEventType.TASK_CLAIM,
            f"Claimed task {task_id}",
            task_info=task_info,
            **kwargs
        )
    
    def task_start(self, task_id: Optional[str] = None, **kwargs):
        """Log task start."""
        task_id = task_id or self.worker_context.current_task_id
        if task_id:
            self._task_start_times[task_id] = time.time()
        
        task_info = {'task_id': task_id} if task_id else None
        
        self._log_with_context(
            logging.INFO,
            WorkerEventType.TASK_START,
            f"Starting task {task_id}",
            task_info=task_info,
            **kwargs
        )
    
    def task_progress(self, progress: str, percentage: Optional[float] = None, 
                     task_id: Optional[str] = None, **kwargs):
        """Log task progress."""
        task_id = task_id or self.worker_context.current_task_id
        
        task_info = {
            'task_id': task_id,
            'progress': progress
        }
        if percentage is not None:
            task_info['percentage'] = percentage
        
        self._log_with_context(
            logging.INFO,
            WorkerEventType.TASK_PROGRESS,
            f"Task {task_id} progress: {progress}",
            task_info=task_info,
            **kwargs
        )
    
    def task_complete(self, task_id: Optional[str] = None, result_summary: Optional[str] = None, **kwargs):
        """Log task completion."""
        task_id = task_id or self.worker_context.current_task_id
        
        # Calculate duration
        duration = None
        if task_id and task_id in self._task_start_times:
            duration = time.time() - self._task_start_times.pop(task_id)
            
            # Update metrics
            self.metrics.tasks_completed += 1
            self.metrics.total_duration += duration
            self.metrics.avg_task_duration = self.metrics.total_duration / (self.metrics.tasks_completed + self.metrics.tasks_failed)
            self.metrics.success_rate = self.metrics.tasks_completed / (self.metrics.tasks_completed + self.metrics.tasks_failed)
        
        task_info = {'task_id': task_id}
        if result_summary:
            task_info['result_summary'] = result_summary
        
        performance = {'duration_seconds': duration} if duration else None
        
        self._log_with_context(
            logging.INFO,
            WorkerEventType.TASK_COMPLETE,
            f"Completed task {task_id}" + (f" in {duration:.2f}s" if duration else ""),
            task_info=task_info,
            performance=performance,
            **kwargs
        )
        
        # Clear current task
        if task_id == self.worker_context.current_task_id:
            self.worker_context.current_task_id = None
    
    def task_fail(self, error: str, task_id: Optional[str] = None, **kwargs):
        """Log task failure."""
        task_id = task_id or self.worker_context.current_task_id
        
        # Calculate duration
        duration = None
        if task_id and task_id in self._task_start_times:
            duration = time.time() - self._task_start_times.pop(task_id)
            
            # Update metrics
            self.metrics.tasks_failed += 1
            if self.metrics.tasks_completed + self.metrics.tasks_failed > 0:
                self.metrics.success_rate = self.metrics.tasks_completed / (self.metrics.tasks_completed + self.metrics.tasks_failed)
        
        task_info = {
            'task_id': task_id,
            'error': error
        }
        
        performance = {'duration_seconds': duration} if duration else None
        
        self._log_with_context(
            logging.ERROR,
            WorkerEventType.TASK_FAIL,
            f"Failed task {task_id}: {error}",
            task_info=task_info,
            performance=performance,
            **kwargs
        )
        
        # Clear current task
        if task_id == self.worker_context.current_task_id:
            self.worker_context.current_task_id = None
    
    def coordination_event(self, event: str, details: Optional[Dict[str, Any]] = None, **kwargs):
        """Log coordination events between workers."""
        self._log_with_context(
            logging.INFO,
            WorkerEventType.COORDINATION,
            f"Coordination: {event}",
            coordination_details=details,
            **kwargs
        )
    
    def intelligence_event(self, event: str, learning_data: Optional[Dict[str, Any]] = None, **kwargs):
        """Log intelligence and learning events."""
        self._log_with_context(
            logging.INFO,
            WorkerEventType.INTELLIGENCE,
            f"Intelligence: {event}",
            learning_data=learning_data,
            **kwargs
        )
    
    def performance_metric(self, metric_name: str, value: float, unit: Optional[str] = None, **kwargs):
        """Log performance metrics."""
        performance = {
            'metric': metric_name,
            'value': value,
            'unit': unit
        }
        
        self._log_with_context(
            logging.INFO,
            WorkerEventType.PERFORMANCE,
            f"Performance metric {metric_name}: {value}" + (f" {unit}" if unit else ""),
            performance=performance,
            **kwargs
        )
    
    def update_metrics(self, **metric_updates):
        """Update worker metrics."""
        for key, value in metric_updates.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
    
    def get_logger(self) -> logging.Logger:
        """Get the underlying logger for custom logging."""
        return self.logger


def setup_worker_logging(log_file: str = "logs/worker_logs.json", 
                        level: int = logging.INFO,
                        include_console: bool = True) -> None:
    """Setup centralized worker logging configuration."""
    
    # Create logs directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create JSON formatter
    json_formatter = WorkerJSONFormatter()
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(json_formatter)
    file_handler.setLevel(level)
    
    # Setup console handler if requested
    handlers = [file_handler]
    if include_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(json_formatter)
        console_handler.setLevel(level)
        handlers.append(console_handler)
    
    # Configure root logger for all worker loggers
    root_logger = logging.getLogger('worker')
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Prevent propagation to avoid duplicate logs
    root_logger.propagate = False


def create_worker_logger(worker_id: str, worker_type: str, 
                        specialization: Optional[str] = None) -> WorkerLogger:
    """Create a new worker logger with context."""
    context = WorkerContext(
        worker_id=worker_id,
        worker_type=worker_type,
        specialization=specialization
    )
    
    return WorkerLogger(context)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    setup_worker_logging()
    
    # Create a worker logger
    worker_logger = create_worker_logger("worker_001", "continuous", "github_issues")
    
    # Simulate worker lifecycle
    with CorrelationIDManager.with_correlation_id() as correlation_id:
        worker_logger.worker_start(specialization="github_issues")
        
        # Simulate task work
        worker_logger.task_claim("task_123", "github_issue")
        worker_logger.task_start()
        worker_logger.task_progress("Analyzing issue", 25)
        worker_logger.task_progress("Creating solution", 75)
        worker_logger.task_complete(result_summary="Issue resolved successfully")
        
        # Simulate coordination
        worker_logger.coordination_event("Load balancing", {"target_worker": "worker_002"})
        
        # Log performance metric
        worker_logger.performance_metric("task_completion_rate", 0.95, "ratio")
        
        worker_logger.worker_stop("normal")
    
    print(f"Demo completed with correlation ID: {correlation_id}")