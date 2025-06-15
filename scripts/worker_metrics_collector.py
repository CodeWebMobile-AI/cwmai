"""
Worker Metrics Collector

Real-time performance tracking and analytics for parallel workers.
Collects, analyzes, and provides actionable insights about worker performance,
resource utilization, and system optimization opportunities.
"""

import asyncio
import time
import threading
import psutil
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import numpy as np
import statistics
from scripts.worker_logging_config import setup_worker_logger, WorkerOperationContext


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"  # Cumulative values (task count, errors)
    GAUGE = "gauge"     # Point-in-time values (CPU, memory)
    HISTOGRAM = "histogram"  # Distribution of values (task duration)
    TIMER = "timer"     # Time-based measurements


@dataclass
class MetricPoint:
    """Individual metric measurement."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric measurements."""
    name: str
    metric_type: MetricType
    unit: str
    description: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))  # Keep last 1000 points
    
    def add_point(self, value: float, labels: Dict[str, str] = None, 
                  metadata: Dict[str, Any] = None):
        """Add a new measurement point."""
        point = MetricPoint(
            timestamp=datetime.now(timezone.utc),
            value=value,
            labels=labels or {},
            metadata=metadata or {}
        )
        self.points.append(point)
    
    def get_latest_value(self) -> Optional[float]:
        """Get the most recent value."""
        return self.points[-1].value if self.points else None
    
    def get_values_in_window(self, window_minutes: int) -> List[float]:
        """Get values within a time window."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        return [p.value for p in self.points if p.timestamp >= cutoff]
    
    def calculate_statistics(self, window_minutes: int = 5) -> Dict[str, float]:
        """Calculate statistics for recent values."""
        values = self.get_values_in_window(window_minutes)
        if not values:
            return {}
        
        stats = {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values)
        }
        
        if len(values) > 1:
            stats['std'] = statistics.stdev(values)
            
            # Calculate percentiles
            sorted_values = sorted(values)
            stats['p95'] = sorted_values[int(0.95 * len(sorted_values))]
            stats['p99'] = sorted_values[int(0.99 * len(sorted_values))]
        
        return stats


@dataclass
class ResourceUsage:
    """System resource usage snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: int
    network_bytes_recv: int
    open_files: int
    thread_count: int


@dataclass
class WorkerPerformanceSnapshot:
    """Performance snapshot for a specific worker."""
    worker_id: str
    timestamp: datetime
    tasks_completed: int
    tasks_failed: int
    average_task_duration: float
    current_load: float
    health_score: float
    resource_usage: ResourceUsage
    specialization: str
    active_time_seconds: float


class PerformanceAnomalyDetector:
    """Detects performance anomalies using statistical methods."""
    
    def __init__(self, sensitivity: float = 2.0):
        """Initialize anomaly detector.
        
        Args:
            sensitivity: Number of standard deviations for anomaly threshold
        """
        self.sensitivity = sensitivity
        self.baselines: Dict[str, Tuple[float, float]] = {}  # metric -> (mean, std)
        self.logger = setup_worker_logger("anomaly_detector")
    
    def update_baseline(self, metric_name: str, values: List[float]):
        """Update baseline statistics for a metric."""
        if len(values) < 3:
            return
        
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        self.baselines[metric_name] = (mean, std)
    
    def detect_anomaly(self, metric_name: str, current_value: float) -> Optional[Dict[str, Any]]:
        """Detect if current value is anomalous."""
        if metric_name not in self.baselines:
            return None
        
        mean, std = self.baselines[metric_name]
        if std == 0:
            return None
        
        z_score = abs(current_value - mean) / std
        
        if z_score > self.sensitivity:
            return {
                'metric': metric_name,
                'current_value': current_value,
                'baseline_mean': mean,
                'baseline_std': std,
                'z_score': z_score,
                'severity': 'high' if z_score > 3.0 else 'medium',
                'detected_at': datetime.now(timezone.utc).isoformat()
            }
        
        return None


class WorkerMetricsCollector:
    """Comprehensive metrics collection and analysis for parallel workers."""
    
    def __init__(self, collection_interval: float = 5.0):
        """Initialize metrics collector.
        
        Args:
            collection_interval: Seconds between metric collections
        """
        self.collection_interval = collection_interval
        self.logger = setup_worker_logger("metrics_collector")
        
        # Metric storage
        self.metrics: Dict[str, MetricSeries] = {}
        self.worker_snapshots: Dict[str, List[WorkerPerformanceSnapshot]] = defaultdict(list)
        self.system_snapshots: List[ResourceUsage] = []
        
        # Performance tracking
        self.task_timers: Dict[str, float] = {}  # task_id -> start_time
        self.worker_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Analysis components
        self.anomaly_detector = PerformanceAnomalyDetector()
        self.performance_alerts: List[Dict[str, Any]] = []
        
        # Collection control
        self._collection_task: Optional[asyncio.Task] = None
        self._shutdown = False
        self.lock = threading.RLock()
        
        # Initialize standard metrics
        self._initialize_standard_metrics()
        
        # Resource monitoring
        self._last_disk_io = psutil.disk_io_counters()
        self._last_network_io = psutil.net_io_counters()
        self._last_io_time = time.time()
    
    def _initialize_standard_metrics(self):
        """Initialize standard system and worker metrics."""
        standard_metrics = [
            # System metrics
            ("system.cpu_percent", MetricType.GAUGE, "percent", "System CPU utilization"),
            ("system.memory_percent", MetricType.GAUGE, "percent", "System memory utilization"),
            ("system.disk_io_rate", MetricType.GAUGE, "MB/s", "Disk I/O rate"),
            ("system.network_io_rate", MetricType.GAUGE, "bytes/s", "Network I/O rate"),
            
            # Worker metrics
            ("worker.task_completion_rate", MetricType.GAUGE, "tasks/min", "Worker task completion rate"),
            ("worker.task_duration", MetricType.HISTOGRAM, "seconds", "Task execution duration"),
            ("worker.error_rate", MetricType.GAUGE, "percent", "Worker error rate"),
            ("worker.health_score", MetricType.GAUGE, "score", "Worker health score"),
            ("worker.load", MetricType.GAUGE, "percent", "Worker current load"),
            
            # Performance metrics
            ("performance.throughput", MetricType.GAUGE, "tasks/hour", "System task throughput"),
            ("performance.success_rate", MetricType.GAUGE, "percent", "Overall success rate"),
            ("performance.average_response_time", MetricType.GAUGE, "seconds", "Average task response time"),
        ]
        
        for name, metric_type, unit, description in standard_metrics:
            self.metrics[name] = MetricSeries(name, metric_type, unit, description)
    
    async def start_collection(self):
        """Start background metrics collection."""
        self.logger.info("Starting metrics collection")
        self._collection_task = asyncio.create_task(self._collection_loop())
    
    async def stop_collection(self):
        """Stop metrics collection."""
        self.logger.info("Stopping metrics collection")
        self._shutdown = True
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
    
    async def _collection_loop(self):
        """Main collection loop."""
        while not self._shutdown:
            try:
                with WorkerOperationContext("metrics_collector", "collect_metrics"):
                    await self._collect_system_metrics()
                    await self._collect_worker_metrics()
                    await self._perform_analysis()
                    
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}", exc_info=True)
                await asyncio.sleep(self.collection_interval * 2)  # Back off on error
    
    async def _collect_system_metrics(self):
        """Collect system-wide resource metrics."""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            # Disk I/O
            current_disk_io = psutil.disk_io_counters()
            current_time = time.time()
            time_delta = current_time - self._last_io_time
            
            if time_delta > 0 and self._last_disk_io:
                disk_read_rate = (current_disk_io.read_bytes - self._last_disk_io.read_bytes) / time_delta / 1024 / 1024  # MB/s
                disk_write_rate = (current_disk_io.write_bytes - self._last_disk_io.write_bytes) / time_delta / 1024 / 1024
                disk_io_rate = disk_read_rate + disk_write_rate
            else:
                disk_io_rate = 0.0
            
            # Network I/O
            current_network_io = psutil.net_io_counters()
            if time_delta > 0 and self._last_network_io:
                network_rate = (
                    (current_network_io.bytes_sent - self._last_network_io.bytes_sent) +
                    (current_network_io.bytes_recv - self._last_network_io.bytes_recv)
                ) / time_delta
            else:
                network_rate = 0.0
            
            # Record metrics
            self.record_metric("system.cpu_percent", cpu_percent)
            self.record_metric("system.memory_percent", memory.percent)
            self.record_metric("system.disk_io_rate", disk_io_rate)
            self.record_metric("system.network_io_rate", network_rate)
            
            # Create system snapshot
            system_snapshot = ResourceUsage(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                disk_io_read_mb=disk_read_rate if time_delta > 0 else 0.0,
                disk_io_write_mb=disk_write_rate if time_delta > 0 else 0.0,
                network_bytes_sent=current_network_io.bytes_sent,
                network_bytes_recv=current_network_io.bytes_recv,
                open_files=len(psutil.Process().open_files()),
                thread_count=threading.active_count()
            )
            
            with self.lock:
                self.system_snapshots.append(system_snapshot)
                # Keep only last 1000 snapshots
                if len(self.system_snapshots) > 1000:
                    self.system_snapshots.pop(0)
            
            # Update for next iteration
            self._last_disk_io = current_disk_io
            self._last_network_io = current_network_io
            self._last_io_time = current_time
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_worker_metrics(self):
        """Collect worker-specific performance metrics."""
        with self.lock:
            for worker_id, state in self.worker_states.items():
                try:
                    # Calculate task completion rate
                    completed_tasks = state.get('tasks_completed', 0)
                    failed_tasks = state.get('tasks_failed', 0)
                    total_tasks = completed_tasks + failed_tasks
                    
                    if total_tasks > 0:
                        error_rate = (failed_tasks / total_tasks) * 100
                        self.record_metric("worker.error_rate", error_rate, 
                                         labels={"worker_id": worker_id})
                    
                    # Record other worker metrics
                    health_score = state.get('health_score', 1.0)
                    current_load = state.get('current_load', 0.0)
                    
                    self.record_metric("worker.health_score", health_score,
                                     labels={"worker_id": worker_id})
                    self.record_metric("worker.load", current_load * 100,
                                     labels={"worker_id": worker_id})
                    
                    # Create worker performance snapshot
                    if self.system_snapshots:
                        latest_system = self.system_snapshots[-1]
                        
                        snapshot = WorkerPerformanceSnapshot(
                            worker_id=worker_id,
                            timestamp=datetime.now(timezone.utc),
                            tasks_completed=completed_tasks,
                            tasks_failed=failed_tasks,
                            average_task_duration=state.get('average_task_duration', 0.0),
                            current_load=current_load,
                            health_score=health_score,
                            resource_usage=latest_system,
                            specialization=state.get('specialization', 'unknown'),
                            active_time_seconds=state.get('active_time_seconds', 0.0)
                        )
                        
                        self.worker_snapshots[worker_id].append(snapshot)
                        # Keep only last 200 snapshots per worker
                        if len(self.worker_snapshots[worker_id]) > 200:
                            self.worker_snapshots[worker_id].pop(0)
                
                except Exception as e:
                    self.logger.error(f"Error collecting metrics for worker {worker_id}: {e}")
    
    async def _perform_analysis(self):
        """Perform real-time analysis and anomaly detection."""
        try:
            # Update baselines for anomaly detection
            for metric_name, metric_series in self.metrics.items():
                recent_values = metric_series.get_values_in_window(10)  # Last 10 minutes
                if len(recent_values) >= 5:
                    self.anomaly_detector.update_baseline(metric_name, recent_values)
            
            # Check for anomalies in recent values
            for metric_name, metric_series in self.metrics.items():
                latest_value = metric_series.get_latest_value()
                if latest_value is not None:
                    anomaly = self.anomaly_detector.detect_anomaly(metric_name, latest_value)
                    if anomaly:
                        self._handle_anomaly(anomaly)
            
            # Calculate derived metrics
            self._calculate_derived_metrics()
            
        except Exception as e:
            self.logger.error(f"Error in metrics analysis: {e}")
    
    def _calculate_derived_metrics(self):
        """Calculate derived performance metrics."""
        try:
            # System throughput
            task_completions = []
            for worker_id in self.worker_states:
                completed = self.worker_states[worker_id].get('tasks_completed', 0)
                task_completions.append(completed)
            
            if task_completions:
                total_throughput = sum(task_completions)
                self.record_metric("performance.throughput", total_throughput)
            
            # Overall success rate
            total_completed = sum(
                state.get('tasks_completed', 0) 
                for state in self.worker_states.values()
            )
            total_failed = sum(
                state.get('tasks_failed', 0) 
                for state in self.worker_states.values()
            )
            
            if total_completed + total_failed > 0:
                success_rate = (total_completed / (total_completed + total_failed)) * 100
                self.record_metric("performance.success_rate", success_rate)
            
            # Average response time across all workers
            response_times = []
            for state in self.worker_states.values():
                avg_duration = state.get('average_task_duration', 0.0)
                if avg_duration > 0:
                    response_times.append(avg_duration)
            
            if response_times:
                avg_response_time = statistics.mean(response_times)
                self.record_metric("performance.average_response_time", avg_response_time)
        
        except Exception as e:
            self.logger.error(f"Error calculating derived metrics: {e}")
    
    def _handle_anomaly(self, anomaly: Dict[str, Any]):
        """Handle detected performance anomaly."""
        self.logger.warning(f"Performance anomaly detected: {anomaly}")
        
        # Add to alerts with automatic expiration
        alert = {
            **anomaly,
            'alert_id': f"anomaly_{len(self.performance_alerts)}_{int(time.time())}",
            'status': 'active',
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        with self.lock:
            self.performance_alerts.append(alert)
            # Keep only last 50 alerts
            if len(self.performance_alerts) > 50:
                self.performance_alerts.pop(0)
    
    def record_metric(self, name: str, value: float, 
                     labels: Dict[str, str] = None, 
                     metadata: Dict[str, Any] = None):
        """Record a metric value."""
        with self.lock:
            if name not in self.metrics:
                # Auto-create metric if it doesn't exist
                self.metrics[name] = MetricSeries(
                    name=name,
                    metric_type=MetricType.GAUGE,
                    unit="unknown",
                    description=f"Auto-created metric: {name}"
                )
            
            self.metrics[name].add_point(value, labels, metadata)
    
    def start_task_timer(self, task_id: str, worker_id: str):
        """Start timing a task execution."""
        with self.lock:
            self.task_timers[task_id] = time.time()
            
            # Initialize worker state if needed
            if worker_id not in self.worker_states:
                self.worker_states[worker_id] = {
                    'tasks_completed': 0,
                    'tasks_failed': 0,
                    'average_task_duration': 0.0,
                    'current_load': 0.0,
                    'health_score': 1.0,
                    'specialization': 'unknown',
                    'active_time_seconds': 0.0
                }
    
    def end_task_timer(self, task_id: str, worker_id: str, success: bool):
        """End timing a task execution and record metrics."""
        with self.lock:
            if task_id not in self.task_timers:
                self.logger.warning(f"Task timer not found for task {task_id}")
                return
            
            duration = time.time() - self.task_timers[task_id]
            del self.task_timers[task_id]
            
            # Record task duration
            self.record_metric("worker.task_duration", duration, 
                             labels={"worker_id": worker_id, "success": str(success)})
            
            # Update worker state
            if worker_id in self.worker_states:
                state = self.worker_states[worker_id]
                
                if success:
                    state['tasks_completed'] = state.get('tasks_completed', 0) + 1
                else:
                    state['tasks_failed'] = state.get('tasks_failed', 0) + 1
                
                # Update average duration (exponential moving average)
                alpha = 0.3
                if state.get('average_task_duration', 0) == 0:
                    state['average_task_duration'] = duration
                else:
                    state['average_task_duration'] = (
                        alpha * duration + (1 - alpha) * state['average_task_duration']
                    )
                
                state['active_time_seconds'] = state.get('active_time_seconds', 0) + duration
    
    def update_worker_state(self, worker_id: str, **kwargs):
        """Update worker state information."""
        with self.lock:
            if worker_id not in self.worker_states:
                self.worker_states[worker_id] = {}
            
            self.worker_states[worker_id].update(kwargs)
    
    def get_metric_statistics(self, metric_name: str, 
                            window_minutes: int = 5) -> Dict[str, Any]:
        """Get statistics for a specific metric."""
        if metric_name not in self.metrics:
            return {}
        
        metric = self.metrics[metric_name]
        stats = metric.calculate_statistics(window_minutes)
        
        return {
            'metric_name': metric_name,
            'window_minutes': window_minutes,
            'statistics': stats,
            'latest_value': metric.get_latest_value(),
            'total_points': len(metric.points)
        }
    
    def get_worker_performance_summary(self, worker_id: str) -> Dict[str, Any]:
        """Get comprehensive performance summary for a worker."""
        with self.lock:
            if worker_id not in self.worker_states:
                return {}
            
            state = self.worker_states[worker_id]
            snapshots = self.worker_snapshots.get(worker_id, [])
            
            # Calculate performance trends
            if len(snapshots) >= 5:
                recent_snapshots = snapshots[-5:]
                health_trend = [s.health_score for s in recent_snapshots]
                load_trend = [s.current_load for s in recent_snapshots]
                
                health_trend_direction = "stable"
                if len(health_trend) >= 3:
                    if health_trend[-1] > health_trend[0] + 0.1:
                        health_trend_direction = "improving"
                    elif health_trend[-1] < health_trend[0] - 0.1:
                        health_trend_direction = "declining"
            else:
                health_trend = []
                load_trend = []
                health_trend_direction = "insufficient_data"
            
            # Get worker-specific metrics
            worker_metrics = {}
            for metric_name, metric_series in self.metrics.items():
                if metric_name.startswith("worker."):
                    # Get points for this specific worker
                    worker_points = [
                        p.value for p in metric_series.points
                        if p.labels.get("worker_id") == worker_id
                    ]
                    if worker_points:
                        worker_metrics[metric_name] = {
                            'current': worker_points[-1] if worker_points else None,
                            'average': statistics.mean(worker_points),
                            'count': len(worker_points)
                        }
            
            return {
                'worker_id': worker_id,
                'current_state': state,
                'performance_trends': {
                    'health_trend_direction': health_trend_direction,
                    'health_values': health_trend,
                    'load_values': load_trend
                },
                'metrics': worker_metrics,
                'snapshot_count': len(snapshots),
                'last_updated': snapshots[-1].timestamp.isoformat() if snapshots else None
            }
    
    def get_system_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive system performance dashboard data."""
        with self.lock:
            # System overview
            system_overview = {
                'total_workers': len(self.worker_states),
                'active_workers': sum(1 for state in self.worker_states.values() 
                                    if state.get('current_load', 0) > 0),
                'total_tasks_completed': sum(state.get('tasks_completed', 0) 
                                           for state in self.worker_states.values()),
                'total_tasks_failed': sum(state.get('tasks_failed', 0) 
                                        for state in self.worker_states.values()),
                'active_alerts': len([a for a in self.performance_alerts 
                                    if a.get('status') == 'active'])
            }
            
            # Current system metrics
            current_metrics = {}
            for metric_name in ['system.cpu_percent', 'system.memory_percent', 
                              'performance.throughput', 'performance.success_rate']:
                if metric_name in self.metrics:
                    latest = self.metrics[metric_name].get_latest_value()
                    stats = self.metrics[metric_name].calculate_statistics(5)
                    current_metrics[metric_name] = {
                        'current': latest,
                        'stats': stats
                    }
            
            # Worker summaries
            worker_summaries = {}
            for worker_id in self.worker_states:
                worker_summaries[worker_id] = self.get_worker_performance_summary(worker_id)
            
            # Recent alerts
            recent_alerts = sorted(self.performance_alerts, 
                                 key=lambda x: x['created_at'], reverse=True)[:10]
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'system_overview': system_overview,
                'current_metrics': current_metrics,
                'worker_summaries': worker_summaries,
                'recent_alerts': recent_alerts,
                'collection_interval': self.collection_interval,
                'uptime_seconds': time.time() - (self._last_io_time if hasattr(self, '_last_io_time') else time.time())
            }
    
    def export_metrics(self, format_type: str = "json", 
                      time_range_minutes: int = 60) -> str:
        """Export metrics in specified format."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=time_range_minutes)
        
        export_data = {
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'time_range_minutes': time_range_minutes,
            'metrics': {}
        }
        
        with self.lock:
            for metric_name, metric_series in self.metrics.items():
                filtered_points = [
                    {
                        'timestamp': p.timestamp.isoformat(),
                        'value': p.value,
                        'labels': p.labels,
                        'metadata': p.metadata
                    }
                    for p in metric_series.points 
                    if p.timestamp >= cutoff_time
                ]
                
                if filtered_points:
                    export_data['metrics'][metric_name] = {
                        'type': metric_series.metric_type.value,
                        'unit': metric_series.unit,
                        'description': metric_series.description,
                        'points': filtered_points
                    }
        
        if format_type.lower() == "json":
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")


# Context manager for automatic task timing
class TaskTimer:
    """Context manager for automatic task timing."""
    
    def __init__(self, metrics_collector: WorkerMetricsCollector, 
                 task_id: str, worker_id: str):
        """Initialize task timer.
        
        Args:
            metrics_collector: Metrics collector instance
            task_id: Unique task identifier
            worker_id: Worker performing the task
        """
        self.metrics_collector = metrics_collector
        self.task_id = task_id
        self.worker_id = worker_id
        self.success = False
    
    def __enter__(self):
        """Start timing the task."""
        self.metrics_collector.start_task_timer(self.task_id, self.worker_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and record results."""
        self.success = exc_type is None
        self.metrics_collector.end_task_timer(self.task_id, self.worker_id, self.success)
    
    def mark_success(self):
        """Manually mark task as successful."""
        self.success = True


# Example usage and demonstration
async def demonstrate_metrics_collector():
    """Demonstrate the metrics collector capabilities."""
    collector = WorkerMetricsCollector(collection_interval=2.0)
    
    try:
        await collector.start_collection()
        
        # Simulate worker activities
        worker_ids = ["swarm_1", "repo_1", "general_1"]
        
        # Register workers
        for worker_id in worker_ids:
            collector.update_worker_state(
                worker_id,
                specialization=f"test_{worker_id}",
                health_score=0.9,
                current_load=0.3
            )
        
        # Simulate task executions
        for i in range(10):
            worker_id = worker_ids[i % len(worker_ids)]
            task_id = f"task_{i}"
            
            # Use task timer context manager
            with TaskTimer(collector, task_id, worker_id):
                # Simulate task work
                await asyncio.sleep(0.1 + (i % 3) * 0.05)  # Variable duration
                
                # Simulate occasional failures
                if i == 7:
                    raise Exception("Simulated task failure")
        
        # Wait for some metrics collection
        await asyncio.sleep(5)
        
        # Get dashboard data
        dashboard = collector.get_system_performance_dashboard()
        print("System Performance Dashboard:")
        print(json.dumps(dashboard, indent=2, default=str))
        
        # Get specific metric statistics
        task_duration_stats = collector.get_metric_statistics("worker.task_duration", 5)
        print(f"\nTask Duration Statistics:")
        print(json.dumps(task_duration_stats, indent=2, default=str))
        
        # Export metrics
        metrics_export = collector.export_metrics("json", 10)
        print(f"\nExported {len(metrics_export)} characters of metrics data")
        
    finally:
        await collector.stop_collection()


if __name__ == "__main__":
    asyncio.run(demonstrate_metrics_collector())