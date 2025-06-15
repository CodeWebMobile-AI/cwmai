"""
Worker Status Reporter

Real-time monitoring and reporting system for parallel workers.
Provides dashboards, alerts, and performance tracking.
"""

import asyncio
import json
import time
import psutil
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from worker_logging_config import create_worker_logger, setup_worker_logging
from worker_intelligence_hub import get_intelligence_hub, WorkerStatus, TaskStatus


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    load_average: Optional[List[float]] = None
    network_io: Optional[Dict[str, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkerMetrics:
    """Individual worker performance metrics."""
    worker_id: str
    status: str
    current_task: Optional[str]
    tasks_completed: int
    tasks_failed: int
    success_rate: float
    avg_duration: float
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    last_activity: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Alert:
    """System alert."""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    component: str
    timestamp: float
    resolved: bool = False
    resolved_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['severity'] = self.severity.value
        return data


class WorkerStatusReporter:
    """Real-time worker status monitoring and reporting."""
    
    def __init__(self, report_interval: float = 30.0, alert_thresholds: Optional[Dict[str, Any]] = None):
        self.report_interval = report_interval
        self.logger = create_worker_logger("status_reporter", "monitoring")
        
        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'cpu_threshold': 90.0,
            'memory_threshold': 85.0,
            'disk_threshold': 95.0,
            'worker_timeout': 300.0,  # 5 minutes
            'task_duration_threshold': 1800.0,  # 30 minutes
            'failure_rate_threshold': 0.3,  # 30%
            'queue_length_threshold': 100
        }
        
        # State tracking
        self.alerts: Dict[str, Alert] = {}
        self.system_metrics_history: List[SystemMetrics] = []
        self.worker_metrics_history: Dict[str, List[WorkerMetrics]] = {}
        self.last_report_time = 0
        
        # Dashboard data
        self.dashboard_data: Dict[str, Any] = {}
        
        # Configuration
        self.max_history_length = 100
        self._running = False
        self._monitoring_task = None
        
        self.logger.worker_start(specialization="status_monitoring")
    
    async def start(self):
        """Start the status reporter."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.intelligence_event("Status reporter started", {
            "report_interval": self.report_interval,
            "thresholds": self.alert_thresholds
        })
    
    async def stop(self):
        """Stop the status reporter."""
        if not self._running:
            return
        
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.worker_stop("normal")
    
    def get_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Load average (Unix systems)
            load_average = None
            try:
                load_average = list(psutil.getloadavg())
            except AttributeError:
                pass  # Windows doesn't have load average
            
            # Network I/O
            network_io = None
            try:
                net_io = psutil.net_io_counters()
                network_io = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                }
            except:
                pass
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                load_average=load_average,
                network_io=network_io
            )
            
        except Exception as e:
            self.logger.intelligence_event("Error collecting system metrics", {"error": str(e)})
            return SystemMetrics(cpu_percent=0.0, memory_percent=0.0, disk_percent=0.0)
    
    async def collect_worker_metrics(self) -> List[WorkerMetrics]:
        """Collect metrics from all workers via intelligence hub."""
        try:
            hub = await get_intelligence_hub()
            worker_status = hub.get_worker_status()
            
            metrics = []
            for worker_id, worker_data in worker_status.get('workers', {}).items():
                worker_metrics = WorkerMetrics(
                    worker_id=worker_id,
                    status=worker_data.get('status', 'unknown'),
                    current_task=worker_data.get('current_task_id'),
                    tasks_completed=worker_data.get('successful_tasks', 0),
                    tasks_failed=worker_data.get('failed_tasks', 0),
                    success_rate=worker_data.get('success_rate', 0.0),
                    avg_duration=worker_data.get('avg_task_duration', 0.0),
                    cpu_usage=worker_data.get('cpu_usage'),
                    memory_usage=worker_data.get('memory_usage'),
                    last_activity=worker_data.get('updated_at')
                )
                metrics.append(worker_metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.intelligence_event("Error collecting worker metrics", {"error": str(e)})
            return []
    
    def check_system_alerts(self, metrics: SystemMetrics):
        """Check system metrics against thresholds and generate alerts."""
        current_time = time.time()
        
        # CPU alert
        if metrics.cpu_percent > self.alert_thresholds['cpu_threshold']:
            self._create_alert(
                "system_cpu_high",
                AlertSeverity.WARNING,
                "High CPU Usage",
                f"CPU usage is {metrics.cpu_percent:.1f}% (threshold: {self.alert_thresholds['cpu_threshold']}%)",
                "system"
            )
        else:
            self._resolve_alert("system_cpu_high")
        
        # Memory alert
        if metrics.memory_percent > self.alert_thresholds['memory_threshold']:
            self._create_alert(
                "system_memory_high",
                AlertSeverity.WARNING,
                "High Memory Usage",
                f"Memory usage is {metrics.memory_percent:.1f}% (threshold: {self.alert_thresholds['memory_threshold']}%)",
                "system"
            )
        else:
            self._resolve_alert("system_memory_high")
        
        # Disk alert
        if metrics.disk_percent > self.alert_thresholds['disk_threshold']:
            self._create_alert(
                "system_disk_high",
                AlertSeverity.ERROR,
                "High Disk Usage",
                f"Disk usage is {metrics.disk_percent:.1f}% (threshold: {self.alert_thresholds['disk_threshold']}%)",
                "system"
            )
        else:
            self._resolve_alert("system_disk_high")
    
    async def check_worker_alerts(self, worker_metrics: List[WorkerMetrics]):
        """Check worker metrics against thresholds and generate alerts."""
        current_time = time.time()
        
        try:
            hub = await get_intelligence_hub()
            task_status = hub.get_task_status()
            queue_length = task_status.get('queue_length', 0)
            
            # Queue length alert
            if queue_length > self.alert_thresholds['queue_length_threshold']:
                self._create_alert(
                    "task_queue_high",
                    AlertSeverity.WARNING,
                    "High Task Queue Length",
                    f"Task queue has {queue_length} pending tasks (threshold: {self.alert_thresholds['queue_length_threshold']})",
                    "task_queue"
                )
            else:
                self._resolve_alert("task_queue_high")
            
        except Exception as e:
            self.logger.intelligence_event("Error checking task queue", {"error": str(e)})
        
        # Worker-specific alerts
        for worker in worker_metrics:
            worker_id = worker.worker_id
            
            # Worker timeout alert
            if (worker.last_activity and 
                current_time - worker.last_activity > self.alert_thresholds['worker_timeout']):
                self._create_alert(
                    f"worker_timeout_{worker_id}",
                    AlertSeverity.ERROR,
                    "Worker Timeout",
                    f"Worker {worker_id} hasn't reported activity for {(current_time - worker.last_activity)/60:.1f} minutes",
                    f"worker_{worker_id}"
                )
            else:
                self._resolve_alert(f"worker_timeout_{worker_id}")
            
            # Worker failure rate alert
            total_tasks = worker.tasks_completed + worker.tasks_failed
            if (total_tasks >= 10 and  # Only alert if worker has done significant work
                worker.success_rate < (1.0 - self.alert_thresholds['failure_rate_threshold'])):
                self._create_alert(
                    f"worker_failure_rate_{worker_id}",
                    AlertSeverity.WARNING,
                    "High Worker Failure Rate",
                    f"Worker {worker_id} has failure rate of {(1.0-worker.success_rate)*100:.1f}% (threshold: {self.alert_thresholds['failure_rate_threshold']*100:.1f}%)",
                    f"worker_{worker_id}"
                )
            else:
                self._resolve_alert(f"worker_failure_rate_{worker_id}")
    
    def _create_alert(self, alert_id: str, severity: AlertSeverity, title: str, message: str, component: str):
        """Create or update an alert."""
        if alert_id in self.alerts and not self.alerts[alert_id].resolved:
            return  # Alert already exists and is not resolved
        
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            title=title,
            message=message,
            component=component,
            timestamp=time.time()
        )
        
        self.alerts[alert_id] = alert
        
        self.logger.coordination_event("Alert created", {
            "alert_id": alert_id,
            "severity": severity.value,
            "title": title,
            "component": component
        })
    
    def _resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        if alert_id in self.alerts and not self.alerts[alert_id].resolved:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolved_at = time.time()
            
            self.logger.coordination_event("Alert resolved", {
                "alert_id": alert_id,
                "duration": time.time() - self.alerts[alert_id].timestamp
            })
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_all_alerts(self, limit: int = 50) -> List[Alert]:
        """Get all alerts (resolved and unresolved)."""
        alerts = list(self.alerts.values())
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return alerts[:limit]
    
    async def generate_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report."""
        # Collect current metrics
        system_metrics = self.get_system_metrics()
        worker_metrics = await self.collect_worker_metrics()
        
        # Store in history
        self.system_metrics_history.append(system_metrics)
        if len(self.system_metrics_history) > self.max_history_length:
            self.system_metrics_history.pop(0)
        
        for worker in worker_metrics:
            worker_id = worker.worker_id
            if worker_id not in self.worker_metrics_history:
                self.worker_metrics_history[worker_id] = []
            
            self.worker_metrics_history[worker_id].append(worker)
            if len(self.worker_metrics_history[worker_id]) > self.max_history_length:
                self.worker_metrics_history[worker_id].pop(0)
        
        # Check for alerts
        self.check_system_alerts(system_metrics)
        await self.check_worker_alerts(worker_metrics)
        
        # Get intelligence hub status
        try:
            hub = await get_intelligence_hub()
            performance_metrics = hub.get_performance_metrics()
            task_status = hub.get_task_status()
        except Exception as e:
            self.logger.intelligence_event("Error getting hub status", {"error": str(e)})
            performance_metrics = {}
            task_status = {}
        
        # Generate report
        report = {
            "timestamp": time.time(),
            "system": {
                "current": system_metrics.to_dict(),
                "history": [m.to_dict() for m in self.system_metrics_history[-10:]]  # Last 10 readings
            },
            "workers": {
                "current": [w.to_dict() for w in worker_metrics],
                "summary": self._generate_worker_summary(worker_metrics),
                "performance": performance_metrics
            },
            "tasks": task_status,
            "alerts": {
                "active": [a.to_dict() for a in self.get_active_alerts()],
                "total_active": len(self.get_active_alerts()),
                "recent": [a.to_dict() for a in self.get_all_alerts(10)]
            },
            "uptime": time.time() - self.last_report_time if self.last_report_time > 0 else 0
        }
        
        self.last_report_time = time.time()
        return report
    
    def _generate_worker_summary(self, worker_metrics: List[WorkerMetrics]) -> Dict[str, Any]:
        """Generate summary statistics for workers."""
        if not worker_metrics:
            return {
                "total_workers": 0,
                "active_workers": 0,
                "busy_workers": 0,
                "total_tasks_completed": 0,
                "total_tasks_failed": 0,
                "average_success_rate": 0.0,
                "average_duration": 0.0
            }
        
        active_workers = [w for w in worker_metrics if w.status in ['available', 'busy']]
        busy_workers = [w for w in worker_metrics if w.status == 'busy']
        
        total_completed = sum(w.tasks_completed for w in worker_metrics)
        total_failed = sum(w.tasks_failed for w in worker_metrics)
        
        # Calculate averages only for workers with tasks
        workers_with_tasks = [w for w in worker_metrics if w.tasks_completed + w.tasks_failed > 0]
        avg_success_rate = sum(w.success_rate for w in workers_with_tasks) / max(len(workers_with_tasks), 1)
        avg_duration = sum(w.avg_duration for w in workers_with_tasks) / max(len(workers_with_tasks), 1)
        
        return {
            "total_workers": len(worker_metrics),
            "active_workers": len(active_workers),
            "busy_workers": len(busy_workers),
            "total_tasks_completed": total_completed,
            "total_tasks_failed": total_failed,
            "average_success_rate": avg_success_rate,
            "average_duration": avg_duration,
            "worker_types": self._count_worker_types(worker_metrics)
        }
    
    def _count_worker_types(self, worker_metrics: List[WorkerMetrics]) -> Dict[str, int]:
        """Count workers by type."""
        type_counts = {}
        for worker in worker_metrics:
            # Try to extract worker type from worker_id (assumes format like "continuous_001")
            worker_type = worker.worker_id.split('_')[0] if '_' in worker.worker_id else 'unknown'
            type_counts[worker_type] = type_counts.get(worker_type, 0) + 1
        return type_counts
    
    async def save_report_to_file(self, report: Dict[str, Any], filename: Optional[str] = None):
        """Save status report to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/worker_status_report_{timestamp}.json"
        
        try:
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.performance_metric("report_saved", 1, "count")
            
        except Exception as e:
            self.logger.intelligence_event("Error saving report", {"error": str(e), "filename": filename})
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Generate status report
                report = await self.generate_status_report()
                
                # Update dashboard data
                self.dashboard_data = report
                
                # Log summary
                worker_summary = report['workers']['summary']
                system_current = report['system']['current']
                active_alerts = len(report['alerts']['active'])
                
                self.logger.performance_metric("monitoring_cycle", 1, "count")
                self.logger.coordination_event("Status report generated", {
                    "workers": worker_summary['total_workers'],
                    "active_workers": worker_summary['active_workers'],
                    "busy_workers": worker_summary['busy_workers'],
                    "cpu_percent": system_current['cpu_percent'],
                    "memory_percent": system_current['memory_percent'],
                    "active_alerts": active_alerts,
                    "tasks_completed": worker_summary['total_tasks_completed'],
                    "avg_success_rate": worker_summary['average_success_rate']
                })
                
                # Save periodic reports
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    await self.save_report_to_file(report)
                
                await asyncio.sleep(self.report_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.intelligence_event("Error in monitoring loop", {"error": str(e)})
                await asyncio.sleep(5)  # Short delay before retrying
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return self.dashboard_data
    
    def print_status_summary(self):
        """Print a concise status summary to console."""
        data = self.dashboard_data
        if not data:
            print("No status data available")
            return
        
        print("\n" + "="*60)
        print("WORKER STATUS SUMMARY")
        print("="*60)
        
        # System status
        system = data.get('system', {}).get('current', {})
        print(f"System: CPU {system.get('cpu_percent', 0):.1f}% | "
              f"Memory {system.get('memory_percent', 0):.1f}% | "
              f"Disk {system.get('disk_percent', 0):.1f}%")
        
        # Worker status
        worker_summary = data.get('workers', {}).get('summary', {})
        print(f"Workers: {worker_summary.get('active_workers', 0)}/{worker_summary.get('total_workers', 0)} active | "
              f"{worker_summary.get('busy_workers', 0)} busy | "
              f"Success rate: {worker_summary.get('average_success_rate', 0)*100:.1f}%")
        
        # Task status
        tasks = data.get('tasks', {})
        print(f"Tasks: {tasks.get('queue_length', 0)} pending | "
              f"{tasks.get('total_tasks', 0)} total")
        
        # Alerts
        alerts = data.get('alerts', {})
        active_alerts = alerts.get('total_active', 0)
        if active_alerts > 0:
            print(f"⚠️  {active_alerts} active alerts")
            for alert in alerts.get('active', [])[:3]:  # Show first 3 alerts
                print(f"   • {alert.get('title', 'Unknown')}: {alert.get('message', '')}")
        else:
            print("✅ No active alerts")
        
        print("="*60)


# Global status reporter instance
_status_reporter: Optional[WorkerStatusReporter] = None

def get_status_reporter() -> WorkerStatusReporter:
    """Get or create the global status reporter."""
    global _status_reporter
    if _status_reporter is None:
        _status_reporter = WorkerStatusReporter()
    return _status_reporter


# Example usage and testing
if __name__ == "__main__":
    async def demo():
        # Setup logging
        setup_worker_logging()
        
        # Create and start status reporter
        reporter = WorkerStatusReporter(report_interval=5.0)  # 5 second intervals for demo
        await reporter.start()
        
        print("Worker Status Reporter Demo")
        print("Monitoring for 30 seconds...")
        
        # Monitor for 30 seconds
        for i in range(6):
            await asyncio.sleep(5)
            reporter.print_status_summary()
        
        await reporter.stop()
        print("Demo completed")
    
    asyncio.run(demo())