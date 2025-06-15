"""
Proactive Monitoring System

Anticipates user needs and system issues before they become problems.
Monitors system health, suggests actions, and provides intelligent alerts.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import psutil
from pathlib import Path

from scripts.state_manager import StateManager
from scripts.task_manager import TaskManager
from scripts.semantic_memory_system import SemanticMemorySystem
from scripts.http_ai_client import HTTPAIClient


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitoringMetric(Enum):
    """Types of metrics to monitor."""
    SYSTEM_HEALTH = "system_health"
    TASK_QUEUE = "task_queue"
    ERROR_RATE = "error_rate"
    PERFORMANCE = "performance"
    DISK_USAGE = "disk_usage"
    MEMORY_USAGE = "memory_usage"
    REPOSITORY_ACTIVITY = "repository_activity"
    AI_USAGE = "ai_usage"
    USER_PATTERNS = "user_patterns"


@dataclass
class Alert:
    """Represents a system alert."""
    id: str
    severity: AlertSeverity
    metric: MonitoringMetric
    title: str
    description: str
    suggestions: List[str] = field(default_factory=list)
    timestamp: datetime = None
    acknowledged: bool = False
    auto_resolved: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class SystemMetrics:
    """Current system metrics."""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    active_tasks: int
    pending_tasks: int
    error_count: int
    uptime_hours: float
    ai_requests_hour: int
    repositories_tracked: int
    last_user_activity: Optional[datetime] = None


@dataclass
class Suggestion:
    """Proactive suggestion for the user."""
    id: str
    title: str
    description: str
    action: str
    priority: float
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class ProactiveMonitor:
    """Base class for proactive monitors."""
    
    def __init__(self, name: str, check_interval: int = 60):
        """Initialize monitor.
        
        Args:
            name: Monitor name
            check_interval: Check interval in seconds
        """
        self.name = name
        self.check_interval = check_interval
        self.logger = logging.getLogger(f"Monitor.{name}")
        self.last_check = None
        self.alerts: List[Alert] = []
        
    async def check(self, metrics: SystemMetrics) -> List[Alert]:
        """Check for issues and return alerts.
        
        Args:
            metrics: Current system metrics
            
        Returns:
            List of new alerts
        """
        raise NotImplementedError


class HealthMonitor(ProactiveMonitor):
    """Monitors overall system health."""
    
    def __init__(self):
        super().__init__("HealthMonitor", check_interval=30)
        self.health_history = []
        
    async def check(self, metrics: SystemMetrics) -> List[Alert]:
        """Check system health."""
        alerts = []
        
        # CPU usage
        if metrics.cpu_percent > 90:
            alerts.append(Alert(
                id=f"cpu_high_{datetime.now().timestamp()}",
                severity=AlertSeverity.CRITICAL,
                metric=MonitoringMetric.SYSTEM_HEALTH,
                title="High CPU Usage",
                description=f"CPU usage is at {metrics.cpu_percent}%",
                suggestions=[
                    "Check for runaway processes",
                    "Consider stopping non-critical tasks",
                    "Review continuous AI worker count"
                ]
            ))
        elif metrics.cpu_percent > 70:
            alerts.append(Alert(
                id=f"cpu_warning_{datetime.now().timestamp()}",
                severity=AlertSeverity.WARNING,
                metric=MonitoringMetric.SYSTEM_HEALTH,
                title="Elevated CPU Usage",
                description=f"CPU usage is at {metrics.cpu_percent}%",
                suggestions=["Monitor for further increases"]
            ))
        
        # Memory usage
        if metrics.memory_percent > 85:
            alerts.append(Alert(
                id=f"memory_high_{datetime.now().timestamp()}",
                severity=AlertSeverity.ERROR,
                metric=MonitoringMetric.MEMORY_USAGE,
                title="High Memory Usage",
                description=f"Memory usage is at {metrics.memory_percent}%",
                suggestions=[
                    "Clear caches and temporary files",
                    "Restart long-running processes",
                    "Check for memory leaks"
                ]
            ))
        
        # Disk usage
        if metrics.disk_usage_percent > 90:
            alerts.append(Alert(
                id=f"disk_critical_{datetime.now().timestamp()}",
                severity=AlertSeverity.CRITICAL,
                metric=MonitoringMetric.DISK_USAGE,
                title="Critical Disk Space",
                description=f"Only {100 - metrics.disk_usage_percent:.1f}% disk space remaining",
                suggestions=[
                    "Clear old log files",
                    "Remove unused repositories",
                    "Archive old data"
                ]
            ))
        
        return alerts


class TaskQueueMonitor(ProactiveMonitor):
    """Monitors task queue health."""
    
    def __init__(self):
        super().__init__("TaskQueueMonitor", check_interval=60)
        self.task_history = []
        
    async def check(self, metrics: SystemMetrics) -> List[Alert]:
        """Check task queue health."""
        alerts = []
        
        # Task buildup
        total_tasks = metrics.active_tasks + metrics.pending_tasks
        
        if metrics.pending_tasks > 50:
            alerts.append(Alert(
                id=f"task_backlog_{datetime.now().timestamp()}",
                severity=AlertSeverity.WARNING,
                metric=MonitoringMetric.TASK_QUEUE,
                title="Task Queue Backlog",
                description=f"{metrics.pending_tasks} tasks pending",
                suggestions=[
                    "Start continuous AI system to process tasks",
                    "Increase worker count",
                    "Review task priorities"
                ]
            ))
        
        # No active workers with pending tasks
        if metrics.active_tasks == 0 and metrics.pending_tasks > 0:
            alerts.append(Alert(
                id=f"no_workers_{datetime.now().timestamp()}",
                severity=AlertSeverity.ERROR,
                metric=MonitoringMetric.TASK_QUEUE,
                title="No Active Workers",
                description="Tasks are queued but no workers are active",
                suggestions=[
                    "Start the continuous AI system",
                    "Check for worker errors",
                    "Manually process critical tasks"
                ]
            ))
        
        return alerts


class ErrorRateMonitor(ProactiveMonitor):
    """Monitors error rates and patterns."""
    
    def __init__(self):
        super().__init__("ErrorRateMonitor", check_interval=120)
        self.error_window = []
        self.error_threshold = 10  # errors per hour
        
    async def check(self, metrics: SystemMetrics) -> List[Alert]:
        """Check error rates."""
        alerts = []
        
        # High error count
        if metrics.error_count > self.error_threshold:
            alerts.append(Alert(
                id=f"high_errors_{datetime.now().timestamp()}",
                severity=AlertSeverity.ERROR,
                metric=MonitoringMetric.ERROR_RATE,
                title="High Error Rate",
                description=f"{metrics.error_count} errors in the last hour",
                suggestions=[
                    "Check recent logs for patterns",
                    "Review recent system changes",
                    "Run system diagnostics"
                ]
            ))
        
        return alerts


class UserActivityMonitor(ProactiveMonitor):
    """Monitors user activity patterns."""
    
    def __init__(self, memory_system: SemanticMemorySystem):
        super().__init__("UserActivityMonitor", check_interval=300)
        self.memory_system = memory_system
        
    async def check(self, metrics: SystemMetrics) -> List[Alert]:
        """Check user activity patterns."""
        alerts = []
        
        # Inactivity detection
        if metrics.last_user_activity:
            inactive_hours = (datetime.now(timezone.utc) - metrics.last_user_activity).total_seconds() / 3600
            
            if inactive_hours > 24:
                alerts.append(Alert(
                    id=f"user_inactive_{datetime.now().timestamp()}",
                    severity=AlertSeverity.INFO,
                    metric=MonitoringMetric.USER_PATTERNS,
                    title="Extended Inactivity",
                    description=f"No user activity for {inactive_hours:.0f} hours",
                    suggestions=[
                        "Run maintenance tasks",
                        "Update repository information",
                        "Process background improvements"
                    ]
                ))
        
        return alerts


class ProactiveMonitoringSystem:
    """Main proactive monitoring system."""
    
    def __init__(self, state_manager: StateManager, task_manager: TaskManager,
                 memory_system: SemanticMemorySystem):
        """Initialize the monitoring system."""
        self.logger = logging.getLogger(__name__)
        self.state_manager = state_manager
        self.task_manager = task_manager
        self.memory_system = memory_system
        self.ai_client = HTTPAIClient(enable_round_robin=True)
        
        # Initialize monitors
        self.monitors = [
            HealthMonitor(),
            TaskQueueMonitor(),
            ErrorRateMonitor(),
            UserActivityMonitor(memory_system)
        ]
        
        # Alert management
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.suggestions: List[Suggestion] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
        
    async def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Proactive monitoring started")
        
    async def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
                
        self.logger.info("Proactive monitoring stopped")
        
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                
                # Run monitors
                new_alerts = []
                for monitor in self.monitors:
                    if self._should_run_monitor(monitor):
                        alerts = await monitor.check(metrics)
                        new_alerts.extend(alerts)
                        monitor.last_check = datetime.now(timezone.utc)
                
                # Process new alerts
                for alert in new_alerts:
                    await self._process_alert(alert)
                
                # Generate suggestions
                await self._generate_suggestions(metrics, new_alerts)
                
                # Clean up old alerts
                await self._cleanup_alerts()
                
                # Sleep before next check
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(30)  # Wait longer on error
                
    def _should_run_monitor(self, monitor: ProactiveMonitor) -> bool:
        """Check if monitor should run."""
        if monitor.last_check is None:
            return True
            
        elapsed = (datetime.now(timezone.utc) - monitor.last_check).total_seconds()
        return elapsed >= monitor.check_interval
        
    async def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Task metrics
        tasks = self.task_manager.get_task_queue()
        active_tasks = len([t for t in tasks if t.get('status') == 'active'])
        pending_tasks = len([t for t in tasks if t.get('status') == 'pending'])
        
        # Error count (from logs)
        error_count = await self._count_recent_errors()
        
        # Uptime
        boot_time = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc)
        uptime_hours = (datetime.now(timezone.utc) - boot_time).total_seconds() / 3600
        
        # AI usage (simplified)
        ai_requests_hour = len(self.memory_system.memories) % 100  # Mock value
        
        # Repository count
        self.state_manager.load_state()
        repositories_tracked = len(self.state_manager.state.get('projects', {}))
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_usage_percent=disk.percent,
            active_tasks=active_tasks,
            pending_tasks=pending_tasks,
            error_count=error_count,
            uptime_hours=uptime_hours,
            ai_requests_hour=ai_requests_hour,
            repositories_tracked=repositories_tracked,
            last_user_activity=self._get_last_user_activity()
        )
        
    async def _count_recent_errors(self) -> int:
        """Count recent errors in logs."""
        error_count = 0
        log_files = ['continuous_ai.log', 'system.log']
        
        for log_file in log_files:
            if Path(log_file).exists():
                try:
                    with open(log_file, 'r') as f:
                        # Read last 1000 lines
                        lines = f.readlines()[-1000:]
                        for line in lines:
                            if 'ERROR' in line or 'CRITICAL' in line:
                                error_count += 1
                except:
                    pass
                    
        return error_count
        
    def _get_last_user_activity(self) -> Optional[datetime]:
        """Get timestamp of last user activity."""
        # Check memory system for recent activity
        if self.memory_system.memories:
            recent_memory = max(self.memory_system.memories.values(), 
                              key=lambda m: m.timestamp)
            return recent_memory.timestamp
        return None
        
    async def _process_alert(self, alert: Alert):
        """Process a new alert."""
        # Check if similar alert already exists
        for existing_id, existing_alert in self.active_alerts.items():
            if (existing_alert.metric == alert.metric and 
                existing_alert.severity == alert.severity and
                not existing_alert.acknowledged):
                # Update existing alert instead
                self.logger.debug(f"Updating existing alert: {existing_id}")
                return
                
        # Add new alert
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Log alert
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨"
        }
        
        self.logger.info(
            f"{severity_emoji.get(alert.severity, '•')} {alert.title}: {alert.description}"
        )
        
        # Auto-resolve if possible
        if alert.severity == AlertSeverity.WARNING:
            await self._try_auto_resolve(alert)
            
    async def _try_auto_resolve(self, alert: Alert):
        """Try to automatically resolve an alert."""
        if alert.metric == MonitoringMetric.TASK_QUEUE and "start continuous AI" in str(alert.suggestions):
            # Could auto-start continuous AI here
            self.logger.info(f"Could auto-resolve: {alert.title}")
            
    async def _generate_suggestions(self, metrics: SystemMetrics, new_alerts: List[Alert]):
        """Generate proactive suggestions based on metrics and alerts."""
        suggestions = []
        
        # Suggest based on time patterns
        current_hour = datetime.now().hour
        
        if current_hour == 2 and metrics.last_user_activity:
            # Late night maintenance window
            inactive_hours = (datetime.now(timezone.utc) - metrics.last_user_activity).total_seconds() / 3600
            if inactive_hours > 4:
                suggestions.append(Suggestion(
                    id="night_maintenance",
                    title="Run Night Maintenance",
                    description="Good time for system maintenance while inactive",
                    action="run_maintenance",
                    priority=0.7,
                    context={"reason": "night_time", "inactive_hours": inactive_hours}
                ))
                
        # Suggest based on metrics
        if metrics.pending_tasks > 10 and metrics.active_tasks == 0:
            suggestions.append(Suggestion(
                id="start_workers",
                title="Start Task Processing",
                description=f"{metrics.pending_tasks} tasks are waiting to be processed",
                action="start_continuous_ai",
                priority=0.9,
                context={"pending_tasks": metrics.pending_tasks}
            ))
            
        # Suggest based on repository activity
        if metrics.repositories_tracked > 5 and metrics.ai_requests_hour < 5:
            suggestions.append(Suggestion(
                id="analyze_repos",
                title="Analyze Repository Health",
                description="It's been quiet - good time to analyze repository health",
                action="analyze_all_repositories",
                priority=0.5,
                context={"repo_count": metrics.repositories_tracked}
            ))
            
        # AI-powered suggestions
        if new_alerts:
            ai_suggestions = await self._generate_ai_suggestions(metrics, new_alerts)
            suggestions.extend(ai_suggestions)
            
        # Update suggestions list
        self.suggestions = sorted(suggestions, key=lambda s: s.priority, reverse=True)[:5]
        
    async def _generate_ai_suggestions(self, metrics: SystemMetrics, 
                                      alerts: List[Alert]) -> List[Suggestion]:
        """Use AI to generate intelligent suggestions."""
        if not alerts:
            return []
            
        alert_summary = "\n".join([
            f"- {a.severity.value}: {a.title} - {a.description}"
            for a in alerts[:5]
        ])
        
        prompt = f"""Based on these system alerts and metrics, suggest proactive actions:

Alerts:
{alert_summary}

Metrics:
- CPU: {metrics.cpu_percent}%
- Memory: {metrics.memory_percent}%
- Pending tasks: {metrics.pending_tasks}
- Active tasks: {metrics.active_tasks}

Suggest 1-3 specific actions in JSON format:
[
  {{
    "title": "Action title",
    "description": "Why this helps",
    "action": "specific_command_or_tool",
    "priority": 0.0-1.0
  }}
]"""
        
        response = await self.ai_client.generate_enhanced_response(prompt, prefill='[')
        
        suggestions = []
        if response.get('content'):
            try:
                content = response['content']
                if not content.startswith('['):
                    content = '[' + content
                    
                ai_suggestions = json.loads(content)
                
                for i, sugg in enumerate(ai_suggestions[:3]):
                    suggestions.append(Suggestion(
                        id=f"ai_suggestion_{i}",
                        title=sugg.get('title', 'AI Suggestion'),
                        description=sugg.get('description', ''),
                        action=sugg.get('action', 'review'),
                        priority=float(sugg.get('priority', 0.5)),
                        context={"source": "ai", "alert_based": True}
                    ))
            except:
                pass
                
        return suggestions
        
    async def _cleanup_alerts(self):
        """Clean up old or resolved alerts."""
        now = datetime.now(timezone.utc)
        
        # Remove acknowledged alerts older than 1 hour
        to_remove = []
        for alert_id, alert in self.active_alerts.items():
            if alert.acknowledged:
                age = (now - alert.timestamp).total_seconds() / 3600
                if age > 1:
                    to_remove.append(alert_id)
                    
        for alert_id in to_remove:
            del self.active_alerts[alert_id]
            
    async def get_current_alerts(self) -> List[Alert]:
        """Get current active alerts."""
        return list(self.active_alerts.values())
        
    async def get_suggestions(self) -> List[Suggestion]:
        """Get current suggestions."""
        return self.suggestions
        
    async def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            self.logger.info(f"Alert acknowledged: {alert_id}")
            
    async def execute_suggestion(self, suggestion_id: str) -> Dict[str, Any]:
        """Execute a suggestion."""
        suggestion = next((s for s in self.suggestions if s.id == suggestion_id), None)
        
        if not suggestion:
            return {"success": False, "error": "Suggestion not found"}
            
        self.logger.info(f"Executing suggestion: {suggestion.title}")
        
        # Execute based on action
        if suggestion.action == "start_continuous_ai":
            # Would integrate with continuous AI system
            return {"success": True, "message": "Would start continuous AI system"}
        elif suggestion.action == "run_maintenance":
            # Would run maintenance tasks
            return {"success": True, "message": "Would run maintenance tasks"}
        else:
            return {"success": True, "message": f"Would execute: {suggestion.action}"}
            
    async def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        metrics = await self._collect_metrics()
        
        return {
            "metrics": {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "disk_usage_percent": metrics.disk_usage_percent,
                "active_tasks": metrics.active_tasks,
                "pending_tasks": metrics.pending_tasks,
                "error_count": metrics.error_count,
                "uptime_hours": metrics.uptime_hours,
                "repositories": metrics.repositories_tracked
            },
            "alerts": {
                "active": len(self.active_alerts),
                "critical": len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL]),
                "list": [a.__dict__ for a in list(self.active_alerts.values())[:5]]
            },
            "suggestions": [s.__dict__ for s in self.suggestions],
            "monitoring_active": self.monitoring_active,
            "last_check": max((m.last_check for m in self.monitors if m.last_check), default=None)
        }