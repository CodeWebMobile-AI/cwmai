"""
Redis Monitoring and Health Management

Comprehensive monitoring system for Redis infrastructure including performance metrics,
health checks, alerting, and automatic diagnostics with real-time dashboards.
"""

import asyncio
import json
import time
import psutil
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging
from .redis_client import RedisClient


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMING = "timing"


@dataclass
class Alert:
    """System alert information."""
    id: str
    metric: str
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'metric': self.metric,
            'severity': self.severity.value,
            'message': self.message,
            'value': self.value,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'metadata': self.metadata
        }


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags
        }


@dataclass
class HealthCheck:
    """Health check configuration and results."""
    name: str
    check_function: Callable
    interval: float
    timeout: float
    enabled: bool = True
    last_run: Optional[datetime] = None
    last_result: bool = True
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    
    def is_failing(self, failure_threshold: int = 3) -> bool:
        """Check if health check is consistently failing."""
        return self.consecutive_failures >= failure_threshold


class RedisMonitoring:
    """Comprehensive Redis monitoring system."""
    
    def __init__(self, redis_client: RedisClient, namespace: str = "monitoring"):
        """Initialize Redis monitoring.
        
        Args:
            redis_client: Redis client instance
            namespace: Monitoring namespace for metrics storage
        """
        self.redis = redis_client
        self.namespace = namespace
        self.logger = logging.getLogger(__name__)
        
        # Monitoring configuration
        self.collection_interval = 30.0  # seconds
        self.metrics_retention = 86400 * 7  # 7 days
        self.alert_retention = 86400 * 30  # 30 days
        self.enable_system_metrics = True
        self.enable_redis_metrics = True
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_check_interval = 60.0  # seconds
        self.failure_threshold = 3
        
        # Alerting
        self.alert_handlers: List[Callable] = []
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Initialize default thresholds
        self._setup_default_thresholds()
        self._setup_default_health_checks()
    
    def _metrics_key(self, metric_name: str) -> str:
        """Create metrics storage key."""
        return f"{self.namespace}:metrics:{metric_name}"
    
    def _alerts_key(self) -> str:
        """Create alerts storage key."""
        return f"{self.namespace}:alerts"
    
    def _health_key(self) -> str:
        """Create health check storage key."""
        return f"{self.namespace}:health"
    
    def _setup_default_thresholds(self):
        """Setup default alert thresholds."""
        self.alert_thresholds = {
            'redis_memory_usage_percent': {
                'warning': 80.0,
                'critical': 95.0
            },
            'redis_connected_clients': {
                'warning': 1000,
                'critical': 1500
            },
            'redis_keyspace_misses_rate': {
                'warning': 0.1,
                'critical': 0.2
            },
            'system_cpu_percent': {
                'warning': 80.0,
                'critical': 95.0
            },
            'system_memory_percent': {
                'warning': 85.0,
                'critical': 95.0
            },
            'redis_response_time_ms': {
                'warning': 100.0,
                'critical': 500.0
            }
        }
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        self.add_health_check(
            "redis_ping",
            self._health_check_redis_ping,
            interval=30.0,
            timeout=5.0
        )
        
        self.add_health_check(
            "redis_memory",
            self._health_check_redis_memory,
            interval=60.0,
            timeout=10.0
        )
        
        self.add_health_check(
            "system_resources",
            self._health_check_system_resources,
            interval=60.0,
            timeout=5.0
        )
    
    async def start(self):
        """Start monitoring system."""
        if not self._monitoring_task:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._health_task = asyncio.create_task(self._health_check_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.logger.info("Redis monitoring system started")
    
    async def stop(self):
        """Stop monitoring system."""
        self._shutdown = True
        
        # Stop background tasks
        for task in [self._monitoring_task, self._health_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._monitoring_task = None
        self._health_task = None
        self._cleanup_task = None
        
        self.logger.info("Redis monitoring system stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._shutdown:
            try:
                # Collect metrics
                if self.enable_redis_metrics:
                    await self._collect_redis_metrics()
                
                if self.enable_system_metrics:
                    await self._collect_system_metrics()
                
                # Check for alerts
                await self._check_alerts()
                
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _health_check_loop(self):
        """Health check monitoring loop."""
        while not self._shutdown:
            try:
                await self._run_health_checks()
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _cleanup_loop(self):
        """Cleanup old metrics and alerts."""
        while not self._shutdown:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Run every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)
    
    async def _collect_redis_metrics(self):
        """Collect Redis-specific metrics."""
        try:
            start_time = time.time()
            
            # Get Redis info
            info = await self.redis.info()
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000  # milliseconds
            
            # Memory metrics
            used_memory = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            
            memory_percent = 0
            if max_memory > 0:
                memory_percent = (used_memory / max_memory) * 100
            
            await self._record_metric("redis_memory_usage_bytes", used_memory, MetricType.GAUGE)
            await self._record_metric("redis_memory_usage_percent", memory_percent, MetricType.GAUGE)
            
            # Client metrics
            connected_clients = info.get('connected_clients', 0)
            await self._record_metric("redis_connected_clients", connected_clients, MetricType.GAUGE)
            
            # Keyspace metrics
            keyspace_hits = info.get('keyspace_hits', 0)
            keyspace_misses = info.get('keyspace_misses', 0)
            total_requests = keyspace_hits + keyspace_misses
            
            miss_rate = 0
            if total_requests > 0:
                miss_rate = keyspace_misses / total_requests
            
            await self._record_metric("redis_keyspace_hits", keyspace_hits, MetricType.COUNTER)
            await self._record_metric("redis_keyspace_misses", keyspace_misses, MetricType.COUNTER)
            await self._record_metric("redis_keyspace_misses_rate", miss_rate, MetricType.GAUGE)
            
            # Performance metrics
            await self._record_metric("redis_response_time_ms", response_time, MetricType.TIMING)
            
            # Commands metrics
            total_commands = info.get('total_commands_processed', 0)
            await self._record_metric("redis_commands_processed", total_commands, MetricType.COUNTER)
            
            # Persistence metrics
            rdb_last_save = info.get('rdb_last_save_time', 0)
            await self._record_metric("redis_rdb_last_save", rdb_last_save, MetricType.GAUGE)
            
            # Replication metrics
            connected_slaves = info.get('connected_slaves', 0)
            await self._record_metric("redis_connected_slaves", connected_slaves, MetricType.GAUGE)
            
        except Exception as e:
            self.logger.error(f"Error collecting Redis metrics: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            await self._record_metric("system_cpu_percent", cpu_percent, MetricType.GAUGE)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            await self._record_metric("system_memory_total", memory.total, MetricType.GAUGE)
            await self._record_metric("system_memory_used", memory.used, MetricType.GAUGE)
            await self._record_metric("system_memory_percent", memory.percent, MetricType.GAUGE)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            await self._record_metric("system_disk_total", disk.total, MetricType.GAUGE)
            await self._record_metric("system_disk_used", disk.used, MetricType.GAUGE)
            await self._record_metric("system_disk_percent", (disk.used / disk.total) * 100, MetricType.GAUGE)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            await self._record_metric("system_network_bytes_sent", net_io.bytes_sent, MetricType.COUNTER)
            await self._record_metric("system_network_bytes_recv", net_io.bytes_recv, MetricType.COUNTER)
            
            # Load average (Unix-like systems only)
            try:
                load_avg = psutil.getloadavg()
                await self._record_metric("system_load_1min", load_avg[0], MetricType.GAUGE)
                await self._record_metric("system_load_5min", load_avg[1], MetricType.GAUGE)
                await self._record_metric("system_load_15min", load_avg[2], MetricType.GAUGE)
            except AttributeError:
                pass  # Not available on Windows
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    async def _record_metric(self, name: str, value: Union[int, float], 
                           metric_type: MetricType, tags: Dict[str, str] = None):
        """Record a metric data point."""
        try:
            metric_point = MetricPoint(
                name=name,
                value=value,
                metric_type=metric_type,
                timestamp=datetime.now(timezone.utc),
                tags=tags or {}
            )
            
            metrics_key = self._metrics_key(name)
            
            # Store metric with timestamp as score for time-series data
            timestamp_score = time.time()
            metric_data = json.dumps(metric_point.to_dict())
            
            await self.redis.zadd(metrics_key, {metric_data: timestamp_score})
            
            # Set expiration to prevent unlimited growth
            await self.redis.expire(metrics_key, self.metrics_retention)
            
        except Exception as e:
            self.logger.error(f"Error recording metric {name}: {e}")
    
    async def _check_alerts(self):
        """Check metrics against alert thresholds."""
        try:
            for metric_name, thresholds in self.alert_thresholds.items():
                # Get latest metric value
                latest_value = await self._get_latest_metric(metric_name)
                
                if latest_value is None:
                    continue
                
                # Check thresholds
                for severity_name, threshold in thresholds.items():
                    severity = AlertSeverity(severity_name)
                    alert_id = f"{metric_name}:{severity_name}"
                    
                    if latest_value >= threshold:
                        # Threshold exceeded
                        if alert_id not in self.active_alerts:
                            alert = Alert(
                                id=alert_id,
                                metric=metric_name,
                                severity=severity,
                                message=f"{metric_name} is {latest_value:.2f}, exceeding {severity_name} threshold of {threshold}",
                                value=latest_value,
                                threshold=threshold,
                                timestamp=datetime.now(timezone.utc)
                            )
                            
                            self.active_alerts[alert_id] = alert
                            await self._store_alert(alert)
                            await self._trigger_alert_handlers(alert)
                    else:
                        # Threshold not exceeded, resolve alert if active
                        if alert_id in self.active_alerts:
                            alert = self.active_alerts.pop(alert_id)
                            alert.resolved = True
                            alert.resolved_at = datetime.now(timezone.utc)
                            
                            await self._store_alert(alert)
                            await self._trigger_alert_handlers(alert)
        
        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")
    
    async def _get_latest_metric(self, metric_name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        try:
            metrics_key = self._metrics_key(metric_name)
            
            # Get the most recent metric (highest score)
            latest = await self.redis.zrevrange(metrics_key, 0, 0, withscores=True)
            
            if latest:
                metric_data = json.loads(latest[0][0])
                return float(metric_data['value'])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting latest metric {metric_name}: {e}")
            return None
    
    async def _store_alert(self, alert: Alert):
        """Store alert in Redis."""
        try:
            alerts_key = self._alerts_key()
            alert_data = json.dumps(alert.to_dict())
            timestamp_score = time.time()
            
            await self.redis.zadd(alerts_key, {alert_data: timestamp_score})
            await self.redis.expire(alerts_key, self.alert_retention)
            
        except Exception as e:
            self.logger.error(f"Error storing alert: {e}")
    
    async def _trigger_alert_handlers(self, alert: Alert):
        """Trigger registered alert handlers."""
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
    
    async def _run_health_checks(self):
        """Run all registered health checks."""
        for check_name, health_check in self.health_checks.items():
            if not health_check.enabled:
                continue
            
            try:
                start_time = time.time()
                
                # Run health check with timeout
                if asyncio.iscoroutinefunction(health_check.check_function):
                    result = await asyncio.wait_for(
                        health_check.check_function(),
                        timeout=health_check.timeout
                    )
                else:
                    result = health_check.check_function()
                
                health_check.last_run = datetime.now(timezone.utc)
                health_check.last_result = result
                health_check.last_error = None
                
                if result:
                    health_check.consecutive_failures = 0
                else:
                    health_check.consecutive_failures += 1
                
                # Store health check result
                await self._store_health_check_result(check_name, health_check)
                
                # Check if health check is consistently failing
                if health_check.is_failing(self.failure_threshold):
                    await self._trigger_health_alert(check_name, health_check)
                
            except asyncio.TimeoutError:
                health_check.consecutive_failures += 1
                health_check.last_error = "Health check timeout"
                self.logger.warning(f"Health check {check_name} timed out")
                
            except Exception as e:
                health_check.consecutive_failures += 1
                health_check.last_error = str(e)
                self.logger.error(f"Health check {check_name} failed: {e}")
    
    async def _store_health_check_result(self, check_name: str, health_check: HealthCheck):
        """Store health check result."""
        try:
            health_key = self._health_key()
            
            result_data = {
                'name': check_name,
                'result': health_check.last_result,
                'timestamp': health_check.last_run.isoformat() if health_check.last_run else None,
                'error': health_check.last_error,
                'consecutive_failures': health_check.consecutive_failures
            }
            
            await self.redis.hset(health_key, check_name, json.dumps(result_data))
            await self.redis.expire(health_key, 86400)  # Expire in 1 day
            
        except Exception as e:
            self.logger.error(f"Error storing health check result: {e}")
    
    async def _trigger_health_alert(self, check_name: str, health_check: HealthCheck):
        """Trigger alert for failing health check."""
        alert_id = f"health_check:{check_name}"
        
        if alert_id not in self.active_alerts:
            alert = Alert(
                id=alert_id,
                metric=f"health_check_{check_name}",
                severity=AlertSeverity.CRITICAL,
                message=f"Health check '{check_name}' has failed {health_check.consecutive_failures} consecutive times",
                value=health_check.consecutive_failures,
                threshold=self.failure_threshold,
                timestamp=datetime.now(timezone.utc),
                metadata={'last_error': health_check.last_error}
            )
            
            self.active_alerts[alert_id] = alert
            await self._store_alert(alert)
            await self._trigger_alert_handlers(alert)
    
    async def _health_check_redis_ping(self) -> bool:
        """Basic Redis ping health check."""
        try:
            result = await self.redis.ping()
            return result is True
        except Exception:
            return False
    
    async def _health_check_redis_memory(self) -> bool:
        """Redis memory usage health check."""
        try:
            info = await self.redis.info('memory')
            used_memory = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            
            if max_memory == 0:
                return True  # No memory limit set
            
            usage_percent = (used_memory / max_memory) * 100
            return usage_percent < 90  # Consider unhealthy if > 90% memory usage
            
        except Exception:
            return False
    
    async def _health_check_system_resources(self) -> bool:
        """System resources health check."""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                return False
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return False
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 90:
                return False
            
            return True
            
        except Exception:
            return False
    
    async def _cleanup_old_data(self):
        """Cleanup old metrics and alerts."""
        try:
            current_time = time.time()
            cutoff_time = current_time - self.metrics_retention
            
            # Cleanup old metrics
            pattern = f"{self.namespace}:metrics:*"
            cursor = 0
            
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
                
                for key in keys:
                    # Remove old metric points
                    await self.redis.zremrangebyscore(key, 0, cutoff_time)
                
                if cursor == 0:
                    break
            
            # Cleanup old alerts
            alerts_key = self._alerts_key()
            alert_cutoff = current_time - self.alert_retention
            await self.redis.zremrangebyscore(alerts_key, 0, alert_cutoff)
            
        except Exception as e:
            self.logger.error(f"Error in cleanup: {e}")
    
    def add_health_check(self, name: str, check_function: Callable, 
                        interval: float = 60.0, timeout: float = 30.0, enabled: bool = True):
        """Add a custom health check.
        
        Args:
            name: Health check name
            check_function: Function that returns True if healthy
            interval: Check interval in seconds
            timeout: Check timeout in seconds
            enabled: Whether check is enabled
        """
        self.health_checks[name] = HealthCheck(
            name=name,
            check_function=check_function,
            interval=interval,
            timeout=timeout,
            enabled=enabled
        )
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler function.
        
        Args:
            handler: Function to handle alerts
        """
        self.alert_handlers.append(handler)
    
    def set_alert_threshold(self, metric_name: str, severity: str, threshold: float):
        """Set alert threshold for a metric.
        
        Args:
            metric_name: Name of the metric
            severity: Severity level (warning, critical)
            threshold: Threshold value
        """
        if metric_name not in self.alert_thresholds:
            self.alert_thresholds[metric_name] = {}
        
        self.alert_thresholds[metric_name][severity] = threshold
    
    async def get_metrics(self, metric_name: str, start_time: datetime = None, 
                         end_time: datetime = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical metrics data.
        
        Args:
            metric_name: Name of the metric
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            limit: Maximum number of data points
            
        Returns:
            List of metric data points
        """
        try:
            metrics_key = self._metrics_key(metric_name)
            
            # Convert timestamps to scores
            start_score = start_time.timestamp() if start_time else 0
            end_score = end_time.timestamp() if end_time else time.time()
            
            # Get metrics in time range
            results = await self.redis.zrangebyscore(
                metrics_key, start_score, end_score, withscores=True
            )
            
            # Limit results
            if len(results) > limit:
                results = results[-limit:]
            
            metrics = []
            for metric_data, _ in results:
                try:
                    metrics.append(json.loads(metric_data))
                except json.JSONDecodeError:
                    continue
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting metrics for {metric_name}: {e}")
            return []
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts."""
        return list(self.active_alerts.values())
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        try:
            health_key = self._health_key()
            health_data = await self.redis.hgetall(health_key)
            
            health_status = {}
            overall_healthy = True
            
            for check_name, result_json in health_data.items():
                try:
                    result_data = json.loads(result_json)
                    health_status[check_name] = result_data
                    
                    if not result_data.get('result', False):
                        overall_healthy = False
                        
                except json.JSONDecodeError:
                    continue
            
            return {
                'overall_healthy': overall_healthy,
                'checks': health_status,
                'active_alerts_count': len(self.active_alerts),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting health status: {e}")
            return {
                'overall_healthy': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        try:
            # Get recent metrics for key indicators
            metrics_data = {}
            key_metrics = [
                'redis_memory_usage_percent',
                'redis_connected_clients',
                'redis_response_time_ms',
                'system_cpu_percent',
                'system_memory_percent'
            ]
            
            for metric in key_metrics:
                latest = await self._get_latest_metric(metric)
                if latest is not None:
                    metrics_data[metric] = latest
            
            # Get health status
            health_status = await self.get_health_status()
            
            # Get active alerts
            active_alerts = await self.get_active_alerts()
            
            return {
                'metrics': metrics_data,
                'health': health_status,
                'alerts': [alert.to_dict() for alert in active_alerts],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }