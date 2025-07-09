"""
Intelligent Monitoring System

Tracks all system activities and provides real-time insights for improvement:
1. Performance monitoring across all components
2. Error tracking and prediction
3. Success pattern identification
4. Resource usage optimization
5. Behavioral anomaly detection
6. Predictive maintenance
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
import logging
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import numpy as np
from enum import Enum

from scripts.http_ai_client import HTTPAIClient
from scripts.state_manager import StateManager
from scripts.mcp_integration import MCPIntegrationHub


class MetricType(Enum):
    """Types of metrics to track."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"


@dataclass
class Metric:
    """Represents a system metric."""
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Represents a system alert."""
    severity: str  # 'info', 'warning', 'error', 'critical'
    category: str  # 'performance', 'error', 'resource', 'anomaly'
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class Insight:
    """Represents an actionable insight."""
    category: str  # 'optimization', 'prediction', 'pattern', 'anomaly'
    title: str
    description: str
    impact: str  # Expected impact
    confidence: float  # 0-1
    recommendations: List[str]
    supporting_data: Dict[str, Any]
    timestamp: datetime


class IntelligentMonitoringSystem:
    """Monitors system health and provides intelligent insights."""
    
    def __init__(self, state_manager: StateManager):
        self.logger = logging.getLogger(__name__)
        self.state_manager = state_manager
        self.ai_client = HTTPAIClient()
        self.mcp_hub = MCPIntegrationHub()
        
        # Metric storage
        self.metrics = defaultdict(lambda: deque(maxlen=10000))  # Keep last 10k points per metric
        self.metric_aggregates = defaultdict(dict)
        
        # Alert management
        self.active_alerts = []
        self.alert_history = deque(maxlen=1000)
        
        # Insight generation
        self.insights = deque(maxlen=100)
        self.applied_insights = []
        
        # Pattern detection
        self.patterns = {
            'performance': defaultdict(list),
            'error': defaultdict(list),
            'usage': defaultdict(list),
            'success': defaultdict(list)
        }
        
        # Baselines for anomaly detection
        self.baselines = {}
        self.anomaly_threshold = 2.5  # Standard deviations
        
        # Configuration
        self.config = {
            'collection_interval': 10,  # seconds
            'analysis_interval': 60,  # seconds
            'baseline_window': 3600,  # 1 hour
            'alert_cooldown': 300,  # 5 minutes
            'insight_generation_interval': 600,  # 10 minutes
            'min_data_points_for_analysis': 10
        }
        
        # State
        self.is_running = False
        self.last_analysis_time = None
        self.last_insight_time = None
        
        # Registered monitors
        self.monitors = {}
        
    async def start(self):
        """Start the monitoring system."""
        print("Starting Intelligent Monitoring System")
        self.is_running = True
        
        # Start monitoring loops
        asyncio.create_task(self._collection_loop())
        asyncio.create_task(self._analysis_loop())
        asyncio.create_task(self._insight_loop())
        
        # Initialize MCP for external monitoring
        await self.mcp_hub.initialize(servers=['brave-search'])
        
    async def stop(self):
        """Stop the monitoring system."""
        self.is_running = False
        await self.mcp_hub.close()
        
    def register_monitor(self, name: str, monitor_func: Callable):
        """Register a custom monitor function."""
        self.monitors[name] = monitor_func
        print(f"Registered monitor: {name}")
        
    async def _collection_loop(self):
        """Main collection loop for metrics."""
        while self.is_running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.config['collection_interval'])
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(30)
                
    async def _analysis_loop(self):
        """Main analysis loop for patterns and anomalies."""
        while self.is_running:
            try:
                await self._analyze_metrics()
                self.last_analysis_time = datetime.now(timezone.utc)
                await asyncio.sleep(self.config['analysis_interval'])
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(60)
                
    async def _insight_loop(self):
        """Main loop for generating insights."""
        while self.is_running:
            try:
                await self._generate_insights()
                self.last_insight_time = datetime.now(timezone.utc)
                await asyncio.sleep(self.config['insight_generation_interval'])
            except Exception as e:
                self.logger.error(f"Error in insight loop: {e}")
                await asyncio.sleep(300)
                
    async def _collect_metrics(self):
        """Collect all system metrics."""
        timestamp = datetime.now(timezone.utc)
        
        # System resource metrics
        await self._collect_resource_metrics(timestamp)
        
        # Component performance metrics
        await self._collect_performance_metrics(timestamp)
        
        # Error metrics
        await self._collect_error_metrics(timestamp)
        
        # Custom monitors
        for name, monitor in self.monitors.items():
            try:
                metrics = await monitor()
                for metric in metrics:
                    # Handle both Metric objects and dict-based metrics
                    if isinstance(metric, Metric):
                        self.record_metric(metric)
                    elif isinstance(metric, dict) and 'name' in metric:
                        # Convert dict to Metric object
                        metric_obj = Metric(
                            name=metric['name'],
                            type=MetricType[metric.get('type', 'GAUGE').upper()],
                            value=metric['value'],
                            timestamp=metric.get('timestamp', datetime.now(timezone.utc)),
                            tags=metric.get('tags', {}),
                            metadata=metric.get('metadata', {})
                        )
                        self.record_metric(metric_obj)
                    else:
                        self.logger.warning(f"Invalid metric from monitor {name}: {metric}")
            except Exception as e:
                self.logger.error(f"Monitor {name} failed: {e}")
                
    async def _collect_resource_metrics(self, timestamp: datetime):
        """Collect system resource metrics."""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric(Metric(
                name="system.cpu.usage",
                type=MetricType.GAUGE,
                value=cpu_percent,
                timestamp=timestamp,
                tags={"resource": "cpu"}
            ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric(Metric(
                name="system.memory.usage",
                type=MetricType.GAUGE,
                value=memory.percent,
                timestamp=timestamp,
                tags={"resource": "memory"}
            ))
            
            self.record_metric(Metric(
                name="system.memory.available",
                type=MetricType.GAUGE,
                value=memory.available / (1024 ** 3),  # GB
                timestamp=timestamp,
                tags={"resource": "memory", "unit": "GB"}
            ))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_metric(Metric(
                name="system.disk.usage",
                type=MetricType.GAUGE,
                value=disk.percent,
                timestamp=timestamp,
                tags={"resource": "disk"}
            ))
            
            # Process metrics
            process = psutil.Process()
            self.record_metric(Metric(
                name="process.cpu.usage",
                type=MetricType.GAUGE,
                value=process.cpu_percent(),
                timestamp=timestamp,
                tags={"process": "cwmai"}
            ))
            
            self.record_metric(Metric(
                name="process.memory.rss",
                type=MetricType.GAUGE,
                value=process.memory_info().rss / (1024 ** 2),  # MB
                timestamp=timestamp,
                tags={"process": "cwmai", "unit": "MB"}
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to collect resource metrics: {e}")
            
    async def _collect_performance_metrics(self, timestamp: datetime):
        """Collect performance metrics from various components."""
        # Get state for analysis
        state = self.state_manager.get_state()
        
        # Task execution metrics
        if 'task_history' in state:
            recent_tasks = state['task_history'][-100:]  # Last 100 tasks
            
            if recent_tasks:
                success_count = sum(1 for t in recent_tasks if t.get('status') == 'completed')
                self.record_metric(Metric(
                    name="tasks.success.rate",
                    type=MetricType.GAUGE,
                    value=success_count / len(recent_tasks) * 100,
                    timestamp=timestamp,
                    tags={"component": "task_manager"}
                ))
                
                # Task execution time
                exec_times = [t.get('execution_time', 0) for t in recent_tasks if t.get('execution_time')]
                if exec_times:
                    self.record_metric(Metric(
                        name="tasks.execution.time",
                        type=MetricType.HISTOGRAM,
                        value=statistics.mean(exec_times),
                        timestamp=timestamp,
                        tags={"component": "task_manager", "stat": "mean"}
                    ))
                    
        # AI response times
        if hasattr(self.ai_client, 'get_metrics'):
            ai_metrics = self.ai_client.get_metrics()
            for provider, metrics in ai_metrics.items():
                if metrics.get('avg_response_time'):
                    self.record_metric(Metric(
                        name="ai.response.time",
                        type=MetricType.GAUGE,
                        value=metrics['avg_response_time'],
                        timestamp=timestamp,
                        tags={"provider": provider}
                    ))
                    
        # Tool performance
        if 'tool_usage' in state:
            for tool_name, usage in state['tool_usage'].items():
                if usage.get('calls', 0) > 0:
                    success_rate = usage.get('successes', 0) / usage['calls'] * 100
                    self.record_metric(Metric(
                        name="tool.success.rate",
                        type=MetricType.GAUGE,
                        value=success_rate,
                        timestamp=timestamp,
                        tags={"tool": tool_name}
                    ))
                    
    async def _collect_error_metrics(self, timestamp: datetime):
        """Collect error metrics."""
        # Parse recent logs for errors
        log_files = list(Path('.').glob('*.log'))
        
        error_counts = defaultdict(int)
        
        for log_file in log_files:
            try:
                # Read last 1000 lines
                lines = log_file.read_text().splitlines()[-1000:]
                
                for line in lines:
                    if 'ERROR' in line:
                        # Extract error type
                        if 'AttributeError' in line:
                            error_counts['AttributeError'] += 1
                        elif 'KeyError' in line:
                            error_counts['KeyError'] += 1
                        elif 'TypeError' in line:
                            error_counts['TypeError'] += 1
                        elif 'ValueError' in line:
                            error_counts['ValueError'] += 1
                        else:
                            error_counts['Other'] += 1
                            
            except Exception:
                pass
                
        # Record error metrics
        for error_type, count in error_counts.items():
            self.record_metric(Metric(
                name="errors.count",
                type=MetricType.COUNTER,
                value=count,
                timestamp=timestamp,
                tags={"error_type": error_type}
            ))
            
    def record_metric(self, metric: Metric):
        """Record a metric."""
        key = f"{metric.name}:{json.dumps(metric.tags, sort_keys=True)}"
        self.metrics[key].append(metric)
        
        # Update aggregates
        self._update_aggregates(key, metric)
        
    def _update_aggregates(self, key: str, metric: Metric):
        """Update metric aggregates."""
        if key not in self.metric_aggregates:
            self.metric_aggregates[key] = {
                'min': metric.value,
                'max': metric.value,
                'sum': metric.value,
                'count': 1,
                'last': metric.value,
                'last_timestamp': metric.timestamp
            }
        else:
            agg = self.metric_aggregates[key]
            agg['min'] = min(agg['min'], metric.value)
            agg['max'] = max(agg['max'], metric.value)
            agg['sum'] += metric.value
            agg['count'] += 1
            agg['last'] = metric.value
            agg['last_timestamp'] = metric.timestamp
            
    async def _analyze_metrics(self):
        """Analyze metrics for patterns and anomalies."""
        print("Analyzing metrics")
        
        # Update baselines
        await self._update_baselines()
        
        # Detect anomalies
        anomalies = await self._detect_anomalies()
        
        # Detect patterns
        patterns = await self._detect_patterns()
        
        # Generate alerts
        await self._generate_alerts(anomalies, patterns)
        
        # Clean up old metrics
        self._cleanup_old_metrics()
        
    async def _update_baselines(self):
        """Update baselines for anomaly detection."""
        window_start = datetime.now(timezone.utc) - timedelta(seconds=self.config['baseline_window'])
        
        for key, metric_list in self.metrics.items():
            recent_metrics = [m for m in metric_list if m.timestamp > window_start]
            
            if len(recent_metrics) >= self.config['min_data_points_for_analysis']:
                values = [m.value for m in recent_metrics]
                
                self.baselines[key] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'median': statistics.median(values),
                    'p95': np.percentile(values, 95) if values else 0,
                    'updated': datetime.now(timezone.utc)
                }
                
    async def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics."""
        anomalies = []
        
        for key, metric_list in self.metrics.items():
            if key not in self.baselines:
                continue
                
            baseline = self.baselines[key]
            if baseline['std'] == 0:
                continue
                
            # Check recent values
            recent_metrics = list(metric_list)[-10:]  # Last 10 values
            
            for metric in recent_metrics:
                # Z-score calculation
                z_score = abs((metric.value - baseline['mean']) / baseline['std'])
                
                if z_score > self.anomaly_threshold:
                    anomalies.append({
                        'metric': metric.name,
                        'tags': metric.tags,
                        'value': metric.value,
                        'baseline_mean': baseline['mean'],
                        'z_score': z_score,
                        'timestamp': metric.timestamp,
                        'severity': 'high' if z_score > 4 else 'medium'
                    })
                    
        return anomalies
        
    async def _detect_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Detect patterns in metrics."""
        detected_patterns = defaultdict(list)
        
        # Performance degradation pattern
        for key, metric_list in self.metrics.items():
            if 'response.time' in key or 'execution.time' in key:
                recent = list(metric_list)[-20:]
                if len(recent) >= 10:
                    # Check for increasing trend
                    values = [m.value for m in recent]
                    if self._is_increasing_trend(values):
                        detected_patterns['performance_degradation'].append({
                            'metric': recent[0].name,
                            'trend': 'increasing',
                            'change': (values[-1] - values[0]) / values[0] * 100,
                            'duration': (recent[-1].timestamp - recent[0].timestamp).total_seconds()
                        })
                        
        # Error spike pattern
        error_metrics = [k for k in self.metrics.keys() if 'error' in k]
        for key in error_metrics:
            recent = list(self.metrics[key])[-10:]
            if recent:
                current_rate = sum(m.value for m in recent[-5:])
                previous_rate = sum(m.value for m in recent[:5])
                
                if previous_rate > 0 and current_rate / previous_rate > 2:
                    detected_patterns['error_spike'].append({
                        'metric': recent[0].name,
                        'tags': recent[0].tags,
                        'increase': current_rate / previous_rate,
                        'current_rate': current_rate
                    })
                    
        # Success rate drop pattern
        for key, metric_list in self.metrics.items():
            if 'success.rate' in key:
                recent = list(metric_list)[-20:]
                if len(recent) >= 10:
                    recent_avg = statistics.mean([m.value for m in recent[-5:]])
                    historical_avg = statistics.mean([m.value for m in recent[:10]])
                    
                    if historical_avg > 0 and recent_avg < historical_avg * 0.8:
                        detected_patterns['success_rate_drop'].append({
                            'metric': recent[0].name,
                            'tags': recent[0].tags,
                            'recent_avg': recent_avg,
                            'historical_avg': historical_avg,
                            'drop_percent': (historical_avg - recent_avg) / historical_avg * 100
                        })
                        
        return dict(detected_patterns)
        
    def _is_increasing_trend(self, values: List[float], threshold: float = 0.7) -> bool:
        """Check if values show an increasing trend."""
        if len(values) < 3:
            return False
            
        # Simple linear regression
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        
        # Check if slope is positive and significant
        mean_value = statistics.mean(values)
        if mean_value > 0:
            normalized_slope = slope / mean_value
            return normalized_slope > threshold / len(values)
            
        return False
        
    async def _generate_alerts(self, anomalies: List[Dict[str, Any]], 
                             patterns: Dict[str, List[Dict[str, Any]]]):
        """Generate alerts from anomalies and patterns."""
        
        # Anomaly alerts
        for anomaly in anomalies:
            alert_key = f"anomaly:{anomaly['metric']}:{json.dumps(anomaly['tags'])}"
            
            if not self._is_alert_active(alert_key):
                alert = Alert(
                    severity='warning' if anomaly['severity'] == 'medium' else 'error',
                    category='anomaly',
                    message=f"Anomaly detected in {anomaly['metric']}",
                    details=anomaly,
                    timestamp=datetime.now(timezone.utc)
                )
                
                self._create_alert(alert, alert_key)
                
        # Pattern alerts
        for pattern_type, occurrences in patterns.items():
            for occurrence in occurrences:
                alert_key = f"pattern:{pattern_type}:{occurrence.get('metric', 'system')}"
                
                if not self._is_alert_active(alert_key):
                    severity = 'error' if pattern_type == 'error_spike' else 'warning'
                    
                    alert = Alert(
                        severity=severity,
                        category='performance' if 'performance' in pattern_type else 'error',
                        message=f"Pattern detected: {pattern_type.replace('_', ' ').title()}",
                        details=occurrence,
                        timestamp=datetime.now(timezone.utc)
                    )
                    
                    self._create_alert(alert, alert_key)
                    
    def _is_alert_active(self, alert_key: str) -> bool:
        """Check if an alert is already active or in cooldown."""
        for alert in self.active_alerts:
            if alert.get('key') == alert_key:
                return True
                
        # Check cooldown
        for alert in self.alert_history:
            if (alert.get('key') == alert_key and 
                (datetime.now(timezone.utc) - alert['timestamp']).total_seconds() < self.config['alert_cooldown']):
                return True
                
        return False
        
    def _create_alert(self, alert: Alert, key: str):
        """Create a new alert."""
        alert_data = {
            'alert': alert,
            'key': key,
            'timestamp': alert.timestamp
        }
        
        self.active_alerts.append(alert_data)
        self.logger.warning(f"Alert created: {alert.message}")
        
        # Trigger alert handlers
        asyncio.create_task(self._handle_alert(alert))
        
    async def _handle_alert(self, alert: Alert):
        """Handle an alert with automated responses."""
        if alert.category == 'performance' and alert.severity in ['error', 'critical']:
            # Try to auto-remediate performance issues
            await self._auto_remediate_performance(alert)
        elif alert.category == 'error' and 'spike' in alert.message:
            # Investigate error spikes
            await self._investigate_error_spike(alert)
            
    async def _auto_remediate_performance(self, alert: Alert):
        """Attempt to auto-remediate performance issues."""
        print(f"Attempting auto-remediation for: {alert.message}")
        
        # Clear caches if memory pressure
        if 'memory' in str(alert.details):
            try:
                import gc
                gc.collect()
                print("Performed garbage collection")
            except:
                pass
                
        # Reduce concurrency if CPU pressure
        if 'cpu' in str(alert.details):
            # Could reduce worker threads, adjust rate limits, etc.
            pass
            
    async def _investigate_error_spike(self, alert: Alert):
        """Investigate error spike causes."""
        error_type = alert.details.get('tags', {}).get('error_type', 'Unknown')
        
        # Generate investigation report
        prompt = f"""Investigate this error spike:
Error type: {error_type}
Details: {json.dumps(alert.details, indent=2)}

Analyze potential causes and suggest fixes.
"""
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        if response.get('content'):
            # Store investigation results
            investigation = {
                'alert': alert.message,
                'error_type': error_type,
                'investigation': response['content'],
                'timestamp': datetime.now(timezone.utc)
            }
            
            # Could trigger automatic fixes based on investigation
            
    async def _generate_insights(self):
        """Generate actionable insights from monitoring data."""
        print("Generating insights")
        
        insights = []
        
        # Performance optimization insights
        insights.extend(await self._generate_performance_insights())
        
        # Resource optimization insights
        insights.extend(await self._generate_resource_insights())
        
        # Pattern-based insights
        insights.extend(await self._generate_pattern_insights())
        
        # Predictive insights
        insights.extend(await self._generate_predictive_insights())
        
        # Store insights
        for insight in insights:
            self.insights.append(insight)
            print(f"Generated insight: {insight.title}")
            
    async def _generate_performance_insights(self) -> List[Insight]:
        """Generate performance optimization insights."""
        insights = []
        
        # Analyze slow operations
        slow_operations = []
        
        for key, metric_list in self.metrics.items():
            if 'time' in key and len(metric_list) > 20:
                recent = list(metric_list)[-20:]
                avg_time = statistics.mean([m.value for m in recent])
                
                if avg_time > 5.0:  # Operations taking >5 seconds
                    slow_operations.append({
                        'operation': recent[0].name,
                        'avg_time': avg_time,
                        'tags': recent[0].tags
                    })
                    
        if slow_operations:
            # Generate optimization suggestions
            top_slow = sorted(slow_operations, key=lambda x: x['avg_time'], reverse=True)[:3]
            
            for op in top_slow:
                insights.append(Insight(
                    category='optimization',
                    title=f"Optimize {op['operation']}",
                    description=f"Operation averaging {op['avg_time']:.2f}s response time",
                    impact="Reduce response time by 50%+",
                    confidence=0.8,
                    recommendations=[
                        "Add caching for frequently accessed data",
                        "Optimize database queries",
                        "Consider async/parallel processing",
                        "Profile code to find bottlenecks"
                    ],
                    supporting_data=op,
                    timestamp=datetime.now(timezone.utc)
                ))
                
        return insights
        
    async def _generate_resource_insights(self) -> List[Insight]:
        """Generate resource optimization insights."""
        insights = []
        
        # Check resource usage trends
        cpu_metrics = self.metrics.get("system.cpu.usage:{}]", [])
        memory_metrics = self.metrics.get("system.memory.usage:{}", [])
        
        if len(cpu_metrics) > 100:
            recent_cpu = [m.value for m in list(cpu_metrics)[-100:]]
            avg_cpu = statistics.mean(recent_cpu)
            
            if avg_cpu > 70:
                insights.append(Insight(
                    category='optimization',
                    title='High CPU Usage Detected',
                    description=f'Average CPU usage is {avg_cpu:.1f}%',
                    impact='Improve system responsiveness and reduce costs',
                    confidence=0.9,
                    recommendations=[
                        'Profile CPU-intensive operations',
                        'Implement request queuing and rate limiting',
                        'Consider horizontal scaling',
                        'Optimize algorithms and data structures'
                    ],
                    supporting_data={'avg_cpu': avg_cpu, 'samples': len(recent_cpu)},
                    timestamp=datetime.now(timezone.utc)
                ))
                
        return insights
        
    async def _generate_pattern_insights(self) -> List[Insight]:
        """Generate insights from detected patterns."""
        insights = []
        
        # Analyze success patterns
        high_success_tools = []
        low_success_tools = []
        
        for key, metric_list in self.metrics.items():
            if 'success.rate' in key and len(metric_list) > 10:
                recent = list(metric_list)[-10:]
                avg_rate = statistics.mean([m.value for m in recent])
                tool_name = recent[0].tags.get('tool', 'unknown')
                
                if avg_rate > 95:
                    high_success_tools.append((tool_name, avg_rate))
                elif avg_rate < 70:
                    low_success_tools.append((tool_name, avg_rate))
                    
        # Suggest replicating success patterns
        if high_success_tools:
            insights.append(Insight(
                category='pattern',
                title='Replicate Successful Tool Patterns',
                description=f'Tools with >95% success rate: {", ".join([t[0] for t in high_success_tools[:3]])}',
                impact='Improve overall system reliability',
                confidence=0.85,
                recommendations=[
                    f'Analyze implementation of {high_success_tools[0][0]}',
                    'Apply similar error handling to other tools',
                    'Create tool templates based on successful patterns',
                    'Document best practices from high-performing tools'
                ],
                supporting_data={'high_success_tools': high_success_tools},
                timestamp=datetime.now(timezone.utc)
            ))
            
        # Suggest improving low success tools
        if low_success_tools:
            insights.append(Insight(
                category='optimization',
                title='Improve Low-Success Tools',
                description=f'Tools with <70% success rate need attention',
                impact='Increase system reliability by 20%+',
                confidence=0.9,
                recommendations=[
                    f'Add retry logic to {low_success_tools[0][0]}',
                    'Implement better error handling',
                    'Add input validation',
                    'Consider redesigning tool architecture'
                ],
                supporting_data={'low_success_tools': low_success_tools},
                timestamp=datetime.now(timezone.utc)
            ))
            
        return insights
        
    async def _generate_predictive_insights(self) -> List[Insight]:
        """Generate predictive insights."""
        insights = []
        
        # Predict resource exhaustion
        memory_metrics = list(self.metrics.get("system.memory.usage:{}", []))[-100:]
        
        if len(memory_metrics) > 50:
            memory_values = [m.value for m in memory_metrics]
            memory_times = [(m.timestamp - memory_metrics[0].timestamp).total_seconds() / 3600 
                           for m in memory_metrics]
            
            # Simple linear regression for trend
            if len(memory_values) > 10:
                slope = np.polyfit(memory_times, memory_values, 1)[0]
                
                if slope > 1:  # Growing >1% per hour
                    current = memory_values[-1]
                    hours_to_full = (95 - current) / slope if slope > 0 else float('inf')
                    
                    if hours_to_full < 24:
                        insights.append(Insight(
                            category='prediction',
                            title='Memory Exhaustion Risk',
                            description=f'Memory usage growing at {slope:.1f}% per hour',
                            impact='Prevent system crash',
                            confidence=0.7,
                            recommendations=[
                                'Investigate memory leaks',
                                'Implement memory cleanup routines',
                                'Add memory monitoring alerts',
                                'Plan for system restart if needed'
                            ],
                            supporting_data={
                                'current_usage': current,
                                'growth_rate': slope,
                                'hours_to_full': hours_to_full
                            },
                            timestamp=datetime.now(timezone.utc)
                        ))
                        
        return insights
        
    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory growth."""
        # Metrics are already limited by deque maxlen
        # Additional cleanup can be added here if needed
        pass
        
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        # Calculate health score
        health_score = 100.0
        
        # Deduct for active alerts
        for alert in self.active_alerts:
            if alert['alert'].severity == 'critical':
                health_score -= 20
            elif alert['alert'].severity == 'error':
                health_score -= 10
            elif alert['alert'].severity == 'warning':
                health_score -= 5
                
        # Deduct for poor metrics
        for key, agg in self.metric_aggregates.items():
            if 'success.rate' in key and agg['last'] < 80:
                health_score -= 5
            elif 'error' in key and agg['last'] > 10:
                health_score -= 3
                
        health_score = max(0, health_score)
        
        return {
            'health_score': health_score,
            'status': self._get_health_status(health_score),
            'active_alerts': len(self.active_alerts),
            'metrics_tracked': len(self.metrics),
            'insights_available': len(self.insights),
            'last_analysis': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'uptime': self._calculate_uptime()
        }
        
    def _get_health_status(self, score: float) -> str:
        """Get health status from score."""
        if score >= 90:
            return 'excellent'
        elif score >= 75:
            return 'good'
        elif score >= 50:
            return 'fair'
        elif score >= 25:
            return 'poor'
        else:
            return 'critical'
            
    def _calculate_uptime(self) -> Dict[str, Any]:
        """Calculate system uptime."""
        if hasattr(self, 'start_time'):
            uptime = datetime.now(timezone.utc) - self.start_time
            return {
                'days': uptime.days,
                'hours': uptime.seconds // 3600,
                'minutes': (uptime.seconds % 3600) // 60
            }
        return {'days': 0, 'hours': 0, 'minutes': 0}
        
    async def get_insights(self, category: Optional[str] = None) -> List[Insight]:
        """Get available insights."""
        if category:
            return [i for i in self.insights if i.category == category]
        return list(self.insights)
        
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        
        for key, agg in self.metric_aggregates.items():
            metric_name = key.split(':')[0]
            
            if metric_name not in summary:
                summary[metric_name] = []
                
            summary[metric_name].append({
                'tags': json.loads(key.split(':', 1)[1]) if ':' in key else {},
                'last_value': agg['last'],
                'min': agg['min'],
                'max': agg['max'],
                'avg': agg['sum'] / agg['count'] if agg['count'] > 0 else 0,
                'count': agg['count']
            })
            
        return summary


# Global instance
monitoring_system = None

def get_monitoring_system(state_manager: StateManager = None) -> IntelligentMonitoringSystem:
    """Get or create the global monitoring system."""
    global monitoring_system
    if monitoring_system is None:
        monitoring_system = IntelligentMonitoringSystem(
            state_manager or StateManager()
        )
    return monitoring_system