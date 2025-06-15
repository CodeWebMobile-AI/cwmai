"""
Redis Analytics Engine

Real-time analytics and insights for Redis operations, performance patterns,
and usage optimization with machine learning-based recommendations.
"""

import asyncio
import json
import time
import statistics
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import defaultdict, deque
import logging
from .redis_client import RedisClient


class AnalyticsType(Enum):
    """Types of analytics."""
    PERFORMANCE = "performance"
    USAGE = "usage"
    PATTERNS = "patterns"
    ANOMALIES = "anomalies"
    PREDICTIONS = "predictions"


class AnomalyType(Enum):
    """Types of anomalies detected."""
    SPIKE = "spike"
    DROP = "drop"
    TREND = "trend"
    OUTLIER = "outlier"
    PATTERN_BREAK = "pattern_break"


@dataclass
class AnalyticsInsight:
    """Analytics insight with recommendations."""
    id: str
    type: AnalyticsType
    title: str
    description: str
    severity: str  # low, medium, high, critical
    confidence: float  # 0.0 to 1.0
    recommendations: List[str]
    metrics: Dict[str, Any]
    timestamp: datetime
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'type': self.type.value,
            'title': self.title,
            'description': self.description,
            'severity': self.severity,
            'confidence': self.confidence,
            'recommendations': self.recommendations,
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }


@dataclass
class PerformancePattern:
    """Performance pattern analysis."""
    metric: str
    pattern_type: str  # daily, weekly, trending, cyclical
    description: str
    peak_hours: List[int]
    low_hours: List[int]
    average_value: float
    peak_value: float
    low_value: float
    trend_direction: str  # increasing, decreasing, stable
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric': self.metric,
            'pattern_type': self.pattern_type,
            'description': self.description,
            'peak_hours': self.peak_hours,
            'low_hours': self.low_hours,
            'average_value': self.average_value,
            'peak_value': self.peak_value,
            'low_value': self.low_value,
            'trend_direction': self.trend_direction,
            'confidence': self.confidence
        }


@dataclass
class AnomalyDetection:
    """Anomaly detection result."""
    metric: str
    anomaly_type: AnomalyType
    value: float
    expected_value: float
    deviation: float
    severity: str
    timestamp: datetime
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric': self.metric,
            'anomaly_type': self.anomaly_type.value,
            'value': self.value,
            'expected_value': self.expected_value,
            'deviation': self.deviation,
            'severity': self.severity,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context
        }


class RedisAnalytics:
    """Advanced Redis analytics and insights engine."""
    
    def __init__(self, redis_client: RedisClient, namespace: str = "analytics"):
        """Initialize Redis analytics engine.
        
        Args:
            redis_client: Redis client instance
            namespace: Analytics namespace for data storage
        """
        self.redis = redis_client
        self.namespace = namespace
        self.logger = logging.getLogger(__name__)
        
        # Analytics configuration
        self.analysis_interval = 300  # 5 minutes
        self.insight_retention = 86400 * 7  # 7 days
        self.anomaly_window = 3600  # 1 hour
        self.pattern_window = 86400 * 7  # 7 days
        
        # Statistical tracking
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self.detected_patterns: Dict[str, PerformancePattern] = {}
        
        # Machine learning features
        self.enable_anomaly_detection = True
        self.enable_pattern_recognition = True
        self.enable_predictive_analysis = True
        self.anomaly_threshold = 2.5  # Standard deviations
        
        # Background tasks
        self._analytics_task: Optional[asyncio.Task] = None
        self._pattern_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Initialize insights storage
        self.insights: Dict[str, AnalyticsInsight] = {}
    
    def _insights_key(self) -> str:
        """Create insights storage key."""
        return f"{self.namespace}:insights"
    
    def _patterns_key(self) -> str:
        """Create patterns storage key."""
        return f"{self.namespace}:patterns"
    
    def _anomalies_key(self) -> str:
        """Create anomalies storage key."""
        return f"{self.namespace}:anomalies"
    
    def _baselines_key(self) -> str:
        """Create baselines storage key."""
        return f"{self.namespace}:baselines"
    
    async def start(self):
        """Start analytics engine."""
        await self._load_baseline_stats()
        
        if not self._analytics_task:
            self._analytics_task = asyncio.create_task(self._analytics_loop())
            self._pattern_task = asyncio.create_task(self._pattern_analysis_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.logger.info("Redis analytics engine started")
    
    async def stop(self):
        """Stop analytics engine."""
        self._shutdown = True
        
        # Stop background tasks
        for task in [self._analytics_task, self._pattern_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Save baseline stats
        await self._save_baseline_stats()
        
        self.logger.info("Redis analytics engine stopped")
    
    async def _analytics_loop(self):
        """Main analytics processing loop."""
        while not self._shutdown:
            try:
                # Collect current metrics
                await self._collect_metrics()
                
                # Run anomaly detection
                if self.enable_anomaly_detection:
                    await self._detect_anomalies()
                
                # Generate insights
                await self._generate_insights()
                
                await asyncio.sleep(self.analysis_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in analytics loop: {e}")
                await asyncio.sleep(self.analysis_interval)
    
    async def _pattern_analysis_loop(self):
        """Pattern recognition processing loop."""
        while not self._shutdown:
            try:
                if self.enable_pattern_recognition:
                    await self._analyze_patterns()
                
                await asyncio.sleep(3600)  # Run every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in pattern analysis loop: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_loop(self):
        """Cleanup old analytics data."""
        while not self._shutdown:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Run every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)
    
    async def _collect_metrics(self):
        """Collect current metrics for analysis."""
        try:
            # Get Redis info
            info = await self.redis.info()
            current_time = time.time()
            
            # Extract key metrics
            metrics = {
                'memory_usage': info.get('used_memory', 0),
                'connected_clients': info.get('connected_clients', 0),
                'commands_per_sec': info.get('instantaneous_ops_per_sec', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'expired_keys': info.get('expired_keys', 0),
                'evicted_keys': info.get('evicted_keys', 0)
            }
            
            # Calculate derived metrics
            total_requests = metrics['keyspace_hits'] + metrics['keyspace_misses']
            if total_requests > 0:
                metrics['hit_rate'] = metrics['keyspace_hits'] / total_requests
            else:
                metrics['hit_rate'] = 1.0
            
            # Add to buffers with timestamp
            for metric_name, value in metrics.items():
                self.metrics_buffer[metric_name].append((current_time, value))
            
            # Update baseline statistics
            await self._update_baseline_stats(metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
    
    async def _update_baseline_stats(self, metrics: Dict[str, float]):
        """Update baseline statistics for metrics."""
        for metric_name, value in metrics.items():
            if metric_name not in self.baseline_stats:
                self.baseline_stats[metric_name] = {
                    'count': 0,
                    'sum': 0.0,
                    'sum_squares': 0.0,
                    'min': value,
                    'max': value,
                    'mean': value,
                    'std': 0.0
                }
            
            stats = self.baseline_stats[metric_name]
            stats['count'] += 1
            stats['sum'] += value
            stats['sum_squares'] += value * value
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)
            
            # Update running mean and standard deviation
            if stats['count'] > 1:
                stats['mean'] = stats['sum'] / stats['count']
                variance = (stats['sum_squares'] / stats['count']) - (stats['mean'] ** 2)
                stats['std'] = max(0, variance) ** 0.5
    
    async def _detect_anomalies(self):
        """Detect anomalies in current metrics."""
        try:
            current_time = datetime.now(timezone.utc)
            anomalies = []
            
            for metric_name, data_points in self.metrics_buffer.items():
                if len(data_points) < 10:  # Need sufficient data
                    continue
                
                baseline = self.baseline_stats.get(metric_name)
                if not baseline or baseline['std'] == 0:
                    continue
                
                # Get recent values
                recent_values = [value for _, value in list(data_points)[-10:]]
                current_value = recent_values[-1]
                
                # Calculate z-score
                z_score = abs(current_value - baseline['mean']) / baseline['std']
                
                if z_score > self.anomaly_threshold:
                    # Determine anomaly type
                    if current_value > baseline['mean']:
                        anomaly_type = AnomalyType.SPIKE
                        severity = "high" if z_score > 4 else "medium"
                    else:
                        anomaly_type = AnomalyType.DROP
                        severity = "high" if z_score > 4 else "medium"
                    
                    anomaly = AnomalyDetection(
                        metric=metric_name,
                        anomaly_type=anomaly_type,
                        value=current_value,
                        expected_value=baseline['mean'],
                        deviation=z_score,
                        severity=severity,
                        timestamp=current_time,
                        context={
                            'baseline_mean': baseline['mean'],
                            'baseline_std': baseline['std'],
                            'recent_trend': self._calculate_trend(recent_values)
                        }
                    )
                    
                    anomalies.append(anomaly)
            
            # Store anomalies
            if anomalies:
                await self._store_anomalies(anomalies)
                await self._generate_anomaly_insights(anomalies)
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 3:
            return "insufficient_data"
        
        # Simple linear regression slope
        x = list(range(len(values)))
        n = len(values)
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    async def _analyze_patterns(self):
        """Analyze historical patterns in metrics."""
        try:
            for metric_name, data_points in self.metrics_buffer.items():
                if len(data_points) < 100:  # Need sufficient data
                    continue
                
                # Extract values and timestamps
                timestamps = [ts for ts, _ in data_points]
                values = [value for _, value in data_points]
                
                # Analyze hourly patterns
                hourly_patterns = self._analyze_hourly_patterns(timestamps, values)
                
                # Detect trend
                trend = self._calculate_trend(values[-50:])  # Recent trend
                
                # Create pattern object
                if hourly_patterns:
                    pattern = PerformancePattern(
                        metric=metric_name,
                        pattern_type="daily",
                        description=f"Daily usage pattern for {metric_name}",
                        peak_hours=hourly_patterns['peak_hours'],
                        low_hours=hourly_patterns['low_hours'],
                        average_value=statistics.mean(values),
                        peak_value=max(values),
                        low_value=min(values),
                        trend_direction=trend,
                        confidence=hourly_patterns['confidence']
                    )
                    
                    self.detected_patterns[metric_name] = pattern
                    await self._store_pattern(pattern)
            
        except Exception as e:
            self.logger.error(f"Error analyzing patterns: {e}")
    
    def _analyze_hourly_patterns(self, timestamps: List[float], values: List[float]) -> Optional[Dict[str, Any]]:
        """Analyze hourly usage patterns."""
        try:
            # Group by hour of day
            hourly_data = defaultdict(list)
            
            for ts, value in zip(timestamps, values):
                hour = datetime.fromtimestamp(ts).hour
                hourly_data[hour].append(value)
            
            # Calculate average for each hour
            hourly_averages = {}
            for hour in range(24):
                if hour in hourly_data:
                    hourly_averages[hour] = statistics.mean(hourly_data[hour])
                else:
                    hourly_averages[hour] = 0
            
            # Find peak and low hours
            if not hourly_averages:
                return None
            
            avg_values = list(hourly_averages.values())
            overall_avg = statistics.mean(avg_values)
            std_dev = statistics.stdev(avg_values) if len(avg_values) > 1 else 0
            
            if std_dev == 0:
                return None
            
            # Hours significantly above/below average
            peak_hours = [hour for hour, avg in hourly_averages.items() 
                         if avg > overall_avg + std_dev]
            low_hours = [hour for hour, avg in hourly_averages.items() 
                        if avg < overall_avg - std_dev]
            
            # Calculate confidence based on data consistency
            confidence = min(1.0, len(set(hourly_data.keys())) / 24)
            
            return {
                'peak_hours': peak_hours,
                'low_hours': low_hours,
                'hourly_averages': hourly_averages,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing hourly patterns: {e}")
            return None
    
    async def _generate_insights(self):
        """Generate actionable insights from analytics."""
        try:
            insights = []
            
            # Performance insights
            insights.extend(await self._generate_performance_insights())
            
            # Usage optimization insights
            insights.extend(await self._generate_usage_insights())
            
            # Pattern-based insights
            insights.extend(await self._generate_pattern_insights())
            
            # Store new insights
            for insight in insights:
                self.insights[insight.id] = insight
                await self._store_insight(insight)
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
    
    async def _generate_performance_insights(self) -> List[AnalyticsInsight]:
        """Generate performance-related insights."""
        insights = []
        
        try:
            # Memory usage insights
            if 'memory_usage' in self.baseline_stats:
                memory_stats = self.baseline_stats['memory_usage']
                recent_memory = [value for _, value in list(self.metrics_buffer['memory_usage'])[-10:]]
                
                if recent_memory:
                    current_memory = recent_memory[-1]
                    trend = self._calculate_trend(recent_memory)
                    
                    if trend == "increasing" and current_memory > memory_stats['mean'] * 1.2:
                        insight = AnalyticsInsight(
                            id=f"memory_growth_{int(time.time())}",
                            type=AnalyticsType.PERFORMANCE,
                            title="Memory Usage Growing",
                            description=f"Memory usage is trending upward and is {((current_memory / memory_stats['mean'] - 1) * 100):.1f}% above baseline",
                            severity="medium",
                            confidence=0.8,
                            recommendations=[
                                "Review memory configuration and maxmemory policy",
                                "Check for memory leaks in client applications",
                                "Consider implementing data expiration policies",
                                "Monitor for large key values"
                            ],
                            metrics={
                                'current_memory': current_memory,
                                'baseline_memory': memory_stats['mean'],
                                'trend': trend
                            },
                            timestamp=datetime.now(timezone.utc),
                            expires_at=datetime.now(timezone.utc) + timedelta(hours=6)
                        )
                        insights.append(insight)
            
            # Connection insights
            if 'connected_clients' in self.baseline_stats:
                client_stats = self.baseline_stats['connected_clients']
                recent_clients = [value for _, value in list(self.metrics_buffer['connected_clients'])[-10:]]
                
                if recent_clients:
                    current_clients = recent_clients[-1]
                    
                    if current_clients > client_stats['mean'] * 1.5:
                        insight = AnalyticsInsight(
                            id=f"high_connections_{int(time.time())}",
                            type=AnalyticsType.PERFORMANCE,
                            title="High Client Connection Count",
                            description=f"Current client connections ({current_clients}) are significantly above normal ({client_stats['mean']:.0f})",
                            severity="medium",
                            confidence=0.9,
                            recommendations=[
                                "Review connection pooling in client applications",
                                "Check for connection leaks",
                                "Consider adjusting timeout settings",
                                "Monitor for denial of service attempts"
                            ],
                            metrics={
                                'current_connections': current_clients,
                                'baseline_connections': client_stats['mean']
                            },
                            timestamp=datetime.now(timezone.utc),
                            expires_at=datetime.now(timezone.utc) + timedelta(hours=4)
                        )
                        insights.append(insight)
        
        except Exception as e:
            self.logger.error(f"Error generating performance insights: {e}")
        
        return insights
    
    async def _generate_usage_insights(self) -> List[AnalyticsInsight]:
        """Generate usage optimization insights."""
        insights = []
        
        try:
            # Hit rate insights
            if 'hit_rate' in self.baseline_stats:
                hit_rate_data = [value for _, value in list(self.metrics_buffer['hit_rate'])[-50:]]
                
                if hit_rate_data:
                    avg_hit_rate = statistics.mean(hit_rate_data)
                    
                    if avg_hit_rate < 0.8:  # Less than 80% hit rate
                        insight = AnalyticsInsight(
                            id=f"low_hit_rate_{int(time.time())}",
                            type=AnalyticsType.USAGE,
                            title="Low Cache Hit Rate",
                            description=f"Cache hit rate is {avg_hit_rate:.1%}, indicating potential inefficiencies",
                            severity="medium" if avg_hit_rate < 0.6 else "low",
                            confidence=0.9,
                            recommendations=[
                                "Review caching strategies in applications",
                                "Increase TTL for frequently accessed keys",
                                "Implement cache warming for critical data",
                                "Analyze query patterns for optimization opportunities"
                            ],
                            metrics={
                                'current_hit_rate': avg_hit_rate,
                                'recommended_hit_rate': 0.9
                            },
                            timestamp=datetime.now(timezone.utc),
                            expires_at=datetime.now(timezone.utc) + timedelta(hours=8)
                        )
                        insights.append(insight)
            
            # Eviction insights
            if 'evicted_keys' in self.metrics_buffer:
                evicted_data = [value for _, value in list(self.metrics_buffer['evicted_keys'])[-20:]]
                
                if len(evicted_data) >= 2:
                    recent_evictions = evicted_data[-1] - evicted_data[0]
                    
                    if recent_evictions > 100:  # Significant evictions
                        insight = AnalyticsInsight(
                            id=f"high_evictions_{int(time.time())}",
                            type=AnalyticsType.USAGE,
                            title="High Key Eviction Rate",
                            description=f"Redis has evicted {recent_evictions} keys recently, indicating memory pressure",
                            severity="high",
                            confidence=0.95,
                            recommendations=[
                                "Increase Redis memory allocation",
                                "Review maxmemory-policy configuration",
                                "Implement more aggressive TTL policies",
                                "Identify and remove unused keys",
                                "Consider data partitioning strategies"
                            ],
                            metrics={
                                'recent_evictions': recent_evictions,
                                'eviction_threshold': 100
                            },
                            timestamp=datetime.now(timezone.utc),
                            expires_at=datetime.now(timezone.utc) + timedelta(hours=4)
                        )
                        insights.append(insight)
        
        except Exception as e:
            self.logger.error(f"Error generating usage insights: {e}")
        
        return insights
    
    async def _generate_pattern_insights(self) -> List[AnalyticsInsight]:
        """Generate pattern-based insights."""
        insights = []
        
        try:
            for metric_name, pattern in self.detected_patterns.items():
                if pattern.confidence > 0.7:  # High confidence patterns
                    
                    # Peak usage recommendations
                    if pattern.peak_hours and len(pattern.peak_hours) <= 4:
                        peak_hours_str = ", ".join(f"{h}:00" for h in pattern.peak_hours)
                        
                        insight = AnalyticsInsight(
                            id=f"peak_pattern_{metric_name}_{int(time.time())}",
                            type=AnalyticsType.PATTERNS,
                            title=f"Predictable Peak Usage Pattern - {metric_name}",
                            description=f"Consistent peak usage occurs at {peak_hours_str} daily",
                            severity="low",
                            confidence=pattern.confidence,
                            recommendations=[
                                "Schedule maintenance outside peak hours",
                                "Pre-warm caches before peak periods",
                                "Consider auto-scaling during peak times",
                                "Monitor resource allocation during peaks"
                            ],
                            metrics=pattern.to_dict(),
                            timestamp=datetime.now(timezone.utc),
                            expires_at=datetime.now(timezone.utc) + timedelta(days=1)
                        )
                        insights.append(insight)
                    
                    # Trend-based insights
                    if pattern.trend_direction == "increasing":
                        insight = AnalyticsInsight(
                            id=f"growth_trend_{metric_name}_{int(time.time())}",
                            type=AnalyticsType.PATTERNS,
                            title=f"Growing {metric_name} Trend",
                            description=f"{metric_name} shows consistent growth pattern",
                            severity="medium",
                            confidence=pattern.confidence,
                            recommendations=[
                                "Plan for capacity expansion",
                                "Monitor growth rate sustainability",
                                "Review scaling policies",
                                "Consider optimization opportunities"
                            ],
                            metrics=pattern.to_dict(),
                            timestamp=datetime.now(timezone.utc),
                            expires_at=datetime.now(timezone.utc) + timedelta(days=2)
                        )
                        insights.append(insight)
        
        except Exception as e:
            self.logger.error(f"Error generating pattern insights: {e}")
        
        return insights
    
    async def _generate_anomaly_insights(self, anomalies: List[AnomalyDetection]):
        """Generate insights from detected anomalies."""
        try:
            for anomaly in anomalies:
                insight = AnalyticsInsight(
                    id=f"anomaly_{anomaly.metric}_{int(time.time())}",
                    type=AnalyticsType.ANOMALIES,
                    title=f"Anomaly Detected in {anomaly.metric}",
                    description=f"{anomaly.anomaly_type.value.title()} detected: {anomaly.value:.2f} (expected: {anomaly.expected_value:.2f})",
                    severity=anomaly.severity,
                    confidence=min(0.9, anomaly.deviation / 10),  # Higher deviation = higher confidence
                    recommendations=[
                        "Investigate recent changes to the system",
                        "Check for unusual client behavior",
                        "Review recent deployments or configuration changes",
                        "Monitor for continued anomalous behavior"
                    ],
                    metrics=anomaly.to_dict(),
                    timestamp=anomaly.timestamp,
                    expires_at=anomaly.timestamp + timedelta(hours=2)
                )
                
                self.insights[insight.id] = insight
                await self._store_insight(insight)
        
        except Exception as e:
            self.logger.error(f"Error generating anomaly insights: {e}")
    
    async def _store_insight(self, insight: AnalyticsInsight):
        """Store insight in Redis."""
        try:
            insights_key = self._insights_key()
            insight_data = json.dumps(insight.to_dict())
            timestamp_score = time.time()
            
            await self.redis.zadd(insights_key, {insight_data: timestamp_score})
            await self.redis.expire(insights_key, self.insight_retention)
            
        except Exception as e:
            self.logger.error(f"Error storing insight: {e}")
    
    async def _store_pattern(self, pattern: PerformancePattern):
        """Store detected pattern in Redis."""
        try:
            patterns_key = self._patterns_key()
            pattern_data = json.dumps(pattern.to_dict())
            
            await self.redis.hset(patterns_key, pattern.metric, pattern_data)
            await self.redis.expire(patterns_key, self.insight_retention)
            
        except Exception as e:
            self.logger.error(f"Error storing pattern: {e}")
    
    async def _store_anomalies(self, anomalies: List[AnomalyDetection]):
        """Store detected anomalies in Redis."""
        try:
            anomalies_key = self._anomalies_key()
            
            for anomaly in anomalies:
                anomaly_data = json.dumps(anomaly.to_dict())
                timestamp_score = time.time()
                
                await self.redis.zadd(anomalies_key, {anomaly_data: timestamp_score})
            
            await self.redis.expire(anomalies_key, self.insight_retention)
            
        except Exception as e:
            self.logger.error(f"Error storing anomalies: {e}")
    
    async def _save_baseline_stats(self):
        """Save baseline statistics to Redis."""
        try:
            baselines_key = self._baselines_key()
            baselines_data = json.dumps(self.baseline_stats)
            
            await self.redis.set(baselines_key, baselines_data)
            await self.redis.expire(baselines_key, 86400 * 30)  # 30 days
            
        except Exception as e:
            self.logger.error(f"Error saving baseline stats: {e}")
    
    async def _load_baseline_stats(self):
        """Load baseline statistics from Redis."""
        try:
            baselines_key = self._baselines_key()
            baselines_data = await self.redis.get(baselines_key)
            
            if baselines_data:
                self.baseline_stats = json.loads(baselines_data)
                self.logger.info("Loaded baseline statistics from Redis")
            
        except Exception as e:
            self.logger.error(f"Error loading baseline stats: {e}")
    
    async def _cleanup_old_data(self):
        """Cleanup old analytics data."""
        try:
            current_time = time.time()
            cutoff_time = current_time - self.insight_retention
            
            # Cleanup old insights
            insights_key = self._insights_key()
            await self.redis.zremrangebyscore(insights_key, 0, cutoff_time)
            
            # Cleanup old anomalies
            anomalies_key = self._anomalies_key()
            await self.redis.zremrangebyscore(anomalies_key, 0, cutoff_time)
            
            # Cleanup expired insights from local cache
            expired_insights = [
                insight_id for insight_id, insight in self.insights.items()
                if insight.expires_at and insight.expires_at <= datetime.now(timezone.utc)
            ]
            
            for insight_id in expired_insights:
                self.insights.pop(insight_id, None)
            
        except Exception as e:
            self.logger.error(f"Error in cleanup: {e}")
    
    async def get_insights(self, insight_type: AnalyticsType = None, 
                          severity: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get analytics insights with optional filtering.
        
        Args:
            insight_type: Filter by insight type
            severity: Filter by severity level
            limit: Maximum number of insights to return
            
        Returns:
            List of insight dictionaries
        """
        try:
            insights_key = self._insights_key()
            
            # Get recent insights
            current_time = time.time()
            start_time = current_time - self.insight_retention
            
            results = await self.redis.zrangebyscore(
                insights_key, start_time, current_time, withscores=True
            )
            
            insights = []
            for insight_data, _ in results:
                try:
                    insight_dict = json.loads(insight_data)
                    
                    # Apply filters
                    if insight_type and insight_dict.get('type') != insight_type.value:
                        continue
                    
                    if severity and insight_dict.get('severity') != severity:
                        continue
                    
                    insights.append(insight_dict)
                    
                except json.JSONDecodeError:
                    continue
            
            # Sort by timestamp (newest first) and limit
            insights.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return insights[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting insights: {e}")
            return []
    
    async def get_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get detected patterns for all metrics."""
        try:
            patterns_key = self._patterns_key()
            pattern_data = await self.redis.hgetall(patterns_key)
            
            patterns = {}
            for metric, data in pattern_data.items():
                try:
                    patterns[metric] = json.loads(data)
                except json.JSONDecodeError:
                    continue
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error getting patterns: {e}")
            return {}
    
    async def get_anomalies(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent anomalies.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of anomaly dictionaries
        """
        try:
            anomalies_key = self._anomalies_key()
            
            # Calculate time range
            current_time = time.time()
            start_time = current_time - (hours * 3600)
            
            results = await self.redis.zrangebyscore(
                anomalies_key, start_time, current_time, withscores=True
            )
            
            anomalies = []
            for anomaly_data, _ in results:
                try:
                    anomalies.append(json.loads(anomaly_data))
                except json.JSONDecodeError:
                    continue
            
            return sorted(anomalies, key=lambda x: x.get('timestamp', ''), reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error getting anomalies: {e}")
            return []
    
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        try:
            return {
                'insights': {
                    'total': len(self.insights),
                    'by_type': {
                        insight_type.value: len([
                            i for i in self.insights.values() 
                            if i.type == insight_type
                        ]) for insight_type in AnalyticsType
                    },
                    'by_severity': {
                        severity: len([
                            i for i in self.insights.values() 
                            if i.severity == severity
                        ]) for severity in ['low', 'medium', 'high', 'critical']
                    }
                },
                'patterns': {
                    'detected': len(self.detected_patterns),
                    'metrics_with_patterns': list(self.detected_patterns.keys())
                },
                'baseline_metrics': list(self.baseline_stats.keys()),
                'monitoring_status': {
                    'anomaly_detection': self.enable_anomaly_detection,
                    'pattern_recognition': self.enable_pattern_recognition,
                    'predictive_analysis': self.enable_predictive_analysis
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting analytics summary: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }