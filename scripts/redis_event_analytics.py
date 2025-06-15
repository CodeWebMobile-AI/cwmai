"""
Redis Event Analytics and Insights

Comprehensive real-time analytics engine for Redis Streams events,
providing intelligent insights, pattern detection, and predictive analytics.
"""

import asyncio
import json
import logging
import statistics
import time
import os
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Callable, Tuple, Union
import uuid
import math

from scripts.redis_integration.redis_client import get_redis_client
from scripts.redis_intelligence_hub import IntelligenceEvent, EventType, EventPriority, RedisIntelligenceHub
from scripts.redis_event_sourcing import RedisEventStore
from scripts.redis_lockfree_adapter import create_lockfree_state_manager
from scripts.mcp_redis_integration import MCPRedisIntegration


class AnalyticsMetric:
    """Analytics metric with statistical calculations."""
    
    def __init__(self, name: str, max_samples: int = 1000):
        """Initialize analytics metric.
        
        Args:
            name: Metric name
            max_samples: Maximum samples to keep in memory
        """
        self.name = name
        self.max_samples = max_samples
        self.samples = deque(maxlen=max_samples)
        self.timestamps = deque(maxlen=max_samples)
        
    def add_sample(self, value: Union[int, float], timestamp: datetime = None):
        """Add sample to metric."""
        self.samples.append(value)
        self.timestamps.append(timestamp or datetime.now(timezone.utc))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistical analysis of metric."""
        if not self.samples:
            return {'count': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'sum': 0, 'median': 0}
        
        samples_list = list(self.samples)
        return {
            'count': len(samples_list),
            'mean': statistics.mean(samples_list),
            'median': statistics.median(samples_list),
            'std': statistics.stdev(samples_list) if len(samples_list) > 1 else 0,
            'min': min(samples_list),
            'max': max(samples_list),
            'sum': sum(samples_list)
        }
    
    def get_trend(self, window_minutes: int = 5) -> str:
        """Get trend analysis for recent samples."""
        if len(self.samples) < 2:
            return 'insufficient_data'
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        recent_samples = []
        
        for i, timestamp in enumerate(self.timestamps):
            if timestamp >= cutoff_time:
                recent_samples.append(self.samples[i])
        
        if len(recent_samples) < 2:
            return 'insufficient_recent_data'
        
        # Simple trend analysis
        first_half = recent_samples[:len(recent_samples)//2]
        second_half = recent_samples[len(recent_samples)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        if second_avg > first_avg * 1.1:
            return 'increasing'
        elif second_avg < first_avg * 0.9:
            return 'decreasing'
        else:
            return 'stable'


class EventPattern:
    """Event pattern detection and analysis."""
    
    def __init__(self, pattern_id: str, event_types: List[EventType], window_seconds: int = 300):
        """Initialize event pattern.
        
        Args:
            pattern_id: Unique pattern identifier
            event_types: Event types to track
            window_seconds: Time window for pattern detection
        """
        self.pattern_id = pattern_id
        self.event_types = event_types
        self.window_seconds = window_seconds
        self.occurrences = deque()
        self.last_detected = None
        
    def add_event(self, event: IntelligenceEvent):
        """Add event for pattern analysis."""
        if event.event_type in self.event_types:
            self.occurrences.append({
                'timestamp': event.timestamp,
                'event_type': event.event_type,
                'worker_id': event.worker_id,
                'data': event.data
            })
            
            # Clean old occurrences
            cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.window_seconds)
            while self.occurrences and self.occurrences[0]['timestamp'] < cutoff_time:
                self.occurrences.popleft()
    
    def detect_pattern(self) -> Optional[Dict[str, Any]]:
        """Detect if pattern is present."""
        if len(self.occurrences) < len(self.event_types):
            return None
        
        # Check if all event types occurred in the window
        recent_types = {occ['event_type'] for occ in self.occurrences}
        
        if all(event_type in recent_types for event_type in self.event_types):
            self.last_detected = datetime.now(timezone.utc)
            return {
                'pattern_id': self.pattern_id,
                'detected_at': self.last_detected.isoformat(),
                'occurrences_count': len(self.occurrences),
                'event_types': [et.value for et in recent_types],
                'worker_coverage': len({occ['worker_id'] for occ in self.occurrences})
            }
        
        return None


class AnomalyDetector:
    """Statistical anomaly detection for events."""
    
    def __init__(self, metric_name: str, threshold_std: float = 2.0, min_samples: int = 10):
        """Initialize anomaly detector.
        
        Args:
            metric_name: Name of metric to monitor
            threshold_std: Standard deviation threshold for anomalies
            min_samples: Minimum samples before detection starts
        """
        self.metric_name = metric_name
        self.threshold_std = threshold_std
        self.min_samples = min_samples
        self.metric = AnalyticsMetric(metric_name)
        self.anomalies_detected = 0
        
    def add_value(self, value: Union[int, float]) -> Optional[Dict[str, Any]]:
        """Add value and check for anomaly."""
        self.metric.add_sample(value)
        
        if len(self.metric.samples) < self.min_samples:
            return None
        
        stats = self.metric.get_stats()
        z_score = abs(value - stats['mean']) / max(stats['std'], 0.001)
        
        if z_score > self.threshold_std:
            self.anomalies_detected += 1
            return {
                'metric_name': self.metric_name,
                'anomaly_value': value,
                'expected_mean': stats['mean'],
                'z_score': z_score,
                'threshold': self.threshold_std,
                'severity': 'high' if z_score > self.threshold_std * 2 else 'medium',
                'detected_at': datetime.now(timezone.utc).isoformat()
            }
        
        return None


class RedisEventAnalytics:
    """Comprehensive Redis event analytics engine."""
    
    def __init__(self,
                 analytics_id: str = None,
                 enable_real_time: bool = True,
                 enable_pattern_detection: bool = True,
                 enable_anomaly_detection: bool = True,
                 enable_predictive_analytics: bool = True):
        """Initialize Redis event analytics engine.
        
        Args:
            analytics_id: Unique analytics engine identifier
            enable_real_time: Enable real-time analytics
            enable_pattern_detection: Enable pattern detection
            enable_anomaly_detection: Enable anomaly detection
            enable_predictive_analytics: Enable predictive analytics
        """
        self.analytics_id = analytics_id or f"analytics_{uuid.uuid4().hex[:8]}"
        self.enable_real_time = enable_real_time
        self.enable_pattern_detection = enable_pattern_detection
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_predictive_analytics = enable_predictive_analytics
        
        self.logger = logging.getLogger(f"{__name__}.RedisEventAnalytics")
        
        # Redis components
        self.redis_client = None
        self.intelligence_hub: Optional[RedisIntelligenceHub] = None
        self.event_store: Optional[RedisEventStore] = None
        self.state_manager = None
        
        # Analytics components
        self.metrics: Dict[str, AnalyticsMetric] = {
            # Initialize critical metric to avoid early access errors
            'event_throughput': AnalyticsMetric('event_throughput')
        }
        self.patterns: Dict[str, EventPattern] = {}
        self.anomaly_detectors: Dict[str, AnomalyDetector] = {}
        
        # Event processing
        self._event_processors: List[Callable] = []
        self._analytics_tasks: List[asyncio.Task] = []
        self._shutdown = False
        
        # Performance tracking
        self._analytics_metrics = {
            'events_processed': 0,
            'insights_generated': 0,
            'anomalies_detected': 0,
            'patterns_detected': 0,
            'predictions_made': 0,
            'processing_time_ms': 0.0
        }
        
        # Real-time state
        self._event_rates = defaultdict(lambda: AnalyticsMetric('event_rate', 100))
        self._worker_performance = defaultdict(lambda: AnalyticsMetric('performance', 500))
        self._task_metrics = defaultdict(lambda: AnalyticsMetric('task_metric', 200))
        
        # Insights storage
        self._insights_history = deque(maxlen=1000)
        self._alert_history = deque(maxlen=500)
        
        # MCP-Redis integration
        self.mcp_redis: Optional[MCPRedisIntegration] = None
        self._use_mcp = os.getenv("USE_MCP_REDIS", "false").lower() == "true"
    
    async def initialize(self):
        """Initialize analytics engine components."""
        try:
            self.logger.info(f"Initializing Redis Event Analytics: {self.analytics_id}")
            
            # Initialize Redis components
            self.redis_client = await get_redis_client()
            self.state_manager = create_lockfree_state_manager(f"event_analytics_{self.analytics_id}")
            await self.state_manager.initialize()
            
            # Initialize intelligence hub integration
            from scripts.redis_intelligence_hub import get_intelligence_hub
            self.intelligence_hub = await get_intelligence_hub()
            
            # Initialize event store integration
            from scripts.redis_event_sourcing import get_event_store
            self.event_store = await get_event_store()
            
            # Register event processors
            await self._register_event_processors()
            
            # Initialize built-in analytics
            await self._initialize_builtin_analytics()
            
            # Start analytics tasks
            await self._start_analytics_tasks()
            
            # Register analytics engine
            await self._register_analytics_engine()
            
            # Initialize MCP-Redis if enabled
            if self._use_mcp:
                try:
                    self.mcp_redis = MCPRedisIntegration()
                    await self.mcp_redis.initialize()
                    self.logger.info("MCP-Redis integration enabled for event analytics")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize MCP-Redis: {e}")
                    self._use_mcp = False
            
            self.logger.info(f"Event Analytics {self.analytics_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Event Analytics: {e}")
            raise
    
    async def _register_event_processors(self):
        """Register event processors with intelligence hub."""
        try:
            # Register for all event types to perform analytics
            for event_type in EventType:
                self.intelligence_hub.register_event_processor(
                    event_type,
                    self._process_event_for_analytics
                )
            
        except Exception as e:
            self.logger.error(f"Error registering event processors: {e}")
            raise
    
    async def _initialize_builtin_analytics(self):
        """Initialize built-in analytics patterns and detectors."""
        try:
            # Built-in metrics
            self.metrics.update({
                'event_throughput': AnalyticsMetric('event_throughput'),
                'worker_count': AnalyticsMetric('worker_count'),
                'task_completion_rate': AnalyticsMetric('task_completion_rate'),
                'error_rate': AnalyticsMetric('error_rate'),
                'ai_response_time': AnalyticsMetric('ai_response_time'),
                'system_load': AnalyticsMetric('system_load')
            })
            
            # Built-in patterns
            if self.enable_pattern_detection:
                self.patterns.update({
                    'worker_lifecycle': EventPattern(
                        'worker_lifecycle',
                        [EventType.WORKER_REGISTRATION, EventType.WORKER_HEARTBEAT, EventType.WORKER_SHUTDOWN],
                        600  # 10 minutes
                    ),
                    'task_workflow': EventPattern(
                        'task_workflow',
                        [EventType.TASK_ASSIGNMENT, EventType.TASK_PROGRESS, EventType.TASK_COMPLETION],
                        1800  # 30 minutes
                    ),
                    'ai_interaction': EventPattern(
                        'ai_interaction',
                        [EventType.AI_REQUEST, EventType.AI_RESPONSE],
                        300  # 5 minutes
                    ),
                    'performance_degradation': EventPattern(
                        'performance_degradation',
                        [EventType.PERFORMANCE_METRIC, EventType.ERROR_EVENT],
                        900  # 15 minutes
                    )
                })
            
            # Built-in anomaly detectors
            if self.enable_anomaly_detection:
                self.anomaly_detectors.update({
                    'event_rate_anomaly': AnomalyDetector('event_rate', 2.5, 20),
                    'response_time_anomaly': AnomalyDetector('response_time', 3.0, 15),
                    'error_rate_anomaly': AnomalyDetector('error_rate', 2.0, 10),
                    'worker_performance_anomaly': AnomalyDetector('worker_performance', 2.5, 25)
                })
            
            self.logger.info("Built-in analytics initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing built-in analytics: {e}")
    
    async def _start_analytics_tasks(self):
        """Start analytics background tasks."""
        try:
            if self.enable_real_time:
                # Real-time analytics processor
                realtime_task = asyncio.create_task(self._realtime_analytics_processor())
                self._analytics_tasks.append(realtime_task)
            
            # Insights generator
            insights_task = asyncio.create_task(self._insights_generator())
            self._analytics_tasks.append(insights_task)
            
            # Pattern detection task
            if self.enable_pattern_detection:
                pattern_task = asyncio.create_task(self._pattern_detection_processor())
                self._analytics_tasks.append(pattern_task)
            
            # Anomaly detection task
            if self.enable_anomaly_detection:
                anomaly_task = asyncio.create_task(self._anomaly_detection_processor())
                self._analytics_tasks.append(anomaly_task)
            
            # Predictive analytics task
            if self.enable_predictive_analytics:
                prediction_task = asyncio.create_task(self._predictive_analytics_processor())
                self._analytics_tasks.append(prediction_task)
            
            # Performance dashboard updater
            dashboard_task = asyncio.create_task(self._dashboard_updater())
            self._analytics_tasks.append(dashboard_task)
            
            self.logger.info(f"Started {len(self._analytics_tasks)} analytics tasks")
            
        except Exception as e:
            self.logger.error(f"Error starting analytics tasks: {e}")
    
    async def _process_event_for_analytics(self, event: IntelligenceEvent, stream_name: str):
        """Process event for analytics."""
        start_time = time.time()
        
        try:
            # Update basic metrics
            await self._update_basic_metrics(event)
            
            # Update event rate metrics
            self._event_rates[event.event_type].add_sample(1)
            
            # Update worker performance metrics
            if event.event_type == EventType.PERFORMANCE_METRIC:
                metric_value = event.data.get('value', 0)
                self._worker_performance[event.worker_id].add_sample(metric_value)
            
            # Update task metrics
            if event.event_type in [EventType.TASK_COMPLETION, EventType.TASK_FAILURE]:
                duration = event.data.get('duration_seconds', 0)
                self._task_metrics['duration'].add_sample(duration)
            
            # Pattern detection
            if self.enable_pattern_detection:
                for pattern in self.patterns.values():
                    pattern.add_event(event)
            
            # Anomaly detection
            if self.enable_anomaly_detection:
                await self._check_event_anomalies(event)
            
            # Update processing metrics
            processing_time = (time.time() - start_time) * 1000
            self._analytics_metrics['events_processed'] += 1
            self._analytics_metrics['processing_time_ms'] += processing_time
            
        except Exception as e:
            self.logger.error(f"Error processing event for analytics: {e}")
    
    async def _update_basic_metrics(self, event: IntelligenceEvent):
        """Update basic analytics metrics."""
        try:
            # Event throughput
            if 'event_throughput' in self.metrics:
                self.metrics['event_throughput'].add_sample(1)
            
            # AI response time
            if event.event_type == EventType.AI_RESPONSE and 'ai_response_time' in self.metrics:
                response_time = event.data.get('duration_ms', 0)
                self.metrics['ai_response_time'].add_sample(response_time)
            
            # Error rate
            if event.event_type == EventType.ERROR_EVENT and 'error_rate' in self.metrics:
                self.metrics['error_rate'].add_sample(1)
            
            # Task completion rate
            if event.event_type == EventType.TASK_COMPLETION and 'task_completion_rate' in self.metrics:
                self.metrics['task_completion_rate'].add_sample(1)
            elif event.event_type == EventType.TASK_FAILURE and 'task_completion_rate' in self.metrics:
                self.metrics['task_completion_rate'].add_sample(0)
            
        except Exception as e:
            self.logger.error(f"Error updating basic metrics: {e}")
    
    async def _check_event_anomalies(self, event: IntelligenceEvent):
        """Check for anomalies in event data."""
        try:
            # Check response time anomalies
            if event.event_type == EventType.AI_RESPONSE:
                response_time = event.data.get('duration_ms', 0)
                anomaly = self.anomaly_detectors['response_time_anomaly'].add_value(response_time)
                
                if anomaly:
                    await self._handle_anomaly(anomaly, event)
            
            # Check performance anomalies
            if event.event_type == EventType.PERFORMANCE_METRIC:
                metric_value = event.data.get('value', 0)
                anomaly = self.anomaly_detectors['worker_performance_anomaly'].add_value(metric_value)
                
                if anomaly:
                    await self._handle_anomaly(anomaly, event)
            
        except Exception as e:
            self.logger.error(f"Error checking event anomalies: {e}")
    
    async def _handle_anomaly(self, anomaly: Dict[str, Any], event: IntelligenceEvent):
        """Handle detected anomaly."""
        try:
            # Create anomaly event
            anomaly_event = IntelligenceEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.ERROR_EVENT,
                worker_id=self.analytics_id,
                timestamp=datetime.now(timezone.utc),
                priority=EventPriority.HIGH if anomaly.get('severity') == 'high' else EventPriority.NORMAL,
                data={
                    'event_subtype': 'anomaly_detected',
                    'anomaly': anomaly,
                    'original_event': {
                        'event_id': event.event_id,
                        'event_type': event.event_type.value,
                        'worker_id': event.worker_id
                    }
                }
            )
            
            await self.intelligence_hub.publish_event(anomaly_event)
            
            # Store in alert history
            self._alert_history.append({
                'type': 'anomaly',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'anomaly': anomaly,
                'event_context': event.event_id
            })
            
            self._analytics_metrics['anomalies_detected'] += 1
            
            self.logger.warning(f"Anomaly detected: {anomaly['metric_name']} = {anomaly['anomaly_value']} (z-score: {anomaly['z_score']:.2f})")
            
        except Exception as e:
            self.logger.error(f"Error handling anomaly: {e}")
    
    async def _realtime_analytics_processor(self):
        """Real-time analytics processing."""
        while not self._shutdown:
            try:
                # Calculate real-time metrics
                current_time = datetime.now(timezone.utc)
                
                # Event rate analytics
                await self._calculate_event_rates()
                
                # Worker analytics
                await self._calculate_worker_analytics()
                
                # System health analytics
                await self._calculate_system_health()
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in real-time analytics processor: {e}")
                await asyncio.sleep(10)
    
    async def _insights_generator(self):
        """Generate intelligent insights from analytics data."""
        while not self._shutdown:
            try:
                insights = []
                
                # Performance insights
                performance_insights = await self._generate_performance_insights()
                insights.extend(performance_insights)
                
                # Capacity insights
                capacity_insights = await self._generate_capacity_insights()
                insights.extend(capacity_insights)
                
                # Efficiency insights
                efficiency_insights = await self._generate_efficiency_insights()
                insights.extend(efficiency_insights)
                
                # Quality insights
                quality_insights = await self._generate_quality_insights()
                insights.extend(quality_insights)
                
                # Publish insights
                for insight in insights:
                    await self._publish_insight(insight)
                
                self._analytics_metrics['insights_generated'] += len(insights)
                
                await asyncio.sleep(300)  # Generate insights every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in insights generator: {e}")
                await asyncio.sleep(60)
    
    async def _pattern_detection_processor(self):
        """Pattern detection processing."""
        while not self._shutdown:
            try:
                detected_patterns = []
                
                for pattern_id, pattern in self.patterns.items():
                    detection = pattern.detect_pattern()
                    if detection:
                        detected_patterns.append(detection)
                
                # Publish detected patterns
                for pattern in detected_patterns:
                    await self._publish_pattern_detection(pattern)
                
                self._analytics_metrics['patterns_detected'] += len(detected_patterns)
                
                await asyncio.sleep(30)  # Check patterns every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in pattern detection processor: {e}")
                await asyncio.sleep(60)
    
    async def _anomaly_detection_processor(self):
        """Anomaly detection processing."""
        while not self._shutdown:
            try:
                # Check event rate anomalies
                await self._check_event_rate_anomalies()
                
                # Check system-wide anomalies
                await self._check_system_anomalies()
                
                await asyncio.sleep(60)  # Check anomalies every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in anomaly detection processor: {e}")
                await asyncio.sleep(120)
    
    async def _predictive_analytics_processor(self):
        """Predictive analytics processing."""
        while not self._shutdown:
            try:
                predictions = []
                
                # Predict system load
                load_prediction = await self._predict_system_load()
                if load_prediction:
                    predictions.append(load_prediction)
                
                # Predict worker capacity needs
                capacity_prediction = await self._predict_capacity_needs()
                if capacity_prediction:
                    predictions.append(capacity_prediction)
                
                # Predict performance trends
                performance_prediction = await self._predict_performance_trends()
                if performance_prediction:
                    predictions.append(performance_prediction)
                
                # Publish predictions
                for prediction in predictions:
                    await self._publish_prediction(prediction)
                
                self._analytics_metrics['predictions_made'] += len(predictions)
                
                await asyncio.sleep(900)  # Generate predictions every 15 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in predictive analytics processor: {e}")
                await asyncio.sleep(300)
    
    async def _dashboard_updater(self):
        """Update analytics dashboard data."""
        while not self._shutdown:
            try:
                dashboard_data = await self._generate_dashboard_data()
                
                # Store dashboard data in Redis
                await self.state_manager.update(
                    f"analytics.dashboard.{self.analytics_id}",
                    dashboard_data,
                    distributed=True
                )
                
                await asyncio.sleep(30)  # Update dashboard every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error updating dashboard: {e}")
                await asyncio.sleep(60)
    
    async def _calculate_event_rates(self):
        """Calculate real-time event rates."""
        try:
            # Calculate rates for each event type
            event_rate_summary = {}
            
            for event_type, metric in self._event_rates.items():
                stats = metric.get_stats()
                trend = metric.get_trend(window_minutes=5)
                
                # Calculate events per minute
                if len(metric.timestamps) > 1:
                    time_span = (metric.timestamps[-1] - metric.timestamps[0]).total_seconds()
                    events_per_minute = (stats['count'] / max(time_span, 1)) * 60
                else:
                    events_per_minute = 0
                
                event_rate_summary[event_type.value if hasattr(event_type, 'value') else str(event_type)] = {
                    'events_per_minute': events_per_minute,
                    'total_events': stats['count'],
                    'trend': trend
                }
            
            # Store event rate summary
            await self.state_manager.update(
                f"analytics.event_rates.{self.analytics_id}",
                event_rate_summary,
                distributed=True
            )
            
            # Update system load metric based on event rates
            total_events_per_minute = sum(
                rate_info['events_per_minute'] 
                for rate_info in event_rate_summary.values()
            )
            self.metrics['system_load'].add_sample(total_events_per_minute)
            
        except Exception as e:
            self.logger.error(f"Error calculating event rates: {e}")
    
    async def _calculate_health_scores(self) -> Dict[str, float]:
        """Calculate health scores for various system components."""
        try:
            health_scores = {}
            
            # Overall system health (0-100)
            system_health = 100.0
            
            # Event throughput health
            throughput_stats = self.metrics['event_throughput'].get_stats()
            if throughput_stats['count'] > 0:
                # Good health if throughput is stable or increasing
                throughput_trend = self.metrics['event_throughput'].get_trend()
                if throughput_trend == 'decreasing':
                    system_health -= 10
                health_scores['throughput_health'] = 90 if throughput_trend == 'decreasing' else 100
            else:
                health_scores['throughput_health'] = 0
                system_health -= 20
            
            # Error rate health
            error_stats = self.metrics['error_rate'].get_stats()
            if error_stats['count'] > 0:
                # Calculate error percentage
                total_events = self.metrics['event_throughput'].get_stats()['count']
                error_percentage = (error_stats['sum'] / max(total_events, 1)) * 100
                
                # Deduct health based on error rate
                if error_percentage > 10:
                    system_health -= 30
                elif error_percentage > 5:
                    system_health -= 20
                elif error_percentage > 1:
                    system_health -= 10
                
                health_scores['error_health'] = max(0, 100 - (error_percentage * 10))
            else:
                health_scores['error_health'] = 100
            
            # Response time health
            response_stats = self.metrics['ai_response_time'].get_stats()
            if response_stats['count'] > 0:
                avg_response_time = response_stats['mean']
                # Good health if response time < 1000ms
                if avg_response_time > 5000:
                    system_health -= 25
                    health_scores['response_health'] = 25
                elif avg_response_time > 2000:
                    system_health -= 15
                    health_scores['response_health'] = 60
                elif avg_response_time > 1000:
                    system_health -= 5
                    health_scores['response_health'] = 85
                else:
                    health_scores['response_health'] = 100
            else:
                health_scores['response_health'] = 100
            
            # Task completion health
            completion_stats = self.metrics['task_completion_rate'].get_stats()
            if completion_stats['count'] > 0:
                completion_rate = completion_stats['mean'] * 100
                health_scores['completion_health'] = completion_rate
                
                if completion_rate < 50:
                    system_health -= 30
                elif completion_rate < 80:
                    system_health -= 15
            else:
                health_scores['completion_health'] = 100
            
            # Worker health (based on active workers)
            worker_count_stats = self.metrics['worker_count'].get_stats()
            if worker_count_stats['count'] > 0:
                current_workers = worker_count_stats['mean']
                if current_workers < 1:
                    system_health -= 50
                    health_scores['worker_health'] = 0
                elif current_workers < 3:
                    system_health -= 20
                    health_scores['worker_health'] = 60
                else:
                    health_scores['worker_health'] = 100
            else:
                health_scores['worker_health'] = 0
                system_health -= 50
            
            # Overall system health
            health_scores['system_health'] = max(0, min(100, system_health))
            
            return health_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating health scores: {e}")
            return {'system_health': 0}
    
    async def _generate_performance_insights(self) -> List[Dict[str, Any]]:
        """Generate insights about system performance."""
        insights = []
        
        try:
            # Response time insights
            response_stats = self.metrics['ai_response_time'].get_stats()
            if response_stats['count'] > 10:
                avg_response = response_stats['mean']
                trend = self.metrics['ai_response_time'].get_trend()
                
                if avg_response > 3000 and trend == 'increasing':
                    insights.append({
                        'type': 'performance_degradation',
                        'severity': 'high',
                        'message': f"AI response times are degrading. Average: {avg_response:.0f}ms, Trend: {trend}",
                        'recommendation': "Consider scaling AI resources or optimizing prompts",
                        'metric': 'ai_response_time',
                        'value': avg_response
                    })
                elif avg_response < 500 and trend == 'stable':
                    insights.append({
                        'type': 'performance_excellent',
                        'severity': 'info',
                        'message': f"AI response times are excellent. Average: {avg_response:.0f}ms",
                        'recommendation': "Current configuration is optimal",
                        'metric': 'ai_response_time',
                        'value': avg_response
                    })
            
            # Task completion insights
            completion_stats = self.metrics['task_completion_rate'].get_stats()
            if completion_stats['count'] > 5:
                completion_rate = completion_stats['mean'] * 100
                
                if completion_rate < 70:
                    insights.append({
                        'type': 'low_completion_rate',
                        'severity': 'high',
                        'message': f"Task completion rate is low: {completion_rate:.1f}%",
                        'recommendation': "Review task complexity and worker capabilities",
                        'metric': 'task_completion_rate',
                        'value': completion_rate
                    })
            
            # Error rate insights
            error_stats = self.metrics['error_rate'].get_stats()
            throughput_stats = self.metrics['event_throughput'].get_stats()
            
            if error_stats['count'] > 0 and throughput_stats['count'] > 0:
                error_percentage = (error_stats['sum'] / throughput_stats['count']) * 100
                
                if error_percentage > 5:
                    insights.append({
                        'type': 'high_error_rate',
                        'severity': 'high',
                        'message': f"Error rate is elevated: {error_percentage:.1f}%",
                        'recommendation': "Investigate error patterns and implement fixes",
                        'metric': 'error_rate',
                        'value': error_percentage
                    })
            
            # Store insights in history
            for insight in insights:
                insight['timestamp'] = datetime.now(timezone.utc).isoformat()
                insight['analytics_id'] = self.analytics_id
                self._insights_history.append(insight)
            
        except Exception as e:
            self.logger.error(f"Error generating performance insights: {e}")
        
        return insights
    
    async def _check_event_rate_anomalies(self):
        """Check for anomalies in event rates."""
        try:
            # Check overall event rate
            throughput_stats = self.metrics['event_throughput'].get_stats()
            if throughput_stats['count'] > 20:
                current_rate = len([t for t in self.metrics['event_throughput'].timestamps 
                                  if t > datetime.now(timezone.utc) - timedelta(minutes=1)])
                
                anomaly = self.anomaly_detectors['event_rate_anomaly'].add_value(current_rate)
                if anomaly:
                    await self._handle_anomaly(anomaly, IntelligenceEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=EventType.PERFORMANCE_METRIC,
                        worker_id=self.analytics_id,
                        timestamp=datetime.now(timezone.utc),
                        priority=EventPriority.HIGH,
                        data={'metric': 'event_rate', 'value': current_rate}
                    ))
            
            # Check individual event type rates
            for event_type, metric in self._event_rates.items():
                if len(metric.samples) > 10:
                    recent_rate = len([t for t in metric.timestamps 
                                     if t > datetime.now(timezone.utc) - timedelta(minutes=1)])
                    
                    # Check if rate is unusually high or low
                    stats = metric.get_stats()
                    if stats['std'] > 0:
                        z_score = abs(recent_rate - stats['mean']) / stats['std']
                        if z_score > 3:
                            self.logger.warning(
                                f"Unusual rate for {event_type}: {recent_rate} events/min "
                                f"(mean: {stats['mean']:.1f}, z-score: {z_score:.1f})"
                            )
            
        except Exception as e:
            self.logger.error(f"Error checking event rate anomalies: {e}")
    
    async def _predict_system_load(self) -> Optional[Dict[str, Any]]:
        """Predict future system load based on historical patterns."""
        try:
            load_stats = self.metrics['system_load'].get_stats()
            
            if load_stats['count'] < 50:
                return None
            
            # Simple linear prediction based on recent trend
            recent_samples = list(self.metrics['system_load'].samples)[-20:]
            if len(recent_samples) < 10:
                return None
            
            # Calculate trend
            first_half_avg = statistics.mean(recent_samples[:10])
            second_half_avg = statistics.mean(recent_samples[10:])
            
            # Predict next hour
            trend_rate = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg > 0 else 0
            predicted_load = second_half_avg * (1 + trend_rate)
            
            # Determine if action needed
            confidence = min(0.9, load_stats['count'] / 100)  # Higher confidence with more data
            
            prediction = {
                'type': 'system_load_prediction',
                'current_load': second_half_avg,
                'predicted_load_1h': predicted_load,
                'trend': 'increasing' if trend_rate > 0.1 else 'decreasing' if trend_rate < -0.1 else 'stable',
                'confidence': confidence,
                'recommendation': None
            }
            
            # Add recommendations
            if predicted_load > load_stats['max'] * 1.2:
                prediction['recommendation'] = "Consider scaling up workers to handle predicted load increase"
                prediction['severity'] = 'high'
            elif predicted_load < load_stats['mean'] * 0.5:
                prediction['recommendation'] = "System load decreasing, consider scaling down to save resources"
                prediction['severity'] = 'low'
            else:
                prediction['severity'] = 'info'
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting system load: {e}")
            return None
    
    async def _generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard data."""
        try:
            # Get current metrics
            metrics_summary = {}
            for name, metric in self.metrics.items():
                metrics_summary[name] = metric.get_stats()
            
            # Get recent insights
            recent_insights = list(self._insights_history)[-10:]
            
            # Get recent alerts
            recent_alerts = list(self._alert_history)[-5:]
            
            # Calculate health scores
            health_scores = await self._calculate_health_scores()
            
            return {
                'analytics_id': self.analytics_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'metrics_summary': metrics_summary,
                'recent_insights': recent_insights,
                'recent_alerts': recent_alerts,
                'health_scores': health_scores,
                'analytics_performance': self._analytics_metrics.copy(),
                'active_patterns': len(self.patterns),
                'active_detectors': len(self.anomaly_detectors)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating dashboard data: {e}")
            return {}
    
    async def _calculate_worker_analytics(self):
        """Calculate analytics for worker performance and status."""
        try:
            worker_analytics = {}
            
            # Analyze each worker's performance
            for worker_id, metric in self._worker_performance.items():
                stats = metric.get_stats()
                trend = metric.get_trend()
                
                worker_analytics[worker_id] = {
                    'performance_stats': stats,
                    'performance_trend': trend,
                    'samples_count': stats['count'],
                    'health_score': min(100, stats['mean']) if stats['count'] > 0 else 0
                }
                
                # Check for performance issues
                if stats['count'] > 5 and stats['mean'] < 50:
                    self.logger.warning(f"Worker {worker_id} showing poor performance: {stats['mean']:.1f}")
            
            # Calculate aggregate worker metrics
            if worker_analytics:
                avg_performance = statistics.mean(
                    wa['performance_stats']['mean'] 
                    for wa in worker_analytics.values() 
                    if wa['performance_stats']['count'] > 0
                )
                active_workers = len([wa for wa in worker_analytics.values() if wa['samples_count'] > 0])
                
                # Update worker count metric
                self.metrics['worker_count'].add_sample(active_workers)
                
                # Store worker analytics
                await self.state_manager.update(
                    f"analytics.workers.{self.analytics_id}",
                    {
                        'worker_details': worker_analytics,
                        'active_workers': active_workers,
                        'average_performance': avg_performance,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    },
                    distributed=True
                )
            
        except Exception as e:
            self.logger.error(f"Error calculating worker analytics: {e}")
    
    async def _calculate_system_health(self):
        """Calculate overall system health metrics."""
        try:
            # Get health scores
            health_scores = await self._calculate_health_scores()
            
            # Calculate additional system metrics
            system_metrics = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'health_scores': health_scores,
                'active_components': {
                    'workers': self.metrics['worker_count'].get_stats()['mean'] if self.metrics['worker_count'].get_stats()['count'] > 0 else 0,
                    'patterns_monitoring': len(self.patterns),
                    'anomaly_detectors': len(self.anomaly_detectors),
                    'metrics_tracked': len(self.metrics)
                },
                'performance_summary': {
                    'avg_response_time': self.metrics['ai_response_time'].get_stats()['mean'] if self.metrics['ai_response_time'].get_stats()['count'] > 0 else 0,
                    'task_completion_rate': self.metrics['task_completion_rate'].get_stats()['mean'] * 100 if self.metrics['task_completion_rate'].get_stats()['count'] > 0 else 0,
                    'error_rate': (self.metrics['error_rate'].get_stats()['sum'] / max(self.metrics['event_throughput'].get_stats()['count'], 1)) * 100 if self.metrics['error_rate'].get_stats()['count'] > 0 else 0
                }
            }
            
            # Store system health
            await self.state_manager.update(
                f"analytics.system_health.{self.analytics_id}",
                system_metrics,
                distributed=True
            )
            
            # Check if system health is critical
            if health_scores.get('system_health', 0) < 50:
                self.logger.warning(f"System health is critical: {health_scores['system_health']}")
                
                # Create alert event
                alert_event = IntelligenceEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=EventType.ERROR_EVENT,
                    worker_id=self.analytics_id,
                    timestamp=datetime.now(timezone.utc),
                    priority=EventPriority.CRITICAL,
                    data={
                        'event_subtype': 'system_health_critical',
                        'health_scores': health_scores,
                        'message': 'System health has dropped below critical threshold'
                    }
                )
                await self.intelligence_hub.publish_event(alert_event)
            
        except Exception as e:
            self.logger.error(f"Error calculating system health: {e}")
    
    async def _generate_capacity_insights(self) -> List[Dict[str, Any]]:
        """Generate insights about system capacity."""
        insights = []
        
        try:
            # Worker capacity insights
            worker_stats = self.metrics['worker_count'].get_stats()
            if worker_stats['count'] > 5:
                current_workers = worker_stats['mean']
                max_workers = worker_stats['max']
                
                # Check if more workers needed
                load_stats = self.metrics['system_load'].get_stats()
                if load_stats['count'] > 0:
                    load_per_worker = load_stats['mean'] / max(current_workers, 1)
                    
                    if load_per_worker > 100:  # High load per worker
                        insights.append({
                            'type': 'capacity_shortage',
                            'severity': 'high',
                            'message': f"High load per worker: {load_per_worker:.0f} events/min/worker",
                            'recommendation': f"Scale up from {current_workers:.0f} to {math.ceil(load_stats['mean'] / 50)} workers",
                            'metric': 'worker_capacity',
                            'value': load_per_worker
                        })
                    elif load_per_worker < 10 and current_workers > 1:  # Low utilization
                        insights.append({
                            'type': 'capacity_excess',
                            'severity': 'low',
                            'message': f"Low worker utilization: {load_per_worker:.0f} events/min/worker",
                            'recommendation': f"Consider scaling down to {max(1, math.ceil(load_stats['mean'] / 30))} workers",
                            'metric': 'worker_capacity',
                            'value': load_per_worker
                        })
            
            # Task queue capacity
            task_duration_stats = self._task_metrics['duration'].get_stats() if 'duration' in self._task_metrics else None
            if task_duration_stats and task_duration_stats['count'] > 10:
                avg_duration = task_duration_stats['mean']
                if avg_duration > 300:  # Tasks taking more than 5 minutes
                    insights.append({
                        'type': 'task_processing_slow',
                        'severity': 'medium',
                        'message': f"Tasks taking long to complete: {avg_duration:.0f} seconds average",
                        'recommendation': "Optimize task processing or increase parallel processing capacity",
                        'metric': 'task_duration',
                        'value': avg_duration
                    })
            
            # Store insights
            for insight in insights:
                insight['timestamp'] = datetime.now(timezone.utc).isoformat()
                insight['analytics_id'] = self.analytics_id
                self._insights_history.append(insight)
            
        except Exception as e:
            self.logger.error(f"Error generating capacity insights: {e}")
        
        return insights
    
    async def _generate_efficiency_insights(self) -> List[Dict[str, Any]]:
        """Generate insights about system efficiency."""
        insights = []
        
        try:
            # Task completion efficiency
            completion_stats = self.metrics['task_completion_rate'].get_stats()
            if completion_stats['count'] > 10:
                completion_rate = completion_stats['mean'] * 100
                trend = self.metrics['task_completion_rate'].get_trend()
                
                if completion_rate < 90 and trend == 'decreasing':
                    insights.append({
                        'type': 'efficiency_declining',
                        'severity': 'medium',
                        'message': f"Task completion efficiency declining: {completion_rate:.1f}% and {trend}",
                        'recommendation': "Review recent changes and task complexity",
                        'metric': 'task_completion_efficiency',
                        'value': completion_rate
                    })
                elif completion_rate > 95 and trend == 'stable':
                    insights.append({
                        'type': 'efficiency_optimal',
                        'severity': 'info',
                        'message': f"System running at optimal efficiency: {completion_rate:.1f}% completion rate",
                        'recommendation': "Maintain current configuration",
                        'metric': 'task_completion_efficiency',
                        'value': completion_rate
                    })
            
            # Resource utilization efficiency
            if self._worker_performance:
                performance_values = [
                    metric.get_stats()['mean'] 
                    for metric in self._worker_performance.values() 
                    if metric.get_stats()['count'] > 0
                ]
                
                if performance_values:
                    avg_performance = statistics.mean(performance_values)
                    performance_variance = statistics.variance(performance_values) if len(performance_values) > 1 else 0
                    
                    if performance_variance > 1000:  # High variance in worker performance
                        insights.append({
                            'type': 'performance_imbalance',
                            'severity': 'medium',
                            'message': f"High variance in worker performance (variance: {performance_variance:.0f})",
                            'recommendation': "Balance workload distribution across workers",
                            'metric': 'performance_variance',
                            'value': performance_variance
                        })
            
            # AI response efficiency
            response_stats = self.metrics['ai_response_time'].get_stats()
            if response_stats['count'] > 20:
                p95_response = sorted(list(self.metrics['ai_response_time'].samples))[int(len(self.metrics['ai_response_time'].samples) * 0.95)] if len(self.metrics['ai_response_time'].samples) > 0 else 0
                
                if p95_response > response_stats['mean'] * 2:
                    insights.append({
                        'type': 'response_time_variability',
                        'severity': 'medium',
                        'message': f"High response time variability. P95: {p95_response:.0f}ms vs Mean: {response_stats['mean']:.0f}ms",
                        'recommendation': "Investigate causes of response time spikes",
                        'metric': 'response_time_p95',
                        'value': p95_response
                    })
            
            # Store insights
            for insight in insights:
                insight['timestamp'] = datetime.now(timezone.utc).isoformat()
                insight['analytics_id'] = self.analytics_id
                self._insights_history.append(insight)
            
        except Exception as e:
            self.logger.error(f"Error generating efficiency insights: {e}")
        
        return insights
    
    async def _generate_quality_insights(self) -> List[Dict[str, Any]]:
        """Generate insights about system quality and reliability."""
        insights = []
        
        try:
            # Error quality insights
            error_stats = self.metrics['error_rate'].get_stats()
            if error_stats.get('sum', 0) > 5:  # At least 5 errors
                # Analyze error patterns
                error_trend = self.metrics['error_rate'].get_trend()
                
                if error_trend == 'increasing':
                    insights.append({
                        'type': 'error_trend_negative',
                        'severity': 'high',
                        'message': f"Error rate is increasing. Total errors: {error_stats.get('sum', 0):.0f}",
                        'recommendation': "Urgent: Investigate and fix root causes of errors",
                        'metric': 'error_trend',
                        'value': error_stats.get('sum', 0)
                    })
            
            # System stability insights
            health_scores = await self._calculate_health_scores()
            system_health = health_scores.get('system_health', 0)
            
            if system_health < 70:
                insights.append({
                    'type': 'system_stability_concern',
                    'severity': 'high' if system_health < 50 else 'medium',
                    'message': f"System stability below threshold: {system_health:.0f}%",
                    'recommendation': "Review all health metrics and address critical issues",
                    'metric': 'system_health',
                    'value': system_health
                })
            
            # Pattern detection quality
            if self.enable_pattern_detection and self.patterns:
                detected_patterns = sum(1 for p in self.patterns.values() if p.last_detected and 
                                      (datetime.now(timezone.utc) - p.last_detected).total_seconds() < 3600)
                
                if detected_patterns > len(self.patterns) * 0.7:
                    insights.append({
                        'type': 'pattern_detection_active',
                        'severity': 'info',
                        'message': f"Pattern detection highly active: {detected_patterns}/{len(self.patterns)} patterns detected recently",
                        'recommendation': "System behavior patterns are being tracked effectively",
                        'metric': 'pattern_detection_rate',
                        'value': detected_patterns
                    })
            
            # Anomaly detection effectiveness
            if self.enable_anomaly_detection:
                total_anomalies = sum(d.anomalies_detected for d in self.anomaly_detectors.values())
                
                if total_anomalies > 10:
                    insights.append({
                        'type': 'high_anomaly_rate',
                        'severity': 'medium',
                        'message': f"Multiple anomalies detected: {total_anomalies} total",
                        'recommendation': "Review anomaly patterns and adjust thresholds if needed",
                        'metric': 'total_anomalies',
                        'value': total_anomalies
                    })
            
            # Store insights
            for insight in insights:
                insight['timestamp'] = datetime.now(timezone.utc).isoformat()
                insight['analytics_id'] = self.analytics_id
                self._insights_history.append(insight)
            
        except Exception as e:
            self.logger.error(f"Error generating quality insights: {e}")
        
        return insights
    
    async def _check_system_anomalies(self):
        """Check for system-wide anomalies."""
        try:
            # Check for correlated anomalies
            recent_anomalies = [
                alert for alert in self._alert_history 
                if alert.get('type') == 'anomaly' and 
                datetime.fromisoformat(alert['timestamp']) > datetime.now(timezone.utc) - timedelta(minutes=5)
            ]
            
            if len(recent_anomalies) > 3:
                # Multiple anomalies detected - possible system issue
                anomaly_types = set(a['anomaly']['metric_name'] for a in recent_anomalies)
                
                system_anomaly = {
                    'type': 'system_wide_anomaly',
                    'severity': 'critical',
                    'message': f"Multiple correlated anomalies detected: {', '.join(anomaly_types)}",
                    'anomaly_count': len(recent_anomalies),
                    'affected_metrics': list(anomaly_types),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                # Publish system anomaly event
                event = IntelligenceEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=EventType.ERROR_EVENT,
                    worker_id=self.analytics_id,
                    timestamp=datetime.now(timezone.utc),
                    priority=EventPriority.CRITICAL,
                    data={
                        'event_subtype': 'system_anomaly',
                        'anomaly': system_anomaly
                    }
                )
                await self.intelligence_hub.publish_event(event)
                
                self._alert_history.append(system_anomaly)
                self.logger.critical(f"System-wide anomaly detected: {system_anomaly['message']}")
            
            # Check for missing events (system freeze)
            throughput_stats = self.metrics['event_throughput'].get_stats()
            if throughput_stats['count'] > 100:
                # Check if events stopped coming
                recent_events = len([
                    t for t in self.metrics['event_throughput'].timestamps 
                    if t > datetime.now(timezone.utc) - timedelta(minutes=2)
                ])
                
                if recent_events == 0:
                    freeze_anomaly = {
                        'type': 'system_freeze',
                        'severity': 'critical',
                        'message': "No events received in the last 2 minutes - possible system freeze",
                        'last_event_time': self.metrics['event_throughput'].timestamps[-1].isoformat() if self.metrics['event_throughput'].timestamps else 'unknown',
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                    
                    self._alert_history.append(freeze_anomaly)
                    self.logger.critical("System freeze detected - no recent events")
            
        except Exception as e:
            self.logger.error(f"Error checking system anomalies: {e}")
    
    async def _predict_capacity_needs(self) -> Optional[Dict[str, Any]]:
        """Predict future capacity needs based on trends."""
        try:
            # Need sufficient data for predictions
            load_stats = self.metrics['system_load'].get_stats()
            worker_stats = self.metrics['worker_count'].get_stats()
            
            if load_stats['count'] < 30 or worker_stats['count'] < 10:
                return None
            
            # Analyze load trend
            load_trend = self.metrics['system_load'].get_trend(window_minutes=30)
            current_load = load_stats['mean']
            current_workers = worker_stats['mean']
            
            # Simple projection
            if load_trend == 'increasing':
                # Estimate 20% increase per hour
                projected_load_1h = current_load * 1.2
                projected_load_4h = current_load * 1.8
            elif load_trend == 'decreasing':
                # Estimate 15% decrease per hour  
                projected_load_1h = current_load * 0.85
                projected_load_4h = current_load * 0.5
            else:
                # Stable
                projected_load_1h = current_load
                projected_load_4h = current_load
            
            # Calculate required workers (target 50 events/min/worker)
            target_load_per_worker = 50
            required_workers_1h = math.ceil(projected_load_1h / target_load_per_worker)
            required_workers_4h = math.ceil(projected_load_4h / target_load_per_worker)
            
            prediction = {
                'type': 'capacity_prediction',
                'current_state': {
                    'load': current_load,
                    'workers': current_workers,
                    'load_per_worker': current_load / max(current_workers, 1)
                },
                'predictions': {
                    '1_hour': {
                        'projected_load': projected_load_1h,
                        'required_workers': required_workers_1h,
                        'worker_change': required_workers_1h - current_workers
                    },
                    '4_hours': {
                        'projected_load': projected_load_4h,
                        'required_workers': required_workers_4h,
                        'worker_change': required_workers_4h - current_workers
                    }
                },
                'trend': load_trend,
                'confidence': 0.7,  # Medium confidence for simple projection
                'recommendation': None
            }
            
            # Add recommendation
            if required_workers_1h > current_workers * 1.5:
                prediction['recommendation'] = f"Urgent: Scale up to {required_workers_1h} workers within the next hour"
                prediction['severity'] = 'high'
            elif required_workers_1h < current_workers * 0.5:
                prediction['recommendation'] = f"Consider scaling down to {required_workers_1h} workers to optimize costs"
                prediction['severity'] = 'low'
            else:
                prediction['recommendation'] = "Current capacity adequate for projected load"
                prediction['severity'] = 'info'
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting capacity needs: {e}")
            return None
    
    async def _predict_performance_trends(self) -> Optional[Dict[str, Any]]:
        """Predict performance trends based on historical data."""
        try:
            # Analyze response time trends
            response_stats = self.metrics['ai_response_time'].get_stats()
            if response_stats['count'] < 50:
                return None
            
            response_trend = self.metrics['ai_response_time'].get_trend(window_minutes=30)
            current_response_time = response_stats['mean']
            
            # Analyze error rate trends
            error_stats = self.metrics['error_rate'].get_stats()
            error_trend = self.metrics['error_rate'].get_trend() if error_stats['count'] > 10 else 'stable'
            
            # Simple trend projection
            if response_trend == 'increasing':
                projected_response_1h = current_response_time * 1.15
                projected_response_4h = current_response_time * 1.5
            elif response_trend == 'decreasing':
                projected_response_1h = current_response_time * 0.9
                projected_response_4h = current_response_time * 0.7
            else:
                projected_response_1h = current_response_time
                projected_response_4h = current_response_time
            
            prediction = {
                'type': 'performance_prediction',
                'metrics': {
                    'response_time': {
                        'current': current_response_time,
                        'trend': response_trend,
                        'projected_1h': projected_response_1h,
                        'projected_4h': projected_response_4h
                    },
                    'error_rate': {
                        'current': error_stats['sum'] if error_stats['count'] > 0 else 0,
                        'trend': error_trend
                    }
                },
                'risk_assessment': None,
                'recommendation': None
            }
            
            # Risk assessment
            if projected_response_1h > 5000:  # 5 second threshold
                prediction['risk_assessment'] = 'high'
                prediction['recommendation'] = "Performance degradation predicted. Take preventive action"
                prediction['severity'] = 'high'
            elif response_trend == 'increasing' and error_trend == 'increasing':
                prediction['risk_assessment'] = 'medium'
                prediction['recommendation'] = "Multiple negative trends detected. Monitor closely"
                prediction['severity'] = 'medium'
            else:
                prediction['risk_assessment'] = 'low'
                prediction['recommendation'] = "Performance trends within acceptable range"
                prediction['severity'] = 'info'
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting performance trends: {e}")
            return None
    
    async def _publish_insight(self, insight: Dict[str, Any]):
        """Publish an insight event."""
        try:
            # Create insight event
            event = IntelligenceEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.ANALYTICS_INSIGHT,
                worker_id=self.analytics_id,
                timestamp=datetime.now(timezone.utc),
                priority=EventPriority.HIGH if insight.get('severity') == 'high' else EventPriority.NORMAL,
                data={
                    'event_subtype': 'analytics_insight',
                    'insight': insight
                }
            )
            
            await self.intelligence_hub.publish_event(event)
            
            self.logger.info(f"Published insight: {insight['type']} - {insight['message']}")
            
        except Exception as e:
            self.logger.error(f"Error publishing insight: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    async def _publish_pattern_detection(self, pattern: Dict[str, Any]):
        """Publish a pattern detection event."""
        try:
            # Create pattern event
            event = IntelligenceEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.ANALYTICS_INSIGHT,
                worker_id=self.analytics_id,
                timestamp=datetime.now(timezone.utc),
                priority=EventPriority.NORMAL,
                data={
                    'event_subtype': 'pattern_detected',
                    'pattern': pattern
                }
            )
            
            await self.intelligence_hub.publish_event(event)
            
            self.logger.info(f"Pattern detected: {pattern['pattern_id']}")
            
        except Exception as e:
            self.logger.error(f"Error publishing pattern detection: {e}")
    
    async def _publish_prediction(self, prediction: Dict[str, Any]):
        """Publish a prediction event."""
        try:
            # Create prediction event  
            event = IntelligenceEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.ANALYTICS_INSIGHT,
                worker_id=self.analytics_id,
                timestamp=datetime.now(timezone.utc),
                priority=EventPriority.HIGH if prediction.get('severity') == 'high' else EventPriority.NORMAL,
                data={
                    'event_subtype': 'analytics_prediction',
                    'prediction': prediction
                }
            )
            
            await self.intelligence_hub.publish_event(event)
            
            self.logger.info(f"Published prediction: {prediction['type']}")
            
        except Exception as e:
            self.logger.error(f"Error publishing prediction: {e}")
    
    async def get_analytics_report(self) -> Dict[str, Any]:
        """Get comprehensive analytics report."""
        try:
            # Generate metrics summary
            metrics_summary = {}
            for name, metric in self.metrics.items():
                stats = metric.get_stats()
                stats['trend'] = metric.get_trend()
                metrics_summary[name] = stats
            
            # Get pattern analysis
            pattern_analysis = {}
            for pattern_id, pattern in self.patterns.items():
                pattern_analysis[pattern_id] = {
                    'last_detected': pattern.last_detected.isoformat() if pattern.last_detected else None,
                    'recent_occurrences': len(pattern.occurrences),
                    'event_types': [et.value for et in pattern.event_types]
                }
            
            # Get anomaly summary
            anomaly_summary = {}
            for detector_name, detector in self.anomaly_detectors.items():
                anomaly_summary[detector_name] = {
                    'anomalies_detected': detector.anomalies_detected,
                    'threshold': detector.threshold_std,
                    'samples_count': len(detector.metric.samples)
                }
            
            return {
                'analytics_id': self.analytics_id,
                'report_timestamp': datetime.now(timezone.utc).isoformat(),
                'analytics_performance': self._analytics_metrics.copy(),
                'metrics_summary': metrics_summary,
                'pattern_analysis': pattern_analysis,
                'anomaly_summary': anomaly_summary,
                'insights_count': len(self._insights_history),
                'alerts_count': len(self._alert_history),
                'capabilities': {
                    'real_time_enabled': self.enable_real_time,
                    'pattern_detection_enabled': self.enable_pattern_detection,
                    'anomaly_detection_enabled': self.enable_anomaly_detection,
                    'predictive_analytics_enabled': self.enable_predictive_analytics
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating analytics report: {e}")
            return {'error': str(e)}
    
    async def _register_analytics_engine(self):
        """Register analytics engine in distributed registry."""
        engine_data = {
            'analytics_id': self.analytics_id,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'status': 'active',
            'capabilities': {
                'real_time': self.enable_real_time,
                'pattern_detection': self.enable_pattern_detection,
                'anomaly_detection': self.enable_anomaly_detection,
                'predictive_analytics': self.enable_predictive_analytics
            },
            'metrics_tracked': len(self.metrics),
            'patterns_configured': len(self.patterns),
            'detectors_active': len(self.anomaly_detectors)
        }
        
        await self.state_manager.update(
            f"analytics_engines.{self.analytics_id}",
            engine_data,
            distributed=True
        )
    
    async def track_event(self, event_data: Dict[str, Any]) -> None:
        """Track an event for analytics.
        
        Args:
            event_data: Event data to track
        """
        try:
            # Validate event_data is a dictionary
            if not isinstance(event_data, dict):
                self.logger.error(f"Invalid event_data type: {type(event_data).__name__} - expected dict")
                return
            
            # Extract event type
            event_type = event_data.get('event_type', 'unknown')
            
            # Update event rate metric
            if event_type in self._event_rates:
                self._event_rates[event_type].add_sample(1)
            
            # Update general metrics
            # event_throughput is always initialized in constructor
            self.metrics['event_throughput'].add_sample(1)
            
            # Track worker performance if applicable
            if 'worker_id' in event_data and 'performance_score' in event_data:
                worker_id = event_data['worker_id']
                score = event_data['performance_score']
                self._worker_performance[worker_id].add_sample(score)
            
            # Track task metrics if applicable
            if 'task_id' in event_data:
                task_id = event_data['task_id']
                if 'duration' in event_data:
                    self._task_metrics[task_id].add_sample(event_data['duration'])
            
            # Note: Full event processing is handled by intelligence hub integration
            # This method is for simple metric tracking only
            
            self._analytics_metrics['events_processed'] += 1
            
        except Exception as e:
            self.logger.error(f"Error tracking event: {e}")
    
    # MCP-Redis Enhanced Analytics Methods
    async def analyze_event_patterns_ai(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Use MCP-Redis to analyze event patterns with AI insights."""
        if not self._use_mcp or not self.mcp_redis:
            # Fallback to basic pattern detection
            return {
                "patterns": [p.pattern_id for p in self.patterns.values() if p.last_detected],
                "message": "MCP-Redis not available for AI analysis"
            }
        
        try:
            analysis = await self.mcp_redis.execute(f"""
                Analyze system event patterns from the last {time_range_hours} hours:
                - Identify recurring event sequences
                - Find correlations between different event types
                - Detect unusual event frequencies or timing
                - Identify event cascades (one event triggering others)
                - Find root causes of error patterns
                - Suggest optimizations to reduce error events
                - Predict likely upcoming events based on patterns
                - Return actionable insights and recommendations
            """)
            
            return analysis if isinstance(analysis, dict) else {"analysis": analysis}
            
        except Exception as e:
            self.logger.error(f"Error in AI pattern analysis: {e}")
            return {"error": str(e)}
    
    async def predict_system_issues(self) -> List[Dict[str, Any]]:
        """Use MCP-Redis to predict potential system issues."""
        if not self._use_mcp or not self.mcp_redis:
            return []
        
        try:
            predictions = await self.mcp_redis.execute("""
                Predict potential system issues based on:
                - Current event patterns and trends
                - Historical failure patterns
                - Resource utilization trends
                - Worker performance degradation
                - Error rate acceleration
                - Anomaly detection results
                
                For each prediction provide:
                - Issue type and description
                - Probability (0-1)
                - Expected time until occurrence
                - Severity level
                - Recommended preventive actions
                - Related metrics to monitor
            """)
            
            return predictions if isinstance(predictions, list) else []
            
        except Exception as e:
            self.logger.error(f"Error predicting issues: {e}")
            return []
    
    async def get_intelligent_insights(self) -> Dict[str, Any]:
        """Get AI-powered insights about system behavior."""
        if not self._use_mcp or not self.mcp_redis:
            # Return basic metrics
            return {
                "total_events": self._analytics_metrics['events_processed'],
                "anomalies": self._analytics_metrics['anomalies_detected'],
                "message": "MCP-Redis not available for intelligent insights"
            }
        
        try:
            insights = await self.mcp_redis.execute("""
                Provide intelligent insights about the system:
                - What are the top performance bottlenecks?
                - Which workers are most/least efficient?
                - What tasks are causing the most errors?
                - Are there any concerning trends?
                - What optimizations would have the biggest impact?
                - Which event patterns indicate healthy vs unhealthy states?
                - What's the overall system health score (0-100)?
                - Provide specific, actionable recommendations
            """)
            
            return insights if isinstance(insights, dict) else {"insights": insights}
            
        except Exception as e:
            self.logger.error(f"Error getting intelligent insights: {e}")
            return {"error": str(e)}
    
    async def correlate_events_with_outcomes(self) -> Dict[str, Any]:
        """Use MCP-Redis to find correlations between events and outcomes."""
        if not self._use_mcp or not self.mcp_redis:
            return {"message": "MCP-Redis not available"}
        
        try:
            correlations = await self.mcp_redis.execute("""
                Analyze correlations between events and outcomes:
                - Which event sequences lead to task failures?
                - What events predict successful task completion?
                - Which worker behaviors correlate with high performance?
                - What event patterns precede system slowdowns?
                - Which events have the strongest impact on system efficiency?
                - Find hidden relationships between seemingly unrelated events
                - Calculate correlation coefficients for key relationships
                - Suggest event handling improvements based on correlations
            """)
            
            return correlations if isinstance(correlations, dict) else {"correlations": correlations}
            
        except Exception as e:
            self.logger.error(f"Error finding correlations: {e}")
            return {"error": str(e)}
    
    async def optimize_event_processing(self) -> Dict[str, Any]:
        """Use MCP-Redis to optimize event processing strategies."""
        if not self._use_mcp or not self.mcp_redis:
            return {"message": "MCP-Redis not available"}
        
        try:
            optimizations = await self.mcp_redis.execute("""
                Optimize event processing based on analysis:
                - Which events can be batched for efficiency?
                - What's the optimal event retention period?
                - Which events should trigger immediate alerts?
                - How to reduce event processing latency?
                - What event filters would reduce noise?
                - Suggest event prioritization strategies
                - Recommend event aggregation approaches
                - Calculate optimal buffer sizes and timeouts
            """)
            
            return optimizations if isinstance(optimizations, dict) else {"optimizations": optimizations}
            
        except Exception as e:
            self.logger.error(f"Error optimizing event processing: {e}")
            return {"error": str(e)}
    
    async def generate_analytics_report(self, report_type: str = "comprehensive") -> Dict[str, Any]:
        """Generate an AI-powered analytics report."""
        if not self._use_mcp or not self.mcp_redis:
            # Return basic report
            return {
                "report_type": report_type,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "events_processed": self._analytics_metrics['events_processed'],
                "insights_generated": self._analytics_metrics['insights_generated'],
                "message": "MCP-Redis not available for AI reporting"
            }
        
        try:
            report = await self.mcp_redis.execute(f"""
                Generate a {report_type} analytics report including:
                - Executive summary of system performance
                - Key metrics and KPIs with trends
                - Significant events and their impact
                - Performance bottlenecks and recommendations
                - Worker efficiency analysis
                - Task completion patterns
                - Error analysis and root causes
                - Predictive insights for next 24-48 hours
                - Top 5 actionable recommendations
                - Risk assessment and mitigation strategies
                
                Format as a structured report with clear sections.
            """)
            
            return report if isinstance(report, dict) else {"report": report}
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown analytics engine."""
        self.logger.info(f"Shutting down Event Analytics: {self.analytics_id}")
        self._shutdown = True
        
        # Stop analytics tasks
        for task in self._analytics_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Update engine status
        await self.state_manager.update(
            f"analytics_engines.{self.analytics_id}.status",
            'shutdown',
            distributed=True
        )
        
        self.logger.info(f"Event Analytics {self.analytics_id} shutdown complete")


# Global analytics engine instance
_global_analytics_engine: Optional[RedisEventAnalytics] = None


async def get_event_analytics(**kwargs) -> RedisEventAnalytics:
    """Get global event analytics engine instance."""
    global _global_analytics_engine
    
    if _global_analytics_engine is None:
        _global_analytics_engine = RedisEventAnalytics(**kwargs)
        await _global_analytics_engine.initialize()
    
    return _global_analytics_engine


async def create_event_analytics(**kwargs) -> RedisEventAnalytics:
    """Create new event analytics engine instance."""
    engine = RedisEventAnalytics(**kwargs)
    await engine.initialize()
    return engine