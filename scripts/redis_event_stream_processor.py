"""
Redis Event Stream Processor

Processes system events in real-time using Redis Streams for
pattern detection, anomaly detection, and intelligent insights.
"""

import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from redis_integration.redis_client import RedisClient
    from redis_integration.redis_streams_manager import RedisStreamsManager
except ImportError:
    from scripts.redis_integration.redis_client import RedisClient
    from scripts.redis_integration.redis_streams_manager import RedisStreamsManager


class EventPattern(Enum):
    """Types of patterns to detect."""
    REPEATED_FAILURE = "repeated_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    TASK_BOTTLENECK = "task_bottleneck"
    WORKER_IMBALANCE = "worker_imbalance"
    RESOURCE_SPIKE = "resource_spike"
    SUCCESS_STREAK = "success_streak"
    ANOMALY = "anomaly"


@dataclass
class DetectedPattern:
    """A detected pattern in the event stream."""
    pattern_type: EventPattern
    severity: str  # low, medium, high, critical
    description: str
    affected_components: List[str]
    metrics: Dict[str, Any]
    recommendations: List[str]
    detected_at: datetime


class RedisEventStreamProcessor:
    """Processes event streams for intelligence and insights."""
    
    def __init__(self, redis_client: RedisClient = None):
        """Initialize event stream processor.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        
        # Stream configuration
        self.event_stream = "cwmai:events:stream"
        self.pattern_stream = "cwmai:patterns:stream"
        self.consumer_group = "event_processors"
        self.consumer_name = "processor_1"
        
        # Stream manager
        self.stream_manager: Optional[RedisStreamsManager] = None
        
        # Pattern detection state
        self.event_window: List[Dict[str, Any]] = []
        self.window_size = 100  # Keep last 100 events
        self.pattern_detectors: Dict[EventPattern, Callable] = {
            EventPattern.REPEATED_FAILURE: self._detect_repeated_failures,
            EventPattern.PERFORMANCE_DEGRADATION: self._detect_performance_degradation,
            EventPattern.TASK_BOTTLENECK: self._detect_task_bottleneck,
            EventPattern.WORKER_IMBALANCE: self._detect_worker_imbalance,
            EventPattern.RESOURCE_SPIKE: self._detect_resource_spike,
            EventPattern.SUCCESS_STREAK: self._detect_success_streak,
            EventPattern.ANOMALY: self._detect_anomalies
        }
        
        # Analytics state
        self.task_metrics: Dict[str, Dict[str, Any]] = {}
        self.worker_metrics: Dict[str, Dict[str, Any]] = {}
        self.system_metrics: Dict[str, float] = {}
        
        self._initialized = False
        self._processor_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize stream processor."""
        if self._initialized:
            return
        
        try:
            # Get Redis client if not provided
            if not self.redis_client:
                from redis_integration.redis_client import get_redis_client
                self.redis_client = await get_redis_client()
            
            # Initialize stream manager
            self.stream_manager = RedisStreamsManager(self.redis_client)
            
            # Create consumer group
            await self.stream_manager.create_consumer_group(
                self.event_stream,
                self.consumer_group
            )
            
            # Start processing
            self._processor_task = asyncio.create_task(self._process_events())
            
            self._initialized = True
            self.logger.info("Event stream processor initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize event processor: {e}")
            raise
    
    async def track_event(self, event_data: Dict[str, Any]):
        """Track an event in the stream.
        
        Args:
            event_data: Event data to track
        """
        if not self._initialized:
            await self.initialize()
        
        # Add timestamp if not present
        if 'timestamp' not in event_data:
            event_data['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        # Add to stream
        await self.stream_manager.produce(self.event_stream, event_data)
    
    async def _process_events(self):
        """Main event processing loop."""
        async def process_message(message):
            """Process a single event."""
            try:
                # Extract data from message object
                data = message.data if hasattr(message, 'data') else message
                
                # Add to event window
                self.event_window.append(data)
                if len(self.event_window) > self.window_size:
                    self.event_window.pop(0)
                
                # Update metrics
                self._update_metrics(data)
                
                # Run pattern detection
                detected_patterns = await self._detect_patterns()
                
                # Publish detected patterns
                for pattern in detected_patterns:
                    await self._publish_pattern(pattern)
                    
            except Exception as e:
                self.logger.error(f"Error processing event: {e}")
        
        # Start consumer
        await self.stream_manager.start_consumer(
            self.event_stream,
            self.consumer_group,
            self.consumer_name,
            process_message
        )
    
    def _update_metrics(self, event: Dict[str, Any]):
        """Update metrics based on event."""
        event_type = event.get('event_type')
        
        if event_type == 'task_completed':
            task_type = event.get('task_type', 'unknown')
            worker_id = event.get('worker_id', 'unknown')
            completion_time = event.get('completion_time', 0)
            success = event.get('success', False)
            
            # Update task metrics
            if task_type not in self.task_metrics:
                self.task_metrics[task_type] = {
                    'total': 0,
                    'success': 0,
                    'failed': 0,
                    'avg_time': 0,
                    'total_time': 0
                }
            
            metrics = self.task_metrics[task_type]
            metrics['total'] += 1
            if success:
                metrics['success'] += 1
            else:
                metrics['failed'] += 1
            
            metrics['total_time'] += completion_time
            metrics['avg_time'] = metrics['total_time'] / metrics['total']
            
            # Update worker metrics
            if worker_id not in self.worker_metrics:
                self.worker_metrics[worker_id] = {
                    'tasks_completed': 0,
                    'tasks_failed': 0,
                    'total_time': 0,
                    'last_task': None
                }
            
            worker = self.worker_metrics[worker_id]
            if success:
                worker['tasks_completed'] += 1
            else:
                worker['tasks_failed'] += 1
            worker['total_time'] += completion_time
            worker['last_task'] = datetime.now(timezone.utc)
    
    async def _detect_patterns(self) -> List[DetectedPattern]:
        """Run all pattern detectors."""
        patterns = []
        
        for pattern_type, detector in self.pattern_detectors.items():
            try:
                detected = await detector()
                if detected:
                    patterns.append(detected)
            except Exception as e:
                self.logger.error(f"Error in pattern detector {pattern_type}: {e}")
        
        return patterns
    
    async def _detect_repeated_failures(self) -> Optional[DetectedPattern]:
        """Detect repeated task failures."""
        # Count recent failures
        recent_events = self.event_window[-20:]  # Last 20 events
        failures = [e for e in recent_events if e.get('event_type') == 'task_completed' and not e.get('success', True)]
        
        if len(failures) >= 5:  # 5 or more failures in last 20 events
            affected_tasks = list(set(f.get('task_type', 'unknown') for f in failures))
            affected_workers = list(set(f.get('worker_id', 'unknown') for f in failures))
            
            return DetectedPattern(
                pattern_type=EventPattern.REPEATED_FAILURE,
                severity='high',
                description=f"Detected {len(failures)} task failures in recent events",
                affected_components=affected_workers,
                metrics={
                    'failure_count': len(failures),
                    'failure_rate': len(failures) / len(recent_events),
                    'affected_task_types': affected_tasks
                },
                recommendations=[
                    "Review error logs for common failure patterns",
                    "Check resource availability",
                    "Consider reducing task complexity",
                    "Verify external service dependencies"
                ],
                detected_at=datetime.now(timezone.utc)
            )
        
        return None
    
    async def _detect_performance_degradation(self) -> Optional[DetectedPattern]:
        """Detect performance degradation."""
        # Compare recent vs historical performance
        if not self.task_metrics:
            return None
        
        degraded_tasks = []
        
        for task_type, metrics in self.task_metrics.items():
            if metrics['total'] < 10:  # Need enough data
                continue
            
            # Check if average time is increasing
            recent_completions = [
                e for e in self.event_window[-30:]
                if e.get('event_type') == 'task_completed' and 
                e.get('task_type') == task_type and
                e.get('success', False)
            ]
            
            if len(recent_completions) >= 5:
                recent_avg = sum(e.get('completion_time', 0) for e in recent_completions) / len(recent_completions)
                overall_avg = metrics['avg_time']
                
                if recent_avg > overall_avg * 1.5:  # 50% slower
                    degraded_tasks.append({
                        'task_type': task_type,
                        'recent_avg': recent_avg,
                        'overall_avg': overall_avg,
                        'degradation': (recent_avg - overall_avg) / overall_avg
                    })
        
        if degraded_tasks:
            return DetectedPattern(
                pattern_type=EventPattern.PERFORMANCE_DEGRADATION,
                severity='medium',
                description=f"Performance degradation detected in {len(degraded_tasks)} task types",
                affected_components=[t['task_type'] for t in degraded_tasks],
                metrics={
                    'degraded_tasks': degraded_tasks,
                    'max_degradation': max(t['degradation'] for t in degraded_tasks)
                },
                recommendations=[
                    "Profile slow tasks to identify bottlenecks",
                    "Check for increased data volumes",
                    "Review recent code changes",
                    "Monitor system resources"
                ],
                detected_at=datetime.now(timezone.utc)
            )
        
        return None
    
    async def _detect_task_bottleneck(self) -> Optional[DetectedPattern]:
        """Detect task processing bottlenecks."""
        # Check for tasks taking significantly longer than average
        bottlenecks = []
        
        for task_type, metrics in self.task_metrics.items():
            if metrics['total'] < 5:
                continue
            
            avg_time = metrics['avg_time']
            
            # Find outliers
            recent_tasks = [
                e for e in self.event_window
                if e.get('event_type') == 'task_completed' and
                e.get('task_type') == task_type
            ]
            
            outliers = [
                t for t in recent_tasks
                if t.get('completion_time', 0) > avg_time * 3  # 3x average
            ]
            
            if len(outliers) >= 2:
                bottlenecks.append({
                    'task_type': task_type,
                    'outlier_count': len(outliers),
                    'max_time': max(t.get('completion_time', 0) for t in outliers),
                    'avg_time': avg_time
                })
        
        if bottlenecks:
            return DetectedPattern(
                pattern_type=EventPattern.TASK_BOTTLENECK,
                severity='medium',
                description=f"Task bottlenecks detected in {len(bottlenecks)} task types",
                affected_components=[b['task_type'] for b in bottlenecks],
                metrics={'bottlenecks': bottlenecks},
                recommendations=[
                    "Break down complex tasks into smaller units",
                    "Implement task timeouts",
                    "Add progress tracking for long-running tasks",
                    "Consider parallel processing"
                ],
                detected_at=datetime.now(timezone.utc)
            )
        
        return None
    
    async def _detect_worker_imbalance(self) -> Optional[DetectedPattern]:
        """Detect imbalanced work distribution."""
        if len(self.worker_metrics) < 2:
            return None
        
        # Calculate work distribution
        worker_loads = {
            worker_id: metrics['tasks_completed'] + metrics['tasks_failed']
            for worker_id, metrics in self.worker_metrics.items()
        }
        
        if not worker_loads:
            return None
        
        avg_load = sum(worker_loads.values()) / len(worker_loads)
        max_load = max(worker_loads.values())
        min_load = min(worker_loads.values())
        
        # Check for significant imbalance
        if max_load > avg_load * 2 and min_load < avg_load * 0.5:
            return DetectedPattern(
                pattern_type=EventPattern.WORKER_IMBALANCE,
                severity='medium',
                description="Significant worker load imbalance detected",
                affected_components=list(worker_loads.keys()),
                metrics={
                    'worker_loads': worker_loads,
                    'max_load': max_load,
                    'min_load': min_load,
                    'avg_load': avg_load,
                    'imbalance_ratio': max_load / max(min_load, 1)
                },
                recommendations=[
                    "Review task assignment algorithm",
                    "Check worker specializations",
                    "Redistribute work queue",
                    "Scale workers based on load"
                ],
                detected_at=datetime.now(timezone.utc)
            )
        
        return None
    
    async def _detect_resource_spike(self) -> Optional[DetectedPattern]:
        """Detect resource usage spikes."""
        # This would integrate with resource monitoring
        # For now, we'll check for rapid task creation
        
        recent_creations = [
            e for e in self.event_window[-50:]
            if e.get('event_type') == 'task_created'
        ]
        
        if len(recent_creations) > 30:  # More than 30 tasks in last 50 events
            return DetectedPattern(
                pattern_type=EventPattern.RESOURCE_SPIKE,
                severity='high',
                description=f"Rapid task creation detected: {len(recent_creations)} tasks",
                affected_components=['task_queue'],
                metrics={
                    'creation_rate': len(recent_creations) / 50,
                    'task_count': len(recent_creations)
                },
                recommendations=[
                    "Review task generation logic",
                    "Implement rate limiting",
                    "Check for infinite loops",
                    "Monitor system resources"
                ],
                detected_at=datetime.now(timezone.utc)
            )
        
        return None
    
    async def _detect_success_streak(self) -> Optional[DetectedPattern]:
        """Detect success streaks for positive reinforcement."""
        recent_completions = [
            e for e in self.event_window[-20:]
            if e.get('event_type') == 'task_completed'
        ]
        
        if not recent_completions:
            return None
        
        success_count = sum(1 for e in recent_completions if e.get('success', False))
        success_rate = success_count / len(recent_completions)
        
        if success_rate >= 0.95 and len(recent_completions) >= 10:
            return DetectedPattern(
                pattern_type=EventPattern.SUCCESS_STREAK,
                severity='low',
                description=f"Excellent performance: {success_rate:.0%} success rate",
                affected_components=[],
                metrics={
                    'success_rate': success_rate,
                    'task_count': len(recent_completions),
                    'success_count': success_count
                },
                recommendations=[
                    "Current configuration is working well",
                    "Consider documenting current best practices",
                    "Monitor for sustained performance"
                ],
                detected_at=datetime.now(timezone.utc)
            )
        
        return None
    
    async def _detect_anomalies(self) -> Optional[DetectedPattern]:
        """Detect anomalous patterns using statistical methods."""
        # Simple anomaly detection based on event frequency
        event_types = {}
        
        for event in self.event_window:
            event_type = event.get('event_type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        # Check for unusual event type distribution
        # This is a simplified approach - could use more sophisticated methods
        anomalies = []
        
        total_events = len(self.event_window)
        for event_type, count in event_types.items():
            frequency = count / total_events
            
            # Flag if any event type is more than 70% of all events
            if frequency > 0.7:
                anomalies.append({
                    'event_type': event_type,
                    'frequency': frequency,
                    'count': count
                })
        
        if anomalies:
            return DetectedPattern(
                pattern_type=EventPattern.ANOMALY,
                severity='medium',
                description="Unusual event distribution detected",
                affected_components=['event_stream'],
                metrics={'anomalies': anomalies},
                recommendations=[
                    "Investigate dominant event types",
                    "Check for stuck processes",
                    "Review event generation logic"
                ],
                detected_at=datetime.now(timezone.utc)
            )
        
        return None
    
    async def _publish_pattern(self, pattern: DetectedPattern):
        """Publish detected pattern to pattern stream."""
        pattern_data = {
            'pattern_type': pattern.pattern_type.value,
            'severity': pattern.severity,
            'description': pattern.description,
            'affected_components': pattern.affected_components,
            'metrics': pattern.metrics,
            'recommendations': pattern.recommendations,
            'detected_at': pattern.detected_at.isoformat()
        }
        
        await self.stream_manager.produce(self.pattern_stream, pattern_data)
        
        self.logger.info(
            f"Pattern detected: {pattern.pattern_type.value} "
            f"(severity: {pattern.severity}) - {pattern.description}"
        )
    
    async def get_recent_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently detected patterns."""
        patterns = []
        
        # Read from pattern stream
        messages = await self.redis_client.xrevrange(
            self.pattern_stream,
            count=limit
        )
        
        for msg_id, data in messages:
            pattern_data = {k.decode(): v.decode() if isinstance(v, bytes) else v for k, v in data.items()}
            pattern_data['id'] = msg_id.decode()
            patterns.append(pattern_data)
        
        return patterns
    
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary."""
        return {
            'task_metrics': self.task_metrics,
            'worker_metrics': self.worker_metrics,
            'event_window_size': len(self.event_window),
            'patterns_detected': await self.get_recent_patterns(5)
        }
    
    async def cleanup(self):
        """Clean up resources."""
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        if self.stream_manager:
            await self.stream_manager.stop_consumer(
                self.event_stream,
                self.consumer_group,
                self.consumer_name
            )
        
        self._initialized = False
        self.logger.info("Event stream processor cleaned up")