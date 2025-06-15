"""
Redis-powered Intelligence Hub (Lock-Free Version)

High-performance event processing hub using Redis Streams for distributed
worker intelligence coordination, real-time analytics, and event sourcing.

This version implements a lock-free approach with:
- Partitioned worker states by worker ID to avoid contention
- Atomic Redis operations for all counters and metrics
- Redis Streams for event coordination instead of state locks
- TTL-based automatic cleanup of stale data
- No distributed locks - all operations are atomic or eventually consistent
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Callable, Union, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum

from scripts.redis_integration.redis_client import get_redis_client
from scripts.redis_integration.redis_streams_manager import RedisStreamsManager
from scripts.redis_lockfree_adapter import create_lockfree_state_manager


class EventType(Enum):
    """Intelligence Hub event types."""
    WORKER_REGISTRATION = "worker_registration"
    WORKER_HEARTBEAT = "worker_heartbeat"
    WORKER_SHUTDOWN = "worker_shutdown"
    TASK_ASSIGNMENT = "task_assignment"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETION = "task_completion"
    TASK_FAILURE = "task_failure"
    AI_REQUEST = "ai_request"
    AI_RESPONSE = "ai_response"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_EVENT = "error_event"
    INTELLIGENCE_UPDATE = "intelligence_update"
    CAPABILITY_DISCOVERY = "capability_discovery"
    COORDINATION_EVENT = "coordination_event"
    ANALYTICS_INSIGHT = "analytics_insight"


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class IntelligenceEvent:
    """Structured intelligence event."""
    event_id: str
    event_type: EventType
    worker_id: str
    timestamp: datetime
    priority: EventPriority
    data: Dict[str, Any]
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for Redis storage."""
        result = {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'worker_id': self.worker_id,
            'timestamp': self.timestamp.isoformat(),
            'priority': str(self.priority.value),
            'data': json.dumps(self.data),
            'retry_count': str(self.retry_count)
        }
        
        # Only include optional fields if they have values
        if self.correlation_id is not None:
            result['correlation_id'] = self.correlation_id
        if self.parent_event_id is not None:
            result['parent_event_id'] = self.parent_event_id
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntelligenceEvent':
        """Create event from dictionary."""
        # Decode bytes to strings if necessary (Redis returns bytes)
        decoded_data = {}
        for key, value in data.items():
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            decoded_data[key] = value
        
        # Parse JSON data field if it's a string
        event_data = decoded_data.get('data', '{}')
        if isinstance(event_data, str):
            event_data = json.loads(event_data)
        
        return cls(
            event_id=decoded_data['event_id'],
            event_type=EventType(decoded_data['event_type']),
            worker_id=decoded_data['worker_id'],
            timestamp=datetime.fromisoformat(decoded_data['timestamp']),
            priority=EventPriority(int(decoded_data['priority'])),
            data=event_data,
            correlation_id=decoded_data.get('correlation_id'),
            parent_event_id=decoded_data.get('parent_event_id'),
            retry_count=int(decoded_data.get('retry_count', 0))
        )


class RedisIntelligenceHub:
    """Redis-powered Intelligence Hub for distributed worker coordination."""
    
    def __init__(self, 
                 hub_id: str = None,
                 max_stream_length: int = 100000,
                 consumer_group: str = "intelligence_processors",
                 enable_analytics: bool = True,
                 enable_event_sourcing: bool = True):
        """Initialize Redis Intelligence Hub.
        
        Args:
            hub_id: Unique hub identifier
            max_stream_length: Maximum events in each stream
            consumer_group: Redis consumer group name
            enable_analytics: Enable real-time analytics
            enable_event_sourcing: Enable event sourcing patterns
        """
        self.hub_id = hub_id or f"intelligence_hub_{uuid.uuid4().hex[:8]}"
        self.max_stream_length = max_stream_length
        self.consumer_group = consumer_group
        self.enable_analytics = enable_analytics
        self.enable_event_sourcing = enable_event_sourcing
        
        self.logger = logging.getLogger(f"{__name__}.RedisIntelligenceHub")
        
        # Redis components
        self.redis_client = None
        self.streams_manager: Optional[RedisStreamsManager] = None
        self.state_manager = None
        
        # Stream configurations
        self.streams = {
            'worker_events': f'intelligence:workers:{self.hub_id}',
            'task_events': f'intelligence:tasks:{self.hub_id}',
            'ai_events': f'intelligence:ai:{self.hub_id}',
            'performance_events': f'intelligence:performance:{self.hub_id}',
            'coordination_events': f'intelligence:coordination:{self.hub_id}',
            'analytics_events': f'intelligence:analytics:{self.hub_id}',
        }
        
        # Event processors
        self._event_processors: Dict[EventType, List[Callable]] = {}
        self._consumer_tasks: List[asyncio.Task] = []
        self._analytics_tasks: List[asyncio.Task] = []
        
        # Hub state
        self._workers_registry: Dict[str, Dict[str, Any]] = {}
        self._task_registry: Dict[str, Dict[str, Any]] = {}
        self._performance_metrics: Dict[str, Any] = {}
        self._shutdown = False
        
        # Processing statistics
        self._processing_stats = {
            'events_processed': 0,
            'events_failed': 0,
            'processing_time_ms': 0.0,
            'last_processed': None,
            'throughput_per_second': 0.0
        }
    
    async def initialize(self):
        """Initialize Intelligence Hub components."""
        try:
            self.logger.info(f"Initializing Redis Intelligence Hub: {self.hub_id}")
            
            # Initialize Redis components
            self.redis_client = await get_redis_client()
            self.streams_manager = RedisStreamsManager(self.redis_client)
            
            # Initialize lock-free state manager with unique component ID
            self.state_manager = create_lockfree_state_manager(f"intelligence_hub_{self.hub_id}")
            await self.state_manager.initialize()
            
            # Initialize Redis streams
            await self._initialize_streams()
            
            # Start event processing
            await self._start_event_processors()
            
            # Start analytics if enabled
            if self.enable_analytics:
                await self._start_analytics_processors()
            
            # Register hub in distributed registry
            await self._register_hub()
            
            self.logger.info(f"Intelligence Hub {self.hub_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Intelligence Hub: {e}")
            raise
    
    async def _initialize_streams(self):
        """Initialize Redis streams and consumer groups."""
        try:
            for stream_name, stream_key in self.streams.items():
                # Create stream with first dummy entry if it doesn't exist
                try:
                    await self.redis_client.xadd(stream_key, {'init': 'true'}, maxlen=self.max_stream_length)
                except Exception:
                    pass  # Stream might already exist
                
                # Create consumer group
                try:
                    await self.redis_client.xgroup_create(
                        stream_key, 
                        self.consumer_group, 
                        id='0', 
                        mkstream=True
                    )
                except Exception:
                    pass  # Consumer group might already exist
                
                self.logger.debug(f"Initialized stream: {stream_key}")
            
        except Exception as e:
            self.logger.error(f"Error initializing streams: {e}")
            raise
    
    async def _start_event_processors(self):
        """Start event processing consumers."""
        try:
            # Start consumer for each stream type
            for stream_name, stream_key in self.streams.items():
                consumer_id = f"{self.hub_id}_{stream_name}"
                
                # Create consumer task
                consumer_task = asyncio.create_task(
                    self._stream_consumer(stream_key, consumer_id, stream_name)
                )
                self._consumer_tasks.append(consumer_task)
                
                self.logger.debug(f"Started consumer for {stream_name}: {consumer_id}")
            
        except Exception as e:
            self.logger.error(f"Error starting event processors: {e}")
            raise
    
    async def _start_analytics_processors(self):
        """Start real-time analytics processors."""
        try:
            # Real-time metrics aggregator
            analytics_task = asyncio.create_task(self._analytics_aggregator())
            self._analytics_tasks.append(analytics_task)
            
            # Performance insights generator
            insights_task = asyncio.create_task(self._insights_generator())
            self._analytics_tasks.append(insights_task)
            
            # Anomaly detector
            anomaly_task = asyncio.create_task(self._anomaly_detector())
            self._analytics_tasks.append(anomaly_task)
            
            self.logger.info("Analytics processors started")
            
        except Exception as e:
            self.logger.error(f"Error starting analytics processors: {e}")
            raise
    
    async def publish_event(self, event: IntelligenceEvent) -> str:
        """Publish event to appropriate Redis stream.
        
        Args:
            event: Intelligence event to publish
            
        Returns:
            Event ID from Redis stream
        """
        try:
            # Determine target stream
            stream_key = self._get_stream_for_event(event)
            
            # Add timestamp if not set
            if not hasattr(event, 'timestamp') or not event.timestamp:
                event.timestamp = datetime.now(timezone.utc)
            
            # Publish to Redis stream
            stream_id = await self.redis_client.xadd(
                stream_key,
                event.to_dict(),
                maxlen=self.max_stream_length
            )
            
            # Update analytics if enabled
            if self.enable_analytics:
                await self._update_publish_analytics(event, stream_key, stream_id)
            
            self.logger.debug(f"Published event {event.event_id} to {stream_key}: {stream_id}")
            return stream_id
            
        except Exception as e:
            self.logger.error(f"Error publishing event {event.event_id}: {e}")
            raise
    
    async def _stream_consumer(self, stream_key: str, consumer_id: str, stream_name: str):
        """Redis stream consumer for processing events."""
        while not self._shutdown:
            try:
                # Read from stream with consumer group
                messages = await self.redis_client.xreadgroup(
                    self.consumer_group,
                    consumer_id,
                    {stream_key: '>'},
                    count=10,
                    block=1000  # 1 second timeout
                )
                
                if not messages:
                    continue
                
                # Process messages
                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        try:
                            await self._process_stream_message(stream_key, msg_id, fields, stream_name)
                            
                            # Acknowledge message
                            await self.redis_client.xack(stream_key, self.consumer_group, msg_id)
                            
                        except Exception as e:
                            self.logger.error(f"Error processing message {msg_id}: {e}")
                            # Don't acknowledge failed messages for retry
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in stream consumer {consumer_id}: {e}")
                await asyncio.sleep(1)  # Brief delay before retry
    
    async def _process_stream_message(self, stream_key: str, msg_id: str, fields: Dict[str, Any], stream_name: str):
        """Process individual stream message."""
        start_time = time.time()
        
        try:
            # Check if this is an initialization message
            if fields.get('init') == 'true':
                self.logger.debug(f"Skipping initialization message in {stream_name}")
                return
            
            # Validate required fields before processing
            required_fields = ['event_id', 'event_type', 'worker_id', 'timestamp', 'priority', 'data']
            missing_fields = [field for field in required_fields if field not in fields]
            
            if missing_fields:
                self.logger.warning(f"Message {msg_id} missing required fields: {missing_fields}. Fields: {list(fields.keys())}")
                self._processing_stats['events_failed'] += 1
                return
            
            # Convert Redis fields to IntelligenceEvent
            event = IntelligenceEvent.from_dict(fields)
            
            # Update processing statistics
            self._processing_stats['events_processed'] += 1
            self._processing_stats['last_processed'] = datetime.now(timezone.utc)
            
            # Route event to appropriate processors
            await self._route_event_to_processors(event, stream_name)
            
            # Update state if event sourcing is enabled
            if self.enable_event_sourcing:
                await self._update_event_sourcing_state(event)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            self._processing_stats['processing_time_ms'] += processing_time
            
            self.logger.debug(f"Processed event {event.event_id} in {processing_time:.2f}ms")
            
        except KeyError as e:
            self._processing_stats['events_failed'] += 1
            self.logger.error(f"KeyError processing stream message: {e}. Message fields: {list(fields.keys())}")
        except Exception as e:
            self._processing_stats['events_failed'] += 1
            self.logger.error(f"Error processing stream message: {e}")
            raise
    
    async def _route_event_to_processors(self, event: IntelligenceEvent, stream_name: str):
        """Route events to registered processors."""
        try:
            # Get processors for this event type
            processors = self._event_processors.get(event.event_type, [])
            
            # Execute all processors
            for processor in processors:
                try:
                    if asyncio.iscoroutinefunction(processor):
                        await processor(event, stream_name)
                    else:
                        processor(event, stream_name)
                except Exception as e:
                    self.logger.error(f"Error in event processor: {e}")
            
            # Handle built-in event types
            await self._handle_builtin_event(event)
            
        except Exception as e:
            self.logger.error(f"Error routing event to processors: {e}")
            raise
    
    async def _handle_builtin_event(self, event: IntelligenceEvent):
        """Handle built-in event types."""
        try:
            if event.event_type == EventType.WORKER_REGISTRATION:
                await self._handle_worker_registration(event)
            
            elif event.event_type == EventType.WORKER_HEARTBEAT:
                await self._handle_worker_heartbeat(event)
                
            elif event.event_type == EventType.WORKER_SHUTDOWN:
                await self._handle_worker_shutdown(event)
                
            elif event.event_type == EventType.TASK_ASSIGNMENT:
                await self._handle_task_assignment(event)
                
            elif event.event_type == EventType.TASK_COMPLETION:
                await self._handle_task_completion(event)
                
            elif event.event_type == EventType.PERFORMANCE_METRIC:
                await self._handle_performance_metric(event)
                
            elif event.event_type == EventType.AI_REQUEST:
                await self._handle_ai_request(event)
                
            elif event.event_type == EventType.AI_RESPONSE:
                await self._handle_ai_response(event)
                
        except Exception as e:
            self.logger.error(f"Error handling built-in event: {e}")
    
    async def _handle_worker_registration(self, event: IntelligenceEvent):
        """Handle worker registration event."""
        worker_data = {
            'worker_id': event.worker_id,
            'registration_time': event.timestamp.isoformat(),
            'capabilities': event.data.get('capabilities', []),
            'status': 'active',
            'last_heartbeat': event.timestamp.isoformat(),
            'performance_metrics': {},
            'task_history': []
        }
        
        self._workers_registry[event.worker_id] = worker_data
        
        # Update distributed state
        await self.state_manager.update(
            f"intelligence.workers.{event.worker_id}",
            worker_data,
            distributed=True
        )
        
        self.logger.info(f"Registered worker: {event.worker_id}")
    
    async def _handle_worker_heartbeat(self, event: IntelligenceEvent):
        """Handle worker heartbeat event."""
        if event.worker_id in self._workers_registry:
            self._workers_registry[event.worker_id]['last_heartbeat'] = event.timestamp.isoformat()
            self._workers_registry[event.worker_id]['status'] = 'active'
            
            # Update performance metrics if provided
            if 'performance' in event.data:
                self._workers_registry[event.worker_id]['performance_metrics'] = event.data['performance']
            
            # Update distributed state
            await self.state_manager.update(
                f"intelligence.workers.{event.worker_id}.last_heartbeat",
                event.timestamp.isoformat(),
                distributed=True
            )
    
    async def _handle_worker_shutdown(self, event: IntelligenceEvent):
        """Handle worker shutdown event."""
        if event.worker_id in self._workers_registry:
            self._workers_registry[event.worker_id]['status'] = 'shutdown'
            self._workers_registry[event.worker_id]['shutdown_time'] = event.timestamp.isoformat()
            
            # Update distributed state
            await self.state_manager.update(
                f"intelligence.workers.{event.worker_id}.status",
                'shutdown',
                distributed=True
            )
            
            # Mark any assigned tasks as needing reassignment
            for task_id, task_data in self._task_registry.items():
                if task_data.get('worker_id') == event.worker_id and task_data.get('status') == 'assigned':
                    task_data['status'] = 'pending'
                    task_data['needs_reassignment'] = True
                    
                    # Publish task reassignment event
                    reassign_event = IntelligenceEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=EventType.TASK_FAILURE,
                        worker_id=event.worker_id,
                        timestamp=datetime.now(timezone.utc),
                        priority=EventPriority.HIGH,
                        data={
                            'task_id': task_id,
                            'reason': 'worker_shutdown',
                            'original_worker': event.worker_id
                        }
                    )
                    await self.publish_event(reassign_event)
            
            self.logger.info(f"Worker {event.worker_id} shutdown handled")
    
    async def _handle_task_assignment(self, event: IntelligenceEvent):
        """Handle task assignment event."""
        task_data = {
            'task_id': event.data.get('task_id'),
            'worker_id': event.worker_id,
            'assigned_time': event.timestamp.isoformat(),
            'task_type': event.data.get('task_type'),
            'priority': event.priority.value,
            'status': 'assigned',
            'progress': 0.0
        }
        
        if task_data['task_id']:
            self._task_registry[task_data['task_id']] = task_data
            
            # Update worker task history
            if event.worker_id in self._workers_registry:
                self._workers_registry[event.worker_id]['task_history'].append(task_data['task_id'])
        
        self.logger.debug(f"Assigned task {task_data['task_id']} to worker {event.worker_id}")
    
    async def _handle_task_completion(self, event: IntelligenceEvent):
        """Handle task completion event."""
        task_id = event.data.get('task_id')
        if task_id and task_id in self._task_registry:
            self._task_registry[task_id].update({
                'status': 'completed',
                'completion_time': event.timestamp.isoformat(),
                'progress': 100.0,
                'result': event.data.get('result'),
                'duration_seconds': event.data.get('duration_seconds', 0)
            })
            
            # Update performance metrics
            await self._update_task_performance_metrics(event)
        
        self.logger.debug(f"Completed task {task_id} by worker {event.worker_id}")
    
    async def _handle_performance_metric(self, event: IntelligenceEvent):
        """Handle performance metric event."""
        metric_type = event.data.get('metric_type')
        metric_value = event.data.get('value')
        
        if metric_type and metric_value is not None:
            metric_key = f"{event.worker_id}.{metric_type}"
            
            if metric_key not in self._performance_metrics:
                self._performance_metrics[metric_key] = []
            
            self._performance_metrics[metric_key].append({
                'timestamp': event.timestamp.isoformat(),
                'value': metric_value,
                'metadata': event.data.get('metadata', {})
            })
            
            # Keep only recent metrics (last 1000 entries)
            if len(self._performance_metrics[metric_key]) > 1000:
                self._performance_metrics[metric_key] = self._performance_metrics[metric_key][-1000:]
    
    async def _handle_ai_request(self, event: IntelligenceEvent):
        """Handle AI request event."""
        # Track AI request patterns for optimization
        request_data = {
            'worker_id': event.worker_id,
            'request_time': event.timestamp.isoformat(),
            'provider': event.data.get('provider'),
            'model': event.data.get('model'),
            'prompt_length': len(event.data.get('prompt', '')),
            'request_type': event.data.get('request_type', 'unknown')
        }
        
        await self.state_manager.update(
            f"intelligence.ai_requests.{event.event_id}",
            request_data,
            distributed=True
        )
    
    async def _handle_ai_response(self, event: IntelligenceEvent):
        """Handle AI response event."""
        # Track AI response patterns and performance
        response_data = {
            'worker_id': event.worker_id,
            'response_time': event.timestamp.isoformat(),
            'duration_ms': event.data.get('duration_ms', 0),
            'success': event.data.get('success', False),
            'provider': event.data.get('provider'),
            'model': event.data.get('model'),
            'token_usage': event.data.get('token_usage', {}),
            'cost_estimate': event.data.get('cost_estimate', 0.0)
        }
        
        await self.state_manager.update(
            f"intelligence.ai_responses.{event.event_id}",
            response_data,
            distributed=True
        )
    
    async def register_worker(self, worker_id: str, capabilities: List[str]) -> str:
        """Register a new worker with the intelligence hub.
        
        Args:
            worker_id: Unique worker identifier
            capabilities: List of worker capabilities
            
        Returns:
            Event ID for the registration
        """
        event = IntelligenceEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.WORKER_REGISTRATION,
            worker_id=worker_id,
            timestamp=datetime.now(timezone.utc),
            priority=EventPriority.NORMAL,
            data={'capabilities': capabilities}
        )
        
        return await self.publish_event(event)
    
    async def worker_heartbeat(self, worker_id: str, performance_data: Dict[str, Any] = None) -> str:
        """Send worker heartbeat to intelligence hub.
        
        Args:
            worker_id: Worker identifier
            performance_data: Optional performance metrics
            
        Returns:
            Event ID for the heartbeat
        """
        event_data = {}
        if performance_data:
            event_data['performance'] = performance_data
        
        event = IntelligenceEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.WORKER_HEARTBEAT,
            worker_id=worker_id,
            timestamp=datetime.now(timezone.utc),
            priority=EventPriority.LOW,
            data=event_data
        )
        
        return await self.publish_event(event)
    
    async def assign_task(self, worker_id: str, task_id: str, task_type: str, 
                         task_data: Dict[str, Any], priority: EventPriority = EventPriority.NORMAL) -> str:
        """Assign task to worker.
        
        Args:
            worker_id: Target worker identifier
            task_id: Unique task identifier
            task_type: Type of task
            task_data: Task data and parameters
            priority: Task priority
            
        Returns:
            Event ID for the assignment
        """
        event = IntelligenceEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.TASK_ASSIGNMENT,
            worker_id=worker_id,
            timestamp=datetime.now(timezone.utc),
            priority=priority,
            data={
                'task_id': task_id,
                'task_type': task_type,
                **task_data
            }
        )
        
        return await self.publish_event(event)
    
    async def report_task_completion(self, worker_id: str, task_id: str, 
                                   result: Dict[str, Any], duration_seconds: float) -> str:
        """Report task completion.
        
        Args:
            worker_id: Worker identifier
            task_id: Completed task identifier
            result: Task result data
            duration_seconds: Task execution time
            
        Returns:
            Event ID for the completion report
        """
        event = IntelligenceEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.TASK_COMPLETION,
            worker_id=worker_id,
            timestamp=datetime.now(timezone.utc),
            priority=EventPriority.NORMAL,
            data={
                'task_id': task_id,
                'result': result,
                'duration_seconds': duration_seconds
            }
        )
        
        return await self.publish_event(event)
    
    async def report_performance_metric(self, worker_id: str, metric_type: str, 
                                      value: Union[int, float], metadata: Dict[str, Any] = None) -> str:
        """Report performance metric.
        
        Args:
            worker_id: Worker identifier
            metric_type: Type of metric (e.g., 'cpu_usage', 'memory_usage')
            value: Metric value
            metadata: Additional metric metadata
            
        Returns:
            Event ID for the metric report
        """
        event = IntelligenceEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.PERFORMANCE_METRIC,
            worker_id=worker_id,
            timestamp=datetime.now(timezone.utc),
            priority=EventPriority.LOW,
            data={
                'metric_type': metric_type,
                'value': value,
                'metadata': metadata or {}
            }
        )
        
        return await self.publish_event(event)
    
    def register_event_processor(self, event_type: EventType, processor: Callable):
        """Register custom event processor.
        
        Args:
            event_type: Event type to process
            processor: Processing function (sync or async)
        """
        if event_type not in self._event_processors:
            self._event_processors[event_type] = []
        
        self._event_processors[event_type].append(processor)
        self.logger.info(f"Registered processor for {event_type.value}")
    
    async def get_worker_registry(self) -> Dict[str, Dict[str, Any]]:
        """Get current worker registry."""
        return self._workers_registry.copy()
    
    async def get_task_registry(self) -> Dict[str, Dict[str, Any]]:
        """Get current task registry."""
        return self._task_registry.copy()
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self._performance_metrics.copy()
    
    async def get_hub_statistics(self) -> Dict[str, Any]:
        """Get comprehensive hub statistics."""
        active_workers = len([w for w in self._workers_registry.values() if w.get('status') == 'active'])
        
        # Calculate throughput
        if self._processing_stats['events_processed'] > 0:
            if self._processing_stats['last_processed']:
                time_diff = (datetime.now(timezone.utc) - self._processing_stats['last_processed']).total_seconds()
                if time_diff > 0:
                    self._processing_stats['throughput_per_second'] = self._processing_stats['events_processed'] / time_diff
        
        return {
            'hub_id': self.hub_id,
            'active_workers': active_workers,
            'total_workers': len(self._workers_registry),
            'active_tasks': len([t for t in self._task_registry.values() if t.get('status') in ['assigned', 'in_progress']]),
            'completed_tasks': len([t for t in self._task_registry.values() if t.get('status') == 'completed']),
            'processing_stats': self._processing_stats.copy(),
            'streams_configured': len(self.streams),
            'analytics_enabled': self.enable_analytics,
            'event_sourcing_enabled': self.enable_event_sourcing
        }
    
    def _get_stream_for_event(self, event: IntelligenceEvent) -> str:
        """Determine appropriate stream for event type."""
        if event.event_type in [EventType.WORKER_REGISTRATION, EventType.WORKER_HEARTBEAT, EventType.WORKER_SHUTDOWN]:
            return self.streams['worker_events']
        elif event.event_type in [EventType.TASK_ASSIGNMENT, EventType.TASK_PROGRESS, EventType.TASK_COMPLETION, EventType.TASK_FAILURE]:
            return self.streams['task_events']
        elif event.event_type in [EventType.AI_REQUEST, EventType.AI_RESPONSE]:
            return self.streams['ai_events']
        elif event.event_type == EventType.PERFORMANCE_METRIC:
            return self.streams['performance_events']
        elif event.event_type in [EventType.COORDINATION_EVENT, EventType.CAPABILITY_DISCOVERY]:
            return self.streams['coordination_events']
        elif event.event_type == EventType.ANALYTICS_INSIGHT:
            return self.streams['analytics_events']
        else:
            return self.streams['coordination_events']  # Default stream
    
    async def _register_hub(self):
        """Register hub in distributed registry."""
        hub_data = {
            'hub_id': self.hub_id,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'streams': list(self.streams.keys()),
            'consumer_group': self.consumer_group,
            'status': 'active'
        }
        
        await self.state_manager.update(
            f"intelligence.hubs.{self.hub_id}",
            hub_data,
            distributed=True
        )
    
    async def _analytics_aggregator(self):
        """Real-time analytics aggregation processor."""
        while not self._shutdown:
            try:
                # Aggregate performance metrics
                await self._aggregate_performance_metrics()
                
                # Aggregate worker statistics
                await self._aggregate_worker_statistics()
                
                # Aggregate task statistics
                await self._aggregate_task_statistics()
                
                # Refresh hub registration TTL (lock-free)
                await self._refresh_hub_registration()
                
                await asyncio.sleep(30)  # Aggregate every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in analytics aggregator: {e}")
                await asyncio.sleep(10)
    
    async def _insights_generator(self):
        """Generate intelligent insights from patterns."""
        while not self._shutdown:
            try:
                # Generate performance insights
                insights = await self._generate_performance_insights()
                
                if insights:
                    # Publish insights as events
                    for insight in insights:
                        insight_event = IntelligenceEvent(
                            event_id=str(uuid.uuid4()),
                            event_type=EventType.ANALYTICS_INSIGHT,
                            worker_id=self.hub_id,
                            timestamp=datetime.now(timezone.utc),
                            priority=EventPriority.NORMAL,
                            data=insight
                        )
                        await self.publish_event(insight_event)
                
                await asyncio.sleep(300)  # Generate insights every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in insights generator: {e}")
                await asyncio.sleep(60)
    
    async def _anomaly_detector(self):
        """Detect anomalies in system behavior."""
        while not self._shutdown:
            try:
                # Detect performance anomalies
                anomalies = await self._detect_performance_anomalies()
                
                if anomalies:
                    # Publish anomaly events
                    for anomaly in anomalies:
                        anomaly_event = IntelligenceEvent(
                            event_id=str(uuid.uuid4()),
                            event_type=EventType.ERROR_EVENT,
                            worker_id=self.hub_id,
                            timestamp=datetime.now(timezone.utc),
                            priority=EventPriority.HIGH,
                            data=anomaly
                        )
                        await self.publish_event(anomaly_event)
                
                await asyncio.sleep(60)  # Check for anomalies every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in anomaly detector: {e}")
                await asyncio.sleep(30)
    
    async def shutdown(self):
        """Shutdown Intelligence Hub."""
        self.logger.info(f"Shutting down Intelligence Hub: {self.hub_id}")
        self._shutdown = True
        
        # Stop consumer tasks
        for task in self._consumer_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop analytics tasks
        for task in self._analytics_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Update hub status
        await self.state_manager.update(
            f"intelligence.hubs.{self.hub_id}.status",
            'shutdown',
            distributed=True
        )
        
        self.logger.info(f"Intelligence Hub {self.hub_id} shutdown complete")
    
    async def _update_publish_analytics(self, event: IntelligenceEvent, stream_key: str, stream_id: str):
        """Update analytics after publishing an event."""
        try:
            # Track event publishing metrics
            current_count = await self.state_manager.get(
                f"intelligence.analytics.events_published.{event.event_type.value}"
            ) or 0
            await self.state_manager.update(
                f"intelligence.analytics.events_published.{event.event_type.value}",
                current_count + 1,
                distributed=True
            )
        except Exception as e:
            self.logger.error(f"Error updating publish analytics: {e}")
    
    async def _update_event_sourcing_state(self, event: IntelligenceEvent):
        """Update event sourcing state based on event."""
        try:
            # Store event in event store
            event_store_key = f"intelligence.event_store.{event.event_type.value}.{event.event_id}"
            await self.state_manager.update(
                event_store_key,
                event.to_dict(),
                distributed=True
            )
            
            # Update event sequence
            sequence_key = f"intelligence.event_sequence.{event.worker_id}"
            current_sequence = await self.state_manager.get(sequence_key) or []
            updated_sequence = current_sequence + [event.event_id]
            await self.state_manager.set(
                sequence_key,
                updated_sequence,
                distributed=True
            )
        except Exception as e:
            self.logger.error(f"Error updating event sourcing state: {e}")
    
    async def _update_task_performance_metrics(self, event: IntelligenceEvent):
        """Update task performance metrics."""
        try:
            task_id = event.data.get('task_id')
            duration = event.data.get('duration_seconds', 0)
            
            if task_id and duration > 0:
                # Update worker task performance
                await self.state_manager.update(
                    f"intelligence.performance.task_durations.{event.worker_id}",
                    lambda durations: (durations or []) + [duration],
                    distributed=True
                )
                
                # Update task type performance
                task_type = self._task_registry.get(task_id, {}).get('task_type', 'unknown')
                await self.state_manager.update(
                    f"intelligence.performance.task_type_durations.{task_type}",
                    lambda durations: (durations or []) + [duration],
                    distributed=True
                )
        except Exception as e:
            self.logger.error(f"Error updating task performance metrics: {e}")
    
    async def _aggregate_performance_metrics(self):
        """Aggregate performance metrics from all workers."""
        try:
            # Aggregate CPU usage
            cpu_metrics = []
            memory_metrics = []
            
            for metric_key, metric_values in self._performance_metrics.items():
                if 'cpu_usage' in metric_key:
                    cpu_metrics.extend([m['value'] for m in metric_values[-10:]])  # Last 10 values
                elif 'memory_usage' in metric_key:
                    memory_metrics.extend([m['value'] for m in metric_values[-10:]])
            
            # Calculate aggregates
            if cpu_metrics:
                avg_cpu = sum(cpu_metrics) / len(cpu_metrics)
                await self.state_manager.update(
                    f"intelligence.aggregates.avg_cpu_usage",
                    avg_cpu,
                    distributed=True
                )
            
            if memory_metrics:
                avg_memory = sum(memory_metrics) / len(memory_metrics)
                await self.state_manager.update(
                    f"intelligence.aggregates.avg_memory_usage",
                    avg_memory,
                    distributed=True
                )
            
            # Aggregate task completion rates
            completed_tasks = len([t for t in self._task_registry.values() if t.get('status') == 'completed'])
            failed_tasks = len([t for t in self._task_registry.values() if t.get('status') == 'failed'])
            
            if completed_tasks + failed_tasks > 0:
                success_rate = completed_tasks / (completed_tasks + failed_tasks)
                await self.state_manager.update(
                    f"intelligence.aggregates.task_success_rate",
                    success_rate,
                    distributed=True
                )
                
        except Exception as e:
            self.logger.error(f"Error aggregating performance metrics: {e}")
    
    async def _aggregate_worker_statistics(self):
        """Aggregate statistics about workers."""
        try:
            active_workers = 0
            idle_workers = 0
            overloaded_workers = 0
            
            current_time = datetime.now(timezone.utc)
            
            for worker_id, worker_data in self._workers_registry.items():
                # Check if worker is active (heartbeat within last 2 minutes)
                last_heartbeat = datetime.fromisoformat(worker_data.get('last_heartbeat', ''))
                if (current_time - last_heartbeat).total_seconds() < 120:
                    active_workers += 1
                    
                    # Check worker load
                    active_tasks = len([t for t in self._task_registry.values() 
                                      if t.get('worker_id') == worker_id and t.get('status') == 'assigned'])
                    
                    if active_tasks == 0:
                        idle_workers += 1
                    elif active_tasks > 5:  # Threshold for overloaded
                        overloaded_workers += 1
            
            # Update aggregated statistics
            await self.state_manager.update(
                "intelligence.aggregates.worker_stats",
                {
                    'active': active_workers,
                    'idle': idle_workers,
                    'overloaded': overloaded_workers,
                    'total': len(self._workers_registry),
                    'timestamp': current_time.isoformat()
                },
                distributed=True
            )
            
        except Exception as e:
            self.logger.error(f"Error aggregating worker statistics: {e}")
    
    async def _aggregate_task_statistics(self):
        """Aggregate statistics about tasks."""
        try:
            task_stats = {
                'pending': 0,
                'assigned': 0,
                'in_progress': 0,
                'completed': 0,
                'failed': 0,
                'total_duration': 0.0,
                'avg_duration': 0.0
            }
            
            durations = []
            
            for task_data in self._task_registry.values():
                status = task_data.get('status', 'unknown')
                if status in task_stats:
                    task_stats[status] += 1
                
                # Calculate duration for completed tasks
                if status == 'completed' and 'duration_seconds' in task_data:
                    duration = task_data['duration_seconds']
                    durations.append(duration)
                    task_stats['total_duration'] += duration
            
            # Calculate average duration
            if durations:
                task_stats['avg_duration'] = sum(durations) / len(durations)
            
            # Update aggregated statistics
            await self.state_manager.update(
                "intelligence.aggregates.task_stats",
                {
                    **task_stats,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                },
                distributed=True
            )
            
        except Exception as e:
            self.logger.error(f"Error aggregating task statistics: {e}")
    
    async def _generate_performance_insights(self) -> List[Dict[str, Any]]:
        """Generate insights from performance data."""
        insights = []
        
        try:
            # Get aggregated stats
            worker_stats = await self.state_manager.get("intelligence.aggregates.worker_stats")
            task_stats = await self.state_manager.get("intelligence.aggregates.task_stats")
            
            # Insight: Worker utilization
            if worker_stats and worker_stats.get('active'):
                active = int(worker_stats.get('active', 0))
                idle = int(worker_stats.get('idle', 0))
                overloaded = int(worker_stats.get('overloaded', 0))
                
                if active > 0:
                    idle_ratio = idle / active
                    if idle_ratio > 0.5:
                        insights.append({
                            'type': 'worker_utilization',
                            'severity': 'warning',
                            'message': f"High idle ratio: {idle_ratio:.2%} of active workers are idle",
                            'recommendation': 'Consider scaling down workers or increasing task distribution'
                        })
                    
                    if overloaded > active * 0.3:
                        insights.append({
                            'type': 'worker_overload',
                            'severity': 'warning',
                            'message': f"{overloaded} workers are overloaded",
                            'recommendation': 'Consider scaling up workers or optimizing task distribution'
                        })
            
            # Insight: Task performance
            if task_stats:
                failed = int(task_stats.get('failed', 0))
                completed = int(task_stats.get('completed', 0))
                avg_duration = float(task_stats.get('avg_duration', 0))
                
                if completed > 0 and failed > completed * 0.1:
                    insights.append({
                        'type': 'high_failure_rate',
                        'severity': 'critical',
                        'message': f"High task failure rate: {failed} failures",
                        'recommendation': 'Investigate task failures and improve error handling'
                    })
                
                if avg_duration > 300:  # 5 minutes
                    insights.append({
                        'type': 'slow_task_execution',
                        'severity': 'warning',
                        'message': f"Average task duration is {avg_duration:.1f} seconds",
                        'recommendation': 'Consider optimizing task processing or breaking down complex tasks'
                    })
            
            # Insight: System throughput
            if self._processing_stats['throughput_per_second'] < 1.0 and self._processing_stats['events_processed'] > 100:
                insights.append({
                    'type': 'low_throughput',
                    'severity': 'warning',
                    'message': f"Low event throughput: {self._processing_stats['throughput_per_second']:.2f} events/second",
                    'recommendation': 'Consider optimizing event processing or scaling infrastructure'
                })
                
        except Exception as e:
            self.logger.error(f"Error generating performance insights: {e}")
        
        return insights
    
    async def _detect_performance_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in system performance."""
        anomalies = []
        
        try:
            current_time = datetime.now(timezone.utc)
            
            # Check for stale workers
            for worker_id, worker_data in self._workers_registry.items():
                last_heartbeat = datetime.fromisoformat(worker_data.get('last_heartbeat', ''))
                time_since_heartbeat = (current_time - last_heartbeat).total_seconds()
                
                if time_since_heartbeat > 300 and worker_data.get('status') == 'active':  # 5 minutes
                    anomalies.append({
                        'type': 'stale_worker',
                        'severity': 'critical',
                        'worker_id': worker_id,
                        'message': f"Worker {worker_id} hasn't sent heartbeat for {time_since_heartbeat:.0f} seconds",
                        'last_heartbeat': worker_data['last_heartbeat']
                    })
            
            # Check for stuck tasks
            for task_id, task_data in self._task_registry.items():
                if task_data.get('status') == 'assigned':
                    assigned_time = datetime.fromisoformat(task_data.get('assigned_time', ''))
                    time_since_assignment = (current_time - assigned_time).total_seconds()
                    
                    if time_since_assignment > 1800:  # 30 minutes
                        anomalies.append({
                            'type': 'stuck_task',
                            'severity': 'warning',
                            'task_id': task_id,
                            'worker_id': task_data.get('worker_id'),
                            'message': f"Task {task_id} has been assigned for {time_since_assignment:.0f} seconds without progress",
                            'assigned_time': task_data['assigned_time']
                        })
            
            # Check for unusual CPU/memory spikes
            for metric_key, metric_values in self._performance_metrics.items():
                if metric_values and len(metric_values) >= 5:
                    recent_values = [m['value'] for m in metric_values[-5:]]
                    avg_value = sum(recent_values) / len(recent_values)
                    
                    if 'cpu_usage' in metric_key and avg_value > 90:
                        anomalies.append({
                            'type': 'high_cpu_usage',
                            'severity': 'warning',
                            'metric': metric_key,
                            'message': f"High CPU usage detected: {avg_value:.1f}%",
                            'values': recent_values
                        })
                    
                    elif 'memory_usage' in metric_key and avg_value > 85:
                        anomalies.append({
                            'type': 'high_memory_usage',
                            'severity': 'warning',
                            'metric': metric_key,
                            'message': f"High memory usage detected: {avg_value:.1f}%",
                            'values': recent_values
                        })
                        
        except Exception as e:
            self.logger.error(f"Error detecting performance anomalies: {e}")
        
        return anomalies
    
    async def _refresh_hub_registration(self):
        """Refresh hub registration TTL (lock-free)."""
        try:
            hub_key = f"intelligence:hubs:{self.hub_id}"
            # Refresh TTL to keep hub active
            await self.redis_client.expire(hub_key, 300)  # 5 minute TTL
        except Exception as e:
            self.logger.debug(f"Error refreshing hub registration: {e}")


# Global intelligence hub instance
_global_intelligence_hub: Optional[RedisIntelligenceHub] = None


async def get_intelligence_hub(hub_id: str = None, **kwargs) -> RedisIntelligenceHub:
    """Get global intelligence hub instance."""
    global _global_intelligence_hub
    
    if _global_intelligence_hub is None:
        _global_intelligence_hub = RedisIntelligenceHub(hub_id=hub_id, **kwargs)
        await _global_intelligence_hub.initialize()
    
    return _global_intelligence_hub


async def create_intelligence_hub(hub_id: str = None, **kwargs) -> RedisIntelligenceHub:
    """Create new intelligence hub instance."""
    hub = RedisIntelligenceHub(hub_id=hub_id, **kwargs)
    await hub.initialize()
    return hub