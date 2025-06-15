"""
Redis Streams Manager

High-performance event processing using Redis Streams for real-time data pipelines,
consumer groups, and distributed event processing with automatic scaling.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional, List, Callable, Union, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
from .redis_client import RedisClient


class StreamMessageStatus(Enum):
    """Stream message processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"


@dataclass
class StreamMessage:
    """Stream message with metadata."""
    id: str
    stream: str
    data: Dict[str, Any]
    timestamp: datetime
    consumer_group: Optional[str] = None
    consumer: Optional[str] = None
    delivery_count: int = 0
    status: StreamMessageStatus = StreamMessageStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'stream': self.stream,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'consumer_group': self.consumer_group,
            'consumer': self.consumer,
            'delivery_count': self.delivery_count,
            'status': self.status.value,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_redis_entry(cls, stream: str, entry_id: str, fields: Dict[str, str]) -> 'StreamMessage':
        """Create from Redis stream entry."""
        return cls(
            id=entry_id,
            stream=stream,
            data=json.loads(fields.get('data', '{}')),
            timestamp=datetime.fromisoformat(fields.get('timestamp', datetime.now(timezone.utc).isoformat())),
            consumer_group=fields.get('consumer_group'),
            consumer=fields.get('consumer'),
            delivery_count=int(fields.get('delivery_count', '0')),
            status=StreamMessageStatus(fields.get('status', 'pending')),
            metadata=json.loads(fields.get('metadata', '{}'))
        )


@dataclass
class ConsumerGroupInfo:
    """Consumer group information."""
    name: str
    stream: str
    consumers: List[str] = field(default_factory=list)
    pending_count: int = 0
    last_delivered_id: str = "0-0"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class StreamProcessingMetrics:
    """Stream processing metrics."""
    messages_produced: int = 0
    messages_consumed: int = 0
    messages_failed: int = 0
    messages_retried: int = 0
    processing_time_total: float = 0.0
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time."""
        return self.processing_time_total / max(self.messages_consumed, 1)
    
    @property
    def throughput(self) -> float:
        """Calculate messages per second."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return (self.messages_produced + self.messages_consumed) / max(uptime, 1)


class RedisStreamsManager:
    """Advanced Redis Streams manager for event processing."""
    
    def __init__(self, redis_client: RedisClient, namespace: str = "streams"):
        """Initialize Redis streams manager.
        
        Args:
            redis_client: Redis client instance
            namespace: Streams namespace for isolation
        """
        self.redis = redis_client
        self.namespace = namespace
        self.logger = logging.getLogger(__name__)
        
        # Consumer management
        self.consumer_groups: Dict[str, ConsumerGroupInfo] = {}
        self.active_consumers: Dict[str, asyncio.Task] = {}
        self.message_handlers: Dict[str, Callable] = {}
        
        # Processing configuration
        self.max_message_length = 1000
        self.consumer_timeout = 30000  # 30 seconds
        self.retry_attempts = 3
        self.retry_delay = 5  # seconds
        self.batch_size = 10
        self.processing_timeout = 60  # seconds
        
        # Metrics and monitoring
        self.metrics = StreamProcessingMetrics()
        self._shutdown = False
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
    
    def _stream_key(self, stream_name: str) -> str:
        """Create namespaced stream key."""
        return f"{self.namespace}:{stream_name}"
    
    def _dlq_key(self, stream_name: str) -> str:
        """Create dead letter queue key."""
        return f"{self.namespace}:dlq:{stream_name}"
    
    def _metrics_key(self, stream_name: str) -> str:
        """Create metrics key."""
        return f"{self.namespace}:metrics:{stream_name}"
    
    async def start(self):
        """Start streams manager background tasks."""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Redis Streams manager started")
    
    async def stop(self):
        """Stop streams manager and cleanup resources."""
        self._shutdown = True
        
        # Stop background tasks
        for task in [self._cleanup_task, self._monitoring_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop all consumers
        for consumer_task in self.active_consumers.values():
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                pass
        
        self.active_consumers.clear()
        self.logger.info("Redis Streams manager stopped")
    
    async def produce(self, stream_name: str, data: Dict[str, Any], 
                     message_id: str = "*", max_length: Optional[int] = None) -> str:
        """Produce message to stream.
        
        Args:
            stream_name: Stream name
            data: Message data
            message_id: Message ID (* for auto-generation)
            max_length: Maximum stream length
            
        Returns:
            Generated message ID
        """
        try:
            stream_key = self._stream_key(stream_name)
            max_length = max_length or self.max_message_length
            
            # Prepare message fields
            fields = {
                'data': json.dumps(data),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'producer_id': str(uuid.uuid4()),
                'status': StreamMessageStatus.PENDING.value
            }
            
            # Add to stream
            message_id = await self.redis.xadd(
                stream_key, 
                fields, 
                id=message_id, 
                maxlen=max_length
            )
            
            self.metrics.messages_produced += 1
            self.logger.debug(f"Produced message {message_id} to stream {stream_name}")
            
            return message_id
            
        except Exception as e:
            self.logger.error(f"Error producing message to stream {stream_name}: {e}")
            raise
    
    async def create_consumer_group(self, stream_name: str, group_name: str, 
                                   start_id: str = "0") -> bool:
        """Create consumer group for stream.
        
        Args:
            stream_name: Stream name
            group_name: Consumer group name
            start_id: Starting message ID
            
        Returns:
            True if created successfully
        """
        try:
            stream_key = self._stream_key(stream_name)
            
            # Create consumer group
            success = await self.redis.xgroup_create(
                stream_key, 
                group_name, 
                id=start_id, 
                mkstream=True
            )
            
            if success:
                self.consumer_groups[f"{stream_name}:{group_name}"] = ConsumerGroupInfo(
                    name=group_name,
                    stream=stream_name
                )
                self.logger.info(f"Created consumer group {group_name} for stream {stream_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error creating consumer group {group_name} for stream {stream_name}: {e}")
            return False
    
    async def start_consumer(self, stream_name: str, group_name: str, 
                           consumer_name: str, handler: Callable) -> bool:
        """Start consumer for processing messages.
        
        Args:
            stream_name: Stream name
            group_name: Consumer group name
            consumer_name: Consumer name
            handler: Message handler function
            
        Returns:
            True if started successfully
        """
        try:
            consumer_key = f"{stream_name}:{group_name}:{consumer_name}"
            
            if consumer_key in self.active_consumers:
                self.logger.warning(f"Consumer {consumer_key} already active")
                return False
            
            # Store handler
            self.message_handlers[consumer_key] = handler
            
            # Start consumer task
            self.active_consumers[consumer_key] = asyncio.create_task(
                self._consumer_loop(stream_name, group_name, consumer_name)
            )
            
            self.logger.info(f"Started consumer {consumer_name} for group {group_name} on stream {stream_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting consumer: {e}")
            return False
    
    async def stop_consumer(self, stream_name: str, group_name: str, consumer_name: str) -> bool:
        """Stop consumer."""
        try:
            consumer_key = f"{stream_name}:{group_name}:{consumer_name}"
            
            if consumer_key in self.active_consumers:
                self.active_consumers[consumer_key].cancel()
                try:
                    await self.active_consumers[consumer_key]
                except asyncio.CancelledError:
                    pass
                del self.active_consumers[consumer_key]
                
                # Remove handler
                self.message_handlers.pop(consumer_key, None)
                
                self.logger.info(f"Stopped consumer {consumer_name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error stopping consumer: {e}")
            return False
    
    async def _consumer_loop(self, stream_name: str, group_name: str, consumer_name: str):
        """Consumer processing loop."""
        stream_key = self._stream_key(stream_name)
        consumer_key = f"{stream_name}:{group_name}:{consumer_name}"
        
        while not self._shutdown:
            try:
                # Read messages from stream
                messages = await self.redis.xreadgroup(
                    group_name,
                    consumer_name,
                    {stream_key: '>'},
                    count=self.batch_size,
                    block=self.consumer_timeout
                )
                
                if messages:
                    for stream, stream_messages in messages:
                        for message_id, fields in stream_messages:
                            await self._process_message(
                                stream_name, group_name, consumer_name,
                                message_id, fields
                            )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in consumer loop for {consumer_key}: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, stream_name: str, group_name: str, 
                              consumer_name: str, message_id: str, fields: Dict[str, str]):
        """Process individual message."""
        consumer_key = f"{stream_name}:{group_name}:{consumer_name}"
        handler = self.message_handlers.get(consumer_key)
        
        if not handler:
            self.logger.error(f"No handler found for consumer {consumer_key}")
            return
        
        start_time = time.time()
        
        try:
            # Create message object
            message = StreamMessage.from_redis_entry(stream_name, message_id, fields)
            message.consumer_group = group_name
            message.consumer = consumer_name
            message.status = StreamMessageStatus.PROCESSING
            
            # Process message with timeout
            if asyncio.iscoroutinefunction(handler):
                await asyncio.wait_for(
                    handler(message), 
                    timeout=self.processing_timeout
                )
            else:
                handler(message)
            
            # Acknowledge message
            stream_key = self._stream_key(stream_name)
            await self.redis.xack(stream_key, group_name, message_id)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.messages_consumed += 1
            self.metrics.processing_time_total += processing_time
            
            self.logger.debug(f"Processed message {message_id} in {processing_time:.3f}s")
            
        except asyncio.TimeoutError:
            self.logger.error(f"Message {message_id} processing timeout")
            await self._handle_failed_message(stream_name, group_name, message_id, "timeout")
        except Exception as e:
            self.logger.error(f"Error processing message {message_id}: {e}")
            await self._handle_failed_message(stream_name, group_name, message_id, str(e))
    
    async def _handle_failed_message(self, stream_name: str, group_name: str, 
                                   message_id: str, error: str):
        """Handle failed message processing."""
        try:
            # Get message info
            stream_key = self._stream_key(stream_name)
            pending_info = await self.redis.xpending_range(
                stream_key, group_name, message_id, message_id, 1
            )
            
            if pending_info:
                delivery_count = pending_info[0]['times_delivered']
                
                if delivery_count < self.retry_attempts:
                    # Retry message
                    await asyncio.sleep(self.retry_delay)
                    self.metrics.messages_retried += 1
                    self.logger.info(f"Retrying message {message_id} (attempt {delivery_count + 1})")
                else:
                    # Move to dead letter queue
                    await self._move_to_dlq(stream_name, group_name, message_id, error)
                    await self.redis.xack(stream_key, group_name, message_id)
                    self.metrics.messages_failed += 1
                    self.logger.warning(f"Moved message {message_id} to DLQ after {delivery_count} attempts")
        
        except Exception as e:
            self.logger.error(f"Error handling failed message {message_id}: {e}")
    
    async def _move_to_dlq(self, stream_name: str, group_name: str, 
                          message_id: str, error: str):
        """Move message to dead letter queue."""
        try:
            dlq_key = self._dlq_key(stream_name)
            
            # Get original message
            stream_key = self._stream_key(stream_name)
            messages = await self.redis.xrange(stream_key, message_id, message_id)
            
            if messages:
                original_fields = messages[0][1]
                
                # Add error information
                dlq_fields = {
                    **original_fields,
                    'dlq_timestamp': datetime.now(timezone.utc).isoformat(),
                    'dlq_error': error,
                    'dlq_group': group_name,
                    'original_id': message_id
                }
                
                await self.redis.xadd(dlq_key, dlq_fields)
        
        except Exception as e:
            self.logger.error(f"Error moving message to DLQ: {e}")
    
    async def get_stream_info(self, stream_name: str) -> Optional[Dict[str, Any]]:
        """Get stream information."""
        try:
            stream_key = self._stream_key(stream_name)
            info = await self.redis.xinfo_stream(stream_key)
            return info
        except Exception as e:
            self.logger.error(f"Error getting stream info for {stream_name}: {e}")
            return None
    
    async def get_consumer_group_info(self, stream_name: str) -> List[Dict[str, Any]]:
        """Get consumer group information."""
        try:
            stream_key = self._stream_key(stream_name)
            groups = await self.redis.xinfo_groups(stream_key)
            return groups
        except Exception as e:
            self.logger.error(f"Error getting consumer group info for {stream_name}: {e}")
            return []
    
    async def get_pending_messages(self, stream_name: str, group_name: str, 
                                  count: int = 10) -> List[Dict[str, Any]]:
        """Get pending messages for consumer group."""
        try:
            stream_key = self._stream_key(stream_name)
            pending = await self.redis.xpending_range(
                stream_key, group_name, '-', '+', count
            )
            return pending
        except Exception as e:
            self.logger.error(f"Error getting pending messages: {e}")
            return []
    
    async def read_messages(self, stream_name: str, group_name: str, 
                           count: int = 10, block: int = 1000) -> List[tuple]:
        """Read messages from stream using consumer group.
        
        Args:
            stream_name: Stream name
            group_name: Consumer group name
            count: Number of messages to read
            block: Blocking timeout in milliseconds
            
        Returns:
            List of tuples (stream_name, message_id, data)
        """
        try:
            stream_key = self._stream_key(stream_name)
            consumer_name = f"consumer_{uuid.uuid4().hex[:8]}"
            
            # Ensure consumer group exists
            try:
                await self.redis.xgroup_create(stream_key, group_name, id='0', mkstream=True)
            except Exception:
                # Group already exists
                pass
            
            # Read messages
            messages = await self.redis.xreadgroup(
                group_name,
                consumer_name,
                {stream_key: '>'},
                count=count,
                block=block
            )
            
            result = []
            if messages:
                for stream, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        # Parse the data field
                        data = {}
                        for key, value in fields.items():
                            if key == 'data':
                                try:
                                    data = json.loads(value)
                                except:
                                    data = {'raw_data': value}
                            else:
                                data[key] = value
                        
                        result.append((stream_name, message_id, data))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error reading messages from stream {stream_name}: {e}")
            return []
    
    async def replay_dlq_messages(self, stream_name: str, max_count: int = 100) -> int:
        """Replay messages from dead letter queue."""
        replayed = 0
        
        try:
            dlq_key = self._dlq_key(stream_name)
            
            # Get DLQ messages
            messages = await self.redis.xrange(dlq_key, '-', '+', count=max_count)
            
            for message_id, fields in messages:
                try:
                    # Extract original data
                    original_data = json.loads(fields.get('data', '{}'))
                    
                    # Republish to main stream
                    new_id = await self.produce(stream_name, original_data)
                    
                    # Remove from DLQ
                    await self.redis.xdel(dlq_key, message_id)
                    
                    replayed += 1
                    self.logger.debug(f"Replayed DLQ message {message_id} as {new_id}")
                    
                except Exception as e:
                    self.logger.error(f"Error replaying DLQ message {message_id}: {e}")
            
            if replayed > 0:
                self.logger.info(f"Replayed {replayed} messages from DLQ for stream {stream_name}")
        
        except Exception as e:
            self.logger.error(f"Error replaying DLQ messages: {e}")
        
        return replayed
    
    async def _cleanup_loop(self):
        """Background cleanup of old messages and metrics."""
        while not self._shutdown:
            try:
                await self._cleanup_old_messages()
                await asyncio.sleep(300)  # 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)
    
    async def _monitoring_loop(self):
        """Background monitoring and metrics collection."""
        while not self._shutdown:
            try:
                await self._update_stream_metrics()
                await asyncio.sleep(60)  # 1 minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_messages(self):
        """Cleanup old processed messages."""
        try:
            # Get all streams in namespace
            pattern = f"{self.namespace}:*"
            cursor = 0
            
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
                
                for key in keys:
                    if key.endswith(':dlq') or key.endswith(':metrics'):
                        continue
                    
                    try:
                        # Trim stream to last 10000 messages
                        await self.redis.xtrim(key, maxlen=10000, approximate=True)
                    except Exception:
                        pass
                
                if cursor == 0:
                    break
        
        except Exception as e:
            self.logger.error(f"Error in cleanup: {e}")
    
    async def _update_stream_metrics(self):
        """Update stream processing metrics."""
        try:
            for consumer_key, info in self.consumer_groups.items():
                stream_name = info.stream
                metrics_key = self._metrics_key(stream_name)
                
                # Store current metrics
                metrics_data = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'messages_produced': self.metrics.messages_produced,
                    'messages_consumed': self.metrics.messages_consumed,
                    'messages_failed': self.metrics.messages_failed,
                    'average_processing_time': self.metrics.average_processing_time,
                    'throughput': self.metrics.throughput
                }
                
                await self.redis.lpush(metrics_key, json.dumps(metrics_data))
                await self.redis.ltrim(metrics_key, 0, 999)  # Keep last 1000 metrics
        
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current processing metrics."""
        return {
            'messages_produced': self.metrics.messages_produced,
            'messages_consumed': self.metrics.messages_consumed,
            'messages_failed': self.metrics.messages_failed,
            'messages_retried': self.metrics.messages_retried,
            'average_processing_time': self.metrics.average_processing_time,
            'throughput': self.metrics.throughput,
            'active_consumers': len(self.active_consumers),
            'consumer_groups': len(self.consumer_groups),
            'uptime_seconds': (datetime.now(timezone.utc) - self.metrics.start_time).total_seconds()
        }