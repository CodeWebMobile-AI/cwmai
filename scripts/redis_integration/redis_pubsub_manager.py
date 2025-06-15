"""
Redis Pub/Sub Manager

Real-time messaging and coordination using Redis Pub/Sub for instant
worker communication, event broadcasting, and distributed coordination.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
from .redis_client import RedisClient


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PubSubMessage:
    """Pub/Sub message with metadata."""
    id: str
    channel: str
    data: Any
    priority: MessagePriority
    timestamp: datetime
    sender_id: str
    message_type: str = "data"
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'channel': self.channel,
            'data': self.data,
            'priority': self.priority.value,
            'timestamp': self.timestamp.isoformat(),
            'sender_id': self.sender_id,
            'message_type': self.message_type,
            'ttl_seconds': self.ttl_seconds,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PubSubMessage':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            channel=data['channel'],
            data=data['data'],
            priority=MessagePriority(data['priority']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            sender_id=data['sender_id'],
            message_type=data.get('message_type', 'data'),
            ttl_seconds=data.get('ttl_seconds'),
            metadata=data.get('metadata', {})
        )
    
    def is_expired(self) -> bool:
        """Check if message is expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds() > self.ttl_seconds


@dataclass
class ChannelSubscription:
    """Channel subscription with handler."""
    channel: str
    pattern: bool
    handler: Callable[[PubSubMessage], None]
    message_filter: Optional[Callable[[PubSubMessage], bool]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message_count: int = 0
    last_message_at: Optional[datetime] = None


class MessageQueue:
    """Priority message queue for pub/sub."""
    
    def __init__(self, max_size: int = 10000):
        """Initialize message queue.
        
        Args:
            max_size: Maximum queue size
        """
        self.max_size = max_size
        self.queues = {
            MessagePriority.CRITICAL: asyncio.Queue(),
            MessagePriority.HIGH: asyncio.Queue(),
            MessagePriority.NORMAL: asyncio.Queue(),
            MessagePriority.LOW: asyncio.Queue()
        }
        self.total_size = 0
        self.lock = asyncio.Lock()
    
    async def put(self, message: PubSubMessage):
        """Put message in appropriate priority queue."""
        async with self.lock:
            if self.total_size >= self.max_size:
                # Remove oldest low priority message
                try:
                    await asyncio.wait_for(self.queues[MessagePriority.LOW].get(), timeout=0.1)
                    self.total_size -= 1
                except asyncio.TimeoutError:
                    pass
            
            await self.queues[message.priority].put(message)
            self.total_size += 1
    
    async def get(self) -> PubSubMessage:
        """Get highest priority message."""
        # Check queues in priority order
        for priority in [MessagePriority.CRITICAL, MessagePriority.HIGH, 
                        MessagePriority.NORMAL, MessagePriority.LOW]:
            try:
                message = await asyncio.wait_for(self.queues[priority].get(), timeout=0.1)
                async with self.lock:
                    self.total_size -= 1
                return message
            except asyncio.TimeoutError:
                continue
        
        # If no messages available, wait for any
        tasks = [asyncio.create_task(queue.get()) for queue in self.queues.values()]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
        
        # Return first completed message
        message = done.pop().result()
        async with self.lock:
            self.total_size -= 1
        return message
    
    def qsize(self) -> int:
        """Get total queue size."""
        return self.total_size


class RedisPubSubManager:
    """Advanced Redis Pub/Sub manager with enterprise features."""
    
    def __init__(self, redis_client: RedisClient, instance_id: str = None):
        """Initialize Redis Pub/Sub manager.
        
        Args:
            redis_client: Redis client instance
            instance_id: Unique instance identifier
        """
        self.redis = redis_client
        self.instance_id = instance_id or str(uuid.uuid4())
        self.logger = logging.getLogger(__name__)
        
        # Subscriptions management
        self.subscriptions: Dict[str, ChannelSubscription] = {}
        # Single Pub/Sub connection for all subscriptions
        self._pubsub = None
        self._listener_task: Optional[asyncio.Task] = None
        
        # Message processing
        self.message_queue = MessageQueue()
        self.message_handlers: Dict[str, List[Callable]] = {}  # Channel -> handlers
        self.pattern_handlers: Dict[str, List[Callable]] = {}  # Pattern -> handlers
        
        # Background tasks
        self._processor_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_dropped': 0,
            'subscriptions_active': 0,
            'start_time': datetime.now(timezone.utc)
        }
        
        # Configuration
        self.max_message_size = 1024 * 1024  # 1MB
        self.enable_message_persistence = False
        self.persistence_ttl = 3600  # 1 hour
        self.batch_size = 100
        self.processing_delay = 0.01  # 10ms
    
    async def start(self):
        """Start pub/sub manager."""
        if not self._processor_task:
            self._processor_task = asyncio.create_task(self._message_processor())
            self.logger.info(f"Redis Pub/Sub manager started (ID: {self.instance_id})")
    
    async def start_processing(self):
        """Alias for start() to maintain compatibility."""
        await self.start()
    
    async def stop_processing(self):
        """Alias for stop() to maintain compatibility."""
        await self.stop()
    
    async def psubscribe(self, *patterns):
        """Pattern subscribe - delegate to subscribe_pattern for each pattern."""
        results = []
        for pattern in patterns:
            # Use a simple handler that can be overridden later
            async def default_handler(channel: str, data: Dict[str, Any]):
                self.logger.debug(f"Received message on {channel}: {data}")
            
            result = await self.subscribe_pattern(pattern, default_handler)
            results.append(result)
        return all(results)
    
    async def stop(self):
        """Stop pub/sub manager."""
        self._shutdown = True

        # Stop message processor
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None

        # Stop listener task and close Pub/Sub connection
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None
        if self._pubsub:
            try:
                await self._pubsub.close()
            except Exception:
                pass
            self._pubsub = None

        # Clear subscriptions and handlers
        self.subscriptions.clear()
        self.message_handlers.clear()
        self.pattern_handlers.clear()
        self.stats['subscriptions_active'] = 0

        self.logger.info(f"Redis Pub/Sub manager stopped (ID: {self.instance_id})")

    async def _ensure_pubsub(self):
        """Ensure a single Pub/Sub connection and start listener."""
        if self._pubsub is None:
            self._pubsub = await self.redis.pubsub()
            self._listener_task = asyncio.create_task(self._listen_loop())

    async def _listen_loop(self):
        """Unified listener for channel and pattern messages."""
        try:
            async for message in self._pubsub.listen():
                if self._shutdown:
                    break
                msg_type = message.get('type')
                if msg_type == 'message':
                    await self._handle_message(message.get('channel'), message.get('data'))
                elif msg_type == 'pmessage':
                    await self._handle_message(message.get('channel'), message.get('data'), message.get('pattern'))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in Pub/Sub listener: {e}")

    async def publish(self, channel: str, data: Any, priority: MessagePriority = MessagePriority.NORMAL,
                     message_type: str = "data", ttl_seconds: Optional[int] = None,
                     metadata: Dict[str, Any] = None) -> bool:
        """Publish message to channel.
        
        Args:
            channel: Channel name
            data: Message data
            priority: Message priority
            message_type: Type of message
            ttl_seconds: Message TTL
            metadata: Additional metadata
            
        Returns:
            True if published successfully
        """
        try:
            # Create message
            message = PubSubMessage(
                id=str(uuid.uuid4()),
                channel=channel,
                data=data,
                priority=priority,
                timestamp=datetime.now(timezone.utc),
                sender_id=self.instance_id,
                message_type=message_type,
                ttl_seconds=ttl_seconds,
                metadata=metadata or {}
            )
            
            # Serialize message
            serialized = json.dumps(message.to_dict(), default=str)
            
            # Check message size
            if len(serialized.encode()) > self.max_message_size:
                self.logger.error(f"Message too large for channel {channel}: {len(serialized)} bytes")
                self.stats['messages_dropped'] += 1
                return False
            
            # Publish to Redis
            subscribers = await self.redis.publish(channel, serialized)
            
            # Optionally persist message
            if self.enable_message_persistence:
                await self._persist_message(message)
            
            self.stats['messages_sent'] += 1
            self.logger.debug(f"Published message to {channel} (subscribers: {subscribers})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error publishing to channel {channel}: {e}")
            self.stats['messages_dropped'] += 1
            return False
    
    async def subscribe(self, channel: str, handler: Callable[[str, Dict[str, Any]], None],
                       message_filter: Optional[Callable[[PubSubMessage], bool]] = None) -> bool:
        """Subscribe to channel.
        
        Args:
            channel: Channel name
            handler: Message handler function (takes channel and data)
            message_filter: Optional message filter
            
        Returns:
            True if subscribed successfully
        """
        try:
            # Store the original handler for later use
            original_handler = handler
            
            # Create a wrapper that handles both PubSubMessage and raw handler styles
            async def handler_wrapper(msg: PubSubMessage):
                if asyncio.iscoroutinefunction(original_handler):
                    # Check handler signature
                    import inspect
                    sig = inspect.signature(original_handler)
                    if len(sig.parameters) == 1:
                        # Handler expects PubSubMessage
                        await original_handler(msg)
                    else:
                        # Handler expects channel and data
                        await original_handler(msg.channel, msg.data)
                else:
                    if len(inspect.signature(original_handler).parameters) == 1:
                        original_handler(msg)
                    else:
                        original_handler(msg.channel, msg.data)
            
            # Create subscription
            subscription = ChannelSubscription(
                channel=channel,
                pattern=False,
                handler=handler_wrapper,
                message_filter=message_filter
            )
            
            self.subscriptions[channel] = subscription
            
            # Ensure a single Pub/Sub connection and subscribe
            await self._ensure_pubsub()
            await self._pubsub.subscribe(channel)
            
            # Add handler
            if channel not in self.message_handlers:
                self.message_handlers[channel] = []
            self.message_handlers[channel].append(handler_wrapper)
            
            self.stats['subscriptions_active'] += 1
            self.logger.info(f"Subscribed to channel: {channel}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error subscribing to channel {channel}: {e}")
            return False
    
    async def subscribe_pattern(self, pattern: str, handler: Callable[[str, Dict[str, Any]], None],
                               message_filter: Optional[Callable[[PubSubMessage], bool]] = None) -> bool:
        """Subscribe to channel pattern.
        
        Args:
            pattern: Channel pattern (supports wildcards)
            handler: Message handler function (takes channel and data)
            message_filter: Optional message filter
            
        Returns:
            True if subscribed successfully
        """
        try:
            # Store the original handler for later use
            original_handler = handler
            
            # Create a wrapper that handles both PubSubMessage and raw handler styles
            async def handler_wrapper(msg: PubSubMessage):
                if asyncio.iscoroutinefunction(original_handler):
                    # Check handler signature
                    import inspect
                    sig = inspect.signature(original_handler)
                    if len(sig.parameters) == 1:
                        # Handler expects PubSubMessage
                        await original_handler(msg)
                    else:
                        # Handler expects channel and data
                        await original_handler(msg.channel, msg.data)
                else:
                    if len(inspect.signature(original_handler).parameters) == 1:
                        original_handler(msg)
                    else:
                        original_handler(msg.channel, msg.data)
            
            # Create subscription
            subscription = ChannelSubscription(
                channel=pattern,
                pattern=True,
                handler=handler_wrapper,
                message_filter=message_filter
            )
            
            self.subscriptions[pattern] = subscription
            
            # Ensure a single Pub/Sub connection and pattern-subscribe
            await self._ensure_pubsub()
            await self._pubsub.psubscribe(pattern)
            
            # Add handler
            if pattern not in self.pattern_handlers:
                self.pattern_handlers[pattern] = []
            self.pattern_handlers[pattern].append(handler_wrapper)
            
            self.stats['subscriptions_active'] += 1
            self.logger.info(f"Subscribed to pattern: {pattern}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error subscribing to pattern {pattern}: {e}")
            return False
    
    async def unsubscribe(self, channel: str) -> bool:
        """Unsubscribe from channel or pattern."""
        try:
            subscription = self.subscriptions.get(channel)
            if not subscription:
                return False

            # Remove subscription and handlers
            del self.subscriptions[channel]
            self.message_handlers.pop(channel, None)
            self.pattern_handlers.pop(channel, None)
            self.stats['subscriptions_active'] -= 1

            # Unsubscribe on the shared Pub/Sub connection
            if self._pubsub:
                if subscription.pattern:
                    await self._pubsub.punsubscribe(channel)
                else:
                    await self._pubsub.unsubscribe(channel)

            self.logger.info(f"Unsubscribed from: {channel}")
            return True
        except Exception as e:
            self.logger.error(f"Error unsubscribing from {channel}: {e}")
            return False
    
    
    async def _handle_message(self, channel: str, data: Any, pattern: str = None):
        """Handle incoming message."""
        try:
            # Parse message
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            message_dict = json.loads(data)
            message = PubSubMessage.from_dict(message_dict)
            
            # Check if message is expired
            if message.is_expired():
                self.logger.debug(f"Dropping expired message on {channel}")
                self.stats['messages_dropped'] += 1
                return
            
            # Add to processing queue
            await self.message_queue.put(message)
            self.stats['messages_received'] += 1
            
            # Update subscription stats
            subscription_key = pattern or channel
            if subscription_key in self.subscriptions:
                subscription = self.subscriptions[subscription_key]
                subscription.message_count += 1
                subscription.last_message_at = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error(f"Error handling message on {channel}: {e}")
            self.stats['messages_dropped'] += 1
    
    async def _message_processor(self):
        """Process messages from queue."""
        batch = []
        
        while not self._shutdown:
            try:
                # Collect batch of messages
                while len(batch) < self.batch_size:
                    try:
                        message = await asyncio.wait_for(
                            self.message_queue.get(), 
                            timeout=self.processing_delay
                        )
                        batch.append(message)
                    except asyncio.TimeoutError:
                        break
                
                # Process batch
                if batch:
                    await self._process_message_batch(batch)
                    batch.clear()
                
                await asyncio.sleep(self.processing_delay)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in message processor: {e}")
    
    async def _process_message_batch(self, messages: List[PubSubMessage]):
        """Process batch of messages."""
        for message in messages:
            try:
                await self._process_single_message(message)
            except Exception as e:
                self.logger.error(f"Error processing message {message.id}: {e}")
    
    async def _process_single_message(self, message: PubSubMessage):
        """Process single message."""
        # Find matching handlers
        handlers = []
        
        # Direct channel handlers
        if message.channel in self.message_handlers:
            handlers.extend(self.message_handlers[message.channel])
        
        # Pattern handlers
        for pattern, pattern_handlers in self.pattern_handlers.items():
            if self._match_pattern(pattern, message.channel):
                handlers.extend(pattern_handlers)
        
        # Execute handlers
        for handler in handlers:
            try:
                # Check message filter if subscription has one
                subscription = None
                for sub in self.subscriptions.values():
                    if handler == sub.handler:
                        subscription = sub
                        break
                
                if subscription and subscription.message_filter:
                    if not subscription.message_filter(message):
                        continue
                
                # Execute handler
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
                    
            except Exception as e:
                self.logger.error(f"Error in message handler: {e}")
    
    def _match_pattern(self, pattern: str, channel: str) -> bool:
        """Check if channel matches pattern."""
        import fnmatch
        return fnmatch.fnmatch(channel, pattern)
    
    async def _persist_message(self, message: PubSubMessage):
        """Persist message for replay/recovery."""
        try:
            key = f"pubsub:messages:{message.channel}:{message.id}"
            data = json.dumps(message.to_dict(), default=str)
            await self.redis.set(key, data, ex=self.persistence_ttl)
        except Exception as e:
            self.logger.error(f"Error persisting message: {e}")
    
    async def get_persisted_messages(self, channel: str, since: datetime = None) -> List[PubSubMessage]:
        """Get persisted messages for channel."""
        messages = []
        
        try:
            pattern = f"pubsub:messages:{channel}:*"
            cursor = 0
            
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
                
                for key in keys:
                    data = await self.redis.get(key)
                    if data:
                        message_dict = json.loads(data)
                        message = PubSubMessage.from_dict(message_dict)
                        
                        if since is None or message.timestamp >= since:
                            messages.append(message)
                
                if cursor == 0:
                    break
            
            # Sort by timestamp
            messages.sort(key=lambda m: m.timestamp)
            
        except Exception as e:
            self.logger.error(f"Error getting persisted messages: {e}")
        
        return messages
    
    async def broadcast(self, channels: List[str], data: Any, 
                      priority: MessagePriority = MessagePriority.NORMAL) -> Dict[str, bool]:
        """Broadcast message to multiple channels."""
        results = {}
        
        for channel in channels:
            results[channel] = await self.publish(
                channel, data, priority, message_type="broadcast"
            )
        
        return results
    
    async def request_response(self, channel: str, request_data: Any, 
                              timeout: float = 30.0) -> Optional[Any]:
        """Send request and wait for response."""
        response_channel = f"response:{self.instance_id}:{uuid.uuid4()}"
        response_received = asyncio.Event()
        response_data = None
        
        async def response_handler(message: PubSubMessage):
            nonlocal response_data
            response_data = message.data
            response_received.set()
        
        try:
            # Subscribe to response channel
            await self.subscribe(response_channel, response_handler)
            
            # Send request with response channel
            request_message = {
                'data': request_data,
                'response_channel': response_channel,
                'request_id': str(uuid.uuid4())
            }
            
            success = await self.publish(channel, request_message, 
                                       message_type="request")
            
            if not success:
                return None
            
            # Wait for response
            try:
                await asyncio.wait_for(response_received.wait(), timeout=timeout)
                return response_data
            except asyncio.TimeoutError:
                self.logger.warning(f"Request timeout on channel {channel}")
                return None
        
        finally:
            await self.unsubscribe(response_channel)
    
    def get_subscription_info(self, channel: str) -> Optional[Dict[str, Any]]:
        """Get subscription information."""
        if channel in self.subscriptions:
            subscription = self.subscriptions[channel]
            return {
                'channel': subscription.channel,
                'pattern': subscription.pattern,
                'created_at': subscription.created_at.isoformat(),
                'message_count': subscription.message_count,
                'last_message_at': subscription.last_message_at.isoformat() if subscription.last_message_at else None
            }
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pub/sub statistics."""
        uptime = (datetime.now(timezone.utc) - self.stats['start_time']).total_seconds()
        
        return {
            'instance_id': self.instance_id,
            'uptime_seconds': uptime,
            'messages_sent': self.stats['messages_sent'],
            'messages_received': self.stats['messages_received'],
            'messages_dropped': self.stats['messages_dropped'],
            'subscriptions_active': self.stats['subscriptions_active'],
            'queue_size': self.message_queue.qsize(),
            'channels': list(self.subscriptions.keys()),
            'throughput_sent': self.stats['messages_sent'] / max(uptime, 1),
            'throughput_received': self.stats['messages_received'] / max(uptime, 1)
        }