"""
Redis Message Queues with Priority and Reliability

Advanced message queue system using Redis with priority scheduling, reliable delivery,
dead letter queues, and comprehensive queue management capabilities.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

from redis_integration import get_redis_client
from redis_lockfree_adapter import create_lockfree_state_manager
from redis_lua_engine import get_lua_engine


class MessagePriority(Enum):
    """Message priority levels."""
    LOWEST = 1
    LOW = 2
    NORMAL = 3
    HIGH = 4
    HIGHEST = 5
    CRITICAL = 6


class MessageStatus(Enum):
    """Message status states."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"
    RETRY = "retry"


class DeliveryMode(Enum):
    """Message delivery modes."""
    AT_LEAST_ONCE = "at_least_once"
    AT_MOST_ONCE = "at_most_once"
    EXACTLY_ONCE = "exactly_once"


@dataclass
class QueueMessage:
    """Queue message structure."""
    message_id: str
    queue_name: str
    payload: Dict[str, Any]
    priority: MessagePriority
    delivery_mode: DeliveryMode
    created_at: datetime
    scheduled_at: Optional[datetime]
    expires_at: Optional[datetime]
    retry_count: int
    max_retries: int
    processing_timeout: int
    correlation_id: Optional[str]
    reply_to: Optional[str]
    message_type: Optional[str]
    headers: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'message_id': self.message_id,
            'queue_name': self.queue_name,
            'payload': json.dumps(self.payload),
            'priority': self.priority.value,
            'delivery_mode': self.delivery_mode.value,
            'created_at': self.created_at.isoformat(),
            'scheduled_at': self.scheduled_at.isoformat() if self.scheduled_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'processing_timeout': self.processing_timeout,
            'correlation_id': self.correlation_id,
            'reply_to': self.reply_to,
            'message_type': self.message_type,
            'headers': json.dumps(self.headers)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueueMessage':
        """Create message from dictionary."""
        return cls(
            message_id=data['message_id'],
            queue_name=data['queue_name'],
            payload=json.loads(data['payload']),
            priority=MessagePriority(data['priority']),
            delivery_mode=DeliveryMode(data['delivery_mode']),
            created_at=datetime.fromisoformat(data['created_at']),
            scheduled_at=datetime.fromisoformat(data['scheduled_at']) if data.get('scheduled_at') else None,
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            retry_count=data['retry_count'],
            max_retries=data['max_retries'],
            processing_timeout=data['processing_timeout'],
            correlation_id=data.get('correlation_id'),
            reply_to=data.get('reply_to'),
            message_type=data.get('message_type'),
            headers=json.loads(data.get('headers', '{}'))
        )


@dataclass
class QueueStats:
    """Queue statistics."""
    queue_name: str
    pending_messages: int
    processing_messages: int
    completed_messages: int
    failed_messages: int
    dead_letter_messages: int
    total_messages: int
    average_processing_time: float
    throughput_per_minute: float
    last_activity: Optional[datetime]


class QueueConsumer:
    """Queue message consumer."""
    
    def __init__(self, 
                 consumer_id: str,
                 queue_name: str,
                 handler: Callable,
                 concurrency: int = 1,
                 prefetch_count: int = 1):
        """Initialize queue consumer.
        
        Args:
            consumer_id: Unique consumer identifier
            queue_name: Queue to consume from
            handler: Message handler function
            concurrency: Number of concurrent message handlers
            prefetch_count: Number of messages to prefetch
        """
        self.consumer_id = consumer_id
        self.queue_name = queue_name
        self.handler = handler
        self.concurrency = concurrency
        self.prefetch_count = prefetch_count
        
        self.logger = logging.getLogger(f"{__name__}.QueueConsumer")
        self.is_active = False
        self._processing_tasks: Set[asyncio.Task] = set()
        self._shutdown = False


class RedisMessageQueue:
    """Advanced Redis message queue with priority and reliability."""
    
    def __init__(self,
                 queue_manager_id: str = None,
                 enable_dead_letter: bool = True,
                 enable_message_ttl: bool = True,
                 enable_priority_scheduling: bool = True,
                 default_processing_timeout: int = 300):
        """Initialize Redis message queue manager.
        
        Args:
            queue_manager_id: Unique queue manager identifier
            enable_dead_letter: Enable dead letter queues
            enable_message_ttl: Enable message TTL
            enable_priority_scheduling: Enable priority-based scheduling
            default_processing_timeout: Default message processing timeout
        """
        self.queue_manager_id = queue_manager_id or f"queue_mgr_{uuid.uuid4().hex[:8]}"
        self.enable_dead_letter = enable_dead_letter
        self.enable_message_ttl = enable_message_ttl
        self.enable_priority_scheduling = enable_priority_scheduling
        self.default_processing_timeout = default_processing_timeout
        
        self.logger = logging.getLogger(f"{__name__}.RedisMessageQueue")
        
        # Redis components
        self.redis_client = None
        self.state_manager = None
        self.lua_engine = None
        
        # Queue management
        self._active_queues: Dict[str, Dict[str, Any]] = {}
        self._active_consumers: Dict[str, QueueConsumer] = {}
        self._queue_stats: Dict[str, QueueStats] = {}
        
        # Background tasks
        self._management_tasks: List[asyncio.Task] = []
        self._shutdown = False
        
        # Performance metrics
        self._metrics = {
            'messages_enqueued': 0,
            'messages_dequeued': 0,
            'messages_processed': 0,
            'messages_failed': 0,
            'dead_letter_messages': 0,
            'average_queue_time': 0.0,
            'average_processing_time': 0.0,
            'throughput_per_second': 0.0
        }
        
        # Redis key prefixes
        self._queue_prefix = "queue:"
        self._processing_prefix = "processing:"
        self._dead_letter_prefix = "dlq:"
        self._message_prefix = "message:"
        self._stats_prefix = "queue_stats:"
    
    async def initialize(self):
        """Initialize message queue manager."""
        try:
            self.logger.info(f"Initializing Redis Message Queue Manager: {self.queue_manager_id}")
            
            # Initialize Redis components
            self.redis_client = await get_redis_client()
            self.state_manager = create_lockfree_state_manager(f"message_queue_{self.queue_system.queue_id}")
            await self.state_manager.initialize()
            self.lua_engine = await get_lua_engine()
            
            # Load Lua scripts
            await self._load_queue_scripts()
            
            # Start management tasks
            await self._start_management_tasks()
            
            # Register queue manager
            await self._register_queue_manager()
            
            self.logger.info(f"Message Queue Manager {self.queue_manager_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Message Queue Manager: {e}")
            raise
    
    async def _load_queue_scripts(self):
        """Load Lua scripts for queue operations."""
        try:
            # Priority enqueue script
            priority_enqueue_script = {
                'script_id': 'priority_enqueue',
                'name': 'Priority Message Enqueue',
                'description': 'Enqueue message with priority and scheduling',
                'script_type': 'business_logic',
                'script_code': """
                local queue_key = KEYS[1]
                local message_key = KEYS[2]
                local message_id = ARGV[1]
                local priority = tonumber(ARGV[2])
                local scheduled_timestamp = tonumber(ARGV[3])
                local message_data = ARGV[4]
                local current_time = tonumber(ARGV[5])
                
                -- Store message data
                redis.call('HSET', message_key, 'data', message_data)
                redis.call('HSET', message_key, 'status', 'pending')
                redis.call('HSET', message_key, 'enqueued_at', current_time)
                
                -- Calculate priority score (higher priority = higher score)
                local priority_score = priority * 1000000
                
                -- If scheduled, use scheduled time, otherwise use current time
                local schedule_time = scheduled_timestamp > 0 and scheduled_timestamp or current_time
                local score = priority_score + schedule_time
                
                -- Add to priority queue
                redis.call('ZADD', queue_key, score, message_id)
                
                -- Set message TTL if applicable
                if scheduled_timestamp > current_time then
                    local ttl = math.ceil((scheduled_timestamp - current_time) + 3600)
                    redis.call('EXPIRE', message_key, ttl)
                else
                    redis.call('EXPIRE', message_key, 3600) -- Default 1 hour
                end
                
                return {1, score}
                """,
                'parameters': ['message_id', 'priority', 'scheduled_timestamp', 'message_data', 'current_time'],
                'key_patterns': ['queue:*', 'message:*'],
                'version': '1.0',
                'created_at': datetime.now(timezone.utc),
                'last_modified': datetime.now(timezone.utc),
                'metadata': {'queue_operation': 'enqueue'}
            }
            
            # Reliable dequeue script
            reliable_dequeue_script = {
                'script_id': 'reliable_dequeue',
                'name': 'Reliable Message Dequeue',
                'description': 'Dequeue message with reliability guarantees',
                'script_type': 'business_logic',
                'script_code': """
                local queue_key = KEYS[1]
                local processing_key = KEYS[2]
                local consumer_id = ARGV[1]
                local processing_timeout = tonumber(ARGV[2])
                local current_time = tonumber(ARGV[3])
                local max_messages = tonumber(ARGV[4]) or 1
                
                local messages = {}
                
                for i = 1, max_messages do
                    -- Get highest priority message that's ready for processing
                    local candidates = redis.call('ZRANGEBYSCORE', queue_key, 0, current_time, 'WITHSCORES', 'LIMIT', 0, 1)
                    
                    if #candidates == 0 then
                        break
                    end
                    
                    local message_id = candidates[1]
                    local score = candidates[2]
                    
                    -- Remove from queue
                    redis.call('ZREM', queue_key, message_id)
                    
                    -- Add to processing set with timeout
                    local processing_info = {
                        consumer_id = consumer_id,
                        started_at = current_time,
                        timeout_at = current_time + processing_timeout,
                        message_id = message_id
                    }
                    
                    redis.call('HSET', processing_key, message_id, cjson.encode(processing_info))
                    redis.call('EXPIRE', processing_key, processing_timeout + 60)
                    
                    -- Get message data
                    local message_key = 'message:' .. message_id
                    local message_data = redis.call('HGET', message_key, 'data')
                    
                    if message_data then
                        -- Update message status
                        redis.call('HSET', message_key, 'status', 'processing')
                        redis.call('HSET', message_key, 'processing_started', current_time)
                        redis.call('HSET', message_key, 'processor', consumer_id)
                        
                        table.insert(messages, {
                            message_id = message_id,
                            data = message_data,
                            score = score
                        })
                    end
                end
                
                return cjson.encode(messages)
                """,
                'parameters': ['consumer_id', 'processing_timeout', 'current_time', 'max_messages'],
                'key_patterns': ['queue:*', 'processing:*', 'message:*'],
                'version': '1.0',
                'created_at': datetime.now(timezone.utc),
                'last_modified': datetime.now(timezone.utc),
                'metadata': {'queue_operation': 'dequeue'}
            }
            
            # Message acknowledgment script
            message_ack_script = {
                'script_id': 'message_ack',
                'name': 'Message Acknowledgment',
                'description': 'Acknowledge message completion or failure',
                'script_type': 'business_logic',
                'script_code': """
                local processing_key = KEYS[1]
                local message_key = KEYS[2]
                local dead_letter_key = KEYS[3]
                local retry_queue_key = KEYS[4]
                local message_id = ARGV[1]
                local consumer_id = ARGV[2]
                local result = ARGV[3]  -- 'success', 'failure', 'retry'
                local current_time = tonumber(ARGV[4])
                local max_retries = tonumber(ARGV[5])
                
                -- Check if message is being processed by this consumer
                local processing_info = redis.call('HGET', processing_key, message_id)
                if not processing_info then
                    return {0, 'message_not_processing'}
                end
                
                local proc_data = cjson.decode(processing_info)
                if proc_data.consumer_id ~= consumer_id then
                    return {0, 'wrong_consumer'}
                end
                
                -- Remove from processing
                redis.call('HDEL', processing_key, message_id)
                
                -- Get current retry count
                local retry_count = tonumber(redis.call('HGET', message_key, 'retry_count') or '0')
                
                if result == 'success' then
                    -- Mark as completed
                    redis.call('HSET', message_key, 'status', 'completed')
                    redis.call('HSET', message_key, 'completed_at', current_time)
                    redis.call('EXPIRE', message_key, 3600) -- Keep for 1 hour
                    return {1, 'completed'}
                    
                elseif result == 'failure' then
                    if retry_count < max_retries then
                        -- Retry the message
                        local new_retry_count = retry_count + 1
                        redis.call('HSET', message_key, 'retry_count', new_retry_count)
                        redis.call('HSET', message_key, 'status', 'retry')
                        redis.call('HSET', message_key, 'last_failure', current_time)
                        
                        -- Add back to queue with delay
                        local delay = math.min(new_retry_count * new_retry_count * 60, 3600) -- Exponential backoff, max 1 hour
                        local retry_time = current_time + delay
                        local priority = 3 -- Normal priority for retries
                        local score = priority * 1000000 + retry_time
                        
                        redis.call('ZADD', retry_queue_key, score, message_id)
                        return {1, 'retrying', new_retry_count}
                    else
                        -- Send to dead letter queue
                        redis.call('HSET', message_key, 'status', 'dead_letter')
                        redis.call('HSET', message_key, 'dead_letter_at', current_time)
                        redis.call('ZADD', dead_letter_key, current_time, message_id)
                        return {1, 'dead_letter'}
                    end
                end
                
                return {0, 'unknown_result'}
                """,
                'parameters': ['message_id', 'consumer_id', 'result', 'current_time', 'max_retries'],
                'key_patterns': ['processing:*', 'message:*', 'dlq:*', 'queue:*'],
                'version': '1.0',
                'created_at': datetime.now(timezone.utc),
                'last_modified': datetime.now(timezone.utc),
                'metadata': {'queue_operation': 'acknowledge'}
            }
            
            # Register scripts with Lua engine
            for script_data in [priority_enqueue_script, reliable_dequeue_script, message_ack_script]:
                from redis_lua_engine import LuaScript, ScriptType
                
                script = LuaScript(
                    script_id=script_data['script_id'],
                    name=script_data['name'],
                    description=script_data['description'],
                    script_type=getattr(ScriptType, script_data['script_type'].upper()),
                    script_code=script_data['script_code'],
                    parameters=script_data['parameters'],
                    key_patterns=script_data['key_patterns'],
                    version=script_data['version'],
                    created_at=script_data['created_at'],
                    last_modified=script_data['last_modified'],
                    metadata=script_data['metadata']
                )
                
                await self.lua_engine.register_script(script)
            
            self.logger.info("Queue Lua scripts loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading queue scripts: {e}")
            raise
    
    async def _start_management_tasks(self):
        """Start background management tasks."""
        try:
            # Timeout monitor
            timeout_task = asyncio.create_task(self._timeout_monitor())
            self._management_tasks.append(timeout_task)
            
            # Statistics updater
            stats_task = asyncio.create_task(self._statistics_updater())
            self._management_tasks.append(stats_task)
            
            # Dead letter processor
            if self.enable_dead_letter:
                dl_task = asyncio.create_task(self._dead_letter_processor())
                self._management_tasks.append(dl_task)
            
            # Message TTL cleaner
            if self.enable_message_ttl:
                ttl_task = asyncio.create_task(self._ttl_cleaner())
                self._management_tasks.append(ttl_task)
            
            # Queue health monitor
            health_task = asyncio.create_task(self._queue_health_monitor())
            self._management_tasks.append(health_task)
            
            self.logger.info(f"Started {len(self._management_tasks)} management tasks")
            
        except Exception as e:
            self.logger.error(f"Error starting management tasks: {e}")
    
    async def create_queue(self,
                          queue_name: str,
                          max_priority: int = 6,
                          default_ttl: int = 3600,
                          enable_dlq: bool = None,
                          max_retries: int = 3) -> bool:
        """Create a new message queue.
        
        Args:
            queue_name: Name of the queue
            max_priority: Maximum priority level
            default_ttl: Default message TTL in seconds
            enable_dlq: Enable dead letter queue
            max_retries: Maximum retry attempts
            
        Returns:
            True if queue created successfully
        """
        try:
            queue_config = {
                'queue_name': queue_name,
                'max_priority': max_priority,
                'default_ttl': default_ttl,
                'enable_dlq': enable_dlq if enable_dlq is not None else self.enable_dead_letter,
                'max_retries': max_retries,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'created_by': self.queue_manager_id,
                'status': 'active'
            }
            
            # Store queue configuration
            await self.state_manager.update(
                f"queue_config.{queue_name}",
                queue_config,
                distributed=True
            )
            
            # Initialize queue statistics
            stats = QueueStats(
                queue_name=queue_name,
                pending_messages=0,
                processing_messages=0,
                completed_messages=0,
                failed_messages=0,
                dead_letter_messages=0,
                total_messages=0,
                average_processing_time=0.0,
                throughput_per_minute=0.0,
                last_activity=None
            )
            
            self._queue_stats[queue_name] = stats
            self._active_queues[queue_name] = queue_config
            
            self.logger.info(f"Created queue: {queue_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating queue {queue_name}: {e}")
            return False
    
    async def enqueue_message(self,
                             queue_name: str,
                             payload: Dict[str, Any],
                             priority: MessagePriority = MessagePriority.NORMAL,
                             delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE,
                             scheduled_at: datetime = None,
                             expires_at: datetime = None,
                             max_retries: int = 3,
                             processing_timeout: int = None,
                             correlation_id: str = None,
                             reply_to: str = None,
                             message_type: str = None,
                             headers: Dict[str, Any] = None) -> Optional[str]:
        """Enqueue a message.
        
        Args:
            queue_name: Target queue name
            payload: Message payload
            priority: Message priority
            delivery_mode: Delivery mode
            scheduled_at: Scheduled delivery time
            expires_at: Message expiration time
            max_retries: Maximum retry attempts
            processing_timeout: Processing timeout in seconds
            correlation_id: Correlation ID
            reply_to: Reply-to queue
            message_type: Message type
            headers: Additional headers
            
        Returns:
            Message ID if enqueued successfully
        """
        try:
            # Check if queue exists
            if queue_name not in self._active_queues:
                self.logger.error(f"Queue {queue_name} does not exist")
                return None
            
            # Create message
            message_id = str(uuid.uuid4())
            current_time = datetime.now(timezone.utc)
            
            message = QueueMessage(
                message_id=message_id,
                queue_name=queue_name,
                payload=payload,
                priority=priority,
                delivery_mode=delivery_mode,
                created_at=current_time,
                scheduled_at=scheduled_at,
                expires_at=expires_at,
                retry_count=0,
                max_retries=max_retries,
                processing_timeout=processing_timeout or self.default_processing_timeout,
                correlation_id=correlation_id,
                reply_to=reply_to,
                message_type=message_type,
                headers=headers or {}
            )
            
            # Use Lua script for atomic enqueue
            queue_key = f"{self._queue_prefix}{queue_name}"
            message_key = f"{self._message_prefix}{message_id}"
            
            scheduled_timestamp = scheduled_at.timestamp() if scheduled_at else 0
            current_timestamp = current_time.timestamp()
            
            result = await self.lua_engine.execute_script(
                'priority_enqueue',
                keys=[queue_key, message_key],
                args=[
                    message_id,
                    priority.value,
                    scheduled_timestamp,
                    json.dumps(message.to_dict()),
                    current_timestamp
                ]
            )
            
            if result and result[0] == 1:
                self._metrics['messages_enqueued'] += 1
                
                # Update queue statistics
                if queue_name in self._queue_stats:
                    self._queue_stats[queue_name].pending_messages += 1
                    self._queue_stats[queue_name].total_messages += 1
                    self._queue_stats[queue_name].last_activity = current_time
                
                self.logger.debug(f"Enqueued message {message_id} to {queue_name}")
                return message_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error enqueuing message to {queue_name}: {e}")
            return None
    
    async def dequeue_messages(self,
                              queue_name: str,
                              consumer_id: str,
                              max_messages: int = 1,
                              processing_timeout: int = None) -> List[QueueMessage]:
        """Dequeue messages from queue.
        
        Args:
            queue_name: Queue to dequeue from
            consumer_id: Consumer identifier
            max_messages: Maximum messages to dequeue
            processing_timeout: Processing timeout in seconds
            
        Returns:
            List of dequeued messages
        """
        try:
            # Check if queue exists
            if queue_name not in self._active_queues:
                self.logger.error(f"Queue {queue_name} does not exist")
                return []
            
            # Use Lua script for reliable dequeue
            queue_key = f"{self._queue_prefix}{queue_name}"
            processing_key = f"{self._processing_prefix}{queue_name}"
            
            timeout = processing_timeout or self.default_processing_timeout
            current_timestamp = time.time()
            
            result = await self.lua_engine.execute_script(
                'reliable_dequeue',
                keys=[queue_key, processing_key],
                args=[consumer_id, timeout, current_timestamp, max_messages]
            )
            
            if not result:
                return []
            
            messages_data = json.loads(result)
            messages = []
            
            for msg_data in messages_data:
                # Parse message data
                message_dict = json.loads(msg_data['data'])
                message = QueueMessage.from_dict(message_dict)
                messages.append(message)
            
            self._metrics['messages_dequeued'] += len(messages)
            
            # Update queue statistics
            if queue_name in self._queue_stats:
                self._queue_stats[queue_name].pending_messages -= len(messages)
                self._queue_stats[queue_name].processing_messages += len(messages)
                self._queue_stats[queue_name].last_activity = datetime.now(timezone.utc)
            
            self.logger.debug(f"Dequeued {len(messages)} messages from {queue_name}")
            return messages
            
        except Exception as e:
            self.logger.error(f"Error dequeuing from {queue_name}: {e}")
            return []
    
    async def acknowledge_message(self,
                                 message: QueueMessage,
                                 consumer_id: str,
                                 success: bool = True,
                                 error_message: str = None) -> bool:
        """Acknowledge message processing result.
        
        Args:
            message: Message to acknowledge
            consumer_id: Consumer identifier
            success: Whether processing was successful
            error_message: Error message if failed
            
        Returns:
            True if acknowledged successfully
        """
        try:
            # Use Lua script for atomic acknowledgment
            processing_key = f"{self._processing_prefix}{message.queue_name}"
            message_key = f"{self._message_prefix}{message.message_id}"
            dead_letter_key = f"{self._dead_letter_prefix}{message.queue_name}"
            retry_queue_key = f"{self._queue_prefix}{message.queue_name}"
            
            result_type = 'success' if success else 'failure'
            current_timestamp = time.time()
            
            result = await self.lua_engine.execute_script(
                'message_ack',
                keys=[processing_key, message_key, dead_letter_key, retry_queue_key],
                args=[
                    message.message_id,
                    consumer_id,
                    result_type,
                    current_timestamp,
                    message.max_retries
                ]
            )
            
            if result and result[0] == 1:
                ack_result = result[1]
                
                # Update metrics and statistics
                if ack_result == 'completed':
                    self._metrics['messages_processed'] += 1
                    if message.queue_name in self._queue_stats:
                        self._queue_stats[message.queue_name].processing_messages -= 1
                        self._queue_stats[message.queue_name].completed_messages += 1
                
                elif ack_result == 'retrying':
                    retry_count = result[2] if len(result) > 2 else 0
                    self.logger.info(f"Message {message.message_id} retrying (attempt {retry_count})")
                    if message.queue_name in self._queue_stats:
                        self._queue_stats[message.queue_name].processing_messages -= 1
                        self._queue_stats[message.queue_name].pending_messages += 1
                
                elif ack_result == 'dead_letter':
                    self._metrics['dead_letter_messages'] += 1
                    self.logger.warning(f"Message {message.message_id} sent to dead letter queue")
                    if message.queue_name in self._queue_stats:
                        self._queue_stats[message.queue_name].processing_messages -= 1
                        self._queue_stats[message.queue_name].dead_letter_messages += 1
                
                # Store error information if failed
                if not success and error_message:
                    await self.redis_client.hset(
                        message_key,
                        'error_message',
                        error_message
                    )
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error acknowledging message {message.message_id}: {e}")
            return False
    
    async def register_consumer(self,
                               consumer_id: str,
                               queue_name: str,
                               handler: Callable,
                               concurrency: int = 1,
                               prefetch_count: int = 1) -> bool:
        """Register a message consumer.
        
        Args:
            consumer_id: Unique consumer identifier
            queue_name: Queue to consume from
            handler: Message handler function
            concurrency: Number of concurrent handlers
            prefetch_count: Number of messages to prefetch
            
        Returns:
            True if registered successfully
        """
        try:
            consumer = QueueConsumer(
                consumer_id=consumer_id,
                queue_name=queue_name,
                handler=handler,
                concurrency=concurrency,
                prefetch_count=prefetch_count
            )
            
            self._active_consumers[consumer_id] = consumer
            
            # Start consumer processing
            await self._start_consumer(consumer)
            
            self.logger.info(f"Registered consumer {consumer_id} for queue {queue_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering consumer {consumer_id}: {e}")
            return False
    
    async def _start_consumer(self, consumer: QueueConsumer):
        """Start consumer processing."""
        try:
            consumer.is_active = True
            
            # Start concurrent processing tasks
            for i in range(consumer.concurrency):
                task = asyncio.create_task(
                    self._consumer_processor(consumer, f"{consumer.consumer_id}_{i}")
                )
                consumer._processing_tasks.add(task)
            
            self.logger.info(f"Started consumer {consumer.consumer_id} with {consumer.concurrency} concurrent processors")
            
        except Exception as e:
            self.logger.error(f"Error starting consumer {consumer.consumer_id}: {e}")
    
    async def _consumer_processor(self, consumer: QueueConsumer, processor_id: str):
        """Consumer message processor."""
        while not self._shutdown and consumer.is_active:
            try:
                # Dequeue messages
                messages = await self.dequeue_messages(
                    consumer.queue_name,
                    processor_id,
                    consumer.prefetch_count
                )
                
                if not messages:
                    await asyncio.sleep(1)  # No messages, wait a bit
                    continue
                
                # Process messages
                for message in messages:
                    try:
                        start_time = time.time()
                        
                        # Call handler
                        if asyncio.iscoroutinefunction(consumer.handler):
                            result = await consumer.handler(message)
                        else:
                            result = consumer.handler(message)
                        
                        processing_time = time.time() - start_time
                        
                        # Acknowledge success
                        await self.acknowledge_message(message, processor_id, True)
                        
                        # Update processing time metrics
                        self._metrics['average_processing_time'] = (
                            self._metrics['average_processing_time'] * 0.9 + processing_time * 0.1
                        )
                        
                    except Exception as handler_error:
                        # Acknowledge failure
                        await self.acknowledge_message(
                            message,
                            processor_id,
                            False,
                            str(handler_error)
                        )
                        self._metrics['messages_failed'] += 1
                        
                        consumer.logger.error(f"Handler error for message {message.message_id}: {handler_error}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                consumer.logger.error(f"Error in consumer processor {processor_id}: {e}")
                await asyncio.sleep(5)
    
    async def get_queue_statistics(self, queue_name: str) -> Optional[QueueStats]:
        """Get queue statistics.
        
        Args:
            queue_name: Queue name
            
        Returns:
            Queue statistics
        """
        return self._queue_stats.get(queue_name)
    
    async def get_all_statistics(self) -> Dict[str, Any]:
        """Get comprehensive queue manager statistics."""
        try:
            queue_stats = {}
            for queue_name, stats in self._queue_stats.items():
                queue_stats[queue_name] = asdict(stats)
            
            return {
                'queue_manager_id': self.queue_manager_id,
                'active_queues': len(self._active_queues),
                'active_consumers': len(self._active_consumers),
                'queue_statistics': queue_stats,
                'performance_metrics': self._metrics.copy(),
                'capabilities': {
                    'dead_letter_enabled': self.enable_dead_letter,
                    'message_ttl_enabled': self.enable_message_ttl,
                    'priority_scheduling_enabled': self.enable_priority_scheduling
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}
    
    async def _timeout_monitor(self):
        """Monitor and handle message processing timeouts."""
        while not self._shutdown:
            try:
                current_time = time.time()
                
                for queue_name in self._active_queues:
                    processing_key = f"{self._processing_prefix}{queue_name}"
                    
                    # Get all processing messages
                    processing_messages = await self.redis_client.hgetall(processing_key)
                    
                    for message_id, processing_info in processing_messages.items():
                        if isinstance(message_id, bytes):
                            message_id = message_id.decode()
                        if isinstance(processing_info, bytes):
                            processing_info = processing_info.decode()
                        
                        try:
                            proc_data = json.loads(processing_info)
                            timeout_at = proc_data.get('timeout_at', 0)
                            
                            if current_time > timeout_at:
                                # Message timed out, return to queue
                                await self._handle_timeout(queue_name, message_id, proc_data)
                                
                        except json.JSONDecodeError:
                            continue
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in timeout monitor: {e}")
                await asyncio.sleep(60)
    
    async def _handle_timeout(self, queue_name: str, message_id: str, proc_data: Dict[str, Any]):
        """Handle message processing timeout."""
        try:
            # Remove from processing
            processing_key = f"{self._processing_prefix}{queue_name}"
            await self.redis_client.hdel(processing_key, message_id)
            
            # Return to queue with retry logic
            message_key = f"{self._message_prefix}{message_id}"
            retry_count = int(await self.redis_client.hget(message_key, 'retry_count') or 0)
            max_retries = int(await self.redis_client.hget(message_key, 'max_retries') or 3)
            
            if retry_count < max_retries:
                # Retry with backoff
                new_retry_count = retry_count + 1
                await self.redis_client.hset(message_key, 'retry_count', new_retry_count)
                await self.redis_client.hset(message_key, 'status', 'timeout_retry')
                
                queue_key = f"{self._queue_prefix}{queue_name}"
                delay = min(new_retry_count * new_retry_count * 60, 3600)  # Exponential backoff
                retry_time = time.time() + delay
                score = MessagePriority.NORMAL.value * 1000000 + retry_time
                
                await self.redis_client.zadd(queue_key, {message_id: score})
                
                self.logger.info(f"Message {message_id} timed out, retrying (attempt {new_retry_count})")
            else:
                # Send to dead letter queue
                await self.redis_client.hset(message_key, 'status', 'dead_letter')
                await self.redis_client.hset(message_key, 'dead_letter_reason', 'timeout')
                
                dead_letter_key = f"{self._dead_letter_prefix}{queue_name}"
                await self.redis_client.zadd(dead_letter_key, {message_id: time.time()})
                
                self.logger.warning(f"Message {message_id} sent to dead letter queue after timeout")
            
        except Exception as e:
            self.logger.error(f"Error handling timeout for message {message_id}: {e}")
    
    async def _register_queue_manager(self):
        """Register queue manager in distributed registry."""
        manager_data = {
            'queue_manager_id': self.queue_manager_id,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'status': 'active',
            'capabilities': {
                'dead_letter': self.enable_dead_letter,
                'message_ttl': self.enable_message_ttl,
                'priority_scheduling': self.enable_priority_scheduling
            },
            'active_queues': len(self._active_queues),
            'active_consumers': len(self._active_consumers)
        }
        
        await self.state_manager.update(
            f"queue_managers.{self.queue_manager_id}",
            manager_data,
            distributed=True
        )
    
    async def shutdown(self):
        """Shutdown queue manager."""
        self.logger.info(f"Shutting down Message Queue Manager: {self.queue_manager_id}")
        self._shutdown = True
        
        # Stop consumers
        for consumer in self._active_consumers.values():
            consumer.is_active = False
            for task in consumer._processing_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
        
        # Stop management tasks
        for task in self._management_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Update manager status
        await self.state_manager.update(
            f"queue_managers.{self.queue_manager_id}.status",
            'shutdown',
            distributed=True
        )
        
        self.logger.info(f"Message Queue Manager {self.queue_manager_id} shutdown complete")


# Global queue manager instance
_global_queue_manager: Optional[RedisMessageQueue] = None


async def get_message_queue(**kwargs) -> RedisMessageQueue:
    """Get global message queue manager instance."""
    global _global_queue_manager
    
    if _global_queue_manager is None:
        _global_queue_manager = RedisMessageQueue(**kwargs)
        await _global_queue_manager.initialize()
    
    return _global_queue_manager


async def create_message_queue(**kwargs) -> RedisMessageQueue:
    """Create new message queue manager instance."""
    manager = RedisMessageQueue(**kwargs)
    await manager.initialize()
    return manager