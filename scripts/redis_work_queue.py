"""
Redis-based Work Queue for Continuous AI

Uses Redis Streams for distributed, persistent work queue management
with support for priority-based scheduling and worker specialization.
"""

import json
import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from dataclasses import asdict

from scripts.work_item_types import WorkItem, TaskPriority
from scripts.redis_integration.redis_client import RedisClient
from scripts.redis_integration.redis_streams_manager import RedisStreamsManager
from scripts.mcp_redis_integration import MCPRedisIntegration


class RedisWorkQueue:
    """Redis-based distributed work queue using Redis Streams."""
    
    def __init__(self, redis_client: RedisClient = None, stream_name: str = "cwmai:work_queue"):
        """Initialize Redis work queue.
        
        Args:
            redis_client: Redis client instance
            stream_name: Name of the Redis stream for work items
        """
        self.redis_client = redis_client
        self.stream_name = stream_name
        self.logger = logging.getLogger(__name__)
        
        # Priority streams for different priority levels
        self.priority_streams = {
            TaskPriority.CRITICAL: f"{stream_name}:critical",
            TaskPriority.HIGH: f"{stream_name}:high",
            TaskPriority.MEDIUM: f"{stream_name}:medium",
            TaskPriority.LOW: f"{stream_name}:low",
            TaskPriority.BACKGROUND: f"{stream_name}:background"
        }
        
        # Consumer group for distributed processing
        self.consumer_group = "cwmai_workers"
        
        # Stream manager for advanced features
        self.stream_manager: Optional[RedisStreamsManager] = None
        
        # Local buffer for performance
        self._local_buffer: List[WorkItem] = []
        self._buffer_size = 10
        self._flush_interval = 5  # seconds
        self._flush_task: Optional[asyncio.Task] = None
        
        # Cached queue size for sync access
        self._cached_queue_size = 0
        self._cache_update_interval = 10  # seconds
        self._last_cache_update = 0
        
        self._initialized = False
        
        # MCP-Redis integration (optional, for enhanced features)
        self.mcp_redis: Optional[MCPRedisIntegration] = None
        self._use_mcp = os.getenv("USE_MCP_REDIS", "false").lower() == "true"
    
    async def initialize(self):
        """Initialize Redis connection and streams."""
        if self._initialized:
            return
        
        try:
            # Ensure we have a shared Redis client (do not create a new pool)
            if not self.redis_client:
                from scripts.redis_integration.redis_client import get_redis_client
                self.redis_client = await get_redis_client()
            
            # Initialize stream manager
            self.stream_manager = RedisStreamsManager(self.redis_client)
            
            # Create consumer groups for each priority stream
            for priority, stream in self.priority_streams.items():
                await self._ensure_stream_and_group(stream)
            
            # Start buffer flush task
            self._flush_task = asyncio.create_task(self._flush_loop())
            
            # Initialize MCP-Redis if enabled
            if self._use_mcp:
                try:
                    self.mcp_redis = MCPRedisIntegration()
                    await self.mcp_redis.initialize()
                    self.logger.info("MCP-Redis integration enabled")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize MCP-Redis: {e}")
                    self._use_mcp = False
            
            self._initialized = True
            self.logger.info("Redis work queue initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis work queue: {e}")
            raise
    
    async def _ensure_stream_and_group(self, stream_name: str):
        """Ensure stream exists and consumer group is created."""
        try:
            # Check if stream exists
            info = await self.redis_client.xinfo_stream(stream_name)
        except:
            # Stream doesn't exist, will be created on first add
            pass
        
        try:
            # Create consumer group
            await self.redis_client.xgroup_create(
                stream_name,
                self.consumer_group,
                id='0',
                mkstream=True
            )
            self.logger.debug(f"Created consumer group for {stream_name}")
        except:
            # Group already exists
            pass
    
    async def add_work(self, work_item: WorkItem):
        """Add a work item to the queue.
        
        Args:
            work_item: Work item to add
        """
        if not self._initialized:
            await self.initialize()
        
        # Add to local buffer for batching
        self._local_buffer.append(work_item)
        
        # Flush if buffer is full
        if len(self._local_buffer) >= self._buffer_size:
            await self._flush_buffer()
    
    async def add_work_batch(self, work_items: List[WorkItem]):
        """Add multiple work items to the queue.
        
        Args:
            work_items: List of work items to add
        """
        if not self._initialized:
            await self.initialize()
        
        # Add all to buffer
        self._local_buffer.extend(work_items)
        
        # Flush if needed
        if len(self._local_buffer) >= self._buffer_size:
            await self._flush_buffer()
    
    async def _flush_buffer(self):
        """Flush local buffer to Redis."""
        if not self._local_buffer:
            return
        
        try:
            # Group by priority
            priority_groups = {}
            for item in self._local_buffer:
                priority = item.priority
                if priority not in priority_groups:
                    priority_groups[priority] = []
                priority_groups[priority].append(item)
            
            # Add to respective priority streams
            for priority, items in priority_groups.items():
                # Debug logging
                self.logger.debug(f"Processing priority: {priority}, type: {type(priority)}, name: {getattr(priority, 'name', None)}, value: {getattr(priority, 'value', None)}")
                
                # Ensure priority is a TaskPriority enum
                if isinstance(priority, str):
                    try:
                        # Handle string representations like "TaskPriority.HIGH"
                        if priority.startswith("TaskPriority."):
                            priority_name = priority.split(".")[-1]
                            priority = TaskPriority[priority_name]
                        else:
                            priority = TaskPriority[priority]
                    except KeyError:
                        self.logger.error(f"Invalid priority string: {priority}. Unable to convert to TaskPriority enum.")
                        # Skip this batch to avoid data corruption
                        continue
                
                # Check if it's an enum by checking for name and value attributes
                if not (hasattr(priority, 'name') and hasattr(priority, 'value')):
                    self.logger.error(f"Priority is not a valid enum: {priority} (type: {type(priority)})")
                    continue
                
                # Try to find the matching enum by value
                matched_priority = None
                for enum_priority in TaskPriority:
                    if enum_priority.value == priority.value:
                        matched_priority = enum_priority
                        break
                
                if matched_priority is None:
                    self.logger.error(f"No TaskPriority enum found with value {priority.value}")
                    continue
                
                if matched_priority not in self.priority_streams:
                    # This should never happen if priority is a valid enum
                    self.logger.error(f"TaskPriority enum {matched_priority} not found in priority_streams dict. Available keys: {list(self.priority_streams.keys())}")
                    continue
                
                stream = self.priority_streams[matched_priority]
                
                # Use pipeline for efficiency
                async with self.redis_client.pipeline() as pipe:
                    for item in items:
                        # Convert WorkItem to Redis-compatible format
                        data = self._serialize_work_item(item)
                        await pipe.xadd(stream, data)
                    
                    results = await pipe.execute()
                    # Log success for each item
                    successful_adds = sum(1 for r in results if r)
                    self.logger.info(f"Successfully added {successful_adds}/{len(items)} items to {matched_priority.name} stream")
                    
                    if successful_adds != len(items):
                        self.logger.error(f"Failed to add {len(items) - successful_adds} items to Redis")
                
                self.logger.info(f"Flushed {len(items)} {matched_priority.name} items to Redis stream: {stream}")
            
            # Clear buffer
            self._local_buffer.clear()
            
        except KeyError as e:
            # This happens when a priority enum is not in priority_streams
            self.logger.error(f"Error flushing buffer to Redis - unknown priority: {e}")
            # Keep items in buffer for retry
        except Exception as e:
            self.logger.error(f"Error flushing buffer to Redis: {type(e).__name__}: {e}")
            # Keep items in buffer for retry
    
    async def _flush_loop(self):
        """Periodically flush buffer to Redis and update cache."""
        while True:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush_buffer()
                
                # Update cached queue size if needed
                current_time = asyncio.get_event_loop().time()
                if current_time - self._last_cache_update > self._cache_update_interval:
                    await self.get_queue_stats()  # This updates the cache
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in flush loop: {e}")
    
    def _serialize_work_item(self, work_item: WorkItem) -> Dict[str, str]:
        """Serialize WorkItem for Redis storage."""
        data = {
            'id': str(work_item.id),
            'task_type': str(work_item.task_type),
            'title': str(work_item.title),
            'description': str(work_item.description),
            'priority': str(work_item.priority.name),
            'priority_value': str(work_item.priority.value),
            'estimated_cycles': str(work_item.estimated_cycles),
            'created_at': work_item.created_at.isoformat() if work_item.created_at else ''
        }
        
        if work_item.repository:
            data['repository'] = str(work_item.repository)
        
        if work_item.dependencies:
            data['dependencies'] = json.dumps(work_item.dependencies)
        
        if work_item.metadata:
            # Ensure metadata is properly serialized
            try:
                data['metadata'] = json.dumps(work_item.metadata)
            except (TypeError, ValueError) as e:
                self.logger.warning(f"Failed to serialize metadata for {work_item.id}: {e}")
                data['metadata'] = json.dumps({})
        
        if work_item.assigned_worker:
            data['assigned_worker'] = str(work_item.assigned_worker)
        
        if work_item.started_at:
            data['started_at'] = work_item.started_at.isoformat()
        
        if work_item.completed_at:
            data['completed_at'] = work_item.completed_at.isoformat()
        
        # Ensure all values are strings
        return {k: str(v) for k, v in data.items()}
    
    def _deserialize_work_item(self, data: Dict[bytes, bytes]) -> WorkItem:
        """Deserialize WorkItem from Redis storage."""
        # Decode bytes to strings
        decoded = {}
        for k, v in data.items():
            # Handle both bytes and string keys/values
            key = k.decode() if isinstance(k, bytes) else k
            value = v.decode() if isinstance(v, bytes) else v
            decoded[key] = value
        
        # Parse priority
        priority = TaskPriority[decoded['priority']]
        
        # Create WorkItem
        work_item = WorkItem(
            id=decoded['id'],
            task_type=decoded['task_type'],
            title=decoded['title'],
            description=decoded['description'],
            priority=priority,
            repository=decoded.get('repository'),
            estimated_cycles=int(decoded.get('estimated_cycles', 1)),
            dependencies=json.loads(decoded.get('dependencies', '[]')),
            metadata=json.loads(decoded.get('metadata', '{}')),
            created_at=datetime.fromisoformat(decoded['created_at'])
        )
        
        # Set optional fields
        if 'assigned_worker' in decoded:
            work_item.assigned_worker = decoded['assigned_worker']
        
        if 'started_at' in decoded:
            work_item.started_at = datetime.fromisoformat(decoded['started_at'])
        
        if 'completed_at' in decoded:
            work_item.completed_at = datetime.fromisoformat(decoded['completed_at'])
        
        return work_item
    
    async def get_work_for_worker(self, worker_id: str, 
                                 specialization: Optional[str] = None,
                                 count: int = 1) -> List[WorkItem]:
        """Get work items for a specific worker.
        
        Args:
            worker_id: ID of the worker
            specialization: Worker's specialization (repository or task type)
            count: Maximum number of items to retrieve
            
        Returns:
            List of work items assigned to the worker
        """
        if not self._initialized:
            await self.initialize()
        
        work_items = []
        
        # Flush buffer first to ensure all work is available
        await self._flush_buffer()
        
        # Check priority streams in order
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, 
                        TaskPriority.MEDIUM, TaskPriority.LOW, TaskPriority.BACKGROUND]:
            
            if len(work_items) >= count:
                break
            
            stream = self.priority_streams[priority]
            
            try:
                self.logger.debug(f"Checking {priority.name} stream: {stream}")
                
                # First try to claim pending messages that might be stuck
                try:
                    # Get pending messages
                    pending_messages = await self.redis_client.xpending_range(
                        stream,
                        self.consumer_group,
                        start='-',
                        end='+',
                        count=count - len(work_items)
                    )
                    
                    self.logger.debug(f"Found {len(pending_messages)} pending messages in {priority.name}")
                    
                    if pending_messages:
                        # Claim old pending messages (older than 1 second to be more aggressive)
                        message_ids = [msg['message_id'] for msg in pending_messages 
                                     if msg['time_since_delivered'] > 1000]  # 1 second
                        
                        if message_ids:
                            self.logger.debug(f"Claiming {len(message_ids)} pending messages from {priority.name}")
                            claimed = await self.redis_client.xclaim(
                                stream,
                                self.consumer_group,
                                worker_id,
                                1000,  # min idle time
                                message_ids
                            )
                            
                            if claimed:
                                # Process claimed messages
                                messages = [(stream.encode() if isinstance(stream, str) else stream, claimed)]
                                self.logger.debug(f"Claimed {len(claimed)} messages")
                            else:
                                messages = []
                                self.logger.debug("No messages claimed")
                        else:
                            messages = []
                            self.logger.debug(f"No messages old enough to claim (all < 1s)")
                    else:
                        messages = []
                except Exception as e:
                    self.logger.debug(f"Error handling pending messages: {e}")
                    import traceback
                    self.logger.debug(traceback.format_exc())
                    messages = []
                
                # If no pending messages claimed, try reading new messages
                if not messages:
                    messages = await self.redis_client.xreadgroup(
                        self.consumer_group,
                        worker_id,
                        {stream: '>'},  # Read only new messages
                        count=count - len(work_items),
                        block=100  # Block for 100ms to wait for new messages
                    )
                
                self.logger.debug(f"Messages to process: {len(messages) if messages else 0}")
                
                if messages:
                    self.logger.debug(f"Processing {len(messages)} message groups")
                    for stream_name, stream_messages in messages:
                        self.logger.debug(f"Stream {stream_name}: {len(stream_messages)} messages")
                        for msg_id, data in stream_messages:
                            work_item = self._deserialize_work_item(data)
                            
                            # Check if worker can handle this work
                            if self._can_worker_handle(worker_id, specialization, work_item):
                                work_item.assigned_worker = worker_id
                                work_item.started_at = datetime.now(timezone.utc)
                                work_items.append(work_item)
                                
                                # Acknowledge the message immediately
                                await self.redis_client.xack(stream_name, self.consumer_group, msg_id)
                                self.logger.info(f"Worker {worker_id} claimed task {work_item.id}")
                            else:
                                # Acknowledge but don't process - prevents message from getting stuck
                                # Re-add to appropriate stream for other workers
                                await self.redis_client.xack(stream_name, self.consumer_group, msg_id)
                                await self.redis_client.xadd(stream_name, self._serialize_work_item(work_item))
                                self.logger.debug(f"Worker {worker_id} cannot handle {work_item.id}, re-queued for others")
                
            except Exception as e:
                self.logger.error(f"Error reading from {priority.name} stream: {e}")
        
        return work_items
    
    def _can_worker_handle(self, worker_id: str, specialization: Optional[str], 
                          work_item: WorkItem) -> bool:
        """Check if a worker can handle a specific work item."""
        # General workers can handle anything
        if not specialization or specialization == "general":
            self.logger.debug(f"Worker {worker_id} is general, can handle {work_item.id}")
            return True
        
        if specialization == "system_tasks":
            # System worker handles system-wide tasks
            can_handle = work_item.repository is None
            self.logger.debug(f"Worker {worker_id} is system_tasks, can handle {work_item.id}: {can_handle}")
            return can_handle
        
        # Check repository match
        if work_item.repository and specialization == work_item.repository:
            self.logger.debug(f"Worker {worker_id} matches repository {work_item.repository}")
            return True
        
        # Check task type match
        if specialization == work_item.task_type:
            self.logger.debug(f"Worker {worker_id} matches task type {work_item.task_type}")
            return True
        
        self.logger.debug(f"Worker {worker_id} cannot handle {work_item.id}")
        return False
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the work queue.
        
        Returns:
            Dictionary with queue statistics
        """
        if not self._initialized:
            await self.initialize()
        
        stats = {
            'buffer_size': len(self._local_buffer),
            'priority_queues': {}
        }
        
        # Get stats for each priority stream
        for priority, stream in self.priority_streams.items():
            try:
                info = await self.redis_client.xinfo_stream(stream)
                
                stats['priority_queues'][priority.name] = {
                    'length': info.get('length', 0),
                    'first_entry': info.get('first-entry'),
                    'last_entry': info.get('last-entry'),
                    'consumer_groups': info.get('groups', 0)
                }
                
                # Get pending messages
                try:
                    pending_info = await self.redis_client.xpending(
                        stream,
                        self.consumer_group
                    )
                    
                    if pending_info and isinstance(pending_info, dict):
                        # xpending returns {'name': group, 'pending': count, ...}
                        stats['priority_queues'][priority.name]['pending'] = pending_info.get('pending', 0)
                    elif pending_info and isinstance(pending_info, (list, tuple)) and len(pending_info) > 0:
                        # Some versions return [count, ...]
                        stats['priority_queues'][priority.name]['pending'] = pending_info[0]
                    else:
                        stats['priority_queues'][priority.name]['pending'] = 0
                except Exception as e:
                    self.logger.debug(f"Error getting pending for {priority.name}: {e}")
                    stats['priority_queues'][priority.name]['pending'] = 0
                
            except Exception as e:
                self.logger.debug(f"No stats for {priority.name}: {e}")
                stats['priority_queues'][priority.name] = {'length': 0}
        
        # Calculate totals based on pending (truly queued) rather than raw stream length
        total_pending = sum(
            q.get('pending', 0)
            for q in stats['priority_queues'].values()
        )
        stats['total_queued'] = total_pending + len(self._local_buffer)
        stats['total_pending'] = total_pending
        
        # Update cached queue size
        self._cached_queue_size = stats['total_queued']
        self._last_cache_update = asyncio.get_event_loop().time()
        
        return stats
    
    def get_queue_size_sync(self) -> int:
        """Get queue size synchronously using cached value.
        
        Returns:
            Cached queue size (may be up to cache_update_interval seconds old)
        """
        # Return cached value plus current buffer size
        return self._cached_queue_size + len(self._local_buffer)
    
    async def cleanup_completed_work(self, older_than_hours: int = 24):
        """Clean up old completed work items.
        
        Args:
            older_than_hours: Remove items older than this many hours
        """
        if not self._initialized:
            await self.initialize()
        
        cutoff_time = int((datetime.now(timezone.utc).timestamp() - 
                          older_than_hours * 3600) * 1000)
        
        cleaned = 0
        
        for stream in self.priority_streams.values():
            try:
                # Trim stream to remove old entries
                # This removes entries older than cutoff_time
                result = await self.redis_client.xtrim(
                    stream,
                    minid=f"{cutoff_time}-0"
                )
                cleaned += result
            except Exception as e:
                self.logger.error(f"Error cleaning up {stream}: {e}")
        
        if cleaned > 0:
            self.logger.info(f"Cleaned up {cleaned} old work items")
        
        return cleaned
    
    async def requeue_stuck_work(self, timeout_minutes: int = 30) -> int:
        """Requeue work items that have been stuck in pending state.
        
        Args:
            timeout_minutes: Consider work stuck after this many minutes
            
        Returns:
            Number of items requeued
        """
        if not self._initialized:
            await self.initialize()
        
        requeued = 0
        timeout_ms = timeout_minutes * 60 * 1000
        
        for priority, stream in self.priority_streams.items():
            try:
                # Get pending messages
                pending = await self.redis_client.xpending_range(
                    stream,
                    self.consumer_group,
                    min='-',
                    max='+',
                    count=100
                )
                
                for entry in pending:
                    msg_id = entry['message_id']
                    consumer = entry['consumer']
                    idle_time = entry['time_since_delivered']
                    
                    if idle_time > timeout_ms:
                        # Claim the message for reprocessing
                        claimed = await self.redis_client.xclaim(
                            stream,
                            self.consumer_group,
                            'requeue_worker',
                            idle_time,
                            [msg_id]
                        )
                        
                        if claimed:
                            # Mark as available for other workers
                            await self.redis_client.xack(
                                stream,
                                self.consumer_group,
                                msg_id
                            )
                            requeued += 1
                            
                            self.logger.info(
                                f"Requeued stuck {priority.name} work item "
                                f"(was assigned to {consumer}, idle for {idle_time//1000}s)"
                            )
                
            except Exception as e:
                self.logger.error(f"Error requeuing stuck work from {stream}: {e}")
        
        return requeued
    
    async def cleanup(self):
        """Clean up resources."""
        # Flush remaining buffer
        await self._flush_buffer()
        
        # Cancel flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Close MCP-Redis if initialized
        if self.mcp_redis:
            await self.mcp_redis.close()
        
        self._initialized = False
        self.logger.info("Redis work queue cleaned up")
    
    # MCP-Redis Enhanced Methods
    async def find_similar_tasks(self, description: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar tasks using MCP-Redis natural language search"""
        if not self._use_mcp or not self.mcp_redis:
            self.logger.warning("MCP-Redis not available for similarity search")
            return []
        
        try:
            return await self.mcp_redis.find_similar_tasks(description, limit)
        except Exception as e:
            self.logger.error(f"Error finding similar tasks: {e}")
            return []
    
    async def get_intelligent_queue_insights(self) -> Dict[str, Any]:
        """Get intelligent insights about queue state using MCP-Redis"""
        if not self._use_mcp or not self.mcp_redis:
            # Fallback to basic stats
            return await self.get_queue_stats()
        
        try:
            # Use natural language query for deeper insights
            insights = await self.mcp_redis.execute("""
                Analyze the work queue and provide:
                - Task distribution by priority and type
                - Average wait time by priority
                - Worker utilization rates
                - Bottlenecks or stuck tasks
                - Recommendations for optimization
            """)
            return insights
        except Exception as e:
            self.logger.error(f"Error getting queue insights: {e}")
            return await self.get_queue_stats()
    
    async def optimize_task_assignment(self, worker_id: str, specialization: Optional[str] = None) -> Optional[WorkItem]:
        """Use MCP-Redis to intelligently assign the best task to a worker"""
        if not self._use_mcp or not self.mcp_redis:
            # Fallback to regular assignment
            items = await self.get_work_for_worker(worker_id, specialization, count=1)
            return items[0] if items else None
        
        try:
            # Use natural language to find the best task
            result = await self.mcp_redis.execute(f"""
                Find the best task for worker '{worker_id}' with specialization '{specialization}':
                - Consider task priority and age
                - Match worker specialization to task type or repository
                - Avoid tasks that have failed multiple times
                - Prefer tasks similar to what this worker has successfully completed before
                - Return the single best matching task
            """)
            
            if result:
                return self._deserialize_work_item(result)
            return None
        except Exception as e:
            self.logger.error(f"Error in intelligent task assignment: {e}")
            items = await self.get_work_for_worker(worker_id, specialization, count=1)
            return items[0] if items else None