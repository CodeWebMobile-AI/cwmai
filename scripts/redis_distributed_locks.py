"""
Redis Distributed Locking System

Advanced distributed locking implementation with deadlock prevention,
lock leasing, priority queuing, and comprehensive lock management.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Set, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

from scripts.redis_integration import get_redis_client
from scripts.redis_lockfree_adapter import create_lockfree_state_manager


class LockType(Enum):
    """Types of distributed locks."""
    EXCLUSIVE = "exclusive"        # Traditional exclusive lock
    SHARED = "shared"             # Reader-writer shared lock
    SEMAPHORE = "semaphore"       # Counting semaphore
    FAIR_QUEUE = "fair_queue"     # Fair queuing lock
    PRIORITY = "priority"         # Priority-based lock
    TIMEOUT = "timeout"           # Auto-expiring lock


class LockPriority(Enum):
    """Lock acquisition priorities."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class LockRequest:
    """Lock acquisition request."""
    request_id: str
    lock_name: str
    lock_type: LockType
    requester_id: str
    priority: LockPriority
    timeout_seconds: int
    requested_at: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert lock request to dictionary."""
        return {
            'request_id': self.request_id,
            'lock_name': self.lock_name,
            'lock_type': self.lock_type.value,
            'requester_id': self.requester_id,
            'priority': self.priority.value,
            'timeout_seconds': self.timeout_seconds,
            'requested_at': self.requested_at.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LockRequest':
        """Create lock request from dictionary."""
        return cls(
            request_id=data['request_id'],
            lock_name=data['lock_name'],
            lock_type=LockType(data['lock_type']),
            requester_id=data['requester_id'],
            priority=LockPriority(data['priority']),
            timeout_seconds=data['timeout_seconds'],
            requested_at=datetime.fromisoformat(data['requested_at']),
            metadata=data.get('metadata', {})
        )


@dataclass
class LockInfo:
    """Information about an acquired lock."""
    lock_id: str
    lock_name: str
    lock_type: LockType
    owner_id: str
    acquired_at: datetime
    expires_at: datetime
    lease_duration: int
    metadata: Dict[str, Any]
    readers: Set[str] = None  # For shared locks
    
    def __post_init__(self):
        if self.readers is None:
            self.readers = set()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert lock info to dictionary."""
        return {
            'lock_id': self.lock_id,
            'lock_name': self.lock_name,
            'lock_type': self.lock_type.value,
            'owner_id': self.owner_id,
            'acquired_at': self.acquired_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'lease_duration': self.lease_duration,
            'metadata': self.metadata,
            'readers': list(self.readers)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LockInfo':
        """Create lock info from dictionary."""
        lock_info = cls(
            lock_id=data['lock_id'],
            lock_name=data['lock_name'],
            lock_type=LockType(data['lock_type']),
            owner_id=data['owner_id'],
            acquired_at=datetime.fromisoformat(data['acquired_at']),
            expires_at=datetime.fromisoformat(data['expires_at']),
            lease_duration=data['lease_duration'],
            metadata=data.get('metadata', {}),
            readers=set(data.get('readers', []))
        )
        return lock_info


class DeadlockDetector:
    """Deadlock detection and prevention system."""
    
    def __init__(self, redis_client):
        """Initialize deadlock detector.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis_client = redis_client
        self.logger = logging.getLogger(f"{__name__}.DeadlockDetector")
        
        # Deadlock detection state
        self._wait_graph_key = "deadlock:wait_graph"
        self._detection_interval = 30  # Check every 30 seconds
        self._detection_task: Optional[asyncio.Task] = None
        self._shutdown = False
    
    async def start_detection(self):
        """Start deadlock detection process."""
        if not self._detection_task:
            self._detection_task = asyncio.create_task(self._detection_loop())
            self.logger.info("Deadlock detection started")
    
    async def stop_detection(self):
        """Stop deadlock detection process."""
        self._shutdown = True
        if self._detection_task:
            self._detection_task.cancel()
            try:
                await self._detection_task
            except asyncio.CancelledError:
                pass
            self._detection_task = None
        self.logger.info("Deadlock detection stopped")
    
    async def add_wait_edge(self, waiting_node: str, holding_node: str, lock_name: str):
        """Add edge to wait-for graph.
        
        Args:
            waiting_node: Node waiting for lock
            holding_node: Node holding lock
            lock_name: Name of lock
        """
        try:
            wait_edge = {
                'waiting': waiting_node,
                'holding': holding_node,
                'lock': lock_name,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            edge_key = f"{waiting_node}:{holding_node}:{lock_name}"
            await self.redis_client.hset(
                self._wait_graph_key,
                edge_key,
                json.dumps(wait_edge)
            )
            
        except Exception as e:
            self.logger.error(f"Error adding wait edge: {e}")
    
    async def remove_wait_edge(self, waiting_node: str, holding_node: str, lock_name: str):
        """Remove edge from wait-for graph.
        
        Args:
            waiting_node: Node that was waiting
            holding_node: Node that was holding
            lock_name: Name of lock
        """
        try:
            edge_key = f"{waiting_node}:{holding_node}:{lock_name}"
            await self.redis_client.hdel(self._wait_graph_key, edge_key)
            
        except Exception as e:
            self.logger.error(f"Error removing wait edge: {e}")
    
    async def _detection_loop(self):
        """Main deadlock detection loop."""
        while not self._shutdown:
            try:
                await self._detect_deadlocks()
                await asyncio.sleep(self._detection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in deadlock detection loop: {e}")
                await asyncio.sleep(5)
    
    async def _detect_deadlocks(self):
        """Detect deadlocks in wait-for graph."""
        try:
            # Get wait-for graph
            wait_graph = await self._build_wait_graph()
            
            if not wait_graph:
                return
            
            # Find cycles using DFS
            deadlocks = self._find_cycles(wait_graph)
            
            if deadlocks:
                self.logger.warning(f"Detected {len(deadlocks)} deadlock(s)")
                
                for deadlock in deadlocks:
                    await self._resolve_deadlock(deadlock)
            
        except Exception as e:
            self.logger.error(f"Error detecting deadlocks: {e}")
    
    async def _build_wait_graph(self) -> Dict[str, List[str]]:
        """Build wait-for graph from Redis data."""
        try:
            if not self.redis_client:
                return {}
            wait_edges = await self.redis_client.hgetall(self._wait_graph_key)
            wait_graph = {}
            
            for edge_key, edge_data in wait_edges.items():
                if isinstance(edge_key, bytes):
                    edge_key = edge_key.decode()
                if isinstance(edge_data, bytes):
                    edge_data = edge_data.decode()
                
                edge = json.loads(edge_data)
                waiting = edge['waiting']
                holding = edge['holding']
                
                if waiting not in wait_graph:
                    wait_graph[waiting] = []
                wait_graph[waiting].append(holding)
            
            return wait_graph
            
        except Exception as e:
            self.logger.error(f"Error building wait graph: {e}")
            return {}
    
    def _find_cycles(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Find cycles in wait-for graph using DFS.
        
        Args:
            graph: Wait-for graph
            
        Returns:
            List of cycles (deadlocks)
        """
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> bool:
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if dfs(neighbor, path):
                    return True
            
            rec_stack.remove(node)
            path.pop()
            return False
        
        for node in graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    async def _resolve_deadlock(self, deadlock: List[str]):
        """Resolve detected deadlock.
        
        Args:
            deadlock: List of nodes in deadlock cycle
        """
        try:
            self.logger.warning(f"Resolving deadlock: {' -> '.join(deadlock)}")
            
            # Simple resolution: abort the youngest transaction
            # This could be made more sophisticated based on priorities, costs, etc.
            
            # For now, we'll just log the deadlock
            # In a real implementation, you'd implement deadlock resolution strategy
            
            deadlock_info = {
                'deadlock_id': str(uuid.uuid4()),
                'nodes': deadlock,
                'detected_at': datetime.now(timezone.utc).isoformat(),
                'resolution_strategy': 'logged'
            }
            
            # Store deadlock information
            await self.redis_client.setex(
                f"deadlock:detected:{deadlock_info['deadlock_id']}",
                3600,  # Keep for 1 hour
                json.dumps(deadlock_info)
            )
            
        except Exception as e:
            self.logger.error(f"Error resolving deadlock: {e}")


class RedisDistributedLockManager:
    """Advanced distributed lock manager with comprehensive features."""
    
    def __init__(self,
                 manager_id: str = None,
                 enable_deadlock_detection: bool = True,
                 enable_lock_monitoring: bool = True,
                 default_lease_duration: int = 30):
        """Initialize distributed lock manager.
        
        Args:
            manager_id: Unique manager identifier
            enable_deadlock_detection: Enable deadlock detection
            enable_lock_monitoring: Enable lock monitoring
            default_lease_duration: Default lock lease duration in seconds
        """
        self.manager_id = manager_id or f"lock_mgr_{uuid.uuid4().hex[:8]}"
        self.enable_deadlock_detection = enable_deadlock_detection
        self.enable_lock_monitoring = enable_lock_monitoring
        self.default_lease_duration = default_lease_duration
        
        self.logger = logging.getLogger(f"{__name__}.RedisDistributedLockManager")
        
        # Redis components
        self.redis_client = None
        self.state_manager = None
        
        # Lock management
        self.deadlock_detector: Optional[DeadlockDetector] = None
        self._active_locks: Dict[str, LockInfo] = {}
        self._lock_queues: Dict[str, List[LockRequest]] = {}
        
        # Background tasks
        self._management_tasks: List[asyncio.Task] = []
        self._shutdown = False
        
        # Performance metrics
        self._metrics = {
            'locks_acquired': 0,
            'locks_released': 0,
            'locks_expired': 0,
            'lock_timeouts': 0,
            'deadlocks_detected': 0,
            'average_wait_time': 0.0,
            'total_wait_time': 0.0
        }
        
        # Lock prefixes
        self._lock_prefix = "lock:"
        self._queue_prefix = "queue:"
        self._semaphore_prefix = "semaphore:"
    
    async def initialize(self):
        """Initialize distributed lock manager."""
        try:
            self.logger.info(f"Initializing Redis Distributed Lock Manager: {self.manager_id}")
            
            # Initialize Redis components
            self.redis_client = await get_redis_client()
            
            # Check if redis client was successfully created
            if not self.redis_client:
                raise ValueError("Failed to get Redis client - returned None")
            
            self.logger.debug(f"Redis client initialized: {type(self.redis_client)}")
            
            self.state_manager = create_lockfree_state_manager(f"distributed_locks_{self.manager_id}")
            await self.state_manager.initialize()
            
            # Initialize deadlock detection
            if self.enable_deadlock_detection:
                self.deadlock_detector = DeadlockDetector(self.redis_client)
                await self.deadlock_detector.start_detection()
            
            # Start management tasks
            await self._start_management_tasks()
            
            # Register lock manager
            await self._register_lock_manager()
            
            self.logger.info(f"Lock Manager {self.manager_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Lock Manager: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    async def _start_management_tasks(self):
        """Start background management tasks."""
        try:
            # TODO: Implement background tasks
            # Lock expiration monitor
            # expiration_task = asyncio.create_task(self._lock_expiration_monitor())
            # self._management_tasks.append(expiration_task)
            
            # Queue processor
            # queue_task = asyncio.create_task(self._queue_processor())
            # self._management_tasks.append(queue_task)
            
            # Lock monitoring
            # if self.enable_lock_monitoring:
            #     monitor_task = asyncio.create_task(self._lock_monitor())
            #     self._management_tasks.append(monitor_task)
            
            self.logger.info(f"Started {len(self._management_tasks)} management tasks")
            
        except Exception as e:
            self.logger.error(f"Error starting management tasks: {e}")
    
    async def acquire_lock(self,
                          lock_name: str,
                          requester_id: str,
                          lock_type: LockType = LockType.EXCLUSIVE,
                          priority: LockPriority = LockPriority.NORMAL,
                          timeout_seconds: int = None,
                          lease_duration: int = None,
                          metadata: Dict[str, Any] = None) -> Optional[str]:
        """Acquire distributed lock.
        
        Args:
            lock_name: Name of lock to acquire
            requester_id: ID of requester
            lock_type: Type of lock
            priority: Lock priority
            timeout_seconds: Acquisition timeout
            lease_duration: Lock lease duration
            metadata: Additional metadata
            
        Returns:
            Lock ID if acquired, None otherwise
        """
        try:
            # Create lock request
            request = LockRequest(
                request_id=str(uuid.uuid4()),
                lock_name=lock_name,
                lock_type=lock_type,
                requester_id=requester_id,
                priority=priority,
                timeout_seconds=timeout_seconds or 60,
                requested_at=datetime.now(timezone.utc),
                metadata=metadata or {}
            )
            
            self.logger.debug(f"Lock acquisition request: {lock_name} by {requester_id}")
            
            # Check if lock can be acquired immediately
            immediate_lock = await self._try_immediate_acquisition(request, lease_duration)
            if immediate_lock:
                return immediate_lock
            
            # Add to queue if not immediately available
            await self._add_to_queue(request)
            
            # Wait for lock acquisition
            return await self._wait_for_lock(request, lease_duration)
            
        except Exception as e:
            self.logger.error(f"Error acquiring lock {lock_name}: {e}")
            return None
    
    async def _try_immediate_acquisition(self, 
                                       request: LockRequest, 
                                       lease_duration: int = None) -> Optional[str]:
        """Try to acquire lock immediately.
        
        Args:
            request: Lock request
            lease_duration: Lock lease duration
            
        Returns:
            Lock ID if acquired, None otherwise
        """
        try:
            lock_key = f"{self._lock_prefix}{request.lock_name}"
            lease_duration = lease_duration or self.default_lease_duration
            
            if request.lock_type == LockType.EXCLUSIVE:
                return await self._acquire_exclusive_lock(request, lock_key, lease_duration)
            
            elif request.lock_type == LockType.SHARED:
                return await self._acquire_shared_lock(request, lock_key, lease_duration)
            
            elif request.lock_type == LockType.SEMAPHORE:
                return await self._acquire_semaphore_lock(request, lock_key, lease_duration)
            
            elif request.lock_type == LockType.FAIR_QUEUE:
                # Fair queue locks always go through the queue
                return None
            
            else:
                return await self._acquire_exclusive_lock(request, lock_key, lease_duration)
            
        except Exception as e:
            self.logger.error(f"Error in immediate acquisition: {e}")
            return None
    
    async def _acquire_exclusive_lock(self, 
                                    request: LockRequest, 
                                    lock_key: str, 
                                    lease_duration: int) -> Optional[str]:
        """Acquire exclusive lock.
        
        Args:
            request: Lock request
            lock_key: Redis lock key
            lease_duration: Lock lease duration
            
        Returns:
            Lock ID if acquired, None otherwise
        """
        try:
            if not self.redis_client:
                self.logger.error("Redis client not initialized")
                return None
            # Try to set lock with expiration
            lock_id = str(uuid.uuid4())
            lock_value = json.dumps({
                'lock_id': lock_id,
                'owner': request.requester_id,
                'acquired_at': datetime.now(timezone.utc).isoformat(),
                'type': request.lock_type.value,
                'metadata': request.metadata
            })
            
            # Use SET NX EX for atomic acquisition
            success = await self.redis_client.set(
                lock_key,
                lock_value,
                nx=True,  # Only set if not exists
                ex=lease_duration  # Expiration
            )
            
            if success:
                # Create lock info
                lock_info = LockInfo(
                    lock_id=lock_id,
                    lock_name=request.lock_name,
                    lock_type=request.lock_type,
                    owner_id=request.requester_id,
                    acquired_at=datetime.now(timezone.utc),
                    expires_at=datetime.now(timezone.utc) + timedelta(seconds=lease_duration),
                    lease_duration=lease_duration,
                    metadata=request.metadata
                )
                
                self._active_locks[lock_id] = lock_info
                self._metrics['locks_acquired'] += 1
                
                self.logger.debug(f"Acquired exclusive lock: {request.lock_name}")
                return lock_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error acquiring exclusive lock: {e}")
            return None
    
    async def _acquire_shared_lock(self, 
                                 request: LockRequest, 
                                 lock_key: str, 
                                 lease_duration: int) -> Optional[str]:
        """Acquire shared (reader) lock.
        
        Args:
            request: Lock request
            lock_key: Redis lock key
            lease_duration: Lock lease duration
            
        Returns:
            Lock ID if acquired, None otherwise
        """
        try:
            shared_key = f"{lock_key}:shared"
            readers_key = f"{lock_key}:readers"
            writer_key = f"{lock_key}:writer"
            
            # Check if there's an active writer
            writer = await self.redis_client.get(writer_key)
            if writer:
                return None  # Cannot acquire shared lock when writer present
            
            # Add to readers set
            lock_id = str(uuid.uuid4())
            reader_info = {
                'lock_id': lock_id,
                'reader_id': request.requester_id,
                'acquired_at': datetime.now(timezone.utc).isoformat(),
                'expires_at': (datetime.now(timezone.utc) + timedelta(seconds=lease_duration)).isoformat()
            }
            
            await self.redis_client.hset(readers_key, lock_id, json.dumps(reader_info))
            await self.redis_client.expire(readers_key, lease_duration)
            
            # Create lock info
            lock_info = LockInfo(
                lock_id=lock_id,
                lock_name=request.lock_name,
                lock_type=request.lock_type,
                owner_id=request.requester_id,
                acquired_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=lease_duration),
                lease_duration=lease_duration,
                metadata=request.metadata
            )
            
            self._active_locks[lock_id] = lock_info
            self._metrics['locks_acquired'] += 1
            
            self.logger.debug(f"Acquired shared lock: {request.lock_name}")
            return lock_id
            
        except Exception as e:
            self.logger.error(f"Error acquiring shared lock: {e}")
            return None
    
    async def _acquire_semaphore_lock(self, 
                                    request: LockRequest, 
                                    lock_key: str, 
                                    lease_duration: int) -> Optional[str]:
        """Acquire semaphore lock.
        
        Args:
            request: Lock request
            lock_key: Redis lock key
            lease_duration: Lock lease duration
            
        Returns:
            Lock ID if acquired, None otherwise
        """
        try:
            semaphore_key = f"{self._semaphore_prefix}{request.lock_name}"
            max_count = request.metadata.get('max_count', 1)
            
            # Get current semaphore count
            current_count = await self.redis_client.get(f"{semaphore_key}:count")
            current_count = int(current_count) if current_count else 0
            
            if current_count >= max_count:
                return None  # Semaphore full
            
            # Increment semaphore count
            lock_id = str(uuid.uuid4())
            
            # Use pipeline for atomicity
            # Get the actual redis connection and create pipeline
            if hasattr(self.redis_client, 'redis') and self.redis_client.redis:
                pipe = self.redis_client.redis.pipeline()
            else:
                # Fallback - execute commands individually
                await self.redis_client.incr(f"{semaphore_key}:count")
                await self.redis_client.expire(f"{semaphore_key}:count", lease_duration)
                await self.redis_client.hset(f"{semaphore_key}:holders", lock_id, json.dumps({
                    'holder_id': request.requester_id,
                    'acquired_at': datetime.now(timezone.utc).isoformat(),
                    'expires_at': (datetime.now(timezone.utc) + timedelta(seconds=lease_duration)).isoformat()
                }))
                await self.redis_client.expire(f"{semaphore_key}:holders", lease_duration)
                
                current_count = await self.redis_client.get(f"{semaphore_key}:count")
                current_count = int(current_count) if current_count else 0
                
                if current_count <= max_count:
                    # Create lock info and return
                    lock_info = LockInfo(
                        lock_id=lock_id,
                        lock_name=request.lock_name,
                        lock_type=request.lock_type,
                        owner_id=request.requester_id,
                        acquired_at=datetime.now(timezone.utc),
                        expires_at=datetime.now(timezone.utc) + timedelta(seconds=lease_duration),
                        lease_duration=lease_duration,
                        metadata=request.metadata
                    )
                    
                    self._active_locks[lock_id] = lock_info
                    self._metrics['locks_acquired'] += 1
                    
                    self.logger.debug(f"Acquired semaphore lock: {request.lock_name}")
                    return lock_id
                else:
                    # Rollback
                    await self.redis_client.decr(f"{semaphore_key}:count")
                    await self.redis_client.hdel(f"{semaphore_key}:holders", lock_id)
                    return None
            
            # Original pipeline code
            pipe = self.redis_client.redis.pipeline()
            pipe.incr(f"{semaphore_key}:count")
            pipe.expire(f"{semaphore_key}:count", lease_duration)
            pipe.hset(f"{semaphore_key}:holders", lock_id, json.dumps({
                'holder_id': request.requester_id,
                'acquired_at': datetime.now(timezone.utc).isoformat(),
                'expires_at': (datetime.now(timezone.utc) + timedelta(seconds=lease_duration)).isoformat()
            }))
            pipe.expire(f"{semaphore_key}:holders", lease_duration)
            
            results = await pipe.execute()
            
            if results[0] <= max_count:  # Successfully acquired
                # Create lock info
                lock_info = LockInfo(
                    lock_id=lock_id,
                    lock_name=request.lock_name,
                    lock_type=request.lock_type,
                    owner_id=request.requester_id,
                    acquired_at=datetime.now(timezone.utc),
                    expires_at=datetime.now(timezone.utc) + timedelta(seconds=lease_duration),
                    lease_duration=lease_duration,
                    metadata=request.metadata
                )
                
                self._active_locks[lock_id] = lock_info
                self._metrics['locks_acquired'] += 1
                
                self.logger.debug(f"Acquired semaphore lock: {request.lock_name}")
                return lock_id
            else:
                # Rollback - too many holders
                await self.redis_client.decr(f"{semaphore_key}:count")
                await self.redis_client.hdel(f"{semaphore_key}:holders", lock_id)
                return None
            
        except Exception as e:
            self.logger.error(f"Error acquiring semaphore lock: {e}")
            return None
    
    async def release_lock(self, lock_id: str, requester_id: str) -> bool:
        """Release distributed lock.
        
        Args:
            lock_id: ID of lock to release
            requester_id: ID of requester
            
        Returns:
            True if released successfully
        """
        try:
            if lock_id not in self._active_locks:
                self.logger.warning(f"Lock {lock_id} not found in active locks")
                return False
            
            lock_info = self._active_locks[lock_id]
            
            # Verify ownership
            if lock_info.owner_id != requester_id:
                self.logger.warning(f"Lock {lock_id} not owned by {requester_id}")
                return False
            
            # Release based on lock type
            success = False
            
            if lock_info.lock_type == LockType.EXCLUSIVE:
                success = await self._release_exclusive_lock(lock_info)
            
            elif lock_info.lock_type == LockType.SHARED:
                success = await self._release_shared_lock(lock_info)
            
            elif lock_info.lock_type == LockType.SEMAPHORE:
                success = await self._release_semaphore_lock(lock_info)
            
            else:
                success = await self._release_exclusive_lock(lock_info)
            
            if success:
                del self._active_locks[lock_id]
                self._metrics['locks_released'] += 1
                
                # Process queue for this lock
                await self._process_lock_queue(lock_info.lock_name)
                
                self.logger.debug(f"Released lock: {lock_info.lock_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error releasing lock {lock_id}: {e}")
            return False
    
    async def _release_exclusive_lock(self, lock_info: LockInfo) -> bool:
        """Release exclusive lock."""
        try:
            lock_key = f"{self._lock_prefix}{lock_info.lock_name}"
            
            # Use Lua script for atomic release with ownership check
            lua_script = """
            local lock_key = KEYS[1]
            local expected_owner = ARGV[1]
            local lock_data = redis.call('GET', lock_key)
            
            if lock_data then
                local lock_info = cjson.decode(lock_data)
                if lock_info.owner == expected_owner then
                    redis.call('DEL', lock_key)
                    return 1
                end
            end
            return 0
            """
            
            # RedisClient has eval method directly
            result = await self.redis_client.eval(
                lua_script,
                1,
                lock_key,
                lock_info.owner_id
            )
            
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Error releasing exclusive lock: {e}")
            return False
    
    async def extend_lock(self, 
                         lock_id: str, 
                         requester_id: str, 
                         additional_seconds: int) -> bool:
        """Extend lock lease.
        
        Args:
            lock_id: ID of lock to extend
            requester_id: ID of requester
            additional_seconds: Additional seconds to extend
            
        Returns:
            True if extended successfully
        """
        try:
            if lock_id not in self._active_locks:
                return False
            
            lock_info = self._active_locks[lock_id]
            
            # Verify ownership
            if lock_info.owner_id != requester_id:
                return False
            
            # Extend in Redis
            lock_key = f"{self._lock_prefix}{lock_info.lock_name}"
            
            if lock_info.lock_type == LockType.EXCLUSIVE:
                success = await self.redis_client.expire(lock_key, additional_seconds)
            else:
                # Handle other lock types
                success = await self.redis_client.expire(lock_key, additional_seconds)
            
            if success:
                # Update local info
                lock_info.expires_at = datetime.now(timezone.utc) + timedelta(seconds=additional_seconds)
                lock_info.lease_duration = additional_seconds
                
                self.logger.debug(f"Extended lock {lock_id} by {additional_seconds} seconds")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error extending lock {lock_id}: {e}")
            return False
    
    async def get_lock_info(self, lock_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a lock.
        
        Args:
            lock_name: Name of lock
            
        Returns:
            Lock information dictionary
        """
        try:
            lock_key = f"{self._lock_prefix}{lock_name}"
            lock_data = await self.redis_client.get(lock_key)
            
            if not lock_data:
                return None
            
            if isinstance(lock_data, bytes):
                lock_data = lock_data.decode()
            
            return json.loads(lock_data)
            
        except Exception as e:
            self.logger.error(f"Error getting lock info for {lock_name}: {e}")
            return None
    
    async def list_active_locks(self, owner_id: str = None) -> List[Dict[str, Any]]:
        """List active locks.
        
        Args:
            owner_id: Filter by owner ID (optional)
            
        Returns:
            List of active lock information
        """
        try:
            active_locks = []
            
            for lock_info in self._active_locks.values():
                if owner_id is None or lock_info.owner_id == owner_id:
                    active_locks.append(lock_info.to_dict())
            
            return active_locks
            
        except Exception as e:
            self.logger.error(f"Error listing active locks: {e}")
            return []
    
    async def _add_to_queue(self, request: LockRequest):
        """Add lock request to priority queue."""
        try:
            queue_key = f"{self._queue_prefix}{request.lock_name}"
            
            # Add to Redis sorted set with priority as score
            score = (request.priority.value * 1000000) + int(request.requested_at.timestamp())
            
            await self.redis_client.zadd(
                queue_key,
                {request.request_id: score}
            )
            
            # Store request details
            request_key = f"request:{request.request_id}"
            await self.redis_client.setex(
                request_key,
                request.timeout_seconds,
                json.dumps(request.to_dict())
            )
            
            # Add to local queue
            if request.lock_name not in self._lock_queues:
                self._lock_queues[request.lock_name] = []
            
            self._lock_queues[request.lock_name].append(request)
            self._lock_queues[request.lock_name].sort(key=lambda r: (r.priority.value, r.requested_at))
            
        except Exception as e:
            self.logger.error(f"Error adding to queue: {e}")
    
    async def _wait_for_lock(self, request: LockRequest, lease_duration: int = None) -> Optional[str]:
        """Wait for lock acquisition from queue."""
        try:
            start_time = time.time()
            timeout = request.timeout_seconds
            
            while time.time() - start_time < timeout:
                # Check if we can acquire the lock now
                lock_id = await self._try_immediate_acquisition(request, lease_duration)
                if lock_id:
                    # Remove from queue
                    await self._remove_from_queue(request)
                    
                    # Update metrics
                    wait_time = time.time() - start_time
                    self._metrics['total_wait_time'] += wait_time
                    self._metrics['average_wait_time'] = (
                        self._metrics['total_wait_time'] / max(self._metrics['locks_acquired'], 1)
                    )
                    
                    return lock_id
                
                # Wait a bit before retrying
                await asyncio.sleep(0.5)
            
            # Timeout
            await self._remove_from_queue(request)
            self._metrics['lock_timeouts'] += 1
            
            self.logger.warning(f"Lock acquisition timeout for {request.lock_name}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error waiting for lock: {e}")
            return None
    
    async def _remove_from_queue(self, request: LockRequest):
        """Remove request from queue."""
        try:
            queue_key = f"{self._queue_prefix}{request.lock_name}"
            await self.redis_client.zrem(queue_key, request.request_id)
            
            request_key = f"request:{request.request_id}"
            await self.redis_client.delete(request_key)
            
            # Remove from local queue
            if request.lock_name in self._lock_queues:
                self._lock_queues[request.lock_name] = [
                    r for r in self._lock_queues[request.lock_name] 
                    if r.request_id != request.request_id
                ]
                
        except Exception as e:
            self.logger.error(f"Error removing from queue: {e}")
    
    async def _queue_processor(self):
        """Process lock queues."""
        while not self._shutdown:
            try:
                # Process each lock queue
                for lock_name in list(self._lock_queues.keys()):
                    await self._process_lock_queue(lock_name)
                
                await asyncio.sleep(1)  # Process every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(5)
    
    async def _process_lock_queue(self, lock_name: str):
        """Process queue for specific lock."""
        try:
            if lock_name not in self._lock_queues:
                return
            
            queue = self._lock_queues[lock_name]
            if not queue:
                return
            
            # Try to satisfy requests in priority order
            satisfied_requests = []
            
            for request in queue[:]:  # Copy to avoid modification during iteration
                lock_id = await self._try_immediate_acquisition(request)
                if lock_id:
                    satisfied_requests.append(request)
                    await self._remove_from_queue(request)
                else:
                    break  # If we can't satisfy this request, stop processing
            
            if satisfied_requests:
                self.logger.debug(f"Satisfied {len(satisfied_requests)} lock requests for {lock_name}")
                
        except Exception as e:
            self.logger.error(f"Error processing lock queue for {lock_name}: {e}")
    
    async def _lock_expiration_monitor(self):
        """Monitor and clean up expired locks."""
        while not self._shutdown:
            try:
                current_time = datetime.now(timezone.utc)
                expired_locks = []
                
                # Check for expired locks
                for lock_id, lock_info in self._active_locks.items():
                    if current_time >= lock_info.expires_at:
                        expired_locks.append(lock_id)
                
                # Clean up expired locks
                for lock_id in expired_locks:
                    lock_info = self._active_locks[lock_id]
                    await self._cleanup_expired_lock(lock_info)
                    del self._active_locks[lock_id]
                    self._metrics['locks_expired'] += 1
                
                if expired_locks:
                    self.logger.info(f"Cleaned up {len(expired_locks)} expired locks")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in lock expiration monitor: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_expired_lock(self, lock_info: LockInfo):
        """Clean up expired lock from Redis."""
        try:
            lock_key = f"{self._lock_prefix}{lock_info.lock_name}"
            
            if lock_info.lock_type == LockType.EXCLUSIVE:
                await self.redis_client.delete(lock_key)
            
            elif lock_info.lock_type == LockType.SHARED:
                readers_key = f"{lock_key}:readers"
                await self.redis_client.hdel(readers_key, lock_info.lock_id)
            
            elif lock_info.lock_type == LockType.SEMAPHORE:
                semaphore_key = f"{self._semaphore_prefix}{lock_info.lock_name}"
                await self.redis_client.decr(f"{semaphore_key}:count")
                await self.redis_client.hdel(f"{semaphore_key}:holders", lock_info.lock_id)
            
            # Process queue for this lock
            await self._process_lock_queue(lock_info.lock_name)
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired lock: {e}")
    
    async def get_lock_statistics(self) -> Dict[str, Any]:
        """Get comprehensive lock statistics."""
        try:
            # Count active locks by type
            lock_types = {}
            for lock_info in self._active_locks.values():
                lock_type = lock_info.lock_type.value
                lock_types[lock_type] = lock_types.get(lock_type, 0) + 1
            
            # Count queued requests
            total_queued = sum(len(queue) for queue in self._lock_queues.values())
            
            return {
                'manager_id': self.manager_id,
                'active_locks': len(self._active_locks),
                'queued_requests': total_queued,
                'lock_types': lock_types,
                'performance_metrics': self._metrics.copy(),
                'capabilities': {
                    'deadlock_detection': self.enable_deadlock_detection,
                    'lock_monitoring': self.enable_lock_monitoring
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting lock statistics: {e}")
            return {}
    
    async def _register_lock_manager(self):
        """Register lock manager in distributed registry."""
        manager_data = {
            'manager_id': self.manager_id,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'status': 'active',
            'capabilities': {
                'deadlock_detection': self.enable_deadlock_detection,
                'lock_monitoring': self.enable_lock_monitoring,
                'supported_lock_types': [lt.value for lt in LockType]
            }
        }
        
        await self.state_manager.update(
            f"lock_managers.{self.manager_id}",
            manager_data,
            distributed=True
        )
    
    async def shutdown(self):
        """Shutdown lock manager."""
        self.logger.info(f"Shutting down Lock Manager: {self.manager_id}")
        self._shutdown = True
        
        # Stop deadlock detection
        if self.deadlock_detector:
            await self.deadlock_detector.stop_detection()
        
        # Stop management tasks
        for task in self._management_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Release all locks owned by this manager
        for lock_id, lock_info in list(self._active_locks.items()):
            await self.release_lock(lock_id, lock_info.owner_id)
        
        # Update manager status
        await self.state_manager.update(
            f"lock_managers.{self.manager_id}.status",
            'shutdown',
            distributed=True
        )
        
        self.logger.info(f"Lock Manager {self.manager_id} shutdown complete")


# Global lock manager instance
_global_lock_manager: Optional[RedisDistributedLockManager] = None


async def get_lock_manager(**kwargs) -> RedisDistributedLockManager:
    """Get global lock manager instance."""
    global _global_lock_manager
    
    if _global_lock_manager is None:
        _global_lock_manager = RedisDistributedLockManager(**kwargs)
        await _global_lock_manager.initialize()
    
    return _global_lock_manager


async def create_lock_manager(**kwargs) -> RedisDistributedLockManager:
    """Create new lock manager instance."""
    manager = RedisDistributedLockManager(**kwargs)
    await manager.initialize()
    return manager