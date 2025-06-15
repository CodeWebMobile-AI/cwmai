"""
Redis Transaction Management with Optimistic Locking

Advanced transaction management system using Redis with optimistic locking,
multi-key transactions, conflict resolution, and comprehensive transaction coordination.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union, Callable, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

from redis_integration import get_redis_client
from redis_lockfree_adapter import create_lockfree_state_manager
from redis_lua_engine import get_lua_engine


class TransactionStatus(Enum):
    """Transaction status states."""
    PENDING = "pending"
    ACTIVE = "active"
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ABORTING = "aborting"
    ABORTED = "aborted"
    FAILED = "failed"
    TIMEOUT = "timeout"


class IsolationLevel(Enum):
    """Transaction isolation levels."""
    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED = "read_committed"
    REPEATABLE_READ = "repeatable_read"
    SERIALIZABLE = "serializable"


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    ABORT = "abort"
    RETRY = "retry"
    OVERWRITE = "overwrite"
    MERGE = "merge"
    CUSTOM = "custom"


@dataclass
class TransactionOperation:
    """Individual transaction operation."""
    operation_id: str
    operation_type: str  # 'get', 'set', 'delete', 'increment', 'custom'
    key: str
    value: Any = None
    expected_version: Optional[int] = None
    condition: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert operation to dictionary."""
        return {
            'operation_id': self.operation_id,
            'operation_type': self.operation_type,
            'key': self.key,
            'value': json.dumps(self.value) if self.value is not None else None,
            'expected_version': self.expected_version,
            'condition': self.condition,
            'metadata': json.dumps(self.metadata or {})
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransactionOperation':
        """Create operation from dictionary."""
        return cls(
            operation_id=data['operation_id'],
            operation_type=data['operation_type'],
            key=data['key'],
            value=json.loads(data['value']) if data.get('value') else None,
            expected_version=data.get('expected_version'),
            condition=data.get('condition'),
            metadata=json.loads(data.get('metadata', '{}'))
        )


@dataclass
class TransactionContext:
    """Transaction execution context."""
    transaction_id: str
    initiator_id: str
    isolation_level: IsolationLevel
    conflict_resolution: ConflictResolution
    status: TransactionStatus
    operations: List[TransactionOperation]
    read_set: Dict[str, Tuple[Any, int]]  # key -> (value, version)
    write_set: Dict[str, Tuple[Any, int]]  # key -> (value, version)
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    timeout_seconds: int
    retry_count: int
    max_retries: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary."""
        return {
            'transaction_id': self.transaction_id,
            'initiator_id': self.initiator_id,
            'isolation_level': self.isolation_level.value,
            'conflict_resolution': self.conflict_resolution.value,
            'status': self.status.value,
            'operations': [op.to_dict() for op in self.operations],
            'read_set': {k: [v[0], v[1]] for k, v in self.read_set.items()},
            'write_set': {k: [v[0], v[1]] for k, v in self.write_set.items()},
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'timeout_seconds': self.timeout_seconds,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'metadata': json.dumps(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransactionContext':
        """Create transaction from dictionary."""
        return cls(
            transaction_id=data['transaction_id'],
            initiator_id=data['initiator_id'],
            isolation_level=IsolationLevel(data['isolation_level']),
            conflict_resolution=ConflictResolution(data['conflict_resolution']),
            status=TransactionStatus(data['status']),
            operations=[TransactionOperation.from_dict(op) for op in data['operations']],
            read_set={k: tuple(v) for k, v in data['read_set'].items()},
            write_set={k: tuple(v) for k, v in data['write_set'].items()},
            created_at=datetime.fromisoformat(data['created_at']),
            started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else None,
            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            timeout_seconds=data['timeout_seconds'],
            retry_count=data['retry_count'],
            max_retries=data['max_retries'],
            metadata=json.loads(data.get('metadata', '{}'))
        )


class OptimisticLockManager:
    """Optimistic locking manager for versioned data."""
    
    def __init__(self, redis_client):
        """Initialize optimistic lock manager.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis_client = redis_client
        self.logger = logging.getLogger(f"{__name__}.OptimisticLockManager")
        
        # Key prefixes
        self._version_prefix = "version:"
        self._lock_prefix = "olock:"
    
    async def get_versioned(self, key: str) -> Tuple[Any, int]:
        """Get value with version.
        
        Args:
            key: Key to get
            
        Returns:
            Tuple of (value, version)
        """
        try:
            # Use pipeline for atomic get of value and version
            pipe = self.redis_client.pipeline()
            pipe.get(key)
            pipe.get(f"{self._version_prefix}{key}")
            results = await pipe.execute()
            
            value = results[0]
            version = results[1]
            
            if value is not None:
                if isinstance(value, bytes):
                    value = value.decode()
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass  # Keep as string if not JSON
            
            version_num = int(version) if version else 0
            
            return value, version_num
            
        except Exception as e:
            self.logger.error(f"Error getting versioned value for {key}: {e}")
            return None, 0
    
    async def set_versioned(self, 
                           key: str, 
                           value: Any, 
                           expected_version: int = None) -> Tuple[bool, int]:
        """Set value with optimistic locking.
        
        Args:
            key: Key to set
            value: Value to set
            expected_version: Expected current version
            
        Returns:
            Tuple of (success, new_version)
        """
        try:
            # Lua script for atomic versioned set
            lua_script = """
            local key = KEYS[1]
            local version_key = KEYS[2]
            local value = ARGV[1]
            local expected_version = tonumber(ARGV[2]) or -1
            
            local current_version = tonumber(redis.call('GET', version_key) or '0')
            
            if expected_version >= 0 and current_version ~= expected_version then
                return {0, current_version, 'version_mismatch'}
            end
            
            local new_version = current_version + 1
            
            redis.call('SET', key, value)
            redis.call('SET', version_key, new_version)
            
            return {1, new_version, 'success'}
            """
            
            value_str = json.dumps(value) if not isinstance(value, str) else value
            version_key = f"{self._version_prefix}{key}"
            
            result = await self.redis_client.eval(
                lua_script,
                2,
                key,
                version_key,
                value_str,
                expected_version or -1
            )
            
            success = bool(result[0])
            new_version = result[1]
            
            return success, new_version
            
        except Exception as e:
            self.logger.error(f"Error setting versioned value for {key}: {e}")
            return False, 0
    
    async def delete_versioned(self, key: str, expected_version: int = None) -> bool:
        """Delete value with optimistic locking.
        
        Args:
            key: Key to delete
            expected_version: Expected current version
            
        Returns:
            True if deleted successfully
        """
        try:
            lua_script = """
            local key = KEYS[1]
            local version_key = KEYS[2]
            local expected_version = tonumber(ARGV[1]) or -1
            
            local current_version = tonumber(redis.call('GET', version_key) or '0')
            
            if expected_version >= 0 and current_version ~= expected_version then
                return {0, current_version, 'version_mismatch'}
            end
            
            redis.call('DEL', key)
            redis.call('DEL', version_key)
            
            return {1, 0, 'deleted'}
            """
            
            version_key = f"{self._version_prefix}{key}"
            
            result = await self.redis_client.eval(
                lua_script,
                2,
                key,
                version_key,
                expected_version or -1
            )
            
            return bool(result[0])
            
        except Exception as e:
            self.logger.error(f"Error deleting versioned value for {key}: {e}")
            return False


class RedisTransactionManager:
    """Advanced Redis transaction manager with optimistic locking."""
    
    def __init__(self,
                 manager_id: str = None,
                 default_isolation: IsolationLevel = IsolationLevel.READ_COMMITTED,
                 default_conflict_resolution: ConflictResolution = ConflictResolution.RETRY,
                 default_timeout: int = 30,
                 enable_deadlock_detection: bool = True):
        """Initialize Redis transaction manager.
        
        Args:
            manager_id: Unique manager identifier
            default_isolation: Default isolation level
            default_conflict_resolution: Default conflict resolution strategy
            default_timeout: Default transaction timeout in seconds
            enable_deadlock_detection: Enable deadlock detection
        """
        self.manager_id = manager_id or f"txn_mgr_{uuid.uuid4().hex[:8]}"
        self.default_isolation = default_isolation
        self.default_conflict_resolution = default_conflict_resolution
        self.default_timeout = default_timeout
        self.enable_deadlock_detection = enable_deadlock_detection
        
        self.logger = logging.getLogger(f"{__name__}.RedisTransactionManager")
        
        # Redis components
        self.redis_client = None
        self.state_manager = None
        self.lua_engine = None
        self.lock_manager: Optional[OptimisticLockManager] = None
        
        # Transaction management
        self._active_transactions: Dict[str, TransactionContext] = {}
        self._transaction_locks: Dict[str, Set[str]] = {}  # txn_id -> locked_keys
        
        # Background tasks
        self._management_tasks: List[asyncio.Task] = []
        self._shutdown = False
        
        # Performance metrics
        self._metrics = {
            'transactions_started': 0,
            'transactions_committed': 0,
            'transactions_aborted': 0,
            'transactions_retried': 0,
            'conflicts_detected': 0,
            'deadlocks_detected': 0,
            'average_transaction_time': 0.0,
            'successful_rate': 0.0
        }
        
        # Redis key prefixes
        self._transaction_prefix = "txn:"
        self._lock_prefix = "txn_lock:"
        self._deadlock_prefix = "deadlock:"
    
    async def initialize(self):
        """Initialize transaction manager."""
        try:
            self.logger.info(f"Initializing Redis Transaction Manager: {self.manager_id}")
            
            # Initialize Redis components
            self.redis_client = await get_redis_client()
            self.state_manager = create_lockfree_state_manager(f"transaction_coordinator_{self.coordinator_id}")
            await self.state_manager.initialize()
            self.lua_engine = await get_lua_engine()
            
            # Initialize optimistic lock manager
            self.lock_manager = OptimisticLockManager(self.redis_client)
            
            # Load transaction Lua scripts
            await self._load_transaction_scripts()
            
            # Start management tasks
            await self._start_management_tasks()
            
            # Register transaction manager
            await self._register_transaction_manager()
            
            self.logger.info(f"Transaction Manager {self.manager_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Transaction Manager: {e}")
            raise
    
    async def _load_transaction_scripts(self):
        """Load Lua scripts for transaction operations."""
        try:
            # Multi-key transaction commit script
            transaction_commit_script = {
                'script_id': 'transaction_commit',
                'name': 'Multi-key Transaction Commit',
                'description': 'Atomic commit of multi-key transaction with version checking',
                'script_type': 'coordination',
                'script_code': """
                local transaction_id = ARGV[1]
                local operations_json = ARGV[2]
                local current_time = tonumber(ARGV[3])
                
                local operations = cjson.decode(operations_json)
                local success = true
                local conflicts = {}
                
                -- Phase 1: Validate all operations
                for i, op in ipairs(operations) do
                    local key = op.key
                    local expected_version = op.expected_version
                    local version_key = 'version:' .. key
                    
                    if expected_version then
                        local current_version = tonumber(redis.call('GET', version_key) or '0')
                        if current_version ~= expected_version then
                            success = false
                            table.insert(conflicts, {
                                key = key,
                                expected = expected_version,
                                actual = current_version
                            })
                        end
                    end
                end
                
                if not success then
                    return {0, 'conflicts', cjson.encode(conflicts)}
                end
                
                -- Phase 2: Execute all operations atomically
                local results = {}
                for i, op in ipairs(operations) do
                    local key = op.key
                    local value = op.value
                    local operation_type = op.operation_type
                    local version_key = 'version:' .. key
                    
                    if operation_type == 'set' then
                        local new_version = tonumber(redis.call('GET', version_key) or '0') + 1
                        redis.call('SET', key, value)
                        redis.call('SET', version_key, new_version)
                        table.insert(results, {key = key, new_version = new_version})
                        
                    elseif operation_type == 'delete' then
                        redis.call('DEL', key)
                        redis.call('DEL', version_key)
                        table.insert(results, {key = key, deleted = true})
                        
                    elseif operation_type == 'increment' then
                        local current_value = tonumber(redis.call('GET', key) or '0')
                        local increment_value = tonumber(value or '1')
                        local new_value = current_value + increment_value
                        local new_version = tonumber(redis.call('GET', version_key) or '0') + 1
                        
                        redis.call('SET', key, new_value)
                        redis.call('SET', version_key, new_version)
                        table.insert(results, {key = key, new_value = new_value, new_version = new_version})
                    end
                end
                
                -- Mark transaction as committed
                local txn_key = 'txn:' .. transaction_id
                redis.call('HSET', txn_key, 'status', 'committed')
                redis.call('HSET', txn_key, 'committed_at', current_time)
                redis.call('EXPIRE', txn_key, 3600) -- Keep for 1 hour
                
                return {1, 'committed', cjson.encode(results)}
                """,
                'parameters': ['transaction_id', 'operations_json', 'current_time'],
                'key_patterns': ['txn:*', 'version:*'],
                'version': '1.0',
                'created_at': datetime.now(timezone.utc),
                'last_modified': datetime.now(timezone.utc),
                'metadata': {'transaction_operation': 'commit'}
            }
            
            # Transaction rollback script
            transaction_rollback_script = {
                'script_id': 'transaction_rollback',
                'name': 'Transaction Rollback',
                'description': 'Rollback transaction and release locks',
                'script_type': 'coordination',
                'script_code': """
                local transaction_id = ARGV[1]
                local locked_keys_json = ARGV[2]
                local current_time = tonumber(ARGV[3])
                
                local locked_keys = cjson.decode(locked_keys_json)
                
                -- Release all locks
                for i, key in ipairs(locked_keys) do
                    local lock_key = 'txn_lock:' .. key
                    redis.call('DEL', lock_key)
                end
                
                -- Mark transaction as aborted
                local txn_key = 'txn:' .. transaction_id
                redis.call('HSET', txn_key, 'status', 'aborted')
                redis.call('HSET', txn_key, 'aborted_at', current_time)
                redis.call('EXPIRE', txn_key, 3600)
                
                return {1, 'aborted', #locked_keys}
                """,
                'parameters': ['transaction_id', 'locked_keys_json', 'current_time'],
                'key_patterns': ['txn:*', 'txn_lock:*'],
                'version': '1.0',
                'created_at': datetime.now(timezone.utc),
                'last_modified': datetime.now(timezone.utc),
                'metadata': {'transaction_operation': 'rollback'}
            }
            
            # Deadlock detection script
            deadlock_detection_script = {
                'script_id': 'deadlock_detection',
                'name': 'Deadlock Detection',
                'description': 'Detect deadlocks in transaction dependency graph',
                'script_type': 'coordination',
                'script_code': """
                local wait_graph_key = KEYS[1]
                local current_time = tonumber(ARGV[1])
                local max_age = tonumber(ARGV[2]) or 300
                
                -- Get all wait-for relationships
                local wait_edges = redis.call('HGETALL', wait_graph_key)
                local graph = {}
                local nodes = {}
                
                -- Build adjacency list
                for i = 1, #wait_edges, 2 do
                    local edge_key = wait_edges[i]
                    local edge_data = cjson.decode(wait_edges[i + 1])
                    
                    -- Skip old edges
                    if current_time - edge_data.timestamp < max_age then
                        local waiter = edge_data.waiter
                        local holder = edge_data.holder
                        
                        if not graph[waiter] then
                            graph[waiter] = {}
                        end
                        table.insert(graph[waiter], holder)
                        
                        nodes[waiter] = true
                        nodes[holder] = true
                    end
                end
                
                -- Simple cycle detection using DFS
                local visited = {}
                local rec_stack = {}
                local cycles = {}
                
                local function dfs(node, path)
                    if rec_stack[node] then
                        -- Found cycle
                        local cycle_start = nil
                        for i, n in ipairs(path) do
                            if n == node then
                                cycle_start = i
                                break
                            end
                        end
                        if cycle_start then
                            local cycle = {}
                            for i = cycle_start, #path do
                                table.insert(cycle, path[i])
                            end
                            table.insert(cycle, node)
                            table.insert(cycles, cycle)
                        end
                        return true
                    end
                    
                    if visited[node] then
                        return false
                    end
                    
                    visited[node] = true
                    rec_stack[node] = true
                    table.insert(path, node)
                    
                    if graph[node] then
                        for _, neighbor in ipairs(graph[node]) do
                            if dfs(neighbor, path) then
                                return true
                            end
                        end
                    end
                    
                    rec_stack[node] = false
                    table.remove(path)
                    return false
                end
                
                -- Check all nodes for cycles
                for node in pairs(nodes) do
                    if not visited[node] then
                        dfs(node, {})
                    end
                end
                
                return cjson.encode(cycles)
                """,
                'parameters': ['current_time', 'max_age'],
                'key_patterns': ['deadlock:wait_graph'],
                'version': '1.0',
                'created_at': datetime.now(timezone.utc),
                'last_modified': datetime.now(timezone.utc),
                'metadata': {'transaction_operation': 'deadlock_detection'}
            }
            
            # Register scripts with Lua engine
            for script_data in [transaction_commit_script, transaction_rollback_script, deadlock_detection_script]:
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
            
            self.logger.info("Transaction Lua scripts loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading transaction scripts: {e}")
            raise
    
    async def _start_management_tasks(self):
        """Start background management tasks."""
        try:
            # Transaction timeout monitor
            timeout_task = asyncio.create_task(self._transaction_timeout_monitor())
            self._management_tasks.append(timeout_task)
            
            # Deadlock detector
            if self.enable_deadlock_detection:
                deadlock_task = asyncio.create_task(self._deadlock_detector())
                self._management_tasks.append(deadlock_task)
            
            # Performance monitor
            perf_task = asyncio.create_task(self._performance_monitor())
            self._management_tasks.append(perf_task)
            
            # Cleanup task
            cleanup_task = asyncio.create_task(self._cleanup_completed_transactions())
            self._management_tasks.append(cleanup_task)
            
            self.logger.info(f"Started {len(self._management_tasks)} management tasks")
            
        except Exception as e:
            self.logger.error(f"Error starting management tasks: {e}")
    
    async def begin_transaction(self,
                               initiator_id: str,
                               isolation_level: IsolationLevel = None,
                               conflict_resolution: ConflictResolution = None,
                               timeout_seconds: int = None,
                               max_retries: int = 3,
                               metadata: Dict[str, Any] = None) -> str:
        """Begin a new transaction.
        
        Args:
            initiator_id: Transaction initiator identifier
            isolation_level: Isolation level for transaction
            conflict_resolution: Conflict resolution strategy
            timeout_seconds: Transaction timeout
            max_retries: Maximum retry attempts
            metadata: Additional metadata
            
        Returns:
            Transaction ID
        """
        try:
            transaction_id = str(uuid.uuid4())
            current_time = datetime.now(timezone.utc)
            
            # Create transaction context
            transaction = TransactionContext(
                transaction_id=transaction_id,
                initiator_id=initiator_id,
                isolation_level=isolation_level or self.default_isolation,
                conflict_resolution=conflict_resolution or self.default_conflict_resolution,
                status=TransactionStatus.PENDING,
                operations=[],
                read_set={},
                write_set={},
                created_at=current_time,
                started_at=None,
                completed_at=None,
                timeout_seconds=timeout_seconds or self.default_timeout,
                retry_count=0,
                max_retries=max_retries,
                metadata=metadata or {}
            )
            
            # Store transaction
            self._active_transactions[transaction_id] = transaction
            self._transaction_locks[transaction_id] = set()
            
            # Persist transaction to Redis
            txn_key = f"{self._transaction_prefix}{transaction_id}"
            await self.redis_client.hset(txn_key, mapping=transaction.to_dict())
            await self.redis_client.expire(txn_key, transaction.timeout_seconds + 60)
            
            self._metrics['transactions_started'] += 1
            
            self.logger.debug(f"Started transaction {transaction_id}")
            return transaction_id
            
        except Exception as e:
            self.logger.error(f"Error beginning transaction: {e}")
            raise
    
    async def add_operation(self,
                           transaction_id: str,
                           operation_type: str,
                           key: str,
                           value: Any = None,
                           condition: str = None) -> bool:
        """Add operation to transaction.
        
        Args:
            transaction_id: Transaction ID
            operation_type: Type of operation
            key: Key to operate on
            value: Value for operation
            condition: Optional condition
            
        Returns:
            True if operation added successfully
        """
        try:
            if transaction_id not in self._active_transactions:
                self.logger.error(f"Transaction {transaction_id} not found")
                return False
            
            transaction = self._active_transactions[transaction_id]
            
            if transaction.status not in [TransactionStatus.PENDING, TransactionStatus.ACTIVE]:
                self.logger.error(f"Cannot add operation to transaction in status {transaction.status.value}")
                return False
            
            # Create operation
            operation = TransactionOperation(
                operation_id=str(uuid.uuid4()),
                operation_type=operation_type,
                key=key,
                value=value,
                condition=condition
            )
            
            # Add to transaction
            transaction.operations.append(operation)
            
            # Update transaction status
            if transaction.status == TransactionStatus.PENDING:
                transaction.status = TransactionStatus.ACTIVE
                transaction.started_at = datetime.now(timezone.utc)
            
            # Update in Redis
            txn_key = f"{self._transaction_prefix}{transaction_id}"
            await self.redis_client.hset(txn_key, mapping=transaction.to_dict())
            
            self.logger.debug(f"Added {operation_type} operation for key {key} to transaction {transaction_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding operation to transaction {transaction_id}: {e}")
            return False
    
    async def read_key(self, transaction_id: str, key: str) -> Tuple[Any, bool]:
        """Read key within transaction context.
        
        Args:
            transaction_id: Transaction ID
            key: Key to read
            
        Returns:
            Tuple of (value, success)
        """
        try:
            if transaction_id not in self._active_transactions:
                return None, False
            
            transaction = self._active_transactions[transaction_id]
            
            # Check write set first (read your own writes)
            if key in transaction.write_set:
                value, version = transaction.write_set[key]
                return value, True
            
            # Check read set for repeatable read isolation
            if (transaction.isolation_level in [IsolationLevel.REPEATABLE_READ, IsolationLevel.SERIALIZABLE] 
                and key in transaction.read_set):
                value, version = transaction.read_set[key]
                return value, True
            
            # Read from storage with version
            value, version = await self.lock_manager.get_versioned(key)
            
            # Add to read set
            transaction.read_set[key] = (value, version)
            
            return value, True
            
        except Exception as e:
            self.logger.error(f"Error reading key {key} in transaction {transaction_id}: {e}")
            return None, False
    
    async def write_key(self, transaction_id: str, key: str, value: Any) -> bool:
        """Write key within transaction context.
        
        Args:
            transaction_id: Transaction ID
            key: Key to write
            value: Value to write
            
        Returns:
            True if write added to write set
        """
        try:
            if transaction_id not in self._active_transactions:
                return False
            
            transaction = self._active_transactions[transaction_id]
            
            # Get current version for optimistic locking
            current_value, current_version = await self.lock_manager.get_versioned(key)
            
            # Add to write set with expected version
            transaction.write_set[key] = (value, current_version)
            
            # Add write operation
            await self.add_operation(transaction_id, 'set', key, value)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing key {key} in transaction {transaction_id}: {e}")
            return False
    
    async def commit_transaction(self, transaction_id: str) -> Tuple[bool, str, Any]:
        """Commit transaction.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            Tuple of (success, status, result)
        """
        try:
            if transaction_id not in self._active_transactions:
                return False, "transaction_not_found", None
            
            transaction = self._active_transactions[transaction_id]
            
            if transaction.status not in [TransactionStatus.ACTIVE, TransactionStatus.PREPARING]:
                return False, f"invalid_status_{transaction.status.value}", None
            
            # Update status
            transaction.status = TransactionStatus.PREPARING
            
            # Validate read set for conflicts (optimistic locking)
            conflicts = await self._validate_read_set(transaction)
            if conflicts:
                return await self._handle_conflict(transaction, conflicts)
            
            # Prepare write operations with expected versions
            write_operations = []
            for key, (value, expected_version) in transaction.write_set.items():
                write_operations.append({
                    'operation_type': 'set',
                    'key': key,
                    'value': json.dumps(value) if not isinstance(value, str) else value,
                    'expected_version': expected_version
                })
            
            # Add other operations
            for op in transaction.operations:
                if op.operation_type != 'set' or op.key not in transaction.write_set:
                    op_dict = op.to_dict()
                    if op.operation_type in ['delete', 'increment']:
                        # Get current version for these operations
                        _, current_version = await self.lock_manager.get_versioned(op.key)
                        op_dict['expected_version'] = current_version
                    write_operations.append(op_dict)
            
            # Execute atomic commit using Lua script
            transaction.status = TransactionStatus.COMMITTING
            current_timestamp = time.time()
            
            result = await self.lua_engine.execute_script(
                'transaction_commit',
                keys=[],
                args=[
                    transaction_id,
                    json.dumps(write_operations),
                    current_timestamp
                ]
            )
            
            if result and result[0] == 1:
                # Successful commit
                transaction.status = TransactionStatus.COMMITTED
                transaction.completed_at = datetime.now(timezone.utc)
                
                # Clean up
                await self._cleanup_transaction(transaction_id)
                
                self._metrics['transactions_committed'] += 1
                self.logger.info(f"Transaction {transaction_id} committed successfully")
                
                return True, "committed", json.loads(result[2])
            else:
                # Commit failed due to conflicts
                conflicts = json.loads(result[2]) if len(result) > 2 else []
                return await self._handle_conflict(transaction, conflicts)
            
        except Exception as e:
            self.logger.error(f"Error committing transaction {transaction_id}: {e}")
            await self.abort_transaction(transaction_id, f"commit_error: {e}")
            return False, "error", str(e)
    
    async def abort_transaction(self, transaction_id: str, reason: str = None) -> bool:
        """Abort transaction.
        
        Args:
            transaction_id: Transaction ID
            reason: Abort reason
            
        Returns:
            True if aborted successfully
        """
        try:
            if transaction_id not in self._active_transactions:
                return False
            
            transaction = self._active_transactions[transaction_id]
            transaction.status = TransactionStatus.ABORTING
            
            # Get locked keys
            locked_keys = list(self._transaction_locks.get(transaction_id, set()))
            
            # Execute rollback script
            current_timestamp = time.time()
            
            result = await self.lua_engine.execute_script(
                'transaction_rollback',
                keys=[],
                args=[
                    transaction_id,
                    json.dumps(locked_keys),
                    current_timestamp
                ]
            )
            
            # Update transaction status
            transaction.status = TransactionStatus.ABORTED
            transaction.completed_at = datetime.now(timezone.utc)
            if reason:
                transaction.metadata['abort_reason'] = reason
            
            # Clean up
            await self._cleanup_transaction(transaction_id)
            
            self._metrics['transactions_aborted'] += 1
            self.logger.info(f"Transaction {transaction_id} aborted: {reason or 'unknown'}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error aborting transaction {transaction_id}: {e}")
            return False
    
    async def _validate_read_set(self, transaction: TransactionContext) -> List[Dict[str, Any]]:
        """Validate read set for conflicts.
        
        Args:
            transaction: Transaction context
            
        Returns:
            List of conflicts
        """
        conflicts = []
        
        try:
            for key, (read_value, read_version) in transaction.read_set.items():
                current_value, current_version = await self.lock_manager.get_versioned(key)
                
                if current_version != read_version:
                    conflicts.append({
                        'key': key,
                        'read_version': read_version,
                        'current_version': current_version,
                        'type': 'version_mismatch'
                    })
            
        except Exception as e:
            self.logger.error(f"Error validating read set: {e}")
        
        return conflicts
    
    async def _handle_conflict(self, 
                              transaction: TransactionContext, 
                              conflicts: List[Dict[str, Any]]) -> Tuple[bool, str, Any]:
        """Handle transaction conflicts.
        
        Args:
            transaction: Transaction context
            conflicts: List of conflicts
            
        Returns:
            Tuple of (success, status, result)
        """
        try:
            self._metrics['conflicts_detected'] += 1
            
            if transaction.conflict_resolution == ConflictResolution.ABORT:
                await self.abort_transaction(transaction.transaction_id, "conflicts_detected")
                return False, "aborted_due_to_conflicts", conflicts
            
            elif transaction.conflict_resolution == ConflictResolution.RETRY:
                if transaction.retry_count < transaction.max_retries:
                    # Retry transaction
                    transaction.retry_count += 1
                    transaction.status = TransactionStatus.PENDING
                    transaction.read_set.clear()
                    
                    # Reset operations but keep write set
                    transaction.operations = [
                        op for op in transaction.operations 
                        if op.key in transaction.write_set
                    ]
                    
                    self._metrics['transactions_retried'] += 1
                    self.logger.info(f"Retrying transaction {transaction.transaction_id} (attempt {transaction.retry_count})")
                    
                    # Exponential backoff
                    delay = min(2 ** transaction.retry_count, 10)
                    await asyncio.sleep(delay)
                    
                    return False, "retrying", conflicts
                else:
                    await self.abort_transaction(transaction.transaction_id, "max_retries_exceeded")
                    return False, "aborted_max_retries", conflicts
            
            else:  # Other strategies not implemented yet
                await self.abort_transaction(transaction.transaction_id, "unsupported_conflict_resolution")
                return False, "aborted_unsupported_resolution", conflicts
        
        except Exception as e:
            self.logger.error(f"Error handling conflict: {e}")
            await self.abort_transaction(transaction.transaction_id, f"conflict_handling_error: {e}")
            return False, "error", str(e)
    
    async def get_transaction_status(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get transaction status.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            Transaction status dictionary
        """
        try:
            if transaction_id in self._active_transactions:
                transaction = self._active_transactions[transaction_id]
                return {
                    'transaction_id': transaction_id,
                    'status': transaction.status.value,
                    'operations_count': len(transaction.operations),
                    'read_set_size': len(transaction.read_set),
                    'write_set_size': len(transaction.write_set),
                    'retry_count': transaction.retry_count,
                    'created_at': transaction.created_at.isoformat(),
                    'started_at': transaction.started_at.isoformat() if transaction.started_at else None,
                    'completed_at': transaction.completed_at.isoformat() if transaction.completed_at else None
                }
            
            # Check Redis for completed transactions
            txn_key = f"{self._transaction_prefix}{transaction_id}"
            txn_data = await self.redis_client.hgetall(txn_key)
            
            if txn_data:
                return {k.decode() if isinstance(k, bytes) else k: 
                       v.decode() if isinstance(v, bytes) else v 
                       for k, v in txn_data.items()}
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting transaction status {transaction_id}: {e}")
            return None
    
    async def get_manager_statistics(self) -> Dict[str, Any]:
        """Get comprehensive transaction manager statistics."""
        try:
            # Calculate success rate
            total_completed = self._metrics['transactions_committed'] + self._metrics['transactions_aborted']
            success_rate = (
                self._metrics['transactions_committed'] / max(total_completed, 1)
            )
            
            return {
                'manager_id': self.manager_id,
                'active_transactions': len(self._active_transactions),
                'performance_metrics': {
                    **self._metrics,
                    'successful_rate': success_rate
                },
                'capabilities': {
                    'deadlock_detection': self.enable_deadlock_detection,
                    'default_isolation': self.default_isolation.value,
                    'default_conflict_resolution': self.default_conflict_resolution.value
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting manager statistics: {e}")
            return {}
    
    async def _cleanup_transaction(self, transaction_id: str):
        """Clean up completed transaction."""
        try:
            # Remove from active transactions
            if transaction_id in self._active_transactions:
                del self._active_transactions[transaction_id]
            
            # Clean up locks
            if transaction_id in self._transaction_locks:
                del self._transaction_locks[transaction_id]
            
        except Exception as e:
            self.logger.error(f"Error cleaning up transaction {transaction_id}: {e}")
    
    async def _register_transaction_manager(self):
        """Register transaction manager in distributed registry."""
        manager_data = {
            'manager_id': self.manager_id,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'status': 'active',
            'capabilities': {
                'deadlock_detection': self.enable_deadlock_detection,
                'isolation_levels': [level.value for level in IsolationLevel],
                'conflict_resolutions': [res.value for res in ConflictResolution]
            }
        }
        
        await self.state_manager.update(
            f"transaction_managers.{self.manager_id}",
            manager_data,
            distributed=True
        )
    
    async def shutdown(self):
        """Shutdown transaction manager."""
        self.logger.info(f"Shutting down Transaction Manager: {self.manager_id}")
        self._shutdown = True
        
        # Abort all active transactions
        for transaction_id in list(self._active_transactions.keys()):
            await self.abort_transaction(transaction_id, "manager_shutdown")
        
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
            f"transaction_managers.{self.manager_id}.status",
            'shutdown',
            distributed=True
        )
        
        self.logger.info(f"Transaction Manager {self.manager_id} shutdown complete")


# Global transaction manager instance
_global_transaction_manager: Optional[RedisTransactionManager] = None


async def get_transaction_manager(**kwargs) -> RedisTransactionManager:
    """Get global transaction manager instance."""
    global _global_transaction_manager
    
    if _global_transaction_manager is None:
        _global_transaction_manager = RedisTransactionManager(**kwargs)
        await _global_transaction_manager.initialize()
    
    return _global_transaction_manager


async def create_transaction_manager(**kwargs) -> RedisTransactionManager:
    """Create new transaction manager instance."""
    manager = RedisTransactionManager(**kwargs)
    await manager.initialize()
    return manager


class TransactionContext:
    """Context manager for Redis transactions."""
    
    def __init__(self, manager: RedisTransactionManager, **kwargs):
        """Initialize transaction context.
        
        Args:
            manager: Transaction manager instance
            **kwargs: Transaction parameters
        """
        self.manager = manager
        self.transaction_kwargs = kwargs
        self.transaction_id = None
    
    async def __aenter__(self):
        """Enter transaction context."""
        initiator_id = self.transaction_kwargs.get('initiator_id', 'context_manager')
        self.transaction_id = await self.manager.begin_transaction(initiator_id, **self.transaction_kwargs)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit transaction context."""
        if self.transaction_id:
            if exc_type is None:
                # No exception, commit transaction
                success, status, result = await self.manager.commit_transaction(self.transaction_id)
                if not success:
                    raise RuntimeError(f"Transaction commit failed: {status}")
            else:
                # Exception occurred, abort transaction
                await self.manager.abort_transaction(self.transaction_id, f"exception: {exc_type.__name__}")
    
    async def read(self, key: str):
        """Read key in transaction."""
        value, success = await self.manager.read_key(self.transaction_id, key)
        if not success:
            raise RuntimeError(f"Failed to read key {key}")
        return value
    
    async def write(self, key: str, value: Any):
        """Write key in transaction."""
        success = await self.manager.write_key(self.transaction_id, key, value)
        if not success:
            raise RuntimeError(f"Failed to write key {key}")
    
    async def add_operation(self, operation_type: str, key: str, value: Any = None, condition: str = None):
        """Add operation to transaction."""
        success = await self.manager.add_operation(
            self.transaction_id, operation_type, key, value, condition
        )
        if not success:
            raise RuntimeError(f"Failed to add operation {operation_type} for key {key}")


async def transaction(**kwargs):
    """Create transaction context manager.
    
    Args:
        **kwargs: Transaction parameters
        
    Returns:
        Transaction context manager
    """
    manager = await get_transaction_manager()
    return TransactionContext(manager, **kwargs)