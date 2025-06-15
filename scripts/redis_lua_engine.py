"""
Redis Lua Script Engine

Advanced Lua scripting system for Redis with atomic operations, complex business logic,
performance optimization, and comprehensive script management capabilities.
"""

import asyncio
import json
import logging
import hashlib
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from redis_integration import get_redis_client
from redis_lockfree_adapter import create_lockfree_state_manager


class ScriptType(Enum):
    """Types of Lua scripts."""
    ATOMIC_OPERATION = "atomic_operation"
    BUSINESS_LOGIC = "business_logic"
    DATA_PROCESSING = "data_processing"
    COORDINATION = "coordination"
    ANALYTICS = "analytics"
    MAINTENANCE = "maintenance"
    OPTIMIZATION = "optimization"


class ScriptPriority(Enum):
    """Script execution priorities."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class LuaScript:
    """Lua script definition."""
    script_id: str
    name: str
    description: str
    script_type: ScriptType
    script_code: str
    parameters: List[str]
    key_patterns: List[str]
    version: str
    created_at: datetime
    last_modified: datetime
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Calculate script hash after initialization."""
        self.script_hash = hashlib.sha1(self.script_code.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert script to dictionary."""
        return {
            'script_id': self.script_id,
            'name': self.name,
            'description': self.description,
            'script_type': self.script_type.value,
            'script_code': self.script_code,
            'script_hash': self.script_hash,
            'parameters': self.parameters,
            'key_patterns': self.key_patterns,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'last_modified': self.last_modified.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LuaScript':
        """Create script from dictionary."""
        script = cls(
            script_id=data['script_id'],
            name=data['name'],
            description=data['description'],
            script_type=ScriptType(data['script_type']),
            script_code=data['script_code'],
            parameters=data['parameters'],
            key_patterns=data['key_patterns'],
            version=data['version'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_modified=datetime.fromisoformat(data['last_modified']),
            metadata=data.get('metadata', {})
        )
        return script


@dataclass
class ScriptExecution:
    """Script execution context and result."""
    execution_id: str
    script_id: str
    script_hash: str
    keys: List[str]
    args: List[str]
    started_at: datetime
    completed_at: Optional[datetime]
    execution_time_ms: Optional[float]
    result: Any
    error: Optional[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert execution to dictionary."""
        return {
            'execution_id': self.execution_id,
            'script_id': self.script_id,
            'script_hash': self.script_hash,
            'keys': self.keys,
            'args': self.args,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'execution_time_ms': self.execution_time_ms,
            'result': self.result,
            'error': self.error,
            'metadata': self.metadata
        }


class RedisLuaEngine:
    """Advanced Redis Lua script engine with comprehensive script management."""
    
    def __init__(self,
                 engine_id: str = None,
                 enable_script_caching: bool = True,
                 enable_performance_monitoring: bool = True,
                 max_execution_time: int = 30):
        """Initialize Redis Lua engine.
        
        Args:
            engine_id: Unique engine identifier
            enable_script_caching: Enable script caching
            enable_performance_monitoring: Enable performance monitoring
            max_execution_time: Maximum script execution time in seconds
        """
        self.engine_id = engine_id or f"lua_engine_{uuid.uuid4().hex[:8]}"
        self.enable_script_caching = enable_script_caching
        self.enable_performance_monitoring = enable_performance_monitoring
        self.max_execution_time = max_execution_time
        
        self.logger = logging.getLogger(f"{__name__}.RedisLuaEngine")
        
        # Redis components
        self.redis_client = None
        self.state_manager = None
        
        # Script management
        self._registered_scripts: Dict[str, LuaScript] = {}
        self._script_cache: Dict[str, str] = {}  # script_hash -> script_sha
        self._execution_history: List[ScriptExecution] = []
        
        # Performance metrics
        self._metrics = {
            'scripts_registered': 0,
            'scripts_executed': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'execution_errors': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Built-in scripts
        self._builtin_scripts = {}
        
    async def initialize(self):
        """Initialize Lua engine components."""
        try:
            self.logger.info(f"Initializing Redis Lua Engine: {self.engine_id}")
            
            # Initialize Redis components
            self.redis_client = await get_redis_client()
            self.state_manager = create_lockfree_state_manager(f"lua_engine_{self.engine_id}")
            await self.state_manager.initialize()
            
            # Load built-in scripts
            await self._load_builtin_scripts()
            
            # Register engine
            await self._register_lua_engine()
            
            self.logger.info(f"Lua Engine {self.engine_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Lua Engine: {e}")
            raise
    
    async def _load_builtin_scripts(self):
        """Load built-in Lua scripts."""
        try:
            # Atomic Counter Script
            counter_script = LuaScript(
                script_id="atomic_counter",
                name="Atomic Counter",
                description="Atomic increment/decrement with bounds checking",
                script_type=ScriptType.ATOMIC_OPERATION,
                script_code="""
                local key = KEYS[1]
                local operation = ARGV[1]  -- 'incr' or 'decr'
                local amount = tonumber(ARGV[2]) or 1
                local min_val = tonumber(ARGV[3])
                local max_val = tonumber(ARGV[4])
                
                local current = redis.call('GET', key)
                if not current then
                    current = 0
                else
                    current = tonumber(current)
                end
                
                local new_val = current
                if operation == 'incr' then
                    new_val = current + amount
                elseif operation == 'decr' then
                    new_val = current - amount
                end
                
                -- Check bounds
                if min_val and new_val < min_val then
                    return {current, 'below_minimum'}
                end
                
                if max_val and new_val > max_val then
                    return {current, 'above_maximum'}
                end
                
                redis.call('SET', key, new_val)
                return {new_val, 'success'}
                """,
                parameters=['operation', 'amount', 'min_val', 'max_val'],
                key_patterns=['counter:*'],
                version="1.0",
                created_at=datetime.now(timezone.utc),
                last_modified=datetime.now(timezone.utc),
                metadata={'builtin': True}
            )
            
            # Distributed Lock Script
            lock_script = LuaScript(
                script_id="distributed_lock",
                name="Distributed Lock",
                description="Atomic lock acquisition with timeout",
                script_type=ScriptType.COORDINATION,
                script_code="""
                local lock_key = KEYS[1]
                local owner_id = ARGV[1]
                local timeout = tonumber(ARGV[2])
                local current_time = tonumber(ARGV[3])
                
                local lock_data = redis.call('GET', lock_key)
                
                if lock_data then
                    local lock_info = cjson.decode(lock_data)
                    
                    -- Check if lock expired
                    if current_time > lock_info.expires_at then
                        -- Lock expired, can acquire
                        local new_lock = {
                            owner = owner_id,
                            acquired_at = current_time,
                            expires_at = current_time + timeout
                        }
                        redis.call('SET', lock_key, cjson.encode(new_lock))
                        redis.call('EXPIRE', lock_key, timeout)
                        return {1, 'acquired', new_lock.expires_at}
                    else
                        -- Lock still active
                        if lock_info.owner == owner_id then
                            -- Owner trying to reacquire - extend lock
                            lock_info.expires_at = current_time + timeout
                            redis.call('SET', lock_key, cjson.encode(lock_info))
                            redis.call('EXPIRE', lock_key, timeout)
                            return {1, 'extended', lock_info.expires_at}
                        else
                            -- Lock held by someone else
                            return {0, 'locked', lock_info.expires_at}
                        end
                    end
                else
                    -- No lock exists, acquire it
                    local new_lock = {
                        owner = owner_id,
                        acquired_at = current_time,
                        expires_at = current_time + timeout
                    }
                    redis.call('SET', lock_key, cjson.encode(new_lock))
                    redis.call('EXPIRE', lock_key, timeout)
                    return {1, 'acquired', new_lock.expires_at}
                end
                """,
                parameters=['owner_id', 'timeout', 'current_time'],
                key_patterns=['lock:*'],
                version="1.0",
                created_at=datetime.now(timezone.utc),
                last_modified=datetime.now(timezone.utc),
                metadata={'builtin': True}
            )
            
            # Rate Limiter Script
            rate_limit_script = LuaScript(
                script_id="rate_limiter",
                name="Rate Limiter",
                description="Token bucket rate limiter with sliding window",
                script_type=ScriptType.BUSINESS_LOGIC,
                script_code="""
                local key = KEYS[1]
                local max_tokens = tonumber(ARGV[1])
                local refill_rate = tonumber(ARGV[2])  -- tokens per second
                local requested_tokens = tonumber(ARGV[3]) or 1
                local current_time = tonumber(ARGV[4])
                
                local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
                local tokens = tonumber(bucket[1]) or max_tokens
                local last_refill = tonumber(bucket[2]) or current_time
                
                -- Calculate tokens to add based on time elapsed
                local time_elapsed = current_time - last_refill
                local tokens_to_add = time_elapsed * refill_rate
                tokens = math.min(max_tokens, tokens + tokens_to_add)
                
                if tokens >= requested_tokens then
                    -- Allow request
                    tokens = tokens - requested_tokens
                    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
                    redis.call('EXPIRE', key, 3600)  -- Expire after 1 hour of inactivity
                    return {1, tokens, 'allowed'}
                else
                    -- Rate limited
                    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
                    redis.call('EXPIRE', key, 3600)
                    local retry_after = (requested_tokens - tokens) / refill_rate
                    return {0, tokens, retry_after}
                end
                """,
                parameters=['max_tokens', 'refill_rate', 'requested_tokens', 'current_time'],
                key_patterns=['rate_limit:*'],
                version="1.0",
                created_at=datetime.now(timezone.utc),
                last_modified=datetime.now(timezone.utc),
                metadata={'builtin': True}
            )
            
            # Task Queue Script
            task_queue_script = LuaScript(
                script_id="priority_task_queue",
                name="Priority Task Queue",
                description="Priority-based task queue with atomic dequeue",
                script_type=ScriptType.DATA_PROCESSING,
                script_code="""
                local queue_key = KEYS[1]
                local processing_key = KEYS[2]
                local worker_id = ARGV[1]
                local max_tasks = tonumber(ARGV[2]) or 1
                local current_time = tonumber(ARGV[3])
                
                local tasks = {}
                
                for i = 1, max_tasks do
                    -- Get highest priority task
                    local task_data = redis.call('ZPOPMAX', queue_key)
                    
                    if #task_data == 0 then
                        break
                    end
                    
                    local task_id = task_data[1]
                    local priority = task_data[2]
                    
                    -- Get task details
                    local task_info = redis.call('HGET', 'tasks:' .. task_id, 'data')
                    
                    if task_info then
                        -- Mark as processing
                        local processing_info = {
                            worker_id = worker_id,
                            started_at = current_time,
                            task_id = task_id,
                            priority = priority
                        }
                        
                        redis.call('HSET', processing_key, task_id, cjson.encode(processing_info))
                        redis.call('EXPIRE', processing_key, 3600)  -- Expire processing info after 1 hour
                        
                        table.insert(tasks, {
                            task_id = task_id,
                            priority = priority,
                            data = cjson.decode(task_info)
                        })
                    end
                end
                
                return cjson.encode(tasks)
                """,
                parameters=['worker_id', 'max_tasks', 'current_time'],
                key_patterns=['queue:*', 'processing:*'],
                version="1.0",
                created_at=datetime.now(timezone.utc),
                last_modified=datetime.now(timezone.utc),
                metadata={'builtin': True}
            )
            
            # Analytics Aggregation Script
            analytics_script = LuaScript(
                script_id="analytics_aggregation",
                name="Analytics Aggregation",
                description="Real-time metrics aggregation with time windows",
                script_type=ScriptType.ANALYTICS,
                script_code="""
                local metric_key = KEYS[1]
                local window_key = KEYS[2]
                local metric_value = tonumber(ARGV[1])
                local timestamp = tonumber(ARGV[2])
                local window_size = tonumber(ARGV[3]) or 300  -- 5 minutes default
                
                -- Add to time series
                redis.call('ZADD', metric_key, timestamp, timestamp .. ':' .. metric_value)
                
                -- Remove old entries outside window
                local cutoff = timestamp - window_size
                redis.call('ZREMRANGEBYSCORE', metric_key, '-inf', cutoff)
                
                -- Calculate aggregations
                local values = redis.call('ZRANGE', metric_key, 0, -1)
                local sum = 0
                local count = 0
                local min_val = nil
                local max_val = nil
                
                for _, entry in ipairs(values) do
                    local value = tonumber(string.match(entry, ':(.+)'))
                    sum = sum + value
                    count = count + 1
                    
                    if not min_val or value < min_val then
                        min_val = value
                    end
                    
                    if not max_val or value > max_val then
                        max_val = value
                    end
                end
                
                local avg = count > 0 and (sum / count) or 0
                
                -- Store aggregated results
                local agg_data = {
                    sum = sum,
                    count = count,
                    average = avg,
                    min = min_val or 0,
                    max = max_val or 0,
                    window_start = cutoff,
                    window_end = timestamp,
                    last_updated = timestamp
                }
                
                redis.call('HSET', window_key, 'data', cjson.encode(agg_data))
                redis.call('EXPIRE', window_key, window_size * 2)
                
                return cjson.encode(agg_data)
                """,
                parameters=['metric_value', 'timestamp', 'window_size'],
                key_patterns=['metrics:*', 'agg:*'],
                version="1.0",
                created_at=datetime.now(timezone.utc),
                last_modified=datetime.now(timezone.utc),
                metadata={'builtin': True}
            )
            
            # Register built-in scripts
            builtin_scripts = [
                counter_script,
                lock_script,
                rate_limit_script,
                task_queue_script,
                analytics_script
            ]
            
            for script in builtin_scripts:
                await self.register_script(script)
                self._builtin_scripts[script.script_id] = script
            
            self.logger.info(f"Loaded {len(builtin_scripts)} built-in Lua scripts")
            
        except Exception as e:
            self.logger.error(f"Error loading built-in scripts: {e}")
    
    async def register_script(self, script: LuaScript) -> bool:
        """Register Lua script with the engine.
        
        Args:
            script: Lua script to register
            
        Returns:
            True if registered successfully
        """
        try:
            # Validate script
            if not await self._validate_script(script):
                return False
            
            # Load script into Redis if caching enabled
            if self.enable_script_caching:
                script_sha = await self.redis_client.script_load(script.script_code)
                self._script_cache[script.script_hash] = script_sha
                self._metrics['cache_misses'] += 1
            
            # Store script
            self._registered_scripts[script.script_id] = script
            self._metrics['scripts_registered'] += 1
            
            # Store in distributed state
            await self.state_manager.update(
                f"lua_scripts.{script.script_id}",
                script.to_dict(),
                distributed=True
            )
            
            self.logger.info(f"Registered Lua script: {script.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering script {script.script_id}: {e}")
            return False
    
    async def _validate_script(self, script: LuaScript) -> bool:
        """Validate Lua script syntax and security.
        
        Args:
            script: Script to validate
            
        Returns:
            True if script is valid
        """
        try:
            # Basic syntax check by loading into Redis
            await self.redis_client.script_load(script.script_code)
            
            # Security checks
            forbidden_patterns = [
                'os.',          # OS operations
                'io.',          # File I/O
                'package.',     # Package loading
                'require',      # Module loading
                'loadfile',     # File loading
                'dofile',       # File execution
            ]
            
            for pattern in forbidden_patterns:
                if pattern in script.script_code:
                    self.logger.warning(f"Script {script.script_id} contains forbidden pattern: {pattern}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Script validation failed: {e}")
            return False
    
    async def execute_script(self,
                           script_id: str,
                           keys: List[str] = None,
                           args: List[Union[str, int, float]] = None,
                           timeout: int = None) -> Any:
        """Execute registered Lua script.
        
        Args:
            script_id: ID of script to execute
            keys: Redis keys for script
            args: Script arguments
            timeout: Execution timeout
            
        Returns:
            Script execution result
        """
        execution_id = str(uuid.uuid4())
        
        try:
            if script_id not in self._registered_scripts:
                raise ValueError(f"Script {script_id} not registered")
            
            script = self._registered_scripts[script_id]
            keys = keys or []
            args = [str(arg) for arg in (args or [])]
            
            # Create execution context
            execution = ScriptExecution(
                execution_id=execution_id,
                script_id=script_id,
                script_hash=script.script_hash,
                keys=keys,
                args=args,
                started_at=datetime.now(timezone.utc),
                completed_at=None,
                execution_time_ms=None,
                result=None,
                error=None,
                metadata={}
            )
            
            self.logger.debug(f"Executing script {script_id} (execution: {execution_id})")
            
            start_time = time.time()
            
            # Execute script
            if self.enable_script_caching and script.script_hash in self._script_cache:
                # Use cached script
                script_sha = self._script_cache[script.script_hash]
                result = await self.redis_client.evalsha(script_sha, len(keys), *keys, *args)
                self._metrics['cache_hits'] += 1
            else:
                # Execute script directly
                result = await self.redis_client.eval(script.script_code, len(keys), *keys, *args)
                
                # Cache if enabled
                if self.enable_script_caching:
                    script_sha = await self.redis_client.script_load(script.script_code)
                    self._script_cache[script.script_hash] = script_sha
                    self._metrics['cache_misses'] += 1
            
            # Complete execution
            execution_time = (time.time() - start_time) * 1000
            execution.completed_at = datetime.now(timezone.utc)
            execution.execution_time_ms = execution_time
            execution.result = result
            
            # Update metrics
            self._metrics['scripts_executed'] += 1
            self._metrics['total_execution_time'] += execution_time
            self._metrics['average_execution_time'] = (
                self._metrics['total_execution_time'] / self._metrics['scripts_executed']
            )
            
            # Store execution history
            if self.enable_performance_monitoring:
                self._execution_history.append(execution)
                
                # Keep only recent executions
                if len(self._execution_history) > 1000:
                    self._execution_history = self._execution_history[-1000:]
            
            self.logger.debug(f"Script {script_id} executed successfully in {execution_time:.2f}ms")
            return result
            
        except Exception as e:
            # Record error
            execution_time = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0
            execution.completed_at = datetime.now(timezone.utc)
            execution.execution_time_ms = execution_time
            execution.error = str(e)
            
            self._metrics['execution_errors'] += 1
            
            if self.enable_performance_monitoring:
                self._execution_history.append(execution)
            
            self.logger.error(f"Error executing script {script_id}: {e}")
            raise
    
    async def execute_atomic_counter(self,
                                   counter_key: str,
                                   operation: str = 'incr',
                                   amount: int = 1,
                                   min_val: int = None,
                                   max_val: int = None) -> Dict[str, Any]:
        """Execute atomic counter operation.
        
        Args:
            counter_key: Counter key
            operation: 'incr' or 'decr'
            amount: Amount to increment/decrement
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Counter operation result
        """
        result = await self.execute_script(
            'atomic_counter',
            keys=[counter_key],
            args=[operation, amount, min_val or '', max_val or '']
        )
        
        return {
            'value': result[0],
            'status': result[1],
            'operation': operation,
            'amount': amount
        }
    
    async def execute_distributed_lock(self,
                                     lock_key: str,
                                     owner_id: str,
                                     timeout: int = 30) -> Dict[str, Any]:
        """Execute distributed lock acquisition.
        
        Args:
            lock_key: Lock key
            owner_id: Owner identifier
            timeout: Lock timeout in seconds
            
        Returns:
            Lock operation result
        """
        current_time = int(time.time())
        
        result = await self.execute_script(
            'distributed_lock',
            keys=[lock_key],
            args=[owner_id, timeout, current_time]
        )
        
        return {
            'acquired': bool(result[0]),
            'status': result[1],
            'expires_at': result[2],
            'owner_id': owner_id
        }
    
    async def execute_rate_limiter(self,
                                 limit_key: str,
                                 max_tokens: int,
                                 refill_rate: float,
                                 requested_tokens: int = 1) -> Dict[str, Any]:
        """Execute rate limiter check.
        
        Args:
            limit_key: Rate limit key
            max_tokens: Maximum tokens in bucket
            refill_rate: Token refill rate per second
            requested_tokens: Tokens requested
            
        Returns:
            Rate limit result
        """
        current_time = time.time()
        
        result = await self.execute_script(
            'rate_limiter',
            keys=[limit_key],
            args=[max_tokens, refill_rate, requested_tokens, current_time]
        )
        
        return {
            'allowed': bool(result[0]),
            'remaining_tokens': result[1],
            'retry_after': result[2] if len(result) > 2 else None
        }
    
    async def execute_task_dequeue(self,
                                 queue_key: str,
                                 processing_key: str,
                                 worker_id: str,
                                 max_tasks: int = 1) -> List[Dict[str, Any]]:
        """Execute priority task dequeue.
        
        Args:
            queue_key: Task queue key
            processing_key: Processing tasks key
            worker_id: Worker identifier
            max_tasks: Maximum tasks to dequeue
            
        Returns:
            List of dequeued tasks
        """
        current_time = int(time.time())
        
        result = await self.execute_script(
            'priority_task_queue',
            keys=[queue_key, processing_key],
            args=[worker_id, max_tasks, current_time]
        )
        
        return json.loads(result) if result else []
    
    async def execute_analytics_aggregation(self,
                                          metric_key: str,
                                          window_key: str,
                                          metric_value: float,
                                          window_size: int = 300) -> Dict[str, Any]:
        """Execute analytics aggregation.
        
        Args:
            metric_key: Metric time series key
            window_key: Aggregation window key
            metric_value: Metric value to add
            window_size: Window size in seconds
            
        Returns:
            Aggregation result
        """
        timestamp = int(time.time())
        
        result = await self.execute_script(
            'analytics_aggregation',
            keys=[metric_key, window_key],
            args=[metric_value, timestamp, window_size]
        )
        
        return json.loads(result)
    
    async def get_script_info(self, script_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered script.
        
        Args:
            script_id: Script identifier
            
        Returns:
            Script information dictionary
        """
        if script_id not in self._registered_scripts:
            return None
        
        script = self._registered_scripts[script_id]
        
        # Get execution statistics
        executions = [e for e in self._execution_history if e.script_id == script_id]
        successful_executions = [e for e in executions if e.error is None]
        failed_executions = [e for e in executions if e.error is not None]
        
        avg_execution_time = 0
        if successful_executions:
            avg_execution_time = sum(e.execution_time_ms for e in successful_executions) / len(successful_executions)
        
        return {
            **script.to_dict(),
            'execution_stats': {
                'total_executions': len(executions),
                'successful_executions': len(successful_executions),
                'failed_executions': len(failed_executions),
                'average_execution_time_ms': avg_execution_time,
                'success_rate': len(successful_executions) / max(len(executions), 1)
            },
            'cached': script.script_hash in self._script_cache
        }
    
    async def list_scripts(self, script_type: ScriptType = None) -> List[Dict[str, Any]]:
        """List registered scripts.
        
        Args:
            script_type: Filter by script type (optional)
            
        Returns:
            List of script information
        """
        scripts = []
        
        for script in self._registered_scripts.values():
            if script_type is None or script.script_type == script_type:
                script_info = await self.get_script_info(script.script_id)
                if script_info:
                    scripts.append(script_info)
        
        return scripts
    
    async def get_execution_history(self, 
                                  script_id: str = None, 
                                  limit: int = 100) -> List[Dict[str, Any]]:
        """Get script execution history.
        
        Args:
            script_id: Filter by script ID (optional)
            limit: Maximum number of executions to return
            
        Returns:
            List of execution records
        """
        executions = self._execution_history
        
        if script_id:
            executions = [e for e in executions if e.script_id == script_id]
        
        # Sort by execution time (most recent first)
        executions.sort(key=lambda e: e.started_at, reverse=True)
        
        return [e.to_dict() for e in executions[:limit]]
    
    async def get_engine_statistics(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        try:
            # Calculate cache hit rate
            total_cache_operations = self._metrics['cache_hits'] + self._metrics['cache_misses']
            cache_hit_rate = (
                self._metrics['cache_hits'] / max(total_cache_operations, 1)
            )
            
            # Get error rate
            total_executions = self._metrics['scripts_executed']
            error_rate = self._metrics['execution_errors'] / max(total_executions, 1)
            
            return {
                'engine_id': self.engine_id,
                'registered_scripts': len(self._registered_scripts),
                'builtin_scripts': len(self._builtin_scripts),
                'cached_scripts': len(self._script_cache),
                'execution_history_size': len(self._execution_history),
                'performance_metrics': self._metrics.copy(),
                'cache_hit_rate': cache_hit_rate,
                'error_rate': error_rate,
                'capabilities': {
                    'script_caching': self.enable_script_caching,
                    'performance_monitoring': self.enable_performance_monitoring,
                    'max_execution_time': self.max_execution_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting engine statistics: {e}")
            return {}
    
    async def clear_script_cache(self) -> bool:
        """Clear script cache.
        
        Returns:
            True if cache cleared successfully
        """
        try:
            self._script_cache.clear()
            await self.redis_client.script_flush()
            
            self.logger.info("Script cache cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing script cache: {e}")
            return False
    
    async def _register_lua_engine(self):
        """Register Lua engine in distributed registry."""
        engine_data = {
            'engine_id': self.engine_id,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'status': 'active',
            'capabilities': {
                'script_caching': self.enable_script_caching,
                'performance_monitoring': self.enable_performance_monitoring,
                'max_execution_time': self.max_execution_time
            },
            'registered_scripts': len(self._registered_scripts),
            'builtin_scripts': len(self._builtin_scripts)
        }
        
        await self.state_manager.update(
            f"lua_engines.{self.engine_id}",
            engine_data,
            distributed=True
        )
    
    async def shutdown(self):
        """Shutdown Lua engine."""
        self.logger.info(f"Shutting down Lua Engine: {self.engine_id}")
        
        # Clear caches
        if self.enable_script_caching:
            await self.clear_script_cache()
        
        # Update engine status
        await self.state_manager.update(
            f"lua_engines.{self.engine_id}.status",
            'shutdown',
            distributed=True
        )
        
        self.logger.info(f"Lua Engine {self.engine_id} shutdown complete")


# Global Lua engine instance
_global_lua_engine: Optional[RedisLuaEngine] = None


async def get_lua_engine(**kwargs) -> RedisLuaEngine:
    """Get global Lua engine instance."""
    global _global_lua_engine
    
    if _global_lua_engine is None:
        _global_lua_engine = RedisLuaEngine(**kwargs)
        await _global_lua_engine.initialize()
    
    return _global_lua_engine


async def create_lua_engine(**kwargs) -> RedisLuaEngine:
    """Create new Lua engine instance."""
    engine = RedisLuaEngine(**kwargs)
    await engine.initialize()
    return engine