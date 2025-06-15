"""
Redis Distributed Locks Manager

High-performance distributed locking system with Redis backend featuring
deadlock prevention, lock renewal, and hierarchical locking patterns.
"""

import asyncio
import time
import uuid
import threading
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging
from contextlib import asynccontextmanager
from .redis_client import RedisClient


class LockType(Enum):
    """Types of distributed locks."""
    EXCLUSIVE = "exclusive"
    SHARED = "shared"
    REENTRANT = "reentrant"
    TIMED = "timed"


class LockStatus(Enum):
    """Lock status states."""
    UNLOCKED = "unlocked"
    LOCKED = "locked"
    EXPIRED = "expired"
    RELEASED = "released"
    FAILED = "failed"


@dataclass
class LockInfo:
    """Lock information and metadata."""
    lock_id: str
    key: str
    owner: str
    lock_type: LockType
    acquired_at: datetime
    expires_at: datetime
    ttl_seconds: int
    reentrant_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if lock is expired."""
        return datetime.now(timezone.utc) >= self.expires_at
    
    def time_remaining(self) -> float:
        """Get remaining lock time in seconds."""
        remaining = (self.expires_at - datetime.now(timezone.utc)).total_seconds()
        return max(0, remaining)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            'lock_id': self.lock_id,
            'key': self.key,
            'owner': self.owner,
            'lock_type': self.lock_type.value,
            'acquired_at': self.acquired_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'ttl_seconds': self.ttl_seconds,
            'reentrant_count': self.reentrant_count,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LockInfo':
        """Create from dictionary."""
        return cls(
            lock_id=data['lock_id'],
            key=data['key'],
            owner=data['owner'],
            lock_type=LockType(data['lock_type']),
            acquired_at=datetime.fromisoformat(data['acquired_at']),
            expires_at=datetime.fromisoformat(data['expires_at']),
            ttl_seconds=data['ttl_seconds'],
            reentrant_count=data.get('reentrant_count', 0),
            metadata=data.get('metadata', {})
        )


@dataclass
class LockStatistics:
    """Lock usage statistics."""
    locks_acquired: int = 0
    locks_released: int = 0
    locks_expired: int = 0
    locks_failed: int = 0
    total_wait_time: float = 0.0
    total_hold_time: float = 0.0
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def success_rate(self) -> float:
        """Calculate lock acquisition success rate."""
        total_attempts = self.locks_acquired + self.locks_failed
        return self.locks_acquired / max(total_attempts, 1)
    
    @property
    def average_wait_time(self) -> float:
        """Calculate average wait time."""
        return self.total_wait_time / max(self.locks_acquired, 1)
    
    @property
    def average_hold_time(self) -> float:
        """Calculate average hold time."""
        return self.total_hold_time / max(self.locks_released, 1)


class RedisLock:
    """Individual Redis distributed lock with context manager support."""
    
    def __init__(self, manager: 'RedisLocksManager', key: str, 
                 lock_type: LockType = LockType.EXCLUSIVE, ttl: int = 30):
        """Initialize Redis lock.
        
        Args:
            manager: Locks manager instance
            key: Lock key
            lock_type: Type of lock
            ttl: Time to live in seconds
        """
        self.manager = manager
        self.key = key
        self.lock_type = lock_type
        self.ttl = ttl
        self.lock_info: Optional[LockInfo] = None
        self._renewal_task: Optional[asyncio.Task] = None
        self._acquired = False
    
    async def acquire(self, timeout: Optional[float] = None, 
                     auto_renewal: bool = True) -> bool:
        """Acquire the lock.
        
        Args:
            timeout: Maximum time to wait for lock
            auto_renewal: Enable automatic lock renewal
            
        Returns:
            True if lock acquired successfully
        """
        success = await self.manager.acquire_lock(
            self.key, self.lock_type, self.ttl, timeout
        )
        
        if success:
            self._acquired = True
            self.lock_info = self.manager.get_lock_info(self.key)
            
            if auto_renewal and self.lock_info:
                self._renewal_task = asyncio.create_task(
                    self._auto_renewal_loop()
                )
        
        return success
    
    async def release(self) -> bool:
        """Release the lock."""
        if not self._acquired:
            return False
        
        # Stop auto-renewal
        if self._renewal_task:
            self._renewal_task.cancel()
            try:
                await self._renewal_task
            except asyncio.CancelledError:
                pass
            self._renewal_task = None
        
        success = await self.manager.release_lock(self.key)
        if success:
            self._acquired = False
            self.lock_info = None
        
        return success
    
    async def renew(self, new_ttl: Optional[int] = None) -> bool:
        """Renew the lock."""
        if not self._acquired or not self.lock_info:
            return False
        
        return await self.manager.renew_lock(self.key, new_ttl or self.ttl)
    
    async def _auto_renewal_loop(self):
        """Auto-renewal background task."""
        while self._acquired and self.lock_info:
            try:
                # Renew when 2/3 of TTL has passed
                renewal_time = self.ttl * 0.66
                await asyncio.sleep(renewal_time)
                
                if self._acquired:
                    await self.renew()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.manager.logger.error(f"Lock auto-renewal error: {e}")
                break
    
    async def __aenter__(self):
        """Async context manager entry."""
        success = await self.acquire()
        if not success:
            raise RuntimeError(f"Failed to acquire lock: {self.key}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.release()
    
    def __repr__(self) -> str:
        status = "ACQUIRED" if self._acquired else "RELEASED"
        return f"RedisLock(key={self.key}, type={self.lock_type.value}, status={status})"


class RedisLocksManager:
    """Advanced Redis distributed locks manager."""
    
    def __init__(self, redis_client: RedisClient, namespace: str = "locks",
                 instance_id: str = None):
        """Initialize Redis locks manager.
        
        Args:
            redis_client: Redis client instance
            namespace: Locks namespace for isolation
            instance_id: Unique instance identifier
        """
        self.redis = redis_client
        self.namespace = namespace
        self.instance_id = instance_id or str(uuid.uuid4())
        self.logger = logging.getLogger(__name__)
        
        # Lock tracking
        self.owned_locks: Dict[str, LockInfo] = {}
        self.lock_waiters: Dict[str, List[asyncio.Event]] = {}
        self._local_lock = threading.RLock()
        
        # Configuration
        self.default_ttl = 30  # seconds
        self.max_wait_time = 300  # 5 minutes
        self.cleanup_interval = 60  # 1 minute
        self.deadlock_detection = True
        
        # Statistics
        self.stats = LockStatistics()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Lua scripts for atomic operations
        self._acquire_script = None
        self._release_script = None
        self._renew_script = None
    
    def _lock_key(self, key: str) -> str:
        """Create namespaced lock key."""
        return f"{self.namespace}:{key}"
    
    def _owner_key(self, key: str) -> str:
        """Create lock owner key."""
        return f"{self.namespace}:owner:{key}"
    
    def _waiters_key(self, key: str) -> str:
        """Create lock waiters key."""
        return f"{self.namespace}:waiters:{key}"
    
    def _deadlock_key(self) -> str:
        """Create deadlock detection key."""
        return f"{self.namespace}:deadlock:{self.instance_id}"
    
    async def start(self):
        """Start locks manager background tasks."""
        await self._load_lua_scripts()
        
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.logger.info(f"Redis Locks manager started (instance: {self.instance_id})")
    
    async def stop(self):
        """Stop locks manager and release all locks."""
        self._shutdown = True
        
        # Stop cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        
        # Release all owned locks
        with self._local_lock:
            for key in list(self.owned_locks.keys()):
                await self.release_lock(key)
        
        self.logger.info(f"Redis Locks manager stopped (instance: {self.instance_id})")
    
    async def _load_lua_scripts(self):
        """Load Lua scripts for atomic operations."""
        # Acquire lock script
        acquire_script = """
        local key = KEYS[1]
        local owner_key = KEYS[2]
        local lock_data = ARGV[1]
        local ttl = tonumber(ARGV[2])
        local owner = ARGV[3]
        local lock_type = ARGV[4]
        
        -- Check if lock exists
        local current_owner = redis.call('GET', owner_key)
        
        if current_owner == false then
            -- Lock doesn't exist, acquire it
            redis.call('SET', key, lock_data, 'EX', ttl)
            redis.call('SET', owner_key, owner, 'EX', ttl)
            return 1
        elseif current_owner == owner and lock_type == 'reentrant' then
            -- Reentrant lock by same owner
            redis.call('SET', key, lock_data, 'EX', ttl)
            redis.call('SET', owner_key, owner, 'EX', ttl)
            return 1
        else
            -- Lock is held by someone else
            return 0
        end
        """
        
        # Release lock script
        release_script = """
        local key = KEYS[1]
        local owner_key = KEYS[2]
        local owner = ARGV[1]
        
        local current_owner = redis.call('GET', owner_key)
        
        if current_owner == owner then
            redis.call('DEL', key, owner_key)
            return 1
        else
            return 0
        end
        """
        
        # Renew lock script
        renew_script = """
        local key = KEYS[1]
        local owner_key = KEYS[2]
        local owner = ARGV[1]
        local ttl = tonumber(ARGV[2])
        local lock_data = ARGV[3]
        
        local current_owner = redis.call('GET', owner_key)
        
        if current_owner == owner then
            redis.call('EXPIRE', key, ttl)
            redis.call('EXPIRE', owner_key, ttl)
            redis.call('SET', key, lock_data)
            return 1
        else
            return 0
        end
        """
        
        # Load scripts into Redis
        self._acquire_script = await self.redis.script_load(acquire_script)
        self._release_script = await self.redis.script_load(release_script)
        self._renew_script = await self.redis.script_load(renew_script)
    
    async def acquire_lock(self, key: str, lock_type: LockType = LockType.EXCLUSIVE,
                          ttl: int = None, timeout: Optional[float] = None) -> bool:
        """Acquire distributed lock.
        
        Args:
            key: Lock key
            lock_type: Type of lock to acquire
            ttl: Time to live in seconds
            timeout: Maximum wait time
            
        Returns:
            True if lock acquired successfully
        """
        ttl = ttl or self.default_ttl
        timeout = timeout or self.max_wait_time
        start_time = time.time()
        
        lock_key = self._lock_key(key)
        owner_key = self._owner_key(key)
        owner_id = f"{self.instance_id}:{threading.get_ident()}"
        
        # Check for deadlock
        if self.deadlock_detection and await self._check_deadlock(key):
            self.logger.warning(f"Potential deadlock detected for lock {key}")
            self.stats.locks_failed += 1
            return False
        
        while True:
            try:
                # Create lock info
                lock_info = LockInfo(
                    lock_id=str(uuid.uuid4()),
                    key=key,
                    owner=owner_id,
                    lock_type=lock_type,
                    acquired_at=datetime.now(timezone.utc),
                    expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl),
                    ttl_seconds=ttl
                )
                
                # Try to acquire lock using Lua script
                result = await self.redis.evalsha(
                    self._acquire_script,
                    2,
                    lock_key, owner_key,
                    json.dumps(lock_info.to_dict()),
                    ttl,
                    owner_id,
                    lock_type.value
                )
                
                if result == 1:
                    # Lock acquired successfully
                    with self._local_lock:
                        self.owned_locks[key] = lock_info
                    
                    wait_time = time.time() - start_time
                    self.stats.locks_acquired += 1
                    self.stats.total_wait_time += wait_time
                    
                    self.logger.debug(f"Acquired lock {key} after {wait_time:.3f}s")
                    return True
                
                # Lock not acquired, check timeout
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    self.stats.locks_failed += 1
                    self.logger.debug(f"Lock acquisition timeout for {key} after {elapsed:.3f}s")
                    return False
                
                # Wait before retry
                await asyncio.sleep(min(0.1, (timeout - elapsed) / 10))
                
            except Exception as e:
                self.logger.error(f"Error acquiring lock {key}: {e}")
                self.stats.locks_failed += 1
                return False
    
    async def release_lock(self, key: str) -> bool:
        """Release distributed lock.
        
        Args:
            key: Lock key to release
            
        Returns:
            True if lock released successfully
        """
        lock_key = self._lock_key(key)
        owner_key = self._owner_key(key)
        owner_id = f"{self.instance_id}:{threading.get_ident()}"
        
        try:
            # Release using Lua script
            result = await self.redis.evalsha(
                self._release_script,
                2,
                lock_key, owner_key,
                owner_id
            )
            
            if result == 1:
                # Update local tracking
                with self._local_lock:
                    if key in self.owned_locks:
                        lock_info = self.owned_locks.pop(key)
                        hold_time = (datetime.now(timezone.utc) - lock_info.acquired_at).total_seconds()
                        self.stats.total_hold_time += hold_time
                
                # Notify waiters
                await self._notify_waiters(key)
                
                self.stats.locks_released += 1
                self.logger.debug(f"Released lock {key}")
                return True
            else:
                self.logger.warning(f"Failed to release lock {key} - not owned by this instance")
                return False
                
        except Exception as e:
            self.logger.error(f"Error releasing lock {key}: {e}")
            return False
    
    async def renew_lock(self, key: str, new_ttl: int) -> bool:
        """Renew distributed lock.
        
        Args:
            key: Lock key to renew
            new_ttl: New time to live in seconds
            
        Returns:
            True if lock renewed successfully
        """
        lock_key = self._lock_key(key)
        owner_key = self._owner_key(key)
        owner_id = f"{self.instance_id}:{threading.get_ident()}"
        
        try:
            with self._local_lock:
                if key not in self.owned_locks:
                    return False
                
                lock_info = self.owned_locks[key]
                lock_info.expires_at = datetime.now(timezone.utc) + timedelta(seconds=new_ttl)
                lock_info.ttl_seconds = new_ttl
            
            # Renew using Lua script
            result = await self.redis.evalsha(
                self._renew_script,
                2,
                lock_key, owner_key,
                owner_id,
                new_ttl,
                json.dumps(lock_info.to_dict())
            )
            
            if result == 1:
                self.logger.debug(f"Renewed lock {key} for {new_ttl}s")
                return True
            else:
                self.logger.warning(f"Failed to renew lock {key}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error renewing lock {key}: {e}")
            return False
    
    async def is_locked(self, key: str) -> bool:
        """Check if key is locked."""
        try:
            lock_key = self._lock_key(key)
            return bool(await self.redis.exists(lock_key))
        except Exception:
            return False
    
    def get_lock_info(self, key: str) -> Optional[LockInfo]:
        """Get lock information."""
        with self._local_lock:
            return self.owned_locks.get(key)
    
    async def get_remote_lock_info(self, key: str) -> Optional[LockInfo]:
        """Get remote lock information from Redis."""
        try:
            lock_key = self._lock_key(key)
            data = await self.redis.get(lock_key)
            
            if data:
                lock_data = json.loads(data)
                return LockInfo.from_dict(lock_data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting remote lock info for {key}: {e}")
            return None
    
    def create_lock(self, key: str, lock_type: LockType = LockType.EXCLUSIVE,
                   ttl: int = None) -> RedisLock:
        """Create a Redis lock instance.
        
        Args:
            key: Lock key
            lock_type: Type of lock
            ttl: Time to live in seconds
            
        Returns:
            RedisLock instance
        """
        return RedisLock(self, key, lock_type, ttl or self.default_ttl)
    
    @asynccontextmanager
    async def lock(self, key: str, lock_type: LockType = LockType.EXCLUSIVE,
                   ttl: int = None, timeout: Optional[float] = None):
        """Context manager for acquiring and releasing locks.
        
        Args:
            key: Lock key
            lock_type: Type of lock
            ttl: Time to live in seconds
            timeout: Maximum wait time
        """
        redis_lock = self.create_lock(key, lock_type, ttl)
        
        try:
            success = await redis_lock.acquire(timeout)
            if not success:
                raise RuntimeError(f"Failed to acquire lock: {key}")
            yield redis_lock
        finally:
            if redis_lock._acquired:
                await redis_lock.release()
    
    async def _notify_waiters(self, key: str):
        """Notify processes waiting for lock."""
        waiters_key = self._waiters_key(key)
        
        try:
            # Add notification to waiters list
            await self.redis.lpush(waiters_key, "notification")
            await self.redis.expire(waiters_key, 60)  # Expire in 1 minute
            
        except Exception as e:
            self.logger.error(f"Error notifying waiters for {key}: {e}")
    
    async def _check_deadlock(self, key: str) -> bool:
        """Simple deadlock detection."""
        try:
            deadlock_key = self._deadlock_key()
            
            # Record lock request
            await self.redis.sadd(deadlock_key, key)
            await self.redis.expire(deadlock_key, 60)
            
            # Check if we have too many pending lock requests
            pending_count = await self.redis.scard(deadlock_key)
            
            return pending_count > 10  # Simple threshold-based detection
            
        except Exception:
            return False
    
    async def _cleanup_loop(self):
        """Background cleanup of expired locks and metrics."""
        while not self._shutdown:
            try:
                await self._cleanup_expired_locks()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(self.cleanup_interval)
    
    async def _cleanup_expired_locks(self):
        """Cleanup expired locks and update statistics."""
        try:
            # Cleanup local tracking
            with self._local_lock:
                expired_keys = []
                for key, lock_info in self.owned_locks.items():
                    if lock_info.is_expired():
                        expired_keys.append(key)
                
                for key in expired_keys:
                    self.owned_locks.pop(key, None)
                    self.stats.locks_expired += len(expired_keys)
            
            # Cleanup Redis keys - this is handled by Redis TTL
            # but we can clean up auxiliary keys
            pattern = f"{self.namespace}:waiters:*"
            cursor = 0
            
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
                
                for key in keys:
                    ttl = await self.redis.ttl(key)
                    if ttl <= 0:
                        await self.redis.delete(key)
                
                if cursor == 0:
                    break
        
        except Exception as e:
            self.logger.error(f"Error in expired locks cleanup: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get lock usage statistics."""
        uptime = (datetime.now(timezone.utc) - self.stats.start_time).total_seconds()
        
        with self._local_lock:
            owned_locks_count = len(self.owned_locks)
        
        return {
            'instance_id': self.instance_id,
            'uptime_seconds': uptime,
            'locks_acquired': self.stats.locks_acquired,
            'locks_released': self.stats.locks_released,
            'locks_expired': self.stats.locks_expired,
            'locks_failed': self.stats.locks_failed,
            'success_rate': self.stats.success_rate,
            'average_wait_time': self.stats.average_wait_time,
            'average_hold_time': self.stats.average_hold_time,
            'currently_owned': owned_locks_count,
            'locks_per_second': (self.stats.locks_acquired + self.stats.locks_released) / max(uptime, 1)
        }


import json  # Add this import at the top if not already present