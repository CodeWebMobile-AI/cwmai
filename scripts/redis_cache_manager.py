"""
Redis Cache Manager

Advanced caching system with Redis persistence, fallback mechanisms,
and performance optimization for the CWMAI API.
"""

import asyncio
import json
import logging
import pickle
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, List, Union, Callable
import redis


class CacheEntry:
    """Cache entry with metadata."""
    
    def __init__(self, value: Any, ttl: Optional[int] = None, tags: Optional[List[str]] = None):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.expires_at = self.created_at + ttl if ttl else None
        self.tags = tags or []
        self.access_count = 0
        self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "value": self.value,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "expires_at": self.expires_at,
            "tags": self.tags,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        entry = cls(data["value"], data["ttl"], data["tags"])
        entry.created_at = data["created_at"]
        entry.expires_at = data["expires_at"]
        entry.access_count = data["access_count"]
        entry.last_accessed = data["last_accessed"]
        return entry


class CacheStatistics:
    """Cache statistics tracking."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.evictions = 0
        self.errors = 0
        self.total_requests = 0
        self.average_response_time = 0.0
        self.start_time = time.time()
    
    def record_hit(self, response_time: float):
        """Record cache hit."""
        self.hits += 1
        self.total_requests += 1
        self._update_avg_response_time(response_time)
    
    def record_miss(self, response_time: float):
        """Record cache miss."""
        self.misses += 1
        self.total_requests += 1
        self._update_avg_response_time(response_time)
    
    def record_set(self):
        """Record cache set operation."""
        self.sets += 1
    
    def record_delete(self):
        """Record cache delete operation."""
        self.deletes += 1
    
    def record_eviction(self):
        """Record cache eviction."""
        self.evictions += 1
    
    def record_error(self):
        """Record cache error."""
        self.errors += 1
    
    def _update_avg_response_time(self, response_time: float):
        """Update average response time."""
        self.average_response_time = (
            (self.average_response_time * (self.total_requests - 1) + response_time)
            / self.total_requests
        )
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    def get_miss_rate(self) -> float:
        """Get cache miss rate."""
        if self.total_requests == 0:
            return 0.0
        return self.misses / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "evictions": self.evictions,
            "errors": self.errors,
            "total_requests": self.total_requests,
            "hit_rate": self.get_hit_rate(),
            "miss_rate": self.get_miss_rate(),
            "average_response_time": self.average_response_time,
            "uptime_seconds": time.time() - self.start_time
        }


class RedisCacheManager:
    """Advanced Redis cache manager with fallback and optimization."""
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        default_ttl: int = 3600,
        max_memory_items: int = 10000,
        namespace: str = "cwmai"
    ):
        """Initialize cache manager."""
        self.redis_client = redis_client
        self.default_ttl = default_ttl
        self.max_memory_items = max_memory_items
        self.namespace = namespace
        self.logger = logging.getLogger(f"{__name__}.RedisCacheManager")
        
        # In-memory fallback cache
        self.memory_cache: Dict[str, CacheEntry] = {}
        
        # Statistics
        self.stats = CacheStatistics()
        
        # Background tasks
        self._cleanup_task = None
        self._start_background_tasks()
    
    def _make_key(self, key: str) -> str:
        """Create namespaced cache key."""
        return f"{self.namespace}:cache:{key}"
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        start_time = time.time()
        namespaced_key = self._make_key(key)
        
        try:
            # Try Redis first
            if self.redis_client:
                try:
                    cached_data = self.redis_client.get(namespaced_key)
                    if cached_data:
                        entry_dict = json.loads(cached_data)
                        entry = CacheEntry.from_dict(entry_dict)
                        
                        if not entry.is_expired():
                            entry.touch()
                            # Update access stats in Redis asynchronously
                            asyncio.create_task(self._update_redis_entry(namespaced_key, entry))
                            
                            self.stats.record_hit(time.time() - start_time)
                            return entry.value
                        else:
                            # Remove expired entry
                            self.redis_client.delete(namespaced_key)
                            
                except Exception as e:
                    self.logger.warning(f"Redis get error for key {key}: {e}")
                    self.stats.record_error()
            
            # Fallback to memory cache
            if namespaced_key in self.memory_cache:
                entry = self.memory_cache[namespaced_key]
                if not entry.is_expired():
                    entry.touch()
                    self.stats.record_hit(time.time() - start_time)
                    return entry.value
                else:
                    del self.memory_cache[namespaced_key]
                    self.stats.record_eviction()
            
            # Cache miss
            self.stats.record_miss(time.time() - start_time)
            return default
            
        except Exception as e:
            self.logger.error(f"Cache get error for key {key}: {e}")
            self.stats.record_error()
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache."""
        try:
            ttl = ttl or self.default_ttl
            namespaced_key = self._make_key(key)
            entry = CacheEntry(value, ttl, tags)
            
            # Try Redis first
            if self.redis_client:
                try:
                    entry_data = json.dumps(entry.to_dict())
                    self.redis_client.setex(namespaced_key, ttl, entry_data)
                    
                    # Also store tags for tag-based operations
                    if tags:
                        for tag in tags:
                            tag_key = f"{self.namespace}:tags:{tag}"
                            self.redis_client.sadd(tag_key, namespaced_key)
                            self.redis_client.expire(tag_key, ttl + 60)  # Slightly longer TTL
                    
                    self.stats.record_set()
                    return True
                    
                except Exception as e:
                    self.logger.warning(f"Redis set error for key {key}: {e}")
                    self.stats.record_error()
            
            # Fallback to memory cache
            await self._ensure_memory_cache_capacity()
            self.memory_cache[namespaced_key] = entry
            self.stats.record_set()
            return True
            
        except Exception as e:
            self.logger.error(f"Cache set error for key {key}: {e}")
            self.stats.record_error()
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            namespaced_key = self._make_key(key)
            deleted = False
            
            # Delete from Redis
            if self.redis_client:
                try:
                    result = self.redis_client.delete(namespaced_key)
                    deleted = result > 0
                except Exception as e:
                    self.logger.warning(f"Redis delete error for key {key}: {e}")
                    self.stats.record_error()
            
            # Delete from memory cache
            if namespaced_key in self.memory_cache:
                del self.memory_cache[namespaced_key]
                deleted = True
            
            if deleted:
                self.stats.record_delete()
            
            return deleted
            
        except Exception as e:
            self.logger.error(f"Cache delete error for key {key}: {e}")
            self.stats.record_error()
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        namespaced_key = self._make_key(key)
        
        # Check Redis
        if self.redis_client:
            try:
                if self.redis_client.exists(namespaced_key):
                    return True
            except Exception as e:
                self.logger.warning(f"Redis exists error for key {key}: {e}")
        
        # Check memory cache
        if namespaced_key in self.memory_cache:
            entry = self.memory_cache[namespaced_key]
            if not entry.is_expired():
                return True
            else:
                del self.memory_cache[namespaced_key]
                self.stats.record_eviction()
        
        return False
    
    async def get_or_set(
        self,
        key: str,
        callback: Callable[[], Any],
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> Any:
        """Get value from cache or set it using callback."""
        value = await self.get(key)
        if value is not None:
            return value
        
        # Generate value using callback
        if asyncio.iscoroutinefunction(callback):
            value = await callback()
        else:
            value = callback()
        
        await self.set(key, value, ttl, tags)
        return value
    
    async def clear_by_tags(self, tags: List[str]) -> int:
        """Clear cache entries by tags."""
        cleared_count = 0
        
        if self.redis_client:
            try:
                for tag in tags:
                    tag_key = f"{self.namespace}:tags:{tag}"
                    keys = self.redis_client.smembers(tag_key)
                    
                    if keys:
                        self.redis_client.delete(*keys)
                        self.redis_client.delete(tag_key)
                        cleared_count += len(keys)
                        
            except Exception as e:
                self.logger.error(f"Redis clear by tags error: {e}")
                self.stats.record_error()
        
        # Clear from memory cache
        keys_to_remove = []
        for key, entry in self.memory_cache.items():
            if any(tag in entry.tags for tag in tags):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.memory_cache[key]
            cleared_count += 1
            self.stats.record_delete()
        
        return cleared_count
    
    async def clear_all(self) -> bool:
        """Clear all cache entries."""
        try:
            cleared = False
            
            # Clear Redis
            if self.redis_client:
                try:
                    keys = self.redis_client.keys(f"{self.namespace}:*")
                    if keys:
                        self.redis_client.delete(*keys)
                        cleared = True
                except Exception as e:
                    self.logger.error(f"Redis clear all error: {e}")
                    self.stats.record_error()
            
            # Clear memory cache
            if self.memory_cache:
                self.memory_cache.clear()
                cleared = True
            
            return cleared
            
        except Exception as e:
            self.logger.error(f"Cache clear all error: {e}")
            self.stats.record_error()
            return False
    
    async def get_info(self) -> Dict[str, Any]:
        """Get cache information and statistics."""
        info = {
            "namespace": self.namespace,
            "default_ttl": self.default_ttl,
            "max_memory_items": self.max_memory_items,
            "storage": {
                "redis_available": self.redis_client is not None,
                "memory_cache_size": len(self.memory_cache)
            },
            "statistics": self.stats.to_dict()
        }
        
        # Add Redis info if available
        if self.redis_client:
            try:
                redis_info = self.redis_client.info()
                info["redis_info"] = {
                    "version": redis_info.get("redis_version"),
                    "connected_clients": redis_info.get("connected_clients"),
                    "used_memory": redis_info.get("used_memory_human"),
                    "keyspace_hits": redis_info.get("keyspace_hits"),
                    "keyspace_misses": redis_info.get("keyspace_misses")
                }
                
                # Count our keys
                our_keys = self.redis_client.keys(f"{self.namespace}:*")
                info["storage"]["redis_cache_keys"] = len(our_keys)
                
            except Exception as e:
                self.logger.warning(f"Error getting Redis info: {e}")
        
        return info
    
    async def _update_redis_entry(self, key: str, entry: CacheEntry):
        """Update Redis entry access statistics."""
        if not self.redis_client:
            return
        
        try:
            entry_data = json.dumps(entry.to_dict())
            ttl = self.redis_client.ttl(key)
            if ttl > 0:
                self.redis_client.setex(key, ttl, entry_data)
        except Exception as e:
            self.logger.debug(f"Error updating Redis entry stats: {e}")
    
    async def _ensure_memory_cache_capacity(self):
        """Ensure memory cache doesn't exceed capacity."""
        if len(self.memory_cache) >= self.max_memory_items:
            # Remove oldest/least accessed entries
            entries_with_keys = [
                (key, entry) for key, entry in self.memory_cache.items()
            ]
            
            # Sort by last access time (LRU)
            entries_with_keys.sort(key=lambda x: x[1].last_accessed)
            
            # Remove 10% of entries
            to_remove = max(1, len(entries_with_keys) // 10)
            for i in range(to_remove):
                key_to_remove = entries_with_keys[i][0]
                del self.memory_cache[key_to_remove]
                self.stats.record_eviction()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        async def cleanup_expired():
            """Clean up expired entries from memory cache."""
            while True:
                try:
                    await asyncio.sleep(60)  # Run every minute
                    
                    expired_keys = []
                    for key, entry in self.memory_cache.items():
                        if entry.is_expired():
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        del self.memory_cache[key]
                        self.stats.record_eviction()
                    
                    if expired_keys:
                        self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                        
                except Exception as e:
                    self.logger.error(f"Cache cleanup error: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_expired())
    
    async def shutdown(self):
        """Shutdown cache manager and cleanup."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Cache manager shutdown complete")