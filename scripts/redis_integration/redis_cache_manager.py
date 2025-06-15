"""
Redis Cache Manager

Advanced caching system with Redis backend featuring semantic similarity,
TTL management, cache warming, compression, and distributed invalidation.
"""

import asyncio
import json
import time
import hashlib
import zlib
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import logging
import numpy as np
from .redis_client import RedisClient


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now(timezone.utc) - self.created_at).total_seconds() > self.ttl_seconds
    
    def update_access(self):
        """Update access statistics."""
        self.accessed_at = datetime.now(timezone.utc)
        self.access_count += 1


class CacheStats:
    """Cache statistics tracking."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.evictions = 0
        self.memory_usage = 0
        self.total_entries = 0
        self.start_time = datetime.now(timezone.utc)
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return {
            'hits': self.hits,
            'misses': self.misses,
            'sets': self.sets,
            'deletes': self.deletes,
            'evictions': self.evictions,
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate,
            'memory_usage': self.memory_usage,
            'total_entries': self.total_entries,
            'uptime_seconds': uptime,
            'operations_per_second': (self.hits + self.misses + self.sets) / max(uptime, 1)
        }


class RedisCacheManager:
    """Advanced Redis cache manager with enterprise features."""
    
    def __init__(self, redis_client: RedisClient, namespace: str = "cache"):
        """Initialize Redis cache manager.
        
        Args:
            redis_client: Redis client instance
            namespace: Cache namespace for key isolation
        """
        self.redis = redis_client
        self.namespace = namespace
        self.stats = CacheStats()
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.default_ttl = 3600  # 1 hour
        self.max_key_length = 250
        self.compression_threshold = 1024  # Compress values > 1KB
        self.enable_compression = True
        self.enable_stats = True
        
        # Semantic similarity configuration
        self.similarity_threshold = 0.8
        self.embedding_cache_ttl = 86400  # 24 hours
        
        # Cache warming configuration
        self.warming_batch_size = 100
        self.warming_delay = 0.01  # 10ms between batches
    
    def _make_key(self, key: str) -> str:
        """Create namespaced cache key."""
        if len(key) > self.max_key_length:
            # Hash long keys
            key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
            key = f"{key[:self.max_key_length-20]}_{key_hash}"
        return f"{self.namespace}:{key}"
    
    def _stats_key(self, stat_type: str) -> str:
        """Create cache stats key."""
        return f"{self.namespace}:stats:{stat_type}"
    
    def _metadata_key(self, key: str) -> str:
        """Create metadata key."""
        return f"{self.namespace}:meta:{key}"
    
    def _embedding_key(self, content_hash: str) -> str:
        """Create embedding cache key."""
        return f"{self.namespace}:embeddings:{content_hash}"
    
    def _tags_key(self, tag: str) -> str:
        """Create tag index key."""
        return f"{self.namespace}:tags:{tag}"
    
    async def _serialize_value(self, value: Any) -> bytes:
        """Serialize and optionally compress value."""
        try:
            # Serialize to JSON
            serialized = json.dumps(value, default=str).encode('utf-8')
            
            # Compress if above threshold
            if self.enable_compression and len(serialized) > self.compression_threshold:
                compressed = zlib.compress(serialized)
                # Include compression flag
                return b'COMPRESSED:' + compressed
            
            return serialized
            
        except Exception as e:
            self.logger.error(f"Serialization error: {e}")
            raise
    
    async def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize and decompress value."""
        try:
            # Check for compression
            if data.startswith(b'COMPRESSED:'):
                compressed_data = data[11:]  # Remove prefix
                decompressed = zlib.decompress(compressed_data)
                return json.loads(decompressed.decode('utf-8'))
            
            return json.loads(data.decode('utf-8'))
            
        except Exception as e:
            self.logger.error(f"Deserialization error: {e}")
            raise
    
    async def _update_stats(self, operation: str, size: int = 0):
        """Update cache statistics."""
        if not self.enable_stats:
            return
        
        if operation == 'hit':
            self.stats.hits += 1
        elif operation == 'miss':
            self.stats.misses += 1
        elif operation == 'set':
            self.stats.sets += 1
            self.stats.memory_usage += size
            self.stats.total_entries += 1
        elif operation == 'delete':
            self.stats.deletes += 1
            self.stats.memory_usage -= size
            self.stats.total_entries -= 1
        elif operation == 'eviction':
            self.stats.evictions += 1
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        cache_key = self._make_key(key)
        
        try:
            # Get value from Redis
            data = await self.redis.get(cache_key)
            
            if data is None:
                await self._update_stats('miss')
                return default
            
            # Deserialize value
            value = await self._deserialize_value(data)
            
            # Update access statistics
            await self._update_access_stats(cache_key)
            await self._update_stats('hit')
            
            return value
            
        except Exception as e:
            self.logger.error(f"Cache get error for key {key}: {e}")
            await self._update_stats('miss')
            return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                 tags: List[str] = None, metadata: Dict[str, Any] = None) -> bool:
        """Set value in cache with optional TTL and tags."""
        cache_key = self._make_key(key)
        ttl = ttl or self.default_ttl
        tags = tags or []
        metadata = metadata or {}
        
        try:
            # Serialize value
            serialized = await self._serialize_value(value)
            
            # Set value with TTL
            success = await self.redis.set(cache_key, serialized, ex=ttl)
            
            if success:
                # Store metadata
                entry_metadata = {
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'size_bytes': len(serialized),
                    'ttl_seconds': ttl,
                    'tags': tags,
                    'metadata': metadata
                }
                
                meta_key = self._metadata_key(key)
                await self.redis.set(meta_key, json.dumps(entry_metadata), ex=ttl)
                
                # Update tag indexes
                for tag in tags:
                    tag_key = self._tags_key(tag)
                    await self.redis.sadd(tag_key, cache_key)
                    await self.redis.expire(tag_key, ttl)
                
                await self._update_stats('set', len(serialized))
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        cache_key = self._make_key(key)
        
        try:
            # Get metadata for size tracking
            meta_key = self._metadata_key(key)
            meta_data = await self.redis.get(meta_key)
            size = 0
            tags = []
            
            if meta_data:
                metadata = json.loads(meta_data)
                size = metadata.get('size_bytes', 0)
                tags = metadata.get('tags', [])
            
            # Delete value and metadata
            deleted = await self.redis.delete(cache_key, meta_key)
            
            # Remove from tag indexes
            for tag in tags:
                tag_key = self._tags_key(tag)
                await self.redis.srem(tag_key, cache_key)
            
            if deleted > 0:
                await self._update_stats('delete', size)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        cache_key = self._make_key(key)
        try:
            return bool(await self.redis.exists(cache_key))
        except Exception as e:
            self.logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for cache key."""
        cache_key = self._make_key(key)
        try:
            # Update both value and metadata TTL
            meta_key = self._metadata_key(key)
            await self.redis.expire(cache_key, ttl)
            return await self.redis.expire(meta_key, ttl)
        except Exception as e:
            self.logger.error(f"Cache expire error for key {key}: {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """Get TTL for cache key."""
        cache_key = self._make_key(key)
        try:
            return await self.redis.ttl(cache_key)
        except Exception as e:
            self.logger.error(f"Cache TTL error for key {key}: {e}")
            return -1
    
    async def clear_by_tag(self, tag: str) -> int:
        """Clear all cache entries with specific tag."""
        tag_key = self._tags_key(tag)
        cleared = 0
        
        try:
            # Get all keys with this tag
            cache_keys = await self.redis.smembers(tag_key)
            
            if cache_keys:
                # Delete cache entries
                deleted = await self.redis.delete(*cache_keys)
                cleared += deleted
                
                # Delete metadata entries
                meta_keys = [self._metadata_key(k.split(':', 1)[1]) for k in cache_keys]
                await self.redis.delete(*meta_keys)
                
                # Delete tag index
                await self.redis.delete(tag_key)
                
                await self._update_stats('delete', cleared * 100)  # Estimate size
            
            return cleared
            
        except Exception as e:
            self.logger.error(f"Cache clear by tag error for tag {tag}: {e}")
            return 0
    
    async def clear_all(self) -> int:
        """Clear all cache entries in namespace."""
        pattern = f"{self.namespace}:*"
        cleared = 0
        
        try:
            # Use SCAN to avoid blocking
            cursor = 0
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=1000)
                
                if keys:
                    deleted = await self.redis.delete(*keys)
                    cleared += deleted
                
                if cursor == 0:
                    break
            
            # Reset stats
            self.stats = CacheStats()
            return cleared
            
        except Exception as e:
            self.logger.error(f"Cache clear all error: {e}")
            return 0
    
    async def get_multi(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        cache_keys = [self._make_key(key) for key in keys]
        results = {}
        
        try:
            # Use pipeline for efficient multi-get
            async with self.redis.pipeline() as pipe:
                for cache_key in cache_keys:
                    pipe.get(cache_key)
                values = await pipe.execute()
            
            # Process results
            for i, (original_key, value) in enumerate(zip(keys, values)):
                if value is not None:
                    try:
                        results[original_key] = await self._deserialize_value(value)
                        await self._update_stats('hit')
                    except Exception:
                        await self._update_stats('miss')
                else:
                    await self._update_stats('miss')
            
            return results
            
        except Exception as e:
            self.logger.error(f"Cache multi-get error: {e}")
            return {}
    
    async def set_multi(self, data: Dict[str, Any], ttl: Optional[int] = None) -> int:
        """Set multiple values in cache."""
        ttl = ttl or self.default_ttl
        set_count = 0
        
        try:
            # Use pipeline for efficient multi-set
            async with self.redis.pipeline() as pipe:
                for key, value in data.items():
                    cache_key = self._make_key(key)
                    serialized = await self._serialize_value(value)
                    pipe.set(cache_key, serialized, ex=ttl)
                
                results = await pipe.execute()
                set_count = sum(1 for r in results if r)
            
            await self._update_stats('set', set_count * 100)  # Estimate size
            return set_count
            
        except Exception as e:
            self.logger.error(f"Cache multi-set error: {e}")
            return 0
    
    async def get_or_set(self, key: str, value_func: Callable, ttl: Optional[int] = None,
                        tags: List[str] = None) -> Any:
        """Get value or set it using provided function."""
        # Try to get existing value
        value = await self.get(key)
        
        if value is not None:
            return value
        
        # Generate new value
        try:
            if asyncio.iscoroutinefunction(value_func):
                new_value = await value_func()
            else:
                new_value = value_func()
            
            # Cache the new value
            await self.set(key, new_value, ttl=ttl, tags=tags)
            return new_value
            
        except Exception as e:
            self.logger.error(f"Cache get_or_set error for key {key}: {e}")
            raise
    
    async def _update_access_stats(self, cache_key: str):
        """Update access statistics for cache entry."""
        try:
            # Increment access count
            access_key = f"{cache_key}:access"
            await self.redis.incr(access_key)
            await self.redis.expire(access_key, self.default_ttl)
        except Exception:
            pass  # Non-critical operation
    
    async def get_cache_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cache entry information."""
        meta_key = self._metadata_key(key)
        cache_key = self._make_key(key)
        
        try:
            # Get metadata
            meta_data = await self.redis.get(meta_key)
            if not meta_data:
                return None
            
            metadata = json.loads(meta_data)
            
            # Get additional info
            ttl = await self.redis.ttl(cache_key)
            access_key = f"{cache_key}:access"
            access_count = await self.redis.get(access_key) or 0
            
            return {
                **metadata,
                'ttl_remaining': ttl,
                'access_count': int(access_count)
            }
            
        except Exception as e:
            self.logger.error(f"Cache info error for key {key}: {e}")
            return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.stats.to_dict()
        
        try:
            # Get Redis memory info
            info = await self.redis.info('memory')
            stats['redis_memory'] = {
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'memory_fragmentation_ratio': info.get('mem_fragmentation_ratio', 1.0)
            }
        except Exception:
            pass
        
        return stats
    
    async def warm_cache(self, data_loader: Callable, keys: List[str] = None,
                        batch_size: int = None) -> int:
        """Warm cache with data from loader function."""
        batch_size = batch_size or self.warming_batch_size
        warmed = 0
        
        try:
            if keys:
                # Warm specific keys
                for i in range(0, len(keys), batch_size):
                    batch_keys = keys[i:i + batch_size]
                    
                    for key in batch_keys:
                        if not await self.exists(key):
                            if asyncio.iscoroutinefunction(data_loader):
                                value = await data_loader(key)
                            else:
                                value = data_loader(key)
                            
                            if value is not None:
                                await self.set(key, value)
                                warmed += 1
                    
                    await asyncio.sleep(self.warming_delay)
            
            return warmed
            
        except Exception as e:
            self.logger.error(f"Cache warming error: {e}")
            return warmed
    
    async def semantic_search(self, query_embedding: List[float], 
                             threshold: float = None, limit: int = 10) -> List[Tuple[str, float]]:
        """Find semantically similar cache entries."""
        threshold = threshold or self.similarity_threshold
        results = []
        
        try:
            # Get all embedding keys
            pattern = f"{self.namespace}:embeddings:*"
            cursor = 0
            
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
                
                for key in keys:
                    embedding_data = await self.redis.get(key)
                    if embedding_data:
                        stored_embedding = json.loads(embedding_data)
                        
                        # Calculate cosine similarity
                        similarity = self._cosine_similarity(query_embedding, stored_embedding)
                        
                        if similarity >= threshold:
                            # Extract original key
                            original_key = key.split(':', 2)[2]  # Remove namespace:embeddings:
                            results.append((original_key, similarity))
                
                if cursor == 0:
                    break
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"Semantic search error: {e}")
            return []
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            np_a = np.array(a)
            np_b = np.array(b)
            return np.dot(np_a, np_b) / (np.linalg.norm(np_a) * np.linalg.norm(np_b))
        except Exception:
            return 0.0
    
    async def store_embedding(self, key: str, embedding: List[float], ttl: Optional[int] = None):
        """Store embedding for semantic similarity search."""
        embedding_key = self._embedding_key(key)
        ttl = ttl or self.embedding_cache_ttl
        
        try:
            serialized = json.dumps(embedding)
            await self.redis.set(embedding_key, serialized, ex=ttl)
        except Exception as e:
            self.logger.error(f"Store embedding error for key {key}: {e}")
    
    async def close(self):
        """Close cache manager and cleanup resources."""
        # Cache manager doesn't own the Redis client
        # Client should be closed by the owner
        pass