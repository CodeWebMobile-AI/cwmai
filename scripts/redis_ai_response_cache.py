"""
Redis-Backed AI Response Cache

Enhanced AI response cache with Redis backend, dual-write migration strategy,
and advanced features including distributed caching, semantic similarity, and analytics.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np

try:
    from scripts.redis_integration import (
        RedisClient, 
        RedisCacheManager, 
        RedisAnalytics,
        get_redis_client
    )
except ImportError:
    from redis_integration import (
        RedisClient, 
        RedisCacheManager, 
        RedisAnalytics,
        get_redis_client
    )

try:
    from scripts.ai_response_cache import CacheEntry, CacheStats, AIResponseCache
except ImportError:
    from ai_response_cache import CacheEntry, CacheStats, AIResponseCache

try:
    from scripts.mcp_redis_integration import MCPRedisIntegration
except ImportError:
    MCPRedisIntegration = None


@dataclass
class RedisEnhancedCacheEntry(CacheEntry):
    """Enhanced cache entry with Redis-specific features."""
    tags: List[str] = None
    access_count: int = 0
    last_accessed: datetime = None
    cache_tier: str = "hot"  # hot, warm, cold
    compression_ratio: float = 1.0
    semantic_cluster: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.last_accessed is None:
            self.last_accessed = self.timestamp


@dataclass
class RedisEnhancedCacheStats(CacheStats):
    """Enhanced cache statistics with Redis metrics."""
    redis_hits: int = 0
    redis_misses: int = 0
    memory_cache_hits: int = 0
    memory_cache_misses: int = 0
    compression_savings: float = 0.0
    average_embedding_similarity: float = 0.0
    cache_warming_requests: int = 0
    distributed_invalidations: int = 0
    
    @property
    def total_hit_rate(self) -> float:
        """Calculate total hit rate across all cache layers."""
        total_requests = self.total_requests
        total_hits = self.cache_hits + self.semantic_hits + self.redis_hits
        return total_hits / max(total_requests, 1)
    
    @property
    def redis_hit_rate(self) -> float:
        """Calculate Redis-specific hit rate."""
        redis_total = self.redis_hits + self.redis_misses
        return self.redis_hits / max(redis_total, 1)


class RedisAIResponseCache:
    """Advanced Redis-backed AI response cache with migration support."""
    
    def __init__(self,
                 redis_client: Optional[RedisClient] = None,
                 fallback_cache: Optional[AIResponseCache] = None,
                 namespace: str = "ai_cache",
                 default_ttl: int = 3600,
                 similarity_threshold: float = 0.85,
                 enable_embeddings: bool = True,
                 enable_compression: bool = True,
                 enable_analytics: bool = True,
                 dual_write_mode: bool = True,
                 migration_mode: str = "gradual"):  # gradual, immediate, readonly
        """Initialize Redis AI response cache.
        
        Args:
            redis_client: Redis client instance
            fallback_cache: Fallback in-memory cache for migration
            namespace: Redis namespace for cache isolation
            default_ttl: Default time-to-live in seconds
            similarity_threshold: Minimum similarity for semantic matches
            enable_embeddings: Enable semantic similarity matching
            enable_compression: Enable value compression
            enable_analytics: Enable advanced analytics
            dual_write_mode: Write to both Redis and fallback cache
            migration_mode: Migration strategy (gradual/immediate/readonly)
        """
        self.redis_client = redis_client
        self.fallback_cache = fallback_cache
        self.namespace = namespace
        self.default_ttl = default_ttl
        self.similarity_threshold = similarity_threshold
        self.enable_embeddings = enable_embeddings
        self.enable_compression = enable_compression
        self.enable_analytics = enable_analytics
        self.dual_write_mode = dual_write_mode
        self.migration_mode = migration_mode
        
        # Initialize Redis components
        self.redis_cache: Optional[RedisCacheManager] = None
        self.analytics: Optional[RedisAnalytics] = None
        
        # Enhanced statistics
        self.stats = RedisEnhancedCacheStats()
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.RedisAIResponseCache")
        
        # Background tasks
        self._migration_task: Optional[asyncio.Task] = None
        self._analytics_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Migration tracking
        self._migration_progress = {
            'total_entries': 0,
            'migrated_entries': 0,
            'failed_migrations': 0,
            'start_time': None
        }
        
        # MCP-Redis integration
        self.mcp_redis: Optional[MCPRedisIntegration] = None
        self._use_mcp = os.getenv("USE_MCP_REDIS", "false").lower() == "true"
    
    async def initialize(self):
        """Initialize Redis cache components and start background tasks."""
        try:
            # Initialize Redis client if not provided
            if self.redis_client is None:
                self.redis_client = await get_redis_client()
            
            # Initialize Redis cache manager
            self.redis_cache = RedisCacheManager(
                self.redis_client, 
                namespace=self.namespace
            )
            
            # Initialize analytics if enabled
            if self.enable_analytics:
                self.analytics = RedisAnalytics(
                    self.redis_client,
                    namespace=f"{self.namespace}_analytics"
                )
                await self.analytics.start()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Initialize MCP-Redis if enabled
            if self._use_mcp:
                try:
                    self.mcp_redis = MCPRedisIntegration()
                    await self.mcp_redis.initialize()
                    self.logger.info("MCP-Redis integration enabled for AI response cache")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize MCP-Redis: {e}")
                    self._use_mcp = False
            
            self.logger.info(f"Redis AI Response Cache initialized (mode: {self.migration_mode})")
            
        except Exception as e:
            self.logger.error(f"Error initializing Redis AI cache: {e}")
            raise
    
    async def _start_background_tasks(self):
        """Start background processing tasks."""
        if self.migration_mode == "gradual" and self.fallback_cache:
            self._migration_task = asyncio.create_task(self._gradual_migration_worker())
        
        if self.enable_analytics:
            self._analytics_task = asyncio.create_task(self._analytics_worker())
        
        self._cleanup_task = asyncio.create_task(self._cleanup_worker())
    
    async def get(self, prompt: str, provider: str, model: str) -> Optional[str]:
        """Get cached response with Redis backend and fallback support.
        
        Args:
            prompt: The AI prompt
            provider: AI provider (anthropic, openai, etc.)
            model: Model name
            
        Returns:
            Cached response content or None if not found
        """
        start_time = time.time()
        self.stats.total_requests += 1
        
        prompt_hash = self._generate_prompt_hash(prompt, provider, model)
        
        # Try Redis cache first (if not in readonly migration mode)
        if self.migration_mode != "readonly" and self.redis_cache:
            try:
                redis_result = await self._get_from_redis(prompt_hash, prompt, provider, model)
                if redis_result:
                    self.stats.redis_hits += 1
                    self.stats.cache_hits += 1
                    self.stats.time_saved += time.time() - start_time
                    
                    # Update analytics
                    if self.enable_analytics:
                        await self._record_cache_hit("redis", provider, model)
                    
                    self.logger.debug(f"Redis cache HIT: {prompt_hash}")
                    return redis_result
                else:
                    self.stats.redis_misses += 1
            except Exception as e:
                self.logger.warning(f"Redis cache error, falling back: {e}")
        
        # Try fallback cache
        if self.fallback_cache:
            try:
                fallback_result = await self.fallback_cache.get(prompt, provider, model)
                if fallback_result:
                    self.stats.memory_cache_hits += 1
                    self.stats.cache_hits += 1
                    self.stats.time_saved += time.time() - start_time
                    
                    # Migrate to Redis if dual-write enabled
                    if self.dual_write_mode and self.redis_cache:
                        asyncio.create_task(self._migrate_entry_to_redis(
                            prompt, fallback_result, provider, model
                        ))
                    
                    self.logger.debug(f"Fallback cache HIT: {prompt_hash}")
                    return fallback_result
                else:
                    self.stats.memory_cache_misses += 1
            except Exception as e:
                self.logger.error(f"Fallback cache error: {e}")
        
        self.stats.cache_misses += 1
        self.logger.debug(f"Cache MISS: {prompt_hash}")
        return None
    
    async def _get_from_redis(self, prompt_hash: str, prompt: str, 
                             provider: str, model: str) -> Optional[str]:
        """Get response from Redis with semantic similarity support."""
        try:
            # Try exact match first
            cache_key = f"exact:{prompt_hash}"
            result = await self.redis_cache.get(cache_key)
            
            if result:
                # Update access statistics
                await self._update_access_stats(cache_key)
                return result
            
            # Try semantic similarity if enabled
            if self.enable_embeddings:
                return await self._semantic_search_redis(prompt, provider, model)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting from Redis: {e}")
            return None
    
    async def _semantic_search_redis(self, prompt: str, provider: str, model: str) -> Optional[str]:
        """Perform semantic similarity search in Redis."""
        try:
            # Generate embedding for query
            query_embedding = self._generate_embedding(prompt)
            if not query_embedding:
                return None
            
            # Search for similar embeddings
            similar_entries = await self.redis_cache.semantic_search(
                query_embedding,
                threshold=self.similarity_threshold,
                limit=5
            )
            
            for cache_key, similarity in similar_entries:
                # Verify provider and model match
                entry_data = await self.redis_cache.get(cache_key)
                if entry_data:
                    try:
                        entry_info = json.loads(entry_data)
                        if (entry_info.get('provider') == provider and 
                            entry_info.get('model') == model):
                            
                            self.stats.semantic_hits += 1
                            self.stats.average_embedding_similarity = (
                                (self.stats.average_embedding_similarity * self.stats.semantic_hits + similarity) / 
                                (self.stats.semantic_hits + 1)
                            )
                            
                            await self._update_access_stats(cache_key)
                            self.logger.debug(f"Semantic cache HIT: {cache_key} (similarity: {similarity:.3f})")
                            
                            return entry_info.get('content')
                    except json.JSONDecodeError:
                        continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return None
    
    async def put(self, 
                  prompt: str, 
                  response: str, 
                  provider: str, 
                  model: str,
                  ttl_seconds: Optional[int] = None,
                  cost_estimate: float = 0.0) -> None:
        """Store response in cache with Redis backend and dual-write support.
        
        Args:
            prompt: The AI prompt
            response: The AI response
            provider: AI provider
            model: Model name
            ttl_seconds: Time-to-live override
            cost_estimate: Estimated cost of this API call
        """
        if not response:
            return
        
        prompt_hash = self._generate_prompt_hash(prompt, provider, model)
        ttl = ttl_seconds or self.default_ttl
        
        # Store in Redis (if not in readonly mode)
        if self.migration_mode != "readonly" and self.redis_cache:
            try:
                await self._store_in_redis(prompt, response, provider, model, ttl, cost_estimate)
            except Exception as e:
                self.logger.error(f"Error storing in Redis: {e}")
        
        # Store in fallback cache (if dual-write enabled or Redis unavailable)
        if self.dual_write_mode and self.fallback_cache:
            try:
                await self.fallback_cache.put(prompt, response, provider, model, ttl, cost_estimate)
            except Exception as e:
                self.logger.error(f"Error storing in fallback cache: {e}")
        
        # Update statistics
        self.stats.cost_saved += cost_estimate
        
        # Record analytics
        if self.enable_analytics:
            await self._record_cache_store(provider, model, len(response), cost_estimate)
        
        self.logger.debug(f"Cache stored: {prompt_hash} (TTL: {ttl}s)")
    
    async def _store_in_redis(self, prompt: str, response: str, provider: str, 
                             model: str, ttl: int, cost_estimate: float):
        """Store entry in Redis with advanced features."""
        try:
            prompt_hash = self._generate_prompt_hash(prompt, provider, model)
            
            # Create enhanced cache entry
            entry = RedisEnhancedCacheEntry(
                content=response,
                embedding=self._generate_embedding(prompt) if self.enable_embeddings else [],
                timestamp=datetime.now(timezone.utc),
                ttl_seconds=ttl,
                provider=provider,
                model=model,
                prompt_hash=prompt_hash,
                usage_count=0,
                cost_estimate=cost_estimate,
                tags=[provider, model, "ai_response"],
                cache_tier="hot"
            )
            
            # Store entry data
            cache_key = f"exact:{prompt_hash}"
            entry_data = json.dumps({
                'content': entry.content,
                'provider': entry.provider,
                'model': entry.model,
                'timestamp': entry.timestamp.isoformat(),
                'cost_estimate': entry.cost_estimate,
                'prompt_hash': entry.prompt_hash
            })
            
            success = await self.redis_cache.set(
                cache_key,
                entry_data,
                ttl=ttl,
                tags=entry.tags,
                metadata={
                    'provider': provider,
                    'model': model,
                    'prompt_length': len(prompt),
                    'response_length': len(response)
                }
            )
            
            if success and self.enable_embeddings and entry.embedding:
                # Store embedding for semantic search
                await self.redis_cache.store_embedding(cache_key, entry.embedding, ttl)
            
        except Exception as e:
            self.logger.error(f"Error storing in Redis: {e}")
            raise
    
    async def _migrate_entry_to_redis(self, prompt: str, response: str, 
                                     provider: str, model: str):
        """Migrate a single entry from fallback cache to Redis."""
        try:
            if self.redis_cache:
                await self._store_in_redis(prompt, response, provider, model, self.default_ttl, 0.0)
                self._migration_progress['migrated_entries'] += 1
                self.logger.debug(f"Migrated entry to Redis: {self._generate_prompt_hash(prompt, provider, model)}")
        except Exception as e:
            self._migration_progress['failed_migrations'] += 1
            self.logger.error(f"Error migrating entry to Redis: {e}")
    
    async def _gradual_migration_worker(self):
        """Background worker for gradual migration from fallback to Redis."""
        if not self.fallback_cache:
            return
        
        self.logger.info("Starting gradual migration to Redis")
        self._migration_progress['start_time'] = datetime.now(timezone.utc)
        
        try:
            # Get fallback cache entries
            fallback_entries = list(self.fallback_cache.cache.items())
            self._migration_progress['total_entries'] = len(fallback_entries)
            
            # Migrate entries gradually
            for i, (prompt_hash, entry) in enumerate(fallback_entries):
                if self._shutdown:
                    break
                
                try:
                    # Reconstruct prompt (this is a limitation - we need to store prompts)
                    # For now, we'll migrate the entry data we have
                    await self._store_in_redis(
                        f"migrated_{prompt_hash}",  # Placeholder prompt
                        entry.content,
                        entry.provider,
                        entry.model,
                        entry.ttl_seconds,
                        entry.cost_estimate
                    )
                    
                    self._migration_progress['migrated_entries'] += 1
                    
                    # Progress logging
                    if (i + 1) % 100 == 0:
                        progress = (i + 1) / len(fallback_entries) * 100
                        self.logger.info(f"Migration progress: {progress:.1f}% ({i + 1}/{len(fallback_entries)})")
                    
                    # Rate limiting to avoid overwhelming Redis
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    self._migration_progress['failed_migrations'] += 1
                    self.logger.error(f"Error migrating entry {prompt_hash}: {e}")
            
            migration_time = (datetime.now(timezone.utc) - self._migration_progress['start_time']).total_seconds()
            self.logger.info(f"Migration completed in {migration_time:.1f}s. "
                           f"Migrated: {self._migration_progress['migrated_entries']}, "
                           f"Failed: {self._migration_progress['failed_migrations']}")
            
        except Exception as e:
            self.logger.error(f"Error in migration worker: {e}")
    
    async def _analytics_worker(self):
        """Background worker for analytics collection."""
        while not self._shutdown:
            try:
                await asyncio.sleep(60)  # Collect analytics every minute
                
                if self.analytics:
                    # Record cache performance metrics
                    await self.analytics._record_metric(
                        "ai_cache_hit_rate",
                        self.stats.total_hit_rate,
                        self.analytics.MetricType.GAUGE
                    )
                    
                    await self.analytics._record_metric(
                        "ai_cache_requests_total",
                        self.stats.total_requests,
                        self.analytics.MetricType.COUNTER
                    )
                    
                    await self.analytics._record_metric(
                        "ai_cache_cost_saved",
                        self.stats.cost_saved,
                        self.analytics.MetricType.COUNTER
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in analytics worker: {e}")
    
    async def _cleanup_worker(self):
        """Background worker for cache cleanup and optimization."""
        while not self._shutdown:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Cleanup fallback cache if available
                if self.fallback_cache:
                    await self.fallback_cache.cleanup_expired()
                
                # Optimize Redis cache
                if self.redis_cache:
                    # This would involve Redis-specific optimizations
                    pass
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup worker: {e}")
    
    async def _update_access_stats(self, cache_key: str):
        """Update access statistics for cache entry."""
        try:
            if self.redis_cache:
                # This would update Redis-based access statistics
                pass
        except Exception as e:
            self.logger.debug(f"Error updating access stats: {e}")
    
    async def _record_cache_hit(self, cache_type: str, provider: str, model: str):
        """Record cache hit for analytics."""
        try:
            if self.analytics:
                await self.analytics._record_metric(
                    f"ai_cache_hit_{cache_type}",
                    1,
                    self.analytics.MetricType.COUNTER,
                    tags={'provider': provider, 'model': model}
                )
        except Exception:
            pass
    
    async def _record_cache_store(self, provider: str, model: str, 
                                 response_size: int, cost: float):
        """Record cache store operation for analytics."""
        try:
            if self.analytics:
                await self.analytics._record_metric(
                    "ai_cache_store_operations",
                    1,
                    self.analytics.MetricType.COUNTER,
                    tags={'provider': provider, 'model': model}
                )
                
                await self.analytics._record_metric(
                    "ai_cache_response_size",
                    response_size,
                    self.analytics.MetricType.HISTOGRAM,
                    tags={'provider': provider, 'model': model}
                )
        except Exception:
            pass
    
    def _generate_prompt_hash(self, prompt: str, provider: str, model: str) -> str:
        """Generate unique hash for prompt/provider/model combination."""
        content = f"{provider}:{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for semantic similarity.
        
        This is a placeholder implementation. In production, you would use
        proper embedding models like sentence-transformers or OpenAI embeddings.
        """
        if not self.enable_embeddings:
            return []
        
        # Simple TF-IDF-style embedding (replace with proper embeddings in production)
        words = text.lower().split()
        vocab_size = 1000
        embedding = [0.0] * vocab_size
        
        for word in words:
            word_hash = hash(word) % vocab_size
            embedding[word_hash] += 1.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    async def warm_cache(self, historical_data: List[Dict[str, Any]]) -> int:
        """Warm cache with historical AI responses.
        
        Args:
            historical_data: List of historical AI interactions
            
        Returns:
            Number of entries added to cache
        """
        added_count = 0
        self.stats.cache_warming_requests += 1
        
        for data in historical_data:
            try:
                prompt = data.get('prompt', '')
                response = data.get('response', '')
                provider = data.get('provider', 'unknown')
                model = data.get('model', 'unknown')
                
                if prompt and response:
                    await self.put(prompt, response, provider, model)
                    added_count += 1
                    
            except Exception as e:
                self.logger.error(f"Error warming cache entry: {e}")
        
        self.logger.info(f"Cache warmed with {added_count} historical entries")
        return added_count
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate cache entries by tags.
        
        Args:
            tags: List of tags to match
            
        Returns:
            Number of entries invalidated
        """
        invalidated = 0
        
        try:
            if self.redis_cache:
                for tag in tags:
                    count = await self.redis_cache.clear_by_tag(tag)
                    invalidated += count
                    self.stats.distributed_invalidations += 1
            
            self.logger.info(f"Invalidated {invalidated} cache entries by tags: {tags}")
            
        except Exception as e:
            self.logger.error(f"Error invalidating by tags: {e}")
        
        return invalidated
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics."""
        base_stats = asdict(self.stats)
        
        # Add migration progress
        base_stats['migration'] = self._migration_progress.copy()
        
        # Add Redis-specific stats
        if self.redis_cache:
            redis_stats = asyncio.create_task(self.redis_cache.get_stats())
            base_stats['redis'] = redis_stats
        
        # Add configuration info
        base_stats['config'] = {
            'migration_mode': self.migration_mode,
            'dual_write_mode': self.dual_write_mode,
            'enable_embeddings': self.enable_embeddings,
            'enable_compression': self.enable_compression,
            'enable_analytics': self.enable_analytics,
            'similarity_threshold': self.similarity_threshold,
            'default_ttl': self.default_ttl
        }
        
        return base_stats
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get detailed migration status."""
        return {
            'progress': self._migration_progress.copy(),
            'mode': self.migration_mode,
            'dual_write_enabled': self.dual_write_mode,
            'redis_available': self.redis_cache is not None,
            'fallback_available': self.fallback_cache is not None,
            'analytics_enabled': self.enable_analytics
        }
    
    async def clear(self):
        """Clear all cache entries from both Redis and fallback."""
        try:
            if self.redis_cache:
                await self.redis_cache.clear_all()
            
            if self.fallback_cache:
                await self.fallback_cache.clear()
            
            self.stats = RedisEnhancedCacheStats()
            self.logger.info("All caches cleared")
            
        except Exception as e:
            self.logger.error(f"Error clearing caches: {e}")
    
    # MCP-Redis Enhanced Methods
    async def find_semantically_similar_responses(self, prompt: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find semantically similar cached responses using MCP-Redis."""
        if not self._use_mcp or not self.mcp_redis:
            # Fallback to basic embedding search if available
            if self.enable_embeddings:
                return await self._find_similar_by_embedding(prompt, limit)
            return []
        
        try:
            similar_responses = await self.mcp_redis.execute(f"""
                Find cached AI responses semantically similar to:
                "{prompt}"
                
                Search criteria:
                - Look for conceptually similar questions
                - Consider different phrasings of the same intent
                - Include responses about related topics
                - Account for synonyms and paraphrasing
                
                Return up to {limit} responses with:
                - Original prompt
                - Cached response
                - Similarity score (0-1)
                - Provider and model used
                - Cache timestamp
                - Explanation of similarity
            """)
            
            return similar_responses if isinstance(similar_responses, list) else []
            
        except Exception as e:
            self.logger.error(f"Error finding similar responses: {e}")
            return []
    
    async def analyze_cache_patterns(self) -> Dict[str, Any]:
        """Analyze cache usage patterns using MCP-Redis."""
        if not self._use_mcp or not self.mcp_redis:
            return {"message": "MCP-Redis not available"}
        
        try:
            analysis = await self.mcp_redis.execute(f"""
                Analyze AI response cache patterns:
                Total cached responses: {self.stats.cache_entries}
                Cache hit rate: {self.stats.hit_rate:.2%}
                
                Analyze:
                - Most frequently accessed prompts
                - Common question patterns
                - Provider/model usage distribution
                - Cache efficiency by prompt type
                - Stale cache entries (old, unused)
                - Redundant entries (near-duplicates)
                - Cost savings from cache hits
                
                Provide:
                - Usage insights
                - Optimization recommendations
                - Cache sizing suggestions
                - TTL optimization per prompt type
            """)
            
            return analysis if isinstance(analysis, dict) else {"analysis": analysis}
            
        except Exception as e:
            self.logger.error(f"Error analyzing cache patterns: {e}")
            return {"error": str(e)}
    
    async def intelligent_cache_invalidation(self, topic: str) -> Dict[str, Any]:
        """Intelligently invalidate related cache entries using MCP-Redis."""
        if not self._use_mcp or not self.mcp_redis:
            # Fallback to tag-based invalidation
            if hasattr(self, 'invalidate_by_tag'):
                await self.invalidate_by_tag(topic)
            return {"invalidated": 0, "method": "tag-based"}
        
        try:
            result = await self.mcp_redis.execute(f"""
                Intelligently invalidate cached responses related to:
                Topic: "{topic}"
                
                Identify and remove:
                - Responses directly about this topic
                - Responses that reference this topic
                - Outdated information about this topic
                - Responses that would be affected by changes to this topic
                
                But keep:
                - General responses that remain valid
                - Historical information that shouldn't change
                
                Return:
                - Number of entries invalidated
                - Categories of invalidated entries
                - Entries kept despite topic match (with reasons)
            """)
            
            return result if isinstance(result, dict) else {"result": result}
            
        except Exception as e:
            self.logger.error(f"Error in intelligent invalidation: {e}")
            return {"error": str(e)}
    
    async def optimize_cache_strategy(self) -> Dict[str, Any]:
        """Optimize caching strategy using MCP-Redis analysis."""
        if not self._use_mcp or not self.mcp_redis:
            return {"message": "MCP-Redis not available"}
        
        try:
            optimization = await self.mcp_redis.execute(f"""
                Optimize AI response cache strategy based on:
                - Current hit rate: {self.stats.hit_rate:.2%}
                - Cache size: {self.stats.cache_entries}
                - Average response time saved: {self.stats.time_saved / max(self.stats.cache_hits, 1):.2f}s
                
                Analyze and recommend:
                - Optimal cache size limits
                - TTL values per prompt category
                - Which responses should always be cached
                - Which responses aren't worth caching
                - Compression settings for large responses
                - Semantic clustering for better retrieval
                - Cost-benefit analysis of current strategy
                
                Provide specific, actionable recommendations.
            """)
            
            return optimization if isinstance(optimization, dict) else {"optimization": optimization}
            
        except Exception as e:
            self.logger.error(f"Error optimizing cache strategy: {e}")
            return {"error": str(e)}
    
    async def predict_cache_value(self, prompt: str, provider: str, model: str) -> Dict[str, Any]:
        """Predict if a prompt is worth caching using MCP-Redis."""
        if not self._use_mcp or not self.mcp_redis:
            # Default: cache everything
            return {"should_cache": True, "confidence": 0.5}
        
        try:
            prediction = await self.mcp_redis.execute(f"""
                Predict if this prompt should be cached:
                Prompt: "{prompt}"
                Provider: {provider}
                Model: {model}
                
                Consider:
                - Likelihood of prompt being repeated
                - Cost of the API call (model pricing)
                - Response size and complexity
                - Prompt specificity vs generality
                - Historical access patterns for similar prompts
                
                Predict:
                - should_cache: boolean
                - cache_value_score: 0-1
                - recommended_ttl: seconds
                - reasoning: explanation
                - expected_reuse_count
            """)
            
            return prediction if isinstance(prediction, dict) else {"should_cache": True}
            
        except Exception as e:
            self.logger.error(f"Error predicting cache value: {e}")
            return {"should_cache": True, "confidence": 0}
    
    async def generate_cache_insights_report(self) -> Dict[str, Any]:
        """Generate comprehensive cache insights using MCP-Redis."""
        if not self._use_mcp or not self.mcp_redis:
            # Return basic stats
            return await self.get_stats()
        
        try:
            report = await self.mcp_redis.execute(f"""
                Generate comprehensive AI cache insights report:
                
                Current metrics:
                - Total requests: {self.stats.total_requests}
                - Hit rate: {self.stats.hit_rate:.2%}
                - Cache entries: {self.stats.cache_entries}
                - Time saved: {self.stats.time_saved:.2f}s
                - Cost saved: ${self.stats.cost_saved:.2f}
                
                Analyze and report on:
                - Cache effectiveness by provider/model
                - Top cached prompts and their value
                - Wasted cache space (unused entries)
                - Semantic clusters in cached content
                - Trends in cache usage over time
                - ROI of caching strategy
                - Recommendations for improvement
                
                Format as executive summary with key metrics and actions.
            """)
            
            return report if isinstance(report, dict) else {"report": report}
            
        except Exception as e:
            self.logger.error(f"Error generating insights report: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown cache and cleanup resources."""
        self.logger.info("Shutting down Redis AI Response Cache")
        self._shutdown = True
        
        # Stop background tasks
        for task in [self._migration_task, self._analytics_task, self._cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Shutdown analytics
        if self.analytics:
            await self.analytics.stop()
        
        # Shutdown fallback cache
        if self.fallback_cache:
            await self.fallback_cache.shutdown()
        
        self.logger.info("Redis AI Response Cache shutdown complete")


# Global cache instance with migration support
_global_redis_cache: Optional[RedisAIResponseCache] = None


async def get_redis_ai_cache(migration_mode: str = "gradual") -> RedisAIResponseCache:
    """Get or create global Redis AI cache instance with migration support."""
    global _global_redis_cache
    
    if _global_redis_cache is None:
        # Import existing cache for migration
        from scripts.ai_response_cache import get_global_cache
        fallback_cache = get_global_cache()
        
        # Create Redis-backed cache
        _global_redis_cache = RedisAIResponseCache(
            fallback_cache=fallback_cache,
            migration_mode=migration_mode,
            dual_write_mode=True
        )
        
        await _global_redis_cache.initialize()
    
    return _global_redis_cache


# Convenience functions with Redis backend
async def cache_ai_response(prompt: str, 
                          response: str, 
                          provider: str, 
                          model: str,
                          cost_estimate: float = 0.0) -> None:
    """Convenience function to cache AI response with Redis backend."""
    cache = await get_redis_ai_cache()
    await cache.put(prompt, response, provider, model, cost_estimate=cost_estimate)


async def get_cached_response(prompt: str, provider: str, model: str) -> Optional[str]:
    """Convenience function to get cached AI response with Redis backend."""
    cache = await get_redis_ai_cache()
    return await cache.get(prompt, provider, model)