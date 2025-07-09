"""
AI Response Cache Module

Intelligent caching system for AI responses with semantic similarity matching.
Provides significant cost savings and latency reduction through smart caching strategies.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from collections import OrderedDict
import numpy as np


@dataclass
class CacheEntry:
    """Represents a cached AI response with metadata."""
    content: str
    embedding: Optional[List[float]]
    timestamp: datetime
    ttl_seconds: int
    provider: str
    model: str
    prompt_hash: str
    usage_count: int = 0
    cost_estimate: float = 0.0


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    semantic_hits: int = 0
    cost_saved: float = 0.0
    time_saved: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits + self.semantic_hits) / self.total_requests
    
    @property
    def exact_hit_rate(self) -> float:
        """Calculate exact match hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests


class AIResponseCache:
    """Intelligent AI response cache with semantic similarity matching."""
    
    def __init__(self, 
                 max_size: int = 10000,
                 default_ttl: int = 3600,
                 similarity_threshold: float = 0.85,
                 enable_embeddings: bool = True):
        """Initialize AI response cache.
        
        Args:
            max_size: Maximum number of cached entries
            default_ttl: Default time-to-live in seconds
            similarity_threshold: Minimum similarity score for semantic matches
            enable_embeddings: Whether to use semantic similarity matching
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.similarity_threshold = similarity_threshold
        self.enable_embeddings = enable_embeddings
        
        # Cache storage (LRU ordered)
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.embedding_cache: Dict[str, List[float]] = {}
        
        # Statistics
        self.stats = CacheStats()
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.AIResponseCache")
        self.logger.info(f"AI Cache initialized: max_size={max_size}, ttl={default_ttl}s, threshold={similarity_threshold}")
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """Periodically clean expired entries."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
    
    def _generate_prompt_hash(self, prompt: str, provider: str, model: str) -> str:
        """Generate unique hash for prompt/provider/model combination."""
        content = f"{provider}:{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate simple embedding for semantic similarity.
        
        Note: This is a simplified embedding. In production, you'd use
        proper embedding models like sentence-transformers or OpenAI embeddings.
        """
        if not self.enable_embeddings:
            return []
        
        # Simple TF-IDF-style embedding (replace with proper embeddings in production)
        words = text.lower().split()
        vocab_size = 1000  # Fixed vocabulary size
        embedding = [0.0] * vocab_size
        
        for word in words:
            # Simple hash-based word to index mapping
            word_hash = hash(word) % vocab_size
            embedding[word_hash] += 1.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        if not embedding1 or not embedding2:
            return 0.0
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def get(self, prompt: str, provider: str, model: str) -> Optional[str]:
        """Get cached response for prompt.
        
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
        
        # Check for exact match first
        if prompt_hash in self.cache:
            entry = self.cache[prompt_hash]
            
            # Check if expired
            if self._is_expired(entry):
                del self.cache[prompt_hash]
                self.logger.debug(f"Cache entry expired: {prompt_hash}")
            else:
                # Move to end (LRU)
                self.cache.move_to_end(prompt_hash)
                entry.usage_count += 1
                
                self.stats.cache_hits += 1
                self.stats.time_saved += time.time() - start_time
                
                self.logger.debug(f"Cache HIT (exact): {prompt_hash}")
                return entry.content
        
        # Check for semantic similarity if enabled
        if self.enable_embeddings:
            prompt_embedding = self._generate_embedding(prompt)
            
            for cached_hash, entry in self.cache.items():
                if entry.provider == provider and entry.model == model and entry.embedding:
                    similarity = self._calculate_similarity(prompt_embedding, entry.embedding)
                    
                    if similarity >= self.similarity_threshold:
                        # Move to end (LRU)
                        self.cache.move_to_end(cached_hash)
                        entry.usage_count += 1
                        
                        self.stats.semantic_hits += 1
                        self.stats.time_saved += time.time() - start_time
                        
                        self.logger.debug(f"Cache HIT (semantic): {cached_hash} (similarity: {similarity:.3f})")
                        return entry.content
        
        self.stats.cache_misses += 1
        self.logger.debug(f"Cache MISS: {prompt_hash}")
        return None
    
    async def put(self, 
                  prompt: str, 
                  response: str, 
                  provider: str, 
                  model: str,
                  ttl_seconds: Optional[int] = None,
                  cost_estimate: float = 0.0) -> None:
        """Store response in cache.
        
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
        
        # Generate embedding for semantic similarity
        embedding = self._generate_embedding(prompt) if self.enable_embeddings else []
        
        # Create cache entry
        entry = CacheEntry(
            content=response,
            embedding=embedding,
            timestamp=datetime.now(timezone.utc),
            ttl_seconds=ttl,
            provider=provider,
            model=model,
            prompt_hash=prompt_hash,
            usage_count=0,
            cost_estimate=cost_estimate
        )
        
        # Store in cache
        self.cache[prompt_hash] = entry
        
        # Enforce size limit (LRU eviction)
        while len(self.cache) > self.max_size:
            oldest_key, oldest_entry = self.cache.popitem(last=False)
            self.logger.debug(f"Cache evicted (LRU): {oldest_key}")
        
        self.logger.debug(f"Cache stored: {prompt_hash} (TTL: {ttl}s)")
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        expiry_time = entry.timestamp + timedelta(seconds=entry.ttl_seconds)
        return datetime.now(timezone.utc) > expiry_time
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        expired_keys = []
        
        for key, entry in self.cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    async def warm_cache(self, historical_data: List[Dict[str, Any]]) -> int:
        """Warm cache with historical AI responses.
        
        Args:
            historical_data: List of historical AI interactions
            
        Returns:
            Number of entries added to cache
        """
        added_count = 0
        
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        return {
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'stats': asdict(self.stats),
            'memory_usage_mb': self._estimate_memory_usage(),
            'enabled_features': {
                'semantic_similarity': self.enable_embeddings,
                'auto_cleanup': self._cleanup_task is not None and not self._cleanup_task.done()
            }
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        # Rough estimation
        total_size = 0
        for entry in self.cache.values():
            total_size += len(entry.content.encode('utf-8'))
            if entry.embedding:
                total_size += len(entry.embedding) * 8  # 8 bytes per float
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    async def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.embedding_cache.clear()
        self.stats = CacheStats()
        self.logger.info("Cache cleared")
    
    async def shutdown(self):
        """Shutdown cache and cleanup resources."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("AI Response Cache shutdown complete")


# Global cache instance
_global_cache: Optional[AIResponseCache] = None


def get_global_cache() -> AIResponseCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = AIResponseCache()
    return _global_cache


async def cache_ai_response(prompt: str, 
                          response: str, 
                          provider: str, 
                          model: str,
                          cost_estimate: float = 0.0) -> None:
    """Convenience function to cache AI response."""
    cache = get_global_cache()
    await cache.put(prompt, response, provider, model, cost_estimate=cost_estimate)


async def get_cached_response(prompt: str, provider: str, model: str) -> Optional[str]:
    """Convenience function to get cached AI response."""
    cache = get_global_cache()
    return await cache.get(prompt, provider, model)