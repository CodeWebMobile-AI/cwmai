"""
Comprehensive Test Suite for CWMAI API Rate Limiting

Tests for Redis-based rate limiting, caching, and API functionality.
"""

import asyncio
import json
import pytest
import redis
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

# Import the modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.rate_limiter import AdvancedRateLimiter, RateLimitRule, RateLimitAlgorithm
from scripts.redis_cache_manager import RedisCacheManager, CacheEntry
from scripts.http_ai_client import HTTPAIClient


class TestAdvancedRateLimiter:
    """Test suite for the advanced rate limiter."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing."""
        mock_redis = Mock()
        mock_redis.pipeline.return_value = mock_redis
        mock_redis.zremrangebyscore.return_value = None
        mock_redis.zcard.return_value = 0
        mock_redis.zadd.return_value = None
        mock_redis.expire.return_value = None
        mock_redis.execute.return_value = [None, 0, None, None]
        mock_redis.get.return_value = None
        mock_redis.incrby.return_value = 1
        mock_redis.keys.return_value = []
        mock_redis.delete.return_value = 1
        mock_redis.hgetall.return_value = {}
        mock_redis.hset.return_value = None
        return mock_redis
    
    @pytest.fixture
    def rate_limiter_with_redis(self, mock_redis):
        """Rate limiter with mocked Redis."""
        return AdvancedRateLimiter(mock_redis)
    
    @pytest.fixture
    def rate_limiter_memory_only(self):
        """Rate limiter with in-memory storage only."""
        return AdvancedRateLimiter(None)
    
    def test_rate_limiter_initialization(self, rate_limiter_with_redis):
        """Test rate limiter initialization."""
        assert rate_limiter_with_redis.redis_client is not None
        assert "api_general" in rate_limiter_with_redis.rules
        assert "api_ai" in rate_limiter_with_redis.rules
        assert "api_tasks" in rate_limiter_with_redis.rules
    
    def test_add_custom_rule(self, rate_limiter_memory_only):
        """Test adding custom rate limiting rules."""
        custom_rule = RateLimitRule("custom", 100, 3600)
        rate_limiter_memory_only.add_rule(custom_rule)
        
        assert "custom" in rate_limiter_memory_only.rules
        assert rate_limiter_memory_only.rules["custom"].limit == 100
        assert rate_limiter_memory_only.rules["custom"].window_seconds == 3600
    
    @pytest.mark.asyncio
    async def test_sliding_window_algorithm_memory(self, rate_limiter_memory_only):
        """Test sliding window algorithm with memory storage."""
        # Test within limit
        result = await rate_limiter_memory_only.check_rate_limit("test_user", "api_general", 1)
        assert result.allowed is True
        assert result.remaining >= 0
        
        # Test multiple requests
        for _ in range(5):
            result = await rate_limiter_memory_only.check_rate_limit("test_user", "api_general", 1)
            assert result.allowed is True
    
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_memory(self, rate_limiter_memory_only):
        """Test rate limit exceeded scenario with memory storage."""
        # Create a strict rule for testing
        strict_rule = RateLimitRule("strict", 3, 60)
        rate_limiter_memory_only.add_rule(strict_rule)
        
        # First 3 requests should be allowed
        for i in range(3):
            result = await rate_limiter_memory_only.check_rate_limit("test_user", "strict", 1)
            assert result.allowed is True
            assert result.remaining == 2 - i
        
        # 4th request should be denied
        result = await rate_limiter_memory_only.check_rate_limit("test_user", "strict", 1)
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after is not None
    
    @pytest.mark.asyncio
    async def test_different_users_separate_limits(self, rate_limiter_memory_only):
        """Test that different users have separate rate limits."""
        strict_rule = RateLimitRule("strict", 2, 60)
        rate_limiter_memory_only.add_rule(strict_rule)
        
        # User 1 reaches limit
        for _ in range(2):
            result = await rate_limiter_memory_only.check_rate_limit("user1", "strict", 1)
            assert result.allowed is True
        
        result = await rate_limiter_memory_only.check_rate_limit("user1", "strict", 1)
        assert result.allowed is False
        
        # User 2 should still be allowed
        result = await rate_limiter_memory_only.check_rate_limit("user2", "strict", 1)
        assert result.allowed is True
    
    @pytest.mark.asyncio
    async def test_token_bucket_algorithm(self, rate_limiter_memory_only):
        """Test token bucket algorithm."""
        token_rule = RateLimitRule("token_test", 10, 60, RateLimitAlgorithm.TOKEN_BUCKET, 20, 10/60)
        rate_limiter_memory_only.add_rule(token_rule)
        
        # Should allow burst initially
        result = await rate_limiter_memory_only.check_rate_limit("test_user", "token_test", 15)
        assert result.allowed is True
        assert result.remaining < 20
    
    @pytest.mark.asyncio
    async def test_rate_limit_info(self, rate_limiter_memory_only):
        """Test getting rate limit information."""
        # Make some requests
        for _ in range(3):
            await rate_limiter_memory_only.check_rate_limit("test_user", "api_general", 1)
        
        info = await rate_limiter_memory_only.get_rate_limit_info("test_user", "api_general")
        
        assert info["rule_name"] == "api_general"
        assert info["algorithm"] == "sliding_window"
        assert info["current_usage"] >= 0
        assert info["limit"] > 0
        assert "percentage_used" in info
    
    @pytest.mark.asyncio
    async def test_reset_rate_limit(self, rate_limiter_memory_only):
        """Test resetting rate limits."""
        # Reach limit
        strict_rule = RateLimitRule("strict", 1, 60)
        rate_limiter_memory_only.add_rule(strict_rule)
        
        result = await rate_limiter_memory_only.check_rate_limit("test_user", "strict", 1)
        assert result.allowed is True
        
        result = await rate_limiter_memory_only.check_rate_limit("test_user", "strict", 1)
        assert result.allowed is False
        
        # Reset and try again
        await rate_limiter_memory_only.reset_rate_limit("test_user", "strict")
        
        result = await rate_limiter_memory_only.check_rate_limit("test_user", "strict", 1)
        assert result.allowed is True
    
    @pytest.mark.asyncio
    async def test_statistics(self, rate_limiter_memory_only):
        """Test getting rate limiter statistics."""
        stats = await rate_limiter_memory_only.get_statistics()
        
        assert "rules" in stats
        assert "storage" in stats
        assert "active_keys" in stats
        assert stats["storage"] == "in_memory"
        assert len(stats["rules"]) > 0


class TestRedisCacheManager:
    """Test suite for the Redis cache manager."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing."""
        mock_redis = Mock()
        mock_redis.get.return_value = None
        mock_redis.setex.return_value = None
        mock_redis.delete.return_value = 1
        mock_redis.exists.return_value = False
        mock_redis.sadd.return_value = None
        mock_redis.expire.return_value = None
        mock_redis.smembers.return_value = set()
        mock_redis.keys.return_value = []
        mock_redis.info.return_value = {
            "redis_version": "6.0.0",
            "connected_clients": 1,
            "used_memory_human": "1M",
            "keyspace_hits": 0,
            "keyspace_misses": 0
        }
        mock_redis.ttl.return_value = 300
        return mock_redis
    
    @pytest.fixture
    def cache_manager_with_redis(self, mock_redis):
        """Cache manager with mocked Redis."""
        return RedisCacheManager(mock_redis, namespace="test")
    
    @pytest.fixture
    def cache_manager_memory_only(self):
        """Cache manager with memory-only storage."""
        return RedisCacheManager(None, namespace="test")
    
    def test_cache_manager_initialization(self, cache_manager_with_redis):
        """Test cache manager initialization."""
        assert cache_manager_with_redis.redis_client is not None
        assert cache_manager_with_redis.namespace == "test"
        assert cache_manager_with_redis.default_ttl == 3600
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get_memory(self, cache_manager_memory_only):
        """Test cache set and get operations with memory storage."""
        # Set a value
        success = await cache_manager_memory_only.set("test_key", "test_value", ttl=300)
        assert success is True
        
        # Get the value
        value = await cache_manager_memory_only.get("test_key")
        assert value == "test_value"
        
        # Get non-existent key
        value = await cache_manager_memory_only.get("non_existent", "default")
        assert value == "default"
    
    @pytest.mark.asyncio
    async def test_cache_expiration_memory(self, cache_manager_memory_only):
        """Test cache expiration with memory storage."""
        # Set a value with short TTL
        await cache_manager_memory_only.set("expire_key", "expire_value", ttl=1)
        
        # Should be available immediately
        value = await cache_manager_memory_only.get("expire_key")
        assert value == "expire_value"
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired
        value = await cache_manager_memory_only.get("expire_key")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_cache_delete_memory(self, cache_manager_memory_only):
        """Test cache deletion with memory storage."""
        # Set and verify
        await cache_manager_memory_only.set("delete_key", "delete_value")
        value = await cache_manager_memory_only.get("delete_key")
        assert value == "delete_value"
        
        # Delete and verify
        deleted = await cache_manager_memory_only.delete("delete_key")
        assert deleted is True
        
        value = await cache_manager_memory_only.get("delete_key")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_cache_exists_memory(self, cache_manager_memory_only):
        """Test cache existence check with memory storage."""
        # Non-existent key
        exists = await cache_manager_memory_only.exists("non_existent")
        assert exists is False
        
        # Set and check
        await cache_manager_memory_only.set("exists_key", "exists_value")
        exists = await cache_manager_memory_only.exists("exists_key")
        assert exists is True
    
    @pytest.mark.asyncio
    async def test_cache_get_or_set_memory(self, cache_manager_memory_only):
        """Test get_or_set functionality with memory storage."""
        def value_generator():
            return "generated_value"
        
        # Should call generator and cache the result
        value = await cache_manager_memory_only.get_or_set("gen_key", value_generator, ttl=300)
        assert value == "generated_value"
        
        # Should return cached value without calling generator again
        def different_generator():
            return "different_value"
        
        value = await cache_manager_memory_only.get_or_set("gen_key", different_generator, ttl=300)
        assert value == "generated_value"  # Should return cached value
    
    @pytest.mark.asyncio
    async def test_cache_clear_by_tags_memory(self, cache_manager_memory_only):
        """Test clearing cache by tags with memory storage."""
        # Set values with tags
        await cache_manager_memory_only.set("tag1_key1", "value1", tags=["tag1", "common"])
        await cache_manager_memory_only.set("tag1_key2", "value2", tags=["tag1"])
        await cache_manager_memory_only.set("tag2_key1", "value3", tags=["tag2", "common"])
        
        # Clear by tag1
        cleared = await cache_manager_memory_only.clear_by_tags(["tag1"])
        assert cleared == 2
        
        # Check what remains
        value1 = await cache_manager_memory_only.get("tag1_key1")
        value2 = await cache_manager_memory_only.get("tag1_key2")
        value3 = await cache_manager_memory_only.get("tag2_key1")
        
        assert value1 is None
        assert value2 is None
        assert value3 == "value3"
    
    @pytest.mark.asyncio
    async def test_cache_clear_all_memory(self, cache_manager_memory_only):
        """Test clearing all cache with memory storage."""
        # Set multiple values
        await cache_manager_memory_only.set("key1", "value1")
        await cache_manager_memory_only.set("key2", "value2")
        await cache_manager_memory_only.set("key3", "value3")
        
        # Clear all
        cleared = await cache_manager_memory_only.clear_all()
        assert cleared is True
        
        # Verify all are gone
        value1 = await cache_manager_memory_only.get("key1")
        value2 = await cache_manager_memory_only.get("key2")
        value3 = await cache_manager_memory_only.get("key3")
        
        assert value1 is None
        assert value2 is None
        assert value3 is None
    
    @pytest.mark.asyncio
    async def test_cache_info(self, cache_manager_memory_only):
        """Test getting cache information."""
        info = await cache_manager_memory_only.get_info()
        
        assert "namespace" in info
        assert "default_ttl" in info
        assert "storage" in info
        assert "statistics" in info
        assert info["namespace"] == "test"
        assert info["storage"]["redis_available"] is False
        assert info["storage"]["memory_cache_size"] >= 0
    
    @pytest.mark.asyncio
    async def test_cache_memory_capacity_management(self, cache_manager_memory_only):
        """Test memory cache capacity management."""
        # Create cache manager with low capacity
        small_cache = RedisCacheManager(None, max_memory_items=5)
        
        # Fill beyond capacity
        for i in range(10):
            await small_cache.set(f"key_{i}", f"value_{i}")
        
        # Should not exceed capacity significantly
        assert len(small_cache.memory_cache) <= 5
    
    def test_cache_entry_functionality(self):
        """Test CacheEntry class functionality."""
        # Test basic creation
        entry = CacheEntry("test_value", ttl=300, tags=["tag1", "tag2"])
        assert entry.value == "test_value"
        assert entry.ttl == 300
        assert entry.tags == ["tag1", "tag2"]
        assert not entry.is_expired()
        
        # Test expiration
        expired_entry = CacheEntry("expired_value", ttl=0)
        time.sleep(0.01)  # Small delay to ensure expiration
        assert expired_entry.is_expired()
        
        # Test touch functionality
        initial_access_count = entry.access_count
        entry.touch()
        assert entry.access_count == initial_access_count + 1
        
        # Test serialization
        entry_dict = entry.to_dict()
        reconstructed = CacheEntry.from_dict(entry_dict)
        assert reconstructed.value == entry.value
        assert reconstructed.ttl == entry.ttl
        assert reconstructed.tags == entry.tags


class TestHTTPAIClient:
    """Test suite for HTTP AI client."""
    
    @pytest.fixture
    def ai_client(self):
        """AI client for testing."""
        return HTTPAIClient()
    
    def test_ai_client_initialization(self, ai_client):
        """Test AI client initialization."""
        assert ai_client.request_count == 0
        assert ai_client.total_response_time == 0.0
        assert ai_client.error_count == 0
        assert isinstance(ai_client.providers_available, dict)
    
    def test_sanitize_headers(self, ai_client):
        """Test header sanitization."""
        headers = {
            "Authorization": "Bearer secret-key",
            "x-api-key": "secret-api-key",
            "Content-Type": "application/json"
        }
        
        sanitized = ai_client._sanitize_headers(headers)
        
        assert sanitized["Authorization"] == "***"
        assert sanitized["x-api-key"] == "***"
        assert sanitized["Content-Type"] == "application/json"
    
    @pytest.mark.asyncio
    async def test_generate_response_no_providers(self):
        """Test response generation with no providers available."""
        # Create client with no API keys
        with patch.dict(os.environ, {}, clear=True):
            client = HTTPAIClient()
            response = await client.generate_enhanced_response("test prompt")
            
            assert response["provider"] == "mock"
            assert "Mock AI response" in response["content"]
            assert response["confidence"] == 0.1
    
    def test_get_research_ai_status(self, ai_client):
        """Test getting research AI status."""
        status = ai_client.get_research_ai_status()
        
        assert "gemini_available" in status
        assert "deepseek_available" in status
        assert "anthropic_primary" in status
        assert "openai_secondary" in status
        assert isinstance(status["gemini_available"], bool)
    
    def test_get_research_capabilities(self, ai_client):
        """Test getting research capabilities."""
        capabilities = ai_client.get_research_capabilities()
        
        assert "available_providers" in capabilities
        assert "research_functions" in capabilities
        assert "analysis_types" in capabilities
        assert "total_providers" in capabilities
        assert "research_ready" in capabilities
        assert isinstance(capabilities["total_providers"], int)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_cache_integration(self):
        """Test integration between rate limiter and cache."""
        # Create components
        rate_limiter = AdvancedRateLimiter(None)
        cache_manager = RedisCacheManager(None)
        
        # Cache rate limit results for performance
        async def cached_rate_limit_check(user_id, rule_name):
            cache_key = f"rate_limit:{user_id}:{rule_name}"
            
            # Try cache first
            cached_result = await cache_manager.get(cache_key)
            if cached_result:
                return cached_result
            
            # Perform rate limit check
            result = await rate_limiter.check_rate_limit(user_id, rule_name)
            
            # Cache the result briefly
            await cache_manager.set(cache_key, result.allowed, ttl=1)
            
            return result.allowed
        
        # Test the integration
        result1 = await cached_rate_limit_check("user1", "api_general")
        result2 = await cached_rate_limit_check("user1", "api_general")  # Should hit cache
        
        assert isinstance(result1, bool)
        assert isinstance(result2, bool)


def run_tests():
    """Run all tests."""
    print("🧪 Running CWMAI API Rate Limiting Tests")
    print("=" * 50)
    
    # Run pytest
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--no-header"
    ])
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    run_tests()