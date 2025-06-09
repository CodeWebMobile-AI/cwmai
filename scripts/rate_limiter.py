"""
Advanced Rate Limiting Module with Redis

Sophisticated rate limiting system with multiple algorithms and Redis persistence.
Supports sliding window, token bucket, and fixed window rate limiting.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, Tuple, List
import redis


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithm types."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitRule:
    """Rate limiting rule configuration."""
    
    def __init__(
        self,
        name: str,
        limit: int,
        window_seconds: int,
        algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW,
        burst_limit: Optional[int] = None,
        refill_rate: Optional[float] = None
    ):
        self.name = name
        self.limit = limit
        self.window_seconds = window_seconds
        self.algorithm = algorithm
        self.burst_limit = burst_limit or limit
        self.refill_rate = refill_rate or (limit / window_seconds)


class RateLimitResult:
    """Result of rate limit check."""
    
    def __init__(
        self,
        allowed: bool,
        remaining: int,
        reset_time: float,
        retry_after: Optional[int] = None,
        rule_name: str = "default"
    ):
        self.allowed = allowed
        self.remaining = remaining
        self.reset_time = reset_time
        self.retry_after = retry_after
        self.rule_name = rule_name
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers."""
        headers = {
            "X-RateLimit-Limit": str(self.remaining + (0 if self.allowed else 1)),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_time)),
            "X-RateLimit-Rule": self.rule_name
        }
        
        if self.retry_after:
            headers["Retry-After"] = str(self.retry_after)
        
        return headers


class AdvancedRateLimiter:
    """Advanced rate limiter with multiple algorithms and Redis persistence."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """Initialize rate limiter."""
        self.redis_client = redis_client
        self.logger = logging.getLogger(f"{__name__}.AdvancedRateLimiter")
        self.in_memory_cache: Dict[str, Any] = {}
        self.rules: Dict[str, RateLimitRule] = {}
        
        # Default rules
        self._setup_default_rules()
        
        # Cleanup task
        self._cleanup_task = None
        if not redis_client:
            self._start_cleanup_task()
    
    def _setup_default_rules(self):
        """Setup default rate limiting rules."""
        self.rules.update({
            "api_general": RateLimitRule("api_general", 60, 60),  # 60/minute
            "api_ai": RateLimitRule("api_ai", 10, 60),  # 10/minute for AI
            "api_tasks": RateLimitRule("api_tasks", 20, 60),  # 20/minute for tasks
            "api_websocket": RateLimitRule("api_websocket", 100, 60),  # 100/minute for WS
            "api_burst": RateLimitRule("api_burst", 10, 1, RateLimitAlgorithm.TOKEN_BUCKET, 20)  # Burst handling
        })
    
    def add_rule(self, rule: RateLimitRule):
        """Add a custom rate limiting rule."""
        self.rules[rule.name] = rule
        self.logger.info(f"Added rate limit rule: {rule.name} - {rule.limit}/{rule.window_seconds}s")
    
    async def check_rate_limit(
        self,
        identifier: str,
        rule_name: str = "api_general",
        increment: int = 1
    ) -> RateLimitResult:
        """Check rate limit for identifier using specified rule."""
        rule = self.rules.get(rule_name)
        if not rule:
            self.logger.warning(f"Rate limit rule '{rule_name}' not found, using default")
            rule = self.rules["api_general"]
        
        # Choose algorithm
        if rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return await self._sliding_window_check(identifier, rule, increment)
        elif rule.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return await self._fixed_window_check(identifier, rule, increment)
        elif rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return await self._token_bucket_check(identifier, rule, increment)
        else:
            return await self._sliding_window_check(identifier, rule, increment)
    
    async def _sliding_window_check(
        self,
        identifier: str,
        rule: RateLimitRule,
        increment: int
    ) -> RateLimitResult:
        """Sliding window rate limiting algorithm."""
        now = time.time()
        window_start = now - rule.window_seconds
        key = f"rl:sw:{rule.name}:{identifier}"
        
        if self.redis_client:
            # Redis-based sliding window
            pipe = self.redis_client.pipeline()
            try:
                # Remove old entries
                pipe.zremrangebyscore(key, 0, window_start)
                # Count current requests in window
                pipe.zcard(key)
                # Add current request with timestamp
                for _ in range(increment):
                    pipe.zadd(key, {str(now + _ * 0.001): now + _ * 0.001})
                # Set expiry
                pipe.expire(key, rule.window_seconds + 1)
                
                results = pipe.execute()
                current_count = results[1] + increment
                
            except Exception as e:
                self.logger.error(f"Redis sliding window error: {e}")
                return RateLimitResult(True, rule.limit - 1, now + rule.window_seconds, rule_name=rule.name)
        else:
            # In-memory sliding window
            if key not in self.in_memory_cache:
                self.in_memory_cache[key] = []
            
            # Remove old entries
            self.in_memory_cache[key] = [
                entry for entry in self.in_memory_cache[key]
                if entry > window_start
            ]
            
            # Add new entries
            for _ in range(increment):
                self.in_memory_cache[key].append(now)
            
            current_count = len(self.in_memory_cache[key])
        
        # Check limit
        allowed = current_count <= rule.limit
        remaining = max(0, rule.limit - current_count)
        reset_time = now + rule.window_seconds
        retry_after = None if allowed else int(rule.window_seconds)
        
        return RateLimitResult(allowed, remaining, reset_time, retry_after, rule.name)
    
    async def _fixed_window_check(
        self,
        identifier: str,
        rule: RateLimitRule,
        increment: int
    ) -> RateLimitResult:
        """Fixed window rate limiting algorithm."""
        now = time.time()
        window = int(now // rule.window_seconds)
        key = f"rl:fw:{rule.name}:{identifier}:{window}"
        
        if self.redis_client:
            try:
                current = self.redis_client.get(key)
                current_count = int(current) if current else 0
                
                if current_count + increment <= rule.limit:
                    new_count = self.redis_client.incrby(key, increment)
                    self.redis_client.expire(key, rule.window_seconds)
                    allowed = True
                    remaining = rule.limit - new_count
                else:
                    allowed = False
                    remaining = 0
                    
            except Exception as e:
                self.logger.error(f"Redis fixed window error: {e}")
                return RateLimitResult(True, rule.limit - 1, now + rule.window_seconds, rule_name=rule.name)
        else:
            if key not in self.in_memory_cache:
                self.in_memory_cache[key] = 0
            
            if self.in_memory_cache[key] + increment <= rule.limit:
                self.in_memory_cache[key] += increment
                allowed = True
                remaining = rule.limit - self.in_memory_cache[key]
            else:
                allowed = False
                remaining = 0
        
        reset_time = (window + 1) * rule.window_seconds
        retry_after = None if allowed else int(reset_time - now)
        
        return RateLimitResult(allowed, remaining, reset_time, retry_after, rule.name)
    
    async def _token_bucket_check(
        self,
        identifier: str,
        rule: RateLimitRule,
        increment: int
    ) -> RateLimitResult:
        """Token bucket rate limiting algorithm."""
        now = time.time()
        key = f"rl:tb:{rule.name}:{identifier}"
        
        if self.redis_client:
            try:
                # Get bucket state
                bucket_data = self.redis_client.hgetall(key)
                
                if bucket_data:
                    tokens = float(bucket_data.get('tokens', rule.burst_limit))
                    last_refill = float(bucket_data.get('last_refill', now))
                else:
                    tokens = rule.burst_limit
                    last_refill = now
                
                # Refill tokens
                time_passed = now - last_refill
                tokens_to_add = time_passed * rule.refill_rate
                tokens = min(rule.burst_limit, tokens + tokens_to_add)
                
                # Check if request can be served
                if tokens >= increment:
                    tokens -= increment
                    allowed = True
                else:
                    allowed = False
                
                # Update bucket state
                self.redis_client.hset(key, mapping={
                    'tokens': tokens,
                    'last_refill': now
                })
                self.redis_client.expire(key, rule.window_seconds * 2)
                
                remaining = int(tokens)
                
            except Exception as e:
                self.logger.error(f"Redis token bucket error: {e}")
                return RateLimitResult(True, rule.limit - 1, now + rule.window_seconds, rule_name=rule.name)
        else:
            if key not in self.in_memory_cache:
                self.in_memory_cache[key] = {
                    'tokens': rule.burst_limit,
                    'last_refill': now
                }
            
            bucket = self.in_memory_cache[key]
            
            # Refill tokens
            time_passed = now - bucket['last_refill']
            tokens_to_add = time_passed * rule.refill_rate
            bucket['tokens'] = min(rule.burst_limit, bucket['tokens'] + tokens_to_add)
            bucket['last_refill'] = now
            
            # Check if request can be served
            if bucket['tokens'] >= increment:
                bucket['tokens'] -= increment
                allowed = True
            else:
                allowed = False
            
            remaining = int(bucket['tokens'])
        
        reset_time = now + (rule.burst_limit - remaining) / rule.refill_rate
        retry_after = None if allowed else int((increment - remaining) / rule.refill_rate)
        
        return RateLimitResult(allowed, remaining, reset_time, retry_after, rule.name)
    
    async def get_rate_limit_info(self, identifier: str, rule_name: str = "api_general") -> Dict[str, Any]:
        """Get current rate limit information for identifier."""
        rule = self.rules.get(rule_name, self.rules["api_general"])
        now = time.time()
        
        # Get current usage without incrementing
        if rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            window_start = now - rule.window_seconds
            key = f"rl:sw:{rule.name}:{identifier}"
            
            if self.redis_client:
                current_count = self.redis_client.zcount(key, window_start, now)
            else:
                current_count = len([
                    entry for entry in self.in_memory_cache.get(key, [])
                    if entry > window_start
                ])
        else:
            # For other algorithms, do a zero-increment check
            result = await self.check_rate_limit(identifier, rule_name, 0)
            current_count = rule.limit - result.remaining
        
        return {
            "rule_name": rule.name,
            "algorithm": rule.algorithm.value,
            "limit": rule.limit,
            "window_seconds": rule.window_seconds,
            "current_usage": current_count,
            "remaining": rule.limit - current_count,
            "reset_time": now + rule.window_seconds,
            "percentage_used": (current_count / rule.limit) * 100
        }
    
    async def reset_rate_limit(self, identifier: str, rule_name: str = "api_general"):
        """Reset rate limit for identifier."""
        rule = self.rules.get(rule_name, self.rules["api_general"])
        
        if self.redis_client:
            patterns = [
                f"rl:sw:{rule.name}:{identifier}",
                f"rl:fw:{rule.name}:{identifier}:*",
                f"rl:tb:{rule.name}:{identifier}"
            ]
            
            for pattern in patterns:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
        else:
            # Remove from in-memory cache
            keys_to_remove = [
                key for key in self.in_memory_cache.keys()
                if identifier in key and rule.name in key
            ]
            for key in keys_to_remove:
                del self.in_memory_cache[key]
        
        self.logger.info(f"Reset rate limit for {identifier} (rule: {rule_name})")
    
    def _start_cleanup_task(self):
        """Start background cleanup task for in-memory cache."""
        async def cleanup():
            while True:
                try:
                    await asyncio.sleep(60)  # Cleanup every minute
                    now = time.time()
                    
                    keys_to_remove = []
                    for key, value in self.in_memory_cache.items():
                        if isinstance(value, list):
                            # Sliding window cleanup
                            if key.startswith("rl:sw:"):
                                window_seconds = 3600  # Default cleanup window
                                self.in_memory_cache[key] = [
                                    entry for entry in value
                                    if entry > now - window_seconds
                                ]
                                if not self.in_memory_cache[key]:
                                    keys_to_remove.append(key)
                        elif isinstance(value, dict) and 'last_refill' in value:
                            # Token bucket cleanup
                            if now - value['last_refill'] > 3600:  # 1 hour timeout
                                keys_to_remove.append(key)
                    
                    for key in keys_to_remove:
                        del self.in_memory_cache[key]
                        
                except Exception as e:
                    self.logger.error(f"Cleanup task error: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup())
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        stats = {
            "rules": {name: {
                "name": rule.name,
                "limit": rule.limit,
                "window_seconds": rule.window_seconds,
                "algorithm": rule.algorithm.value
            } for name, rule in self.rules.items()},
            "storage": "redis" if self.redis_client else "in_memory",
            "active_keys": 0
        }
        
        if self.redis_client:
            try:
                # Count active rate limit keys
                patterns = ["rl:*"]
                total_keys = 0
                for pattern in patterns:
                    keys = self.redis_client.keys(pattern)
                    total_keys += len(keys)
                stats["active_keys"] = total_keys
            except Exception as e:
                self.logger.error(f"Error getting Redis statistics: {e}")
        else:
            stats["active_keys"] = len(self.in_memory_cache)
        
        return stats