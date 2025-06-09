"""
Rate Limiter Module

Sophisticated Redis-based rate limiting for API requests with real-time monitoring,
adaptive thresholds, and comprehensive admin dashboard capabilities.
"""

import json
import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import redis
from dataclasses import dataclass, asdict


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    ADAPTIVE = "adaptive"


class RateLimitTier(Enum):
    """Rate limit tiers for different user types."""
    BASIC = "basic"
    PREMIUM = "premium"
    ADMIN = "admin"
    SYSTEM = "system"


@dataclass
class RateLimitRule:
    """Rate limit rule configuration."""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_allowance: int
    strategy: RateLimitStrategy
    tier: RateLimitTier


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining_requests: int
    reset_time: datetime
    retry_after: Optional[int]
    current_usage: int
    tier: str
    strategy: str


class RateLimiter:
    """Redis-based rate limiter with real-time monitoring and adaptive capabilities."""
    
    # Default rate limit configurations
    DEFAULT_RULES = {
        RateLimitTier.BASIC: RateLimitRule(
            requests_per_minute=10,
            requests_per_hour=300,
            requests_per_day=1000,
            burst_allowance=5,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            tier=RateLimitTier.BASIC
        ),
        RateLimitTier.PREMIUM: RateLimitRule(
            requests_per_minute=30,
            requests_per_hour=1000,
            requests_per_day=5000,
            burst_allowance=15,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            tier=RateLimitTier.PREMIUM
        ),
        RateLimitTier.ADMIN: RateLimitRule(
            requests_per_minute=100,
            requests_per_hour=5000,
            requests_per_day=20000,
            burst_allowance=50,
            strategy=RateLimitStrategy.ADAPTIVE,
            tier=RateLimitTier.ADMIN
        ),
        RateLimitTier.SYSTEM: RateLimitRule(
            requests_per_minute=1000,
            requests_per_hour=50000,
            requests_per_day=100000,
            burst_allowance=200,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            tier=RateLimitTier.SYSTEM
        )
    }
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize the rate limiter.
        
        Args:
            redis_url: Redis connection URL (optional, uses environment variable if not provided)
        """
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        
        # Initialize Redis connection with connection pooling
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=20
            )
            # Test connection
            self.redis_client.ping()
            self.redis_available = True
        except Exception as e:
            logging.warning(f"Redis connection failed: {e}. Rate limiting will use fallback mode.")
            self.redis_client = None
            self.redis_available = False
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.RateLimiter")
        
        # Rate limit rules
        self.rules = self.DEFAULT_RULES.copy()
        
        # Metrics tracking
        self.metrics = {
            "total_requests": 0,
            "blocked_requests": 0,
            "allowed_requests": 0,
            "errors": 0,
            "start_time": datetime.now(timezone.utc).isoformat()
        }
        
        # Fallback storage for when Redis is unavailable
        self.fallback_storage = {}
        
        self.logger.info(f"RateLimiter initialized - Redis available: {self.redis_available}")
    
    def check_rate_limit(self, client_id: str, tier: RateLimitTier = RateLimitTier.BASIC, 
                        endpoint: str = "default") -> RateLimitResult:
        """Check if request is allowed under rate limits.
        
        Args:
            client_id: Unique identifier for the client
            tier: Rate limit tier for the client
            endpoint: API endpoint being accessed
            
        Returns:
            RateLimitResult object with decision and metadata
        """
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            rule = self.rules.get(tier, self.DEFAULT_RULES[RateLimitTier.BASIC])
            
            # Create composite key
            key_base = f"rate_limit:{client_id}:{tier.value}:{endpoint}"
            
            # Check rate limit based on strategy
            if rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
                result = self._check_token_bucket(key_base, rule)
            elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
                result = self._check_sliding_window(key_base, rule)
            elif rule.strategy == RateLimitStrategy.FIXED_WINDOW:
                result = self._check_fixed_window(key_base, rule)
            elif rule.strategy == RateLimitStrategy.ADAPTIVE:
                result = self._check_adaptive_limit(key_base, rule, client_id)
            else:
                # Default to sliding window
                result = self._check_sliding_window(key_base, rule)
            
            # Update metrics
            if result.allowed:
                self.metrics["allowed_requests"] += 1
            else:
                self.metrics["blocked_requests"] += 1
            
            # Log real-time activity for monitoring
            self._log_activity(client_id, tier, endpoint, result, time.time() - start_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            self.metrics["errors"] += 1
            
            # Fail open - allow request if rate limiter has errors
            return RateLimitResult(
                allowed=True,
                remaining_requests=1000,
                reset_time=datetime.now(timezone.utc) + timedelta(minutes=1),
                retry_after=None,
                current_usage=0,
                tier=tier.value,
                strategy="error_fallback"
            )
    
    def _check_token_bucket(self, key_base: str, rule: RateLimitRule) -> RateLimitResult:
        """Token bucket rate limiting implementation."""
        key = f"{key_base}:bucket"
        
        if self.redis_available:
            # Use Redis-based token bucket
            return self._redis_token_bucket(key, rule)
        else:
            # Use fallback implementation
            return self._fallback_token_bucket(key, rule)
    
    def _redis_token_bucket(self, key: str, rule: RateLimitRule) -> RateLimitResult:
        """Redis-based token bucket implementation."""
        now = time.time()
        bucket_key = f"{key}:tokens"
        last_refill_key = f"{key}:last_refill"
        
        # Token bucket parameters
        capacity = rule.requests_per_minute + rule.burst_allowance
        refill_rate = rule.requests_per_minute / 60.0  # tokens per second
        
        try:
            # Get current bucket state
            pipe = self.redis_client.pipeline()
            pipe.get(bucket_key)
            pipe.get(last_refill_key)
            tokens, last_refill = pipe.execute()
            
            # Initialize if first time
            if tokens is None:
                tokens = capacity
                last_refill = now
            else:
                tokens = float(tokens)
                last_refill = float(last_refill) if last_refill else now
            
            # Calculate tokens to add
            time_passed = now - last_refill
            tokens_to_add = time_passed * refill_rate
            tokens = min(capacity, tokens + tokens_to_add)
            
            # Check if request can be allowed
            if tokens >= 1:
                tokens -= 1
                allowed = True
                
                # Update bucket state
                pipe = self.redis_client.pipeline()
                pipe.setex(bucket_key, 3600, str(tokens))  # 1 hour TTL
                pipe.setex(last_refill_key, 3600, str(now))
                pipe.execute()
            else:
                allowed = False
            
            # Calculate reset time
            if tokens == 0:
                reset_time = datetime.fromtimestamp(now + (1 / refill_rate), timezone.utc)
                retry_after = int(1 / refill_rate) + 1
            else:
                reset_time = datetime.fromtimestamp(now + ((1 - tokens) / refill_rate), timezone.utc)
                retry_after = None
            
            return RateLimitResult(
                allowed=allowed,
                remaining_requests=int(tokens),
                reset_time=reset_time,
                retry_after=retry_after,
                current_usage=capacity - int(tokens),
                tier=rule.tier.value,
                strategy=rule.strategy.value
            )
            
        except Exception as e:
            self.logger.error(f"Redis token bucket failed: {e}")
            # Fallback to allowing request
            return RateLimitResult(
                allowed=True,
                remaining_requests=rule.requests_per_minute,
                reset_time=datetime.now(timezone.utc) + timedelta(minutes=1),
                retry_after=None,
                current_usage=0,
                tier=rule.tier.value,
                strategy="redis_error_fallback"
            )
    
    def _check_sliding_window(self, key_base: str, rule: RateLimitRule) -> RateLimitResult:
        """Sliding window rate limiting implementation."""
        if self.redis_available:
            return self._redis_sliding_window(key_base, rule)
        else:
            return self._fallback_sliding_window(key_base, rule)
    
    def _redis_sliding_window(self, key_base: str, rule: RateLimitRule) -> RateLimitResult:
        """Redis-based sliding window implementation."""
        now = time.time()
        minute_key = f"{key_base}:minute"
        hour_key = f"{key_base}:hour"
        day_key = f"{key_base}:day"
        
        try:
            # Use sorted sets to track request timestamps
            pipe = self.redis_client.pipeline()
            
            # Remove old entries and count current
            minute_ago = now - 60
            hour_ago = now - 3600
            day_ago = now - 86400
            
            # Clean old entries and add current request
            pipe.zremrangebyscore(minute_key, 0, minute_ago)
            pipe.zremrangebyscore(hour_key, 0, hour_ago)
            pipe.zremrangebyscore(day_key, 0, day_ago)
            
            # Count current requests
            pipe.zcard(minute_key)
            pipe.zcard(hour_key)
            pipe.zcard(day_key)
            
            results = pipe.execute()
            minute_count = results[3]
            hour_count = results[4]
            day_count = results[5]
            
            # Check limits
            allowed = (
                minute_count < rule.requests_per_minute and
                hour_count < rule.requests_per_hour and
                day_count < rule.requests_per_day
            )
            
            if allowed:
                # Add current request
                pipe = self.redis_client.pipeline()
                pipe.zadd(minute_key, {str(now): now})
                pipe.zadd(hour_key, {str(now): now})
                pipe.zadd(day_key, {str(now): now})
                pipe.expire(minute_key, 120)  # 2 minutes TTL
                pipe.expire(hour_key, 7200)  # 2 hours TTL
                pipe.expire(day_key, 172800)  # 2 days TTL
                pipe.execute()
            
            # Calculate reset time and retry after
            reset_time = datetime.fromtimestamp(now + 60, timezone.utc)
            retry_after = 60 if not allowed else None
            
            # Calculate remaining requests (use most restrictive limit)
            remaining_minute = max(0, rule.requests_per_minute - minute_count)
            remaining_hour = max(0, rule.requests_per_hour - hour_count)
            remaining_day = max(0, rule.requests_per_day - day_count)
            remaining = min(remaining_minute, remaining_hour, remaining_day)
            
            return RateLimitResult(
                allowed=allowed,
                remaining_requests=remaining,
                reset_time=reset_time,
                retry_after=retry_after,
                current_usage=max(minute_count, hour_count / 60, day_count / 1440),
                tier=rule.tier.value,
                strategy=rule.strategy.value
            )
            
        except Exception as e:
            self.logger.error(f"Redis sliding window failed: {e}")
            return RateLimitResult(
                allowed=True,
                remaining_requests=rule.requests_per_minute,
                reset_time=datetime.now(timezone.utc) + timedelta(minutes=1),
                retry_after=None,
                current_usage=0,
                tier=rule.tier.value,
                strategy="redis_error_fallback"
            )
    
    def _check_fixed_window(self, key_base: str, rule: RateLimitRule) -> RateLimitResult:
        """Fixed window rate limiting implementation."""
        now = time.time()
        window_start = int(now / 60) * 60  # 1-minute windows
        key = f"{key_base}:window:{window_start}"
        
        try:
            if self.redis_available:
                # Increment counter for this window
                current_count = self.redis_client.incr(key)
                if current_count == 1:
                    # Set expiry for the key
                    self.redis_client.expire(key, 120)  # 2 minutes TTL
            else:
                # Fallback implementation
                if key not in self.fallback_storage:
                    self.fallback_storage[key] = {"count": 0, "expires": window_start + 120}
                
                # Clean expired entries
                self.fallback_storage = {
                    k: v for k, v in self.fallback_storage.items() 
                    if v["expires"] > now
                }
                
                self.fallback_storage[key]["count"] += 1
                current_count = self.fallback_storage[key]["count"]
            
            allowed = current_count <= rule.requests_per_minute
            reset_time = datetime.fromtimestamp(window_start + 60, timezone.utc)
            retry_after = int(window_start + 60 - now) if not allowed else None
            
            return RateLimitResult(
                allowed=allowed,
                remaining_requests=max(0, rule.requests_per_minute - current_count),
                reset_time=reset_time,
                retry_after=retry_after,
                current_usage=current_count,
                tier=rule.tier.value,
                strategy=rule.strategy.value
            )
            
        except Exception as e:
            self.logger.error(f"Fixed window check failed: {e}")
            return RateLimitResult(
                allowed=True,
                remaining_requests=rule.requests_per_minute,
                reset_time=datetime.now(timezone.utc) + timedelta(minutes=1),
                retry_after=None,
                current_usage=0,
                tier=rule.tier.value,
                strategy="error_fallback"
            )
    
    def _check_adaptive_limit(self, key_base: str, rule: RateLimitRule, client_id: str) -> RateLimitResult:
        """Adaptive rate limiting based on client behavior and system load."""
        # Get client history and system metrics
        history_key = f"{key_base}:history"
        system_load_key = "system:load"
        
        try:
            if self.redis_available:
                # Get client behavior metrics
                client_history = self.redis_client.get(history_key)
                system_load = self.redis_client.get(system_load_key) or "0.5"
                
                if client_history:
                    history = json.loads(client_history)
                else:
                    history = {"success_rate": 1.0, "avg_response_time": 100, "error_rate": 0.0}
                
                system_load = float(system_load)
            else:
                # Fallback - use default behavior
                history = {"success_rate": 1.0, "avg_response_time": 100, "error_rate": 0.0}
                system_load = 0.5
            
            # Calculate adaptive multiplier based on:
            # 1. Client success rate (good clients get higher limits)
            # 2. System load (reduce limits under high load)
            # 3. Error rate (reduce limits for error-prone clients)
            
            success_multiplier = min(2.0, history["success_rate"] * 1.5)
            load_multiplier = max(0.3, 1.5 - system_load)
            error_multiplier = max(0.5, 1.0 - history["error_rate"])
            
            adaptive_multiplier = success_multiplier * load_multiplier * error_multiplier
            
            # Create adaptive rule
            adaptive_rule = RateLimitRule(
                requests_per_minute=int(rule.requests_per_minute * adaptive_multiplier),
                requests_per_hour=int(rule.requests_per_hour * adaptive_multiplier),
                requests_per_day=int(rule.requests_per_day * adaptive_multiplier),
                burst_allowance=int(rule.burst_allowance * adaptive_multiplier),
                strategy=RateLimitStrategy.SLIDING_WINDOW,  # Use sliding window for adaptive
                tier=rule.tier
            )
            
            # Use sliding window with adaptive limits
            result = self._check_sliding_window(key_base, adaptive_rule)
            result.strategy = "adaptive"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Adaptive rate limiting failed: {e}")
            # Fallback to standard sliding window
            return self._check_sliding_window(key_base, rule)
    
    def _fallback_token_bucket(self, key: str, rule: RateLimitRule) -> RateLimitResult:
        """Fallback token bucket implementation without Redis."""
        now = time.time()
        
        if key not in self.fallback_storage:
            self.fallback_storage[key] = {
                "tokens": rule.requests_per_minute + rule.burst_allowance,
                "last_refill": now
            }
        
        bucket = self.fallback_storage[key]
        
        # Clean old entries
        self.fallback_storage = {
            k: v for k, v in self.fallback_storage.items()
            if now - v.get("last_refill", 0) < 3600  # Keep for 1 hour
        }
        
        # Refill tokens
        time_passed = now - bucket["last_refill"]
        refill_rate = rule.requests_per_minute / 60.0
        capacity = rule.requests_per_minute + rule.burst_allowance
        
        bucket["tokens"] = min(capacity, bucket["tokens"] + time_passed * refill_rate)
        bucket["last_refill"] = now
        
        # Check if request allowed
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            allowed = True
        else:
            allowed = False
        
        reset_time = datetime.fromtimestamp(now + 60, timezone.utc)
        retry_after = 60 if not allowed else None
        
        return RateLimitResult(
            allowed=allowed,
            remaining_requests=int(bucket["tokens"]),
            reset_time=reset_time,
            retry_after=retry_after,
            current_usage=capacity - int(bucket["tokens"]),
            tier=rule.tier.value,
            strategy="fallback_" + rule.strategy.value
        )
    
    def _fallback_sliding_window(self, key_base: str, rule: RateLimitRule) -> RateLimitResult:
        """Fallback sliding window implementation without Redis."""
        now = time.time()
        key = f"{key_base}:requests"
        
        if key not in self.fallback_storage:
            self.fallback_storage[key] = []
        
        requests = self.fallback_storage[key]
        
        # Remove old requests
        minute_ago = now - 60
        requests = [req_time for req_time in requests if req_time > minute_ago]
        self.fallback_storage[key] = requests
        
        # Check limit
        if len(requests) < rule.requests_per_minute:
            requests.append(now)
            allowed = True
        else:
            allowed = False
        
        return RateLimitResult(
            allowed=allowed,
            remaining_requests=max(0, rule.requests_per_minute - len(requests)),
            reset_time=datetime.fromtimestamp(now + 60, timezone.utc),
            retry_after=60 if not allowed else None,
            current_usage=len(requests),
            tier=rule.tier.value,
            strategy="fallback_" + rule.strategy.value
        )
    
    def _log_activity(self, client_id: str, tier: RateLimitTier, endpoint: str, 
                     result: RateLimitResult, duration: float) -> None:
        """Log activity for real-time monitoring."""
        activity = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "client_id": client_id,
            "tier": tier.value,
            "endpoint": endpoint,
            "allowed": result.allowed,
            "remaining": result.remaining_requests,
            "usage": result.current_usage,
            "strategy": result.strategy,
            "duration_ms": round(duration * 1000, 2)
        }
        
        # Store recent activity for monitoring
        if self.redis_available:
            try:
                activity_key = "rate_limiter:activity"
                self.redis_client.lpush(activity_key, json.dumps(activity))
                self.redis_client.ltrim(activity_key, 0, 999)  # Keep last 1000 activities
                self.redis_client.expire(activity_key, 3600)  # 1 hour TTL
            except Exception as e:
                self.logger.error(f"Failed to log activity: {e}")
        
        # Log for admin monitoring
        if not result.allowed:
            self.logger.warning(f"Rate limit exceeded - Client: {client_id}, Tier: {tier.value}, Endpoint: {endpoint}")
        else:
            self.logger.debug(f"Rate limit check - Client: {client_id}, Remaining: {result.remaining_requests}")
    
    def get_client_stats(self, client_id: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific client."""
        stats = {
            "client_id": client_id,
            "current_time": datetime.now(timezone.utc).isoformat(),
            "rate_limits": {},
            "recent_activity": []
        }
        
        try:
            if self.redis_available:
                # Get rate limit status for each tier
                for tier in RateLimitTier:
                    key_base = f"rate_limit:{client_id}:{tier.value}:default"
                    rule = self.rules.get(tier, self.DEFAULT_RULES[tier])
                    
                    if rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
                        # Get current usage from sliding window
                        minute_key = f"{key_base}:minute"
                        hour_key = f"{key_base}:hour"
                        day_key = f"{key_base}:day"
                        
                        now = time.time()
                        minute_ago = now - 60
                        hour_ago = now - 3600
                        day_ago = now - 86400
                        
                        pipe = self.redis_client.pipeline()
                        pipe.zcount(minute_key, minute_ago, now)
                        pipe.zcount(hour_key, hour_ago, now)
                        pipe.zcount(day_key, day_ago, now)
                        
                        results = pipe.execute()
                        
                        stats["rate_limits"][tier.value] = {
                            "requests_per_minute": f"{results[0]}/{rule.requests_per_minute}",
                            "requests_per_hour": f"{results[1]}/{rule.requests_per_hour}",
                            "requests_per_day": f"{results[2]}/{rule.requests_per_day}",
                            "strategy": rule.strategy.value
                        }
                
                # Get recent activity
                activity_key = "rate_limiter:activity"
                recent_activities = self.redis_client.lrange(activity_key, 0, 19)  # Last 20
                
                for activity_json in recent_activities:
                    activity = json.loads(activity_json)
                    if activity["client_id"] == client_id:
                        stats["recent_activity"].append(activity)
            
        except Exception as e:
            self.logger.error(f"Failed to get client stats: {e}")
            stats["error"] = str(e)
        
        return stats
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide rate limiting metrics."""
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "redis_available": self.redis_available,
            "total_requests": self.metrics["total_requests"],
            "allowed_requests": self.metrics["allowed_requests"],
            "blocked_requests": self.metrics["blocked_requests"],
            "error_count": self.metrics["errors"],
            "block_rate": 0.0,
            "uptime_seconds": 0,
            "active_clients": 0,
            "recent_activity": []
        }
        
        try:
            # Calculate metrics
            if self.metrics["total_requests"] > 0:
                metrics["block_rate"] = self.metrics["blocked_requests"] / self.metrics["total_requests"]
            
            start_time = datetime.fromisoformat(self.metrics["start_time"].replace("Z", "+00:00"))
            metrics["uptime_seconds"] = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            if self.redis_available:
                # Get active clients and recent activity
                activity_key = "rate_limiter:activity"
                recent_activities = self.redis_client.lrange(activity_key, 0, 99)  # Last 100
                
                active_clients = set()
                activity_list = []
                
                for activity_json in recent_activities:
                    activity = json.loads(activity_json)
                    active_clients.add(activity["client_id"])
                    activity_list.append(activity)
                
                metrics["active_clients"] = len(active_clients)
                metrics["recent_activity"] = activity_list[:20]  # Return last 20
            
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    def update_client_tier(self, client_id: str, new_tier: RateLimitTier) -> bool:
        """Update a client's rate limit tier."""
        try:
            if self.redis_available:
                tier_key = f"client_tier:{client_id}"
                self.redis_client.setex(tier_key, 86400, new_tier.value)  # 24 hours TTL
            else:
                # Fallback storage
                self.fallback_storage[f"tier:{client_id}"] = new_tier.value
            
            self.logger.info(f"Updated client {client_id} tier to {new_tier.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update client tier: {e}")
            return False
    
    def get_client_tier(self, client_id: str) -> RateLimitTier:
        """Get a client's current rate limit tier."""
        try:
            if self.redis_available:
                tier_key = f"client_tier:{client_id}"
                tier_value = self.redis_client.get(tier_key)
                if tier_value:
                    return RateLimitTier(tier_value)
            else:
                # Check fallback storage
                tier_value = self.fallback_storage.get(f"tier:{client_id}")
                if tier_value:
                    return RateLimitTier(tier_value)
            
        except Exception as e:
            self.logger.error(f"Failed to get client tier: {e}")
        
        # Default to basic tier
        return RateLimitTier.BASIC
    
    def reset_client_limits(self, client_id: str) -> bool:
        """Reset rate limits for a specific client."""
        try:
            if self.redis_available:
                # Get all keys for this client
                pattern = f"rate_limit:{client_id}:*"
                keys = self.redis_client.keys(pattern)
                
                if keys:
                    self.redis_client.delete(*keys)
                
                self.logger.info(f"Reset rate limits for client {client_id}")
                return True
            else:
                # Clear from fallback storage
                keys_to_remove = [k for k in self.fallback_storage.keys() if client_id in k]
                for key in keys_to_remove:
                    del self.fallback_storage[key]
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to reset client limits: {e}")
            return False
    
    def close(self) -> None:
        """Close Redis connection."""
        try:
            if self.redis_client:
                self.redis_client.close()
                self.logger.info("Redis connection closed")
        except Exception as e:
            self.logger.error(f"Error closing Redis connection: {e}")