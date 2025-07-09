"""
Redis Compatibility Wrapper

Import this instead of redis to use the hybrid state manager.

Example:
    # Before:
    import redis
    client = redis.Redis()
    
    # After:
    import redis_compat as redis
    client = redis.Redis()
"""

from scripts.hybrid_state_manager import create_redis_replacement as Redis
from scripts.hybrid_state_manager import RedisReplacement as StrictRedis

# Common Redis exceptions (no-ops for compatibility)
class RedisError(Exception):
    pass

class ConnectionError(RedisError):
    pass

class TimeoutError(RedisError):
    pass

# Export common names
__all__ = ['Redis', 'StrictRedis', 'RedisError', 'ConnectionError', 'TimeoutError']
