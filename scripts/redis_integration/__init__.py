"""
Redis Integration Package

Enterprise-grade Redis integration for the enhanced worker intelligence system.
Provides distributed state management, caching, real-time coordination, and analytics.
"""

from .redis_client import RedisClient, get_redis_client, close_redis_client
from .redis_config import RedisConfig
from .redis_connection_pool import get_connection_pool, close_connection_pool, SingletonConnectionPool
from .redis_cache_manager import RedisCacheManager
from .redis_state_manager import RedisStateManager
from .redis_pubsub_manager import RedisPubSubManager
from .redis_streams_manager import RedisStreamsManager
from .redis_locks_manager import RedisLocksManager
from .redis_monitoring import RedisMonitoring
from .redis_analytics import RedisAnalytics

__all__ = [
    'RedisClient',
    'get_redis_client',
    'close_redis_client',
    'RedisConfig',
    'get_connection_pool',
    'close_connection_pool',
    'SingletonConnectionPool',
    'RedisCacheManager',
    'RedisStateManager',
    'RedisPubSubManager',
    'RedisStreamsManager',
    'RedisLocksManager',
    'RedisMonitoring',
    'RedisAnalytics'
]

__version__ = '1.0.0'