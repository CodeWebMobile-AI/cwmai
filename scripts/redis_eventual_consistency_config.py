"""
Redis Eventual Consistency Configuration

This module defines which data types should use eventual consistency vs strong consistency,
providing configuration for TTLs, refresh intervals, and consistency patterns.
"""

from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from datetime import timedelta
import re


class ConsistencyLevel(Enum):
    """Defines different consistency levels for data operations."""
    STRONG = "strong"  # Immediate consistency required
    EVENTUAL = "eventual"  # Can tolerate delays
    WEAK = "weak"  # Can tolerate significant delays and potential data loss
    BEST_EFFORT = "best_effort"  # No consistency guarantees


class DataCategory(Enum):
    """Categories of data with different consistency requirements."""
    # Critical data requiring strong consistency
    FINANCIAL = "financial"
    USER_AUTH = "user_auth"
    TASK_STATE = "task_state"
    CRITICAL_CONFIG = "critical_config"
    
    # Data that can use eventual consistency
    METRICS = "metrics"
    LOGS = "logs"
    STATUS_UPDATES = "status_updates"
    ANALYTICS = "analytics"
    CACHE = "cache"
    
    # Data with weak consistency requirements
    TELEMETRY = "telemetry"
    DEBUG_INFO = "debug_info"
    TEMPORARY_STATE = "temporary_state"
    
    # Best effort data
    PERFORMANCE_METRICS = "performance_metrics"
    USER_PREFERENCES = "user_preferences"
    NOTIFICATIONS = "notifications"


# Consistency configuration for each data category
CONSISTENCY_CONFIG: Dict[DataCategory, Dict[str, Union[ConsistencyLevel, int, timedelta]]] = {
    # Strong consistency required
    DataCategory.FINANCIAL: {
        "level": ConsistencyLevel.STRONG,
        "ttl": None,  # No expiration
        "sync_timeout": timedelta(seconds=5),
        "retry_attempts": 3,
        "require_ack": True,
    },
    DataCategory.USER_AUTH: {
        "level": ConsistencyLevel.STRONG,
        "ttl": timedelta(hours=24),
        "sync_timeout": timedelta(seconds=5),
        "retry_attempts": 3,
        "require_ack": True,
    },
    DataCategory.TASK_STATE: {
        "level": ConsistencyLevel.STRONG,
        "ttl": timedelta(days=7),
        "sync_timeout": timedelta(seconds=3),
        "retry_attempts": 2,
        "require_ack": True,
    },
    DataCategory.CRITICAL_CONFIG: {
        "level": ConsistencyLevel.STRONG,
        "ttl": None,
        "sync_timeout": timedelta(seconds=5),
        "retry_attempts": 3,
        "require_ack": True,
    },
    
    # Eventual consistency acceptable
    DataCategory.METRICS: {
        "level": ConsistencyLevel.EVENTUAL,
        "ttl": timedelta(hours=24),
        "refresh_interval": timedelta(minutes=5),
        "sync_timeout": timedelta(seconds=1),
        "retry_attempts": 1,
        "require_ack": False,
    },
    DataCategory.LOGS: {
        "level": ConsistencyLevel.EVENTUAL,
        "ttl": timedelta(days=30),
        "refresh_interval": timedelta(minutes=1),
        "sync_timeout": timedelta(milliseconds=500),
        "retry_attempts": 0,
        "require_ack": False,
    },
    DataCategory.STATUS_UPDATES: {
        "level": ConsistencyLevel.EVENTUAL,
        "ttl": timedelta(hours=1),
        "refresh_interval": timedelta(seconds=30),
        "sync_timeout": timedelta(seconds=2),
        "retry_attempts": 1,
        "require_ack": False,
    },
    DataCategory.ANALYTICS: {
        "level": ConsistencyLevel.EVENTUAL,
        "ttl": timedelta(days=90),
        "refresh_interval": timedelta(hours=1),
        "sync_timeout": timedelta(seconds=5),
        "retry_attempts": 1,
        "require_ack": False,
    },
    DataCategory.CACHE: {
        "level": ConsistencyLevel.EVENTUAL,
        "ttl": timedelta(minutes=15),
        "refresh_interval": timedelta(minutes=5),
        "sync_timeout": timedelta(milliseconds=100),
        "retry_attempts": 0,
        "require_ack": False,
    },
    
    # Weak consistency
    DataCategory.TELEMETRY: {
        "level": ConsistencyLevel.WEAK,
        "ttl": timedelta(days=7),
        "refresh_interval": timedelta(hours=6),
        "sync_timeout": timedelta(milliseconds=100),
        "retry_attempts": 0,
        "require_ack": False,
    },
    DataCategory.DEBUG_INFO: {
        "level": ConsistencyLevel.WEAK,
        "ttl": timedelta(hours=12),
        "refresh_interval": timedelta(hours=1),
        "sync_timeout": timedelta(milliseconds=50),
        "retry_attempts": 0,
        "require_ack": False,
    },
    DataCategory.TEMPORARY_STATE: {
        "level": ConsistencyLevel.WEAK,
        "ttl": timedelta(minutes=30),
        "refresh_interval": timedelta(minutes=10),
        "sync_timeout": timedelta(milliseconds=50),
        "retry_attempts": 0,
        "require_ack": False,
    },
    
    # Best effort
    DataCategory.PERFORMANCE_METRICS: {
        "level": ConsistencyLevel.BEST_EFFORT,
        "ttl": timedelta(days=1),
        "refresh_interval": timedelta(minutes=15),
        "sync_timeout": timedelta(milliseconds=10),
        "retry_attempts": 0,
        "require_ack": False,
    },
    DataCategory.USER_PREFERENCES: {
        "level": ConsistencyLevel.BEST_EFFORT,
        "ttl": timedelta(days=365),
        "refresh_interval": timedelta(days=1),
        "sync_timeout": timedelta(seconds=1),
        "retry_attempts": 0,
        "require_ack": False,
    },
    DataCategory.NOTIFICATIONS: {
        "level": ConsistencyLevel.BEST_EFFORT,
        "ttl": timedelta(days=7),
        "refresh_interval": timedelta(hours=1),
        "sync_timeout": timedelta(milliseconds=100),
        "retry_attempts": 0,
        "require_ack": False,
    },
}


# Key pattern mappings to data categories
KEY_PATTERNS: List[Tuple[re.Pattern, DataCategory]] = [
    # Financial patterns
    (re.compile(r"^(payment|transaction|billing|invoice):"), DataCategory.FINANCIAL),
    
    # Authentication patterns
    (re.compile(r"^(auth|session|token|user:auth):"), DataCategory.USER_AUTH),
    
    # Task state patterns
    (re.compile(r"^(task:state|workflow:state|job:status):"), DataCategory.TASK_STATE),
    
    # Configuration patterns
    (re.compile(r"^(config:critical|system:config|security:config):"), DataCategory.CRITICAL_CONFIG),
    
    # Metrics patterns
    (re.compile(r"^(metric|stats|counter|gauge):"), DataCategory.METRICS),
    
    # Log patterns
    (re.compile(r"^(log|audit|event):"), DataCategory.LOGS),
    
    # Status patterns
    (re.compile(r"^(status|health|heartbeat):"), DataCategory.STATUS_UPDATES),
    
    # Analytics patterns
    (re.compile(r"^(analytics|report|aggregation):"), DataCategory.ANALYTICS),
    
    # Cache patterns
    (re.compile(r"^(cache|temp:cache|computed):"), DataCategory.CACHE),
    
    # Telemetry patterns
    (re.compile(r"^(telemetry|monitoring|trace):"), DataCategory.TELEMETRY),
    
    # Debug patterns
    (re.compile(r"^(debug|trace:debug|dev):"), DataCategory.DEBUG_INFO),
    
    # Temporary patterns
    (re.compile(r"^(tmp|temp|ephemeral):"), DataCategory.TEMPORARY_STATE),
    
    # Performance patterns
    (re.compile(r"^(perf|benchmark|timing):"), DataCategory.PERFORMANCE_METRICS),
    
    # Preference patterns
    (re.compile(r"^(pref|user:pref|settings):"), DataCategory.USER_PREFERENCES),
    
    # Notification patterns
    (re.compile(r"^(notify|alert|message):"), DataCategory.NOTIFICATIONS),
]


def get_data_category(key: str) -> Optional[DataCategory]:
    """
    Determine the data category for a given Redis key.
    
    Args:
        key: Redis key to categorize
        
    Returns:
        DataCategory or None if no pattern matches
    """
    for pattern, category in KEY_PATTERNS:
        if pattern.match(key):
            return category
    return None


def get_consistency_level(key: str) -> ConsistencyLevel:
    """
    Get the consistency level for a given Redis key.
    
    Args:
        key: Redis key
        
    Returns:
        ConsistencyLevel (defaults to STRONG if unknown)
    """
    category = get_data_category(key)
    if category and category in CONSISTENCY_CONFIG:
        return CONSISTENCY_CONFIG[category]["level"]
    return ConsistencyLevel.STRONG  # Default to strong consistency


def get_ttl(key: str) -> Optional[int]:
    """
    Get the TTL in seconds for a given Redis key.
    
    Args:
        key: Redis key
        
    Returns:
        TTL in seconds or None for no expiration
    """
    category = get_data_category(key)
    if category and category in CONSISTENCY_CONFIG:
        ttl = CONSISTENCY_CONFIG[category].get("ttl")
        if isinstance(ttl, timedelta):
            return int(ttl.total_seconds())
    return None


def get_refresh_interval(key: str) -> Optional[int]:
    """
    Get the refresh interval in seconds for eventual consistency data.
    
    Args:
        key: Redis key
        
    Returns:
        Refresh interval in seconds or None
    """
    category = get_data_category(key)
    if category and category in CONSISTENCY_CONFIG:
        interval = CONSISTENCY_CONFIG[category].get("refresh_interval")
        if isinstance(interval, timedelta):
            return int(interval.total_seconds())
    return None


def get_sync_timeout(key: str) -> float:
    """
    Get the synchronization timeout for a given Redis key.
    
    Args:
        key: Redis key
        
    Returns:
        Timeout in seconds
    """
    category = get_data_category(key)
    if category and category in CONSISTENCY_CONFIG:
        timeout = CONSISTENCY_CONFIG[category].get("sync_timeout", timedelta(seconds=5))
        if isinstance(timeout, timedelta):
            return timeout.total_seconds()
    return 5.0  # Default 5 seconds


def should_require_ack(key: str) -> bool:
    """
    Determine if acknowledgment is required for a given Redis key.
    
    Args:
        key: Redis key
        
    Returns:
        True if acknowledgment required
    """
    category = get_data_category(key)
    if category and category in CONSISTENCY_CONFIG:
        return CONSISTENCY_CONFIG[category].get("require_ack", True)
    return True  # Default to requiring acknowledgment


def get_retry_attempts(key: str) -> int:
    """
    Get the number of retry attempts for a given Redis key.
    
    Args:
        key: Redis key
        
    Returns:
        Number of retry attempts
    """
    category = get_data_category(key)
    if category and category in CONSISTENCY_CONFIG:
        return CONSISTENCY_CONFIG[category].get("retry_attempts", 3)
    return 3  # Default to 3 attempts


def can_use_eventual_consistency(key: str) -> bool:
    """
    Check if a key can use eventual consistency.
    
    Args:
        key: Redis key
        
    Returns:
        True if eventual consistency is acceptable
    """
    level = get_consistency_level(key)
    return level in [ConsistencyLevel.EVENTUAL, ConsistencyLevel.WEAK, ConsistencyLevel.BEST_EFFORT]


def requires_strong_consistency(key: str) -> bool:
    """
    Check if a key requires strong consistency.
    
    Args:
        key: Redis key
        
    Returns:
        True if strong consistency is required
    """
    return get_consistency_level(key) == ConsistencyLevel.STRONG


def get_consistency_config(key: str) -> Dict[str, Union[ConsistencyLevel, int, float, bool]]:
    """
    Get the complete consistency configuration for a key.
    
    Args:
        key: Redis key
        
    Returns:
        Dictionary with all consistency settings
    """
    category = get_data_category(key)
    if category and category in CONSISTENCY_CONFIG:
        config = CONSISTENCY_CONFIG[category].copy()
        # Convert timedelta to seconds
        if "ttl" in config and isinstance(config["ttl"], timedelta):
            config["ttl"] = int(config["ttl"].total_seconds())
        if "refresh_interval" in config and isinstance(config["refresh_interval"], timedelta):
            config["refresh_interval"] = int(config["refresh_interval"].total_seconds())
        if "sync_timeout" in config and isinstance(config["sync_timeout"], timedelta):
            config["sync_timeout"] = config["sync_timeout"].total_seconds()
        return config
    
    # Default configuration
    return {
        "level": ConsistencyLevel.STRONG,
        "ttl": None,
        "sync_timeout": 5.0,
        "retry_attempts": 3,
        "require_ack": True,
    }


# Bulk operation configurations
BULK_OPERATION_CONFIG = {
    "eventual_batch_size": 1000,  # Larger batches for eventual consistency
    "strong_batch_size": 100,     # Smaller batches for strong consistency
    "eventual_pipeline_size": 500,
    "strong_pipeline_size": 50,
    "eventual_parallel_workers": 10,
    "strong_parallel_workers": 3,
}


def get_bulk_config(consistency_level: ConsistencyLevel) -> Dict[str, int]:
    """
    Get bulk operation configuration based on consistency level.
    
    Args:
        consistency_level: The consistency level
        
    Returns:
        Configuration dictionary for bulk operations
    """
    if consistency_level in [ConsistencyLevel.EVENTUAL, ConsistencyLevel.WEAK, ConsistencyLevel.BEST_EFFORT]:
        return {
            "batch_size": BULK_OPERATION_CONFIG["eventual_batch_size"],
            "pipeline_size": BULK_OPERATION_CONFIG["eventual_pipeline_size"],
            "parallel_workers": BULK_OPERATION_CONFIG["eventual_parallel_workers"],
        }
    else:
        return {
            "batch_size": BULK_OPERATION_CONFIG["strong_batch_size"],
            "pipeline_size": BULK_OPERATION_CONFIG["strong_pipeline_size"],
            "parallel_workers": BULK_OPERATION_CONFIG["strong_parallel_workers"],
        }


# Example usage and testing
if __name__ == "__main__":
    # Test key categorization
    test_keys = [
        "payment:12345",
        "auth:session:user123",
        "task:state:worker1",
        "metric:cpu:usage",
        "log:error:2024",
        "cache:api:response",
        "tmp:processing:data",
        "unknown:key:type",
    ]
    
    print("Key Consistency Analysis:")
    print("-" * 60)
    
    for key in test_keys:
        category = get_data_category(key)
        level = get_consistency_level(key)
        ttl = get_ttl(key)
        can_eventual = can_use_eventual_consistency(key)
        
        print(f"Key: {key}")
        print(f"  Category: {category.value if category else 'Unknown'}")
        print(f"  Consistency: {level.value}")
        print(f"  TTL: {ttl if ttl else 'No expiration'}")
        print(f"  Can use eventual: {can_eventual}")
        print()