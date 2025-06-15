"""
Redis Configuration Management

Centralized configuration for Redis connection, clustering, and optimization settings.
Supports development, production, and cluster configurations.
"""

import os
import ssl
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum


class RedisMode(Enum):
    """Redis deployment modes."""
    STANDALONE = "standalone"
    CLUSTER = "cluster"
    SENTINEL = "sentinel"


class RedisEnvironment(Enum):
    """Redis environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class RedisConnectionConfig:
    """Redis connection configuration."""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    ssl_enabled: bool = False
    ssl_cert_reqs: int = ssl.CERT_NONE
    ssl_ca_certs: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    socket_keepalive: bool = True
    socket_keepalive_options: Optional[Dict[int, int]] = None
    health_check_interval: int = 30
    max_connections: int = 50
    retry_on_timeout: bool = True
    decode_responses: bool = True
    encoding: str = "utf-8"
    # Circuit breaker configuration
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: float = 60.0


@dataclass
class RedisClusterConfig:
    """Redis cluster configuration."""
    startup_nodes: List[Dict[str, Any]] = field(default_factory=list)
    max_connections_per_node: int = 50
    skip_full_coverage_check: bool = False
    readonly_mode: bool = False
    decode_responses: bool = True
    health_check_interval: int = 30


@dataclass
class RedisSentinelConfig:
    """Redis Sentinel configuration."""
    sentinels: List[tuple] = field(default_factory=list)
    service_name: str = "mymaster"
    socket_timeout: float = 0.1
    sentinel_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RedisPerformanceConfig:
    """Redis performance optimization settings."""
    max_memory: str = "4gb"
    max_memory_policy: str = "allkeys-lru"
    max_memory_samples: int = 5
    save_config: List[str] = field(default_factory=lambda: ["900 1", "300 10", "60 10000"])
    rdb_compression: bool = True
    rdb_checksum: bool = True
    tcp_keepalive: int = 60
    tcp_backlog: int = 511
    timeout: int = 300
    hash_max_ziplist_entries: int = 512
    hash_max_ziplist_value: int = 64
    list_max_ziplist_size: int = -2
    set_max_intset_entries: int = 512
    zset_max_ziplist_entries: int = 128
    zset_max_ziplist_value: int = 64


class RedisConfig:
    """Centralized Redis configuration management."""
    
    def __init__(self, environment: RedisEnvironment = None):
        """Initialize Redis configuration.
        
        Args:
            environment: Target environment (development, staging, production)
        """
        self.environment = environment or self._detect_environment()
        self.mode = self._get_redis_mode()
        self.connection_config = self._build_connection_config()
        self.cluster_config = self._build_cluster_config()
        self.sentinel_config = self._build_sentinel_config()
        self.performance_config = self._build_performance_config()
    
    def _detect_environment(self) -> RedisEnvironment:
        """Auto-detect environment from environment variables."""
        env = os.getenv('REDIS_ENVIRONMENT', 'development').lower()
        try:
            return RedisEnvironment(env)
        except ValueError:
            return RedisEnvironment.DEVELOPMENT
    
    def _get_redis_mode(self) -> RedisMode:
        """Get Redis deployment mode from environment."""
        mode = os.getenv('REDIS_MODE', 'standalone').lower()
        try:
            return RedisMode(mode)
        except ValueError:
            return RedisMode.STANDALONE
    
    def _build_connection_config(self) -> RedisConnectionConfig:
        """Build connection configuration based on environment."""
        config = RedisConnectionConfig()
        
        # Base configuration
        config.host = os.getenv('REDIS_HOST', 'localhost')
        config.port = int(os.getenv('REDIS_PORT', '6379'))
        config.password = os.getenv('REDIS_PASSWORD')
        config.db = int(os.getenv('REDIS_DB', '0'))
        
        # SSL configuration
        config.ssl_enabled = os.getenv('REDIS_SSL_ENABLED', 'false').lower() == 'true'
        if config.ssl_enabled:
            config.ssl_cert_reqs = ssl.CERT_REQUIRED
            config.ssl_ca_certs = os.getenv('REDIS_SSL_CA_CERTS')
            config.ssl_certfile = os.getenv('REDIS_SSL_CERTFILE')
            config.ssl_keyfile = os.getenv('REDIS_SSL_KEYFILE')
        
        # Performance tuning based on environment
        if self.environment == RedisEnvironment.PRODUCTION:
            # Increase pool size to avoid excessive connection failures
            config.max_connections = int(os.getenv('REDIS_MAX_CONNECTIONS', '5000'))
            config.socket_timeout = float(os.getenv('REDIS_SOCKET_TIMEOUT', '30.0'))  # Increased timeout
            config.socket_connect_timeout = float(os.getenv('REDIS_SOCKET_CONNECT_TIMEOUT', '30.0'))  # Increased timeout
            config.health_check_interval = int(os.getenv('REDIS_HEALTH_CHECK_INTERVAL', '15'))
        elif self.environment == RedisEnvironment.DEVELOPMENT:
            # Increase pool size to avoid excessive connection failures
            config.max_connections = int(os.getenv('REDIS_MAX_CONNECTIONS', '5000'))
            config.socket_timeout = float(os.getenv('REDIS_SOCKET_TIMEOUT', '30.0'))  # Increased timeout
            config.socket_connect_timeout = float(os.getenv('REDIS_SOCKET_CONNECT_TIMEOUT', '30.0'))  # Increased timeout
            config.health_check_interval = int(os.getenv('REDIS_HEALTH_CHECK_INTERVAL', '60'))
        
        # Circuit breaker configuration
        config.circuit_breaker_enabled = os.getenv('REDIS_CIRCUIT_BREAKER_ENABLED', 'true').lower() == 'true'
        config.circuit_breaker_failure_threshold = int(os.getenv('REDIS_CIRCUIT_BREAKER_THRESHOLD', '50'))  # Increased from 5
        config.circuit_breaker_timeout = float(os.getenv('REDIS_CIRCUIT_BREAKER_TIMEOUT', '60.0'))
        
        return config
    
    def _build_cluster_config(self) -> RedisClusterConfig:
        """Build cluster configuration."""
        config = RedisClusterConfig()
        
        if self.mode == RedisMode.CLUSTER:
            # Parse cluster nodes from environment
            nodes_str = os.getenv('REDIS_CLUSTER_NODES', 'localhost:7000,localhost:7001,localhost:7002')
            for node_str in nodes_str.split(','):
                host, port = node_str.strip().split(':')
                config.startup_nodes.append({'host': host, 'port': int(port)})
            
            config.max_connections_per_node = int(os.getenv('REDIS_CLUSTER_MAX_CONNECTIONS', '50'))
            config.skip_full_coverage_check = os.getenv('REDIS_CLUSTER_SKIP_COVERAGE', 'false').lower() == 'true'
        
        return config
    
    def _build_sentinel_config(self) -> RedisSentinelConfig:
        """Build sentinel configuration."""
        config = RedisSentinelConfig()
        
        if self.mode == RedisMode.SENTINEL:
            # Parse sentinel nodes from environment
            sentinels_str = os.getenv('REDIS_SENTINELS', 'localhost:26379')
            for sentinel_str in sentinels_str.split(','):
                host, port = sentinel_str.strip().split(':')
                config.sentinels.append((host, int(port)))
            
            config.service_name = os.getenv('REDIS_SENTINEL_SERVICE', 'mymaster')
            config.socket_timeout = float(os.getenv('REDIS_SENTINEL_TIMEOUT', '0.1'))
        
        return config
    
    def _build_performance_config(self) -> RedisPerformanceConfig:
        """Build performance configuration."""
        config = RedisPerformanceConfig()
        
        if self.environment == RedisEnvironment.PRODUCTION:
            config.max_memory = os.getenv('REDIS_MAX_MEMORY', '8gb')
            config.max_memory_policy = os.getenv('REDIS_MAX_MEMORY_POLICY', 'allkeys-lru')
            config.tcp_keepalive = int(os.getenv('REDIS_TCP_KEEPALIVE', '60'))
            config.timeout = int(os.getenv('REDIS_TIMEOUT', '300'))
        
        return config
    
    def get_connection_url(self) -> str:
        """Get Redis connection URL."""
        config = self.connection_config
        
        # Build URL components
        scheme = 'rediss' if config.ssl_enabled else 'redis'
        auth = f':{config.password}@' if config.password else ''
        host_port = f'{config.host}:{config.port}'
        db = f'/{config.db}' if config.db != 0 else ''
        
        return f'{scheme}://{auth}{host_port}{db}'
    
    def get_connection_kwargs(self) -> Dict[str, Any]:
        """Get Redis connection keyword arguments."""
        config = self.connection_config
        kwargs = {
            'host': config.host,
            'port': config.port,
            'db': config.db,
            'password': config.password,
            'socket_timeout': config.socket_timeout,
            'socket_connect_timeout': config.socket_connect_timeout,
            'socket_keepalive': config.socket_keepalive,
            'socket_keepalive_options': config.socket_keepalive_options,
            'health_check_interval': config.health_check_interval,
            'retry_on_timeout': config.retry_on_timeout,
            'decode_responses': config.decode_responses,
            'encoding': config.encoding,
        }
        
        # Add SSL configuration if enabled
        if config.ssl_enabled:
            kwargs.update({
                'ssl': True,
                'ssl_cert_reqs': config.ssl_cert_reqs,
                'ssl_ca_certs': config.ssl_ca_certs,
                'ssl_certfile': config.ssl_certfile,
                'ssl_keyfile': config.ssl_keyfile,
            })
        
        # Filter out None values and problematic socket keepalive options
        filtered_kwargs = {}
        for k, v in kwargs.items():
            if v is not None:
                # Skip socket_keepalive_options if None or empty
                if k == 'socket_keepalive_options' and (v is None or not v):
                    continue
                filtered_kwargs[k] = v
        return filtered_kwargs
    
    def get_pool_kwargs(self) -> Dict[str, Any]:
        """Get connection pool keyword arguments."""
        kwargs = self.get_connection_kwargs()
        kwargs['max_connections'] = self.connection_config.max_connections
        return kwargs
    
    def get_cluster_kwargs(self) -> Dict[str, Any]:
        """Get cluster connection keyword arguments."""
        if self.mode != RedisMode.CLUSTER:
            raise ValueError("Cluster configuration only available in cluster mode")
        
        config = self.cluster_config
        return {
            'startup_nodes': config.startup_nodes,
            'max_connections_per_node': config.max_connections_per_node,
            'skip_full_coverage_check': config.skip_full_coverage_check,
            'readonly_mode': config.readonly_mode,
            'decode_responses': config.decode_responses,
            'health_check_interval': config.health_check_interval,
        }
    
    def get_sentinel_kwargs(self) -> Dict[str, Any]:
        """Get sentinel connection keyword arguments."""
        if self.mode != RedisMode.SENTINEL:
            raise ValueError("Sentinel configuration only available in sentinel mode")
        
        config = self.sentinel_config
        return {
            'sentinels': config.sentinels,
            'service_name': config.service_name,
            'socket_timeout': config.socket_timeout,
            'sentinel_kwargs': config.sentinel_kwargs,
        }
    
    def generate_redis_conf(self) -> str:
        """Generate Redis configuration file content."""
        config = self.performance_config
        
        conf_lines = [
            f"# Redis Configuration - {self.environment.value}",
            f"# Generated automatically",
            "",
            "# Memory Management",
            f"maxmemory {config.max_memory}",
            f"maxmemory-policy {config.max_memory_policy}",
            f"maxmemory-samples {config.max_memory_samples}",
            "",
            "# Persistence",
        ]
        
        for save_rule in config.save_config:
            conf_lines.append(f"save {save_rule}")
        
        conf_lines.extend([
            f"rdbcompression {'yes' if config.rdb_compression else 'no'}",
            f"rdbchecksum {'yes' if config.rdb_checksum else 'no'}",
            "",
            "# Network",
            f"tcp-keepalive {config.tcp_keepalive}",
            f"tcp-backlog {config.tcp_backlog}",
            f"timeout {config.timeout}",
            "",
            "# Performance Tuning",
            f"hash-max-ziplist-entries {config.hash_max_ziplist_entries}",
            f"hash-max-ziplist-value {config.hash_max_ziplist_value}",
            f"list-max-ziplist-size {config.list_max_ziplist_size}",
            f"set-max-intset-entries {config.set_max_intset_entries}",
            f"zset-max-ziplist-entries {config.zset_max_ziplist_entries}",
            f"zset-max-ziplist-value {config.zset_max_ziplist_value}",
        ])
        
        if self.mode == RedisMode.CLUSTER:
            conf_lines.extend([
                "",
                "# Cluster Configuration",
                "cluster-enabled yes",
                "cluster-config-file nodes.conf",
                "cluster-node-timeout 15000",
                "cluster-require-full-coverage no",
            ])
        
        return "\n".join(conf_lines)
    
    def validate(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        # Validate connection config
        if not self.connection_config.host:
            issues.append("Redis host is required")
        
        if not (1 <= self.connection_config.port <= 65535):
            issues.append("Redis port must be between 1 and 65535")
        
        if self.connection_config.max_connections <= 0:
            issues.append("Max connections must be positive")
        
        # Validate cluster config
        if self.mode == RedisMode.CLUSTER:
            if not self.cluster_config.startup_nodes:
                issues.append("Cluster startup nodes are required in cluster mode")
            
            if len(self.cluster_config.startup_nodes) < 3:
                issues.append("Cluster requires at least 3 nodes")
        
        # Validate sentinel config
        if self.mode == RedisMode.SENTINEL:
            if not self.sentinel_config.sentinels:
                issues.append("Sentinel nodes are required in sentinel mode")
            
            if not self.sentinel_config.service_name:
                issues.append("Sentinel service name is required")
        
        # Validate SSL config
        if self.connection_config.ssl_enabled:
            if not self.connection_config.ssl_ca_certs:
                issues.append("SSL CA certificates are required when SSL is enabled")
        
        return issues
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"RedisConfig(env={self.environment.value}, mode={self.mode.value}, host={self.connection_config.host}:{self.connection_config.port})"


# Global configuration instance
_redis_config = None


def get_redis_config(environment: RedisEnvironment = None) -> RedisConfig:
    """Get global Redis configuration instance."""
    global _redis_config
    if _redis_config is None or (environment and _redis_config.environment != environment):
        _redis_config = RedisConfig(environment)
    return _redis_config


def set_redis_config(config: RedisConfig):
    """Set global Redis configuration instance."""
    global _redis_config
    _redis_config = config