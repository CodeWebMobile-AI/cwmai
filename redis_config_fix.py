#!/usr/bin/env python3
"""
Fixed Redis configuration with reasonable limits
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')

# Reasonable Redis configuration
REDIS_CONFIG = {
    # Connection settings
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', 6379)),
    'password': os.getenv('REDIS_PASSWORD', None),
    'db': int(os.getenv('REDIS_DB', 0)),
    
    # Pool settings - MUCH more reasonable
    'max_connections': 100,  # Down from 5000
    'socket_timeout': 5.0,  # Down from 30.0
    'socket_connect_timeout': 5.0,  # Down from 30.0
    'socket_keepalive': True,
    'socket_keepalive_options': {
        1: 1,  # TCP_KEEPIDLE
        2: 1,  # TCP_KEEPINTVL
        3: 5,  # TCP_KEEPCNT
    },
    
    # Circuit breaker settings
    'circuit_breaker_failure_threshold': 5,  # Down from 50
    'circuit_breaker_recovery_timeout': 60,
    'circuit_breaker_expected_exception': Exception,
    
    # Health check settings
    'health_check_interval': 30,
    'retry_on_timeout': True,
    'retry_on_error': [ConnectionError, TimeoutError],
    'retry': {
        'retries': 3,
        'backoff': 0.5,
    }
}

# Update .env.local with new values
def update_env_file():
    """Update .env.local with reasonable Redis settings"""
    env_updates = {
        'REDIS_MAX_CONNECTIONS': '100',
        'REDIS_SOCKET_TIMEOUT': '5.0',
        'REDIS_SOCKET_CONNECT_TIMEOUT': '5.0',
        'REDIS_CIRCUIT_BREAKER_THRESHOLD': '5',
        'REDIS_HEALTH_CHECK_INTERVAL': '30',
    }
    
    env_path = '.env.local'
    
    # Read current env file
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Update or add Redis settings
    updated_lines = []
    updated_keys = set()
    
    for line in lines:
        key = line.split('=')[0].strip()
        if key in env_updates:
            updated_lines.append(f"{key}={env_updates[key]}\n")
            updated_keys.add(key)
        else:
            updated_lines.append(line)
    
    # Add any missing keys
    for key, value in env_updates.items():
        if key not in updated_keys:
            updated_lines.append(f"{key}={value}\n")
    
    # Write back
    with open(env_path, 'w') as f:
        f.writelines(updated_lines)
    
    print("✓ Updated .env.local with reasonable Redis settings")

if __name__ == "__main__":
    update_env_file()
    print("\nNew Redis configuration:")
    for key, value in REDIS_CONFIG.items():
        if not key.startswith('socket_keepalive_options'):
            print(f"  {key}: {value}")