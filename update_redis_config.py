#!/usr/bin/env python3
"""
Update Redis configuration for better stability and error handling.

This script updates the Redis configuration to:
1. Increase timeouts to prevent timeout errors
2. Enable keepalive for long connections
3. Improve connection pool settings
4. Add better error handling defaults
"""

import os
import json
import sys

def update_redis_config():
    """Update Redis configuration in environment or config files."""
    
    print("Updating Redis Configuration")
    print("=" * 50)
    
    # Check for .env.local file
    env_file = ".env.local"
    env_vars = {}
    
    if os.path.exists(env_file):
        print(f"Found {env_file}, reading current configuration...")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    
    # Recommended Redis configuration
    recommended_config = {
        # Increase timeouts
        'REDIS_SOCKET_TIMEOUT': '30',  # 30 seconds for slow operations
        'REDIS_SOCKET_CONNECT_TIMEOUT': '10',  # 10 seconds for connection
        'REDIS_SOCKET_KEEPALIVE': 'true',  # Enable keepalive
        'REDIS_SOCKET_KEEPALIVE_OPTIONS': '1,3,5',  # TCP keepalive options
        
        # Connection pool settings
        'REDIS_POOL_MAX_CONNECTIONS': '100',  # Increase pool size
        'REDIS_POOL_MIN_IDLE': '10',  # Minimum idle connections
        'REDIS_CONNECTION_RETRY_TIMES': '5',  # Retry failed connections
        'REDIS_CONNECTION_RETRY_DELAY': '1.0',  # Delay between retries
        
        # Error handling
        'REDIS_DECODE_RESPONSES': 'true',  # Decode responses automatically
        'REDIS_RETRY_ON_TIMEOUT': 'true',  # Retry on timeout
        'REDIS_RETRY_ON_ERROR': 'true',  # Retry on connection errors
        
        # Health monitoring
        'REDIS_HEALTH_CHECK_INTERVAL': '30',  # Health check every 30s
        'REDIS_CIRCUIT_BREAKER_THRESHOLD': '10',  # Open circuit after 10 failures
        'REDIS_CIRCUIT_BREAKER_TIMEOUT': '60',  # Try to close after 60s
    }
    
    # Update environment variables
    updates_made = []
    for key, recommended_value in recommended_config.items():
        current_value = env_vars.get(key)
        if current_value != recommended_value:
            env_vars[key] = recommended_value
            updates_made.append(f"{key}: {current_value} -> {recommended_value}")
    
    if updates_made:
        print("\nConfiguration updates:")
        for update in updates_made:
            print(f"  {update}")
        
        # Write updated configuration
        with open(env_file, 'w') as f:
            f.write("# Redis Configuration (Updated for stability)\n")
            for key, value in sorted(env_vars.items()):
                if key.startswith('REDIS_'):
                    f.write(f"{key}={value}\n")
            
            # Write non-Redis variables
            f.write("\n# Other Configuration\n")
            for key, value in sorted(env_vars.items()):
                if not key.startswith('REDIS_'):
                    f.write(f"{key}={value}\n")
        
        print(f"\n✓ Updated {env_file} with {len(updates_made)} changes")
    else:
        print("\n✓ Redis configuration is already optimal")
    
    # Create a Redis config override file
    redis_config_override = {
        "socket_timeout": 30.0,
        "socket_connect_timeout": 10.0,
        "socket_keepalive": True,
        "socket_keepalive_options": [1, 3, 5],
        "health_check_interval": 30,
        "decode_responses": True,
        "retry_on_timeout": True,
        "retry_on_error": [
            "ConnectionError",
            "TimeoutError",
            "ConnectionRefusedError"
        ],
        "max_connections": 100,
        "retry": {
            "retries": 5,
            "backoff": "exponential",
            "base_delay": 1.0,
            "max_delay": 10.0
        },
        "circuit_breaker": {
            "failure_threshold": 10,
            "recovery_timeout": 60,
            "half_open_requests": 3
        }
    }
    
    config_file = "scripts/redis_config_override.json"
    with open(config_file, 'w') as f:
        json.dump(redis_config_override, f, indent=2)
    
    print(f"✓ Created {config_file} with optimized settings")
    
    # Create a helper script to apply the configuration
    apply_script = """#!/usr/bin/env python3
\"\"\"Apply Redis configuration overrides.\"\"\"

import os
import json

# Load configuration override
with open('scripts/redis_config_override.json', 'r') as f:
    override = json.load(f)

# Set environment variables
for key, value in override.items():
    if not isinstance(value, (dict, list)):
        env_key = f"REDIS_{key.upper()}"
        os.environ[env_key] = str(value)

print("Redis configuration overrides applied")
"""
    
    with open("apply_redis_config.py", 'w') as f:
        f.write(apply_script)
    
    os.chmod("apply_redis_config.py", 0o755)
    print("✓ Created apply_redis_config.py helper script")
    
    print("\nNext steps:")
    print("1. Review the updated configuration in .env.local")
    print("2. Run: python apply_redis_config.py before starting the system")
    print("3. Restart the continuous AI system to apply changes")
    print("4. Monitor logs for improved stability")


if __name__ == "__main__":
    update_redis_config()