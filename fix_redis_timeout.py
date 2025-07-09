#!/usr/bin/env python3
"""
Fix Redis Timeout Issues
Increases timeout settings and improves connection handling
"""

import os
import sys
import json
from pathlib import Path

def fix_redis_timeouts():
    """Fix Redis timeout issues by updating configuration"""
    
    print("🔧 Fixing Redis timeout issues...")
    
    # Update Redis config to increase timeouts
    config_file = Path("scripts/redis_integration/redis_config.py")
    if config_file.exists():
        content = config_file.read_text()
        
        # Increase socket timeout from 5.0 to 30.0 seconds
        content = content.replace("socket_timeout: float = 5.0", "socket_timeout: float = 30.0")
        content = content.replace("socket_connect_timeout: float = 5.0", "socket_connect_timeout: float = 30.0")
        
        # Increase circuit breaker timeout
        content = content.replace("circuit_breaker_timeout: float = 60.0", "circuit_breaker_timeout: float = 120.0")
        
        config_file.write_text(content)
        print("✅ Updated Redis config timeouts")
    
    # Update MCP-Redis timeout settings
    mcp_config_file = Path("mcp_config.json")
    if mcp_config_file.exists():
        with open(mcp_config_file, 'r') as f:
            config = json.load(f)
        
        # Add timeout settings for redis server
        if 'servers' in config and 'redis' in config['servers']:
            config['servers']['redis']['env'] = config['servers']['redis'].get('env', {})
            config['servers']['redis']['env']['REDIS_TIMEOUT'] = '30'
            config['servers']['redis']['env']['REDIS_CONNECT_TIMEOUT'] = '30'
        
        with open(mcp_config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print("✅ Updated MCP-Redis timeout settings")
    
    # Create a Redis health check script
    health_check = '''#!/usr/bin/env python3
"""Redis Health Check and Recovery"""

import redis
import time
import sys

def check_redis_health():
    """Check Redis health and attempt recovery if needed"""
    try:
        # Try to connect with longer timeout
        r = redis.Redis(host='localhost', port=6379, decode_responses=True, 
                       socket_timeout=30, socket_connect_timeout=30)
        
        # Test connection
        response = r.ping()
        if response:
            print("✅ Redis is healthy")
            return True
    except redis.ConnectionError:
        print("❌ Redis connection failed")
        return False
    except redis.TimeoutError:
        print("❌ Redis timeout - server may be overloaded")
        return False
    except Exception as e:
        print(f"❌ Redis error: {e}")
        return False

def restart_redis():
    """Attempt to restart Redis"""
    import subprocess
    
    print("🔄 Attempting to restart Redis...")
    
    # Try different methods to restart Redis
    try:
        # Method 1: brew services
        subprocess.run(["brew", "services", "restart", "redis"], check=True)
        print("✅ Redis restarted via brew")
        return True
    except:
        pass
    
    try:
        # Method 2: Direct redis-server
        subprocess.run(["redis-cli", "shutdown"], check=False)
        time.sleep(2)
        subprocess.Popen(["redis-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        print("✅ Redis restarted directly")
        return True
    except:
        pass
    
    print("❌ Could not restart Redis automatically")
    return False

if __name__ == "__main__":
    if not check_redis_health():
        if restart_redis():
            time.sleep(5)
            if check_redis_health():
                print("✅ Redis recovered successfully")
                sys.exit(0)
            else:
                print("❌ Redis still unhealthy after restart")
                sys.exit(1)
        else:
            print("❌ Manual intervention required")
            print("Try: brew services restart redis")
            sys.exit(1)
    else:
        sys.exit(0)
'''
    
    health_check_file = Path("check_redis_health.py")
    health_check_file.write_text(health_check)
    os.chmod(health_check_file, 0o755)
    print("✅ Created Redis health check script")
    
    print("\n📋 Recommendations:")
    print("1. Run './check_redis_health.py' to check Redis status")
    print("2. If Redis is down, restart it with: brew services restart redis")
    print("3. Monitor Redis memory usage: redis-cli info memory")
    print("4. Consider increasing max memory: redis-cli config set maxmemory 8gb")
    print("5. Enable Redis persistence: redis-cli config set save '900 1 300 10 60 10000'")

if __name__ == "__main__":
    fix_redis_timeouts()