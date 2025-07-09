#!/usr/bin/env python3
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
