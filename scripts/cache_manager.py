"""
Cache Manager - Stub module for custom tools
"""

import json
import time
from pathlib import Path
from typing import Any, Optional, Dict
import hashlib


class CacheManager:
    """Simple cache manager for custom tools"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_key(self, key: str) -> str:
        """Generate a cache key"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str, max_age: int = 3600) -> Optional[Any]:
        """Get value from cache if not expired"""
        cache_key = self._get_cache_key(key)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            if time.time() - data['timestamp'] < max_age:
                return data['value']
        
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        cache_key = self._get_cache_key(key)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        data = {
            'timestamp': time.time(),
            'value': value
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    
    def clear(self):
        """Clear all cache"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()


# Default instance
cache_manager = CacheManager()