#!/usr/bin/env python3
"""Debug cache test"""

code = '''
import logging
from typing import Dict, Any

class CacheOptimizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.name = "cache_optimizer"
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0
        
    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply cache optimization with real logic."""
        action = context.get('action', 'get')
        
        if action == 'get':
            key = context.get('key', '')
            if key in self.cache:
                self.hit_count += 1
                return {
                    'success': True,
                    'data': self.cache[key],
                    'hit': True,
                    'stats': {
                        'hits': self.hit_count,
                        'misses': self.miss_count,
                        'hit_rate': self.hit_count / (self.hit_count + self.miss_count)
                    }
                }
            else:
                self.miss_count += 1
                return {
                    'success': False,
                    'hit': False,
                    'stats': {
                        'hits': self.hit_count,
                        'misses': self.miss_count
                    }
                }
                
        elif action == 'set':
            key = context.get('key', '')
            value = context.get('value', None)
            self.cache[key] = value
            return {
                'success': True,
                'action': 'set',
                'key': key,
                'cache_size': len(self.cache)
            }
            
        elif action == 'clear':
            old_size = len(self.cache)
            self.cache.clear()
            self.hit_count = 0
            self.miss_count = 0
            return {
                'success': True,
                'action': 'clear',
                'cleared_items': old_size
            }
            
        return {'success': False, 'error': 'Unknown action'}

# Test cache-specific functionality
instance = CacheOptimizer()

# Test 1: Basic apply
result = instance.apply({'test': True})
print(f"Test 1 result: {result}")

# Test 2: Cache store
test_data = {'key': 'test_key', 'value': 'test_value', 'action': 'set'}  # Fixed: action should be 'set' not 'store'
result = instance.apply(test_data)
print(f"Test 2 result: {result}")

# Test 3: Cache retrieve  
result = instance.apply({'action': 'get', 'key': 'test_key'})  # Fixed: Use 'get' action
print(f"Test 3 result: {result}")
'''

exec(code)