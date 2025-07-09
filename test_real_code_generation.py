#!/usr/bin/env python3
"""
Test that the evolutionary system now generates real code, not placeholders
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.evolutionary_code_generator_fix import RealCodeEvolutionFix

async def test_code_validation():
    """Test the code validation system."""
    
    print("🧪 Testing Real Code Generation Validation\n")
    
    validator = RealCodeEvolutionFix()
    
    # Test 1: Placeholder code (should fail)
    print("Test 1: Validating placeholder code...")
    placeholder_code = '''
class TestImprovement:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.name = "test_improvement"
        
    def apply(self, context):
        # TODO: Implement this
        self.logger.info("Applying improvement")
        return {'success': True, 'improvement': self.name}
'''
    
    is_valid, error_msg, _ = validator.validate_generated_code(
        placeholder_code, 
        {'name': 'test_improvement', 'type': 'test'}
    )
    
    print(f"  Valid: {is_valid}")
    print(f"  Error: {error_msg}")
    assert not is_valid, "Placeholder code should not validate"
    print("  ✅ Correctly rejected placeholder code\n")
    
    # Test 2: Real implementation (should pass)
    print("Test 2: Validating real implementation...")
    real_code = '''
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
'''
    
    is_valid, error_msg, test_results = validator.validate_generated_code(
        real_code,
        {'name': 'cache_optimizer', 'type': 'cache', 'code': real_code}
    )
    
    print(f"  Valid: {is_valid}")
    print(f"  Error: {error_msg}")
    if test_results:
        print(f"  Tests run: {test_results['tests_run']}")
        print(f"  Tests passed: {test_results['tests_passed']}")
    
    assert is_valid, f"Real code should validate, but got: {error_msg}"
    print("  ✅ Correctly validated real implementation\n")
    
    # Test 3: Create implementation prompt
    print("Test 3: Testing implementation prompt generation...")
    improvement = {
        'name': 'intelligent_rate_limiter',
        'type': 'performance',
        'description': 'Adaptive rate limiting based on system load'
    }
    
    prompt = validator.create_implementation_prompt(improvement)
    assert 'REAL, EXECUTABLE code' in prompt
    assert 'not placeholders' in prompt
    print("  ✅ Implementation prompt includes real code requirements\n")
    
    print("✨ All tests passed! The system now validates real code generation.")

if __name__ == "__main__":
    asyncio.run(test_code_validation())