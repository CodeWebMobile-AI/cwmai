#!/usr/bin/env python3
"""Debug test execution"""

import tempfile
import subprocess
import sys

# Simple test code
test_code = '''
import logging
from typing import Dict, Any

class CacheOptimizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.name = "cache_optimizer"
        self.cache = {}
        
    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': True, 'data': 'test'}

# Test code
def run_tests():
    tests_run = 0
    tests_passed = 0
    
    try:
        instance = CacheOptimizer()
        tests_run += 1
        tests_passed += 1
        print("Instance created")
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"TESTS_RUN:{tests_run}")
    print(f"TESTS_PASSED:{tests_passed}")

if __name__ == "__main__":
    run_tests()
'''

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(test_code)
    f.flush()
    
    print(f"Test file: {f.name}")
    
    # Run it
    result = subprocess.run(
        [sys.executable, f.name],
        capture_output=True,
        text=True
    )
    
    print(f"Return code: {result.returncode}")
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")
    
import os
os.unlink(f.name)