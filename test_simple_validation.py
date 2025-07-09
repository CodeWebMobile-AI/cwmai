#!/usr/bin/env python3
"""Simple test of code validation"""

from scripts.evolutionary_code_generator_fix import RealCodeEvolutionFix

# Test code
code = '''
import logging
from typing import Dict, Any

class TestClass:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Real implementation
        data = context.get('data', 0)
        result = data * 2
        return {
            'success': True,
            'result': result,
            'processed': True
        }
'''

validator = RealCodeEvolutionFix()

# First check if it detects as real code (not placeholder)
is_placeholder = False
for indicator in ["return {'success': True}", "# TODO", "pass  # TODO"]:
    if indicator in code:
        is_placeholder = True
        break

print(f"Contains placeholder patterns: {is_placeholder}")

# Parse and check for real implementation
import ast
tree = ast.parse(code)

# Find apply method
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'apply':
        print(f"Found apply method with {len(node.body)} statements")
        
        # Check for logic beyond logging/return
        has_logic = False
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                has_logic = True
                print(f"  - Found assignment: {ast.unparse(stmt.targets[0])}")
            elif isinstance(stmt, ast.AugAssign):
                has_logic = True
                print(f"  - Found augmented assignment")
            elif isinstance(stmt, ast.If):
                has_logic = True
                print(f"  - Found if statement")
            elif isinstance(stmt, ast.For):
                has_logic = True
                print(f"  - Found for loop")
                
        print(f"Has real logic: {has_logic}")

# Now test with validator
print("\nRunning validator...")
is_valid, error_msg, test_results = validator.validate_generated_code(
    code, 
    {'name': 'test_class', 'type': 'test', 'code': code}
)

print(f"Valid: {is_valid}")
print(f"Error: {error_msg}")
if test_results:
    print(f"Test results: {test_results}")