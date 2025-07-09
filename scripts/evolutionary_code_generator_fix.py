#!/usr/bin/env python3
"""
Fix for the evolutionary system to generate and test REAL code, not placeholders
"""

import ast
import traceback
import subprocess
import tempfile
import os
from typing import Dict, Any, Tuple, Optional
import logging

class RealCodeEvolutionFix:
    """Ensures the evolution system generates and tests real, working code."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_generated_code(self, code: str, improvement: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict]]:
        """
        Validate that generated code is real implementation, not placeholder.
        Returns: (is_valid, error_message, test_results)
        """
        try:
            # Parse the code
            tree = ast.parse(code)
            
            # Check for placeholder patterns
            placeholder_indicators = [
                "return {'success': True",  # Placeholder return
                "# Implement",  # TODO comment
                "# TODO",
                "pass  # TODO",
                "raise NotImplementedError"
            ]
            
            for indicator in placeholder_indicators:
                if indicator in code:
                    return False, f"Placeholder code detected: contains '{indicator}'", None
                    
            # Find the apply method
            apply_method = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'apply':
                    apply_method = node
                    break
                    
            if not apply_method:
                return False, "No 'apply' method found", None
                
            # Check if apply method has real implementation
            method_body = ast.unparse(apply_method.body) if hasattr(ast, 'unparse') else str(apply_method.body)
            
            # Check for minimal implementation
            if len(apply_method.body) <= 2:  # Just logging and return
                return False, "Apply method has no real implementation", None
                
            # Check if there's actual logic beyond logging
            has_logic = False
            for stmt in apply_method.body:
                if not isinstance(stmt, (ast.Expr, ast.Return)):
                    has_logic = True
                    break
                elif isinstance(stmt, ast.Expr) and not self._is_logging_statement(stmt):
                    has_logic = True
                    break
                    
            if not has_logic:
                return False, "Apply method contains only logging and return statements", None
                
            # Run actual tests
            test_results = self._run_implementation_tests(code, improvement)
            
            if not test_results['all_passed']:
                return False, f"Tests failed: {test_results['error']}", test_results
                
            return True, "Code validated and tested successfully", test_results
            
        except SyntaxError as e:
            return False, f"Syntax error: {e}", None
        except Exception as e:
            return False, f"Validation error: {e}", None
            
    def _is_logging_statement(self, node: ast.Expr) -> bool:
        """Check if a statement is just logging."""
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute):
                return node.value.func.attr in ['info', 'debug', 'warning', 'error']
        return False
        
    def _run_implementation_tests(self, code: str, improvement: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run actual tests on the implementation.
        """
        test_results = {
            'all_passed': False,
            'tests_run': 0,
            'tests_passed': 0,
            'error': None,
            'performance_metrics': {}
        }
        
        try:
            # Create a test file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Write the implementation
                f.write(code)
                f.write('\n\n')
                
                # Write test code
                test_code = self._generate_test_code(improvement)
                f.write(test_code)
                f.flush()
                
                # Run the tests
                import sys
                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    test_results['error'] = f"Test execution failed: {result.stderr}"
                    # Still try to parse output for partial results
                    output_lines = result.stdout.strip().split('\n') if result.stdout else []
                else:
                    output_lines = result.stdout.strip().split('\n') if result.stdout else []
                
                # Parse test output
                for line in output_lines:
                    if line.startswith('TESTS_RUN:'):
                        test_results['tests_run'] = int(line.split(':')[1])
                    elif line.startswith('TESTS_PASSED:'):
                        test_results['tests_passed'] = int(line.split(':')[1])
                    elif line.startswith('METRIC:'):
                        parts = line.split(':')
                        if len(parts) >= 3:
                            metric_name = parts[1]
                            try:
                                metric_value = float(parts[2])
                                test_results['performance_metrics'][metric_name] = metric_value
                            except ValueError:
                                pass
                                
                test_results['all_passed'] = (
                    test_results['tests_run'] > 0 and 
                    test_results['tests_run'] == test_results['tests_passed'] and
                    result.returncode == 0
                )
                
                if not test_results['all_passed'] and not test_results['error']:
                    if test_results['tests_run'] == 0:
                        test_results['error'] = "No tests were run"
                    elif result.returncode != 0:
                        test_results['error'] = f"Tests failed with return code {result.returncode}"
                    else:
                        test_results['error'] = f"Only {test_results['tests_passed']} of {test_results['tests_run']} tests passed"
                        
        except subprocess.TimeoutExpired:
            test_results['error'] = "Test execution timed out"
        except Exception as e:
            test_results['error'] = f"Test execution error: {e}"
        finally:
            # Clean up
            if 'f' in locals() and os.path.exists(f.name):
                os.unlink(f.name)
                
        return test_results
        
    def _generate_test_code(self, improvement: Dict[str, Any]) -> str:
        """Generate test code based on improvement type."""
        
        improvement_type = improvement.get('type', 'generic')
        
        # Base test template
        test_template = '''
# Test the implementation
import time
import sys

def run_tests():
    tests_run = 0
    tests_passed = 0
    
    try:
        # Create instance
        instance = {class_name}()
        tests_run += 1
        tests_passed += 1
        print("✓ Instance creation successful")
    except Exception as e:
        print(f"✗ Instance creation failed: {{e}}")
        return tests_run, tests_passed
        
    # Test apply method
    try:
        start_time = time.time()
        result = instance.apply({{'test': True}})
        execution_time = time.time() - start_time
        tests_run += 1
        
        # Verify result structure
        assert isinstance(result, dict), "Result must be a dictionary"
        assert 'success' in result, "Result must have 'success' key"
        
        # For real implementation, success should depend on actual work done
        if result.get('success') and len(result) <= 2:
            print("✗ Apply method returns success without doing real work")
        else:
            tests_passed += 1
            print("✓ Apply method executed successfully")
            
        print(f"METRIC:execution_time:{{execution_time}}")
        
    except Exception as e:
        print(f"✗ Apply method failed: {{e}}")
        
    {additional_tests}
    
    return tests_run, tests_passed

if __name__ == "__main__":
    tests_run, tests_passed = run_tests()
    print(f"TESTS_RUN:{{tests_run}}")
    print(f"TESTS_PASSED:{{tests_passed}}")
    if tests_run != tests_passed:
        sys.exit(1)
'''
        
        # Get class name from code
        code = improvement.get('code', '')
        class_name = self._extract_class_name(code)
        
        # Add type-specific tests
        additional_tests = self._get_type_specific_tests(improvement_type)
        
        return test_template.format(
            class_name=class_name,
            additional_tests=additional_tests
        )
        
    def _extract_class_name(self, code: str) -> str:
        """Extract the main class name from code."""
        try:
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    return node.name
        except:
            pass
        return "UnknownClass"
        
    def _get_type_specific_tests(self, improvement_type: str) -> str:
        """Get additional tests based on improvement type."""
        
        type_tests = {
            'cache': '''
    # Test cache-specific functionality
    try:
        # Test cache operations - store a value
        store_result = instance.apply({'action': 'set', 'key': 'test_key', 'value': 'test_value'})
        tests_run += 1
        if store_result.get('success'):
            tests_passed += 1
            print("✓ Cache store operation working")
        else:
            print("✗ Cache store operation failed")
            
        # Test cache retrieval
        result = instance.apply({'action': 'get', 'key': 'test_key'})
        tests_run += 1
        if result.get('success') and result.get('data') == 'test_value':
            tests_passed += 1
            print("✓ Cache retrieve operation working")
        else:
            print("✗ Cache retrieve operation failed")
    except Exception as e:
        print(f"✗ Cache operations error: {e}")
''',
            'monitoring': '''
    # Test monitoring functionality
    try:
        # Test metric collection
        metrics_result = instance.apply({'action': 'collect_metrics'})
        tests_run += 1
        
        if 'metrics' in metrics_result and isinstance(metrics_result['metrics'], dict):
            tests_passed += 1
            print("✓ Monitoring metrics collection working")
        else:
            print("✗ Monitoring not properly implemented")
    except:
        print("✗ Monitoring functionality not implemented")
''',
            'security': '''
    # Test security functionality
    try:
        # Test security scan
        scan_result = instance.apply({'action': 'scan', 'target': 'test_system'})
        tests_run += 1
        
        if 'vulnerabilities' in scan_result or 'scan_results' in scan_result:
            tests_passed += 1
            print("✓ Security scanning implemented")
        else:
            print("✗ Security scanning not properly implemented")
    except:
        print("✗ Security functionality not implemented")
'''
        }
        
        return type_tests.get(improvement_type, '''
    # Generic functionality test
    try:
        # Test that the implementation does something meaningful
        result_with_data = instance.apply({'data': 'test_input', 'action': 'process'})
        tests_run += 1
        
        # Check that result contains more than just success flag
        if len(result_with_data) > 2 or 'output' in result_with_data:
            tests_passed += 1
            print("✓ Implementation produces meaningful output")
        else:
            print("✗ Implementation does not produce meaningful output")
    except:
        print("✗ Generic functionality test failed")
''')
    
    def enhance_improvement_generation(self, original_generator):
        """
        Wrap the original improvement generator to ensure real code generation.
        """
        def wrapped_generator(*args, **kwargs):
            # Call original generator
            improvement = original_generator(*args, **kwargs)
            
            if 'code' in improvement:
                # Validate the generated code
                is_valid, error_msg, test_results = self.validate_generated_code(
                    improvement['code'], 
                    improvement
                )
                
                if not is_valid:
                    self.logger.warning(f"Generated placeholder code: {error_msg}")
                    
                    # Force regeneration with explicit requirements
                    improvement['generation_requirements'] = {
                        'must_have_real_implementation': True,
                        'no_placeholders': True,
                        'must_pass_tests': True,
                        'previous_error': error_msg
                    }
                    
                    # Mark as needing regeneration
                    improvement['needs_regeneration'] = True
                    improvement['fitness'] = 0.1  # Low fitness for placeholder
                else:
                    # Real implementation detected
                    improvement['validated'] = True
                    improvement['test_results'] = test_results
                    
                    # Adjust fitness based on test results
                    if test_results:
                        test_score = test_results['tests_passed'] / max(1, test_results['tests_run'])
                        improvement['fitness'] = improvement.get('fitness', 0.5) * test_score
                        
            return improvement
            
        return wrapped_generator
        
    def create_implementation_prompt(self, improvement: Dict[str, Any]) -> str:
        """
        Create a prompt that forces real implementation generation.
        """
        return f"""Generate a COMPLETE, WORKING implementation for: {improvement.get('name', 'Unknown')}

Type: {improvement.get('type', 'enhancement')}
Description: {improvement.get('description', 'No description')}

CRITICAL REQUIREMENTS:
1. Generate REAL, EXECUTABLE code - not placeholders or comments
2. The apply() method must contain actual implementation logic
3. Do NOT just return {{'success': True}} - implement the actual functionality
4. Include proper error handling
5. The code must be testable and produce measurable results

Example of what NOT to generate:
```python
def apply(self, context):
    # TODO: Implement this
    self.logger.info("Applying improvement")
    return {{'success': True}}
```

Example of what TO generate:
```python
def apply(self, context):
    try:
        # Actual implementation
        input_data = context.get('data', {{}})
        processed_data = self._process_data(input_data)
        results = self._analyze_results(processed_data)
        
        return {{
            'success': True,
            'processed_items': len(processed_data),
            'results': results,
            'metrics': self._calculate_metrics(results)
        }}
    except Exception as e:
        return {{'success': False, 'error': str(e)}}
```

Now generate the COMPLETE implementation:
"""


# Module-level function to apply fixes
def apply_real_code_generation_fixes():
    """Apply fixes to ensure real code generation in the evolution system."""
    
    print("🔧 Applying real code generation fixes...")
    
    # Create the fix implementation
    fixer = RealCodeEvolutionFix()
    
    # Return the fixer instance
    return fixer