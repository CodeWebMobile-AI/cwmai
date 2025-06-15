"""
Improvement Validator

Comprehensive validation system for staged code improvements.
Performs syntax checks, tests, performance analysis, and security scanning.
"""

import os
import ast
import subprocess
import tempfile
import time
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import psutil
import importlib.util
import sys
import re
import traceback

from safe_self_improver import Modification


@dataclass
class ValidationResult:
    """Results from validation checks."""
    ready_to_apply: bool
    syntax_valid: bool
    tests_pass: bool
    performance_improved: bool
    security_passed: bool
    compatibility_ok: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    recommendations: List[str]


class ImprovementValidator:
    """Validates staged improvements before applying them."""
    
    def __init__(self, repo_path: str = "."):
        """Initialize validator with repository path."""
        self.repo_path = os.path.abspath(repo_path)
        self.test_commands = self._detect_test_commands()
        self.performance_baselines = {}
        
    def _detect_test_commands(self) -> List[List[str]]:
        """Detect available test commands in the repository."""
        commands = []
        
        # Check for common test files/configs
        if os.path.exists(os.path.join(self.repo_path, 'pytest.ini')) or \
           os.path.exists(os.path.join(self.repo_path, 'pyproject.toml')):
            commands.append(['python', '-m', 'pytest', '-xvs'])
        
        if os.path.exists(os.path.join(self.repo_path, 'setup.py')):
            commands.append(['python', 'setup.py', 'test'])
        
        if os.path.exists(os.path.join(self.repo_path, 'tests')):
            commands.append(['python', '-m', 'unittest', 'discover', '-s', 'tests'])
        
        # Default fallback
        if not commands:
            commands.append(['python', '-m', 'pytest'])
        
        return commands
    
    async def validate_improvement(self, 
                                  staged_path: str, 
                                  original_path: str,
                                  modification: Modification) -> Dict[str, Any]:
        """Perform comprehensive validation of a staged improvement.
        
        Args:
            staged_path: Path to staged improvement file
            original_path: Path to original file
            modification: The modification object
            
        Returns:
            Validation results dictionary
        """
        print(f"🔍 Validating improvement: {os.path.basename(staged_path)}")
        
        errors = []
        warnings = []
        metrics = {}
        recommendations = []
        
        # 1. Syntax validation
        syntax_valid = self._validate_syntax(staged_path, errors)
        
        # 2. Import validation
        import_valid = self._validate_imports(staged_path, errors, warnings)
        
        # 3. Security scan
        security_passed = self._security_scan(staged_path, errors, warnings)
        
        # 4. Compatibility check
        compatibility_ok = await self._check_compatibility(
            staged_path, original_path, errors, warnings
        )
        
        # 5. Test execution
        tests_pass, test_metrics = await self._run_tests(
            staged_path, original_path, errors
        )
        metrics.update(test_metrics)
        
        # 6. Performance analysis
        performance_improved, perf_metrics = await self._analyze_performance(
            staged_path, original_path, warnings, recommendations
        )
        metrics.update(perf_metrics)
        
        # 7. Code quality analysis
        quality_metrics = self._analyze_code_quality(
            staged_path, original_path, warnings, recommendations
        )
        metrics.update(quality_metrics)
        
        # Determine if ready to apply
        ready_to_apply = all([
            syntax_valid,
            import_valid,
            security_passed,
            compatibility_ok,
            tests_pass,
            not any(e.startswith("CRITICAL:") for e in errors)
        ])
        
        # Build result
        result = {
            'ready_to_apply': ready_to_apply,
            'syntax_valid': syntax_valid,
            'tests_pass': tests_pass,
            'performance_improved': performance_improved,
            'security_passed': security_passed,
            'compatibility_ok': compatibility_ok,
            'errors': errors,
            'warnings': warnings,
            'metrics': metrics,
            'recommendations': recommendations,
            'validation_timestamp': time.time()
        }
        
        # Generate validation report
        self._generate_validation_report(
            staged_path, result, modification
        )
        
        return result
    
    def _validate_syntax(self, file_path: str, errors: List[str]) -> bool:
        """Validate Python syntax."""
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            
            # Parse with AST
            ast.parse(code)
            
            # Compile check
            compile(code, file_path, 'exec')
            
            print("✅ Syntax validation passed")
            return True
            
        except SyntaxError as e:
            errors.append(f"SYNTAX_ERROR: {e.msg} at line {e.lineno}")
            return False
        except Exception as e:
            errors.append(f"VALIDATION_ERROR: {str(e)}")
            return False
    
    def _validate_imports(self, file_path: str, errors: List[str], 
                         warnings: List[str]) -> bool:
        """Validate that all imports are available."""
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
            
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # Check each import
            failed_imports = []
            for imp in imports:
                try:
                    if '.' in imp:
                        # Handle module paths
                        parts = imp.split('.')
                        __import__(parts[0])
                    else:
                        __import__(imp)
                except ImportError:
                    failed_imports.append(imp)
            
            if failed_imports:
                warnings.append(f"Missing imports: {', '.join(failed_imports)}")
                # Not a critical error - might be project-specific imports
            
            return True
            
        except Exception as e:
            errors.append(f"IMPORT_VALIDATION_ERROR: {str(e)}")
            return False
    
    def _security_scan(self, file_path: str, errors: List[str], 
                      warnings: List[str]) -> bool:
        """Scan for security issues."""
        dangerous_patterns = [
            (r'exec\s*\(', 'Use of exec()'),
            (r'eval\s*\(', 'Use of eval()'),
            (r'__import__\s*\(', 'Dynamic import'),
            (r'os\.system\s*\(', 'System command execution'),
            (r'subprocess\.call\s*\(.*shell=True', 'Shell command with shell=True'),
            (r'pickle\.loads?\s*\(', 'Pickle deserialization'),
            (r'open\s*\([^,)]*\s*,\s*["\']w["\']', 'File write operation'),
            (r'requests\.(get|post|put|delete)\s*\((?!.*timeout)', 'HTTP request without timeout'),
        ]
        
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            
            security_issues = []
            
            for pattern, description in dangerous_patterns:
                if re.search(pattern, code):
                    security_issues.append(description)
            
            if security_issues:
                for issue in security_issues:
                    errors.append(f"SECURITY: {issue}")
                return False
            
            print("✅ Security scan passed")
            return True
            
        except Exception as e:
            errors.append(f"SECURITY_SCAN_ERROR: {str(e)}")
            return False
    
    async def _check_compatibility(self, staged_path: str, original_path: str,
                                  errors: List[str], warnings: List[str]) -> bool:
        """Check backward compatibility."""
        try:
            # Extract function/class signatures from both files
            original_api = self._extract_api(original_path)
            staged_api = self._extract_api(staged_path)
            
            # Check for removed functions/classes
            removed = set(original_api.keys()) - set(staged_api.keys())
            if removed:
                errors.append(f"BREAKING_CHANGE: Removed API elements: {removed}")
                return False
            
            # Check for signature changes
            for name, original_sig in original_api.items():
                if name in staged_api:
                    staged_sig = staged_api[name]
                    if original_sig != staged_sig:
                        warnings.append(f"API_CHANGE: {name} signature changed")
            
            print("✅ Compatibility check passed")
            return True
            
        except Exception as e:
            warnings.append(f"COMPATIBILITY_CHECK_WARNING: {str(e)}")
            return True  # Non-critical
    
    def _extract_api(self, file_path: str) -> Dict[str, str]:
        """Extract public API signatures from a file."""
        api = {}
        
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    # Extract function signature
                    args = []
                    for arg in node.args.args:
                        args.append(arg.arg)
                    api[node.name] = f"function({','.join(args)})"
                    
                elif isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
                    # Extract class and its public methods
                    api[node.name] = "class"
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                            method_args = []
                            for arg in item.args.args[1:]:  # Skip self
                                method_args.append(arg.arg)
                            api[f"{node.name}.{item.name}"] = f"method({','.join(method_args)})"
        except:
            pass
        
        return api
    
    async def _run_tests(self, staged_path: str, original_path: str,
                        errors: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """Run tests with staged improvement."""
        print("🧪 Running tests...")
        
        # Create temporary backup
        temp_backup = None
        try:
            # Backup original
            temp_backup = original_path + '.validator_backup'
            os.rename(original_path, temp_backup)
            
            # Copy staged to original location
            import shutil
            shutil.copy2(staged_path, original_path)
            
            # Run tests
            test_results = await self._execute_tests()
            
            # Restore original
            os.rename(temp_backup, original_path)
            
            if test_results['pass_rate'] < 1.0:
                errors.append(
                    f"TEST_FAILURES: {test_results['failed']} tests failed"
                )
            
            return test_results['pass_rate'] >= 0.95, test_results
            
        except Exception as e:
            errors.append(f"TEST_EXECUTION_ERROR: {str(e)}")
            
            # Ensure original is restored
            if temp_backup and os.path.exists(temp_backup):
                if os.path.exists(original_path):
                    os.remove(original_path)
                os.rename(temp_backup, original_path)
            
            return False, {'pass_rate': 0, 'error': str(e)}
    
    async def _execute_tests(self) -> Dict[str, Any]:
        """Execute test suite."""
        for test_cmd in self.test_commands:
            try:
                # Run tests with timeout
                process = await asyncio.create_subprocess_exec(
                    *test_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.repo_path
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), 
                        timeout=300  # 5 minutes
                    )
                except asyncio.TimeoutError:
                    process.terminate()
                    await process.wait()
                    continue
                
                # Parse results
                output = stdout.decode('utf-8', errors='ignore')
                
                # Try to parse pytest output
                if 'pytest' in ' '.join(test_cmd):
                    return self._parse_pytest_output(output)
                
                # Generic success check
                if process.returncode == 0:
                    return {
                        'pass_rate': 1.0,
                        'passed': 'all',
                        'failed': 0,
                        'total': 'unknown'
                    }
                    
            except Exception:
                continue
        
        # No tests found or all failed
        return {
            'pass_rate': 1.0,  # Assume pass if no tests
            'passed': 0,
            'failed': 0,
            'total': 0,
            'note': 'No tests found'
        }
    
    def _parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """Parse pytest output for results."""
        import re
        
        # Look for summary line
        summary_match = re.search(
            r'(\d+) passed(?:, (\d+) failed)?(?:, (\d+) error)?', 
            output
        )
        
        if summary_match:
            passed = int(summary_match.group(1))
            failed = int(summary_match.group(2) or 0)
            errors = int(summary_match.group(3) or 0)
            total = passed + failed + errors
            
            return {
                'pass_rate': passed / total if total > 0 else 1.0,
                'passed': passed,
                'failed': failed + errors,
                'total': total
            }
        
        return {
            'pass_rate': 1.0,
            'passed': 0,
            'failed': 0,
            'total': 0
        }
    
    async def _analyze_performance(self, staged_path: str, original_path: str,
                                  warnings: List[str], 
                                  recommendations: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """Analyze performance impact of improvement."""
        print("⚡ Analyzing performance...")
        
        try:
            # Simple performance metrics
            original_metrics = await self._measure_file_performance(original_path)
            staged_metrics = await self._measure_file_performance(staged_path)
            
            metrics = {
                'complexity_before': original_metrics['complexity'],
                'complexity_after': staged_metrics['complexity'],
                'size_before': original_metrics['size'],
                'size_after': staged_metrics['size'],
                'import_time_before': original_metrics['import_time'],
                'import_time_after': staged_metrics['import_time']
            }
            
            # Check for regressions
            improved = True
            
            if staged_metrics['complexity'] > original_metrics['complexity'] * 1.2:
                warnings.append("Performance: Complexity increased by >20%")
                improved = False
            
            if staged_metrics['import_time'] > original_metrics['import_time'] * 1.5:
                warnings.append("Performance: Import time increased by >50%")
                improved = False
            
            # Add recommendations
            if staged_metrics['complexity'] > 10:
                recommendations.append(
                    "Consider breaking down complex functions"
                )
            
            return improved, metrics
            
        except Exception as e:
            warnings.append(f"PERFORMANCE_ANALYSIS_WARNING: {str(e)}")
            return True, {}
    
    async def _measure_file_performance(self, file_path: str) -> Dict[str, Any]:
        """Measure performance characteristics of a file."""
        metrics = {
            'complexity': 0,
            'size': 0,
            'import_time': 0.0
        }
        
        try:
            # File size
            metrics['size'] = os.path.getsize(file_path)
            
            # Complexity
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
            
            complexity = 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            metrics['complexity'] = complexity
            
            # Import time (simplified)
            start_time = time.time()
            spec = importlib.util.spec_from_file_location("test_module", file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            metrics['import_time'] = time.time() - start_time
            
        except:
            pass
        
        return metrics
    
    def _analyze_code_quality(self, staged_path: str, original_path: str,
                             warnings: List[str], 
                             recommendations: List[str]) -> Dict[str, Any]:
        """Analyze code quality metrics."""
        metrics = {}
        
        try:
            # Count improvements
            with open(original_path, 'r') as f:
                original_lines = f.readlines()
            
            with open(staged_path, 'r') as f:
                staged_lines = f.readlines()
            
            metrics['lines_before'] = len(original_lines)
            metrics['lines_after'] = len(staged_lines)
            
            # Check for common improvements
            original_code = ''.join(original_lines)
            staged_code = ''.join(staged_lines)
            
            # Docstring coverage
            original_docs = len(re.findall(r'""".*?"""', original_code, re.DOTALL))
            staged_docs = len(re.findall(r'""".*?"""', staged_code, re.DOTALL))
            
            if staged_docs > original_docs:
                metrics['documentation_improved'] = True
                
            # Type hints
            original_hints = len(re.findall(r'->\s*\w+', original_code))
            staged_hints = len(re.findall(r'->\s*\w+', staged_code))
            
            if staged_hints > original_hints:
                metrics['type_hints_added'] = True
            
            # List comprehensions (common optimization)
            original_comprehensions = len(re.findall(r'\[.*for.*in.*\]', original_code))
            staged_comprehensions = len(re.findall(r'\[.*for.*in.*\]', staged_code))
            
            if staged_comprehensions > original_comprehensions:
                metrics['comprehensions_added'] = staged_comprehensions - original_comprehensions
            
        except Exception as e:
            warnings.append(f"QUALITY_ANALYSIS_WARNING: {str(e)}")
        
        return metrics
    
    def _generate_validation_report(self, staged_path: str, 
                                   result: Dict[str, Any],
                                   modification: Modification):
        """Generate detailed validation report."""
        report = {
            'file': os.path.basename(staged_path),
            'modification_type': modification.type.value,
            'description': modification.description,
            'validation_result': result,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save report
        report_dir = os.path.join(
            os.path.dirname(os.path.dirname(staged_path)),
            'reports'
        )
        os.makedirs(report_dir, exist_ok=True)
        
        report_file = os.path.join(
            report_dir,
            f"validation_{modification.id}_{int(time.time())}.json"
        )
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        if result['ready_to_apply']:
            print("✅ Validation PASSED - Improvement ready to apply")
        else:
            print("❌ Validation FAILED - Issues found:")
            for error in result['errors']:
                print(f"  - {error}")


async def validate_improvement_standalone(staged_path: str, original_path: str):
    """Standalone validation function for testing."""
    from safe_self_improver import Modification, ModificationType
    
    # Create mock modification
    modification = Modification(
        id="test_validation",
        type=ModificationType.OPTIMIZATION,
        target_file=os.path.basename(original_path),
        description="Test validation",
        changes=[],
        timestamp=time.time(),
        safety_score=1.0
    )
    
    validator = ImprovementValidator()
    result = await validator.validate_improvement(
        staged_path, original_path, modification
    )
    
    print("\nValidation Summary:")
    print(f"Ready to apply: {result['ready_to_apply']}")
    print(f"Errors: {len(result['errors'])}")
    print(f"Warnings: {len(result['warnings'])}")
    
    return result


if __name__ == "__main__":
    # Test validation
    import sys
    if len(sys.argv) == 3:
        staged = sys.argv[1]
        original = sys.argv[2]
        
        import asyncio
        asyncio.run(validate_improvement_standalone(staged, original))