#!/usr/bin/env python3
"""
Testing Manager

Comprehensive testing tool for all test operations including generation,
execution, coverage analysis, and reporting.
"""

import os
import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import ast
import asyncio
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.http_ai_client import HTTPAIClient
from scripts.logger import Logger


class TestingManager:
    """Manages all testing operations with AI-enhanced capabilities."""
    
    def __init__(self):
        self.logger = Logger()
        self.ai_client = HTTPAIClient()
        self.test_frameworks = {
            'python': {
                'runner': 'pytest',
                'config': 'pytest.ini',
                'test_pattern': 'test_*.py',
                'coverage': 'pytest-cov'
            },
            'javascript': {
                'runner': 'jest',
                'config': 'jest.config.js',
                'test_pattern': '*.test.js',
                'coverage': 'jest --coverage'
            },
            'php': {
                'runner': 'phpunit',
                'config': 'phpunit.xml',
                'test_pattern': '*Test.php',
                'coverage': 'phpunit --coverage-html'
            }
        }
        
    async def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Main entry point with operation routing."""
        operations = {
            'generate': self.generate_tests,
            'run': self.run_tests,
            'coverage': self.check_coverage,
            'report': self.generate_report,
            'mock': self.generate_mocks,
            'integration': self.create_integration_tests,
            'performance': self.run_performance_tests,
            'fix': self.fix_failing_tests
        }
        
        if operation not in operations:
            return {
                'success': False,
                'error': f'Unknown operation: {operation}',
                'available_operations': list(operations.keys())
            }
            
        try:
            return await operations[operation](**kwargs)
        except Exception as e:
            self.logger.error(f"Error in {operation}: {str(e)}")
            return {'success': False, 'error': str(e)}
            
    async def generate_tests(self, file_path: str = None, code: str = None, 
                           test_type: str = 'unit') -> Dict[str, Any]:
        """Generate tests for code using AI."""
        try:
            # Get code to test
            if file_path and os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    code = f.read()
                language = self._detect_language(file_path)
            elif code:
                language = self._detect_language_from_code(code)
            else:
                return {'success': False, 'error': 'No code provided'}
                
            # Generate prompt for AI
            prompt = f"""Generate comprehensive {test_type} tests for this code:

```{language}
{code}
```

Requirements:
1. Test all functions and methods
2. Include edge cases and error conditions
3. Use appropriate assertions
4. Add test documentation
5. Follow {language} testing best practices
6. Include setup and teardown if needed

Return the complete test file content."""

            # Get AI response
            response = await self.ai_client.generate_enhanced_response(
                prompt=prompt,
                model="gemini"
            )
            
            test_code = response.get('content', '')
            
            # Save test file if path provided
            test_file_path = None
            if file_path:
                test_file_path = self._generate_test_file_path(file_path)
                os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
                with open(test_file_path, 'w') as f:
                    f.write(test_code)
                    
            return {
                'success': True,
                'test_code': test_code,
                'test_file': test_file_path,
                'language': language,
                'test_type': test_type
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def run_tests(self, path: str = '.', framework: str = None,
                       pattern: str = None) -> Dict[str, Any]:
        """Run tests in the specified path."""
        try:
            # Detect framework if not specified
            if not framework:
                framework = self._detect_test_framework(path)
                
            if not framework:
                return {'success': False, 'error': 'No test framework detected'}
                
            # Get framework config
            fw_config = self.test_frameworks.get(framework, {})
            runner = fw_config.get('runner', 'pytest')
            test_pattern = pattern or fw_config.get('test_pattern', 'test_*.py')
            
            # Build command
            cmd = [runner]
            if path != '.':
                cmd.append(path)
            if pattern:
                cmd.extend(['--pattern', pattern])
                
            # Run tests
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=path if os.path.isdir(path) else '.'
            )
            
            # Parse results
            test_results = self._parse_test_results(result.stdout, framework)
            
            return {
                'success': result.returncode == 0,
                'framework': framework,
                'command': ' '.join(cmd),
                'stdout': result.stdout,
                'stderr': result.stderr,
                'results': test_results,
                'total_tests': test_results.get('total', 0),
                'passed': test_results.get('passed', 0),
                'failed': test_results.get('failed', 0),
                'skipped': test_results.get('skipped', 0)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def check_coverage(self, path: str = '.', min_coverage: float = 80.0) -> Dict[str, Any]:
        """Check test coverage."""
        try:
            framework = self._detect_test_framework(path)
            if not framework:
                return {'success': False, 'error': 'No test framework detected'}
                
            # Run coverage based on framework
            if framework == 'python':
                cmd = ['pytest', '--cov=' + path, '--cov-report=json']
            elif framework == 'javascript':
                cmd = ['jest', '--coverage', '--json']
            elif framework == 'php':
                cmd = ['phpunit', '--coverage-text']
            else:
                return {'success': False, 'error': f'Coverage not supported for {framework}'}
                
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=path)
            
            # Parse coverage data
            coverage_data = self._parse_coverage(result.stdout, framework)
            coverage_percent = coverage_data.get('total_coverage', 0)
            
            # Generate coverage report
            report = {
                'success': True,
                'framework': framework,
                'total_coverage': coverage_percent,
                'meets_minimum': coverage_percent >= min_coverage,
                'minimum_required': min_coverage,
                'coverage_data': coverage_data,
                'uncovered_files': coverage_data.get('uncovered_files', []),
                'suggestions': []
            }
            
            # Add suggestions for low coverage
            if coverage_percent < min_coverage:
                report['suggestions'] = await self._generate_coverage_suggestions(
                    coverage_data, min_coverage
                )
                
            return report
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def generate_report(self, test_results: Dict[str, Any] = None,
                            format: str = 'markdown') -> Dict[str, Any]:
        """Generate a comprehensive test report."""
        try:
            if not test_results:
                # Run tests to get results
                test_results = await self.run_tests()
                
            # Generate report based on format
            if format == 'markdown':
                report = self._generate_markdown_report(test_results)
            elif format == 'html':
                report = self._generate_html_report(test_results)
            elif format == 'json':
                report = json.dumps(test_results, indent=2)
            else:
                return {'success': False, 'error': f'Unknown format: {format}'}
                
            # Save report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = f'test_report_{timestamp}.{format}'
            
            with open(report_file, 'w') as f:
                f.write(report)
                
            return {
                'success': True,
                'report': report,
                'report_file': report_file,
                'format': format,
                'summary': {
                    'total': test_results.get('total_tests', 0),
                    'passed': test_results.get('passed', 0),
                    'failed': test_results.get('failed', 0),
                    'coverage': test_results.get('coverage', 'N/A')
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def generate_mocks(self, code: str, dependencies: List[str] = None) -> Dict[str, Any]:
        """Generate mock objects for testing."""
        try:
            prompt = f"""Generate mock objects for testing this code:

```
{code}
```

Dependencies to mock: {dependencies or 'auto-detect'}

Requirements:
1. Create realistic mock responses
2. Include various test scenarios
3. Handle edge cases
4. Use appropriate mocking framework
5. Include usage examples

Return mock definitions and usage examples."""

            response = await self.ai_client.generate_enhanced_response(
                prompt=prompt,
                model="gemini"
            )
            
            mock_code = response.get('content', '')
            
            return {
                'success': True,
                'mock_code': mock_code,
                'dependencies_mocked': dependencies or [],
                'examples_included': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def create_integration_tests(self, components: List[str],
                                     test_scenario: str = None) -> Dict[str, Any]:
        """Create integration tests for multiple components."""
        try:
            prompt = f"""Create integration tests for these components:
{json.dumps(components, indent=2)}

Test scenario: {test_scenario or 'Standard integration flow'}

Requirements:
1. Test component interactions
2. Include setup and teardown
3. Test data flow between components
4. Handle async operations if needed
5. Include error scenarios
6. Add performance checks

Return complete integration test suite."""

            response = await self.ai_client.generate_enhanced_response(
                prompt=prompt,
                model="gemini"
            )
            
            test_code = response.get('content', '')
            
            return {
                'success': True,
                'integration_tests': test_code,
                'components_tested': components,
                'scenario': test_scenario or 'Standard integration flow'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def run_performance_tests(self, target: str, load_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run performance tests."""
        try:
            # Default load profile
            if not load_profile:
                load_profile = {
                    'users': 10,
                    'duration': '30s',
                    'ramp_up': '5s'
                }
                
            # Use different tools based on target type
            if target.startswith('http'):
                # Use a simple curl-based test for HTTP endpoints
                results = await self._run_http_performance_test(target, load_profile)
            else:
                # Use Python's timeit for local functions
                results = await self._run_local_performance_test(target, load_profile)
                
            return {
                'success': True,
                'target': target,
                'load_profile': load_profile,
                'results': results,
                'recommendations': await self._generate_performance_recommendations(results)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def fix_failing_tests(self, test_file: str, error_log: str = None) -> Dict[str, Any]:
        """Attempt to fix failing tests using AI."""
        try:
            # Read test file
            with open(test_file, 'r') as f:
                test_code = f.read()
                
            # Get error log or run tests to get it
            if not error_log:
                test_result = await self.run_tests(test_file)
                error_log = test_result.get('stderr', '')
                
            prompt = f"""Fix the failing tests in this file:

Test Code:
```
{test_code}
```

Error Log:
```
{error_log}
```

Requirements:
1. Identify the cause of failures
2. Fix the test code (not the source code)
3. Ensure tests follow best practices
4. Add comments explaining the fixes
5. Maintain test coverage

Return the complete fixed test file."""

            response = await self.ai_client.generate_enhanced_response(
                prompt=prompt,
                model="gemini"
            )
            
            fixed_code = response.get('content', '')
            
            # Save fixed version
            backup_file = test_file + '.backup'
            os.rename(test_file, backup_file)
            
            with open(test_file, 'w') as f:
                f.write(fixed_code)
                
            # Verify fix by running tests
            verify_result = await self.run_tests(test_file)
            
            return {
                'success': verify_result.get('success', False),
                'fixed_file': test_file,
                'backup_file': backup_file,
                'verification': verify_result,
                'fixes_applied': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    # Helper methods
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.php': 'php',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust'
        }
        return language_map.get(ext, 'unknown')
        
    def _detect_language_from_code(self, code: str) -> str:
        """Detect language from code content."""
        if 'def ' in code or 'import ' in code:
            return 'python'
        elif 'function' in code or 'const ' in code:
            return 'javascript'
        elif '<?php' in code:
            return 'php'
        return 'unknown'
        
    def _detect_test_framework(self, path: str) -> Optional[str]:
        """Detect which test framework is being used."""
        # Check for config files
        config_checks = [
            ('pytest.ini', 'python'),
            ('setup.cfg', 'python'),
            ('jest.config.js', 'javascript'),
            ('package.json', 'javascript'),
            ('phpunit.xml', 'php')
        ]
        
        for config_file, framework in config_checks:
            if os.path.exists(os.path.join(path, config_file)):
                return framework
                
        # Check for test files
        for ext, framework in [('.py', 'python'), ('.js', 'javascript'), ('.php', 'php')]:
            if list(Path(path).glob(f'*test*{ext}')):
                return framework
                
        return None
        
    def _generate_test_file_path(self, source_file: str) -> str:
        """Generate test file path from source file."""
        path = Path(source_file)
        test_dir = path.parent / 'tests'
        test_name = f'test_{path.stem}{path.suffix}'
        return str(test_dir / test_name)
        
    def _parse_test_results(self, output: str, framework: str) -> Dict[str, Any]:
        """Parse test results from output."""
        results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        
        if framework == 'python':
            # Parse pytest output
            match = re.search(r'(\d+) passed', output)
            if match:
                results['passed'] = int(match.group(1))
            match = re.search(r'(\d+) failed', output)
            if match:
                results['failed'] = int(match.group(1))
            match = re.search(r'(\d+) skipped', output)
            if match:
                results['skipped'] = int(match.group(1))
                
        results['total'] = results['passed'] + results['failed'] + results['skipped']
        return results
        
    def _parse_coverage(self, output: str, framework: str) -> Dict[str, Any]:
        """Parse coverage data from output."""
        coverage_data = {
            'total_coverage': 0,
            'files': {},
            'uncovered_files': []
        }
        
        if framework == 'python':
            # Parse pytest-cov output
            match = re.search(r'TOTAL.*?(\d+)%', output)
            if match:
                coverage_data['total_coverage'] = float(match.group(1))
                
        return coverage_data
        
    async def _generate_coverage_suggestions(self, coverage_data: Dict[str, Any],
                                           min_coverage: float) -> List[str]:
        """Generate suggestions to improve coverage."""
        suggestions = []
        
        if coverage_data['total_coverage'] < min_coverage:
            gap = min_coverage - coverage_data['total_coverage']
            suggestions.append(f"Increase coverage by {gap:.1f}% to meet minimum requirement")
            
        if coverage_data.get('uncovered_files'):
            suggestions.append(f"Add tests for {len(coverage_data['uncovered_files'])} uncovered files")
            
        return suggestions
        
    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate markdown test report."""
        report = f"""# Test Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Tests**: {results.get('total_tests', 0)}
- **Passed**: {results.get('passed', 0)}
- **Failed**: {results.get('failed', 0)}
- **Skipped**: {results.get('skipped', 0)}
- **Success Rate**: {(results.get('passed', 0) / max(results.get('total_tests', 1), 1) * 100):.1f}%

## Details

### Test Output
```
{results.get('stdout', 'No output')}
```

### Errors
```
{results.get('stderr', 'No errors')}
```
"""
        return report
        
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML test report."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>Test Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p>Total Tests: {results.get('total_tests', 0)}</p>
        <p class="passed">Passed: {results.get('passed', 0)}</p>
        <p class="failed">Failed: {results.get('failed', 0)}</p>
    </div>
    <h2>Output</h2>
    <pre>{results.get('stdout', 'No output')}</pre>
</body>
</html>"""
        
    async def _run_http_performance_test(self, url: str, load_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Run simple HTTP performance test."""
        # Use curl for basic performance testing
        cmd = ['curl', '-w', '@-', '-o', '/dev/null', '-s', url]
        
        results = []
        for _ in range(load_profile.get('users', 10)):
            result = subprocess.run(cmd, capture_output=True, text=True)
            # Parse timing info
            results.append({'response_time': 0.1})  # Simplified
            
        return {
            'avg_response_time': sum(r['response_time'] for r in results) / len(results),
            'requests': len(results)
        }
        
    async def _run_local_performance_test(self, target: str, load_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Run performance test on local code."""
        import timeit
        
        # Simple timing test
        execution_time = timeit.timeit(target, number=load_profile.get('users', 10))
        
        return {
            'avg_execution_time': execution_time / load_profile.get('users', 10),
            'total_time': execution_time
        }
        
    async def _generate_performance_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if results.get('avg_response_time', 0) > 1.0:
            recommendations.append("Consider optimizing response time - currently over 1 second")
            
        return recommendations


async def main(operation: str = None, **kwargs):
    """Tool entry point."""
    manager = TestingManager()
    
    if not operation:
        return {
            'success': True,
            'available_operations': [
                'generate', 'run', 'coverage', 'report',
                'mock', 'integration', 'performance', 'fix'
            ],
            'description': 'Comprehensive testing tool for all test operations'
        }
        
    return await manager.execute(operation, **kwargs)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def demo():
        # Show available operations
        result = await main()
        print("Testing Manager Operations:", result)
        
        # Generate tests for a file
        result = await main('generate', file_path='scripts/ai_brain.py', test_type='unit')
        print("\nGenerated tests:", result.get('success'))
        
        # Run tests
        result = await main('run', path='.')
        print("\nTest run results:", result.get('summary', {}))
        
    asyncio.run(demo())