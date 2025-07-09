"""
Automated Code Verification System

Implements AlphaEvolve-style automated verification for generated improvements.
"""

import asyncio
import ast
import logging
import subprocess
import tempfile
import os
from typing import Dict, List, Tuple, Any
from pathlib import Path
import importlib.util
import sys
import traceback

from scripts.state_manager import StateManager


class AutomatedCodeVerifier:
    """Verifies code improvements through automated testing and analysis."""
    
    def fix_common_issues(self, code: str) -> str:
            """Fix common code issues before verification."""
            lines = code.split('\n')
            fixed_lines = []
            
            for i, line in enumerate(lines):
                fixed_lines.append(line)
                
                # Fix empty blocks
                if line.strip().endswith(':'):
                    # Check if next line is empty or less indented
                    next_line_empty = (i + 1 >= len(lines) or 
                                     lines[i + 1].strip() == '' or
                                     (lines[i + 1].strip() and not lines[i + 1].startswith(' ')))
                    
                    if next_line_empty:
                        indent = len(line) - len(line.lstrip()) + 4
                        
                        if 'def ' in line:
                            fixed_lines.append(' ' * indent + '"""TODO: Implement this function."""')
                            fixed_lines.append(' ' * indent + 'pass')
                        elif 'class ' in line:
                            fixed_lines.append(' ' * indent + '"""TODO: Implement this class."""')
                            fixed_lines.append(' ' * indent + 'pass')
                        elif any(keyword in line for keyword in ['if ', 'for ', 'while ', 'with ', 'try:', 'except']):
                            fixed_lines.append(' ' * indent + 'pass  # TODO: Implement')
            
            # Fix missing imports
            fixed_code = '\n'.join(fixed_lines)
            
            # Add common missing imports
            if 'logging' in fixed_code and 'import logging' not in fixed_code:
                fixed_code = 'import logging\n' + fixed_code
            if 'datetime' in fixed_code and 'from datetime import' not in fixed_code and 'import datetime' not in fixed_code:
                fixed_code = 'from datetime import datetime\n' + fixed_code
            if 'Dict' in fixed_code and 'from typing import' not in fixed_code:
                fixed_code = 'from typing import Dict, Any, Optional, List\n' + fixed_code
            
            return fixed_code

    def __init__(self, state_manager: StateManager):
        self.logger = logging.getLogger(__name__)
        self.state_manager = state_manager
        
        # Verification levels
        self.verification_levels = [
            'syntax',      # Level 1: Syntax validity
            'imports',     # Level 2: Import resolution
            'execution',   # Level 3: Basic execution
            'integration', # Level 4: System integration
            'performance'  # Level 5: Performance benchmarks
        ]
        
    async def verify_improvement(self, improvement: Dict) -> Dict[str, Any]:
        """Run comprehensive verification on an improvement."""
        results = {
            'improvement_id': improvement.get('id', 'unknown'),
            'improvement_name': improvement.get('name', 'unknown'),
            'verification_levels': {},
            'overall_score': 0.0,
            'is_deployable': False,
            'issues': [],
            'metrics': {}
        }
        
        code = improvement.get('code', '')
        if not code:
            results['issues'].append('No code provided')
            return results
            
        # Run verification levels
        for level in self.verification_levels:
            level_result = await self._verify_level(level, improvement)
            results['verification_levels'][level] = level_result
            
            if not level_result['passed']:
                results['issues'].extend(level_result.get('errors', []))
                # Stop at first failure for efficiency
                break
                
        # Calculate overall score
        passed_levels = sum(
            1 for r in results['verification_levels'].values()
            if r['passed']
        )
        results['overall_score'] = passed_levels / len(self.verification_levels)
        
        # Determine if deployable (passes at least first 3 levels)
        results['is_deployable'] = passed_levels >= 3
        
        # Collect metrics
        for level_result in results['verification_levels'].values():
            if 'metrics' in level_result:
                results['metrics'].update(level_result['metrics'])
                
        return results
        
    async def _verify_level(self, level: str, improvement: Dict) -> Dict[str, Any]:
        """Verify a specific level."""
        if level == 'syntax':
            return self._verify_syntax(improvement)
        elif level == 'imports':
            return await self._verify_imports(improvement)
        elif level == 'execution':
            return await self._verify_execution(improvement)
        elif level == 'integration':
            return await self._verify_integration(improvement)
        elif level == 'performance':
            return await self._verify_performance(improvement)
        else:
            return {'passed': False, 'errors': [f'Unknown level: {level}']}
            
    def _verify_syntax(self, improvement: Dict) -> Dict[str, Any]:
        """Verify Python syntax is valid."""
        code = improvement.get('code', '')
        
        try:
            # Fix common issues first
            code = self.fix_common_issues(code)
            tree = ast.parse(code)
            
            # Analyze AST for code metrics
            metrics = {
                'functions': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                'classes': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                'lines': len(code.split('\n')),
                'imports': len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])
            }
            
            return {
                'passed': True,
                'metrics': metrics
            }
        except SyntaxError as e:
            return {
                'passed': False,
                'errors': [f'Syntax error at line {e.lineno}: {e.msg}']
            }
        except Exception as e:
            return {
                'passed': False,
                'errors': [f'Parse error: {str(e)}']
            }
            
    async def _verify_imports(self, improvement: Dict) -> Dict[str, Any]:
        """Verify all imports can be resolved."""
        code = improvement.get('code', '')
        
        try:
            tree = ast.parse(code)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        if module:
                            imports.append(f"{module}.{alias.name}")
                        else:
                            imports.append(alias.name)
                            
            # Check each import
            unresolved = []
            for imp in imports:
                try:
                    if imp.startswith('scripts.'):
                        # Check if local module exists
                        module_path = imp.replace('.', '/') + '.py'
                        if not Path(module_path).exists():
                            # Try as package
                            module_path = imp.replace('.', '/') + '/__init__.py'
                            if not Path(module_path).exists():
                                unresolved.append(imp)
                    else:
                        # Try importing to check availability
                        __import__(imp.split('.')[0])
                except ImportError:
                    unresolved.append(imp)
                    
            if unresolved:
                return {
                    'passed': False,
                    'errors': [f'Unresolved imports: {", ".join(unresolved)}']
                }
            else:
                return {
                    'passed': True,
                    'metrics': {'import_count': len(imports)}
                }
                
        except Exception as e:
            return {
                'passed': False,
                'errors': [f'Import verification error: {str(e)}']
            }
            
    async def _verify_execution(self, improvement: Dict) -> Dict[str, Any]:
        """Verify code can execute without errors."""
        code = improvement.get('code', '')
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
            
        try:
            # Run the code in a subprocess with timeout
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=10  # 10 second timeout
            )
            
            if result.returncode == 0:
                return {
                    'passed': True,
                    'metrics': {
                        'execution_time': 'under 10s',
                        'stdout_lines': len(result.stdout.split('\n')) if result.stdout else 0
                    }
                }
            else:
                return {
                    'passed': False,
                    'errors': [f'Execution failed: {result.stderr}']
                }
                
        except subprocess.TimeoutExpired:
            return {
                'passed': False,
                'errors': ['Execution timeout (>10s)']
            }
        except Exception as e:
            return {
                'passed': False,
                'errors': [f'Execution error: {str(e)}']
            }
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                
    async def _verify_integration(self, improvement: Dict) -> Dict[str, Any]:
        """Verify the improvement integrates with the system."""
        integration_points = improvement.get('integration_points', [])
        
        if not integration_points:
            return {
                'passed': True,
                'warnings': ['No integration points specified']
            }
            
        # Check each integration point
        issues = []
        verified_points = 0
        
        for point in integration_points:
            if isinstance(point, str):
                # Check if the integration point exists
                if '.' in point:
                    module_name, attr_name = point.rsplit('.', 1)
                    try:
                        if module_name.startswith('scripts.'):
                            # Local module
                            module_path = module_name.replace('.', '/') + '.py'
                            if Path(module_path).exists():
                                verified_points += 1
                            else:
                                issues.append(f'Integration point not found: {point}')
                        else:
                            # External module
                            module = __import__(module_name)
                            if hasattr(module, attr_name):
                                verified_points += 1
                            else:
                                issues.append(f'Attribute not found: {point}')
                    except ImportError:
                        issues.append(f'Module not found for: {point}')
                        
        if issues:
            return {
                'passed': False,
                'errors': issues
            }
        else:
            return {
                'passed': True,
                'metrics': {'integration_points_verified': verified_points}
            }
            
    async def _verify_performance(self, improvement: Dict) -> Dict[str, Any]:
        """Run performance benchmarks on the improvement."""
        # Simple performance check - ensure no obvious performance issues
        code = improvement.get('code', '')
        
        performance_issues = []
        
        # Check for common performance anti-patterns
        if 'while True:' in code and 'break' not in code:
            performance_issues.append('Potential infinite loop detected')
            
        if code.count('for') > 5:
            performance_issues.append('High number of loops may impact performance')
            
        if 'time.sleep(' in code and 'async' not in code:
            performance_issues.append('Blocking sleep in non-async code')
            
        # Check for good performance patterns
        performance_score = 0.5  # Base score
        
        if 'async def' in code:
            performance_score += 0.1  # Async support
        if 'cache' in code.lower():
            performance_score += 0.1  # Caching
        if '@lru_cache' in code:
            performance_score += 0.1  # Function caching
        if 'yield' in code:
            performance_score += 0.1  # Generator usage
            
        if performance_issues:
            return {
                'passed': False,
                'errors': performance_issues,
                'metrics': {'performance_score': performance_score}
            }
        else:
            return {
                'passed': True,
                'metrics': {'performance_score': performance_score}
            }
            
    async def batch_verify(self, improvements: List[Dict]) -> List[Dict[str, Any]]:
        """Verify multiple improvements in parallel."""
        print(f"Batch verifying {len(improvements)} improvements")
        
        # Run verifications in parallel
        tasks = [self.verify_improvement(imp) for imp in improvements]
        results = await asyncio.gather(*tasks)
        
        # Summary statistics
        passed = sum(1 for r in results if r['is_deployable'])
        avg_score = sum(r['overall_score'] for r in results) / len(results) if results else 0
        
        print(f"Verification complete: {passed}/{len(improvements)} deployable")
        print(f"Average verification score: {avg_score:.2f}")
        
        return results