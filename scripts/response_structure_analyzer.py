#!/usr/bin/env python3
"""
Response Structure Analyzer
===========================

Analyzes method return structures to help generated tools navigate data correctly.
"""

import ast
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import re


class ResponseStructureAnalyzer:
    """Analyzes response structures from methods to prevent navigation errors."""
    
    def __init__(self):
        self.discovered_structures = {}
        
    async def analyze_method_response(self, module_name: str, method_name: str) -> Dict[str, Any]:
        """Analyze what a method returns by examining its implementation and tests."""
        
        structure = {
            'method': method_name,
            'module': module_name,
            'return_type': None,
            'structure': {},
            'examples': []
        }
        
        # Find the method implementation
        module_path = self._find_module_path(module_name)
        if not module_path:
            return structure
            
        # 1. Check return type hints
        return_type = self._get_return_type_hint(module_path, method_name)
        structure['return_type'] = return_type
        
        # 2. Analyze method body for return statements
        return_patterns = self._analyze_return_statements(module_path, method_name)
        structure['structure'] = return_patterns
        
        # 3. Find test files that might show expected structure
        test_examples = self._find_test_examples(module_name, method_name)
        structure['examples'] = test_examples
        
        # 4. Check for docstring examples
        docstring_info = self._extract_docstring_info(module_path, method_name)
        if docstring_info:
            structure['docstring'] = docstring_info
            
        return structure
    
    def _find_module_path(self, module_name: str) -> Optional[str]:
        """Find the path to a module file."""
        # Common patterns for module locations
        possible_paths = [
            f"scripts/{module_name.lower()}.py",
            f"scripts/{module_name}.py",
            f"scripts/{self._camel_to_snake(module_name)}.py"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
                
        # Search for it
        try:
            result = subprocess.run(
                ["find", "scripts", "-name", "*.py", "-exec", "grep", "-l", f"class {module_name}", "{}", "+"],
                capture_output=True, text=True
            )
            if result.stdout:
                return result.stdout.strip().split('\n')[0]
        except:
            pass
            
        return None
    
    def _get_return_type_hint(self, module_path: str, method_name: str) -> Optional[str]:
        """Extract return type hint from method signature."""
        try:
            with open(module_path, 'r') as f:
                content = f.read()
                
            # Look for method with return type
            pattern = rf'def\s+{method_name}\s*\([^)]*\)\s*->\s*([^:]+):'
            match = re.search(pattern, content)
            if match:
                return match.group(1).strip()
                
        except Exception as e:
            print(f"Error getting return type: {e}")
            
        return None
    
    def _analyze_return_statements(self, module_path: str, method_name: str) -> Dict[str, Any]:
        """Analyze return statements in the method."""
        patterns = {
            'dict_keys': set(),
            'nested_structures': [],
            'common_patterns': []
        }
        
        try:
            # Get method body
            result = subprocess.run(
                ["grep", "-A", "50", f"def {method_name}", module_path],
                capture_output=True, text=True
            )
            
            if result.stdout:
                # Find dictionary returns
                dict_returns = re.findall(r'return\s*{([^}]+)}', result.stdout)
                for dict_content in dict_returns:
                    # Extract keys
                    keys = re.findall(r'["\'](\w+)["\']:', dict_content)
                    patterns['dict_keys'].update(keys)
                
                # Find nested access patterns (e.g., result['data']['items'])
                nested = re.findall(r'\[[\'"]([\w_]+)[\'"]\]\[[\'"]([\w_]+)[\'"]\]', result.stdout)
                patterns['nested_structures'].extend(nested)
                
                # Common return patterns
                if 'success' in str(patterns['dict_keys']) and 'result' in str(patterns['dict_keys']):
                    patterns['common_patterns'].append('standard_response')
                    
        except Exception as e:
            print(f"Error analyzing returns: {e}")
            
        patterns['dict_keys'] = list(patterns['dict_keys'])
        return patterns
    
    def _find_test_examples(self, module_name: str, method_name: str) -> List[Dict[str, Any]]:
        """Find test examples showing expected return structure."""
        examples = []
        
        try:
            # Find test files
            snake_name = self._camel_to_snake(module_name)
            test_patterns = [
                f"test_{snake_name}.py",
                f"test_{method_name}.py",
                f"{snake_name}_test.py"
            ]
            
            for pattern in test_patterns:
                result = subprocess.run(
                    ["find", ".", "-name", pattern],
                    capture_output=True, text=True
                )
                
                if result.stdout:
                    test_file = result.stdout.strip()
                    # Look for assertions about the method's return
                    assertion_result = subprocess.run(
                        ["grep", "-A", "3", "-B", "3", method_name, test_file],
                        capture_output=True, text=True
                    )
                    
                    if assertion_result.stdout:
                        # Extract expected structures from assertions
                        # e.g., assert result['health_score'] == 90
                        access_patterns = re.findall(
                            r'result\[[\'"]([\w_]+)[\'"]\](?:\[[\'"]([\w_]+)[\'"]\])?',
                            assertion_result.stdout
                        )
                        
                        for pattern in access_patterns:
                            example = {'access_path': list(pattern)}
                            if pattern not in [e.get('access_path') for e in examples]:
                                examples.append(example)
                                
        except Exception as e:
            print(f"Error finding test examples: {e}")
            
        return examples
    
    def _extract_docstring_info(self, module_path: str, method_name: str) -> Optional[Dict[str, Any]]:
        """Extract information from method docstring."""
        try:
            with open(module_path, 'r') as f:
                content = f.read()
                
            # Parse the AST to get docstrings properly
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == method_name:
                    docstring = ast.get_docstring(node)
                    if docstring:
                        # Look for Returns: section
                        returns_match = re.search(r'Returns?:\s*\n\s*([^:]+?)(?:\n\n|\Z)', docstring, re.MULTILINE)
                        if returns_match:
                            return {
                                'returns_description': returns_match.group(1).strip(),
                                'full_docstring': docstring
                            }
                            
        except Exception as e:
            print(f"Error extracting docstring: {e}")
            
        return None
    
    def _camel_to_snake(self, name: str) -> str:
        """Convert CamelCase to snake_case."""
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    
    def generate_access_helper(self, structure_info: Dict[str, Any]) -> str:
        """Generate helper code for safely accessing nested structures."""
        
        helper_code = """
def safe_get(data, *keys, default=None):
    \"\"\"Safely navigate nested dictionaries.\"\"\"
    result = data
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    return result
"""
        
        # Add specific accessors based on discovered structure
        if structure_info.get('structure', {}).get('dict_keys'):
            keys = structure_info['structure']['dict_keys']
            if 'health_score' in str(structure_info):
                helper_code += """
# Based on discovered structure:
# health_score = safe_get(result, 'health_metrics', 'health_score')
"""
        
        return helper_code