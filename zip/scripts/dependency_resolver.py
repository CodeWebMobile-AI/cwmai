#!/usr/bin/env python3
"""
Dependency Resolver - Smart Import Resolution System
Analyzes and fixes import dependencies for auto-generated tools
"""

import ast
import os
import re
import sys
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
import importlib.util
import json

class DependencyResolver:
    """Smart dependency analyzer and import fixer for generated tools"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.module_cache = {}
        self.import_mappings = self._build_import_mappings()
        
    def _build_import_mappings(self) -> Dict[str, str]:
        """Build a mapping of module names to their actual paths"""
        mappings = {}
        scripts_dir = self.project_root / "scripts"
        
        if scripts_dir.exists():
            for py_file in scripts_dir.glob("*.py"):
                module_name = py_file.stem
                mappings[module_name] = f"scripts.{module_name}"
                
        # Add custom_tools directory
        custom_tools_dir = scripts_dir / "custom_tools"
        if custom_tools_dir.exists():
            for py_file in custom_tools_dir.glob("*.py"):
                module_name = py_file.stem
                mappings[module_name] = f"scripts.custom_tools.{module_name}"
                
        return mappings
        
    def analyze_imports(self, code: str) -> List[str]:
        """Detect all imports required by a piece of code"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
            
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
                        
        return list(set(imports))
        
    def analyze_undefined_names(self, code: str) -> Set[str]:
        """Find all undefined names in the code that might need imports"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
            
        defined_names = set()
        used_names = set()
        
        # Collect defined names
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined_names.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined_names.add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    defined_names.add(name.split('.')[0])
                    
        # Collect used names
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                used_names.add(node.value.id)
                
        # Python builtins
        builtins = set(dir(__builtins__))
        
        # Find undefined names
        undefined = used_names - defined_names - builtins
        
        return undefined
        
    def suggest_imports(self, undefined_names: Set[str]) -> Dict[str, str]:
        """Suggest imports for undefined names based on project structure"""
        suggestions = {}
        
        # Common standard library imports
        stdlib_mappings = {
            'datetime': 'datetime',
            'json': 'json',
            'os': 'os',
            'sys': 'sys',
            'Path': 'pathlib.Path',
            'List': 'typing.List',
            'Dict': 'typing.Dict',
            'Set': 'typing.Set',
            'Optional': 'typing.Optional',
            'Any': 'typing.Any',
            'Tuple': 'typing.Tuple',
            'Union': 'typing.Union',
            'asyncio': 'asyncio',
            'logging': 'logging',
            're': 're',
            'time': 'time',
            'uuid': 'uuid',
            'redis': 'redis',
            'Redis': 'redis.Redis',
        }
        
        for name in undefined_names:
            # Check standard library
            if name in stdlib_mappings:
                suggestions[name] = stdlib_mappings[name]
            # Check project modules
            elif name in self.import_mappings:
                suggestions[name] = self.import_mappings[name]
            # Check if it's a class from a known module
            else:
                for module_name, module_path in self.import_mappings.items():
                    if self._module_contains_name(module_path, name):
                        suggestions[name] = f"{module_path}.{name}"
                        break
                        
        return suggestions
        
    def _module_contains_name(self, module_path: str, name: str) -> bool:
        """Check if a module contains a specific name"""
        try:
            # Convert module path to file path
            file_path = self.project_root / module_path.replace('.', '/') + '.py'
            if not file_path.exists():
                return False
                
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Simple check - could be enhanced with AST parsing
            return re.search(rf'\b(class|def)\s+{name}\b', content) is not None
        except:
            return False
            
    def fix_import_paths(self, code: str) -> str:
        """Correct import statements based on actual project structure"""
        lines = code.split('\n')
        fixed_lines = []
        imports_section = []
        
        for line in lines:
            # Fix import statements
            if line.strip().startswith(('import ', 'from ')):
                fixed_line = self._fix_import_line(line)
                imports_section.append(fixed_line)
            else:
                fixed_lines.append(line)
                
        # Add missing imports
        undefined = self.analyze_undefined_names('\n'.join(fixed_lines))
        suggestions = self.suggest_imports(undefined)
        
        for name, import_path in suggestions.items():
            if '.' in import_path:
                module, attr = import_path.rsplit('.', 1)
                import_line = f"from {module} import {attr}"
            else:
                import_line = f"import {import_path}"
                
            if import_line not in imports_section:
                imports_section.append(import_line)
                
        # Combine and sort imports
        if imports_section:
            # Group imports: standard library, third-party, local
            stdlib_imports = []
            third_party_imports = []
            local_imports = []
            
            for imp in imports_section:
                if self._is_stdlib_import(imp):
                    stdlib_imports.append(imp)
                elif self._is_local_import(imp):
                    local_imports.append(imp)
                else:
                    third_party_imports.append(imp)
                    
            # Sort each group
            all_imports = []
            if stdlib_imports:
                all_imports.extend(sorted(set(stdlib_imports)))
            if third_party_imports:
                if stdlib_imports:
                    all_imports.append('')
                all_imports.extend(sorted(set(third_party_imports)))
            if local_imports:
                if stdlib_imports or third_party_imports:
                    all_imports.append('')
                all_imports.extend(sorted(set(local_imports)))
                
            # Add imports at the beginning
            result_lines = all_imports + ['', ''] + fixed_lines
        else:
            result_lines = fixed_lines
            
        return '\n'.join(result_lines)
        
    def _fix_import_line(self, line: str) -> str:
        """Fix a single import line"""
        # Handle relative imports
        if 'from .' in line:
            # Convert relative to absolute
            match = re.match(r'from\s+\.+(\w+)?', line)
            if match:
                module = match.group(1) or ''
                if module in self.import_mappings:
                    return line.replace(f'.{module}', self.import_mappings[module])
                    
        # Fix incorrect module paths
        for module_name, correct_path in self.import_mappings.items():
            # Fix "import module_name" -> "import scripts.module_name"
            if re.match(rf'^import\s+{module_name}$', line.strip()):
                return f"import {correct_path}"
            # Fix "from module_name import ..." -> "from scripts.module_name import ..."
            if re.match(rf'^from\s+{module_name}\s+import', line.strip()):
                return line.replace(f"from {module_name}", f"from {correct_path}")
                
        return line
        
    def _is_stdlib_import(self, import_line: str) -> bool:
        """Check if an import is from the standard library"""
        stdlib_modules = {
            'os', 'sys', 'json', 'datetime', 'time', 're', 'logging',
            'asyncio', 'typing', 'pathlib', 'collections', 'itertools',
            'functools', 'contextlib', 'uuid', 'random', 'math', 'copy'
        }
        
        match = re.match(r'(?:from\s+)?(\w+)', import_line)
        if match:
            module = match.group(1)
            return module in stdlib_modules
        return False
        
    def _is_local_import(self, import_line: str) -> bool:
        """Check if an import is from the local project"""
        return 'scripts.' in import_line or 'custom_tools' in import_line
        
    def generate_fallbacks(self, missing_modules: List[str]) -> Dict[str, str]:
        """Generate fallback implementations for missing modules"""
        fallbacks = {}
        
        for module in missing_modules:
            # Generate a mock implementation
            fallback_code = f"""
# Fallback implementation for {module}
class Mock{module.capitalize()}:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        
    def __getattr__(self, name):
        def method(*args, **kwargs):
            print(f"Mock {module}.{name} called with args={args}, kwargs={kwargs}")
            return None
        return method

# Create default instance
{module} = Mock{module.capitalize()}()
"""
            fallbacks[module] = fallback_code
            
        return fallbacks
        
    def validate_imports(self, code: str) -> Tuple[bool, List[str]]:
        """Validate that all imports in the code can be resolved"""
        errors = []
        imports = self.analyze_imports(code)
        
        for import_spec in imports:
            if '.' in import_spec:
                module = import_spec.split('.')[0]
            else:
                module = import_spec
                
            # Try to import the module
            try:
                spec = importlib.util.find_spec(module)
                if spec is None:
                    errors.append(f"Cannot find module: {module}")
            except (ImportError, ModuleNotFoundError) as e:
                errors.append(f"Import error for {module}: {str(e)}")
                
        return len(errors) == 0, errors
        
    def optimize_imports(self, code: str) -> str:
        """Optimize imports by removing unused ones and organizing them"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code
            
        # Collect all imports and their usage
        imports_info = {}
        used_names = set()
        
        # First pass: collect imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    imports_info[name] = f"import {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    imports_info[name] = f"from {module} import {alias.name}"
                    
        # Second pass: find used names
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
                    
        # Keep only used imports
        used_imports = []
        for name, import_stmt in imports_info.items():
            if name in used_names:
                used_imports.append(import_stmt)
                
        # Remove old imports and add optimized ones
        lines = code.split('\n')
        new_lines = []
        import_section_done = False
        
        for line in lines:
            if not import_section_done and line.strip().startswith(('import ', 'from ')):
                continue  # Skip old imports
            else:
                if not import_section_done and line.strip() and not line.strip().startswith('#'):
                    # End of import section, add optimized imports
                    import_section_done = True
                    if used_imports:
                        new_lines.extend(sorted(set(used_imports)))
                        new_lines.append('')
                new_lines.append(line)
                
        return '\n'.join(new_lines)


if __name__ == "__main__":
    # Example usage and testing
    resolver = DependencyResolver()
    
    # Test code with import issues
    test_code = """
def analyze_repository(repo_path):
    # Missing imports for Path, json, logging
    logger = logging.getLogger(__name__)
    path = Path(repo_path)
    
    data = {"files": []}
    for file in path.glob("*.py"):
        with open(file) as f:
            content = f.read()
        data["files"].append(str(file))
        
    return json.dumps(data)
"""
    
    print("Original code:")
    print(test_code)
    print("\n" + "="*50 + "\n")
    
    # Fix imports
    fixed_code = resolver.fix_import_paths(test_code)
    print("Fixed code:")
    print(fixed_code)