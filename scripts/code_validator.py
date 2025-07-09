"""
Code Validator
==============

Validates Python code and provides intelligent fixes.
"""

import ast
import re
from typing import Dict, List, Any, Optional, Tuple


def validate_python_code(code: str) -> Dict[str, Any]:
    """Validate Python code and return issues found."""
    
    issues = []
    
    # Check syntax
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        issues.append({
            'type': 'syntax_error',
            'message': str(e),
            'line': e.lineno,
            'severity': 'error'
        })
        return {
            'valid': False,
            'issues': issues,
            'ast': None
        }
    
    # Check for required tool elements
    tool_checks = check_tool_structure(code, tree)
    issues.extend(tool_checks)
    
    # Check imports
    import_checks = check_imports(tree)
    issues.extend(import_checks)
    
    # Check for common issues
    common_checks = check_common_issues(code, tree)
    issues.extend(common_checks)
    
    return {
        'valid': len([i for i in issues if i['severity'] == 'error']) == 0,
        'issues': issues,
        'ast': tree
    }


def check_tool_structure(code: str, tree: ast.AST) -> List[Dict[str, Any]]:
    """Check if code follows tool structure requirements."""
    issues = []
    
    # Check for __description__
    if '__description__' not in code:
        issues.append({
            'type': 'missing_description',
            'message': 'Tool is missing __description__ variable',
            'severity': 'error',
            'fix': 'Add __description__ = "Tool description"'
        })
    
    # Check for __parameters__
    if '__parameters__' not in code:
        issues.append({
            'type': 'missing_parameters',
            'message': 'Tool is missing __parameters__ variable',
            'severity': 'error', 
            'fix': 'Add __parameters__ = {}'
        })
    
    # Check for main function
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    main_func = None
    
    for func in functions:
        if not func.name.startswith('_'):
            main_func = func
            break
    
    if not main_func:
        issues.append({
            'type': 'missing_main_function',
            'message': 'Tool is missing main function',
            'severity': 'error'
        })
    else:
        # Check if function is async
        if not isinstance(main_func, ast.AsyncFunctionDef):
            issues.append({
                'type': 'sync_function',
                'message': f'Function {main_func.name} should be async',
                'severity': 'warning',
                'fix': f'Change "def {main_func.name}" to "async def {main_func.name}"'
            })
        
        # Check function parameters
        if not main_func.args.kwarg:
            issues.append({
                'type': 'missing_kwargs',
                'message': f'Function {main_func.name} should accept **kwargs',
                'severity': 'warning',
                'fix': f'Add **kwargs parameter to {main_func.name}'
            })
        
        # Check for export
        if f'{main_func.name}_tool' not in code:
            issues.append({
                'type': 'missing_export',
                'message': f'Tool is missing export: {main_func.name}_tool = {main_func.name}',
                'severity': 'error',
                'fix': f'Add "{main_func.name}_tool = {main_func.name}" at the end'
            })
    
    return issues


def check_imports(tree: ast.AST) -> List[Dict[str, Any]]:
    """Check imports for common issues."""
    issues = []
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    'module': alias.name,
                    'alias': alias.asname,
                    'type': 'import'
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append({
                    'module': module,
                    'name': alias.name,
                    'alias': alias.asname,
                    'type': 'from'
                })
    
    # Check for missing typing import
    has_typing = any(imp['module'] == 'typing' or 
                    (imp['type'] == 'from' and imp['module'] == 'typing')
                    for imp in imports)
    
    if not has_typing:
        issues.append({
            'type': 'missing_typing',
            'message': 'Missing typing import',
            'severity': 'warning',
            'fix': 'Add "from typing import Dict, Any"'
        })
    
    # Check for problematic imports
    problematic_imports = {
        'scripts.mcp_github_issue_creator': 'scripts.github_issue_creator',
        'scripts.alert_manager': 'scripts.logger',
        'scripts.metrics_tracker': 'scripts.state_manager'
    }
    
    for imp in imports:
        module = imp['module']
        if module in problematic_imports:
            issues.append({
                'type': 'problematic_import',
                'message': f'Import {module} should be {problematic_imports[module]}',
                'severity': 'error',
                'fix': f'Change {module} to {problematic_imports[module]}'
            })
    
    return issues


def check_common_issues(code: str, tree: ast.AST) -> List[Dict[str, Any]]:
    """Check for common coding issues."""
    issues = []
    
    # Check for proper error handling
    has_try_except = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            has_try_except = True
            # Check if except is too broad
            for handler in node.handlers:
                if handler.type is None or (
                    isinstance(handler.type, ast.Name) and handler.type.id == 'Exception'
                ):
                    # Check if error is properly handled
                    has_return = any(isinstance(n, ast.Return) for n in ast.walk(handler))
                    if not has_return:
                        issues.append({
                            'type': 'unhandled_exception',
                            'message': 'Exception caught but not properly handled',
                            'severity': 'warning',
                            'fix': 'Add return statement in except block'
                        })
    
    if not has_try_except:
        issues.append({
            'type': 'missing_error_handling',
            'message': 'Tool lacks error handling',
            'severity': 'warning',
            'fix': 'Wrap main logic in try/except block'
        })
    
    # Check for proper returns
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for func in functions:
        if not func.name.startswith('_'):
            returns = [node for node in ast.walk(func) if isinstance(node, ast.Return)]
            if not returns:
                issues.append({
                    'type': 'missing_return',
                    'message': f'Function {func.name} has no return statement',
                    'severity': 'error',
                    'fix': f'Add return statement to {func.name}'
                })
            else:
                # Check if returns follow expected format
                for ret in returns:
                    if ret.value and isinstance(ret.value, ast.Dict):
                        # Check for success key
                        has_success = any(
                            isinstance(k, ast.Constant) and k.value == 'success'
                            for k in ret.value.keys
                        )
                        if not has_success:
                            issues.append({
                                'type': 'missing_success_key',
                                'message': 'Return dict should have "success" key',
                                'severity': 'warning',
                                'fix': 'Add "success": True/False to return dict'
                            })
    
    # Check for hardcoded values that should be parameters
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) > 20 and any(word in node.value.lower() for word in ['path', 'file', 'url']):
                issues.append({
                    'type': 'hardcoded_value',
                    'message': f'Hardcoded value found: {node.value[:50]}...',
                    'severity': 'warning',
                    'fix': 'Consider making this a parameter'
                })
    
    return issues


def auto_fix_code(code: str, issues: List[Dict[str, Any]]) -> str:
    """Automatically fix common issues in code."""
    
    fixed_code = code
    
    for issue in issues:
        if issue['type'] == 'missing_description' and 'fix' in issue:
            # Add after imports
            lines = fixed_code.split('\n')
            import_end = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith(('from', 'import', '#', '"""')):
                    import_end = i
                    break
            lines.insert(import_end, '\n__description__ = "Tool function"')
            fixed_code = '\n'.join(lines)
        
        elif issue['type'] == 'missing_parameters':
            if '__description__' in fixed_code:
                fixed_code = fixed_code.replace(
                    '__description__',
                    '__description__\n__parameters__ = {}'
                )
            else:
                lines = fixed_code.split('\n')
                lines.append('\n__parameters__ = {}')
                fixed_code = '\n'.join(lines)
        
        elif issue['type'] == 'missing_typing':
            # Add typing import at the top
            lines = fixed_code.split('\n')
            import_added = False
            for i, line in enumerate(lines):
                if line.startswith('from') or line.startswith('import'):
                    lines.insert(i, 'from typing import Dict, Any, List, Optional')
                    import_added = True
                    break
            if not import_added:
                lines.insert(0, 'from typing import Dict, Any, List, Optional')
            fixed_code = '\n'.join(lines)
        
        elif issue['type'] == 'problematic_import' and 'fix' in issue:
            # Extract old and new module names
            if 'Change' in issue['fix']:
                parts = issue['fix'].split(' to ')
                if len(parts) == 2:
                    old_module = parts[0].replace('Change ', '')
                    new_module = parts[1]
                    fixed_code = fixed_code.replace(old_module, new_module)
        
        elif issue['type'] == 'missing_export' and 'fix' in issue:
            # Add export at the end
            export_line = issue['fix'].replace('Add "', '').replace('" at the end', '')
            if export_line not in fixed_code:
                fixed_code += f'\n\n{export_line}'
        
        elif issue['type'] == 'sync_function' and 'fix' in issue:
            # Change def to async def
            if 'Change "def' in issue['fix']:
                func_name = issue['fix'].split('"')[1].replace('def ', '')
                fixed_code = re.sub(
                    rf'\bdef\s+{func_name}\b',
                    f'async def {func_name}',
                    fixed_code
                )
    
    return fixed_code