"""
Real Improvement Validator

This module ensures that all improvements created by the system are real and functional,
not just placeholder stubs.
"""

import ast
import asyncio
import importlib.util
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timezone


class RealImprovementValidator:
    """Validates that improvements are real and functional."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_criteria = {
            'tool': self._validate_tool,
            'workflow': self._validate_workflow,
            'error_fix': self._validate_error_fix,
            'optimization': self._validate_optimization,
            'analysis_tool': self._validate_analysis_tool
        }
        
    async def validate_improvement(self, improvement: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that an improvement is real and functional.
        
        Returns:
            Tuple of (is_valid, issues_found)
        """
        imp_type = improvement.get('type', 'unknown')
        
        if imp_type not in self.validation_criteria:
            return False, [f"Unknown improvement type: {imp_type}"]
            
        validator = self.validation_criteria[imp_type]
        return await validator(improvement)
        
    async def _validate_tool(self, improvement: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a tool improvement."""
        issues = []
        tool_name = improvement.get('name', '')
        
        # Check if tool file exists
        tool_path = Path(f"scripts/custom_tools/{tool_name}.py")
        if not tool_path.exists():
            return False, [f"Tool file not found: {tool_path}"]
            
        # Read and analyze the code
        try:
            code = tool_path.read_text()
            
            # Check for stub/placeholder patterns
            stub_patterns = [
                'return {}',
                'return []',
                'pass',
                '"completed"',
                'return {"results": []}',
                'return {"analysis": "',
                '# TODO:',
                'raise NotImplementedError'
            ]
            
            for pattern in stub_patterns:
                if pattern in code and len(code) < 500:  # Small file with stub pattern
                    issues.append(f"Tool appears to be a stub (contains '{pattern}')")
                    
            # Parse AST to check for real implementation
            tree = ast.parse(code)
            functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            
            if not functions:
                issues.append("No functions found in tool")
            else:
                # Check function complexity
                for func in functions:
                    # Count meaningful statements (not just pass or return literal)
                    meaningful_stmts = 0
                    for stmt in ast.walk(func):
                        if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.For, ast.While, 
                                           ast.If, ast.With, ast.Try, ast.Call)):
                            meaningful_stmts += 1
                            
                    if meaningful_stmts < 3:
                        issues.append(f"Function '{func.name}' has insufficient implementation")
                        
            # Check for imports (real tools use libraries)
            imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
            if len(imports) < 2:
                issues.append("Tool has minimal imports, suggesting limited functionality")
                
            # Try to import and run basic validation
            spec = importlib.util.spec_from_file_location(tool_name, tool_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Check for required attributes/functions
                if not any(hasattr(module, attr) for attr in ['main', 'run', 'execute', tool_name]):
                    issues.append("Tool missing main execution function")
                    
        except Exception as e:
            issues.append(f"Failed to validate tool: {str(e)}")
            
        return len(issues) == 0, issues
        
    async def _validate_workflow(self, improvement: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a workflow improvement."""
        issues = []
        workflow_name = improvement.get('name', '')
        
        # Check multiple possible workflow locations
        possible_paths = [
            Path(f"scripts/workflows/{workflow_name}.py"),
            Path(f"scripts/workflows/{workflow_name}.json"),
            Path(f"workflows/{workflow_name}.yaml"),
        ]
        
        workflow_found = False
        for path in possible_paths:
            if path.exists():
                workflow_found = True
                
                # Validate workflow content
                content = path.read_text()
                
                # Check for minimal/stub workflow
                if len(content) < 100:
                    issues.append("Workflow file is too small to be functional")
                    
                if path.suffix == '.json':
                    try:
                        import json
                        workflow_data = json.loads(content)
                        
                        # Check for required workflow fields
                        if 'steps' not in workflow_data or len(workflow_data.get('steps', [])) < 2:
                            issues.append("Workflow has insufficient steps")
                            
                    except json.JSONDecodeError:
                        issues.append("Invalid JSON in workflow file")
                        
                break
                
        if not workflow_found:
            # Check if workflow was added to system_workflows.py
            system_workflows_path = Path("scripts/system_workflows.py")
            if system_workflows_path.exists():
                content = system_workflows_path.read_text()
                if workflow_name not in content:
                    issues.append(f"Workflow '{workflow_name}' not found in any location")
            else:
                issues.append(f"Workflow '{workflow_name}' not found")
                
        return len(issues) == 0, issues
        
    async def _validate_error_fix(self, improvement: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate an error fix improvement."""
        issues = []
        
        # Error fixes should modify existing files
        files_modified = improvement.get('files_modified', [])
        if not files_modified:
            issues.append("Error fix claims no files were modified")
            
        # Check that modifications are real
        for file_info in files_modified:
            file_path = Path(file_info.get('path', ''))
            if not file_path.exists():
                issues.append(f"Modified file does not exist: {file_path}")
                
        return len(issues) == 0, issues
        
    async def _validate_optimization(self, improvement: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate an optimization improvement."""
        issues = []
        
        # Optimizations should have measurable impact
        metrics = improvement.get('metrics', {})
        if not metrics:
            issues.append("Optimization has no performance metrics")
            
        # Check for before/after comparison
        if 'before' not in metrics or 'after' not in metrics:
            issues.append("Optimization missing before/after metrics")
            
        return len(issues) == 0, issues
        
    async def _validate_analysis_tool(self, improvement: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate an analysis tool improvement."""
        # Analysis tools are validated the same as regular tools
        return await self._validate_tool(improvement)
        
    async def enforce_real_improvements(self, improvement_engine):
        """
        Monkey-patch the improvement engine to enforce real improvements.
        """
        original_create = improvement_engine.create_real_improvement
        
        async def validated_create():
            # Try original creation
            result = await original_create()
            
            if result.get('success'):
                # Validate the improvement
                is_valid, issues = await self.validate_improvement(result)
                
                if not is_valid:
                    self.logger.error(f"Rejected fake improvement: {result.get('name')}")
                    for issue in issues:
                        self.logger.error(f"  - {issue}")
                        
                    # Force a real improvement instead
                    raise ValueError(f"Improvement validation failed: {'; '.join(issues)}")
                    
            return result
            
        improvement_engine.create_real_improvement = validated_create
        
    def generate_real_tool_code(self, tool_name: str, tool_description: str) -> str:
        """Generate real, functional tool code instead of stubs."""
        # Map tool types to real implementations
        tool_templates = {
            'analyzer': self._generate_analyzer_template,
            'scanner': self._generate_scanner_template,
            'monitor': self._generate_monitor_template,
            'detector': self._generate_detector_template,
            'optimizer': self._generate_optimizer_template
        }
        
        # Determine tool type from name
        tool_type = 'analyzer'  # default
        for key in tool_templates:
            if key in tool_name.lower():
                tool_type = key
                break
                
        return tool_templates[tool_type](tool_name, tool_description)
        
    def _generate_analyzer_template(self, name: str, description: str) -> str:
        """Generate a real analyzer tool."""
        return f'''"""
{description}

This is a fully functional {name} tool.
"""

import ast
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class {name.title().replace('_', '')}:
    """Main analyzer class."""
    
    def __init__(self):
        self.results = defaultdict(list)
        self.metrics = {{}}
        
    async def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single file."""
        try:
            content = file_path.read_text()
            tree = ast.parse(content, filename=str(file_path))
            
            # Perform analysis
            complexity = self._calculate_complexity(tree)
            issues = self._find_issues(tree)
            suggestions = self._generate_suggestions(complexity, issues)
            
            return {{
                'file': str(file_path),
                'complexity': complexity,
                'issues': issues,
                'suggestions': suggestions
            }}
            
        except Exception as e:
            logger.error(f"Failed to analyze {{file_path}}: {{e}}")
            return {{
                'file': str(file_path),
                'error': str(e)
            }}
            
    def _calculate_complexity(self, tree: ast.AST) -> Dict[str, int]:
        """Calculate code complexity metrics."""
        metrics = {{
            'lines': len(tree.body),
            'functions': sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)),
            'classes': sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef)),
            'branches': sum(1 for n in ast.walk(tree) if isinstance(n, (ast.If, ast.For, ast.While))),
        }}
        
        # Cyclomatic complexity approximation
        metrics['cyclomatic'] = metrics['branches'] + 1
        
        return metrics
        
    def _find_issues(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find potential issues in the code."""
        issues = []
        
        # Check for common issues
        for node in ast.walk(tree):
            # Long functions
            if isinstance(node, ast.FunctionDef):
                if len(node.body) > 50:
                    issues.append({{
                        'type': 'long_function',
                        'line': node.lineno,
                        'name': node.name,
                        'severity': 'medium'
                    }})
                    
            # Nested loops
            if isinstance(node, ast.For):
                for child in ast.walk(node):
                    if child != node and isinstance(child, ast.For):
                        issues.append({{
                            'type': 'nested_loop',
                            'line': node.lineno,
                            'severity': 'low'
                        }})
                        break
                        
        return issues
        
    def _generate_suggestions(self, complexity: Dict, issues: List) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        if complexity['cyclomatic'] > 10:
            suggestions.append("Consider breaking down complex functions")
            
        if complexity['functions'] > 20:
            suggestions.append("Consider splitting into multiple modules")
            
        if len(issues) > 5:
            suggestions.append("Focus on addressing high-severity issues first")
            
        return suggestions


async def analyze(**kwargs) -> Dict[str, Any]:
    """Main entry point for the analyzer."""
    target = kwargs.get('target', '.')
    file_pattern = kwargs.get('pattern', '**/*.py')
    
    analyzer = {name.title().replace('_', '')}()
    results = {{
        'summary': {{}},
        'files': [],
        'total_issues': 0
    }}
    
    # Find files to analyze
    path = Path(target)
    if path.is_file():
        files = [path]
    else:
        files = list(path.glob(file_pattern))[:100]  # Limit to 100 files
        
    # Analyze each file
    for file_path in files:
        if file_path.suffix == '.py':
            file_result = await analyzer.analyze_file(file_path)
            results['files'].append(file_result)
            
            if 'issues' in file_result:
                results['total_issues'] += len(file_result['issues'])
                
    # Generate summary
    results['summary'] = {{
        'files_analyzed': len(results['files']),
        'total_issues': results['total_issues'],
        'status': 'completed'
    }}
    
    return results


# Make the tool callable
{name} = analyze
'''
        
    def _generate_scanner_template(self, name: str, description: str) -> str:
        """Generate a real scanner tool."""
        return f'''"""
{description}

This is a fully functional {name} tool.
"""

import re
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Pattern
import json

logger = logging.getLogger(__name__)


class {name.title().replace('_', '')}:
    """Main scanner class."""
    
    def __init__(self):
        self.patterns = self._load_patterns()
        self.findings = []
        
    def _load_patterns(self) -> Dict[str, List[Pattern]]:
        """Load scanning patterns."""
        return {{
            'security': [
                re.compile(r'eval\\s*\\([^)]+\\)', re.I),
                re.compile(r'exec\\s*\\([^)]+\\)', re.I),
                re.compile(r'__import__\\s*\\([^)]+\\)', re.I),
                re.compile(r'os\\.system\\s*\\([^)]+\\)', re.I),
            ],
            'credentials': [
                re.compile(r'["\']?password["\']?\\s*[:=]\\s*["\'][^"\']+["\']', re.I),
                re.compile(r'["\']?api_key["\']?\\s*[:=]\\s*["\'][^"\']+["\']', re.I),
                re.compile(r'["\']?secret["\']?\\s*[:=]\\s*["\'][^"\']+["\']', re.I),
            ],
            'quality': [
                re.compile(r'#\\s*TODO:', re.I),
                re.compile(r'#\\s*FIXME:', re.I),
                re.compile(r'#\\s*HACK:', re.I),
            ]
        }}
        
    async def scan_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan a single file."""
        findings = []
        
        try:
            content = file_path.read_text()
            lines = content.split('\\n')
            
            for category, patterns in self.patterns.items():
                for pattern in patterns:
                    for line_num, line in enumerate(lines, 1):
                        if pattern.search(line):
                            findings.append({{
                                'file': str(file_path),
                                'line': line_num,
                                'category': category,
                                'pattern': pattern.pattern,
                                'content': line.strip()[:100],
                                'severity': self._determine_severity(category)
                            }})
                            
        except Exception as e:
            logger.error(f"Failed to scan {{file_path}}: {{e}}")
            
        return findings
        
    def _determine_severity(self, category: str) -> str:
        """Determine finding severity."""
        severity_map = {{
            'security': 'high',
            'credentials': 'critical',
            'quality': 'low'
        }}
        return severity_map.get(category, 'medium')


async def scan(**kwargs) -> Dict[str, Any]:
    """Main entry point for the scanner."""
    target = kwargs.get('target', '.')
    recursive = kwargs.get('recursive', True)
    
    scanner = {name.title().replace('_', '')}()
    results = {{
        'findings': [],
        'summary': {{}},
        'stats': {{}}
    }}
    
    # Find files to scan
    path = Path(target)
    if path.is_file():
        files = [path]
    else:
        pattern = '**/*' if recursive else '*'
        files = [f for f in path.glob(pattern) if f.is_file() and f.suffix in ['.py', '.js', '.yml', '.json']]
        
    # Scan files
    for file_path in files[:200]:  # Limit to 200 files
        findings = await scanner.scan_file(file_path)
        results['findings'].extend(findings)
        
    # Generate summary
    by_severity = {{}}
    by_category = {{}}
    
    for finding in results['findings']:
        sev = finding['severity']
        cat = finding['category']
        
        by_severity[sev] = by_severity.get(sev, 0) + 1
        by_category[cat] = by_category.get(cat, 0) + 1
        
    results['summary'] = {{
        'total_findings': len(results['findings']),
        'by_severity': by_severity,
        'by_category': by_category,
        'files_scanned': len(files)
    }}
    
    return results


# Make the tool callable
{name} = scan
'''
        
    def _generate_monitor_template(self, name: str, description: str) -> str:
        """Generate a real monitor tool."""
        return self._generate_analyzer_template(name, description)  # Similar structure
        
    def _generate_detector_template(self, name: str, description: str) -> str:
        """Generate a real detector tool."""
        return self._generate_scanner_template(name, description)  # Similar structure
        
    def _generate_optimizer_template(self, name: str, description: str) -> str:
        """Generate a real optimizer tool."""
        return self._generate_analyzer_template(name, description)  # Similar structure