"""
Analyze code complexity and suggest simplifications

This is a fully functional code_complexity_analyzer tool.
"""

import ast
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class CodeComplexityAnalyzer:
    """Main analyzer class."""
    
    def __init__(self):
        self.results = defaultdict(list)
        self.metrics = {}
        
    async def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single file."""
        try:
            content = file_path.read_text()
            tree = ast.parse(content, filename=str(file_path))
            
            # Perform analysis
            complexity = self._calculate_complexity(tree)
            issues = self._find_issues(tree)
            suggestions = self._generate_suggestions(complexity, issues)
            
            return {
                'file': str(file_path),
                'complexity': complexity,
                'issues': issues,
                'suggestions': suggestions
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return {
                'file': str(file_path),
                'error': str(e)
            }
            
    def _calculate_complexity(self, tree: ast.AST) -> Dict[str, int]:
        """Calculate code complexity metrics."""
        metrics = {
            'lines': len(tree.body),
            'functions': sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)),
            'classes': sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef)),
            'branches': sum(1 for n in ast.walk(tree) if isinstance(n, (ast.If, ast.For, ast.While))),
        }
        
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
                    issues.append({
                        'type': 'long_function',
                        'line': node.lineno,
                        'name': node.name,
                        'severity': 'medium'
                    })
                    
            # Nested loops
            if isinstance(node, ast.For):
                for child in ast.walk(node):
                    if child != node and isinstance(child, ast.For):
                        issues.append({
                            'type': 'nested_loop',
                            'line': node.lineno,
                            'severity': 'low'
                        })
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
    
    analyzer = CodeComplexityAnalyzer()
    results = {
        'summary': {},
        'files': [],
        'total_issues': 0
    }
    
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
    results['summary'] = {
        'files_analyzed': len(results['files']),
        'total_issues': results['total_issues'],
        'status': 'completed'
    }
    
    return results


# Make the tool callable
code_complexity_analyzer = analyze
