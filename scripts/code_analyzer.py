"""
Code Analyzer Module

Analyzes code patterns, complexity, and quality metrics.
"""

import ast
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging


class CodeAnalyzer:
    """Analyzes code for patterns, complexity, and improvement opportunities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Python file for various metrics."""
        try:
            path = Path(file_path)
            if not path.exists() or not path.suffix == '.py':
                return {'error': 'Invalid Python file'}
                
            content = path.read_text()
            
            return {
                'file_path': file_path,
                'metrics': self._calculate_metrics(content),
                'patterns': self._detect_patterns(content),
                'complexity': self._calculate_complexity(content),
                'issues': self._detect_issues(content)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
            return {'error': str(e)}
            
    def _calculate_metrics(self, content: str) -> Dict[str, int]:
        """Calculate basic code metrics."""
        lines = content.split('\n')
        return {
            'total_lines': len(lines),
            'code_lines': len([l for l in lines if l.strip() and not l.strip().startswith('#')]),
            'comment_lines': len([l for l in lines if l.strip().startswith('#')]),
            'blank_lines': len([l for l in lines if not l.strip()]),
            'functions': len(re.findall(r'^def\s+\w+', content, re.MULTILINE)),
            'classes': len(re.findall(r'^class\s+\w+', content, re.MULTILINE))
        }
        
    def _detect_patterns(self, content: str) -> List[str]:
        """Detect common code patterns."""
        patterns = []
        
        # Check for common patterns
        if 'async def' in content:
            patterns.append('async_await')
        if '@property' in content:
            patterns.append('properties')
        if 'try:' in content and 'except' in content:
            patterns.append('error_handling')
        if 'logging' in content or 'logger' in content:
            patterns.append('logging')
        if '@cache' in content or 'lru_cache' in content:
            patterns.append('caching')
            
        return patterns
        
    def _calculate_complexity(self, content: str) -> Dict[str, Any]:
        """Calculate code complexity metrics."""
        try:
            tree = ast.parse(content)
            
            # Simple complexity calculation
            complexity = {
                'cyclomatic_complexity': self._count_decision_points(tree),
                'nesting_depth': self._max_nesting_depth(tree),
                'function_count': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            }
            
            return complexity
        except:
            return {'cyclomatic_complexity': 0, 'nesting_depth': 0, 'function_count': 0}
            
    def _count_decision_points(self, tree: ast.AST) -> int:
        """Count decision points for cyclomatic complexity."""
        count = 1  # Base complexity
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                count += 1
            elif isinstance(node, ast.BoolOp):
                count += len(node.values) - 1
        return count
        
    def _max_nesting_depth(self, tree: ast.AST, depth: int = 0) -> int:
        """Calculate maximum nesting depth."""
        max_depth = depth
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                child_depth = self._max_nesting_depth(node, depth + 1)
                max_depth = max(max_depth, child_depth)
        return max_depth
        
    def _detect_issues(self, content: str) -> List[Dict[str, str]]:
        """Detect potential code issues."""
        issues = []
        
        # Check for common issues
        if 'except:' in content or 'except Exception:' in content:
            issues.append({
                'type': 'broad_exception',
                'description': 'Broad exception handling detected'
            })
            
        if re.search(r'def\s+\w+\([^)]{100,}\)', content):
            issues.append({
                'type': 'long_parameter_list',
                'description': 'Function with very long parameter list'
            })
            
        # Check for hardcoded values
        if re.search(r'["\'](?:password|token|key|secret)["\']', content, re.IGNORECASE):
            issues.append({
                'type': 'potential_secret',
                'description': 'Potential hardcoded secret detected'
            })
            
        return issues
        
    def suggest_improvements(self, analysis: Dict[str, Any]) -> List[str]:
        """Suggest improvements based on analysis."""
        suggestions = []
        
        metrics = analysis.get('metrics', {})
        complexity = analysis.get('complexity', {})
        issues = analysis.get('issues', [])
        
        # Complexity suggestions
        if complexity.get('cyclomatic_complexity', 0) > 10:
            suggestions.append('Consider breaking down complex functions')
            
        if complexity.get('nesting_depth', 0) > 4:
            suggestions.append('Reduce nesting depth by extracting functions')
            
        # Size suggestions
        if metrics.get('code_lines', 0) > 500:
            suggestions.append('Consider splitting this module into smaller files')
            
        # Issue-based suggestions
        for issue in issues:
            if issue['type'] == 'broad_exception':
                suggestions.append('Use specific exception types instead of broad catches')
            elif issue['type'] == 'potential_secret':
                suggestions.append('Move secrets to environment variables')
                
        return suggestions