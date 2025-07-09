"""
Universal Counter Tool

A powerful unified tool that handles all counting, calculation, and metric operations.
Consolidates 18 different counting tools into one intelligent system.

Capabilities:
- Count files, lines, functions, repositories, workers
- Calculate statistics, averages, complexity metrics
- Find patterns, TODOs, async functions
- Analyze project metrics and opportunities
"""

import os
import re
import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from collections import defaultdict

from scripts.state_manager import StateManager
from scripts.repository_analyzer import RepositoryAnalyzer
from scripts.http_ai_client import HTTPAIClient


class UniversalCounter:
    """Unified counting and metrics system."""
    
    def __init__(self):
        self.state_manager = StateManager()
        self.repo_analyzer = RepositoryAnalyzer()
        self.ai_client = HTTPAIClient()
        self.project_root = Path(__file__).parent.parent.parent
        
    async def count(self, target: str, **kwargs) -> Dict[str, Any]:
        """Universal counting entry point.
        
        Args:
            target: What to count (files, lines, functions, repositories, etc.)
            **kwargs: Additional parameters based on target
            
        Returns:
            Count results with detailed breakdown
        """
        # Normalize target
        target = target.lower().replace('-', '_').replace(' ', '_')
        
        # Route to appropriate counter
        counters = {
            'files': self._count_files,
            'lines': self._count_lines,
            'functions': self._count_functions,
            'repositories': self._count_repositories,
            'workers': self._count_workers,
            'python_files': self._count_python_files,
            'markdown_files': self._count_markdown_files,
            'todos': self._count_todos,
            'async_functions': self._count_async_functions,
            'complexity': self._calculate_complexity,
            'statistics': self._calculate_statistics,
            'issues': self._count_issues,
            'opportunities': self._count_opportunities
        }
        
        # Find best match
        handler = None
        for key, func in counters.items():
            if key in target or target in key:
                handler = func
                break
                
        if not handler:
            # Use AI to understand intent
            handler = await self._intelligent_count
            kwargs['original_target'] = target
            
        return await handler(**kwargs)
        
    async def _count_files(self, directory: str = '.', pattern: str = '*', recursive: bool = True, **kwargs) -> Dict[str, Any]:
        """Count files in directory."""
        path = Path(directory) if directory != '.' else self.project_root
        
        if not path.exists():
            return {"error": f"Directory {directory} not found"}
            
        files = []
        if recursive:
            files = list(path.rglob(pattern))
        else:
            files = list(path.glob(pattern))
            
        # Categorize by type
        by_extension = defaultdict(int)
        by_size = defaultdict(int)
        total_size = 0
        
        for f in files:
            if f.is_file():
                ext = f.suffix or 'no_extension'
                by_extension[ext] += 1
                
                size = f.stat().st_size
                total_size += size
                
                if size < 1024:
                    by_size['<1KB'] += 1
                elif size < 1024 * 100:
                    by_size['1-100KB'] += 1
                elif size < 1024 * 1024:
                    by_size['100KB-1MB'] += 1
                else:
                    by_size['>1MB'] += 1
                    
        return {
            "total_files": len(files),
            "directory": str(path),
            "pattern": pattern,
            "by_extension": dict(by_extension),
            "by_size": dict(by_size),
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "largest_files": sorted(
                [(f.name, f.stat().st_size) for f in files if f.is_file()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
        
    async def _count_lines(self, language: str = None, directory: str = '.', **kwargs) -> Dict[str, Any]:
        """Count lines of code by language."""
        path = Path(directory) if directory != '.' else self.project_root
        
        language_extensions = {
            'python': ['.py'],
            'javascript': ['.js', '.jsx'],
            'typescript': ['.ts', '.tsx'],
            'java': ['.java'],
            'c++': ['.cpp', '.cc', '.cxx', '.hpp'],
            'go': ['.go'],
            'rust': ['.rs'],
            'markdown': ['.md'],
            'json': ['.json'],
            'yaml': ['.yml', '.yaml']
        }
        
        results = {}
        total_lines = 0
        total_code_lines = 0
        total_comment_lines = 0
        total_blank_lines = 0
        
        # Get extensions to check
        if language:
            extensions = language_extensions.get(language.lower(), [f'.{language}'])
        else:
            extensions = [ext for exts in language_extensions.values() for ext in exts]
            
        for ext in extensions:
            ext_stats = {
                'files': 0,
                'total_lines': 0,
                'code_lines': 0,
                'comment_lines': 0,
                'blank_lines': 0
            }
            
            for file_path in path.rglob(f'*{ext}'):
                if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            
                        ext_stats['files'] += 1
                        ext_stats['total_lines'] += len(lines)
                        
                        for line in lines:
                            line = line.strip()
                            if not line:
                                ext_stats['blank_lines'] += 1
                            elif line.startswith('#') or line.startswith('//') or line.startswith('/*'):
                                ext_stats['comment_lines'] += 1
                            else:
                                ext_stats['code_lines'] += 1
                                
                    except:
                        pass
                        
            if ext_stats['files'] > 0:
                results[ext] = ext_stats
                total_lines += ext_stats['total_lines']
                total_code_lines += ext_stats['code_lines']
                total_comment_lines += ext_stats['comment_lines']
                total_blank_lines += ext_stats['blank_lines']
                
        return {
            "by_extension": results,
            "total_lines": total_lines,
            "total_code_lines": total_code_lines,
            "total_comment_lines": total_comment_lines,
            "total_blank_lines": total_blank_lines,
            "code_to_comment_ratio": round(total_code_lines / max(total_comment_lines, 1), 2)
        }
        
    async def _count_functions(self, pattern: str = None, async_only: bool = False, **kwargs) -> Dict[str, Any]:
        """Count functions in Python files."""
        functions = []
        async_functions = []
        classes = []
        
        for py_file in self.project_root.rglob('*.py'):
            if any(part.startswith('.') for part in py_file.parts):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if not pattern or pattern.lower() in node.name.lower():
                            if async_only and not isinstance(node, ast.AsyncFunctionDef):
                                continue
                                
                            func_info = {
                                'name': node.name,
                                'file': str(py_file.relative_to(self.project_root)),
                                'line': node.lineno,
                                'async': isinstance(node, ast.AsyncFunctionDef),
                                'args': len(node.args.args)
                            }
                            
                            functions.append(func_info)
                            if func_info['async']:
                                async_functions.append(func_info)
                                
                    elif isinstance(node, ast.ClassDef):
                        classes.append({
                            'name': node.name,
                            'file': str(py_file.relative_to(self.project_root)),
                            'line': node.lineno
                        })
                        
            except:
                pass
                
        return {
            "total_functions": len(functions),
            "async_functions": len(async_functions),
            "sync_functions": len(functions) - len(async_functions),
            "total_classes": len(classes),
            "functions": functions[:20] if not pattern else functions,  # Limit output
            "by_file": self._group_by_file(functions),
            "most_complex": sorted(functions, key=lambda x: x['args'], reverse=True)[:10]
        }
        
    async def _count_repositories(self, **kwargs) -> Dict[str, Any]:
        """Count repositories in the system."""
        state = self.state_manager.get_state()
        projects = state.get('projects', {})
        
        by_status = defaultdict(int)
        by_health = defaultdict(int)
        total_issues = 0
        total_stars = 0
        
        for repo, data in projects.items():
            # Status
            if isinstance(data, dict):
                health = data.get('health_score', 0)
                if health >= 80:
                    by_status['healthy'] += 1
                elif health >= 60:
                    by_status['moderate'] += 1
                else:
                    by_status['needs_attention'] += 1
                    
                # Health score ranges
                range_key = f"{(health // 10) * 10}-{((health // 10) + 1) * 10}"
                by_health[range_key] += 1
                
                # Metrics
                metrics = data.get('metrics', {})
                total_issues += metrics.get('issues_open', 0)
                total_stars += metrics.get('stars', 0)
                
        return {
            "total_repositories": len(projects),
            "by_status": dict(by_status),
            "by_health_range": dict(by_health),
            "total_open_issues": total_issues,
            "total_stars": total_stars,
            "average_health": round(sum(p.get('health_score', 0) for p in projects.values() if isinstance(p, dict)) / max(len(projects), 1), 1),
            "repositories": list(projects.keys())
        }
        
    async def _count_workers(self, **kwargs) -> Dict[str, Any]:
        """Count active workers in the system."""
        try:
            # Check orchestrator state
            orch_state_file = self.project_root / "continuous_orchestrator_state.json"
            if orch_state_file.exists():
                with open(orch_state_file, 'r') as f:
                    state = json.load(f)
                    
                return {
                    "active_workers": state.get('active_workers', 0),
                    "queued_tasks": len(state.get('task_queue', [])),
                    "completed_tasks": state.get('completed_tasks', 0),
                    "worker_status": state.get('worker_status', {}),
                    "uptime_seconds": state.get('uptime', 0)
                }
            else:
                return {
                    "active_workers": 0,
                    "message": "Continuous AI system not running"
                }
                
        except Exception as e:
            return {"error": str(e)}
            
    async def _count_python_files(self, directory: str = None, **kwargs) -> Dict[str, Any]:
        """Count Python files specifically."""
        return await self._count_files(directory=directory or '.', pattern='*.py', **kwargs)
        
    async def _count_markdown_files(self, **kwargs) -> Dict[str, Any]:
        """Count Markdown files."""
        return await self._count_files(pattern='*.md', **kwargs)
        
    async def _count_todos(self, **kwargs) -> Dict[str, Any]:
        """Count TODO comments in code."""
        todos = []
        todo_pattern = re.compile(r'#\s*(TODO|FIXME|HACK|XXX|OPTIMIZE|BUG):\s*(.+)', re.IGNORECASE)
        
        for py_file in self.project_root.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        match = todo_pattern.search(line)
                        if match:
                            todos.append({
                                'file': str(py_file.relative_to(self.project_root)),
                                'line': line_num,
                                'type': match.group(1).upper(),
                                'message': match.group(2).strip()
                            })
            except:
                pass
                
        # Group by type
        by_type = defaultdict(list)
        for todo in todos:
            by_type[todo['type']].append(todo)
            
        return {
            "total_todos": len(todos),
            "by_type": {k: len(v) for k, v in by_type.items()},
            "todos": todos[:50],  # First 50
            "by_file": self._group_by_file(todos)
        }
        
    async def _count_async_functions(self, **kwargs) -> Dict[str, Any]:
        """Count async functions specifically."""
        return await self._count_functions(async_only=True, **kwargs)
        
    async def _calculate_complexity(self, **kwargs) -> Dict[str, Any]:
        """Calculate code complexity metrics."""
        complexity_data = []
        
        for py_file in self.project_root.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                # Simple complexity metrics
                lines = content.count('\n')
                functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
                classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
                imports = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)))
                
                # Cyclomatic complexity approximation
                branches = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)))
                
                complexity_data.append({
                    'file': str(py_file.relative_to(self.project_root)),
                    'lines': lines,
                    'functions': functions,
                    'classes': classes,
                    'imports': imports,
                    'complexity': branches + 1,  # McCabe complexity approximation
                    'complexity_per_function': round((branches + 1) / max(functions, 1), 2)
                })
                
            except:
                pass
                
        # Sort by complexity
        most_complex = sorted(complexity_data, key=lambda x: x['complexity'], reverse=True)[:10]
        
        return {
            "total_files_analyzed": len(complexity_data),
            "average_complexity": round(sum(f['complexity'] for f in complexity_data) / max(len(complexity_data), 1), 2),
            "total_lines": sum(f['lines'] for f in complexity_data),
            "total_functions": sum(f['functions'] for f in complexity_data),
            "total_classes": sum(f['classes'] for f in complexity_data),
            "most_complex_files": most_complex
        }
        
    async def _calculate_statistics(self, **kwargs) -> Dict[str, Any]:
        """Calculate comprehensive project statistics."""
        stats = {
            "files": await self._count_files(),
            "lines": await self._count_lines(),
            "repositories": await self._count_repositories(),
            "complexity": await self._calculate_complexity()
        }
        
        # Summary
        return {
            "project_size": {
                "total_files": stats['files']['total_files'],
                "total_lines": stats['lines']['total_lines'],
                "size_mb": stats['files']['total_size_mb']
            },
            "code_metrics": {
                "code_lines": stats['lines']['total_code_lines'],
                "comment_lines": stats['lines']['total_comment_lines'],
                "average_complexity": stats['complexity']['average_complexity']
            },
            "repository_health": {
                "total_repos": stats['repositories']['total_repositories'],
                "average_health": stats['repositories']['average_health'],
                "total_issues": stats['repositories']['total_open_issues']
            },
            "languages": list(stats['lines']['by_extension'].keys()),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    async def _count_issues(self, **kwargs) -> Dict[str, Any]:
        """Count GitHub issues across repositories."""
        state = self.state_manager.get_state()
        projects = state.get('projects', {})
        
        total_issues = 0
        by_repo = {}
        
        for repo, data in projects.items():
            if isinstance(data, dict):
                issues = data.get('metrics', {}).get('issues_open', 0)
                total_issues += issues
                by_repo[repo] = issues
                
        return {
            "total_open_issues": total_issues,
            "by_repository": by_repo,
            "repositories_with_issues": sum(1 for count in by_repo.values() if count > 0),
            "average_per_repo": round(total_issues / max(len(projects), 1), 1)
        }
        
    async def _count_opportunities(self, **kwargs) -> Dict[str, Any]:
        """Count business/improvement opportunities."""
        opportunities = {
            "code_improvements": 0,
            "performance_optimizations": 0,
            "security_fixes": 0,
            "feature_additions": 0,
            "refactoring_needs": 0
        }
        
        # Analyze TODOs for opportunities
        todos = await self._count_todos()
        for todo_type, count in todos['by_type'].items():
            if todo_type in ['OPTIMIZE', 'PERFORMANCE']:
                opportunities['performance_optimizations'] += count
            elif todo_type in ['SECURITY', 'HACK']:
                opportunities['security_fixes'] += count
            elif todo_type == 'TODO':
                opportunities['feature_additions'] += count
            elif todo_type in ['FIXME', 'BUG']:
                opportunities['code_improvements'] += count
                
        # Analyze complexity for refactoring needs
        complexity = await self._calculate_complexity()
        opportunities['refactoring_needs'] = len([f for f in complexity.get('most_complex_files', []) if f['complexity'] > 10])
        
        return {
            "total_opportunities": sum(opportunities.values()),
            "by_type": opportunities,
            "priority_areas": sorted(opportunities.items(), key=lambda x: x[1], reverse=True)[:3]
        }
        
    async def _intelligent_count(self, original_target: str, **kwargs) -> Dict[str, Any]:
        """Use AI to understand what to count."""
        prompt = f"""User wants to count: {original_target}
        
Available counting operations:
- files, lines, functions, repositories, workers
- python_files, markdown_files, todos, async_functions
- complexity, statistics, issues, opportunities

Determine what they want to count and provide the result."""
        
        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        if response.get('success'):
            # Extract the operation from AI response
            content = response.get('content', '').lower()
            
            # Try to match with our operations
            for operation in ['files', 'lines', 'functions', 'repositories', 'workers', 
                            'python_files', 'markdown_files', 'todos', 'async_functions',
                            'complexity', 'statistics', 'issues', 'opportunities']:
                if operation in content:
                    return await self.count(operation, **kwargs)
                    
        return {
            "error": f"Could not understand what to count: {original_target}",
            "suggestion": "Try: count('files'), count('lines'), count('functions'), etc."
        }
        
    def _group_by_file(self, items: List[Dict]) -> Dict[str, List]:
        """Group items by file."""
        by_file = defaultdict(list)
        for item in items:
            by_file[item.get('file', 'unknown')].append(item)
        return dict(sorted(by_file.items(), key=lambda x: len(x[1]), reverse=True)[:10])


async def main(**kwargs):
    """Entry point for the universal counter."""
    counter = UniversalCounter()
    
    # Determine what to count
    if 'target' in kwargs:
        return await counter.count(**kwargs)
    elif 'what' in kwargs:
        return await counter.count(target=kwargs.pop('what'), **kwargs)
    else:
        # Default to statistics
        return await counter._calculate_statistics()


if __name__ == "__main__":
    # Test the counter
    import asyncio
    
    async def test():
        counter = UniversalCounter()
        
        # Test different counting operations
        print("Files:", await counter.count('files'))
        print("\nPython files:", await counter.count('python_files'))
        print("\nRepositories:", await counter.count('repositories'))
        print("\nStatistics:", await counter.count('statistics'))
        
    asyncio.run(test())