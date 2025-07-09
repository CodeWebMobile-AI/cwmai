"""
Code Analyzer

Unified tool for all code analysis operations including complexity analysis,
security scanning, dependency checking, and quality assessment.
Consolidates 16 analysis tools into one comprehensive system.

Capabilities:
- Advanced code analysis and metrics
- Complexity and quality assessment
- Security vulnerability scanning
- Dependency analysis and checking
- Performance profiling
- Code smell detection
- Best practices validation
- Technology stack analysis
"""

import ast
import os
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timezone
from collections import defaultdict, Counter
import hashlib

from scripts.http_ai_client import HTTPAIClient
from scripts.state_manager import StateManager


class CodeAnalyzer:
    """Comprehensive code analysis system."""
    
    def __init__(self):
        self.ai_client = HTTPAIClient()
        self.state_manager = StateManager()
        self.project_root = Path(__file__).parent.parent.parent
        
        # Security patterns to check
        self.security_patterns = {
            'hardcoded_secrets': [
                r'api[_-]?key\s*=\s*["\'][^"\']+["\']',
                r'password\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']'
            ],
            'sql_injection': [
                r'execute\s*\(\s*["\'].*%s.*["\']',
                r'cursor\.execute\s*\(\s*f["\']',
                r'query\s*=\s*["\'].*\+.*["\']'
            ],
            'command_injection': [
                r'subprocess\.\w+\s*\([^)]*shell\s*=\s*True',
                r'os\.system\s*\(',
                r'eval\s*\(',
                r'exec\s*\('
            ],
            'path_traversal': [
                r'open\s*\([^)]*\.\./.*\)',
                r'Path\s*\([^)]*\.\./.*\)'
            ]
        }
        
        # Code quality indicators
        self.quality_indicators = {
            'good': [
                r'def\s+test_',  # Has tests
                r'"""[\s\S]+?"""',  # Has docstrings
                r'# type:',  # Type hints
                r'@\w+',  # Uses decorators
                r'try:',  # Error handling
                r'logger\.',  # Logging
            ],
            'bad': [
                r'except\s*:',  # Bare except
                r'import\s+\*',  # Star imports
                r'global\s+',  # Global variables
                r'TODO|FIXME|HACK',  # Unfinished work
                r'print\s*\(',  # Debug prints
            ]
        }
        
    async def analyze(self, target: str = '.', analysis_type: str = 'comprehensive', **kwargs) -> Dict[str, Any]:
        """Main analysis entry point.
        
        Args:
            target: File or directory to analyze
            analysis_type: Type of analysis (comprehensive, security, quality, etc.)
            **kwargs: Additional parameters
            
        Returns:
            Analysis results
        """
        target_path = Path(target) if target != '.' else self.project_root
        
        if not target_path.exists():
            return {"error": f"Target {target} not found"}
            
        # Route to appropriate analyzer
        analyzers = {
            'comprehensive': self._comprehensive_analysis,
            'security': self._security_analysis,
            'complexity': self._complexity_analysis,
            'quality': self._quality_analysis,
            'dependencies': self._dependency_analysis,
            'performance': self._performance_analysis,
            'technology': self._technology_analysis,
            'best_practices': self._best_practices_analysis,
            'monetization': self._monetization_analysis,
            'create_tool': self._analyze_for_tool_creation
        }
        
        analyzer = analyzers.get(analysis_type, self._comprehensive_analysis)
        return await analyzer(target_path, **kwargs)
        
    async def _comprehensive_analysis(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Perform comprehensive code analysis."""
        results = {
            "path": str(path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {}
        }
        
        # Run all analyses
        results['security'] = await self._security_analysis(path, detailed=False)
        results['complexity'] = await self._complexity_analysis(path, detailed=False)
        results['quality'] = await self._quality_analysis(path, detailed=False)
        results['dependencies'] = await self._dependency_analysis(path, detailed=False)
        results['technology'] = await self._technology_analysis(path)
        
        # Generate summary
        results['summary'] = {
            'overall_health': self._calculate_overall_health(results),
            'security_score': results['security'].get('score', 0),
            'quality_score': results['quality'].get('score', 0),
            'complexity_score': results['complexity'].get('score', 0),
            'recommendations': self._generate_recommendations(results)
        }
        
        return results
        
    async def _security_analysis(self, path: Path, detailed: bool = True, **kwargs) -> Dict[str, Any]:
        """Analyze code for security vulnerabilities."""
        vulnerabilities = defaultdict(list)
        total_files = 0
        scanned_files = 0
        
        # Define file extensions to scan
        scan_extensions = {'.py', '.js', '.java', '.go', '.php', '.rb', '.cs'}
        
        for file_path in self._get_files(path, scan_extensions):
            total_files += 1
            try:
                content = file_path.read_text(encoding='utf-8')
                scanned_files += 1
                
                # Check each security pattern
                for vuln_type, patterns in self.security_patterns.items():
                    for pattern in patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            vulnerabilities[vuln_type].append({
                                'file': str(file_path.relative_to(self.project_root)),
                                'line': line_num,
                                'code': match.group(0)[:100],
                                'severity': self._get_severity(vuln_type)
                            })
                            
            except Exception as e:
                continue
                
        # Calculate security score
        total_vulns = sum(len(vulns) for vulns in vulnerabilities.values())
        score = max(0, 100 - (total_vulns * 10))
        
        result = {
            'score': score,
            'scanned_files': scanned_files,
            'total_files': total_files,
            'vulnerabilities_found': total_vulns,
            'by_type': {k: len(v) for k, v in vulnerabilities.items()}
        }
        
        if detailed:
            result['vulnerabilities'] = dict(vulnerabilities)
            result['recommendations'] = self._generate_security_recommendations(vulnerabilities)
            
        return result
        
    async def _complexity_analysis(self, path: Path, detailed: bool = True, **kwargs) -> Dict[str, Any]:
        """Analyze code complexity."""
        complexity_data = []
        total_complexity = 0
        
        for py_file in self._get_files(path, {'.py'}):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                file_complexity = self._calculate_file_complexity(tree, py_file, content)
                complexity_data.append(file_complexity)
                total_complexity += file_complexity['complexity']
                
            except Exception:
                continue
                
        # Calculate metrics
        avg_complexity = total_complexity / max(len(complexity_data), 1)
        score = max(0, 100 - (avg_complexity * 5))
        
        # Find most complex files
        most_complex = sorted(complexity_data, key=lambda x: x['complexity'], reverse=True)[:10]
        
        result = {
            'score': score,
            'average_complexity': round(avg_complexity, 2),
            'files_analyzed': len(complexity_data),
            'most_complex_files': most_complex if detailed else len(most_complex)
        }
        
        if detailed:
            result['distribution'] = self._get_complexity_distribution(complexity_data)
            result['recommendations'] = self._generate_complexity_recommendations(complexity_data)
            
        return result
        
    async def _quality_analysis(self, path: Path, detailed: bool = True, **kwargs) -> Dict[str, Any]:
        """Analyze code quality."""
        quality_metrics = {
            'good_practices': defaultdict(int),
            'bad_practices': defaultdict(int),
            'files_analyzed': 0,
            'total_lines': 0,
            'comment_lines': 0,
            'blank_lines': 0
        }
        
        for file_path in self._get_files(path, {'.py', '.js', '.java', '.go'}):
            quality_metrics['files_analyzed'] += 1
            try:
                content = file_path.read_text(encoding='utf-8')
                lines = content.split('\n')
                quality_metrics['total_lines'] += len(lines)
                
                # Count line types
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        quality_metrics['blank_lines'] += 1
                    elif stripped.startswith(('#', '//', '/*', '*')):
                        quality_metrics['comment_lines'] += 1
                        
                # Check quality indicators
                for indicator_type, patterns in self.quality_indicators.items():
                    for pattern in patterns:
                        matches = len(re.findall(pattern, content, re.MULTILINE))
                        if matches > 0:
                            pattern_name = self._get_pattern_name(pattern)
                            quality_metrics[f'{indicator_type}_practices'][pattern_name] += matches
                            
            except Exception:
                continue
                
        # Calculate quality score
        good_score = sum(quality_metrics['good_practices'].values())
        bad_score = sum(quality_metrics['bad_practices'].values())
        score = min(100, max(0, 70 + (good_score * 2) - (bad_score * 5)))
        
        # Calculate code-to-comment ratio
        code_lines = quality_metrics['total_lines'] - quality_metrics['comment_lines'] - quality_metrics['blank_lines']
        comment_ratio = quality_metrics['comment_lines'] / max(code_lines, 1)
        
        result = {
            'score': score,
            'files_analyzed': quality_metrics['files_analyzed'],
            'code_to_comment_ratio': round(comment_ratio, 2),
            'good_practices_found': sum(quality_metrics['good_practices'].values()),
            'bad_practices_found': sum(quality_metrics['bad_practices'].values())
        }
        
        if detailed:
            result['good_practices'] = dict(quality_metrics['good_practices'])
            result['bad_practices'] = dict(quality_metrics['bad_practices'])
            result['recommendations'] = self._generate_quality_recommendations(quality_metrics)
            
        return result
        
    async def _dependency_analysis(self, path: Path, detailed: bool = True, **kwargs) -> Dict[str, Any]:
        """Analyze project dependencies."""
        dependencies = {
            'python': self._analyze_python_deps(path),
            'javascript': self._analyze_js_deps(path),
            'total': 0,
            'outdated': [],
            'security_issues': []
        }
        
        # Count total dependencies
        for lang, deps in dependencies.items():
            if isinstance(deps, dict):
                dependencies['total'] += len(deps.get('packages', []))
                
        # Simple outdated check (would need real version checking)
        # This is a placeholder - real implementation would check against package registries
        
        result = {
            'total_dependencies': dependencies['total'],
            'by_language': {k: len(v.get('packages', [])) if isinstance(v, dict) else 0 
                           for k, v in dependencies.items() if k not in ['total', 'outdated', 'security_issues']}
        }
        
        if detailed:
            result['dependencies'] = dependencies
            result['recommendations'] = self._generate_dependency_recommendations(dependencies)
            
        return result
        
    async def _performance_analysis(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Analyze code for performance issues."""
        performance_issues = []
        
        # Performance anti-patterns
        patterns = {
            'nested_loops': r'for\s+.*:\s*\n\s*for\s+.*:',
            'inefficient_string_concat': r'[\w\s]+=\s*[\w\s]+\s*\+\s*["\']',
            'sync_in_async': r'async\s+def.*\n.*time\.sleep',
            'large_list_comprehension': r'\[.*for.*for.*\]',
            'repeated_calculations': r'for.*:\s*\n.*len\s*\('
        }
        
        for file_path in self._get_files(path, {'.py'}):
            try:
                content = file_path.read_text(encoding='utf-8')
                
                for issue_type, pattern in patterns.items():
                    matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        performance_issues.append({
                            'type': issue_type,
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': line_num,
                            'severity': 'medium'
                        })
                        
            except Exception:
                continue
                
        return {
            'issues_found': len(performance_issues),
            'by_type': Counter(issue['type'] for issue in performance_issues),
            'top_issues': performance_issues[:10],
            'recommendations': self._generate_performance_recommendations(performance_issues)
        }
        
    async def _technology_analysis(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Analyze technology stack."""
        tech_stack = {
            'languages': defaultdict(int),
            'frameworks': [],
            'databases': [],
            'tools': []
        }
        
        # Count files by extension
        for file_path in path.rglob('*'):
            if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
                ext = file_path.suffix
                if ext:
                    tech_stack['languages'][ext] += 1
                    
        # Detect frameworks and tools from files
        framework_indicators = {
            'django': ['django', 'manage.py', 'settings.py'],
            'flask': ['flask', 'app.py'],
            'react': ['react', 'package.json', 'jsx'],
            'vue': ['vue', '.vue'],
            'docker': ['Dockerfile', 'docker-compose'],
            'kubernetes': ['k8s', '.yaml', 'deployment']
        }
        
        for tech, indicators in framework_indicators.items():
            for root, dirs, files in os.walk(path):
                if any(indicator in str(root).lower() or any(indicator in f.lower() for f in files) 
                      for indicator in indicators):
                    if tech not in tech_stack['frameworks']:
                        tech_stack['frameworks'].append(tech)
                        
        # Sort languages by usage
        sorted_langs = sorted(tech_stack['languages'].items(), key=lambda x: x[1], reverse=True)
        
        return {
            'primary_language': sorted_langs[0][0] if sorted_langs else 'unknown',
            'all_languages': dict(sorted_langs[:10]),
            'frameworks': tech_stack['frameworks'],
            'total_files': sum(tech_stack['languages'].values())
        }
        
    async def _best_practices_analysis(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Analyze adherence to best practices."""
        practices = {
            'has_readme': (path / 'README.md').exists(),
            'has_tests': any(path.rglob('test_*.py')) or any(path.rglob('*_test.py')),
            'has_ci': any(path.rglob('.github/workflows/*.yml')),
            'has_gitignore': (path / '.gitignore').exists(),
            'has_requirements': (path / 'requirements.txt').exists() or (path / 'package.json').exists(),
            'has_license': (path / 'LICENSE').exists() or (path / 'LICENSE.md').exists(),
            'has_contributing': (path / 'CONTRIBUTING.md').exists(),
            'has_code_of_conduct': (path / 'CODE_OF_CONDUCT.md').exists()
        }
        
        score = sum(10 for practice, exists in practices.items() if exists)
        
        return {
            'score': score,
            'practices': practices,
            'missing': [k for k, v in practices.items() if not v],
            'recommendations': self._generate_best_practices_recommendations(practices)
        }
        
    async def _monetization_analysis(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Analyze project for monetization opportunities."""
        # Analyze project characteristics
        tech_analysis = await self._technology_analysis(path)
        quality_analysis = await self._quality_analysis(path, detailed=False)
        
        # Use AI to suggest monetization strategies
        prompt = f"""Analyze this project for monetization opportunities:
Technology: {json.dumps(tech_analysis, indent=2)}
Quality Score: {quality_analysis['score']}

Suggest:
1. Business models that fit this project
2. Monetization strategies
3. Target markets
4. Revenue potential
5. Implementation steps"""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        if response.get('success'):
            return {
                'analysis': response.get('content', ''),
                'project_type': self._determine_project_type(tech_analysis),
                'market_fit': self._assess_market_fit(tech_analysis, quality_analysis)
            }
            
        return {'error': 'Monetization analysis failed'}
        
    async def _analyze_for_tool_creation(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Analyze code to help create a tool like Claude."""
        # This would analyze how Claude creates tools
        structure = {
            'tool_patterns': [],
            'common_imports': [],
            'function_signatures': [],
            'design_patterns': []
        }
        
        # Look for tool creation patterns
        for py_file in self._get_files(path, {'.py'}):
            if 'tool' in py_file.stem.lower():
                try:
                    content = py_file.read_text(encoding='utf-8')
                    tree = ast.parse(content)
                    
                    # Extract patterns
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            structure['function_signatures'].append({
                                'name': node.name,
                                'args': [arg.arg for arg in node.args.args],
                                'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
                            })
                            
                except Exception:
                    continue
                    
        return {
            'tool_creation_patterns': structure,
            'recommendations': [
                "Use async functions for I/O operations",
                "Return structured Dict[str, Any] results",
                "Include error handling with try/except",
                "Add comprehensive docstrings",
                "Follow existing tool naming conventions"
            ]
        }
        
    def _get_files(self, path: Path, extensions: Set[str]) -> List[Path]:
        """Get all files with specified extensions."""
        files = []
        for ext in extensions:
            files.extend(path.rglob(f'*{ext}'))
        return [f for f in files if f.is_file() and not any(part.startswith('.') for part in f.parts)]
        
    def _calculate_file_complexity(self, tree: ast.AST, file_path: Path, content: str) -> Dict[str, Any]:
        """Calculate complexity metrics for a file."""
        complexity = 1  # Base complexity
        functions = 0
        classes = 0
        max_depth = 0
        
        for node in ast.walk(tree):
            # Increment complexity for control structures
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                functions += 1
            elif isinstance(node, ast.ClassDef):
                classes += 1
                
        # Calculate nesting depth
        lines = content.split('\n')
        for line in lines:
            depth = (len(line) - len(line.lstrip())) // 4  # Assuming 4-space indent
            max_depth = max(max_depth, depth)
            
        return {
            'file': str(file_path.relative_to(self.project_root)),
            'complexity': complexity,
            'functions': functions,
            'classes': classes,
            'max_depth': max_depth,
            'lines': len(lines)
        }
        
    def _get_severity(self, vuln_type: str) -> str:
        """Get severity level for vulnerability type."""
        high_severity = ['sql_injection', 'command_injection', 'hardcoded_secrets']
        medium_severity = ['path_traversal']
        
        if vuln_type in high_severity:
            return 'high'
        elif vuln_type in medium_severity:
            return 'medium'
        else:
            return 'low'
            
    def _get_pattern_name(self, pattern: str) -> str:
        """Get human-readable name for pattern."""
        pattern_names = {
            r'def\s+test_': 'has_tests',
            r'"""[\s\S]+?"""': 'has_docstrings',
            r'# type:': 'has_type_hints',
            r'@\w+': 'uses_decorators',
            r'try:': 'has_error_handling',
            r'logger\.': 'uses_logging',
            r'except\s*:': 'bare_except',
            r'import\s+\*': 'star_imports',
            r'global\s+': 'global_variables',
            r'TODO|FIXME|HACK': 'unfinished_work',
            r'print\s*\(': 'debug_prints'
        }
        
        return pattern_names.get(pattern, pattern[:20])
        
    def _get_complexity_distribution(self, data: List[Dict]) -> Dict[str, int]:
        """Get distribution of complexity scores."""
        distribution = defaultdict(int)
        
        for item in data:
            complexity = item['complexity']
            if complexity <= 5:
                distribution['low'] += 1
            elif complexity <= 10:
                distribution['medium'] += 1
            elif complexity <= 20:
                distribution['high'] += 1
            else:
                distribution['very_high'] += 1
                
        return dict(distribution)
        
    def _analyze_python_deps(self, path: Path) -> Dict[str, Any]:
        """Analyze Python dependencies."""
        deps = {'packages': [], 'files': []}
        
        # Check requirements.txt
        req_file = path / 'requirements.txt'
        if req_file.exists():
            deps['files'].append('requirements.txt')
            try:
                content = req_file.read_text()
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        pkg = line.split('==')[0].split('>=')[0].split('<=')[0]
                        deps['packages'].append(pkg)
            except:
                pass
                
        # Check setup.py
        setup_file = path / 'setup.py'
        if setup_file.exists():
            deps['files'].append('setup.py')
            
        # Check Pipfile
        pipfile = path / 'Pipfile'
        if pipfile.exists():
            deps['files'].append('Pipfile')
            
        return deps
        
    def _analyze_js_deps(self, path: Path) -> Dict[str, Any]:
        """Analyze JavaScript dependencies."""
        deps = {'packages': [], 'files': []}
        
        # Check package.json
        pkg_file = path / 'package.json'
        if pkg_file.exists():
            deps['files'].append('package.json')
            try:
                with open(pkg_file) as f:
                    data = json.load(f)
                    deps['packages'].extend(data.get('dependencies', {}).keys())
                    deps['packages'].extend(data.get('devDependencies', {}).keys())
            except:
                pass
                
        return deps
        
    def _calculate_overall_health(self, results: Dict[str, Any]) -> int:
        """Calculate overall health score."""
        scores = []
        
        if 'security' in results:
            scores.append(results['security'].get('score', 0))
        if 'quality' in results:
            scores.append(results['quality'].get('score', 0))
        if 'complexity' in results:
            scores.append(results['complexity'].get('score', 0))
            
        return round(sum(scores) / max(len(scores), 1))
        
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations."""
        recommendations = []
        
        # Security recommendations
        if results.get('security', {}).get('score', 100) < 70:
            recommendations.append("Address security vulnerabilities immediately")
            
        # Quality recommendations
        if results.get('quality', {}).get('score', 100) < 70:
            recommendations.append("Improve code quality by reducing bad practices")
            
        # Complexity recommendations
        if results.get('complexity', {}).get('score', 100) < 70:
            recommendations.append("Refactor complex code to improve maintainability")
            
        return recommendations
        
    def _generate_security_recommendations(self, vulnerabilities: Dict) -> List[str]:
        """Generate security-specific recommendations."""
        recs = []
        
        if vulnerabilities.get('hardcoded_secrets'):
            recs.append("Move secrets to environment variables or secure vaults")
        if vulnerabilities.get('sql_injection'):
            recs.append("Use parameterized queries to prevent SQL injection")
        if vulnerabilities.get('command_injection'):
            recs.append("Avoid shell=True and validate all user inputs")
            
        return recs
        
    def _generate_complexity_recommendations(self, data: List[Dict]) -> List[str]:
        """Generate complexity recommendations."""
        recs = []
        
        high_complexity = [d for d in data if d['complexity'] > 10]
        if high_complexity:
            recs.append(f"Refactor {len(high_complexity)} files with high complexity")
            
        deep_nesting = [d for d in data if d.get('max_depth', 0) > 5]
        if deep_nesting:
            recs.append(f"Reduce nesting depth in {len(deep_nesting)} files")
            
        return recs
        
    def _generate_quality_recommendations(self, metrics: Dict) -> List[str]:
        """Generate quality recommendations."""
        recs = []
        
        if metrics['bad_practices'].get('bare_except', 0) > 0:
            recs.append("Replace bare except clauses with specific exceptions")
        if metrics['bad_practices'].get('global_variables', 0) > 0:
            recs.append("Eliminate global variables for better encapsulation")
        if metrics['good_practices'].get('has_tests', 0) == 0:
            recs.append("Add unit tests to improve reliability")
            
        return recs
        
    def _generate_dependency_recommendations(self, deps: Dict) -> List[str]:
        """Generate dependency recommendations."""
        recs = []
        
        if deps['total'] > 50:
            recs.append("Consider reducing dependencies to minimize complexity")
        if not any(deps.get(lang, {}).get('files') for lang in ['python', 'javascript']):
            recs.append("Add proper dependency management files")
            
        return recs
        
    def _generate_performance_recommendations(self, issues: List[Dict]) -> List[str]:
        """Generate performance recommendations."""
        recs = []
        
        issue_types = Counter(issue['type'] for issue in issues)
        
        if issue_types.get('nested_loops', 0) > 0:
            recs.append("Optimize nested loops using better algorithms")
        if issue_types.get('inefficient_string_concat', 0) > 0:
            recs.append("Use join() instead of + for string concatenation")
            
        return recs
        
    def _generate_best_practices_recommendations(self, practices: Dict) -> List[str]:
        """Generate best practices recommendations."""
        recs = []
        
        if not practices['has_readme']:
            recs.append("Add README.md with project overview and setup instructions")
        if not practices['has_tests']:
            recs.append("Implement unit tests for better reliability")
        if not practices['has_ci']:
            recs.append("Set up continuous integration for automated testing")
            
        return recs
        
    def _determine_project_type(self, tech: Dict) -> str:
        """Determine project type from technology analysis."""
        primary_lang = tech.get('primary_language', '').lower()
        frameworks = tech.get('frameworks', [])
        
        if 'react' in frameworks or 'vue' in frameworks:
            return 'web_frontend'
        elif 'django' in frameworks or 'flask' in frameworks:
            return 'web_backend'
        elif primary_lang == '.py':
            return 'python_library'
        elif primary_lang in ['.js', '.ts']:
            return 'javascript_library'
        else:
            return 'general_software'
            
    def _assess_market_fit(self, tech: Dict, quality: Dict) -> str:
        """Assess market fit based on analysis."""
        score = quality.get('score', 0)
        
        if score >= 80:
            return 'high'
        elif score >= 60:
            return 'medium'
        else:
            return 'low'


async def main(target: str = '.', analysis_type: str = 'comprehensive', **kwargs):
    """Entry point for code analyzer."""
    analyzer = CodeAnalyzer()
    return await analyzer.analyze(target, analysis_type, **kwargs)
    

if __name__ == "__main__":
    # Test the analyzer
    import asyncio
    
    async def test():
        analyzer = CodeAnalyzer()
        
        # Test different analyses
        print("Comprehensive:", await analyzer.analyze('.', 'comprehensive'))
        print("\nSecurity:", await analyzer.analyze('.', 'security'))
        print("\nQuality:", await analyzer.analyze('.', 'quality'))
        
    asyncio.run(test())