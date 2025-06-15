"""
Context-Aware Improver

Understands broader codebase context to make more intelligent improvements.
"""

import os
import ast
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict

from ai_brain import IntelligentAIBrain
from ai_code_analyzer import AICodeAnalyzer, CodeImprovement
from improvement_learning_system import ImprovementLearningSystem
from safe_self_improver import ModificationType


@dataclass
class CodeContext:
    """Represents the context of code within a project."""
    file_path: str
    imports: Dict[str, List[str]]  # module -> [imported names]
    exports: List[str]  # functions/classes exported
    dependencies: Set[str]  # files this depends on
    dependents: Set[str]  # files that depend on this
    api_surface: List[str]  # public API elements
    test_coverage: Optional[float] = None
    last_modified: Optional[float] = None
    complexity_score: int = 0


class ContextAwareImprover:
    """Makes improvements considering the broader codebase context."""
    
    def __init__(self, ai_brain: IntelligentAIBrain, repo_path: str):
        """Initialize with AI brain and repository path.
        
        Args:
            ai_brain: The AI brain for intelligent analysis
            repo_path: Path to the repository
        """
        self.ai_brain = ai_brain
        self.repo_path = repo_path
        self.analyzer = AICodeAnalyzer(ai_brain)
        self.learning_system = ImprovementLearningSystem()
        
        self.dependency_graph = nx.DiGraph()
        self.file_contexts = {}
        self.api_registry = defaultdict(list)  # api_name -> [file_paths]
        
        # Build initial context
        self._build_codebase_context()
    
    def _build_codebase_context(self):
        """Build understanding of the codebase structure."""
        print("Building codebase context...")
        
        # Find all Python files
        python_files = []
        for root, _, files in os.walk(self.repo_path):
            # Skip hidden directories and common non-source directories
            if any(part.startswith('.') for part in root.split(os.sep)):
                continue
            if any(skip in root for skip in ['__pycache__', 'venv', 'env', 'node_modules']):
                continue
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    python_files.append(file_path)
        
        # Analyze each file
        for file_path in python_files:
            try:
                context = self._analyze_file_context(file_path)
                self.file_contexts[file_path] = context
                
                # Build dependency graph
                self.dependency_graph.add_node(file_path)
                for dep in context.dependencies:
                    self.dependency_graph.add_edge(file_path, dep)
                
                # Register APIs
                for api in context.api_surface:
                    self.api_registry[api].append(file_path)
                    
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
    
    def _analyze_file_context(self, file_path: str) -> CodeContext:
        """Analyze a single file's context."""
        with open(file_path, 'r') as f:
            content = f.read()
        
        context = CodeContext(
            file_path=file_path,
            imports={},
            exports=[],
            dependencies=set(),
            dependents=set(),
            api_surface=[],
            last_modified=os.path.getmtime(file_path)
        )
        
        try:
            tree = ast.parse(content)
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name
                        context.imports[module] = ['*']
                        self._resolve_dependency(module, file_path, context)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        names = [n.name for n in node.names]
                        context.imports[node.module] = names
                        self._resolve_dependency(node.module, file_path, context)
                
                # Extract exports (top-level functions and classes)
                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                    context.exports.append(node.name)
                    if not node.name.startswith('_'):
                        context.api_surface.append(node.name)
                        
                elif isinstance(node, ast.ClassDef) and node.col_offset == 0:
                    context.exports.append(node.name)
                    if not node.name.startswith('_'):
                        context.api_surface.append(node.name)
            
            # Calculate complexity
            context.complexity_score = self._calculate_file_complexity(tree)
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        
        return context
    
    def _resolve_dependency(self, module: str, from_file: str, context: CodeContext):
        """Resolve a module import to a file dependency."""
        # Handle relative imports
        if module.startswith('.'):
            base_dir = os.path.dirname(from_file)
            levels = len(module) - len(module.lstrip('.'))
            
            for _ in range(levels - 1):
                base_dir = os.path.dirname(base_dir)
            
            module_path = module.lstrip('.')
            if module_path:
                potential_file = os.path.join(base_dir, module_path.replace('.', os.sep) + '.py')
                if os.path.exists(potential_file):
                    context.dependencies.add(potential_file)
        else:
            # Handle absolute imports within the project
            parts = module.split('.')
            if parts[0] in ['scripts', 'tests']:  # Known project packages
                potential_file = os.path.join(self.repo_path, module.replace('.', os.sep) + '.py')
                if os.path.exists(potential_file):
                    context.dependencies.add(potential_file)
    
    def _calculate_file_complexity(self, tree: ast.AST) -> int:
        """Calculate complexity score for a file."""
        complexity = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity += 1
                # Add complexity for control structures
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        complexity += 1
            elif isinstance(node, ast.ClassDef):
                complexity += 2
        
        return complexity
    
    async def find_improvements_with_context(self, file_path: str,
                                           max_improvements: int = 10) -> List[CodeImprovement]:
        """Find improvements considering the file's context in the codebase.
        
        Args:
            file_path: Path to the file to analyze
            max_improvements: Maximum number of improvements to return
            
        Returns:
            List of context-aware improvements
        """
        # Get file context
        if file_path not in self.file_contexts:
            self.file_contexts[file_path] = self._analyze_file_context(file_path)
        
        context = self.file_contexts[file_path]
        
        # Determine impact radius
        impact_radius = self._calculate_impact_radius(file_path)
        
        # Build enhanced context for AI
        enhanced_context = {
            'file_type': self._determine_file_type(file_path),
            'imports': dict(context.imports),
            'exports': context.exports,
            'api_surface': context.api_surface,
            'complexity': context.complexity_score,
            'dependencies': list(context.dependencies),
            'dependents': list(self._get_dependents(file_path)),
            'impact_radius': impact_radius,
            'critical_path': self._is_on_critical_path(file_path),
            'test_coverage': self._estimate_test_coverage(file_path)
        }
        
        # Get AI improvements with context
        improvements = await self.analyzer.analyze_code_for_improvements(
            self._read_file(file_path),
            file_path,
            enhanced_context
        )
        
        # Score and filter improvements based on context
        scored_improvements = []
        for improvement in improvements:
            score = await self._score_improvement_with_context(improvement, context, impact_radius)
            if score > 0.5:  # Threshold
                improvement.confidence = score
                scored_improvements.append((score, improvement))
        
        # Sort by score and return top improvements
        scored_improvements.sort(key=lambda x: x[0], reverse=True)
        return [imp for _, imp in scored_improvements[:max_improvements]]
    
    def _calculate_impact_radius(self, file_path: str) -> Dict[str, Any]:
        """Calculate how changes to this file might impact the codebase."""
        impact = {
            'direct_dependents': 0,
            'transitive_dependents': 0,
            'api_consumers': 0,
            'critical': False
        }
        
        # Direct dependents
        if file_path in self.dependency_graph:
            direct = list(self.dependency_graph.predecessors(file_path))
            impact['direct_dependents'] = len(direct)
            
            # Transitive dependents (up to 2 levels)
            transitive = set()
            for dep in direct:
                transitive.update(self.dependency_graph.predecessors(dep))
            impact['transitive_dependents'] = len(transitive)
        
        # API consumers
        context = self.file_contexts.get(file_path)
        if context:
            for api in context.api_surface:
                impact['api_consumers'] += len(self.api_registry.get(api, [])) - 1
        
        # Determine if critical
        impact['critical'] = (
            impact['direct_dependents'] > 5 or
            impact['api_consumers'] > 3 or
            'auth' in file_path.lower() or
            'security' in file_path.lower()
        )
        
        return impact
    
    def _get_dependents(self, file_path: str) -> Set[str]:
        """Get files that depend on this file."""
        if file_path in self.dependency_graph:
            return set(self.dependency_graph.predecessors(file_path))
        return set()
    
    def _is_on_critical_path(self, file_path: str) -> bool:
        """Check if file is on a critical path in the dependency graph."""
        # Simple heuristic: file is critical if it's imported by many files
        # or if it's in the import chain of main entry points
        dependents = self._get_dependents(file_path)
        
        if len(dependents) > 5:
            return True
        
        # Check for main entry points
        entry_points = ['main.py', 'app.py', '__main__.py', 'run_', 'server.py']
        for dep in dependents:
            if any(entry in os.path.basename(dep) for entry in entry_points):
                return True
        
        return False
    
    def _estimate_test_coverage(self, file_path: str) -> float:
        """Estimate test coverage for a file."""
        # Simple heuristic: look for corresponding test file
        base_name = os.path.basename(file_path).replace('.py', '')
        test_patterns = [
            f'test_{base_name}.py',
            f'{base_name}_test.py',
            f'tests/test_{base_name}.py',
            f'tests/{base_name}_test.py'
        ]
        
        for pattern in test_patterns:
            test_path = os.path.join(self.repo_path, pattern)
            if os.path.exists(test_path):
                # Rough estimate based on test file size
                test_size = os.path.getsize(test_path)
                source_size = os.path.getsize(file_path)
                return min(1.0, test_size / source_size)
        
        return 0.0
    
    async def _score_improvement_with_context(self, improvement: CodeImprovement,
                                            context: CodeContext,
                                            impact_radius: Dict[str, Any]) -> float:
        """Score an improvement considering its context."""
        # Base score from AI confidence
        score = improvement.confidence
        
        # Adjust based on impact radius
        if impact_radius['critical']:
            # Be more conservative with critical files
            score *= 0.7
            
            # But boost if it's a security or bug fix
            if improvement.type == ModificationType.SECURITY:
                score *= 1.5
        
        # Adjust based on file complexity
        if context.complexity_score > 50:
            # Complex files need more careful changes
            score *= 0.8
        
        # Boost for well-tested files
        test_coverage = self._estimate_test_coverage(context.file_path)
        if test_coverage > 0.7:
            score *= 1.2
        
        # Learn from past improvements
        learning_score = self.learning_system.score_improvement(
            improvement,
            {'file_type': self._determine_file_type(context.file_path)}
        )
        
        # Weighted average
        final_score = 0.6 * score + 0.4 * learning_score
        
        # Ask AI for risk assessment if high impact
        if impact_radius['direct_dependents'] > 3:
            risk_score = await self._assess_improvement_risk(improvement, context, impact_radius)
            final_score *= (1.0 - risk_score * 0.5)
        
        return min(1.0, final_score)
    
    async def _assess_improvement_risk(self, improvement: CodeImprovement,
                                     context: CodeContext,
                                     impact_radius: Dict[str, Any]) -> float:
        """Ask AI to assess the risk of an improvement."""
        prompt = f"""
Assess the risk of this code improvement:

File: {context.file_path}
Improvement Type: {improvement.type.value}
Description: {improvement.description}

Context:
- Direct dependents: {impact_radius['direct_dependents']}
- API consumers: {impact_radius['api_consumers']}
- File exports: {', '.join(context.api_surface[:5])}

Original code:
{improvement.original_code}

Improved code:
{improvement.improved_code}

Rate the risk from 0.0 (no risk) to 1.0 (very risky).
Consider:
1. Breaking changes to API
2. Behavioral changes
3. Performance regressions
4. Compatibility issues

Return just the risk score as a number.
"""
        
        response_dict = await self.ai_brain.generate_enhanced_response(prompt)
        response = response_dict.get('content', '')
        
        try:
            risk = float(response.strip())
            return max(0.0, min(1.0, risk))
        except:
            return 0.5  # Default medium risk
    
    def _determine_file_type(self, file_path: str) -> str:
        """Determine the type of file."""
        path_lower = file_path.lower()
        base_name = os.path.basename(path_lower)
        
        if 'test' in path_lower:
            return 'test'
        elif base_name == '__init__.py':
            return 'package_init'
        elif base_name == 'setup.py':
            return 'setup'
        elif any(x in path_lower for x in ['cli', 'command', 'cmd']):
            return 'cli'
        elif any(x in path_lower for x in ['api', 'route', 'endpoint']):
            return 'api'
        elif any(x in path_lower for x in ['model', 'schema', 'entity']):
            return 'model'
        elif any(x in path_lower for x in ['util', 'helper', 'tool']):
            return 'utility'
        elif any(x in path_lower for x in ['config', 'settings']):
            return 'configuration'
        else:
            return 'general'
    
    def _read_file(self, file_path: str) -> str:
        """Read file contents."""
        with open(file_path, 'r') as f:
            return f.read()
    
    def get_improvement_impact_analysis(self, file_path: str,
                                      improvement: CodeImprovement) -> Dict[str, Any]:
        """Analyze the potential impact of an improvement."""
        context = self.file_contexts.get(file_path)
        if not context:
            return {'error': 'No context available'}
        
        impact_radius = self._calculate_impact_radius(file_path)
        
        # Analyze what might be affected
        affected_apis = []
        for api in context.api_surface:
            # Check if the improvement affects this API
            if api in improvement.original_code:
                affected_apis.append(api)
        
        # Find potentially affected files
        affected_files = []
        if affected_apis:
            for api in affected_apis:
                affected_files.extend(self.api_registry.get(api, []))
        
        return {
            'file': file_path,
            'improvement_type': improvement.type.value,
            'affected_apis': affected_apis,
            'potentially_affected_files': list(set(affected_files) - {file_path}),
            'direct_dependents': impact_radius['direct_dependents'],
            'total_impact': impact_radius['transitive_dependents'],
            'risk_level': 'high' if impact_radius['critical'] else 'medium' if affected_apis else 'low',
            'recommended_tests': self._recommend_tests(improvement, affected_apis)
        }
    
    def _recommend_tests(self, improvement: CodeImprovement,
                        affected_apis: List[str]) -> List[str]:
        """Recommend tests for an improvement."""
        tests = improvement.test_suggestions.copy()
        
        # Add API-specific tests
        for api in affected_apis:
            tests.append(f"Test {api} still works as expected")
            tests.append(f"Test {api} performance hasn't degraded")
        
        # Add general tests based on improvement type
        if improvement.type == ModificationType.OPTIMIZATION:
            tests.append("Benchmark before and after performance")
            tests.append("Verify output remains identical")
        elif improvement.type == ModificationType.REFACTORING:
            tests.append("Run all existing tests")
            tests.append("Check for any behavioral changes")
        elif improvement.type == ModificationType.SECURITY:
            tests.append("Security scan the changes")
            tests.append("Test with malicious inputs")
        
        return list(set(tests))  # Remove duplicates
    
    def generate_context_report(self) -> Dict[str, Any]:
        """Generate a report on the codebase context."""
        total_files = len(self.file_contexts)
        total_dependencies = sum(len(c.dependencies) for c in self.file_contexts.values())
        
        # Find most connected files
        connectivity = []
        for file_path, context in self.file_contexts.items():
            dependents = len(self._get_dependents(file_path))
            dependencies = len(context.dependencies)
            connectivity.append({
                'file': os.path.relpath(file_path, self.repo_path),
                'dependents': dependents,
                'dependencies': dependencies,
                'total_connections': dependents + dependencies,
                'complexity': context.complexity_score
            })
        
        connectivity.sort(key=lambda x: x['total_connections'], reverse=True)
        
        # Find critical paths
        critical_files = []
        for file_path in self.file_contexts:
            if self._is_on_critical_path(file_path):
                critical_files.append(os.path.relpath(file_path, self.repo_path))
        
        return {
            'summary': {
                'total_files': total_files,
                'total_dependencies': total_dependencies,
                'average_complexity': sum(c.complexity_score for c in self.file_contexts.values()) / total_files if total_files > 0 else 0
            },
            'most_connected': connectivity[:10],
            'critical_files': critical_files[:10],
            'api_registry_size': len(self.api_registry),
            'dependency_graph_stats': {
                'nodes': self.dependency_graph.number_of_nodes(),
                'edges': self.dependency_graph.number_of_edges()
            }
        }