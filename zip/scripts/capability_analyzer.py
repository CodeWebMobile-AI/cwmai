"""
Capability Analyzer

Analyzes the AI system's current capabilities, identifies gaps,
and benchmarks performance to guide self-improvement.
"""

import os
import ast
import json
import importlib.util
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import inspect
import asyncio


class CapabilityAnalyzer:
    """Understands what the AI can and cannot do."""
    
    def __init__(self, ai_brain=None):
        """Initialize the capability analyzer.
        
        Args:
            ai_brain: AI brain instance for advanced analysis
        """
        self.ai_brain = ai_brain
        self.base_path = Path(__file__).parent
        self.capability_map = {}
        self.performance_benchmarks = {}
        self.gap_analysis = {}
        
    async def analyze_current_capabilities(self) -> Dict[str, Any]:
        """Scan all modules and document available capabilities.
        
        Returns:
            Comprehensive capability analysis
        """
        print("Analyzing current system capabilities...")
        
        capabilities = {
            'modules': {},
            'tools': {},
            'integrations': {},
            'ai_functions': {},
            'total_functions': 0,
            'total_classes': 0,
            'capability_coverage': {}
        }
        
        # Scan all Python files in scripts directory
        for py_file in self.base_path.glob("*.py"):
            if py_file.name.startswith('__'):
                continue
                
            module_capabilities = await self._analyze_module(py_file)
            if module_capabilities:
                capabilities['modules'][py_file.stem] = module_capabilities
                capabilities['total_functions'] += len(module_capabilities.get('functions', []))
                capabilities['total_classes'] += len(module_capabilities.get('classes', []))
        
        # Analyze available tools
        capabilities['tools'] = self._analyze_available_tools()
        
        # Analyze AI integrations
        capabilities['integrations'] = self._analyze_ai_integrations()
        
        # Analyze AI-specific functions
        capabilities['ai_functions'] = await self._analyze_ai_functions()
        
        # Calculate capability coverage
        capabilities['capability_coverage'] = await self._calculate_coverage()
        
        self.capability_map = capabilities
        return capabilities
    
    async def _analyze_module(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a single Python module for capabilities.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Module capability analysis
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            module_info = {
                'description': ast.get_docstring(tree) or "No description",
                'classes': [],
                'functions': [],
                'imports': [],
                'capabilities': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                        'docstring': ast.get_docstring(node)
                    }
                    module_info['classes'].append(class_info)
                    
                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                    func_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node),
                        'async': isinstance(node, ast.AsyncFunctionDef)
                    }
                    module_info['functions'].append(func_info)
                    
                elif isinstance(node, ast.Import):
                    module_info['imports'].extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_info['imports'].append(node.module)
            
            # Determine capabilities based on content
            module_info['capabilities'] = self._infer_capabilities(module_info, file_path.stem)
            
            return module_info
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
    
    def _infer_capabilities(self, module_info: Dict[str, Any], module_name: str) -> List[str]:
        """Infer capabilities from module analysis.
        
        Args:
            module_info: Module analysis data
            module_name: Name of the module
            
        Returns:
            List of inferred capabilities
        """
        capabilities = []
        
        # Infer from module name
        capability_keywords = {
            'task': ['task generation', 'task management'],
            'ai_brain': ['AI reasoning', 'multi-model AI'],
            'swarm': ['swarm intelligence', 'distributed decision making'],
            'learning': ['machine learning', 'pattern recognition'],
            'context': ['context gathering', 'external research'],
            'charter': ['dynamic goals', 'adaptive purpose'],
            'validator': ['validation', 'quality control'],
            'improver': ['self-improvement', 'code optimization'],
            'project': ['project creation', 'repository management'],
            'quantum': ['quantum optimization', 'advanced algorithms']
        }
        
        for keyword, caps in capability_keywords.items():
            if keyword in module_name.lower():
                capabilities.extend(caps)
        
        # Infer from class/function names
        all_names = []
        all_names.extend([c['name'].lower() for c in module_info.get('classes', [])])
        all_names.extend([f['name'].lower() for f in module_info.get('functions', [])])
        
        if any('generate' in name for name in all_names):
            capabilities.append('generation')
        if any('analyze' in name for name in all_names):
            capabilities.append('analysis')
        if any('optimize' in name for name in all_names):
            capabilities.append('optimization')
        if any('learn' in name for name in all_names):
            capabilities.append('learning')
            
        return list(set(capabilities))  # Remove duplicates
    
    def _analyze_available_tools(self) -> Dict[str, List[str]]:
        """Analyze available external tools and integrations.
        
        Returns:
            Tool analysis
        """
        tools = {
            'version_control': ['git'],
            'languages': ['python'],
            'ai_providers': [],
            'apis': [],
            'databases': []
        }
        
        # Check for AI provider API keys
        if os.getenv('OPENAI_API_KEY'):
            tools['ai_providers'].append('openai')
        if os.getenv('ANTHROPIC_API_KEY'):
            tools['ai_providers'].append('anthropic')
        if os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY'):
            tools['ai_providers'].append('gemini')
        if os.getenv('DEEPSEEK_API_KEY'):
            tools['ai_providers'].append('deepseek')
            
        # Check for GitHub integration
        if os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT'):
            tools['apis'].append('github')
            
        return tools
    
    def _analyze_ai_integrations(self) -> Dict[str, Any]:
        """Analyze AI model integrations and capabilities.
        
        Returns:
            AI integration analysis
        """
        integrations = {
            'primary_ai': None,
            'research_ai': [],
            'specialized_ai': {},
            'total_providers': 0
        }
        
        # Determine primary AI
        if os.getenv('ANTHROPIC_API_KEY'):
            integrations['primary_ai'] = 'anthropic'
            integrations['total_providers'] += 1
            
        # Research AI providers
        for provider in ['openai', 'gemini', 'deepseek']:
            key_names = {
                'openai': 'OPENAI_API_KEY',
                'gemini': ['GOOGLE_API_KEY', 'GEMINI_API_KEY'],
                'deepseek': 'DEEPSEEK_API_KEY'
            }
            
            keys = key_names.get(provider, [])
            if isinstance(keys, str):
                keys = [keys]
                
            if any(os.getenv(key) for key in keys):
                integrations['research_ai'].append(provider)
                integrations['total_providers'] += 1
                
        return integrations
    
    async def _analyze_ai_functions(self) -> Dict[str, List[str]]:
        """Analyze AI-specific functions and capabilities.
        
        Returns:
            AI function analysis
        """
        ai_functions = {
            'generation': [],
            'analysis': [],
            'research': [],
            'decision_making': [],
            'learning': []
        }
        
        # Check ai_brain capabilities
        try:
            ai_brain_path = self.base_path / "ai_brain.py"
            if ai_brain_path.exists():
                module_info = await self._analyze_module(ai_brain_path)
                if module_info:
                    for func in module_info.get('functions', []):
                        name = func['name'].lower()
                        if 'generate' in name:
                            ai_functions['generation'].append(func['name'])
                        elif 'analyze' in name:
                            ai_functions['analysis'].append(func['name'])
                        elif 'research' in name:
                            ai_functions['research'].append(func['name'])
                        elif 'decide' in name or 'choose' in name:
                            ai_functions['decision_making'].append(func['name'])
                        elif 'learn' in name:
                            ai_functions['learning'].append(func['name'])
        except Exception as e:
            print(f"Error analyzing AI functions: {e}")
            
        return ai_functions
    
    async def _calculate_coverage(self) -> Dict[str, float]:
        """Calculate capability coverage across different domains.
        
        Returns:
            Coverage percentages by domain
        """
        # Define expected capabilities for a complete AI system
        expected_capabilities = {
            'core': ['task management', 'AI reasoning', 'validation', 'learning'],
            'advanced': ['self-improvement', 'swarm intelligence', 'dynamic goals'],
            'integration': ['github', 'multi-model AI', 'external research'],
            'creation': ['project creation', 'task generation', 'tool generation']
        }
        
        coverage = {}
        
        for domain, expected in expected_capabilities.items():
            found = 0
            for capability in expected:
                # Check if capability exists in our system
                if self._has_capability(capability):
                    found += 1
                    
            coverage[domain] = (found / len(expected)) * 100 if expected else 0
            
        # Overall coverage
        all_expected = sum(len(caps) for caps in expected_capabilities.values())
        all_found = sum(
            1 for domain in expected_capabilities 
            for cap in expected_capabilities[domain] 
            if self._has_capability(cap)
        )
        coverage['overall'] = (all_found / all_expected) * 100 if all_expected else 0
        
        return coverage
    
    def _has_capability(self, capability: str) -> bool:
        """Check if system has a specific capability.
        
        Args:
            capability: Capability to check
            
        Returns:
            Whether capability exists
        """
        capability_lower = capability.lower()
        
        # Check in module capabilities
        for module_data in self.capability_map.get('modules', {}).values():
            if any(capability_lower in cap.lower() for cap in module_data.get('capabilities', [])):
                return True
                
        # Check in tools
        tools = self.capability_map.get('tools', {})
        if capability_lower in str(tools).lower():
            return True
            
        # Check in AI functions
        ai_funcs = self.capability_map.get('ai_functions', {})
        if capability_lower in str(ai_funcs).lower():
            return True
            
        return False
    
    async def identify_gaps(self) -> Dict[str, Any]:
        """Identify capability gaps and missing functionality.
        
        Returns:
            Gap analysis with recommendations
        """
        if not self.capability_map:
            await self.analyze_current_capabilities()
            
        gaps = {
            'missing_capabilities': [],
            'weak_areas': [],
            'integration_gaps': [],
            'recommendations': []
        }
        
        # Check for missing core capabilities
        essential_capabilities = [
            ('automated testing', 'Automated test generation and execution'),
            ('performance profiling', 'Detailed performance analysis tools'),
            ('visual understanding', 'Ability to analyze diagrams and images'),
            ('multi-language support', 'Analysis of non-Python languages'),
            ('predictive modeling', 'ML models for outcome prediction'),
            ('experiment design', 'Automated experiment creation and execution'),
            ('cross-domain learning', 'Transfer learning from other domains')
        ]
        
        for capability, description in essential_capabilities:
            if not self._has_capability(capability):
                gaps['missing_capabilities'].append({
                    'capability': capability,
                    'description': description,
                    'priority': 'high' if 'automated' in capability else 'medium'
                })
        
        # Identify weak areas based on coverage
        coverage = self.capability_map.get('capability_coverage', {})
        for domain, percentage in coverage.items():
            if percentage < 70:
                gaps['weak_areas'].append({
                    'domain': domain,
                    'coverage': percentage,
                    'improvement_needed': 70 - percentage
                })
        
        # Check integration gaps
        if not self.capability_map.get('tools', {}).get('ai_providers'):
            gaps['integration_gaps'].append('No AI providers configured')
            
        if len(self.capability_map.get('integrations', {}).get('research_ai', [])) < 2:
            gaps['integration_gaps'].append('Limited research AI diversity')
        
        # Generate recommendations using AI if available
        if self.ai_brain:
            gaps['recommendations'] = await self._generate_recommendations(gaps)
        else:
            gaps['recommendations'] = self._generate_basic_recommendations(gaps)
            
        self.gap_analysis = gaps
        return gaps
    
    async def _generate_recommendations(self, gaps: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate AI-powered recommendations for closing gaps.
        
        Args:
            gaps: Gap analysis data
            
        Returns:
            List of recommendations
        """
        prompt = f"""
        Based on this capability gap analysis, provide specific recommendations:
        
        Missing capabilities: {json.dumps(gaps['missing_capabilities'], indent=2)}
        Weak areas: {json.dumps(gaps['weak_areas'], indent=2)}
        Integration gaps: {gaps['integration_gaps']}
        
        For each gap, suggest:
        1. Implementation approach
        2. Required resources
        3. Priority order
        4. Expected impact
        
        Format as a list of actionable recommendations.
        """
        
        try:
            response = await self.ai_brain.generate_response(prompt)
            # Parse AI response into structured recommendations
            return self._parse_recommendations(response.get('content', ''))
        except Exception as e:
            print(f"Error generating AI recommendations: {e}")
            return self._generate_basic_recommendations(gaps)
    
    def _generate_basic_recommendations(self, gaps: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate basic recommendations without AI.
        
        Args:
            gaps: Gap analysis data
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Recommend addressing high-priority missing capabilities first
        for missing in gaps['missing_capabilities']:
            if missing['priority'] == 'high':
                recommendations.append({
                    'action': f"Implement {missing['capability']}",
                    'reason': missing['description'],
                    'priority': 'high',
                    'approach': 'Research existing solutions and create custom implementation'
                })
        
        # Recommend improving weak areas
        for weak in gaps['weak_areas']:
            recommendations.append({
                'action': f"Improve {weak['domain']} capabilities",
                'reason': f"Currently at {weak['coverage']:.1f}% coverage",
                'priority': 'medium',
                'approach': f"Need to add {weak['improvement_needed']:.1f}% more functionality"
            })
            
        return recommendations
    
    def _parse_recommendations(self, ai_response: str) -> List[Dict[str, str]]:
        """Parse AI response into structured recommendations.
        
        Args:
            ai_response: AI-generated recommendation text
            
        Returns:
            Structured recommendations
        """
        # Simple parsing - in production would be more sophisticated
        recommendations = []
        lines = ai_response.split('\n')
        
        current_rec = {}
        for line in lines:
            line = line.strip()
            if line.startswith('1.') or line.startswith('-'):
                if current_rec:
                    recommendations.append(current_rec)
                current_rec = {'action': line.lstrip('1.- ')}
            elif 'approach:' in line.lower():
                current_rec['approach'] = line.split(':', 1)[1].strip()
            elif 'priority:' in line.lower():
                current_rec['priority'] = line.split(':', 1)[1].strip()
            elif 'impact:' in line.lower():
                current_rec['impact'] = line.split(':', 1)[1].strip()
                
        if current_rec:
            recommendations.append(current_rec)
            
        return recommendations
    
    async def benchmark_performance(self) -> Dict[str, Any]:
        """Benchmark current system performance.
        
        Returns:
            Performance metrics and comparisons
        """
        print("Benchmarking system performance...")
        
        benchmarks = {
            'task_performance': await self._benchmark_task_performance(),
            'code_quality': await self._benchmark_code_quality(),
            'ai_effectiveness': await self._benchmark_ai_effectiveness(),
            'resource_efficiency': self._benchmark_resource_efficiency(),
            'improvement_rate': await self._calculate_improvement_rate()
        }
        
        # Compare to baseline or industry standards
        benchmarks['comparison'] = self._compare_to_standards(benchmarks)
        
        self.performance_benchmarks = benchmarks
        return benchmarks
    
    async def _benchmark_task_performance(self) -> Dict[str, float]:
        """Benchmark task completion performance.
        
        Returns:
            Task performance metrics
        """
        # Analyze task_state.json if it exists
        task_state_path = self.base_path.parent / "task_state.json"
        
        metrics = {
            'completion_rate': 0.0,
            'average_time_hours': 0.0,
            'success_rate': 0.0,
            'tasks_per_day': 0.0
        }
        
        if task_state_path.exists():
            try:
                with open(task_state_path, 'r') as f:
                    task_state = json.load(f)
                    
                tasks = task_state.get('tasks', {})
                if tasks:
                    completed = sum(1 for t in tasks.values() if t.get('status') == 'completed')
                    total = len(tasks)
                    
                    metrics['completion_rate'] = (completed / total * 100) if total > 0 else 0
                    metrics['success_rate'] = task_state.get('success_rate', 0.0) * 100
                    
                    # Calculate average completion time
                    completion_times = []
                    for task in tasks.values():
                        if task.get('completed_at') and task.get('created_at'):
                            # Would calculate time difference here
                            pass
                            
            except Exception as e:
                print(f"Error reading task state: {e}")
                
        return metrics
    
    async def _benchmark_code_quality(self) -> Dict[str, float]:
        """Benchmark code quality metrics.
        
        Returns:
            Code quality metrics
        """
        metrics = {
            'documentation_coverage': 0.0,
            'test_coverage': 0.0,
            'complexity_score': 0.0,
            'maintainability_index': 0.0
        }
        
        total_functions = 0
        documented_functions = 0
        
        # Analyze documentation coverage
        for module_data in self.capability_map.get('modules', {}).values():
            for func in module_data.get('functions', []):
                total_functions += 1
                if func.get('docstring'):
                    documented_functions += 1
                    
        if total_functions > 0:
            metrics['documentation_coverage'] = (documented_functions / total_functions) * 100
            
        # Estimate other metrics (would use proper tools in production)
        metrics['test_coverage'] = 40.0  # Placeholder
        metrics['complexity_score'] = 15.0  # Placeholder (lower is better)
        metrics['maintainability_index'] = 65.0  # Placeholder (higher is better)
        
        return metrics
    
    async def _benchmark_ai_effectiveness(self) -> Dict[str, float]:
        """Benchmark AI system effectiveness.
        
        Returns:
            AI effectiveness metrics
        """
        metrics = {
            'response_quality': 0.0,
            'task_understanding': 0.0,
            'learning_rate': 0.0,
            'adaptation_speed': 0.0
        }
        
        # Check if outcome learning system has data
        outcome_path = self.base_path / "outcome_history.json"
        if outcome_path.exists():
            try:
                with open(outcome_path, 'r') as f:
                    outcomes = json.load(f)
                    
                if outcomes:
                    # Calculate average value score
                    value_scores = [o.get('value_score', 0) for o in outcomes[-20:]]
                    if value_scores:
                        metrics['response_quality'] = sum(value_scores) / len(value_scores) * 100
                        
                    # Calculate improvement trend
                    if len(value_scores) > 10:
                        early = sum(value_scores[:5]) / 5
                        late = sum(value_scores[-5:]) / 5
                        metrics['learning_rate'] = ((late - early) / early * 100) if early > 0 else 0
                        
            except Exception as e:
                print(f"Error analyzing outcomes: {e}")
                
        # Estimate other metrics
        metrics['task_understanding'] = 75.0  # Placeholder
        metrics['adaptation_speed'] = 60.0  # Placeholder
        
        return metrics
    
    def _benchmark_resource_efficiency(self) -> Dict[str, float]:
        """Benchmark resource usage efficiency.
        
        Returns:
            Resource efficiency metrics
        """
        return {
            'api_efficiency': 80.0,  # Placeholder - API calls per task
            'compute_efficiency': 70.0,  # Placeholder - CPU usage
            'memory_efficiency': 85.0,  # Placeholder - Memory usage
            'cost_efficiency': 75.0  # Placeholder - Cost per outcome
        }
    
    async def _calculate_improvement_rate(self) -> Dict[str, float]:
        """Calculate system improvement rate over time.
        
        Returns:
            Improvement rate metrics
        """
        return {
            'capability_growth': 5.0,  # Placeholder - % new capabilities per week
            'performance_improvement': 3.0,  # Placeholder - % performance gain per week
            'quality_improvement': 4.0,  # Placeholder - % quality improvement per week
            'efficiency_improvement': 2.5  # Placeholder - % efficiency gain per week
        }
    
    def _compare_to_standards(self, benchmarks: Dict[str, Any]) -> Dict[str, str]:
        """Compare benchmarks to industry standards or goals.
        
        Args:
            benchmarks: Current benchmark data
            
        Returns:
            Comparison analysis
        """
        comparisons = {}
        
        # Define target standards
        standards = {
            'task_performance': {'completion_rate': 80, 'success_rate': 90},
            'code_quality': {'documentation_coverage': 80, 'test_coverage': 80},
            'ai_effectiveness': {'response_quality': 85, 'learning_rate': 5},
            'resource_efficiency': {'api_efficiency': 90, 'cost_efficiency': 85}
        }
        
        for category, metrics in benchmarks.items():
            if category in standards and isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if metric in standards[category]:
                        target = standards[category][metric]
                        if value >= target:
                            comparisons[f"{category}.{metric}"] = f"✅ Above target ({value:.1f}% vs {target}%)"
                        else:
                            comparisons[f"{category}.{metric}"] = f"❌ Below target ({value:.1f}% vs {target}%)"
                            
        return comparisons
    
    def get_capability_summary(self) -> Dict[str, Any]:
        """Get a summary of system capabilities.
        
        Returns:
            Capability summary
        """
        if not self.capability_map:
            return {"error": "No capability analysis available. Run analyze_current_capabilities() first."}
            
        summary = {
            'total_modules': len(self.capability_map.get('modules', {})),
            'total_functions': self.capability_map.get('total_functions', 0),
            'total_classes': self.capability_map.get('total_classes', 0),
            'ai_providers': len(self.capability_map.get('tools', {}).get('ai_providers', [])),
            'capability_coverage': self.capability_map.get('capability_coverage', {}),
            'top_capabilities': self._get_top_capabilities(),
            'gaps_identified': len(self.gap_analysis.get('missing_capabilities', [])) if self.gap_analysis else 0
        }
        
        return summary
    
    def _get_top_capabilities(self) -> List[str]:
        """Get the top system capabilities.
        
        Returns:
            List of top capabilities
        """
        all_capabilities = []
        
        for module_data in self.capability_map.get('modules', {}).values():
            all_capabilities.extend(module_data.get('capabilities', []))
            
        # Count occurrences and return most common
        from collections import Counter
        capability_counts = Counter(all_capabilities)
        
        return [cap for cap, _ in capability_counts.most_common(10)]


async def main():
    """Test the capability analyzer."""
    analyzer = CapabilityAnalyzer()
    
    # Analyze current capabilities
    print("=== Analyzing Current Capabilities ===")
    capabilities = await analyzer.analyze_current_capabilities()
    print(f"Found {len(capabilities['modules'])} modules")
    print(f"Total functions: {capabilities['total_functions']}")
    print(f"Total classes: {capabilities['total_classes']}")
    print(f"AI providers: {capabilities['tools']['ai_providers']}")
    
    # Identify gaps
    print("\n=== Identifying Capability Gaps ===")
    gaps = await analyzer.identify_gaps()
    print(f"Missing capabilities: {len(gaps['missing_capabilities'])}")
    for missing in gaps['missing_capabilities'][:3]:
        print(f"  - {missing['capability']}: {missing['description']}")
    
    # Benchmark performance
    print("\n=== Benchmarking Performance ===")
    benchmarks = await analyzer.benchmark_performance()
    print("Task Performance:")
    for metric, value in benchmarks['task_performance'].items():
        print(f"  - {metric}: {value:.1f}%")
    
    # Get summary
    print("\n=== Capability Summary ===")
    summary = analyzer.get_capability_summary()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())