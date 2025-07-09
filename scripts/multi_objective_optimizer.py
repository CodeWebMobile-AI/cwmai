"""
Multi-Objective Optimization for AI Assistant Improvements

Inspired by AlphaEvolve's ability to optimize for multiple objectives simultaneously.
"""

import asyncio
import ast
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime, timezone
import numpy as np

from scripts.http_ai_client import HTTPAIClient
from scripts.state_manager import StateManager


class MultiObjectiveOptimizer:
    """Optimizes improvements across multiple dimensions simultaneously."""
    
    def __init__(self, state_manager: StateManager):
        self.logger = logging.getLogger(__name__)
        self.state_manager = state_manager
        self.ai_client = HTTPAIClient()
        
        # Define optimization objectives
        self.objectives = {
            'performance': {
                'weight': 0.25,
                'metrics': ['response_time', 'throughput', 'resource_usage']
            },
            'quality': {
                'weight': 0.3,
                'metrics': ['accuracy', 'relevance', 'completeness']
            },
            'maintainability': {
                'weight': 0.2,
                'metrics': ['code_complexity', 'documentation', 'test_coverage']
            },
            'innovation': {
                'weight': 0.25,
                'metrics': ['novelty', 'capability_expansion', 'user_value']
            }
        }
        
    async def evaluate_improvement(self, improvement: Dict[str, Any]) -> Dict[str, float]:
            """Evaluate improvement across multiple objectives with REAL scoring."""
            scores = {}
            
            # 1. Performance Score (based on actual analysis)
            perf_score = 0.0
            
            # Check for performance-related keywords
            code = improvement.get('code', '')
            description = improvement.get('description', '').lower()
            
            # Performance indicators
            if 'cache' in description or 'cache' in code:
                perf_score += 0.2
            if 'async' in code or 'await' in code:
                perf_score += 0.15
            if 'optimize' in description or 'performance' in description:
                perf_score += 0.15
            if '@lru_cache' in code or 'functools.cache' in code:
                perf_score += 0.2
            if 'threading' in code or 'multiprocessing' in code:
                perf_score += 0.2
            if 'batch' in description or 'bulk' in description:
                perf_score += 0.1
                
            scores['performance'] = min(1.0, perf_score)
            
            # 2. Quality Score (based on code analysis)
            quality_score = 0.3  # Base score
            
            try:
                # Parse the code
                if improvement.get('code'):
                    tree = ast.parse(improvement['code'])
                    
                    # Check for quality indicators
                    has_docstrings = any(isinstance(node, ast.FunctionDef) and ast.get_docstring(node) 
                                       for node in ast.walk(tree))
                    has_type_hints = 'typing' in improvement['code'] or '->' in improvement['code']
                    has_error_handling = 'try:' in improvement['code'] or 'except' in improvement['code']
                    has_logging = 'logger' in improvement['code'] or 'logging' in improvement['code']
                    has_tests = 'test_' in improvement['code'] or 'assert' in improvement['code']
                    
                    if has_docstrings: quality_score += 0.15
                    if has_type_hints: quality_score += 0.15
                    if has_error_handling: quality_score += 0.2
                    if has_logging: quality_score += 0.1
                    if has_tests: quality_score += 0.1
                    
            except SyntaxError:
                quality_score = 0.1  # Syntax error = low quality
                
            scores['quality'] = min(1.0, quality_score)
            
            # 3. Maintainability Score
            maint_score = 0.2  # Base
            
            if improvement.get('code'):
                lines = improvement['code'].split('\n')
                
                # Check complexity
                if len(lines) < 100:
                    maint_score += 0.2  # Not too long
                if len(lines) < 50:
                    maint_score += 0.1  # Nice and short
                    
                # Check for good practices
                if 'class' in improvement['code']:
                    maint_score += 0.2  # OOP
                if 'def ' in improvement['code']:
                    maint_score += 0.1  # Functions
                if improvement['code'].count('def ') > 3:
                    maint_score += 0.1  # Well-structured
                if '# TODO' not in improvement['code']:
                    maint_score += 0.1  # Complete implementation
                    
            scores['maintainability'] = min(1.0, maint_score)
            
            # 4. Innovation Score (based on uniqueness)
            innovation_score = 0.3  # Base
            
            # Check for innovative features
            innovative_keywords = [
                'machine learning', 'ml', 'ai', 'neural', 'predict',
                'adaptive', 'dynamic', 'intelligent', 'smart',
                'blockchain', 'quantum', 'parallel', 'distributed',
                'real-time', 'streaming', 'event-driven', 'reactive'
            ]
            
            for keyword in innovative_keywords:
                if keyword in description:
                    innovation_score += 0.1
                    
            # Boost for unique improvement types
            imp_type = improvement.get('type', '').lower()
            if imp_type not in ['optimization', 'tool']:
                innovation_score += 0.2
                
            scores['innovation'] = min(1.0, innovation_score)
            
            # 5. Overall Score (weighted average)
            scores['overall'] = (
                scores['performance'] * 0.3 +
                scores['quality'] * 0.3 +
                scores['maintainability'] * 0.2 +
                scores['innovation'] * 0.2
            )
            
            # Add timestamp for diversity
            scores['_evaluated_at'] = datetime.now().timestamp()
            
            return scores

    async def _evaluate_objective(self, improvement: Dict, objective: str, metrics: List[str]) -> float:
        """Evaluate a specific objective."""
        if objective == 'performance':
            return await self._evaluate_performance(improvement)
        elif objective == 'quality':
            return await self._evaluate_quality(improvement)
        elif objective == 'maintainability':
            return await self._evaluate_maintainability(improvement)
        elif objective == 'innovation':
            return await self._evaluate_innovation(improvement)
        else:
            return 0.5  # Default middle score
            
    async def _evaluate_performance(self, improvement: Dict) -> float:
        """Evaluate performance impact of improvement."""
        score = 0.5  # Base score
        
        # Check for performance-related keywords
        code = improvement.get('code', '')
        description = improvement.get('description', '')
        
        performance_indicators = [
            ('async', 0.1),  # Async operations
            ('cache', 0.1),  # Caching mechanisms
            ('optimize', 0.05),  # Optimization mentions
            ('parallel', 0.1),  # Parallel processing
            ('batch', 0.05),  # Batch operations
            ('index', 0.05),  # Indexing for faster access
            ('stream', 0.05)  # Streaming for memory efficiency
        ]
        
        for indicator, boost in performance_indicators:
            if indicator in code.lower() or indicator in description.lower():
                score += boost
                
        # Check for anti-patterns
        if 'sleep(' in code and 'async' not in code:
            score -= 0.2  # Blocking sleep
        if code.count('for') > 5:
            score -= 0.1  # Too many loops
            
        return min(1.0, max(0.0, score))
        
    async def _evaluate_quality(self, improvement: Dict) -> float:
        """Evaluate code quality and functionality."""
        score = 0.5  # Base score
        
        code = improvement.get('code', '')
        
        # Quality indicators
        quality_checks = [
            ('"""', 0.1),  # Docstrings
            ('try:', 0.1),  # Error handling
            ('logging', 0.05),  # Logging
            ('typing', 0.05),  # Type hints
            ('test', 0.1),  # Testing mentions
            ('validate', 0.05),  # Validation
            ('sanitize', 0.05)  # Input sanitization
        ]
        
        for check, boost in quality_checks:
            if check in code:
                score += boost
                
        # Check code structure
        if len(code.split('\n')) > 20:  # Substantial implementation
            score += 0.1
            
        # Penalize poor practices
        if 'pass' in code and code.count('pass') > 2:
            score -= 0.2  # Too many placeholder passes
        if 'TODO' in code:
            score -= 0.1  # Incomplete implementation
            
        return min(1.0, max(0.0, score))
        
    async def _evaluate_maintainability(self, improvement: Dict) -> float:
        """Evaluate how maintainable the improvement is."""
        score = 0.5  # Base score
        
        code = improvement.get('code', '')
        
        # Maintainability indicators
        lines = code.split('\n')
        
        # Function/class count
        functions = code.count('def ')
        classes = code.count('class ')
        
        if functions > 0:
            avg_lines_per_function = len(lines) / functions
            if 10 <= avg_lines_per_function <= 50:
                score += 0.2  # Good function size
            elif avg_lines_per_function > 100:
                score -= 0.2  # Functions too large
                
        # Documentation
        docstring_count = code.count('"""')
        if docstring_count >= functions + classes:
            score += 0.2  # Well documented
            
        # Naming conventions
        import re
        snake_case_vars = len(re.findall(r'\b[a-z_]+\b', code))
        camel_case_vars = len(re.findall(r'\b[A-Z][a-zA-Z]+\b', code))
        
        if snake_case_vars > camel_case_vars * 2:  # Pythonic naming
            score += 0.1
            
        return min(1.0, max(0.0, score))
        
    async def _evaluate_innovation(self, improvement: Dict) -> float:
        """Evaluate how innovative the improvement is."""
        # Use AI to evaluate innovation
        prompt = f"""Rate the innovation level of this improvement on a scale of 0-1:

Name: {improvement.get('name')}
Type: {improvement.get('type')}
Description: {improvement.get('description')}
Metrics: {improvement.get('metrics', [])}

Consider:
1. Does it add new capabilities?
2. Does it solve problems in a novel way?
3. Does it expand what the system can do?
4. Is it using cutting-edge techniques?

Respond with just a number between 0 and 1.
"""
        
        try:
            response = await self.ai_client.generate_enhanced_response(prompt)
            # Handle dict response
            if isinstance(response, dict):
                response_text = response.get('content', '0.5')
            else:
                response_text = str(response)
            score = float(response_text.strip())
            return min(1.0, max(0.0, score))
        except:
            return 0.5  # Default if evaluation fails
            
    def pareto_rank(self, improvements: List[Dict], scores: Dict[str, Dict[str, float]]) -> List[Tuple[Dict, int]]:
        """Rank improvements using Pareto dominance (AlphaEvolve-style)."""
        n = len(improvements)
        domination_count = [0] * n
        
        # Calculate domination relationships
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                    
                # Check if improvement i dominates improvement j
                dominates = True
                strictly_better = False
                
                for objective in self.objectives:
                    score_i = scores[improvements[i]['id']][objective]
                    score_j = scores[improvements[j]['id']][objective]
                    
                    if score_i < score_j:
                        dominates = False
                        break
                    elif score_i > score_j:
                        strictly_better = True
                        
                if dominates and strictly_better:
                    domination_count[j] += 1
                    
        # Create ranking (lower domination count = better rank)
        ranked = []
        for i, improvement in enumerate(improvements):
            ranked.append((improvement, domination_count[i]))
            
        ranked.sort(key=lambda x: x[1])  # Sort by domination count
        
        return ranked
        
    async def optimize_for_objectives(self, improvements: List[Dict]) -> List[Dict]:
        """Optimize and rank improvements for multiple objectives."""
        print(f"Evaluating {len(improvements)} improvements across {len(self.objectives)} objectives")
        
        # Evaluate all improvements
        scores = {}
        for improvement in improvements:
            imp_id = improvement.get('id', improvement['name'])
            scores[imp_id] = await self.evaluate_improvement(improvement)
            improvement['multi_objective_scores'] = scores[imp_id]
            
        # Pareto ranking
        ranked = self.pareto_rank(improvements, scores)
        
        # Add ranking info to improvements
        for improvement, rank in ranked:
            improvement['pareto_rank'] = rank
            
        # Sort by overall score as secondary criteria
        ranked_improvements = [imp for imp, _ in ranked]
        ranked_improvements.sort(
            key=lambda x: x['multi_objective_scores']['overall'],
            reverse=True
        )
        
        # Log results
        print("Top 3 improvements by multi-objective optimization:")
        for i, imp in enumerate(ranked_improvements[:3], 1):
            scores = imp['multi_objective_scores']
            print(f"{i}. {imp['name']} - Overall: {scores['overall']:.3f}")
            for obj in self.objectives:
                print(f"   {obj}: {scores[obj]:.3f}")
                
        return ranked_improvements