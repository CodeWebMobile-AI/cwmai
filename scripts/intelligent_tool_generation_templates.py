#!/usr/bin/env python3
"""
Intelligent Tool Generation Templates with AI-Enhanced Capabilities
Provides smart context, learning-based templates, and AI-driven tool generation
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import ast
import logging
from datetime import datetime
import asyncio
from collections import defaultdict, Counter
import re
import numpy as np
from dataclasses import dataclass, field

# Import AI capabilities
from scripts.ai_brain import AIBrain
from scripts.intelligent_self_improver import IntelligentSelfImprover
from scripts.capability_synthesizer import CapabilitySynthesizer
from scripts.improvement_learning_system import ImprovementLearningSystem
from scripts.semantic_memory_system import SemanticMemorySystem
from scripts.knowledge_graph_builder import KnowledgeGraphBuilder
from scripts.http_ai_client import HTTPAIClient


@dataclass
class ToolGenerationMetrics:
    """Metrics for tracking tool generation success."""
    generation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time: float = 0.0
    error_patterns: Dict[str, int] = field(default_factory=dict)
    successful_patterns: Dict[str, float] = field(default_factory=dict)
    category_performance: Dict[str, float] = field(default_factory=dict)


@dataclass
class ToolRequirementAnalysis:
    """Analysis of tool requirements."""
    primary_category: str
    confidence: float
    detected_operations: List[str]
    suggested_imports: List[str]
    similar_tools: List[Tuple[str, float]]  # (tool_name, similarity_score)
    complexity_score: float
    security_considerations: List[str]
    performance_requirements: Dict[str, Any]


class IntelligentToolGenerationTemplates:
    """AI-Enhanced tool generation templates with learning capabilities."""
    
    def __init__(self):
        # Initialize base components
        self.templates = self._load_enhanced_templates()
        self.common_patterns = self._load_intelligent_patterns()
        self.error_fixes = self._load_error_fixes()
        self.discovered_scripts = self._discover_available_scripts()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize AI components
        self.ai_brain = AIBrain()
        self.self_improver = IntelligentSelfImprover()
        self.capability_synthesizer = CapabilitySynthesizer()
        self.learning_system = ImprovementLearningSystem()
        self.semantic_memory = SemanticMemorySystem()
        self.knowledge_graph = KnowledgeGraphBuilder()
        self.ai_client = HTTPAIClient()
        
        # Learning data
        self.generation_metrics = defaultdict(ToolGenerationMetrics)
        self.successful_tools_cache = {}
        self.pattern_effectiveness = defaultdict(float)
        
        # Load historical data
        self._load_learning_data()
    
    def _load_enhanced_templates(self) -> Dict[str, str]:
        """Load AI-enhanced templates with learning capabilities."""
        base_templates = self._load_base_templates()
        
        # Enhance templates with AI hooks
        enhanced_templates = {}
        for category, template in base_templates.items():
            enhanced_templates[category] = self._enhance_template_with_ai(template, category)
        
        return enhanced_templates
    
    def _enhance_template_with_ai(self, template: str, category: str) -> str:
        """Enhance a template with AI-driven features."""
        # Add performance tracking
        performance_tracking = '''
# Performance tracking for AI learning
_start_time = time.time()
_error_occurred = False

try:
    # Original logic here
    pass
finally:
    _execution_time = time.time() - _start_time
    # Report metrics for learning
    if hasattr(self, '_report_metrics'):
        self._report_metrics({
            'category': '{category}',
            'execution_time': _execution_time,
            'error_occurred': _error_occurred
        })
'''
        
        # Add intelligent error handling
        intelligent_error_handling = '''
except Exception as e:
    _error_occurred = True
    # Intelligent error analysis
    error_pattern = self._analyze_error_pattern(e) if hasattr(self, '_analyze_error_pattern') else str(e)
    
    # Attempt self-healing if possible
    if hasattr(self, '_attempt_self_healing'):
        healing_result = self._attempt_self_healing(error_pattern)
        if healing_result['success']:
            return healing_result['result']
    
    # Log for learning
    self._log_error_for_learning(error_pattern) if hasattr(self, '_log_error_for_learning') else None
    
    return {"error": f"Intelligent error analysis: {error_pattern}"}
'''
        
        # Enhance template
        enhanced = template.replace(
            "# TODO: Implement", 
            f"# AI-Enhanced Implementation\n{performance_tracking}\n# TODO: Implement"
        )
        
        # Add intelligent error handling
        enhanced = enhanced.replace(
            "except Exception as e:",
            intelligent_error_handling
        )
        
        return enhanced
    
    def analyze_requirements(self, name: str, description: str, 
                           requirements: str) -> ToolRequirementAnalysis:
        """Perform AI-driven analysis of tool requirements."""
        # Combine all text for analysis
        full_context = f"{name} {description} {requirements}"
        
        # Use semantic memory to find similar tools
        similar_tools = self.semantic_memory.search(
            full_context, 
            category="generated_tools",
            top_k=5
        )
        
        # Detect operations using NLP patterns
        detected_operations = self._detect_operations(full_context)
        
        # Determine category with AI
        category, confidence = self._determine_category_with_ai(
            full_context, detected_operations
        )
        
        # Get suggested imports based on operations
        suggested_imports = self._suggest_intelligent_imports(
            detected_operations, category, similar_tools
        )
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(
            detected_operations, requirements
        )
        
        # Identify security considerations
        security_considerations = self._identify_security_considerations(
            full_context, detected_operations
        )
        
        # Determine performance requirements
        performance_requirements = self._analyze_performance_requirements(
            full_context, category
        )
        
        return ToolRequirementAnalysis(
            primary_category=category,
            confidence=confidence,
            detected_operations=detected_operations,
            suggested_imports=suggested_imports,
            similar_tools=[(tool['name'], tool['score']) for tool in similar_tools],
            complexity_score=complexity_score,
            security_considerations=security_considerations,
            performance_requirements=performance_requirements
        )
    
    def _detect_operations(self, text: str) -> List[str]:
        """Detect operations mentioned in requirements using NLP."""
        operations = []
        
        # Operation patterns
        operation_patterns = {
            'file_read': r'\b(read|load|open|parse)\s+\w*\s*(file|document|data)',
            'file_write': r'\b(write|save|export|dump)\s+\w*\s*(file|document|data)',
            'api_call': r'\b(api|endpoint|request|fetch|get|post)\b',
            'data_processing': r'\b(process|transform|analyze|filter|aggregate)\b',
            'validation': r'\b(validate|check|verify|ensure)\b',
            'async_operation': r'\b(async|concurrent|parallel|await)\b',
            'caching': r'\b(cache|store|persist|memory)\b',
            'monitoring': r'\b(monitor|track|log|metric|report)\b',
            'error_handling': r'\b(error|exception|failure|retry|recover)\b',
            'security': r'\b(auth|secure|encrypt|permission|access)\b'
        }
        
        text_lower = text.lower()
        for op_name, pattern in operation_patterns.items():
            if re.search(pattern, text_lower):
                operations.append(op_name)
        
        return operations
    
    def _determine_category_with_ai(self, text: str, 
                                  operations: List[str]) -> Tuple[str, float]:
        """Use AI to determine the most appropriate category."""
        # Build features for category determination
        features = {
            'text_length': len(text),
            'operation_count': len(operations),
            'has_file_ops': any('file' in op for op in operations),
            'has_data_ops': any('data' in op for op in operations),
            'has_system_ops': any('system' in op or 'monitor' in op for op in operations),
            'has_git_ops': 'git' in text.lower() or 'repository' in text.lower()
        }
        
        # Use AI brain for decision
        category_scores = {
            'file_operations': 0.0,
            'data_analysis': 0.0,
            'system_operations': 0.0,
            'git_operations': 0.0
        }
        
        # Score based on operations and features
        if features['has_file_ops']:
            category_scores['file_operations'] += 0.4
        if features['has_data_ops']:
            category_scores['data_analysis'] += 0.4
        if features['has_system_ops']:
            category_scores['system_operations'] += 0.4
        if features['has_git_ops']:
            category_scores['git_operations'] += 0.4
        
        # Add learning-based adjustments
        for category in category_scores:
            historical_performance = self.generation_metrics[category].success_count / \
                max(self.generation_metrics[category].generation_count, 1)
            category_scores[category] += historical_performance * 0.2
        
        # Get best category
        best_category = max(category_scores.items(), key=lambda x: x[1])
        
        return best_category[0], best_category[1]
    
    def _suggest_intelligent_imports(self, operations: List[str], 
                                   category: str, 
                                   similar_tools: List[Dict]) -> List[str]:
        """Suggest imports based on AI analysis."""
        imports = set()
        
        # Base imports for category
        category_imports = {
            'file_operations': ['pathlib.Path', 'os', 'shutil'],
            'data_analysis': ['pandas', 'numpy', 'collections.Counter'],
            'system_operations': ['subprocess', 'psutil', 'platform'],
            'git_operations': ['subprocess', 'git']
        }
        
        imports.update(category_imports.get(category, []))
        
        # Operation-specific imports
        operation_imports = {
            'async_operation': ['asyncio', 'aiofiles'],
            'api_call': ['aiohttp', 'requests'],
            'caching': ['scripts.redis_ai_response_cache', 'functools.lru_cache'],
            'monitoring': ['logging', 'scripts.worker_metrics_collector'],
            'validation': ['jsonschema', 'pydantic'],
            'data_processing': ['pandas', 'numpy', 'itertools']
        }
        
        for op in operations:
            if op in operation_imports:
                imports.update(operation_imports[op])
        
        # Learn from similar tools
        for tool in similar_tools[:3]:
            if 'imports' in tool:
                imports.update(tool['imports'])
        
        return list(imports)
    
    def _calculate_complexity_score(self, operations: List[str], 
                                  requirements: str) -> float:
        """Calculate tool complexity score."""
        score = 0.0
        
        # Base complexity from operations
        score += len(operations) * 0.1
        
        # Complexity indicators in requirements
        complexity_indicators = [
            'complex', 'advanced', 'sophisticated', 'multi-step',
            'concurrent', 'distributed', 'optimize', 'performance'
        ]
        
        req_lower = requirements.lower()
        for indicator in complexity_indicators:
            if indicator in req_lower:
                score += 0.15
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def _identify_security_considerations(self, text: str, 
                                        operations: List[str]) -> List[str]:
        """Identify security considerations."""
        considerations = []
        
        # Check for sensitive operations
        if 'file_write' in operations:
            considerations.append("Validate file paths to prevent directory traversal")
        
        if 'api_call' in operations:
            considerations.append("Implement proper authentication and rate limiting")
        
        if any(word in text.lower() for word in ['password', 'token', 'secret', 'key']):
            considerations.append("Never log or expose sensitive credentials")
        
        if 'subprocess' in text or 'execute' in text:
            considerations.append("Sanitize inputs to prevent command injection")
        
        return considerations
    
    def _analyze_performance_requirements(self, text: str, 
                                        category: str) -> Dict[str, Any]:
        """Analyze performance requirements."""
        requirements = {
            'needs_async': False,
            'needs_caching': False,
            'needs_batching': False,
            'timeout_seconds': 30,
            'memory_limit_mb': None
        }
        
        text_lower = text.lower()
        
        # Detect async needs
        if any(word in text_lower for word in ['concurrent', 'parallel', 'async', 'real-time']):
            requirements['needs_async'] = True
        
        # Detect caching needs
        if any(word in text_lower for word in ['cache', 'frequent', 'repeated', 'optimize']):
            requirements['needs_caching'] = True
        
        # Detect batching needs
        if any(word in text_lower for word in ['batch', 'bulk', 'many', 'large']):
            requirements['needs_batching'] = True
        
        # Adjust timeout based on operation type
        if category == 'data_analysis' or 'large' in text_lower:
            requirements['timeout_seconds'] = 300
        
        return requirements
    
    def generate_intelligent_tool(self, name: str, description: str,
                                requirements: str) -> Dict[str, Any]:
        """Generate a tool using AI-enhanced process."""
        try:
            # Analyze requirements
            analysis = self.analyze_requirements(name, description, requirements)
            
            # Store in semantic memory for future reference
            self.semantic_memory.store({
                'name': name,
                'description': description,
                'requirements': requirements,
                'analysis': analysis.__dict__,
                'timestamp': datetime.now().isoformat()
            }, category='tool_requirements')
            
            # Get base template
            template = self.get_template(analysis.primary_category)
            
            # Create enhanced prompt
            prompt = self._create_intelligent_prompt(
                name, description, requirements, analysis, template
            )
            
            # Generate tool code using AI
            generated_code = self._generate_with_ai(prompt)
            
            # Validate and improve generated code
            validated_code = self._validate_and_improve(generated_code, analysis)
            
            # Test the generated tool
            test_results = self._test_generated_tool(validated_code, name)
            
            # Store successful generation for learning
            if test_results['success']:
                self._store_successful_generation(
                    name, validated_code, analysis, test_results
                )
            
            # Update metrics
            self._update_generation_metrics(
                analysis.primary_category, 
                test_results['success']
            )
            
            return {
                'success': test_results['success'],
                'code': validated_code,
                'analysis': analysis.__dict__,
                'test_results': test_results,
                'confidence': analysis.confidence
            }
            
        except Exception as e:
            self.logger.error(f"Tool generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'analysis': None
            }
    
    def _create_intelligent_prompt(self, name: str, description: str,
                                 requirements: str, analysis: ToolRequirementAnalysis,
                                 template: str) -> str:
        """Create an AI-enhanced prompt with rich context."""
        # Get examples from similar successful tools
        similar_examples = self._get_similar_tool_examples(analysis.similar_tools)
        
        # Get pattern recommendations from learning system
        pattern_recommendations = self.learning_system.get_recommendations({
            'category': analysis.primary_category,
            'operations': analysis.detected_operations,
            'complexity': analysis.complexity_score
        })
        
        prompt = f"""You are an expert Python developer creating an AI-enhanced tool for the CWMAI system.

TASK: Generate a complete, production-ready Python tool module with AI capabilities.

TOOL SPECIFICATION:
- Name: {name}
- Description: {description}
- Requirements: {requirements}
- Category: {analysis.primary_category} (confidence: {analysis.confidence:.2f})

REQUIREMENT ANALYSIS:
- Detected Operations: {', '.join(analysis.detected_operations)}
- Complexity Score: {analysis.complexity_score:.2f}
- Security Considerations: {json.dumps(analysis.security_considerations, indent=2)}
- Performance Requirements: {json.dumps(analysis.performance_requirements, indent=2)}

SUGGESTED IMPORTS:
{chr(10).join(f'- {imp}' for imp in analysis.suggested_imports)}

SIMILAR SUCCESSFUL TOOLS:
{similar_examples}

LEARNING-BASED RECOMMENDATIONS:
{json.dumps(pattern_recommendations, indent=2)}

TEMPLATE TO FOLLOW:
{template}

QUALITY REQUIREMENTS:
✓ Implement all detected operations
✓ Include comprehensive error handling with recovery
✓ Add performance tracking for AI learning
✓ Implement suggested security measures
✓ Follow async patterns if needed ({analysis.performance_requirements['needs_async']})
✓ Add caching if beneficial ({analysis.performance_requirements['needs_caching']})
✓ Include detailed logging for debugging
✓ Make the tool self-documenting
✓ Ensure cross-platform compatibility
✓ Add input validation and type checking

AVOID:
✗ Hardcoded values (use parameters)
✗ Blocking operations in async functions
✗ Silent failures (always log errors)
✗ Memory leaks or unbounded growth
✗ Security vulnerabilities
✗ Platform-specific code without checks

Generate ONLY the Python code. Make it production-ready and AI-enhanced.
"""
        
        return prompt
    
    def _get_similar_tool_examples(self, similar_tools: List[Tuple[str, float]]) -> str:
        """Get examples from similar successful tools."""
        examples = []
        
        for tool_name, similarity in similar_tools[:2]:
            if tool_name in self.successful_tools_cache:
                tool_data = self.successful_tools_cache[tool_name]
                examples.append(f"""
Tool: {tool_name} (similarity: {similarity:.2f})
Key patterns used:
{chr(10).join(f'- {pattern}' for pattern in tool_data.get('patterns', [])[:3])}
""")
        
        return '\n'.join(examples) if examples else "No similar tools found in cache."
    
    def _generate_with_ai(self, prompt: str) -> str:
        """Generate tool code using AI."""
        # Use AI client to generate code
        response = asyncio.run(self.ai_client.query(prompt))
        
        # Extract code from response
        code = self._extract_code_from_response(response)
        
        return code
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from AI response."""
        # Look for code blocks
        code_pattern = r'```python\n(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0]
        
        # If no code blocks, assume entire response is code
        return response
    
    def _validate_and_improve(self, code: str, 
                            analysis: ToolRequirementAnalysis) -> str:
        """Validate and improve generated code."""
        try:
            # Parse AST to check syntax
            tree = ast.parse(code)
            
            # Use intelligent self-improver
            improvements = self.self_improver.analyze_code(code)
            
            # Apply high-confidence improvements
            improved_code = code
            for improvement in improvements:
                if improvement['confidence'] > 0.8:
                    improved_code = self._apply_improvement(
                        improved_code, improvement
                    )
            
            # Ensure all required operations are implemented
            improved_code = self._ensure_operations_implemented(
                improved_code, analysis.detected_operations
            )
            
            # Add missing imports
            improved_code = self._add_missing_imports(
                improved_code, analysis.suggested_imports
            )
            
            return improved_code
            
        except SyntaxError as e:
            self.logger.error(f"Syntax error in generated code: {e}")
            # Attempt to fix common syntax errors
            return self._fix_syntax_errors(code, str(e))
    
    def _test_generated_tool(self, code: str, tool_name: str) -> Dict[str, Any]:
        """Test the generated tool."""
        try:
            # Create temporary module
            test_module = type('TestModule', (), {})
            
            # Execute code in module namespace
            exec(code, test_module.__dict__)
            
            # Check if main function exists
            if not hasattr(test_module, tool_name):
                return {
                    'success': False,
                    'error': f"Function '{tool_name}' not found in generated code"
                }
            
            # Run basic tests
            func = getattr(test_module, tool_name)
            
            # Test with no arguments
            if asyncio.iscoroutinefunction(func):
                result = asyncio.run(func())
            else:
                result = func()
            
            # Check result format
            if not isinstance(result, dict):
                return {
                    'success': False,
                    'error': f"Function must return dict, got {type(result)}"
                }
            
            return {
                'success': True,
                'test_output': result,
                'is_async': asyncio.iscoroutinefunction(func)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _store_successful_generation(self, name: str, code: str,
                                   analysis: ToolRequirementAnalysis,
                                   test_results: Dict) -> None:
        """Store successful generation for future learning."""
        # Extract patterns from code
        patterns = self._extract_patterns_from_code(code)
        
        # Store in cache
        self.successful_tools_cache[name] = {
            'code': code,
            'analysis': analysis.__dict__,
            'patterns': patterns,
            'test_results': test_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in semantic memory
        self.semantic_memory.store({
            'name': name,
            'category': analysis.primary_category,
            'operations': analysis.detected_operations,
            'patterns': patterns,
            'complexity': analysis.complexity_score,
            'code_snippet': code[:500]  # Store first 500 chars
        }, category='generated_tools')
        
        # Update learning system
        self.learning_system.record_outcome({
            'category': analysis.primary_category,
            'patterns': patterns,
            'success': True,
            'execution_time': test_results.get('execution_time', 0)
        })
    
    def _extract_patterns_from_code(self, code: str) -> List[str]:
        """Extract notable patterns from generated code."""
        patterns = []
        
        # Pattern detection
        pattern_checks = {
            'async_await': r'async\s+def.*await',
            'error_recovery': r'except.*:\s*\n.*retry|recover|fallback',
            'validation': r'if\s+not\s+.*:\s*\n.*return.*error',
            'logging': r'logger\.|logging\.',
            'caching': r'@cache|lru_cache|cache\[',
            'type_hints': r'def.*\(.*:.*\).*->',
            'docstring': r'""".*"""',
            'performance_tracking': r'time\.time\(\)|perf_counter'
        }
        
        for pattern_name, pattern_regex in pattern_checks.items():
            if re.search(pattern_regex, code, re.DOTALL | re.IGNORECASE):
                patterns.append(pattern_name)
        
        return patterns
    
    def _update_generation_metrics(self, category: str, success: bool) -> None:
        """Update generation metrics for learning."""
        metrics = self.generation_metrics[category]
        metrics.generation_count += 1
        
        if success:
            metrics.success_count += 1
        else:
            metrics.failure_count += 1
        
        # Calculate success rate
        metrics.category_performance[category] = \
            metrics.success_count / metrics.generation_count
        
        # Save metrics
        self._save_learning_data()
    
    def _load_learning_data(self) -> None:
        """Load historical learning data."""
        learning_file = Path(__file__).parent / 'tool_generation_learning.json'
        
        if learning_file.exists():
            try:
                with open(learning_file, 'r') as f:
                    data = json.load(f)
                    
                # Restore metrics
                for category, metrics_data in data.get('metrics', {}).items():
                    metrics = ToolGenerationMetrics(**metrics_data)
                    self.generation_metrics[category] = metrics
                
                # Restore successful tools cache
                self.successful_tools_cache = data.get('successful_tools', {})
                
                # Restore pattern effectiveness
                self.pattern_effectiveness = defaultdict(
                    float, data.get('pattern_effectiveness', {})
                )
                
            except Exception as e:
                self.logger.error(f"Error loading learning data: {e}")
    
    def _save_learning_data(self) -> None:
        """Save learning data for persistence."""
        learning_file = Path(__file__).parent / 'tool_generation_learning.json'
        
        try:
            data = {
                'metrics': {
                    category: metrics.__dict__
                    for category, metrics in self.generation_metrics.items()
                },
                'successful_tools': self.successful_tools_cache,
                'pattern_effectiveness': dict(self.pattern_effectiveness),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(learning_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving learning data: {e}")
    
    def get_generation_report(self) -> Dict[str, Any]:
        """Get a report on tool generation performance."""
        report = {
            'total_generations': sum(
                m.generation_count for m in self.generation_metrics.values()
            ),
            'overall_success_rate': 0.0,
            'category_performance': {},
            'top_patterns': [],
            'recent_failures': []
        }
        
        # Calculate overall success rate
        total_success = sum(m.success_count for m in self.generation_metrics.values())
        total_count = report['total_generations']
        
        if total_count > 0:
            report['overall_success_rate'] = total_success / total_count
        
        # Category performance
        for category, metrics in self.generation_metrics.items():
            if metrics.generation_count > 0:
                report['category_performance'][category] = {
                    'success_rate': metrics.success_count / metrics.generation_count,
                    'total_generated': metrics.generation_count
                }
        
        # Top patterns by effectiveness
        sorted_patterns = sorted(
            self.pattern_effectiveness.items(),
            key=lambda x: x[1],
            reverse=True
        )
        report['top_patterns'] = sorted_patterns[:10]
        
        return report
    
    # Keep all base methods from original implementation
    def _load_base_templates(self) -> Dict[str, str]:
        """Load base templates (from original implementation)."""
        # This would include all the original templates
        return {
            # ... (include all original templates here)
        }
    
    def _load_error_fixes(self) -> Dict[str, str]:
        """Load common error fixes (from original implementation)."""
        return {
            "missing_state_manager_import": "from scripts.state_manager import StateManager",
            "missing_path_import": "from pathlib import Path",
            "missing_typing_import": "from typing import Dict, Any, List, Optional",
            "missing_json_import": "import json",
            "missing_os_import": "import os",
            "missing_datetime_import": "from datetime import datetime, timedelta",
            "missing_asyncio_import": "import asyncio",
            "undefined_timezone": "from datetime import timezone",
            "undefined_counter": "from collections import Counter",
            "undefined_defaultdict": "from collections import defaultdict"
        }
    
    def _discover_available_scripts(self) -> Dict[str, Dict[str, Any]]:
        """Discover available scripts (enhanced from original)."""
        # Use original discovery logic but add AI categorization
        discovered = {}
        # ... (original discovery logic)
        return discovered


def demonstrate_intelligent_system():
    """Demonstrate the intelligent tool generation system."""
    generator = IntelligentToolGenerationTemplates()
    
    # Example: Generate an intelligent tool
    result = generator.generate_intelligent_tool(
        name="smart_log_analyzer",
        description="Analyze log files for patterns and anomalies using AI",
        requirements="Parse multiple log formats, detect anomalies, generate insights, support real-time monitoring"
    )
    
    if result['success']:
        print("Tool generated successfully!")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Category: {result['analysis']['primary_category']}")
        print(f"Detected operations: {', '.join(result['analysis']['detected_operations'])}")
        print("\nGenerated code preview:")
        print(result['code'][:500] + "...")
    else:
        print(f"Generation failed: {result.get('error', 'Unknown error')}")
    
    # Show generation report
    report = generator.get_generation_report()
    print(f"\nGeneration Report:")
    print(f"Total generations: {report['total_generations']}")
    print(f"Overall success rate: {report['overall_success_rate']:.2%}")
    print(f"Top patterns: {report['top_patterns'][:3]}")


if __name__ == "__main__":
    demonstrate_intelligent_system()