"""
Enhanced Dynamic Method Creator with Rich Context

Creates methods with comprehensive understanding of expectations.
"""

import ast
import inspect
import types
import asyncio
import re
from typing import Any, Dict, Optional, List, Type, Tuple, get_type_hints
import logging

from scripts.dynamic_method_creator import DynamicMethodCreator
try:
    from scripts.return_type_analyzer import ReturnTypeAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    ReturnTypeAnalyzer = None

logger = logging.getLogger(__name__)


class EnhancedDynamicMethodCreator(DynamicMethodCreator):
    """Enhanced method creator with richer context awareness."""
    
    def __init__(self):
        super().__init__()
        self.execution_history = {}
        self.error_patterns = {}
        self.type_analyzer = ReturnTypeAnalyzer() if ANALYZER_AVAILABLE else None
        
    async def create_method(self, obj: Any, method_name: str, 
                          call_context: Dict[str, Any]) -> Optional[callable]:
        """Create method with enhanced context."""
        
        # Gather all available context
        enhanced_context = await self._gather_enhanced_context(
            obj, method_name, call_context
        )
        
        # Generate method with rich context
        return await super().create_method(obj, method_name, enhanced_context)
    
    async def _gather_enhanced_context(self, obj: Any, method_name: str,
                                     call_context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather comprehensive context for method creation."""
        
        enhanced_context = call_context.copy()
        
        # 1. Expected Return Type
        enhanced_context['expected_return'] = self._infer_return_type(
            obj, method_name, call_context
        )
        
        # 2. Data Flow Context
        enhanced_context['data_flow'] = self._analyze_data_flow(
            call_context.get('stack_trace', [])
        )
        
        # 3. Business Rules
        enhanced_context['business_rules'] = await self._extract_business_rules(
            obj, method_name
        )
        
        # 4. Related Methods
        enhanced_context['related_methods'] = self._find_related_methods(
            obj, method_name
        )
        
        # 5. Test Examples
        enhanced_context['test_examples'] = self._find_test_examples(
            obj.__class__.__name__, method_name
        )
        
        # 6. Error History
        enhanced_context['common_errors'] = self._get_error_patterns(
            obj.__class__.__name__, method_name
        )
        
        # 7. Performance Expectations
        enhanced_context['performance'] = self._infer_performance_requirements(
            method_name, call_context
        )
        
        # 8. State Dependencies
        enhanced_context['state_dependencies'] = self._analyze_state_dependencies(
            obj, method_name
        )
        
        # 9. Integration Points
        enhanced_context['integrations'] = self._identify_integrations(
            obj, method_name
        )
        
        # 10. Domain Vocabulary
        enhanced_context['domain_terms'] = self._extract_domain_vocabulary(
            obj, method_name
        )
        
        return enhanced_context
    
    def _infer_return_type(self, obj: Any, method_name: str, 
                           call_context: Dict[str, Any]) -> Dict[str, Any]:
        """Infer expected return type from usage context."""
        
        # Use the specialized analyzer if available
        if self.type_analyzer:
            analysis = self.type_analyzer.analyze_return_type(method_name, call_context)
            
            return {
                'type': self.type_analyzer.generate_type_hint(analysis),
                'shape': analysis.get('generic_args', None),
                'constraints': analysis.get('constraints', []),
                'examples': analysis.get('examples', []),
                'confidence': analysis.get('confidence', 0.0),
                'nullable': analysis.get('nullable', True)
            }
        
        # Fallback to basic inference
        return_context = {
            'type': 'Any',
            'shape': None,
            'constraints': [],
            'examples': [],
            'confidence': 0.3,
            'nullable': True
        }
        
        # Basic method name patterns
        if method_name.startswith('get_'):
            if 'list' in method_name or 'all' in method_name:
                return_context['type'] = 'List[Dict[str, Any]]'
                return_context['confidence'] = 0.7
            elif 'count' in method_name:
                return_context['type'] = 'int'
                return_context['constraints'] = ['value >= 0']
                return_context['confidence'] = 0.8
            elif 'rate' in method_name or 'score' in method_name:
                return_context['type'] = 'float'
                return_context['constraints'] = ['0.0 <= value <= 1.0']
                return_context['confidence'] = 0.8
                
        elif method_name.startswith('is_') or method_name.startswith('has_'):
            return_context['type'] = 'bool'
            return_context['nullable'] = False
            return_context['confidence'] = 0.9
            
        return return_context
    
    def _analyze_data_flow(self, stack_trace: List[str]) -> Dict[str, Any]:
        """Analyze how data flows through the call stack."""
        
        data_flow = {
            'source': None,
            'transformations': [],
            'destination': None,
            'async_chain': False
        }
        
        # Parse stack trace to understand data flow
        for frame in stack_trace:
            if 'await' in frame:
                data_flow['async_chain'] = True
            
            # Look for data sources
            if 'database' in frame or 'redis' in frame:
                data_flow['source'] = 'database'
            elif 'api' in frame or 'http' in frame:
                data_flow['source'] = 'external_api'
            elif 'file' in frame or 'json' in frame:
                data_flow['source'] = 'file_system'
        
        return data_flow
    
    async def _extract_business_rules(self, obj: Any, method_name: str) -> List[str]:
        """Extract business rules from documentation and code."""
        
        rules = []
        
        # Check class docstring
        class_doc = inspect.getdoc(obj.__class__)
        if class_doc:
            # Look for rules, constraints, requirements
            if 'must' in class_doc or 'should' in class_doc:
                rules.append(f"Class documentation: {class_doc}")
        
        # Check for validation methods
        for attr_name in dir(obj):
            if 'validate' in attr_name or 'check' in attr_name:
                rules.append(f"Validation method exists: {attr_name}")
        
        # Domain-specific rules based on method name
        if 'rate' in method_name:
            rules.append("Return value should be between 0.0 and 1.0")
        if 'count' in method_name:
            rules.append("Return value should be non-negative integer")
        if 'repository' in method_name:
            rules.append("Should handle repository name format: 'owner/repo'")
            
        return rules
    
    def _find_related_methods(self, obj: Any, method_name: str) -> List[Dict[str, Any]]:
        """Find methods that might be related or similar."""
        
        related = []
        
        # Extract components from method name
        parts = method_name.split('_')
        
        for name in dir(obj):
            if name.startswith('_'):
                continue
                
            # Check for similar patterns
            similarity_score = 0
            for part in parts:
                if part in name:
                    similarity_score += 1
                    
            if similarity_score > 0 and name != method_name:
                try:
                    method = getattr(obj, name)
                    if callable(method):
                        related.append({
                            'name': name,
                            'similarity': similarity_score,
                            'signature': str(inspect.signature(method)),
                            'is_async': inspect.iscoroutinefunction(method)
                        })
                except:
                    pass
                    
        # Sort by similarity
        related.sort(key=lambda x: x['similarity'], reverse=True)
        return related[:5]  # Top 5 related methods
    
    def _find_test_examples(self, class_name: str, method_name: str) -> List[str]:
        """Find test examples if available."""
        
        examples = []
        
        # Would search for test files
        test_patterns = [
            f"test_{class_name.lower()}.py",
            f"{class_name.lower()}_test.py",
            f"test_{method_name}.py"
        ]
        
        # For now, provide common patterns
        if method_name.startswith('get_'):
            examples.append("result = await obj.{method_name}()")
            examples.append("assert isinstance(result, list)")
            
        return examples
    
    def _get_error_patterns(self, class_name: str, method_name: str) -> List[Dict[str, str]]:
        """Get common error patterns for this type of method."""
        
        patterns = []
        
        # Common patterns based on method type
        if 'database' in class_name.lower() or 'repository' in method_name:
            patterns.append({
                'error': 'ConnectionError',
                'handling': 'Retry with exponential backoff'
            })
            
        if 'api' in class_name.lower() or 'http' in method_name:
            patterns.append({
                'error': 'TimeoutError',
                'handling': 'Set reasonable timeout, provide fallback'
            })
            
        return patterns
    
    def _infer_performance_requirements(self, method_name: str, 
                                      call_context: Dict[str, Any]) -> Dict[str, Any]:
        """Infer performance requirements."""
        
        requirements = {
            'latency': 'normal',
            'caching': False,
            'batch_capable': False
        }
        
        # Check if called in a loop
        stack = call_context.get('stack_trace', [])
        for frame in stack:
            if 'for' in frame and 'in' in frame:
                requirements['batch_capable'] = True
                requirements['caching'] = True
                
        # Method name patterns
        if 'batch' in method_name or 'bulk' in method_name:
            requirements['batch_capable'] = True
            
        if 'cache' in method_name or 'cached' in method_name:
            requirements['caching'] = True
            
        return requirements
    
    def _analyze_state_dependencies(self, obj: Any, method_name: str) -> List[str]:
        """Analyze what state the method might depend on."""
        
        dependencies = []
        
        # Check instance attributes
        for attr in dir(obj):
            if not attr.startswith('_') and not callable(getattr(obj, attr, None)):
                # This is a state attribute
                if any(part in attr for part in method_name.split('_')):
                    dependencies.append(attr)
                    
        return dependencies
    
    def _identify_integrations(self, obj: Any, method_name: str) -> List[str]:
        """Identify external systems this method might integrate with."""
        
        integrations = []
        
        # Check imports in the module
        if hasattr(obj, '__module__'):
            module = inspect.getmodule(obj)
            if module:
                for name, val in inspect.getmembers(module):
                    if inspect.ismodule(val):
                        if 'github' in str(val):
                            integrations.append('GitHub API')
                        elif 'redis' in str(val):
                            integrations.append('Redis')
                        elif 'openai' in str(val):
                            integrations.append('OpenAI API')
                            
        return integrations
    
    def _extract_domain_vocabulary(self, obj: Any, method_name: str) -> Dict[str, str]:
        """Extract domain-specific vocabulary."""
        
        vocabulary = {}
        
        # Common terms in this domain
        if 'repository' in method_name or 'repo' in method_name:
            vocabulary.update({
                'repository': 'GitHub repository in format owner/repo',
                'issue': 'GitHub issue with number and state',
                'pr': 'Pull request',
                'commit': 'Git commit with SHA'
            })
            
        if 'task' in method_name:
            vocabulary.update({
                'task': 'Unit of work with type, priority, and status',
                'work_item': 'Executable task with implementation details',
                'priority': 'HIGH, MEDIUM, or LOW',
                'task_type': 'FEATURE, BUG_FIX, OPTIMIZATION, etc.'
            })
            
        return vocabulary
    
    async def _generate_method_code(self, class_name: str, method_name: str,
                                   existing_methods: Dict[str, Dict[str, Any]],
                                   context: Dict[str, Any]) -> Optional[str]:
        """Generate method code with enhanced context."""
        
        prompt = f"""
Create a missing method with comprehensive context understanding.

CLASS: {class_name}
METHOD: {method_name}
CALLED WITH: args={context.get('args', ())}, kwargs={context.get('kwargs', {})}

=== EXPECTED BEHAVIOR ===
Return Type: {context.get('expected_return', {}).get('type', 'unknown')}
Constraints: {context.get('expected_return', {}).get('constraints', [])}
Examples: {context.get('expected_return', {}).get('examples', [])}

=== DATA FLOW ===
Source: {context.get('data_flow', {}).get('source', 'unknown')}
Async Chain: {context.get('data_flow', {}).get('async_chain', False)}

=== BUSINESS RULES ===
{chr(10).join(context.get('business_rules', []))}

=== RELATED METHODS ===
{self._format_related_methods(context.get('related_methods', []))}

=== COMMON ERRORS TO HANDLE ===
{self._format_error_patterns(context.get('common_errors', []))}

=== PERFORMANCE REQUIREMENTS ===
{context.get('performance', {})}

=== STATE DEPENDENCIES ===
The method may need to access: {context.get('state_dependencies', [])}

=== INTEGRATIONS ===
External systems: {context.get('integrations', [])}

=== DOMAIN VOCABULARY ===
{self._format_vocabulary(context.get('domain_terms', {}))}

=== EXISTING METHODS IN CLASS ===
{self._format_existing_methods(existing_methods)}

Generate a REAL, WORKING implementation that:
1. Matches the expected return type and constraints
2. Handles all common errors appropriately  
3. Integrates with the identified external systems
4. Follows the established patterns from related methods
5. Respects business rules and domain vocabulary
6. Meets performance requirements (caching, batching if needed)

CRITICAL: Make it actually work, not just a stub!
"""
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        return response.get('content', '')
    
    def _format_related_methods(self, methods: List[Dict[str, Any]]) -> str:
        """Format related methods for prompt."""
        if not methods:
            return "No related methods found"
            
        lines = []
        for method in methods[:3]:  # Top 3
            lines.append(f"- {method['name']}{method['signature']} "
                        f"(similarity: {method['similarity']}, "
                        f"async: {method['is_async']})")
        return '\n'.join(lines)
    
    def _format_error_patterns(self, patterns: List[Dict[str, str]]) -> str:
        """Format error patterns for prompt."""
        if not patterns:
            return "No specific error patterns identified"
            
        lines = []
        for pattern in patterns:
            lines.append(f"- {pattern['error']}: {pattern['handling']}")
        return '\n'.join(lines)
    
    def _format_vocabulary(self, vocabulary: Dict[str, str]) -> str:
        """Format domain vocabulary for prompt."""
        if not vocabulary:
            return "No domain-specific terms identified"
            
        lines = []
        for term, definition in vocabulary.items():
            lines.append(f"- {term}: {definition}")
        return '\n'.join(lines)