"""
Return Type Analyzer

Analyzes code context to determine expected return types for missing methods.
"""

import ast
import re
import inspect
from typing import Any, Dict, Optional, List, Type, Union
import logging

logger = logging.getLogger(__name__)


class ReturnTypeAnalyzer:
    """Analyzes expected return types from usage context."""
    
    def __init__(self):
        self.type_patterns = {
            # Variable name patterns
            'list_patterns': ['list', 'items', 'results', 'history', 'records', 'entries'],
            'dict_patterns': ['data', 'info', 'details', 'config', 'params', 'context'],
            'bool_patterns': ['is_', 'has_', 'can_', 'should_', 'exists', 'valid'],
            'float_patterns': ['rate', 'score', 'percentage', 'ratio', 'probability'],
            'int_patterns': ['count', 'number', 'total', 'size', 'length', 'id'],
            'str_patterns': ['name', 'text', 'message', 'description', 'url', 'path']
        }
        
    def analyze_return_type(self, method_name: str, call_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze expected return type from method name and usage context.
        
        Args:
            method_name: Name of the missing method
            call_context: Context about how the method is called
            
        Returns:
            Dictionary with type information and constraints
        """
        analysis = {
            'primary_type': 'Any',
            'generic_args': [],
            'nullable': True,
            'constraints': [],
            'examples': [],
            'confidence': 0.0
        }
        
        # Analyze from multiple sources
        name_analysis = self._analyze_from_name(method_name)
        usage_analysis = self._analyze_from_usage(call_context)
        pattern_analysis = self._analyze_from_patterns(method_name, call_context)
        
        # Combine analyses with weights
        analysis = self._combine_analyses(
            name_analysis, 
            usage_analysis, 
            pattern_analysis
        )
        
        return analysis
    
    def _analyze_from_name(self, method_name: str) -> Dict[str, Any]:
        """Analyze return type from method name patterns."""
        
        analysis = {
            'primary_type': 'Any',
            'generic_args': [],
            'nullable': True,
            'constraints': [],
            'confidence': 0.0
        }
        
        name_lower = method_name.lower()
        
        # Boolean methods
        if (method_name.startswith('is_') or 
            method_name.startswith('has_') or 
            method_name.startswith('can_') or
            method_name.startswith('should_')):
            analysis['primary_type'] = 'bool'
            analysis['nullable'] = False
            analysis['confidence'] = 0.9
            return analysis
        
        # Collection methods
        if 'list' in name_lower or 'all' in name_lower:
            analysis['primary_type'] = 'List'
            analysis['generic_args'] = ['Dict[str, Any]']
            analysis['confidence'] = 0.8
        
        elif 'dict' in name_lower or 'map' in name_lower:
            analysis['primary_type'] = 'Dict'
            analysis['generic_args'] = ['str', 'Any']
            analysis['confidence'] = 0.8
            
        # Numeric methods
        elif any(pattern in name_lower for pattern in ['count', 'number', 'total']):
            analysis['primary_type'] = 'int'
            analysis['nullable'] = False
            analysis['constraints'] = ['value >= 0']
            analysis['confidence'] = 0.85
            
        elif any(pattern in name_lower for pattern in ['rate', 'score', 'percentage']):
            analysis['primary_type'] = 'float'
            analysis['nullable'] = False
            analysis['constraints'] = ['0.0 <= value <= 1.0']
            analysis['confidence'] = 0.85
            
        # String methods
        elif 'name' in name_lower or 'path' in name_lower:
            analysis['primary_type'] = 'str'
            analysis['confidence'] = 0.7
            
        return analysis
    
    def _analyze_from_usage(self, call_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze return type from how the result is used."""
        
        analysis = {
            'primary_type': 'Any',
            'generic_args': [],
            'nullable': True,
            'constraints': [],
            'confidence': 0.0
        }
        
        # Get the code context where method is called
        caller_info = call_context.get('caller_info', {})
        code_context = caller_info.get('code_context', '')
        
        if not code_context:
            return analysis
            
        # Look for variable assignment
        assignment_match = re.search(r'(\w+)\s*=\s*(?:await\s+)?.*?\.(\w+)\(', code_context)
        if assignment_match:
            var_name = assignment_match.group(1)
            
            # Check variable name patterns
            for pattern_type, patterns in self.type_patterns.items():
                if any(p in var_name.lower() for p in patterns):
                    if 'list' in pattern_type:
                        analysis['primary_type'] = 'List'
                        analysis['confidence'] = 0.7
                    elif 'dict' in pattern_type:
                        analysis['primary_type'] = 'Dict'
                        analysis['confidence'] = 0.7
                    elif 'bool' in pattern_type:
                        analysis['primary_type'] = 'bool'
                        analysis['nullable'] = False
                        analysis['confidence'] = 0.8
                    elif 'float' in pattern_type:
                        analysis['primary_type'] = 'float'
                        analysis['nullable'] = False
                        analysis['confidence'] = 0.8
                    elif 'int' in pattern_type:
                        analysis['primary_type'] = 'int'
                        analysis['nullable'] = False
                        analysis['confidence'] = 0.8
                    break
        
        # Look for how the result is used after assignment
        usage_patterns = self._find_usage_patterns(code_context)
        if usage_patterns:
            analysis = self._refine_from_usage_patterns(analysis, usage_patterns)
            
        return analysis
    
    def _analyze_from_patterns(self, method_name: str, 
                               call_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze from common patterns in the codebase."""
        
        analysis = {
            'primary_type': 'Any',
            'generic_args': [],
            'nullable': True,
            'constraints': [],
            'confidence': 0.0
        }
        
        # Check for async patterns
        stack_trace = call_context.get('stack_trace', [])
        is_async = any('await' in frame for frame in stack_trace)
        
        # Repository pattern
        if 'repository' in method_name or 'repo' in method_name:
            if 'list' in method_name or 'all' in method_name:
                analysis['primary_type'] = 'List'
                analysis['generic_args'] = ['Dict[str, Any]']
                analysis['confidence'] = 0.75
            elif 'get' in method_name:
                analysis['primary_type'] = 'Optional[Dict[str, Any]]'
                analysis['confidence'] = 0.75
                
        # Task/Work pattern
        elif 'task' in method_name or 'work' in method_name:
            if 'create' in method_name:
                analysis['primary_type'] = 'Dict[str, Any]'
                analysis['nullable'] = False
                analysis['confidence'] = 0.8
                
        return analysis
    
    def _find_usage_patterns(self, code_context: str) -> List[str]:
        """Find patterns of how the return value is used."""
        
        patterns = []
        
        # Look for method calls on the result
        if '.append(' in code_context:
            patterns.append('list_method')
        if '.items()' in code_context or '.keys()' in code_context:
            patterns.append('dict_method')
        if 'len(' in code_context:
            patterns.append('has_length')
        if '[' in code_context and ']' in code_context:
            patterns.append('indexable')
        if 'for ' in code_context and ' in ' in code_context:
            patterns.append('iterable')
            
        return patterns
    
    def _refine_from_usage_patterns(self, analysis: Dict[str, Any], 
                                   patterns: List[str]) -> Dict[str, Any]:
        """Refine type analysis based on usage patterns."""
        
        if 'list_method' in patterns or 'indexable' in patterns:
            if analysis['primary_type'] == 'Any':
                analysis['primary_type'] = 'List'
                analysis['confidence'] = max(analysis['confidence'], 0.6)
                
        if 'dict_method' in patterns:
            if analysis['primary_type'] == 'Any':
                analysis['primary_type'] = 'Dict'
                analysis['confidence'] = max(analysis['confidence'], 0.6)
                
        if 'iterable' in patterns and analysis['primary_type'] == 'Any':
            analysis['primary_type'] = 'List'  # Default to List for iterables
            analysis['confidence'] = max(analysis['confidence'], 0.5)
            
        return analysis
    
    def _combine_analyses(self, *analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Combine multiple analyses with confidence weighting."""
        
        # Find the analysis with highest confidence
        best_analysis = max(analyses, key=lambda a: a.get('confidence', 0))
        
        # If confidence is too low, check for consensus
        if best_analysis['confidence'] < 0.7:
            # Count type votes
            type_votes = {}
            for analysis in analyses:
                if analysis['confidence'] > 0.3:
                    ptype = analysis['primary_type']
                    type_votes[ptype] = type_votes.get(ptype, 0) + analysis['confidence']
            
            # If there's a clear winner, use it
            if type_votes:
                best_type = max(type_votes.items(), key=lambda x: x[1])
                if best_type[1] > 1.0:  # Combined confidence
                    best_analysis['primary_type'] = best_type[0]
                    best_analysis['confidence'] = min(best_type[1] / len(analyses), 0.9)
        
        # Add examples based on type
        best_analysis['examples'] = self._generate_examples(best_analysis['primary_type'])
        
        return best_analysis
    
    def _generate_examples(self, primary_type: str) -> List[str]:
        """Generate example return values for a type."""
        
        examples = []
        
        if primary_type == 'List':
            examples = [
                "[]",
                "[{'id': 1, 'name': 'example'}]",
                "[item1, item2, item3]"
            ]
        elif primary_type == 'Dict':
            examples = [
                "{}",
                "{'status': 'success', 'data': ...}",
                "{'id': 123, 'name': 'example', 'active': True}"
            ]
        elif primary_type == 'bool':
            examples = ["True", "False"]
        elif primary_type == 'int':
            examples = ["0", "42", "len(items)"]
        elif primary_type == 'float':
            examples = ["0.0", "0.85", "1.0"]
        elif primary_type == 'str':
            examples = ["''", "'example'", "f'Result: {value}'"]
            
        return examples
    
    def generate_type_hint(self, analysis: Dict[str, Any]) -> str:
        """Generate a Python type hint from analysis."""
        
        primary = analysis['primary_type']
        
        if primary == 'Any':
            return 'Any'
            
        if analysis['generic_args']:
            args = ', '.join(analysis['generic_args'])
            hint = f"{primary}[{args}]"
        else:
            hint = primary
            
        if analysis['nullable'] and not hint.startswith('Optional'):
            hint = f"Optional[{hint}]"
            
        return hint