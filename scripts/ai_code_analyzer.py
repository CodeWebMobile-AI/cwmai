"""
AI-Powered Code Analyzer

Uses AI to understand and analyze code semantically rather than relying on regex patterns.
"""

import ast
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import os
import re

from ai_brain import IntelligentAIBrain
from safe_self_improver import ModificationType


@dataclass
class CodeImprovement:
    """Represents an AI-suggested code improvement."""
    type: ModificationType
    description: str
    original_code: str
    improved_code: str
    explanation: str
    confidence: float
    line_start: int
    line_end: int
    impact_analysis: Dict[str, Any] = field(default_factory=dict)
    test_suggestions: List[str] = field(default_factory=list)


class AICodeAnalyzer:
    """Analyzes code using AI for intelligent improvements."""
    
    def __init__(self, ai_brain: IntelligentAIBrain):
        """Initialize with AI brain."""
        self.ai_brain = ai_brain
        self.improvement_templates = self._load_improvement_templates()
        
    def _load_improvement_templates(self) -> Dict[str, str]:
        """Load templates for different improvement types."""
        return {
            'optimization': """
Analyze for performance optimizations:
- Loops that could be list/dict comprehensions
- Inefficient algorithms (O(n²) that could be O(n))
- Repeated computations that could be cached
- String concatenation in loops
- Unnecessary type conversions
""",
            'pythonic': """
Analyze for Pythonic improvements:
- Using built-in functions (zip, enumerate, any, all)
- Context managers for resource handling
- Dictionary get() vs if-in checks
- F-strings vs format/concatenation
- Idiomatic Python patterns
""",
            'quality': """
Analyze for code quality:
- Function complexity (should be broken down?)
- Variable naming clarity
- Magic numbers that should be constants
- Duplicate code patterns
- Error handling completeness
""",
            'security': """
Analyze for security issues:
- SQL injection vulnerabilities
- Command injection risks
- Path traversal vulnerabilities
- Unsafe deserialization
- Hardcoded secrets
""",
            'documentation': """
Analyze for documentation needs:
- Missing function/class docstrings
- Unclear parameter descriptions
- Missing return type hints
- Complex logic without comments
- API documentation completeness
"""
        }
    
    async def analyze_code_for_improvements(self, code: str, file_path: str,
                                          context: Optional[Dict[str, Any]] = None) -> List[CodeImprovement]:
        """Analyze code using AI to find improvements.
        
        Args:
            code: The source code to analyze
            file_path: Path to the file being analyzed
            context: Additional context about the code
            
        Returns:
            List of code improvements
        """
        # First, get AST information
        ast_info = self._extract_ast_info(code)
        
        # Build comprehensive prompt
        prompt = self._build_analysis_prompt(code, file_path, ast_info, context)
        
        # Get AI analysis
        response_dict = await self.ai_brain.generate_enhanced_response(prompt)
        ai_response = response_dict.get('content', '')
        
        # Parse AI response into improvements
        improvements = self._parse_ai_improvements(ai_response, code)
        
        # Validate and enhance improvements
        validated_improvements = await self._validate_improvements(improvements, code)
        
        return validated_improvements
    
    def _extract_ast_info(self, code: str) -> Dict[str, Any]:
        """Extract structural information from code using AST."""
        try:
            tree = ast.parse(code)
            
            info = {
                'functions': [],
                'classes': [],
                'imports': [],
                'global_vars': [],
                'complexity_score': 0
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)],
                        'docstring': ast.get_docstring(node),
                        'line_start': node.lineno,
                        'complexity': self._calculate_complexity(node)
                    }
                    info['functions'].append(func_info)
                    
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'bases': [b.id for b in node.bases if isinstance(b, ast.Name)],
                        'methods': [],
                        'docstring': ast.get_docstring(node),
                        'line_start': node.lineno
                    }
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_info['methods'].append(item.name)
                    
                    info['classes'].append(class_info)
                    
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        info['imports'].append(alias.name)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        info['imports'].append(node.module)
            
            info['complexity_score'] = sum(f['complexity'] for f in info['functions'])
            
            return info
            
        except Exception as e:
            return {'error': str(e), 'functions': [], 'classes': [], 'imports': []}
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a node."""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
                
        return complexity
    
    def _build_analysis_prompt(self, code: str, file_path: str, 
                             ast_info: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
        """Build comprehensive prompt for AI analysis."""
        prompt_parts = [
            "Analyze this Python code for improvements. Be specific and provide exact code changes.",
            f"\nFile: {file_path}",
            f"\nStructure: {len(ast_info['functions'])} functions, {len(ast_info['classes'])} classes",
            f"\nComplexity score: {ast_info.get('complexity_score', 'N/A')}",
        ]
        
        if context:
            prompt_parts.append(f"\nContext: {json.dumps(context, indent=2)}")
        
        prompt_parts.append("\nAnalyze for these improvement categories:")
        
        # Add category-specific analysis
        for category, template in self.improvement_templates.items():
            prompt_parts.append(f"\n{category.upper()}:{template}")
        
        prompt_parts.append(f"\nCode to analyze:\n```python\n{code}\n```")
        
        prompt_parts.append("""
Return improvements as JSON array with this structure:
{
  "improvements": [
    {
      "type": "optimization|pythonic|quality|security|documentation",
      "description": "Brief description of the improvement",
      "original_code": "Exact code to be replaced (including indentation)",
      "improved_code": "The improved version",
      "explanation": "Why this improvement is beneficial",
      "confidence": 0.0-1.0,
      "line_start": line_number,
      "line_end": line_number,
      "impact": {
        "performance": "high|medium|low|none",
        "readability": "improved|same|reduced",
        "maintainability": "improved|same|reduced"
      },
      "tests_needed": ["Description of tests to verify this change"]
    }
  ]
}

Important:
- Only suggest improvements you're confident about (confidence > 0.7)
- Preserve exact indentation in original_code and improved_code
- Ensure improved_code is syntactically correct
- Consider the broader context and avoid breaking changes
- Suggest meaningful improvements, not trivial changes
""")
        
        return '\n'.join(prompt_parts)
    
    def _parse_ai_improvements(self, ai_response: str, original_code: str) -> List[CodeImprovement]:
        """Parse AI response into CodeImprovement objects."""
        improvements = []
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*"improvements".*\}', ai_response, re.DOTALL)
            if not json_match:
                return improvements
            
            data = json.loads(json_match.group())
            
            for imp_data in data.get('improvements', []):
                # Map string type to enum
                type_mapping = {
                    'optimization': ModificationType.OPTIMIZATION,
                    'pythonic': ModificationType.OPTIMIZATION,
                    'quality': ModificationType.REFACTORING,
                    'security': ModificationType.SECURITY,
                    'documentation': ModificationType.DOCUMENTATION
                }
                
                imp_type = type_mapping.get(
                    imp_data.get('type', 'optimization'),
                    ModificationType.OPTIMIZATION
                )
                
                improvement = CodeImprovement(
                    type=imp_type,
                    description=imp_data.get('description', ''),
                    original_code=imp_data.get('original_code', ''),
                    improved_code=imp_data.get('improved_code', ''),
                    explanation=imp_data.get('explanation', ''),
                    confidence=float(imp_data.get('confidence', 0.8)),
                    line_start=int(imp_data.get('line_start', 1)),
                    line_end=int(imp_data.get('line_end', 1)),
                    impact_analysis=imp_data.get('impact', {}),
                    test_suggestions=imp_data.get('tests_needed', [])
                )
                
                improvements.append(improvement)
                
        except Exception as e:
            print(f"Error parsing AI improvements: {e}")
            
        return improvements
    
    async def _validate_improvements(self, improvements: List[CodeImprovement], 
                                   original_code: str) -> List[CodeImprovement]:
        """Validate and enhance improvements."""
        validated = []
        
        for improvement in improvements:
            # Check if original code exists in file
            if improvement.original_code not in original_code:
                # Try to find fuzzy match
                improvement = await self._fuzzy_match_improvement(improvement, original_code)
                if not improvement:
                    continue
            
            # Validate syntax of improved code
            if not self._is_valid_python(improvement.improved_code, original_code, improvement):
                continue
            
            # Enhance with additional analysis
            improvement = await self._enhance_improvement(improvement, original_code)
            
            validated.append(improvement)
        
        return validated
    
    async def _fuzzy_match_improvement(self, improvement: CodeImprovement, 
                                     original_code: str) -> Optional[CodeImprovement]:
        """Try to fuzzy match the improvement if exact match fails."""
        lines = original_code.split('\n')
        
        # Try to find similar code around the specified line numbers
        start = max(0, improvement.line_start - 1)
        end = min(len(lines), improvement.line_end)
        
        if start < len(lines):
            # Get the code block around the specified lines
            code_block = '\n'.join(lines[start:end])
            
            # Ask AI to correct the match
            prompt = f"""
The following improvement has incorrect original_code matching.
Find the correct code to replace in this block:

Code block (lines {start+1}-{end}):
{code_block}

Improvement description: {improvement.description}
Incorrectly matched original: {improvement.original_code}
Suggested improvement: {improvement.improved_code}

Return the exact original_code that should be replaced, preserving indentation.
"""
            
            response_dict = await self.ai_brain.generate_enhanced_response(prompt)
            response = response_dict.get('content', '')
            
            # Extract corrected original code
            if response and response.strip() in original_code:
                improvement.original_code = response.strip()
                return improvement
        
        return None
    
    def _is_valid_python(self, code: str, full_code: str, improvement: CodeImprovement) -> bool:
        """Check if the improved code is valid Python."""
        try:
            # Try to parse just the improved code
            ast.parse(improvement.improved_code)
            
            # Try to parse the full code with replacement
            modified_code = full_code.replace(
                improvement.original_code,
                improvement.improved_code
            )
            ast.parse(modified_code)
            
            return True
            
        except SyntaxError:
            return False
    
    async def _enhance_improvement(self, improvement: CodeImprovement, 
                                  original_code: str) -> CodeImprovement:
        """Enhance improvement with additional analysis."""
        # Calculate actual line numbers if not provided
        if improvement.line_start == 1 and improvement.line_end == 1:
            lines = original_code.split('\n')
            for i, line in enumerate(lines):
                if improvement.original_code.split('\n')[0] in line:
                    improvement.line_start = i + 1
                    improvement.line_end = i + len(improvement.original_code.split('\n'))
                    break
        
        # Add specific test suggestions based on improvement type
        if not improvement.test_suggestions:
            improvement.test_suggestions = self._generate_test_suggestions(improvement)
        
        return improvement
    
    def _generate_test_suggestions(self, improvement: CodeImprovement) -> List[str]:
        """Generate test suggestions for an improvement."""
        suggestions = []
        
        if improvement.type == ModificationType.OPTIMIZATION:
            suggestions.extend([
                "Test that output remains identical to original",
                "Benchmark performance improvement",
                "Test edge cases (empty input, large input)"
            ])
        elif improvement.type == ModificationType.SECURITY:
            suggestions.extend([
                "Test with malicious input",
                "Verify security vulnerability is fixed",
                "Test authorization/authentication if applicable"
            ])
        elif improvement.type == ModificationType.REFACTORING:
            suggestions.extend([
                "Test all existing functionality still works",
                "Test new structure maintains same behavior",
                "Check for any breaking API changes"
            ])
        
        return suggestions
    
    async def analyze_file_with_context(self, file_path: str, 
                                      related_files: Optional[List[str]] = None) -> List[CodeImprovement]:
        """Analyze a file considering its context in the project."""
        with open(file_path, 'r') as f:
            code = f.read()
        
        # Build context
        context = {
            'file_type': self._determine_file_type(file_path),
            'imports': self._extract_imports_context(code),
            'related_files': related_files or [],
            'file_size': len(code.split('\n')),
            'last_modified': os.path.getmtime(file_path)
        }
        
        # If we have related files, analyze dependencies
        if related_files:
            context['dependencies'] = await self._analyze_dependencies(file_path, related_files)
        
        return await self.analyze_code_for_improvements(code, file_path, context)
    
    def _determine_file_type(self, file_path: str) -> str:
        """Determine the type of Python file."""
        path_lower = file_path.lower()
        
        if 'test' in path_lower:
            return 'test'
        elif 'setup.py' in path_lower:
            return 'setup'
        elif '__init__.py' in path_lower:
            return 'package_init'
        elif 'cli' in path_lower or 'command' in path_lower:
            return 'cli'
        elif 'api' in path_lower or 'route' in path_lower:
            return 'api'
        elif 'model' in path_lower:
            return 'model'
        elif 'util' in path_lower or 'helper' in path_lower:
            return 'utility'
        else:
            return 'general'
    
    def _extract_imports_context(self, code: str) -> Dict[str, List[str]]:
        """Extract import context from code."""
        imports = {
            'stdlib': [],
            'third_party': [],
            'local': []
        }
        
        try:
            tree = ast.parse(code)
            
            stdlib_modules = {
                'os', 'sys', 'time', 'datetime', 'json', 're', 'math',
                'random', 'collections', 'itertools', 'functools'
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        if module in stdlib_modules:
                            imports['stdlib'].append(alias.name)
                        elif module.startswith('.'):
                            imports['local'].append(alias.name)
                        else:
                            imports['third_party'].append(alias.name)
                            
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]
                        if module in stdlib_modules:
                            imports['stdlib'].append(node.module)
                        elif node.level > 0:  # Relative import
                            imports['local'].append(node.module or '')
                        else:
                            imports['third_party'].append(node.module)
                            
        except:
            pass
            
        return imports
    
    async def _analyze_dependencies(self, file_path: str, 
                                  related_files: List[str]) -> Dict[str, Any]:
        """Analyze how this file relates to others."""
        dependencies = {
            'imports_from': [],
            'imported_by': [],
            'shared_functions': []
        }
        
        # This would analyze the related files to understand dependencies
        # Simplified for now
        
        return dependencies