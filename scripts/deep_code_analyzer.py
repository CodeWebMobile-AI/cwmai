#!/usr/bin/env python3
"""
Deep Code Analyzer - Advanced semantic code understanding and analysis
"""

import ast
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import importlib.util
import inspect
from pathlib import Path

from scripts.http_ai_client import HTTPAIClient
import logging

logger = logging.getLogger(__name__)


@dataclass
class CodeIntent:
    """Represents the analyzed intent of code"""
    purpose: str
    inputs: List[str]
    outputs: List[str]
    side_effects: List[str]
    patterns_used: List[str]
    quality_score: float
    improvement_areas: List[str]


@dataclass
class RefactoringProposal:
    """Represents a code refactoring suggestion"""
    description: str
    rationale: str
    original_code: str
    refactored_code: str
    impact: str
    risk_level: str  # low, medium, high
    estimated_improvement: Dict[str, Any]


class DeepCodeAnalyzer:
    """Advanced code analysis with semantic understanding"""
    
    def __init__(self):
        self.ai_client = HTTPAIClient()
        self.pattern_library = self._load_pattern_library()
        self.language_patterns = self._load_language_patterns()
        
    def _load_pattern_library(self) -> Dict[str, Any]:
        """Load common code patterns and anti-patterns"""
        return {
            "design_patterns": {
                "singleton": r"class.*\(.*\):.*(?:__instance|_instance).*=.*None",
                "factory": r"class.*Factory.*:|def.*create.*\(",
                "observer": r"def.*(?:attach|detach|notify).*\(",
                "decorator": r"@\w+|def.*\(.*func.*\).*:",
                "strategy": r"class.*Strategy.*:|def.*execute.*\("
            },
            "anti_patterns": {
                "god_class": {"max_methods": 20, "max_lines": 500},
                "long_method": {"max_lines": 50},
                "deep_nesting": {"max_depth": 4},
                "duplicate_code": {"min_similarity": 0.8},
                "magic_numbers": r"(?<!\w)\d+(?!\w)(?![\.\d])"
            },
            "code_smells": {
                "long_parameter_list": {"max_params": 5},
                "feature_envy": "accessing_other_class_data_frequently",
                "data_clumps": "same_parameters_in_multiple_methods",
                "primitive_obsession": "overuse_of_primitives"
            }
        }
        
    def _load_language_patterns(self) -> Dict[str, Any]:
        """Load cross-language patterns and best practices"""
        return {
            "functional": {
                "map_filter_reduce": "Use functional programming patterns",
                "immutability": "Prefer immutable data structures",
                "pure_functions": "Functions without side effects"
            },
            "object_oriented": {
                "solid_principles": "Single responsibility, Open/closed, etc.",
                "composition_over_inheritance": "Favor composition",
                "dependency_injection": "Inject dependencies"
            },
            "async_patterns": {
                "async_await": "Modern async patterns",
                "promises": "Promise-based async",
                "reactive": "Reactive programming patterns"
            }
        }
    
    def analyze_code_intent(self, code: str, context: Optional[Dict] = None) -> CodeIntent:
        """Understand what code is trying to do at a semantic level"""
        try:
            # Parse code structure
            tree = ast.parse(code)
            
            # Extract code components
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            
            # Analyze patterns
            detected_patterns = self._detect_patterns(code)
            
            # Use AI to understand intent
            ai_analysis = self._analyze_with_ai(code, context)
            
            # Extract inputs/outputs
            inputs, outputs = self._analyze_data_flow(tree)
            
            # Detect side effects
            side_effects = self._detect_side_effects(tree)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(code, tree)
            
            # Identify improvement areas
            improvement_areas = self._identify_improvements(code, tree, detected_patterns)
            
            return CodeIntent(
                purpose=ai_analysis.get("purpose", "Unknown"),
                inputs=inputs,
                outputs=outputs,
                side_effects=side_effects,
                patterns_used=detected_patterns,
                quality_score=quality_score,
                improvement_areas=improvement_areas
            )
            
        except Exception as e:
            logger.info(f"Error analyzing code intent: {str(e)}")
            return CodeIntent(
                purpose="Error in analysis",
                inputs=[],
                outputs=[],
                side_effects=[],
                patterns_used=[],
                quality_score=0.0,
                improvement_areas=["Failed to analyze code"]
            )
    
    def suggest_refactoring(self, code: str, intent: Optional[CodeIntent] = None) -> List[RefactoringProposal]:
        """Propose better implementations while maintaining functionality"""
        proposals = []
        
        try:
            # Analyze intent if not provided
            if not intent:
                intent = self.analyze_code_intent(code)
            
            # Check for design pattern improvements
            pattern_proposals = self._suggest_pattern_improvements(code, intent)
            proposals.extend(pattern_proposals)
            
            # Check for code smell fixes
            smell_proposals = self._fix_code_smells(code)
            proposals.extend(smell_proposals)
            
            # Check for performance optimizations
            perf_proposals = self._suggest_performance_improvements(code)
            proposals.extend(perf_proposals)
            
            # Check for readability improvements
            readability_proposals = self._improve_readability(code)
            proposals.extend(readability_proposals)
            
            # Use AI for advanced refactoring suggestions
            ai_proposals = self._get_ai_refactoring_suggestions(code, intent)
            proposals.extend(ai_proposals)
            
            # Apply cross-language best practices
            cross_lang_proposals = self._apply_cross_language_patterns(code, intent)
            proposals.extend(cross_lang_proposals)
            
        except Exception as e:
            logger.info(f"Error suggesting refactoring: {str(e)}")
        
        return proposals
    
    def _detect_patterns(self, code: str) -> List[str]:
        """Detect design patterns in code"""
        detected = []
        
        for pattern_name, pattern_regex in self.pattern_library["design_patterns"].items():
            if re.search(pattern_regex, code, re.MULTILINE | re.IGNORECASE):
                detected.append(pattern_name)
        
        return detected
    
    def _analyze_with_ai(self, code: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Use AI to understand code purpose and intent"""
        prompt = f"""Analyze this code and determine its intent and purpose:

```python
{code}
```

Context: {json.dumps(context) if context else 'None'}

Provide a JSON response with:
1. purpose: Brief description of what the code does
2. main_functionality: List of key functions
3. design_choices: Notable design decisions
4. potential_issues: Any identified problems
"""
        
        try:
            response = self.ai_client.generate_response(prompt)
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.info(f"AI analysis error: {str(e)}")
        
        return {"purpose": "Unable to determine", "main_functionality": []}
    
    def _analyze_data_flow(self, tree: ast.AST) -> Tuple[List[str], List[str]]:
        """Analyze inputs and outputs of code"""
        inputs = []
        outputs = []
        
        class DataFlowVisitor(ast.NodeVisitor):
            def __init__(self):
                self.inputs = set()
                self.outputs = set()
                
            def visit_FunctionDef(self, node):
                # Function parameters are inputs
                for arg in node.args.args:
                    self.inputs.add(arg.arg)
                
                # Return values are outputs
                for child in ast.walk(node):
                    if isinstance(child, ast.Return) and child.value:
                        if isinstance(child.value, ast.Name):
                            self.outputs.add(child.value.id)
                
                self.generic_visit(node)
        
        visitor = DataFlowVisitor()
        visitor.visit(tree)
        
        return list(visitor.inputs), list(visitor.outputs)
    
    def _detect_side_effects(self, tree: ast.AST) -> List[str]:
        """Detect side effects in code"""
        side_effects = []
        
        class SideEffectVisitor(ast.NodeVisitor):
            def __init__(self):
                self.effects = []
                
            def visit_Call(self, node):
                # Detect I/O operations
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['print', 'open', 'write']:
                        self.effects.append(f"I/O operation: {node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['write', 'save', 'dump', 'send']:
                        self.effects.append(f"External operation: {node.func.attr}")
                
                self.generic_visit(node)
                
            def visit_Import(self, node):
                # Track external dependencies
                for alias in node.names:
                    self.effects.append(f"External dependency: {alias.name}")
                self.generic_visit(node)
        
        visitor = SideEffectVisitor()
        visitor.visit(tree)
        
        return visitor.effects
    
    def _calculate_quality_score(self, code: str, tree: ast.AST) -> float:
        """Calculate code quality score (0-1)"""
        score = 1.0
        penalties = 0.0
        
        # Check complexity
        complexity = self._calculate_complexity(tree)
        if complexity > 10:
            penalties += 0.1 * (complexity - 10) / 10
        
        # Check line length
        lines = code.split('\n')
        long_lines = sum(1 for line in lines if len(line) > 80)
        if long_lines > 0:
            penalties += 0.05 * (long_lines / len(lines))
        
        # Check for anti-patterns
        for pattern_name, pattern in self.pattern_library["anti_patterns"].items():
            if isinstance(pattern, dict):
                if pattern_name == "magic_numbers":
                    magic_numbers = len(re.findall(self.pattern_library["anti_patterns"]["magic_numbers"], code))
                    if magic_numbers > 3:
                        penalties += 0.1
            
        # Check documentation
        if '"""' not in code and "'''" not in code:
            penalties += 0.15  # No docstrings
        
        return max(0.0, score - penalties)
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _identify_improvements(self, code: str, tree: ast.AST, patterns: List[str]) -> List[str]:
        """Identify areas for improvement"""
        improvements = []
        
        # Check for missing patterns
        if "error_handling" not in code.lower() and "try" not in code:
            improvements.append("Add error handling")
        
        # Check for missing type hints
        has_type_hints = any('->' in line for line in code.split('\n'))
        if not has_type_hints:
            improvements.append("Add type hints")
        
        # Check for missing tests
        if "test" not in code.lower() and "assert" not in code:
            improvements.append("Add unit tests")
        
        # Check for optimization opportunities
        if "for" in code and "append" in code:
            improvements.append("Consider list comprehension")
        
        return improvements
    
    def _suggest_pattern_improvements(self, code: str, intent: CodeIntent) -> List[RefactoringProposal]:
        """Suggest design pattern improvements"""
        proposals = []
        
        # Example: Suggest Factory pattern for object creation
        if "if" in code and "create" in code.lower():
            proposals.append(RefactoringProposal(
                description="Implement Factory Pattern",
                rationale="Multiple conditional object creation can be simplified with Factory pattern",
                original_code=code[:200] + "...",
                refactored_code="Use Factory pattern for object creation",
                impact="Improved maintainability and extensibility",
                risk_level="medium",
                estimated_improvement={"maintainability": "+30%", "complexity": "-20%"}
            ))
        
        return proposals
    
    def _fix_code_smells(self, code: str) -> List[RefactoringProposal]:
        """Fix identified code smells"""
        proposals = []
        
        # Check for long methods
        functions = re.findall(r'def\s+\w+\s*\([^)]*\):[^}]+', code, re.MULTILINE)
        for func in functions:
            lines = func.count('\n')
            if lines > self.pattern_library["anti_patterns"]["long_method"]["max_lines"]:
                proposals.append(RefactoringProposal(
                    description="Split long method",
                    rationale=f"Method has {lines} lines, exceeding recommended {self.pattern_library['anti_patterns']['long_method']['max_lines']}",
                    original_code=func[:100] + "...",
                    refactored_code="Split into smaller, focused methods",
                    impact="Improved readability and testability",
                    risk_level="low",
                    estimated_improvement={"readability": "+40%", "testability": "+30%"}
                ))
        
        return proposals
    
    def _suggest_performance_improvements(self, code: str) -> List[RefactoringProposal]:
        """Suggest performance optimizations"""
        proposals = []
        
        # Check for list concatenation in loops
        if "for" in code and "+=" in code and "[]" in code:
            proposals.append(RefactoringProposal(
                description="Use list comprehension or extend()",
                rationale="List concatenation in loops is inefficient",
                original_code="for item in items: result += [process(item)]",
                refactored_code="result = [process(item) for item in items]",
                impact="O(n²) to O(n) complexity improvement",
                risk_level="low",
                estimated_improvement={"performance": "+50%", "readability": "+20%"}
            ))
        
        return proposals
    
    def _improve_readability(self, code: str) -> List[RefactoringProposal]:
        """Suggest readability improvements"""
        proposals = []
        
        # Check for complex conditionals
        complex_conditions = re.findall(r'if\s+[^:]{50,}:', code)
        if complex_conditions:
            proposals.append(RefactoringProposal(
                description="Extract complex conditionals",
                rationale="Complex conditionals reduce readability",
                original_code=complex_conditions[0],
                refactored_code="Extract to descriptive boolean methods",
                impact="Improved code clarity",
                risk_level="low",
                estimated_improvement={"readability": "+35%", "maintainability": "+25%"}
            ))
        
        return proposals
    
    def _get_ai_refactoring_suggestions(self, code: str, intent: CodeIntent) -> List[RefactoringProposal]:
        """Get AI-powered refactoring suggestions"""
        prompt = f"""Suggest refactoring for this code while maintaining its functionality:

```python
{code}
```

Intent: {intent.purpose}
Current patterns: {', '.join(intent.patterns_used)}
Quality score: {intent.quality_score}

Provide specific, actionable refactoring suggestions with:
1. What to change
2. Why it's better
3. Example of the refactored code
"""
        
        try:
            response = self.ai_client.generate_response(prompt)
            # Parse AI suggestions into proposals
            # This is simplified - in production, parse structured response
            if "refactor" in response.lower():
                return [RefactoringProposal(
                    description="AI-suggested refactoring",
                    rationale="Based on code analysis and best practices",
                    original_code=code[:100] + "...",
                    refactored_code="See AI suggestions",
                    impact="Multiple improvements",
                    risk_level="medium",
                    estimated_improvement={"overall": "+25%"}
                )]
        except Exception as e:
            logger.info(f"AI refactoring error: {str(e)}")
        
        return []
    
    def _apply_cross_language_patterns(self, code: str, intent: CodeIntent) -> List[RefactoringProposal]:
        """Apply patterns from other languages/paradigms"""
        proposals = []
        
        # Example: Apply functional patterns to imperative code
        if "for" in code and "append" in code:
            proposals.append(RefactoringProposal(
                description="Apply functional programming pattern",
                rationale="Functional approach is more concise and often more performant",
                original_code="result = []\nfor x in data:\n    result.append(transform(x))",
                refactored_code="result = list(map(transform, data))",
                impact="More concise and functional style",
                risk_level="low",
                estimated_improvement={"conciseness": "+40%", "functional_style": "+100%"}
            ))
        
        return proposals
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze an entire file"""
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            
            intent = self.analyze_code_intent(code, {"file_path": file_path})
            refactorings = self.suggest_refactoring(code, intent)
            
            return {
                "file_path": file_path,
                "intent": intent.__dict__,
                "refactoring_proposals": [r.__dict__ for r in refactorings],
                "summary": {
                    "quality_score": intent.quality_score,
                    "patterns_detected": len(intent.patterns_used),
                    "improvements_suggested": len(refactorings),
                    "high_priority_fixes": sum(1 for r in refactorings if r.risk_level == "low")
                }
            }
            
        except Exception as e:
            logger.info(f"Error analyzing file {file_path}: {str(e)}")
            return {"error": str(e)}
    
    def compare_implementations(self, code1: str, code2: str) -> Dict[str, Any]:
        """Compare two implementations of the same functionality"""
        intent1 = self.analyze_code_intent(code1)
        intent2 = self.analyze_code_intent(code2)
        
        comparison = {
            "quality_difference": intent2.quality_score - intent1.quality_score,
            "pattern_comparison": {
                "code1_patterns": intent1.patterns_used,
                "code2_patterns": intent2.patterns_used,
                "unique_to_code1": list(set(intent1.patterns_used) - set(intent2.patterns_used)),
                "unique_to_code2": list(set(intent2.patterns_used) - set(intent1.patterns_used))
            },
            "recommendation": "Code 2 is better" if intent2.quality_score > intent1.quality_score else "Code 1 is better",
            "detailed_comparison": {
                "readability": self._compare_readability(code1, code2),
                "performance": self._compare_performance(code1, code2),
                "maintainability": self._compare_maintainability(intent1, intent2)
            }
        }
        
        return comparison
    
    def _compare_readability(self, code1: str, code2: str) -> Dict[str, Any]:
        """Compare readability of two code snippets"""
        return {
            "lines_of_code": len(code2.split('\n')) - len(code1.split('\n')),
            "average_line_length": sum(len(line) for line in code2.split('\n')) / len(code2.split('\n')) - 
                                  sum(len(line) for line in code1.split('\n')) / len(code1.split('\n')),
            "comment_ratio": code2.count('#') / len(code2.split('\n')) - code1.count('#') / len(code1.split('\n'))
        }
    
    def _compare_performance(self, code1: str, code2: str) -> Dict[str, Any]:
        """Compare performance characteristics"""
        # Simplified comparison - in production, use actual profiling
        return {
            "loop_count_diff": code2.count('for') - code1.count('for'),
            "function_calls_diff": code2.count('(') - code1.count('('),
            "estimated_complexity_diff": self._estimate_complexity_diff(code1, code2)
        }
    
    def _compare_maintainability(self, intent1: CodeIntent, intent2: CodeIntent) -> Dict[str, Any]:
        """Compare maintainability aspects"""
        return {
            "improvement_areas_diff": len(intent2.improvement_areas) - len(intent1.improvement_areas),
            "side_effects_diff": len(intent2.side_effects) - len(intent1.side_effects),
            "pattern_sophistication": len(intent2.patterns_used) - len(intent1.patterns_used)
        }
    
    def _estimate_complexity_diff(self, code1: str, code2: str) -> int:
        """Estimate complexity difference between two code snippets"""
        try:
            tree1 = ast.parse(code1)
            tree2 = ast.parse(code2)
            return self._calculate_complexity(tree2) - self._calculate_complexity(tree1)
        except:
            return 0


# Example usage and testing
if __name__ == "__main__":
    analyzer = DeepCodeAnalyzer()
    
    # Test code sample
    test_code = '''
def process_data(data_list):
    result = []
    for item in data_list:
        if item > 10:
            result.append(item * 2)
        elif item > 5:
            result.append(item + 5)
        else:
            result.append(item)
    return result
'''
    
    # Analyze intent
    intent = analyzer.analyze_code_intent(test_code)
    print(f"Code Intent: {intent.purpose}")
    print(f"Quality Score: {intent.quality_score}")
    print(f"Patterns: {intent.patterns_used}")
    print(f"Improvements: {intent.improvement_areas}")
    
    # Get refactoring suggestions
    refactorings = analyzer.suggest_refactoring(test_code, intent)
    for i, proposal in enumerate(refactorings):
        print(f"\nRefactoring {i+1}: {proposal.description}")
        print(f"Rationale: {proposal.rationale}")
        print(f"Impact: {proposal.impact}")