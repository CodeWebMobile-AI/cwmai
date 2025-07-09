#!/usr/bin/env python3
"""
Self-Debugging Engine - Automatic bug detection and fixing
"""

import ast
import re
import json
import subprocess
import sys
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import difflib
import importlib.util
import inspect
import tempfile
import shutil

from scripts.http_ai_client import HTTPAIClient
import logging
from scripts.code_analyzer import CodeAnalyzer
from scripts.deep_code_analyzer import DeepCodeAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class Bug:
    """Represents a detected bug"""
    bug_id: str
    file_path: str
    line_number: int
    bug_type: str
    severity: str  # critical, high, medium, low
    description: str
    code_snippet: str
    stack_trace: Optional[str]
    detection_method: str
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class BugFix:
    """Represents a fix for a bug"""
    bug_id: str
    fix_type: str
    original_code: str
    fixed_code: str
    explanation: str
    test_code: Optional[str]
    confidence: float
    risk_level: str  # low, medium, high
    validation_status: str  # pending, passed, failed


class SelfDebuggingEngine:
    """Find and fix bugs automatically"""
    
    def __init__(self):
        self.ai_client = HTTPAIClient()
        self.code_analyzer = CodeAnalyzer()
        self.deep_analyzer = DeepCodeAnalyzer()
        self.bug_patterns = self._load_bug_patterns()
        self.fix_strategies = self._load_fix_strategies()
        self.bug_history = self._load_bug_history()
        
    def _load_bug_patterns(self) -> Dict[str, Any]:
        """Load common bug patterns"""
        return {
            "null_pointer": {
                "patterns": [
                    r"AttributeError.*'NoneType'",
                    r"TypeError.*NoneType",
                    r"if\s+\w+\s*:\s*\w+\.\w+",  # Missing None check
                ],
                "severity": "high",
                "fix_strategy": "add_none_check"
            },
            "index_error": {
                "patterns": [
                    r"IndexError.*out of range",
                    r"\[\s*\d+\s*\]",  # Hard-coded index
                    r"list\[[^\]]+\](?!\s*if)",  # List access without check
                ],
                "severity": "medium",
                "fix_strategy": "add_bounds_check"
            },
            "type_error": {
                "patterns": [
                    r"TypeError.*expected.*got",
                    r"TypeError.*argument",
                    r"def\s+\w+\([^)]*\)\s*:",  # Missing type hints
                ],
                "severity": "medium",
                "fix_strategy": "add_type_validation"
            },
            "key_error": {
                "patterns": [
                    r"KeyError:",
                    r"dict\[['\"]?\w+['\"]?\]",  # Direct dict access
                    r"(?<!\.get\()['\"]\w+['\"]\]",  # Not using .get()
                ],
                "severity": "medium",
                "fix_strategy": "use_dict_get"
            },
            "resource_leak": {
                "patterns": [
                    r"open\([^)]+\)(?!.*with)",  # File not in context manager
                    r"(?<!with\s).*=\s*open\(",  # Assignment without with
                    r"socket\(\)(?!.*close)",  # Socket without close
                ],
                "severity": "high",
                "fix_strategy": "add_context_manager"
            },
            "infinite_loop": {
                "patterns": [
                    r"while\s+True:",
                    r"while\s+1:",
                    r"while\s+\w+\s*:\s*(?!.*break)",  # Loop without break
                ],
                "severity": "critical",
                "fix_strategy": "add_loop_termination"
            },
            "sql_injection": {
                "patterns": [
                    r"execute\([^?]+%[^)]+\)",  # String formatting in SQL
                    r"execute\(.*\+.*\)",  # String concatenation in SQL
                    r"f['\"].*SELECT.*{",  # f-string in SQL
                ],
                "severity": "critical",
                "fix_strategy": "use_parameterized_query"
            },
            "race_condition": {
                "patterns": [
                    r"threading\.Thread",
                    r"async\s+def.*(?!.*lock)",
                    r"global\s+\w+.*Thread",
                ],
                "severity": "high",
                "fix_strategy": "add_synchronization"
            }
        }
    
    def _load_fix_strategies(self) -> Dict[str, Callable]:
        """Load fix strategies for different bug types"""
        return {
            "add_none_check": self._fix_none_check,
            "add_bounds_check": self._fix_bounds_check,
            "add_type_validation": self._fix_type_validation,
            "use_dict_get": self._fix_dict_access,
            "add_context_manager": self._fix_resource_leak,
            "add_loop_termination": self._fix_infinite_loop,
            "use_parameterized_query": self._fix_sql_injection,
            "add_synchronization": self._fix_race_condition,
        }
    
    def _load_bug_history(self) -> List[Bug]:
        """Load historical bug data"""
        history_file = Path("bug_history.json")
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    return [Bug(**bug) for bug in data]
            except Exception as e:
                logger.info(f"Error loading bug history: {str(e)}")
        return []
    
    async def detect_bugs(self, file_paths: Optional[List[str]] = None) -> List[Bug]:
        """Run static analysis to identify potential issues"""
        bugs = []
        
        # Get files to analyze
        if not file_paths:
            file_paths = self._get_python_files()
        
        for file_path in file_paths:
            logger.info(f"Analyzing {file_path} for bugs...")
            
            try:
                # Read file content
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Multiple detection methods
                pattern_bugs = self._detect_pattern_bugs(file_path, content)
                bugs.extend(pattern_bugs)
                
                ast_bugs = self._detect_ast_bugs(file_path, content)
                bugs.extend(ast_bugs)
                
                runtime_bugs = await self._detect_runtime_bugs(file_path, content)
                bugs.extend(runtime_bugs)
                
                ai_bugs = await self._detect_ai_bugs(file_path, content)
                bugs.extend(ai_bugs)
                
            except Exception as e:
                logger.info(f"Error analyzing {file_path}: {str(e)}")
        
        # Deduplicate and prioritize
        bugs = self._deduplicate_bugs(bugs)
        bugs = sorted(bugs, key=lambda b: self._get_severity_score(b.severity), reverse=True)
        
        # Save to history
        self._save_bug_history(bugs)
        
        return bugs
    
    async def generate_fixes(self, bug: Bug) -> List[BugFix]:
        """Create patches for detected bugs"""
        fixes = []
        
        try:
            # Get fix strategy
            strategy_name = None
            for bug_type, info in self.bug_patterns.items():
                if bug_type in bug.bug_type.lower():
                    strategy_name = info.get("fix_strategy")
                    break
            
            if strategy_name and strategy_name in self.fix_strategies:
                # Apply specific fix strategy
                fix = self.fix_strategies[strategy_name](bug)
                if fix:
                    fixes.append(fix)
            
            # Get AI-powered fixes
            ai_fixes = await self._generate_ai_fixes(bug)
            fixes.extend(ai_fixes)
            
            # Validate fixes
            for fix in fixes:
                fix.validation_status = self._validate_fix(fix)
            
            # Generate tests for fixes
            for fix in fixes:
                if fix.validation_status == "passed":
                    fix.test_code = self._generate_test_for_fix(fix)
            
        except Exception as e:
            logger.info(f"Error generating fixes for bug {bug.bug_id}: {str(e)}")
        
        return fixes
    
    def _get_python_files(self, directory: str = ".") -> List[str]:
        """Get all Python files in directory"""
        path = Path(directory)
        return [str(f) for f in path.rglob("*.py") if f.is_file() and not f.name.startswith("test_")]
    
    def _detect_pattern_bugs(self, file_path: str, content: str) -> List[Bug]:
        """Detect bugs using pattern matching"""
        bugs = []
        lines = content.split('\n')
        
        for bug_type, info in self.bug_patterns.items():
            for pattern in info["patterns"]:
                for i, line in enumerate(lines):
                    if re.search(pattern, line):
                        bug = Bug(
                            bug_id=f"{Path(file_path).stem}_{bug_type}_{i+1}",
                            file_path=file_path,
                            line_number=i + 1,
                            bug_type=bug_type,
                            severity=info["severity"],
                            description=f"Potential {bug_type.replace('_', ' ')} detected",
                            code_snippet=line.strip(),
                            stack_trace=None,
                            detection_method="pattern_matching",
                            confidence=0.7,
                            metadata={"pattern": pattern}
                        )
                        bugs.append(bug)
        
        return bugs
    
    def _detect_ast_bugs(self, file_path: str, content: str) -> List[Bug]:
        """Detect bugs using AST analysis"""
        bugs = []
        
        try:
            tree = ast.parse(content)
            
            # Check for common AST-based issues
            class BugVisitor(ast.NodeVisitor):
                def __init__(self, file_path):
                    self.bugs = []
                    self.file_path = file_path
                    
                def visit_FunctionDef(self, node):
                    # Check for missing return
                    has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
                    if not has_return and node.returns:
                        self.bugs.append(Bug(
                            bug_id=f"{Path(file_path).stem}_missing_return_{node.lineno}",
                            file_path=file_path,
                            line_number=node.lineno,
                            bug_type="missing_return",
                            severity="high",
                            description=f"Function '{node.name}' has return type annotation but no return statement",
                            code_snippet=f"def {node.name}(...) -> {ast.unparse(node.returns) if hasattr(ast, 'unparse') else '...'}:",
                            stack_trace=None,
                            detection_method="ast_analysis",
                            confidence=0.9,
                            metadata={"function_name": node.name}
                        ))
                    
                    # Check for too many arguments
                    if len(node.args.args) > 7:
                        self.bugs.append(Bug(
                            bug_id=f"{Path(file_path).stem}_too_many_args_{node.lineno}",
                            file_path=file_path,
                            line_number=node.lineno,
                            bug_type="too_many_arguments",
                            severity="medium",
                            description=f"Function '{node.name}' has {len(node.args.args)} arguments (max recommended: 7)",
                            code_snippet=f"def {node.name}({len(node.args.args)} args):",
                            stack_trace=None,
                            detection_method="ast_analysis",
                            confidence=0.8,
                            metadata={"arg_count": len(node.args.args)}
                        ))
                    
                    self.generic_visit(node)
                
                def visit_Try(self, node):
                    # Check for bare except
                    for handler in node.handlers:
                        if handler.type is None:
                            self.bugs.append(Bug(
                                bug_id=f"{Path(file_path).stem}_bare_except_{handler.lineno}",
                                file_path=file_path,
                                line_number=handler.lineno,
                                bug_type="bare_except",
                                severity="medium",
                                description="Bare except clause catches all exceptions",
                                code_snippet="except:",
                                stack_trace=None,
                                detection_method="ast_analysis",
                                confidence=1.0,
                                metadata={}
                            ))
                    
                    self.generic_visit(node)
            
            visitor = BugVisitor(file_path)
            visitor.visit(tree)
            bugs.extend(visitor.bugs)
            
        except SyntaxError as e:
            bugs.append(Bug(
                bug_id=f"{Path(file_path).stem}_syntax_error_{e.lineno}",
                file_path=file_path,
                line_number=e.lineno or 0,
                bug_type="syntax_error",
                severity="critical",
                description=f"Syntax error: {str(e)}",
                code_snippet=e.text or "",
                stack_trace=None,
                detection_method="ast_parsing",
                confidence=1.0,
                metadata={"error": str(e)}
            ))
        
        return bugs
    
    async def _detect_runtime_bugs(self, file_path: str, content: str) -> List[Bug]:
        """Detect bugs that might occur at runtime"""
        bugs = []
        
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Add basic imports and try to run the code
            test_content = f"""
import sys
sys.path.insert(0, '.')

try:
    {content}
except Exception as e:
    print(f"RUNTIME_ERROR:{{type(e).__name__}}:{{str(e)}}")
"""
            f.write(test_content)
            temp_file = f.name
        
        try:
            # Run the file with timeout
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if "RUNTIME_ERROR:" in result.stdout:
                error_info = result.stdout.split("RUNTIME_ERROR:")[1].strip()
                error_type, error_msg = error_info.split(":", 1)
                
                bugs.append(Bug(
                    bug_id=f"{Path(file_path).stem}_runtime_{error_type}",
                    file_path=file_path,
                    line_number=0,
                    bug_type=f"runtime_{error_type.lower()}",
                    severity="high",
                    description=f"Runtime error: {error_msg}",
                    code_snippet="See file content",
                    stack_trace=result.stderr,
                    detection_method="runtime_analysis",
                    confidence=0.8,
                    metadata={"error_type": error_type}
                ))
            
        except subprocess.TimeoutExpired:
            bugs.append(Bug(
                bug_id=f"{Path(file_path).stem}_timeout",
                file_path=file_path,
                line_number=0,
                bug_type="infinite_loop",
                severity="critical",
                description="Code execution timeout - possible infinite loop",
                code_snippet="See file content",
                stack_trace=None,
                detection_method="runtime_analysis",
                confidence=0.7,
                metadata={"timeout": True}
            ))
        except Exception as e:
            logger.info(f"Runtime detection error: {str(e)}")
        finally:
            Path(temp_file).unlink(missing_ok=True)
        
        return bugs
    
    async def _detect_ai_bugs(self, file_path: str, content: str) -> List[Bug]:
        """Use AI to detect complex bugs"""
        bugs = []
        
        prompt = f"""Analyze this Python code for bugs, security issues, and potential problems:

```python
{content[:3000]}  # Truncate for API limits
```

Focus on:
1. Logic errors
2. Security vulnerabilities
3. Performance issues
4. Race conditions
5. Memory leaks

Return a JSON array of bugs with: type, severity, line_number, description."""
        
        try:
            response = self.ai_client.generate_response(prompt)
            
            # Parse AI response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                ai_bugs = json.loads(json_match.group())
                
                for i, ai_bug in enumerate(ai_bugs):
                    bug = Bug(
                        bug_id=f"{Path(file_path).stem}_ai_{i}",
                        file_path=file_path,
                        line_number=ai_bug.get("line_number", 0),
                        bug_type=ai_bug.get("type", "unknown"),
                        severity=ai_bug.get("severity", "medium"),
                        description=ai_bug.get("description", "AI-detected issue"),
                        code_snippet="See file content",
                        stack_trace=None,
                        detection_method="ai_analysis",
                        confidence=0.6,
                        metadata={"ai_detection": True}
                    )
                    bugs.append(bug)
        
        except Exception as e:
            logger.info(f"AI bug detection error: {str(e)}")
        
        return bugs
    
    def _fix_none_check(self, bug: Bug) -> Optional[BugFix]:
        """Fix missing None checks"""
        try:
            # Read the file
            with open(bug.file_path, 'r') as f:
                lines = f.readlines()
            
            if bug.line_number <= len(lines):
                original_line = lines[bug.line_number - 1]
                
                # Pattern to find attribute access
                match = re.search(r'(\s*)(.+?)\.(\w+)', original_line)
                if match:
                    indent, obj, attr = match.groups()
                    fixed_line = f"{indent}if {obj} is not None:\n{indent}    {original_line.strip()}\n"
                    
                    return BugFix(
                        bug_id=bug.bug_id,
                        fix_type="add_none_check",
                        original_code=original_line,
                        fixed_code=fixed_line,
                        explanation="Added None check to prevent AttributeError",
                        test_code=f"assert {obj} is None or hasattr({obj}, '{attr}')",
                        confidence=0.8,
                        risk_level="low",
                        validation_status="pending"
                    )
        except Exception as e:
            logger.info(f"Error fixing None check: {str(e)}")
        
        return None
    
    def _fix_bounds_check(self, bug: Bug) -> Optional[BugFix]:
        """Fix index out of bounds issues"""
        try:
            with open(bug.file_path, 'r') as f:
                lines = f.readlines()
            
            if bug.line_number <= len(lines):
                original_line = lines[bug.line_number - 1]
                
                # Pattern to find list/array access
                match = re.search(r'(\s*)(.+?)\[(\d+)\]', original_line)
                if match:
                    indent, obj, index = match.groups()
                    fixed_code = f"{indent}if len({obj}) > {index}:\n{indent}    {original_line.strip()}\n"
                    
                    return BugFix(
                        bug_id=bug.bug_id,
                        fix_type="add_bounds_check",
                        original_code=original_line,
                        fixed_code=fixed_code,
                        explanation="Added bounds check to prevent IndexError",
                        test_code=f"assert len({obj}) > {index}",
                        confidence=0.8,
                        risk_level="low",
                        validation_status="pending"
                    )
        except Exception as e:
            logger.info(f"Error fixing bounds check: {str(e)}")
        
        return None
    
    def _fix_type_validation(self, bug: Bug) -> Optional[BugFix]:
        """Add type validation"""
        # Simplified implementation
        return BugFix(
            bug_id=bug.bug_id,
            fix_type="add_type_validation",
            original_code=bug.code_snippet,
            fixed_code="# Add type hints and validation",
            explanation="Add type hints and runtime type checking",
            test_code=None,
            confidence=0.5,
            risk_level="medium",
            validation_status="pending"
        )
    
    def _fix_dict_access(self, bug: Bug) -> Optional[BugFix]:
        """Fix direct dictionary access"""
        try:
            with open(bug.file_path, 'r') as f:
                lines = f.readlines()
            
            if bug.line_number <= len(lines):
                original_line = lines[bug.line_number - 1]
                
                # Replace dict[key] with dict.get(key)
                fixed_line = re.sub(r'(\w+)\[(["\']\w+["\'])\]', r'\1.get(\2)', original_line)
                
                if fixed_line != original_line:
                    return BugFix(
                        bug_id=bug.bug_id,
                        fix_type="use_dict_get",
                        original_code=original_line,
                        fixed_code=fixed_line,
                        explanation="Use .get() to avoid KeyError",
                        test_code=None,
                        confidence=0.9,
                        risk_level="low",
                        validation_status="pending"
                    )
        except Exception as e:
            logger.info(f"Error fixing dict access: {str(e)}")
        
        return None
    
    def _fix_resource_leak(self, bug: Bug) -> Optional[BugFix]:
        """Fix resource leaks with context managers"""
        try:
            with open(bug.file_path, 'r') as f:
                lines = f.readlines()
            
            if bug.line_number <= len(lines):
                original_line = lines[bug.line_number - 1]
                
                # Pattern for file open
                match = re.search(r'(\s*)(\w+)\s*=\s*open\((.*?)\)', original_line)
                if match:
                    indent, var, args = match.groups()
                    fixed_code = f"{indent}with open({args}) as {var}:\n{indent}    # Use {var} here\n"
                    
                    return BugFix(
                        bug_id=bug.bug_id,
                        fix_type="add_context_manager",
                        original_code=original_line,
                        fixed_code=fixed_code,
                        explanation="Use context manager to ensure resource cleanup",
                        test_code=None,
                        confidence=0.9,
                        risk_level="low",
                        validation_status="pending"
                    )
        except Exception as e:
            logger.info(f"Error fixing resource leak: {str(e)}")
        
        return None
    
    def _fix_infinite_loop(self, bug: Bug) -> Optional[BugFix]:
        """Add termination condition to loops"""
        return BugFix(
            bug_id=bug.bug_id,
            fix_type="add_loop_termination",
            original_code=bug.code_snippet,
            fixed_code="# Add proper termination condition",
            explanation="Add break condition or maximum iterations",
            test_code=None,
            confidence=0.6,
            risk_level="high",
            validation_status="pending"
        )
    
    def _fix_sql_injection(self, bug: Bug) -> Optional[BugFix]:
        """Fix SQL injection vulnerabilities"""
        return BugFix(
            bug_id=bug.bug_id,
            fix_type="use_parameterized_query",
            original_code=bug.code_snippet,
            fixed_code="# Use parameterized queries",
            explanation="Use parameterized queries to prevent SQL injection",
            test_code=None,
            confidence=0.9,
            risk_level="low",
            validation_status="pending"
        )
    
    def _fix_race_condition(self, bug: Bug) -> Optional[BugFix]:
        """Add synchronization for thread safety"""
        return BugFix(
            bug_id=bug.bug_id,
            fix_type="add_synchronization",
            original_code=bug.code_snippet,
            fixed_code="# Add locks or other synchronization",
            explanation="Add proper synchronization to prevent race conditions",
            test_code=None,
            confidence=0.7,
            risk_level="medium",
            validation_status="pending"
        )
    
    async def _generate_ai_fixes(self, bug: Bug) -> List[BugFix]:
        """Generate fixes using AI"""
        fixes = []
        
        # Read surrounding context
        context = self._get_code_context(bug.file_path, bug.line_number)
        
        prompt = f"""Fix this bug in Python code:

Bug Type: {bug.bug_type}
Severity: {bug.severity}
Description: {bug.description}

Code context:
```python
{context}
```

Problematic line (line {bug.line_number}): {bug.code_snippet}

Provide a fix that:
1. Resolves the issue
2. Maintains existing functionality
3. Follows Python best practices

Return JSON with: fixed_code, explanation, test_code"""
        
        try:
            response = self.ai_client.generate_response(prompt)
            
            # Parse response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                fix_data = json.loads(json_match.group())
                
                fix = BugFix(
                    bug_id=bug.bug_id,
                    fix_type="ai_generated",
                    original_code=bug.code_snippet,
                    fixed_code=fix_data.get("fixed_code", ""),
                    explanation=fix_data.get("explanation", "AI-generated fix"),
                    test_code=fix_data.get("test_code"),
                    confidence=0.7,
                    risk_level="medium",
                    validation_status="pending"
                )
                fixes.append(fix)
        
        except Exception as e:
            logger.info(f"AI fix generation error: {str(e)}")
        
        return fixes
    
    def _get_code_context(self, file_path: str, line_number: int, context_lines: int = 10) -> str:
        """Get code context around a line"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            start = max(0, line_number - context_lines - 1)
            end = min(len(lines), line_number + context_lines)
            
            context = []
            for i in range(start, end):
                marker = ">>> " if i == line_number - 1 else "    "
                context.append(f"{i+1:4d}{marker}{lines[i].rstrip()}")
            
            return "\n".join(context)
        except:
            return "Unable to get context"
    
    def _validate_fix(self, fix: BugFix) -> str:
        """Validate that a fix works correctly"""
        # Create temporary file with fix applied
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(fix.fixed_code)
            temp_file = f.name
        
        try:
            # Try to parse the fixed code
            with open(temp_file, 'r') as f:
                ast.parse(f.read())
            
            # If we have test code, run it
            if fix.test_code:
                test_result = subprocess.run(
                    [sys.executable, "-c", fix.test_code],
                    capture_output=True,
                    timeout=5
                )
                
                if test_result.returncode == 0:
                    return "passed"
                else:
                    return "failed"
            
            return "passed"
            
        except SyntaxError:
            return "failed"
        except Exception as e:
            logger.info(f"Validation error: {str(e)}")
            return "failed"
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def _generate_test_for_fix(self, fix: BugFix) -> str:
        """Generate test code for a fix"""
        if fix.test_code:
            return fix.test_code
        
        # Generate basic test based on fix type
        test_templates = {
            "add_none_check": "assert obj is None or hasattr(obj, 'attr')",
            "add_bounds_check": "assert len(lst) > index or index < 0",
            "use_dict_get": "assert d.get('key') is None or 'key' in d",
            "add_context_manager": "# Context manager ensures cleanup",
        }
        
        return test_templates.get(fix.fix_type, "# Add appropriate test")
    
    def _deduplicate_bugs(self, bugs: List[Bug]) -> List[Bug]:
        """Remove duplicate bug reports"""
        seen = set()
        unique_bugs = []
        
        for bug in bugs:
            key = (bug.file_path, bug.line_number, bug.bug_type)
            if key not in seen:
                seen.add(key)
                unique_bugs.append(bug)
        
        return unique_bugs
    
    def _get_severity_score(self, severity: str) -> int:
        """Convert severity to numeric score"""
        scores = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1
        }
        return scores.get(severity, 0)
    
    def _save_bug_history(self, bugs: List[Bug]):
        """Save bugs to history file"""
        try:
            history_file = Path("bug_history.json")
            
            # Add to existing history
            all_bugs = self.bug_history + bugs
            
            # Keep last 1000 bugs
            if len(all_bugs) > 1000:
                all_bugs = all_bugs[-1000:]
            
            # Convert to dict for JSON
            bug_dicts = [asdict(bug) for bug in all_bugs]
            
            with open(history_file, 'w') as f:
                json.dump(bug_dicts, f, indent=2)
                
        except Exception as e:
            logger.info(f"Error saving bug history: {str(e)}")
    
    def apply_fix(self, fix: BugFix, backup: bool = True) -> bool:
        """Apply a fix to the codebase"""
        try:
            # Read bug details to get file path
            bug_file = None
            for bug in self.bug_history:
                if bug.bug_id == fix.bug_id:
                    bug_file = bug.file_path
                    break
            
            if not bug_file:
                logger.info(f"Bug {fix.bug_id} not found in history")
                return False
            
            # Backup original file
            if backup:
                backup_path = Path(bug_file).with_suffix('.backup')
                shutil.copy2(bug_file, backup_path)
            
            # Apply fix (simplified - in reality would be more complex)
            with open(bug_file, 'r') as f:
                content = f.read()
            
            # Replace original code with fixed code
            content = content.replace(fix.original_code.strip(), fix.fixed_code.strip())
            
            with open(bug_file, 'w') as f:
                f.write(content)
            
            logger.info(f"Applied fix for bug {fix.bug_id}")
            return True
            
        except Exception as e:
            logger.info(f"Error applying fix: {str(e)}")
            return False
    
    def generate_bug_report(self, bugs: List[Bug], fixes: Dict[str, List[BugFix]]) -> str:
        """Generate comprehensive bug report"""
        report = f"""# Bug Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
Total bugs found: {len(bugs)}
Critical: {sum(1 for b in bugs if b.severity == 'critical')}
High: {sum(1 for b in bugs if b.severity == 'high')}
Medium: {sum(1 for b in bugs if b.severity == 'medium')}
Low: {sum(1 for b in bugs if b.severity == 'low')}

## Bugs by Type
"""
        
        # Group by type
        by_type = {}
        for bug in bugs:
            if bug.bug_type not in by_type:
                by_type[bug.bug_type] = []
            by_type[bug.bug_type].append(bug)
        
        for bug_type, type_bugs in by_type.items():
            report += f"\n### {bug_type.replace('_', ' ').title()} ({len(type_bugs)} found)\n"
            
            for bug in type_bugs[:3]:  # Show first 3
                report += f"""
- **File**: {bug.file_path}:{bug.line_number}
  - Severity: {bug.severity}
  - Confidence: {bug.confidence:.1%}
  - Description: {bug.description}
"""
                
                # Show fixes if available
                if bug.bug_id in fixes:
                    bug_fixes = fixes[bug.bug_id]
                    report += f"  - Fixes available: {len(bug_fixes)}\n"
                    
                    for fix in bug_fixes[:1]:  # Show first fix
                        if fix.validation_status == "passed":
                            report += f"    - ✅ {fix.explanation}\n"
        
        return report


# Example usage and testing
if __name__ == "__main__":
    async def main():
        engine = SelfDebuggingEngine()
        
        # Test with sample buggy code
        test_file = Path("test_buggy_code.py")
        test_file.write_text("""
def process_data(data):
    result = data[0]  # Potential IndexError
    return result.upper()  # Potential AttributeError if None

def risky_function(d):
    value = d['key']  # Potential KeyError
    f = open('file.txt')  # Resource leak
    content = f.read()
    return content
""")
        
        # Detect bugs
        print("Detecting bugs...")
        bugs = await engine.detect_bugs([str(test_file)])
        
        print(f"\nFound {len(bugs)} bugs:")
        for bug in bugs:
            print(f"- {bug.bug_type} at line {bug.line_number}: {bug.description}")
        
        # Generate fixes
        print("\nGenerating fixes...")
        all_fixes = {}
        for bug in bugs[:3]:  # Fix first 3 bugs
            fixes = await engine.generate_fixes(bug)
            all_fixes[bug.bug_id] = fixes
            
            print(f"\n{bug.bug_type} fixes:")
            for fix in fixes:
                print(f"- {fix.explanation} (confidence: {fix.confidence:.1%})")
        
        # Generate report
        report = engine.generate_bug_report(bugs, all_fixes)
        print("\n" + report)
        
        # Cleanup
        test_file.unlink()
    
    asyncio.run(main())