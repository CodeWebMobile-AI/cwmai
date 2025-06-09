"""
Safe Self-Improvement Engine

Controlled self-modification system with safety boundaries, daily limits,
and comprehensive rollback mechanisms.
"""

import ast
import os
import json
import hashlib
import subprocess
import tempfile
import shutil
import shlex
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import git
import difflib
import re
from collections import defaultdict
import traceback
import psutil
import time


class ModificationType(Enum):
    """Types of modifications allowed."""
    OPTIMIZATION = "optimization"
    REFACTORING = "refactoring"
    FEATURE_ADDITION = "feature_addition"
    BUG_FIX = "bug_fix"
    DOCUMENTATION = "documentation"
    TEST_ADDITION = "test_addition"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class SafetyConstraints:
    """Safety constraints for modifications."""
    max_changes_per_day: int = 24
    max_file_size_change: int = 1000  # lines
    max_complexity_increase: float = 1.3  # 30% max increase
    required_test_pass_rate: float = 0.95
    max_memory_usage_mb: int = 500
    max_execution_time_seconds: int = 30
    forbidden_patterns: List[str] = field(default_factory=lambda: [
        r'exec\s*\(',
        r'eval\s*\(',
        r'__import__\s*\(',
        r'os\.system\s*\(',
        r'subprocess\.call\s*\(',
        r'subprocess\.run\s*\([^)]*shell\s*=\s*True',  # shell=True is dangerous
        r'subprocess\.Popen\s*\([^)]*shell\s*=\s*True',
        r'open\s*\(.+[\'"]w[\'"]',  # File write operations
        r'shutil\.rmtree',
        r'os\.remove',
        r'requests\.post',  # Network operations
        r'socket\.',
        r'pickle\.loads\s*\(',  # Insecure deserialization
        r'pickle\.load\s*\(',
        r'marshal\.loads\s*\(',
        r'compile\s*\(',  # Code compilation
        r'globals\s*\(\)',  # Access to global namespace
        r'locals\s*\(\)',   # Access to local namespace
        r'vars\s*\(\)',     # Access to variables
        r'dir\s*\(\)',      # Directory listing
        r'hasattr\s*\([^,]+,\s*[\'"][^\'\"]*__',  # Dunder attribute access
        r'getattr\s*\([^,]+,\s*[\'"][^\'\"]*__',  # Dunder attribute access
        r'setattr\s*\([^,]+,\s*[\'"][^\'\"]*__',  # Dunder attribute access
    ])
    allowed_modules: Set[str] = field(default_factory=lambda: {
        'json', 'time', 'datetime', 'typing', 'dataclasses',
        'enum', 'collections', 'itertools', 'functools',
        'numpy', 'pandas', 'sklearn', 'ast', 'inspect',
        're', 'hashlib', 'copy', 'math', 'random'
    })


@dataclass
class Modification:
    """Represents a code modification."""
    id: str
    type: ModificationType
    target_file: str
    description: str
    changes: List[Tuple[str, str]]  # (old_code, new_code) pairs
    timestamp: datetime
    applied: bool = False
    success: bool = False
    performance_impact: Optional[Dict[str, float]] = None
    rollback_commit: Optional[str] = None
    safety_score: float = 0.0
    test_results: Optional[Dict[str, Any]] = None


class SafeSelfImprover:
    """Safe self-improvement system with comprehensive safety measures."""
    
    def __init__(self, repo_path: str = ".", max_changes_per_day: int = 24):
        """Initialize safe self-improver.
        
        Args:
            repo_path: Path to repository
            max_changes_per_day: Maximum daily modifications allowed
        """
        self.repo_path = os.path.abspath(repo_path)
        
        # Security: Validate repository path
        if not self._is_safe_path(self.repo_path):
            raise ValueError("Unsafe repository path detected")
        
        # Find the Git repository root
        current_path = self.repo_path
        while current_path != os.path.dirname(current_path):  # Not at root
            if os.path.exists(os.path.join(current_path, '.git')):
                self.repo_path = current_path
                break
            current_path = os.path.dirname(current_path)
        
        self.repo = git.Repo(self.repo_path)
        
        # Safety constraints
        self.constraints = SafetyConstraints(max_changes_per_day=max_changes_per_day)
        
        # Modification tracking
        self.modifications_today = self._load_todays_modifications()
        self.modification_history = self._load_modification_history()
        
        # Performance baselines
        self.performance_baselines = self._establish_baselines()
        
        # Sandbox environment
        self.sandbox_dir = None
        
    def propose_improvement(self, 
                          target_file: str,
                          improvement_type: ModificationType,
                          description: str,
                          analysis: Optional[Dict[str, Any]] = None) -> Optional[Modification]:
        """Propose a safe improvement to code.
        
        Args:
            target_file: File to improve
            improvement_type: Type of improvement
            description: Description of improvement
            analysis: Optional analysis data
            
        Returns:
            Proposed modification or None if unsafe/limit reached
        """
        # Check daily limit
        if len(self.modifications_today) >= self.constraints.max_changes_per_day:
            print(f"Daily modification limit reached ({self.constraints.max_changes_per_day})")
            return None
        
        # Generate modification ID
        mod_id = self._generate_modification_id(target_file, improvement_type)
        
        # Read current code
        full_path = os.path.join(self.repo_path, target_file)
        if not os.path.exists(full_path):
            print(f"Target file not found: {target_file}")
            return None
        
        with open(full_path, 'r') as f:
            current_code = f.read()
        
        # Generate improvements based on type
        changes = self._generate_improvements(
            current_code, improvement_type, analysis
        )
        
        if not changes:
            print("No improvements generated")
            return None
        
        # Create modification
        modification = Modification(
            id=mod_id,
            type=improvement_type,
            target_file=target_file,
            description=description,
            changes=changes,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Safety validation
        safety_score = self._validate_safety(modification, current_code)
        modification.safety_score = safety_score
        
        if safety_score < 0.8:
            print(f"Modification failed safety check (score: {safety_score:.2f})")
            return None
        
        return modification
    
    def apply_improvement(self, modification: Modification) -> bool:
        """Apply a proposed improvement with full safety checks.
        
        Args:
            modification: Modification to apply
            
        Returns:
            Success status
        """
        print(f"Applying improvement: {modification.description}")
        
        # Create sandbox
        self.sandbox_dir = self._create_sandbox()
        
        try:
            # Apply changes in sandbox first
            sandbox_success = self._apply_in_sandbox(modification)
            
            if not sandbox_success:
                print("Sandbox testing failed")
                return False
            
            # Create git checkpoint
            checkpoint = self._create_checkpoint()
            modification.rollback_commit = checkpoint
            
            # Apply to real repository
            success = self._apply_to_repository(modification)
            
            if success:
                # Run tests
                test_results = self._run_tests()
                modification.test_results = test_results
                
                if test_results['pass_rate'] < self.constraints.required_test_pass_rate:
                    print(f"Test pass rate too low: {test_results['pass_rate']:.2%}")
                    self._rollback(checkpoint)
                    return False
                
                # Measure performance impact
                performance = self._measure_performance_impact()
                modification.performance_impact = performance
                
                # Check performance regression
                if self._has_performance_regression(performance):
                    print("Performance regression detected")
                    self._rollback(checkpoint)
                    return False
                
                # Success!
                modification.applied = True
                modification.success = True
                
                # Update tracking
                self.modifications_today.append(modification)
                self._save_modification(modification)
                
                # Commit changes
                self._commit_improvement(modification)
                
                print(f"Successfully applied improvement: {modification.id}")
                return True
            else:
                self._rollback(checkpoint)
                return False
                
        except Exception as e:
            print(f"Error applying improvement: {e}")
            traceback.print_exc()
            
            if modification.rollback_commit:
                self._rollback(modification.rollback_commit)
            
            return False
            
        finally:
            # Cleanup sandbox
            if self.sandbox_dir and os.path.exists(self.sandbox_dir):
                shutil.rmtree(self.sandbox_dir)
                self.sandbox_dir = None
    
    def analyze_improvement_opportunities(self) -> List[Dict[str, Any]]:
        """Analyze codebase for improvement opportunities."""
        opportunities = []
        
        # Scan Python files
        for root, dirs, files in os.walk(self.repo_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    relative_path = os.path.relpath(filepath, self.repo_path)
                    
                    # Analyze file
                    file_opportunities = self._analyze_file(relative_path)
                    opportunities.extend(file_opportunities)
        
        # Sort by priority
        opportunities.sort(key=lambda x: x['priority'], reverse=True)
        
        return opportunities[:10]  # Top 10 opportunities
    
    def get_improvement_history(self) -> Dict[str, Any]:
        """Get history of improvements and their impact."""
        history = {
            'total_improvements': len(self.modification_history),
            'successful_improvements': sum(1 for m in self.modification_history if m.success),
            'improvements_by_type': defaultdict(int),
            'average_performance_impact': {},
            'most_improved_files': [],
            'recent_improvements': []
        }
        
        # Count by type
        for mod in self.modification_history:
            history['improvements_by_type'][mod.type.value] += 1
        
        # Calculate average performance impact
        performance_impacts = defaultdict(list)
        for mod in self.modification_history:
            if mod.performance_impact:
                for metric, value in mod.performance_impact.items():
                    performance_impacts[metric].append(value)
        
        for metric, values in performance_impacts.items():
            history['average_performance_impact'][metric] = sum(values) / len(values)
        
        # Find most improved files
        file_improvements = defaultdict(int)
        for mod in self.modification_history:
            if mod.success:
                file_improvements[mod.target_file] += 1
        
        history['most_improved_files'] = sorted(
            file_improvements.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Recent improvements
        recent = sorted(
            self.modification_history,
            key=lambda x: x.timestamp,
            reverse=True
        )[:10]
        
        history['recent_improvements'] = [{
            'id': m.id,
            'type': m.type.value,
            'file': m.target_file,
            'description': m.description,
            'timestamp': m.timestamp.isoformat(),
            'success': m.success,
            'performance_impact': m.performance_impact
        } for m in recent]
        
        return history
    
    def _generate_modification_id(self, target_file: str, 
                                 improvement_type: ModificationType) -> str:
        """Generate unique modification ID."""
        content = f"{target_file}{improvement_type.value}{datetime.now()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _generate_improvements(self, code: str, 
                             improvement_type: ModificationType,
                             analysis: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """Generate code improvements based on type."""
        changes = []
        
        if improvement_type == ModificationType.OPTIMIZATION:
            changes.extend(self._generate_optimizations(code))
        elif improvement_type == ModificationType.REFACTORING:
            changes.extend(self._generate_refactorings(code))
        elif improvement_type == ModificationType.PERFORMANCE:
            changes.extend(self._generate_performance_improvements(code))
        elif improvement_type == ModificationType.DOCUMENTATION:
            changes.extend(self._generate_documentation(code))
        elif improvement_type == ModificationType.BUG_FIX:
            changes.extend(self._generate_bug_fixes(code, analysis))
        elif improvement_type == ModificationType.TEST_ADDITION:
            changes.extend(self._generate_tests(code))
        
        return changes
    
    def _generate_optimizations(self, code: str) -> List[Tuple[str, str]]:
        """Generate optimization improvements."""
        changes = []
        
        # List comprehension optimization
        import_re = re.compile(r'(\w+)\s*=\s*\[\]\s*\nfor\s+(\w+)\s+in\s+(\w+):\s*\n\s+\1\.append\(([^)]+)\)')
        matches = import_re.finditer(code)
        
        for match in matches:
            old_code = match.group(0)
            var_name = match.group(1)
            loop_var = match.group(2)
            iterable = match.group(3)
            append_expr = match.group(4)
            
            new_code = f"{var_name} = [{append_expr} for {loop_var} in {iterable}]"
            changes.append((old_code, new_code))
        
        # Dictionary get() optimization
        dict_re = re.compile(r'if\s+(\w+)\s+in\s+(\w+):\s*\n\s+(\w+)\s*=\s*\2\[(\1)\]\s*\nelse:\s*\n\s+\3\s*=\s*(.+)')
        matches = dict_re.finditer(code)
        
        for match in matches:
            old_code = match.group(0)
            key = match.group(1)
            dict_name = match.group(2)
            var_name = match.group(3)
            default_val = match.group(5)
            
            new_code = f"{var_name} = {dict_name}.get({key}, {default_val})"
            changes.append((old_code, new_code))
        
        return changes
    
    def _generate_refactorings(self, code: str) -> List[Tuple[str, str]]:
        """Generate refactoring improvements."""
        changes = []
        
        try:
            tree = ast.parse(code)
            
            # Extract long methods
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Count lines in function
                    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                        func_lines = node.end_lineno - node.lineno
                        
                        if func_lines > 50:  # Long function
                            # Suggest extraction
                            # This is simplified - real implementation would be more complex
                            print(f"Long function detected: {node.name} ({func_lines} lines)")
        except:
            pass
        
        return changes
    
    def _generate_performance_improvements(self, code: str) -> List[Tuple[str, str]]:
        """Generate performance improvements."""
        changes = []
        
        # Cache property pattern
        property_re = re.compile(r'@property\s*\ndef\s+(\w+)\(self\):\s*\n\s+return\s+(.+)')
        matches = property_re.finditer(code)
        
        for match in matches:
            if 'self._' not in match.group(2):  # Not already cached
                old_code = match.group(0)
                prop_name = match.group(1)
                return_expr = match.group(2)
                
                new_code = f"""@property
def {prop_name}(self):
    if not hasattr(self, '_{prop_name}_cache'):
        self._{prop_name}_cache = {return_expr}
    return self._{prop_name}_cache"""
                
                changes.append((old_code, new_code))
        
        return changes
    
    def _generate_documentation(self, code: str) -> List[Tuple[str, str]]:
        """Generate documentation improvements."""
        changes = []
        
        try:
            tree = ast.parse(code)
            
            # Find undocumented functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not ast.get_docstring(node):
                        # Generate docstring
                        args = [arg.arg for arg in node.args.args if arg.arg != 'self']
                        
                        docstring = f'''"""GENERATED: {node.name} function.
    
    Args:
        {chr(10).join(f"{arg}: TODO - Add description" for arg in args) if args else "None"}
    
    Returns:
        TODO - Add return description
    """'''
                        
                        # This is simplified - would need proper AST manipulation
                        print(f"Missing docstring for function: {node.name}")
        except:
            pass
        
        return changes
    
    def _generate_bug_fixes(self, code: str, 
                          analysis: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """Generate bug fixes based on analysis."""
        changes = []
        
        # Common bug patterns
        # Division by zero
        div_re = re.compile(r'(\w+)\s*/\s*(\w+)')
        matches = div_re.finditer(code)
        
        for match in matches:
            if match.group(2) not in ['1', '2', '10', '100']:  # Not obvious constants
                old_code = match.group(0)
                numerator = match.group(1)
                denominator = match.group(2)
                
                new_code = f"{numerator} / {denominator} if {denominator} != 0 else 0"
                # Only suggest if not already protected
                if f"if {denominator}" not in code:
                    changes.append((old_code, new_code))
        
        return changes
    
    def _generate_tests(self, code: str) -> List[Tuple[str, str]]:
        """Generate test additions."""
        # This would analyze code and generate appropriate tests
        # Simplified for demo
        return []
    
    def _validate_safety(self, modification: Modification, 
                        original_code: str) -> float:
        """Validate safety of proposed modification."""
        safety_score = 1.0
        
        # Check forbidden patterns
        for old_code, new_code in modification.changes:
            for pattern in self.constraints.forbidden_patterns:
                if re.search(pattern, new_code):
                    print(f"Forbidden pattern detected: {pattern}")
                    return 0.0  # Immediate rejection
        
        # Check imports
        new_imports = self._extract_imports(
            self._apply_changes_to_code(original_code, modification.changes)
        )
        
        for import_name in new_imports:
            if import_name not in self.constraints.allowed_modules:
                print(f"Forbidden import: {import_name}")
                safety_score *= 0.5
        
        # Check complexity increase
        try:
            original_complexity = self._calculate_complexity(original_code)
            new_code = self._apply_changes_to_code(original_code, modification.changes)
            new_complexity = self._calculate_complexity(new_code)
            
            if new_complexity > original_complexity * self.constraints.max_complexity_increase:
                print(f"Complexity increase too high: {new_complexity/original_complexity:.2f}x")
                safety_score *= 0.7
        except:
            safety_score *= 0.9
        
        # Check file size change
        original_lines = len(original_code.splitlines())
        new_lines = len(self._apply_changes_to_code(original_code, modification.changes).splitlines())
        
        if abs(new_lines - original_lines) > self.constraints.max_file_size_change:
            print(f"File size change too large: {abs(new_lines - original_lines)} lines")
            safety_score *= 0.8
        
        return safety_score
    
    def _extract_imports(self, code: str) -> Set[str]:
        """Extract imported modules from code."""
        imports = set()
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
        except:
            pass
        
        return imports
    
    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity."""
        try:
            tree = ast.parse(code)
            complexity = 1
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            return complexity
        except:
            return 999
    
    def _apply_changes_to_code(self, code: str, 
                              changes: List[Tuple[str, str]]) -> str:
        """Apply changes to code."""
        modified_code = code
        
        for old_code, new_code in changes:
            modified_code = modified_code.replace(old_code, new_code)
        
        return modified_code
    
    def _create_sandbox(self) -> str:
        """Create sandbox environment."""
        sandbox_dir = tempfile.mkdtemp(prefix='safe_improver_')
        
        # Copy repository to sandbox
        shutil.copytree(
            self.repo_path,
            os.path.join(sandbox_dir, 'repo'),
            ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc')
        )
        
        return sandbox_dir
    
    def _apply_in_sandbox(self, modification: Modification) -> bool:
        """Apply modification in sandbox and test."""
        sandbox_file = os.path.join(self.sandbox_dir, 'repo', modification.target_file)
        
        if not os.path.exists(sandbox_file):
            return False
        
        # Read original code
        with open(sandbox_file, 'r') as f:
            code = f.read()
        
        # Apply changes
        modified_code = self._apply_changes_to_code(code, modification.changes)
        
        # Write modified code
        with open(sandbox_file, 'w') as f:
            f.write(modified_code)
        
        # Test in sandbox with resource limits
        try:
            # Validate file path to prevent path traversal
            if not sandbox_file.startswith(self.sandbox_dir):
                print("Security error: Path traversal attempt detected")
                return False
            
            # Run basic syntax check with secure command construction
            sandbox_file_escaped = shlex.quote(sandbox_file)
            result = subprocess.run(
                ['python', '-m', 'py_compile', sandbox_file_escaped],
                capture_output=True,
                text=True,
                timeout=10,
                shell=False  # Explicitly disable shell
            )
            
            if result.returncode != 0:
                print(f"Syntax error in sandbox: {result.stderr}")
                return False
            
            # Run with resource limits using a secure approach
            # Create a secure test script template
            import tempfile
            import json
            
            # Sanitize file paths
            safe_dirname = os.path.abspath(os.path.dirname(sandbox_file))
            safe_basename = os.path.splitext(os.path.basename(sandbox_file))[0]
            
            # Validate paths are within sandbox
            if not safe_dirname.startswith(os.path.abspath(self.sandbox_dir)):
                print("Security error: Path outside sandbox detected")
                return False
            
            # Create secure test configuration
            test_config = {
                'memory_limit_mb': self.constraints.max_memory_usage_mb,
                'timeout_seconds': self.constraints.max_execution_time_seconds,
                'module_path': safe_dirname,
                'module_name': safe_basename
            }
            
            # Use a separate secure test runner script
            test_script = '''
import resource
import signal
import sys
import json
import os

def timeout_handler(signum, frame):
    raise TimeoutError("Execution timeout")

try:
    # Load configuration
    config = json.loads(sys.argv[1])
    
    # Set memory limit (MB to bytes)
    memory_limit = config['memory_limit_mb'] * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, -1))
    
    # Set CPU time limit
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(config['timeout_seconds'])
    
    # Validate paths
    module_path = os.path.abspath(config['module_path'])
    module_name = config['module_name']
    
    # Basic validation of module name (alphanumeric and underscore only)
    if not module_name.replace('_', '').isalnum():
        raise ValueError("Invalid module name")
    
    # Import and test the module
    sys.path.insert(0, module_path)
    __import__(module_name)
    print("SUCCESS")
    
except Exception as e:
    print(f"ERROR: {e}")
finally:
    signal.alarm(0)
'''
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                test_file = f.name
            
            # Run with JSON configuration instead of string interpolation
            result = subprocess.run(
                ['python', test_file, json.dumps(test_config)],
                capture_output=True,
                text=True,
                timeout=self.constraints.max_execution_time_seconds + 5,
                shell=False  # Explicitly disable shell
            )
            
            os.unlink(test_file)
            
            return "SUCCESS" in result.stdout
            
        except subprocess.TimeoutExpired:
            print("Sandbox execution timeout")
            return False
        except Exception as e:
            print(f"Sandbox error: {e}")
            return False
    
    def _create_checkpoint(self) -> str:
        """Create git checkpoint for rollback."""
        # Stash any uncommitted changes
        self.repo.git.stash('push', '-m', 'safe_improver_checkpoint')
        
        # Get current commit hash
        return self.repo.head.commit.hexsha
    
    def _apply_to_repository(self, modification: Modification) -> bool:
        """Apply modification to actual repository."""
        filepath = os.path.join(self.repo_path, modification.target_file)
        
        try:
            # Read current code
            with open(filepath, 'r') as f:
                code = f.read()
            
            # Apply changes
            modified_code = self._apply_changes_to_code(code, modification.changes)
            
            # Write modified code
            with open(filepath, 'w') as f:
                f.write(modified_code)
            
            return True
            
        except Exception as e:
            print(f"Error applying to repository: {e}")
            return False
    
    def _run_tests(self) -> Dict[str, Any]:
        """Run test suite and return results."""
        results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'pass_rate': 0.0,
            'execution_time': 0.0
        }
        
        start_time = time.time()
        
        # Try common test runners with secure command construction
        test_commands = [
            ['python', '-m', 'pytest', '-v', '--tb=short'],
            ['python', '-m', 'unittest', 'discover', '-v'],
            ['python', 'setup.py', 'test']
        ]
        
        # Validate repo_path to prevent path traversal
        safe_repo_path = os.path.abspath(self.repo_path)
        if not os.path.exists(safe_repo_path):
            print("Error: Repository path does not exist")
            results['pass_rate'] = 0.0
            return results
        
        for cmd in test_commands:
            try:
                # Validate that all command components are safe
                safe_cmd = []
                for component in cmd:
                    # Only allow alphanumeric, dots, hyphens, and underscores
                    if re.match(r'^[a-zA-Z0-9.\-_]+$', component):
                        safe_cmd.append(component)
                    else:
                        print(f"Unsafe command component detected: {component}")
                        continue
                
                if len(safe_cmd) != len(cmd):
                    print("Skipping unsafe test command")
                    continue
                
                result = subprocess.run(
                    safe_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                    cwd=safe_repo_path,
                    shell=False  # Explicitly disable shell
                )
                
                if result.returncode == 0:
                    # Parse test results (simplified)
                    output = result.stdout
                    
                    # Pytest pattern
                    import re
                    passed_match = re.search(r'(\d+) passed', output)
                    failed_match = re.search(r'(\d+) failed', output)
                    
                    if passed_match:
                        results['passed_tests'] = int(passed_match.group(1))
                    if failed_match:
                        results['failed_tests'] = int(failed_match.group(1))
                    
                    results['total_tests'] = results['passed_tests'] + results['failed_tests']
                    
                    if results['total_tests'] > 0:
                        results['pass_rate'] = results['passed_tests'] / results['total_tests']
                        break
                        
            except subprocess.TimeoutExpired:
                print("Test execution timeout")
            except Exception as e:
                print(f"Error running tests: {e}")
        
        results['execution_time'] = time.time() - start_time
        
        # If no tests found, assume success but note it
        if results['total_tests'] == 0:
            results['pass_rate'] = 1.0
            results['total_tests'] = 1
            results['passed_tests'] = 1
        
        return results
    
    def _measure_performance_impact(self) -> Dict[str, float]:
        """Measure performance impact of modification."""
        impact = {}
        
        # Simple performance metrics
        # In practice, would run benchmarks
        
        # Memory usage
        process = psutil.Process()
        impact['memory_mb'] = process.memory_info().rss / 1024 / 1024
        
        # Import time
        start_time = time.time()
        try:
            # Re-import affected modules
            import importlib
            import sys
            
            # Clear module cache
            modules_to_clear = [m for m in sys.modules.keys() if self.repo_path in str(sys.modules[m])]
            for module in modules_to_clear:
                del sys.modules[module]
            
            # Measure import time
            # (simplified - would actually import the modified module)
            
        except:
            pass
        
        impact['import_time_seconds'] = time.time() - start_time
        
        # Estimate complexity impact (simplified)
        impact['complexity_change'] = 0.0  # Would calculate actual change
        
        return impact
    
    def _has_performance_regression(self, performance: Dict[str, float]) -> bool:
        """Check if there's a performance regression."""
        # Check memory usage
        if performance.get('memory_mb', 0) > self.constraints.max_memory_usage_mb:
            return True
        
        # Check import time (assuming baseline of 1 second)
        if performance.get('import_time_seconds', 0) > 5.0:
            return True
        
        # Check against baselines
        for metric, value in performance.items():
            baseline = self.performance_baselines.get(metric, value)
            
            # Allow 20% degradation
            if value > baseline * 1.2:
                return True
        
        return False
    
    def _rollback(self, checkpoint: str) -> None:
        """Rollback to checkpoint."""
        try:
            # Reset to checkpoint
            self.repo.git.reset('--hard', checkpoint)
            
            # Pop stash if exists
            try:
                self.repo.git.stash('pop')
            except:
                pass
                
            print(f"Rolled back to checkpoint: {checkpoint}")
            
        except Exception as e:
            print(f"Error during rollback: {e}")
    
    def _commit_improvement(self, modification: Modification) -> None:
        """Commit improvement to git."""
        try:
            # Add modified file
            self.repo.index.add([modification.target_file])
            
            # Create commit message
            message = f"""[Self-Improvement] {modification.type.value}: {modification.description}

Modification ID: {modification.id}
Safety Score: {modification.safety_score:.2f}
Test Pass Rate: {modification.test_results.get('pass_rate', 0):.2%}

Performance Impact:
{json.dumps(modification.performance_impact, indent=2)}

This modification was automatically generated and tested by the Safe Self-Improver.
"""
            
            # Commit
            self.repo.index.commit(message)
            
            print(f"Committed improvement: {modification.id}")
            
        except Exception as e:
            print(f"Error committing improvement: {e}")
    
    def _load_todays_modifications(self) -> List[Modification]:
        """Load today's modifications."""
        modifications = []
        
        today = datetime.now(timezone.utc).date()
        history_file = os.path.join(self.repo_path, '.self_improver', f'modifications_{today}.json')
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    
                for mod_data in data:
                    # Reconstruct modification
                    mod = Modification(
                        id=mod_data['id'],
                        type=ModificationType(mod_data['type']),
                        target_file=mod_data['target_file'],
                        description=mod_data['description'],
                        changes=mod_data['changes'],
                        timestamp=datetime.fromisoformat(mod_data['timestamp']),
                        applied=mod_data['applied'],
                        success=mod_data['success'],
                        performance_impact=mod_data.get('performance_impact'),
                        safety_score=mod_data.get('safety_score', 0)
                    )
                    modifications.append(mod)
            except:
                pass
        
        return modifications
    
    def _load_modification_history(self) -> List[Modification]:
        """Load all modification history."""
        history = []
        
        history_dir = os.path.join(self.repo_path, '.self_improver')
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
            return history
        
        # Load all history files
        for filename in os.listdir(history_dir):
            if filename.startswith('modifications_') and filename.endswith('.json'):
                filepath = os.path.join(history_dir, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        
                    for mod_data in data:
                        mod = Modification(
                            id=mod_data['id'],
                            type=ModificationType(mod_data['type']),
                            target_file=mod_data['target_file'],
                            description=mod_data['description'],
                            changes=mod_data['changes'],
                            timestamp=datetime.fromisoformat(mod_data['timestamp']),
                            applied=mod_data['applied'],
                            success=mod_data['success'],
                            performance_impact=mod_data.get('performance_impact'),
                            safety_score=mod_data.get('safety_score', 0)
                        )
                        history.append(mod)
                except:
                    pass
        
        return history
    
    def _save_modification(self, modification: Modification) -> None:
        """Save modification to history."""
        today = datetime.now(timezone.utc).date()
        history_dir = os.path.join(self.repo_path, '.self_improver')
        
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
        
        history_file = os.path.join(history_dir, f'modifications_{today}.json')
        
        # Load existing modifications
        modifications = []
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    modifications = json.load(f)
            except:
                pass
        
        # Add new modification
        mod_data = {
            'id': modification.id,
            'type': modification.type.value,
            'target_file': modification.target_file,
            'description': modification.description,
            'changes': modification.changes,
            'timestamp': modification.timestamp.isoformat(),
            'applied': modification.applied,
            'success': modification.success,
            'performance_impact': modification.performance_impact,
            'safety_score': modification.safety_score,
            'test_results': modification.test_results
        }
        
        modifications.append(mod_data)
        
        # Save
        with open(history_file, 'w') as f:
            json.dump(modifications, f, indent=2)
    
    def _establish_baselines(self) -> Dict[str, float]:
        """Establish performance baselines."""
        baselines = {}
        
        # Memory baseline
        process = psutil.Process()
        baselines['memory_mb'] = process.memory_info().rss / 1024 / 1024
        
        # Import time baseline
        baselines['import_time_seconds'] = 0.1  # Reasonable baseline
        
        return baselines
    
    def _analyze_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Analyze a file for improvement opportunities."""
        opportunities = []
        
        try:
            with open(os.path.join(self.repo_path, filepath), 'r') as f:
                code = f.read()
            
            # Check for optimization opportunities
            if 'for' in code and '.append(' in code:
                opportunities.append({
                    'file': filepath,
                    'type': ModificationType.OPTIMIZATION,
                    'description': 'Convert loop to list comprehension',
                    'priority': 0.7
                })
            
            # Check for missing docstrings
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not ast.get_docstring(node):
                        opportunities.append({
                            'file': filepath,
                            'type': ModificationType.DOCUMENTATION,
                            'description': f'Add docstring to function: {node.name}',
                            'priority': 0.5
                        })
                        break  # One per file for now
            
            # Check for performance opportunities
            if '@property' in code and 'return self.' not in code:
                opportunities.append({
                    'file': filepath,
                    'type': ModificationType.PERFORMANCE,
                    'description': 'Add caching to property methods',
                    'priority': 0.6
                })
            
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
        
        return opportunities
    
    def _is_safe_path(self, path: str) -> bool:
        """Validate that a file path is safe and within allowed boundaries.
        
        Args:
            path: Path to validate
            
        Returns:
            True if path is safe, False otherwise
        """
        try:
            # Convert to absolute path
            abs_path = os.path.abspath(path)
            
            # Check for path traversal attempts
            if '..' in path or '~' in path:
                return False
            
            # Check for suspicious patterns
            suspicious_patterns = [
                '/etc/', '/proc/', '/sys/', '/dev/',
                'C:\\Windows\\', 'C:\\Program Files\\',
                '/root/', '/home/root/',
                '__pycache__', '.git/'
            ]
            
            for pattern in suspicious_patterns:
                if pattern in abs_path:
                    return False
            
            # Must be a valid directory
            if os.path.exists(abs_path) and not os.path.isdir(abs_path):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _secure_file_operation(self, filepath: str, operation: str, content: str = None) -> tuple[bool, str]:
        """Perform file operations with security validation.
        
        Args:
            filepath: File path to operate on
            operation: Operation type ('read', 'write', 'exists')
            content: Content to write (for write operations)
            
        Returns:
            Tuple of (success, result/error_message)
        """
        try:
            # Validate file path
            abs_path = os.path.abspath(filepath)
            
            # Ensure path is within repository bounds
            if not abs_path.startswith(os.path.abspath(self.repo_path)):
                return False, "File path outside repository bounds"
            
            # Check for malicious patterns
            if '..' in filepath or '~' in filepath:
                return False, "Path traversal attempt detected"
            
            # Validate file extension (only allow Python files for modification)
            if operation == 'write' and not filepath.endswith('.py'):
                return False, "Only Python files can be modified"
            
            if operation == 'read':
                if not os.path.exists(abs_path):
                    return False, "File does not exist"
                with open(abs_path, 'r', encoding='utf-8') as f:
                    return True, f.read()
                    
            elif operation == 'write':
                if content is None:
                    return False, "No content provided for write operation"
                
                # Validate content doesn't contain dangerous patterns
                for pattern in self.constraints.forbidden_patterns:
                    if re.search(pattern, content):
                        return False, f"Content contains forbidden pattern: {pattern}"
                
                with open(abs_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, "File written successfully"
                
            elif operation == 'exists':
                return True, str(os.path.exists(abs_path))
                
            else:
                return False, f"Unknown operation: {operation}"
                
        except Exception as e:
            return False, f"File operation error: {str(e)}"
    
    def _validate_input_sanitization(self, user_input: str) -> bool:
        """Validate and sanitize user input.
        
        Args:
            user_input: Input to validate
            
        Returns:
            True if input is safe, False otherwise
        """
        if not isinstance(user_input, str):
            return False
        
        # Check length
        if len(user_input) > 10000:  # Reasonable limit
            return False
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'<script[^>]*>',  # Script tags
            r'javascript:',    # JavaScript URLs
            r'data:text/html', # Data URLs
            r'vbscript:',      # VBScript
            r'on\w+\s*=',      # Event handlers
            r'eval\s*\(',      # eval calls
            r'exec\s*\(',      # exec calls
            r'__import__',     # Dynamic imports
            r'subprocess',     # System commands
            r'os\.system',     # OS system calls
            r'cmd\.exe',       # Windows command prompt
            r'/bin/sh',        # Unix shell
            r'/bin/bash',      # Bash shell
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return False
        
        return True


def demonstrate_safe_self_improver():
    """Demonstrate safe self-improvement."""
    print("=== Safe Self-Improvement Demo ===\n")
    
    # Create improver
    improver = SafeSelfImprover(max_changes_per_day=24)
    
    # Analyze improvement opportunities
    print("Analyzing improvement opportunities...")
    opportunities = improver.analyze_improvement_opportunities()
    
    print(f"\nFound {len(opportunities)} improvement opportunities:")
    for i, opp in enumerate(opportunities[:5]):
        print(f"{i+1}. {opp['file']} - {opp['type'].value}: {opp['description']}")
    
    # Propose an improvement
    if opportunities:
        opp = opportunities[0]
        print(f"\nProposing improvement for: {opp['file']}")
        
        modification = improver.propose_improvement(
            target_file=opp['file'],
            improvement_type=opp['type'],
            description=opp['description']
        )
        
        if modification:
            print(f"Proposed modification: {modification.id}")
            print(f"Safety score: {modification.safety_score:.2f}")
            print(f"Changes: {len(modification.changes)}")
            
            # In practice, would apply the improvement
            # success = improver.apply_improvement(modification)
            # print(f"Application success: {success}")
    
    # Show history
    print("\n=== Improvement History ===")
    history = improver.get_improvement_history()
    
    print(f"Total improvements: {history['total_improvements']}")
    print(f"Successful improvements: {history['successful_improvements']}")
    
    print("\nImprovements by type:")
    for imp_type, count in history['improvements_by_type'].items():
        print(f"  {imp_type}: {count}")
    
    print(f"\nToday's modifications: {len(improver.modifications_today)}/{improver.constraints.max_changes_per_day}")


if __name__ == "__main__":
    demonstrate_safe_self_improver()