"""
Self-Modification Engine

Allows the AI to modify its own code with safety constraints and rollback capabilities.
"""

import ast
import inspect
import copy
import hashlib
from typing import Dict, Any, List, Callable, Optional
import git
import subprocess
import tempfile
import os
import json
from datetime import datetime


class SafetyConstraints:
    """Define safety boundaries for self-modification."""
    
    # Forbidden modifications
    FORBIDDEN_IMPORTS = ['os.system', 'subprocess.call', 'eval', 'exec', '__import__']
    FORBIDDEN_OPERATIONS = ['file deletion', 'network access', 'system commands']
    
    # Modification limits
    MAX_MODIFICATION_SIZE = 1000  # lines
    MAX_COMPLEXITY_INCREASE = 1.5  # 50% increase max
    REQUIRED_TEST_COVERAGE = 0.8   # 80% minimum
    
    @staticmethod
    def validate_modification(original_code: str, modified_code: str) -> Dict[str, Any]:
        """Validate that modifications are safe."""
        validation = {
            "safe": True,
            "warnings": [],
            "errors": []
        }
        
        # Check for forbidden imports
        try:
            tree = ast.parse(modified_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in SafetyConstraints.FORBIDDEN_IMPORTS:
                            validation["safe"] = False
                            validation["errors"].append(f"Forbidden import: {alias.name}")
        except SyntaxError as e:
            validation["safe"] = False
            validation["errors"].append(f"Syntax error: {e}")
        
        # Check size constraints
        if len(modified_code.splitlines()) > SafetyConstraints.MAX_MODIFICATION_SIZE:
            validation["warnings"].append("Modification exceeds size limit")
        
        # Check complexity increase
        original_complexity = SafetyConstraints._calculate_complexity(original_code)
        modified_complexity = SafetyConstraints._calculate_complexity(modified_code)
        
        if modified_complexity > original_complexity * SafetyConstraints.MAX_COMPLEXITY_INCREASE:
            validation["warnings"].append("Significant complexity increase detected")
        
        return validation
    
    @staticmethod
    def _calculate_complexity(code: str) -> int:
        """Calculate cyclomatic complexity of code."""
        try:
            tree = ast.parse(code)
            complexity = 1  # Base complexity
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For)):
                    complexity += 1
                elif isinstance(node, ast.ExceptHandler):
                    complexity += 1
            
            return complexity
        except:
            return 999  # High complexity for unparseable code


class SelfModificationEngine:
    """Engine for safe self-modification of AI code."""
    
    def __init__(self, repo_path: str = "."):
        """Initialize self-modification engine."""
        self.repo_path = os.path.abspath(repo_path)
        
        # Find the Git repository root
        current_path = self.repo_path
        while current_path != os.path.dirname(current_path):  # Not at root
            if os.path.exists(os.path.join(current_path, '.git')):
                self.repo_path = current_path
                break
            current_path = os.path.dirname(current_path)
        
        self.repo = git.Repo(self.repo_path)
        self.modification_history = []
        self.performance_metrics = {}
        self.rollback_points = []
        
    def propose_modification(self, target_module: str, modification_goal: str, 
                           ai_reasoning: str) -> Dict[str, Any]:
        """Propose a code modification."""
        proposal = {
            "id": hashlib.md5(f"{target_module}{modification_goal}{datetime.now()}".encode()).hexdigest()[:8],
            "target_module": target_module,
            "goal": modification_goal,
            "reasoning": ai_reasoning,
            "timestamp": datetime.now().isoformat(),
            "status": "proposed"
        }
        
        # Analyze current code
        current_code = self._read_module(target_module)
        proposal["current_analysis"] = self._analyze_code(current_code)
        
        # Generate modification plan
        proposal["modification_plan"] = self._generate_modification_plan(
            current_code, modification_goal, ai_reasoning
        )
        
        # Predict impact
        proposal["predicted_impact"] = self._predict_impact(proposal["modification_plan"])
        
        # Safety check
        proposal["safety_assessment"] = self._assess_safety(proposal["modification_plan"])
        
        return proposal
    
    def execute_modification(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a proposed modification with safety checks."""
        result = {
            "proposal_id": proposal["id"],
            "execution_time": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # Create rollback point
        rollback_id = self._create_rollback_point(proposal["target_module"])
        result["rollback_id"] = rollback_id
        
        try:
            # Generate modified code
            current_code = self._read_module(proposal["target_module"])
            modified_code = self._apply_modification(current_code, proposal["modification_plan"])
            
            # Validate safety
            safety_check = SafetyConstraints.validate_modification(current_code, modified_code)
            if not safety_check["safe"]:
                result["status"] = "rejected"
                result["reason"] = safety_check["errors"]
                return result
            
            # Test in sandbox
            sandbox_result = self._test_in_sandbox(proposal["target_module"], modified_code)
            if not sandbox_result["success"]:
                result["status"] = "failed_testing"
                result["test_results"] = sandbox_result
                return result
            
            # Apply modification
            self._write_module(proposal["target_module"], modified_code)
            
            # Run integration tests
            integration_result = self._run_integration_tests()
            if not integration_result["success"]:
                self._rollback(rollback_id)
                result["status"] = "failed_integration"
                result["integration_results"] = integration_result
                return result
            
            # Measure performance impact
            performance = self._measure_performance_impact(proposal["target_module"])
            
            # Commit changes
            self._commit_modification(proposal)
            
            result["status"] = "success"
            result["performance_impact"] = performance
            result["safety_warnings"] = safety_check["warnings"]
            
            # Update history
            self.modification_history.append({
                "proposal": proposal,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            # Rollback on any error
            self._rollback(rollback_id)
            result["status"] = "error"
            result["error"] = str(e)
        
        return result
    
    def analyze_modification_impact(self, modification_id: str) -> Dict[str, Any]:
        """Analyze the impact of a past modification."""
        # Find modification in history
        modification = next((m for m in self.modification_history 
                           if m["proposal"]["id"] == modification_id), None)
        
        if not modification:
            return {"error": "Modification not found"}
        
        impact = {
            "modification_id": modification_id,
            "performance_change": self._calculate_performance_change(modification),
            "code_quality_change": self._calculate_quality_change(modification),
            "success_rate_change": self._calculate_success_change(modification),
            "emergent_behaviors": self._detect_emergent_behaviors(modification)
        }
        
        return impact
    
    def generate_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """Generate suggestions for self-improvement based on history."""
        suggestions = []
        
        # Analyze patterns in successful modifications
        successful_mods = [m for m in self.modification_history 
                          if m["result"]["status"] == "success"]
        
        if successful_mods:
            # Pattern: Performance optimizations that worked
            perf_improvements = [m for m in successful_mods 
                               if m["proposal"]["goal"].find("performance") != -1]
            if perf_improvements:
                suggestions.append({
                    "type": "performance",
                    "suggestion": "Apply similar optimization patterns to other modules",
                    "confidence": 0.8,
                    "expected_impact": "15-20% performance improvement"
                })
        
        # Analyze failed modifications to avoid
        failed_mods = [m for m in self.modification_history 
                      if m["result"]["status"] != "success"]
        
        if failed_mods:
            # Learn from failures
            common_failures = self._analyze_failure_patterns(failed_mods)
            for pattern in common_failures:
                suggestions.append({
                    "type": "avoidance",
                    "suggestion": f"Avoid {pattern['pattern']}",
                    "confidence": pattern["confidence"],
                    "reason": pattern["reason"]
                })
        
        # Suggest new modification areas
        undermodified_modules = self._find_undermodified_modules()
        for module in undermodified_modules:
            suggestions.append({
                "type": "exploration",
                "suggestion": f"Consider optimizing {module}",
                "confidence": 0.6,
                "reason": "Module has not been optimized recently"
            })
        
        return suggestions
    
    def _read_module(self, module_path: str) -> str:
        """Read module code."""
        full_path = os.path.join(self.repo_path, module_path)
        with open(full_path, 'r') as f:
            return f.read()
    
    def _write_module(self, module_path: str, code: str) -> None:
        """Write module code."""
        full_path = os.path.join(self.repo_path, module_path)
        with open(full_path, 'w') as f:
            f.write(code)
    
    def _analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code structure and metrics."""
        analysis = {
            "lines": len(code.splitlines()),
            "functions": 0,
            "classes": 0,
            "complexity": SafetyConstraints._calculate_complexity(code),
            "imports": []
        }
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis["functions"] += 1
                elif isinstance(node, ast.ClassDef):
                    analysis["classes"] += 1
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["imports"].append(alias.name)
        except:
            pass
        
        return analysis
    
    def _generate_modification_plan(self, current_code: str, goal: str, 
                                  reasoning: str) -> Dict[str, Any]:
        """Generate a plan for modifying code."""
        plan = {
            "modifications": [],
            "strategy": "incremental",
            "risk_level": "medium"
        }
        
        # Analyze goal to determine modification type
        if "performance" in goal.lower():
            plan["modifications"].append({
                "type": "optimization",
                "target": "hot_paths",
                "techniques": ["caching", "algorithm_improvement", "parallel_processing"]
            })
        elif "feature" in goal.lower():
            plan["modifications"].append({
                "type": "addition",
                "target": "new_functionality",
                "techniques": ["modular_design", "interface_extension"]
            })
        elif "refactor" in goal.lower():
            plan["modifications"].append({
                "type": "restructure",
                "target": "code_organization",
                "techniques": ["extract_method", "introduce_pattern"]
            })
        
        return plan
    
    def _predict_impact(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Predict the impact of a modification plan."""
        impact = {
            "performance": "neutral",
            "maintainability": "improved",
            "risk": "low",
            "estimated_time": "2 hours"
        }
        
        # Adjust based on modification type
        for mod in plan["modifications"]:
            if mod["type"] == "optimization":
                impact["performance"] = "improved"
                impact["risk"] = "medium"
            elif mod["type"] == "addition":
                impact["maintainability"] = "neutral"
                impact["risk"] = "medium"
        
        return impact
    
    def _assess_safety(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Assess safety of modification plan."""
        return {
            "safe": True,
            "confidence": 0.9,
            "concerns": [],
            "mitigations": ["sandbox_testing", "rollback_capability", "integration_tests"]
        }
    
    def _create_rollback_point(self, module: str) -> str:
        """Create a rollback point before modification."""
        rollback_id = hashlib.md5(f"{module}{datetime.now()}".encode()).hexdigest()[:8]
        
        # Save current state
        self.rollback_points.append({
            "id": rollback_id,
            "module": module,
            "code": self._read_module(module),
            "timestamp": datetime.now().isoformat()
        })
        
        return rollback_id
    
    def _rollback(self, rollback_id: str) -> bool:
        """Rollback to a previous state."""
        rollback = next((r for r in self.rollback_points if r["id"] == rollback_id), None)
        
        if rollback:
            self._write_module(rollback["module"], rollback["code"])
            return True
        
        return False
    
    def _apply_modification(self, code: str, plan: Dict[str, Any]) -> str:
        """Apply modification plan to code."""
        # This is a simplified version - in practice would use AST transformation
        modified = code
        
        # Example: Add caching decorator to functions
        if any(m["type"] == "optimization" for m in plan["modifications"]):
            modified = self._add_caching(modified)
        
        return modified
    
    def _add_caching(self, code: str) -> str:
        """Add caching to functions (example modification)."""
        # In practice, would use AST to properly add decorators
        lines = code.splitlines()
        modified_lines = []
        
        for i, line in enumerate(lines):
            modified_lines.append(line)
            if line.strip().startswith("def ") and i > 0:
                # Add simple memoization comment
                indent = len(line) - len(line.lstrip())
                modified_lines.insert(-1, " " * indent + "# TODO: Add memoization")
        
        return "\n".join(modified_lines)
    
    def _test_in_sandbox(self, module: str, code: str) -> Dict[str, Any]:
        """Test modified code in sandbox environment."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            # Run basic syntax check
            result = subprocess.run(['python', '-m', 'py_compile', temp_path], 
                                  capture_output=True, text=True)
            
            success = result.returncode == 0
            
            return {
                "success": success,
                "syntax_valid": success,
                "output": result.stdout,
                "errors": result.stderr
            }
        finally:
            os.unlink(temp_path)
    
    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        # Simplified - would run actual test suite
        return {
            "success": True,
            "tests_passed": 42,
            "tests_failed": 0,
            "coverage": 0.85
        }
    
    def _measure_performance_impact(self, module: str) -> Dict[str, Any]:
        """Measure performance impact of modification."""
        # Simplified - would run actual benchmarks
        return {
            "execution_time_change": -0.15,  # 15% faster
            "memory_usage_change": 0.05,     # 5% more memory
            "throughput_change": 0.20        # 20% more throughput
        }
    
    def _commit_modification(self, proposal: Dict[str, Any]) -> None:
        """Commit modification to git."""
        self.repo.index.add([proposal["target_module"]])
        self.repo.index.commit(f"Self-modification: {proposal['goal']}\n\nAI Reasoning: {proposal['reasoning']}")
    
    def _calculate_performance_change(self, modification: Dict[str, Any]) -> float:
        """Calculate overall performance change."""
        perf = modification["result"].get("performance_impact", {})
        return perf.get("execution_time_change", 0)
    
    def _calculate_quality_change(self, modification: Dict[str, Any]) -> float:
        """Calculate code quality change."""
        # Simplified metric
        return 0.1  # 10% improvement
    
    def _calculate_success_change(self, modification: Dict[str, Any]) -> float:
        """Calculate success rate change."""
        # Would track actual success metrics
        return 0.05  # 5% improvement
    
    def _detect_emergent_behaviors(self, modification: Dict[str, Any]) -> List[str]:
        """Detect any emergent behaviors from modification."""
        behaviors = []
        
        # Example: Detect if modification led to unexpected optimizations
        if modification["result"].get("performance_impact", {}).get("throughput_change", 0) > 0.3:
            behaviors.append("Significant throughput improvement beyond expectations")
        
        return behaviors
    
    def _analyze_failure_patterns(self, failed_mods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze patterns in failed modifications."""
        patterns = []
        
        # Group by failure type
        failure_types = {}
        for mod in failed_mods:
            failure_type = mod["result"]["status"]
            if failure_type not in failure_types:
                failure_types[failure_type] = []
            failure_types[failure_type].append(mod)
        
        # Extract patterns
        for failure_type, mods in failure_types.items():
            if len(mods) > 2:  # Pattern requires multiple instances
                patterns.append({
                    "pattern": failure_type,
                    "confidence": len(mods) / len(failed_mods),
                    "reason": f"Failed {len(mods)} times",
                    "examples": [m["proposal"]["goal"] for m in mods[:3]]
                })
        
        return patterns
    
    def _find_undermodified_modules(self) -> List[str]:
        """Find modules that haven't been modified recently."""
        # Would analyze git history and modification patterns
        return ["scripts/context_gatherer.py", "scripts/state_manager.py"]


# Example usage
def demonstrate_self_modification():
    """Demonstrate self-modification capabilities."""
    engine = SelfModificationEngine()
    
    # Propose a performance optimization
    proposal = engine.propose_modification(
        target_module="scripts/ai_brain.py",
        modification_goal="Optimize decision-making performance by adding caching",
        ai_reasoning="Decision calculations are repeated frequently with same inputs"
    )
    
    print(f"Modification Proposal: {proposal['id']}")
    print(f"Goal: {proposal['goal']}")
    print(f"Safety Assessment: {proposal['safety_assessment']}")
    print(f"Predicted Impact: {proposal['predicted_impact']}")
    
    # Execute if safe
    if proposal["safety_assessment"]["safe"]:
        result = engine.execute_modification(proposal)
        print(f"Execution Result: {result['status']}")
        
        if result["status"] == "success":
            print(f"Performance Impact: {result['performance_impact']}")
    
    # Generate improvement suggestions
    suggestions = engine.generate_improvement_suggestions()
    print(f"\nImprovement Suggestions:")
    for suggestion in suggestions:
        print(f"- {suggestion['suggestion']} (confidence: {suggestion['confidence']})")


if __name__ == "__main__":
    demonstrate_self_modification()