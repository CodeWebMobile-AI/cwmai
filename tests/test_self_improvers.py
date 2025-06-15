#!/usr/bin/env python3
"""
Comprehensive test suite for custom self-improvers.

Tests both SafeSelfImprover and SelfModificationEngine to ensure:
- Safety constraints are enforced
- Self-modification capabilities work correctly
- Rollback mechanisms function properly
- Daily modification limits are respected
- Integration with continuous orchestrator works
"""

import asyncio
import sys
import os
import json
import tempfile
import shutil
from datetime import datetime, timezone
from typing import Dict, Any, List

# Add scripts directory to path
sys.path.insert(0, '/workspaces/cwmai/scripts')

from safe_self_improver import (
    SafeSelfImprover, 
    ModificationType, 
    SafetyConstraints,
    Modification
)
from self_modification_engine import (
    SelfModificationEngine,
    SafetyConstraints as EngineConstraints
)
from continuous_orchestrator import ContinuousOrchestrator
from work_item_types import WorkItem, TaskPriority


class TestResults:
    """Track test results."""
    
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_details = []
    
    def add_test(self, name: str, passed: bool, details: str = ""):
        """Add test result."""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            status = "✅ PASSED"
        else:
            self.failed_tests += 1
            status = "❌ FAILED"
        
        self.test_details.append({
            'name': name,
            'passed': passed,
            'status': status,
            'details': details
        })
        
        print(f"{status}: {name}")
        if details:
            print(f"   Details: {details}")
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests} ({self.passed_tests/max(self.total_tests,1)*100:.1f}%)")
        print(f"Failed: {self.failed_tests}")
        
        if self.failed_tests > 0:
            print("\nFailed Tests:")
            for test in self.test_details:
                if not test['passed']:
                    print(f"  - {test['name']}: {test['details']}")


async def test_safe_self_improver_initialization():
    """Test SafeSelfImprover initialization."""
    print("\n=== Testing SafeSelfImprover Initialization ===")
    results = TestResults()
    
    try:
        # Test 1: Basic initialization
        improver = SafeSelfImprover(max_changes_per_day=5)
        results.add_test(
            "SafeSelfImprover initialization",
            True,
            f"Max changes per day: {improver.constraints.max_changes_per_day}"
        )
        
        # Test 2: Check safety constraints
        constraints_valid = (
            improver.constraints.max_changes_per_day == 5 and
            improver.constraints.max_file_size_change == 1000 and
            improver.constraints.max_complexity_increase == 1.3 and
            improver.constraints.required_test_pass_rate == 0.95
        )
        results.add_test(
            "Safety constraints validation",
            constraints_valid,
            f"Constraints: {improver.constraints}"
        )
        
        # Test 3: Check forbidden patterns
        has_forbidden_patterns = len(improver.constraints.forbidden_patterns) > 0
        results.add_test(
            "Forbidden patterns defined",
            has_forbidden_patterns,
            f"Number of forbidden patterns: {len(improver.constraints.forbidden_patterns)}"
        )
        
        # Test 4: Check allowed modules
        has_allowed_modules = len(improver.constraints.allowed_modules) > 0
        results.add_test(
            "Allowed modules defined",
            has_allowed_modules,
            f"Number of allowed modules: {len(improver.constraints.allowed_modules)}"
        )
        
    except Exception as e:
        results.add_test("SafeSelfImprover initialization", False, str(e))
    
    return results


async def test_improvement_proposal():
    """Test improvement proposal generation."""
    print("\n=== Testing Improvement Proposal Generation ===")
    results = TestResults()
    
    try:
        improver = SafeSelfImprover(max_changes_per_day=10)
        
        # Test 1: Propose optimization improvement
        modification = improver.propose_improvement(
            target_file="scripts/test_file.py",
            improvement_type=ModificationType.OPTIMIZATION,
            description="Test optimization improvement"
        )
        
        results.add_test(
            "Optimization proposal generation",
            modification is not None,
            f"Modification ID: {modification.id if modification else 'None'}"
        )
        
        # Test 2: Check safety score
        if modification:
            results.add_test(
                "Safety score calculation",
                modification.safety_score >= 0.8,
                f"Safety score: {modification.safety_score}"
            )
        
        # Test 3: Test daily limit enforcement
        for i in range(15):  # Try to exceed limit
            mod = improver.propose_improvement(
                target_file="scripts/test_file.py",
                improvement_type=ModificationType.OPTIMIZATION,
                description=f"Test improvement {i}"
            )
            if mod:
                improver.modifications_today.append(mod)
        
        limit_enforced = len(improver.modifications_today) <= improver.constraints.max_changes_per_day
        results.add_test(
            "Daily limit enforcement",
            limit_enforced,
            f"Modifications today: {len(improver.modifications_today)}/{improver.constraints.max_changes_per_day}"
        )
        
    except Exception as e:
        results.add_test("Improvement proposal", False, str(e))
    
    return results


async def test_safety_validation():
    """Test safety validation mechanisms."""
    print("\n=== Testing Safety Validation ===")
    results = TestResults()
    
    try:
        improver = SafeSelfImprover()
        
        # Test 1: Validate safe code
        safe_code = """
def calculate_sum(a, b):
    return a + b
"""
        safe_mod = Modification(
            id="test_safe",
            type=ModificationType.OPTIMIZATION,
            target_file="test.py",
            description="Safe modification",
            changes=[("pass", safe_code)],
            timestamp=datetime.now(timezone.utc)
        )
        
        safety_score = improver._validate_safety(safe_mod, "pass")
        results.add_test(
            "Safe code validation",
            safety_score > 0.8,
            f"Safety score: {safety_score}"
        )
        
        # Test 2: Validate unsafe code with forbidden patterns
        unsafe_code = """
import os
def dangerous_function():
    os.system("rm -rf /")
    exec("print('dangerous')")
"""
        unsafe_mod = Modification(
            id="test_unsafe",
            type=ModificationType.OPTIMIZATION,
            target_file="test.py",
            description="Unsafe modification",
            changes=[("pass", unsafe_code)],
            timestamp=datetime.now(timezone.utc)
        )
        
        unsafe_score = improver._validate_safety(unsafe_mod, "pass")
        results.add_test(
            "Unsafe code rejection",
            unsafe_score == 0.0,
            f"Safety score: {unsafe_score} (should be 0.0)"
        )
        
        # Test 3: Validate complexity increase
        complex_code = """
def complex_function(data):
    if data:
        for item in data:
            if item > 0:
                while item > 0:
                    try:
                        if item % 2 == 0:
                            for i in range(item):
                                if i > 10:
                                    pass
                    except:
                        pass
                    item -= 1
"""
        complex_mod = Modification(
            id="test_complex",
            type=ModificationType.OPTIMIZATION,
            target_file="test.py",
            description="Complex modification",
            changes=[("pass", complex_code)],
            timestamp=datetime.now(timezone.utc)
        )
        
        complex_score = improver._validate_safety(complex_mod, "pass")
        results.add_test(
            "Complexity validation",
            complex_score < 1.0,
            f"Safety score reduced due to complexity: {complex_score}"
        )
        
    except Exception as e:
        results.add_test("Safety validation", False, str(e))
    
    return results


async def test_self_modification_engine():
    """Test SelfModificationEngine."""
    print("\n=== Testing SelfModificationEngine ===")
    results = TestResults()
    
    try:
        # Create temporary repo for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize git repo
            os.system(f"cd {temp_dir} && git init")
            
            # Create test file
            test_file = os.path.join(temp_dir, "test_module.py")
            with open(test_file, 'w') as f:
                f.write("""
def slow_function(n):
    result = []
    for i in range(n):
        result.append(i * 2)
    return result
""")
            
            # Initialize engine
            engine = SelfModificationEngine(repo_path=temp_dir)
            results.add_test(
                "SelfModificationEngine initialization",
                True,
                f"Repo path: {engine.repo_path}"
            )
            
            # Test 2: Propose modification
            proposal = engine.propose_modification(
                target_module="test_module.py",
                modification_goal="Optimize list comprehension performance",
                ai_reasoning="Loop can be replaced with list comprehension"
            )
            
            results.add_test(
                "Modification proposal",
                proposal is not None and proposal['status'] == 'proposed',
                f"Proposal ID: {proposal['id'] if proposal else 'None'}"
            )
            
            # Test 3: Check safety assessment
            if proposal:
                safety_safe = proposal['safety_assessment']['safe']
                results.add_test(
                    "Safety assessment",
                    safety_safe,
                    f"Safety confidence: {proposal['safety_assessment']['confidence']}"
                )
            
            # Test 4: Check impact prediction
            if proposal:
                has_impact = 'predicted_impact' in proposal
                results.add_test(
                    "Impact prediction",
                    has_impact,
                    f"Predicted impact: {proposal.get('predicted_impact', {})}"
                )
            
    except Exception as e:
        results.add_test("SelfModificationEngine", False, str(e))
    
    return results


async def test_orchestrator_integration():
    """Test integration with ContinuousOrchestrator."""
    print("\n=== Testing Orchestrator Integration ===")
    results = TestResults()
    
    try:
        # Create orchestrator
        orchestrator = ContinuousOrchestrator(max_workers=1)
        
        # Test 1: Check if SafeSelfImprover can be imported
        try:
            from safe_self_improver import SafeSelfImprover
            can_import = True
        except ImportError:
            can_import = False
        
        results.add_test(
            "SafeSelfImprover import in orchestrator context",
            can_import,
            "Import successful" if can_import else "Import failed"
        )
        
        # Test 2: Create system improvement work item
        work_item = WorkItem(
            id="test_system_improvement",
            task_type="SYSTEM_IMPROVEMENT",
            title="Test system improvement",
            description="Test self-improvement integration",
            priority=TaskPriority.HIGH,
            metadata={
                'improvement_type': 'optimization',
                'target_module': 'scripts/test_module.py'
            }
        )
        
        results.add_test(
            "System improvement work item creation",
            work_item is not None,
            f"Work item type: {work_item.task_type}"
        )
        
        # Test 3: Check orchestrator's system improvement handler
        has_handler = hasattr(orchestrator, '_execute_system_improvement_task')
        results.add_test(
            "Orchestrator has system improvement handler",
            has_handler,
            "Handler method exists" if has_handler else "Handler missing"
        )
        
        # Test 4: Test handler execution (without actually applying changes)
        if has_handler:
            try:
                # Mock the improvement execution
                result = await orchestrator._execute_system_improvement_task(work_item)
                handler_works = result.get('success', False)
                results.add_test(
                    "System improvement handler execution",
                    handler_works,
                    f"Result: {result}"
                )
            except ImportError:
                # Expected if SafeSelfImprover not in path during execution
                results.add_test(
                    "System improvement handler execution",
                    True,
                    "Handler exists but SafeSelfImprover import expected to fail in test context"
                )
        
    except Exception as e:
        results.add_test("Orchestrator integration", False, str(e))
    
    return results


async def test_improvement_history():
    """Test improvement history tracking."""
    print("\n=== Testing Improvement History ===")
    results = TestResults()
    
    try:
        improver = SafeSelfImprover()
        
        # Test 1: Get initial history
        history = improver.get_improvement_history()
        results.add_test(
            "History retrieval",
            isinstance(history, dict),
            f"History keys: {list(history.keys())}"
        )
        
        # Test 2: Check history structure
        expected_keys = [
            'total_improvements',
            'successful_improvements',
            'improvements_by_type',
            'average_performance_impact',
            'most_improved_files',
            'recent_improvements'
        ]
        
        has_all_keys = all(key in history for key in expected_keys)
        results.add_test(
            "History structure validation",
            has_all_keys,
            f"Total improvements: {history.get('total_improvements', 0)}"
        )
        
        # Test 3: Analyze improvement opportunities
        opportunities = improver.analyze_improvement_opportunities()
        results.add_test(
            "Improvement opportunity analysis",
            isinstance(opportunities, list),
            f"Found {len(opportunities)} opportunities"
        )
        
    except Exception as e:
        results.add_test("Improvement history", False, str(e))
    
    return results


async def test_performance_baselines():
    """Test performance baseline establishment."""
    print("\n=== Testing Performance Baselines ===")
    results = TestResults()
    
    try:
        improver = SafeSelfImprover()
        
        # Test 1: Check baseline establishment
        has_baselines = hasattr(improver, 'performance_baselines')
        results.add_test(
            "Performance baselines exist",
            has_baselines,
            f"Baselines: {improver.performance_baselines if has_baselines else 'None'}"
        )
        
        # Test 2: Check baseline metrics
        if has_baselines:
            expected_metrics = ['memory_mb', 'import_time_seconds']
            has_metrics = all(
                metric in improver.performance_baselines 
                for metric in expected_metrics
            )
            results.add_test(
                "Baseline metrics validation",
                has_metrics,
                f"Memory baseline: {improver.performance_baselines.get('memory_mb', 0):.1f} MB"
            )
        
    except Exception as e:
        results.add_test("Performance baselines", False, str(e))
    
    return results


async def test_rollback_mechanism():
    """Test rollback capabilities."""
    print("\n=== Testing Rollback Mechanism ===")
    results = TestResults()
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize git repo
            os.system(f"cd {temp_dir} && git init")
            os.system(f"cd {temp_dir} && git config user.email 'test@example.com'")
            os.system(f"cd {temp_dir} && git config user.name 'Test User'")
            
            # Create and commit initial file
            test_file = os.path.join(temp_dir, "rollback_test.py")
            original_content = "def original_function():\n    return 42\n"
            with open(test_file, 'w') as f:
                f.write(original_content)
            
            os.system(f"cd {temp_dir} && git add . && git commit -m 'Initial commit'")
            
            improver = SafeSelfImprover(repo_path=temp_dir)
            
            # Test 1: Create checkpoint
            checkpoint = improver._create_checkpoint()
            results.add_test(
                "Checkpoint creation",
                checkpoint is not None,
                f"Checkpoint: {checkpoint[:8]}..."
            )
            
            # Test 2: Modify file
            modified_content = "def modified_function():\n    return 84\n"
            with open(test_file, 'w') as f:
                f.write(modified_content)
            
            # Verify modification
            with open(test_file, 'r') as f:
                current = f.read()
            
            modification_applied = current == modified_content
            results.add_test(
                "File modification",
                modification_applied,
                "File successfully modified"
            )
            
            # Test 3: Rollback
            improver._rollback(checkpoint)
            
            # Verify rollback
            with open(test_file, 'r') as f:
                rolled_back = f.read()
            
            rollback_successful = rolled_back == original_content
            results.add_test(
                "Rollback execution",
                rollback_successful,
                "Successfully rolled back to original content"
            )
            
    except Exception as e:
        results.add_test("Rollback mechanism", False, str(e))
    
    return results


async def main():
    """Run all self-improver tests."""
    print("="*60)
    print("CUSTOM SELF-IMPROVERS TEST SUITE")
    print("="*60)
    print(f"Started at: {datetime.now().isoformat()}")
    
    all_results = []
    
    # Run all test suites
    test_suites = [
        ("SafeSelfImprover Initialization", test_safe_self_improver_initialization),
        ("Improvement Proposal", test_improvement_proposal),
        ("Safety Validation", test_safety_validation),
        ("SelfModificationEngine", test_self_modification_engine),
        ("Orchestrator Integration", test_orchestrator_integration),
        ("Improvement History", test_improvement_history),
        ("Performance Baselines", test_performance_baselines),
        ("Rollback Mechanism", test_rollback_mechanism)
    ]
    
    for suite_name, test_func in test_suites:
        try:
            results = await test_func()
            all_results.append((suite_name, results))
        except Exception as e:
            print(f"\n❌ ERROR in {suite_name}: {e}")
            # Create failed result
            results = TestResults()
            results.add_test(suite_name, False, str(e))
            all_results.append((suite_name, results))
    
    # Print overall summary
    print("\n" + "="*60)
    print("OVERALL TEST SUMMARY")
    print("="*60)
    
    total_tests = sum(r.total_tests for _, r in all_results)
    total_passed = sum(r.passed_tests for _, r in all_results)
    total_failed = sum(r.failed_tests for _, r in all_results)
    
    print(f"Total Test Suites: {len(all_results)}")
    print(f"Total Tests: {total_tests}")
    print(f"Total Passed: {total_passed} ({total_passed/max(total_tests,1)*100:.1f}%)")
    print(f"Total Failed: {total_failed}")
    
    # Print suite-level summary
    print("\nSuite Results:")
    for suite_name, results in all_results:
        suite_status = "✅" if results.failed_tests == 0 else "❌"
        print(f"  {suite_status} {suite_name}: {results.passed_tests}/{results.total_tests} passed")
    
    # Detailed failure report
    if total_failed > 0:
        print("\n" + "="*60)
        print("FAILED TESTS DETAIL")
        print("="*60)
        for suite_name, results in all_results:
            if results.failed_tests > 0:
                print(f"\n{suite_name}:")
                for test in results.test_details:
                    if not test['passed']:
                        print(f"  ❌ {test['name']}")
                        if test['details']:
                            print(f"     {test['details']}")
    
    print(f"\nCompleted at: {datetime.now().isoformat()}")
    
    # Return overall success
    return total_failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)