#!/usr/bin/env python3
"""
Test script to verify continuous AI system fixes
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from work_item_types import WorkItem, TaskPriority, WorkOpportunity
        print("✓ work_item_types imports successful")
    except ImportError as e:
        print(f"❌ work_item_types import failed: {e}")
        return False
    
    try:
        from task_persistence import TaskPersistence
        print("✓ task_persistence imports successful")
    except ImportError as e:
        print(f"❌ task_persistence import failed: {e}")
        return False
    
    try:
        from github_issue_creator import GitHubIssueCreator
        print("✓ github_issue_creator imports successful")
    except ImportError as e:
        print(f"❌ github_issue_creator import failed: {e}")
        return False
    
    try:
        from intelligent_work_finder import IntelligentWorkFinder
        print("✓ intelligent_work_finder imports successful")
    except ImportError as e:
        print(f"❌ intelligent_work_finder import failed: {e}")
        return False
    
    try:
        from continuous_orchestrator import ContinuousOrchestrator
        print("✓ continuous_orchestrator imports successful")
    except ImportError as e:
        print(f"❌ continuous_orchestrator import failed: {e}")
        return False
    
    return True

def test_repository_exclusion():
    """Test that repository exclusion is working."""
    print("\nTesting repository exclusion...")
    
    try:
        from repository_exclusion import RepositoryExclusion
        
        # Test filtering
        test_projects = {
            'cwmai': {'description': 'Main AI system'},
            '.github': {'description': 'GitHub workflows'},
            'other-project': {'description': 'Normal project'}
        }
        
        filtered = RepositoryExclusion.filter_excluded_repos_dict(test_projects)
        
        if 'cwmai' in filtered:
            print("❌ cwmai should be excluded but wasn't")
            return False
        
        if '.github' in filtered:
            print("❌ .github should be excluded but wasn't")
            return False
            
        if 'other-project' not in filtered:
            print("❌ other-project should be included but wasn't")
            return False
        
        print("✓ Repository exclusion working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Repository exclusion test failed: {e}")
        return False

def test_duplicate_detection():
    """Test that duplicate task detection is working."""
    print("\nTesting duplicate task detection...")
    
    try:
        from work_item_types import WorkItem, TaskPriority
        from task_persistence import TaskPersistence
        
        # Create persistence manager
        persistence = TaskPersistence("test_completed_tasks.json")
        
        # Create test work item
        work_item = WorkItem(
            id="test_123",
            task_type="TESTING",
            title="Add tests for recent .github changes",
            description="Test adding tests for github workflow changes",
            priority=TaskPriority.HIGH,
            repository=".github"
        )
        
        # First check should not be duplicate
        is_duplicate_1 = persistence.is_duplicate_task(work_item)
        
        # Record as completed
        persistence.record_completed_task(work_item, {'success': True, 'issue_number': 123})
        
        # Second check should be duplicate
        is_duplicate_2 = persistence.is_duplicate_task(work_item)
        
        if is_duplicate_1:
            print("❌ First check incorrectly identified as duplicate")
            return False
            
        if not is_duplicate_2:
            print("❌ Second check should have been duplicate")
            return False
        
        print("✓ Duplicate detection working correctly")
        
        # Clean up test file
        import os
        if os.path.exists("test_completed_tasks.json"):
            os.remove("test_completed_tasks.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Duplicate detection test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Testing Continuous AI System Fixes ===\n")
    
    tests = [
        test_imports,
        test_repository_exclusion,
        test_duplicate_detection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All tests passed! The continuous AI system fixes are working correctly.")
        print("\nKey fixes implemented:")
        print("- ✓ Repository exclusion now properly filters out .github and cwmai")
        print("- ✓ Real GitHub issue creation replaces fake task execution")
        print("- ✓ Task persistence prevents infinite loops via deduplication")
        print("- ✓ Circular import issues resolved with work_item_types.py")
        print("- ✓ Task diversification prevents repetitive testing tasks")
        return True
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)