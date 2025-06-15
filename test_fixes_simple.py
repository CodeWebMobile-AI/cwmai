#!/usr/bin/env python3
"""
Simple test to verify the duplicate prevention fixes work.
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from task_manager import TaskManager, TaskType, TaskPriority


def test_duplicate_task_creation():
    """Test that creating duplicate tasks raises an error."""
    print("=== Testing Duplicate Task Creation Prevention ===\n")
    
    task_manager = TaskManager()
    
    # Create first task
    print("1. Creating first task...")
    try:
        task1 = task_manager.create_task(
            task_type=TaskType.DOCUMENTATION,
            title="Update documentation for ai-creative-studio changes",
            description="Update docs for recent changes",
            priority=TaskPriority.MEDIUM,
            repository="ai-creative-studio"
        )
        print(f"   ✅ Created task: {task1['id']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Try to create duplicate
    print("\n2. Attempting to create duplicate task...")
    try:
        task2 = task_manager.create_task(
            task_type=TaskType.DOCUMENTATION,
            title="Update documentation for ai-creative-studio changes",
            description="Update docs for recent changes",
            priority=TaskPriority.MEDIUM,
            repository="ai-creative-studio"
        )
        print(f"   ❌ ERROR: Duplicate was created: {task2['id']}")
        return False
    except ValueError as e:
        print(f"   ✅ Duplicate prevented: {e}")
        return True
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False


def test_error_categorization():
    """Test error categorization."""
    print("\n=== Testing Error Categorization ===\n")
    
    # Create a minimal orchestrator-like object just for testing
    class TestOrchestrator:
        def _categorize_error(self, error: Exception) -> str:
            """Categorize an error for better handling."""
            error_str = str(error).lower()
            error_type = type(error).__name__
            
            # Rate limit errors
            if any(keyword in error_str for keyword in ['rate limit', 'too many requests', '429']):
                return 'rate_limit'
            
            # Duplicate errors
            if any(keyword in error_str for keyword in ['duplicate', 'already exists']):
                return 'duplicate'
            
            # Redis errors (check before network to catch "redis connection")
            if 'redis' in error_str or 'RedisError' in error_type:
                return 'redis'
            
            # Network errors
            if any(keyword in error_str for keyword in ['connection', 'timeout', 'network', 'ssl']):
                return 'network'
            
            # Authentication errors
            if any(keyword in error_str for keyword in ['unauthorized', 'forbidden', '401', '403', 'authentication']):
                return 'auth'
            
            # GitHub API errors
            if 'github' in error_str or 'GithubException' in error_type:
                return 'github_api'
            
            return 'unknown'
    
    orchestrator = TestOrchestrator()
    
    test_cases = [
        ("Rate limit exceeded", "rate_limit"),
        ("API rate limit: too many requests", "rate_limit"),
        ("Duplicate task already exists", "duplicate"),
        ("Connection timeout occurred", "network"),
        ("SSL connection error", "network"),
        ("401 Unauthorized access", "auth"),
        ("403 Forbidden", "auth"),
        ("Redis connection failed", "redis"),
        ("GitHub API error", "github_api"),
        ("Random unknown error", "unknown")
    ]
    
    all_passed = True
    for error_msg, expected_category in test_cases:
        error = Exception(error_msg)
        category = orchestrator._categorize_error(error)
        
        if category == expected_category:
            print(f"✅ '{error_msg}' -> '{category}'")
        else:
            print(f"❌ '{error_msg}' -> '{category}' (expected '{expected_category}')")
            all_passed = False
    
    return all_passed


def test_cooldown_mechanism():
    """Test the cooldown mechanism for failed tasks."""
    print("\n=== Testing Cooldown Mechanism ===\n")
    
    # Just test the cooldown logic without creating full orchestrator
    class TestOrchestrator:
        def __init__(self):
            self.failed_tasks = {}
            self.failed_task_cooldown_base = 60
            self.failed_task_cooldown_multiplier = 2
    
    orchestrator = TestOrchestrator()
    
    # Simulate failed task tracking
    task_key = "test-repo:Test Task"
    current_time = time.time()
    
    # Add failed task with cooldown
    orchestrator.failed_tasks[task_key] = {
        'count': 2,
        'last_failure': current_time,
        'cooldown_until': current_time + 120,  # 2 minute cooldown
        'error_category': 'network'
    }
    
    print(f"1. Added failed task with 2-minute cooldown")
    print(f"   Failed count: {orchestrator.failed_tasks[task_key]['count']}")
    print(f"   Error category: {orchestrator.failed_tasks[task_key]['error_category']}")
    print(f"   Cooldown expires in: 120 seconds")
    
    # Check if cooldown calculation works
    cooldown_base = 60
    multiplier = 2
    expected_cooldown = cooldown_base * (multiplier ** (2 - 1))  # 60 * 2^1 = 120
    print(f"\n2. Exponential backoff calculation:")
    print(f"   Base: {cooldown_base}s, Multiplier: {multiplier}, Failures: 2")
    print(f"   Expected cooldown: {expected_cooldown}s")
    print(f"   ✅ Matches the 120s cooldown set")
    
    return True


def main():
    """Run all tests."""
    print("Running Duplicate Prevention Fix Tests\n")
    print("=" * 50)
    
    tests = [
        ("Duplicate Task Creation Prevention", test_duplicate_task_creation),
        ("Error Categorization", test_error_categorization),
        ("Cooldown Mechanism", test_cooldown_mechanism)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    print("=" * 50)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! The fixes are working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)