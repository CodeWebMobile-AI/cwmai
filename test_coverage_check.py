#!/usr/bin/env python3
"""
Quick test coverage verification script.

This script performs a basic test run to verify our unit tests
are working correctly and can provide initial coverage estimates.
"""

import unittest
import sys
import os
from pathlib import Path
import importlib.util

def test_import_capability():
    """Test that we can import our test modules."""
    print("🔍 Testing import capability...")
    
    test_files = [
        "test_unit_state_manager.py",
        "test_unit_context_gatherer.py", 
        "test_unit_task_manager.py",
        "test_unit_ai_brain_factory.py",
        "test_unit_environment_validator.py",
        "test_unit_http_ai_client.py"
    ]
    
    success_count = 0
    
    for test_file in test_files:
        try:
            # Try to import the test module
            file_path = Path(__file__).parent / test_file
            if file_path.exists():
                spec = importlib.util.spec_from_file_location(test_file[:-3], file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"  ✅ {test_file} - Import successful")
                success_count += 1
            else:
                print(f"  ❌ {test_file} - File not found")
        except Exception as e:
            print(f"  ❌ {test_file} - Import failed: {e}")
    
    print(f"\n📊 Import Results: {success_count}/{len(test_files)} successful")
    return success_count == len(test_files)

def run_sample_tests():
    """Run a sample of tests to verify functionality."""
    print("\n🧪 Running sample tests...")
    
    try:
        # Add current directory to path
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        # Import and run a simple test
        from test_unit_state_manager import TestStateManagerInitialization
        
        # Create a test suite with just a few tests
        suite = unittest.TestSuite()
        suite.addTest(TestStateManagerInitialization('test_init_with_default_parameters'))
        suite.addTest(TestStateManagerInitialization('test_init_with_custom_parameters'))
        
        # Run the tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        print(f"\n📊 Sample Test Results:")
        print(f"  • Tests run: {result.testsRun}")
        print(f"  • Failures: {len(result.failures)}")
        print(f"  • Errors: {len(result.errors)}")
        
        return len(result.failures) == 0 and len(result.errors) == 0
        
    except Exception as e:
        print(f"❌ Failed to run sample tests: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_test_structure():
    """Check the structure and quality of our test files."""
    print("\n🔍 Checking test structure...")
    
    test_files = list(Path(__file__).parent.glob("test_unit_*.py"))
    
    total_test_classes = 0
    total_test_methods = 0
    
    for test_file in test_files:
        try:
            with open(test_file, 'r') as f:
                content = f.read()
                
            # Count test classes and methods
            test_classes = content.count("class Test")
            test_methods = content.count("def test_")
            
            total_test_classes += test_classes
            total_test_methods += test_methods
            
            print(f"  📄 {test_file.name}:")
            print(f"     • Test classes: {test_classes}")
            print(f"     • Test methods: {test_methods}")
            
        except Exception as e:
            print(f"  ❌ Error reading {test_file.name}: {e}")
    
    print(f"\n📊 Total Test Structure:")
    print(f"  • Test files: {len(test_files)}")
    print(f"  • Test classes: {total_test_classes}")
    print(f"  • Test methods: {total_test_methods}")
    
    # Quality assessment
    print(f"\n🎯 Quality Assessment:")
    if len(test_files) >= 5:
        print(f"  ✅ Good test file coverage ({len(test_files)} files)")
    else:
        print(f"  ⚠️  Limited test file coverage ({len(test_files)} files)")
    
    if total_test_methods >= 100:
        print(f"  ✅ Excellent test method coverage ({total_test_methods} methods)")
    elif total_test_methods >= 50:
        print(f"  ✅ Good test method coverage ({total_test_methods} methods)")
    else:
        print(f"  ⚠️  Limited test method coverage ({total_test_methods} methods)")
    
    avg_methods_per_file = total_test_methods / len(test_files) if test_files else 0
    if avg_methods_per_file >= 20:
        print(f"  ✅ Comprehensive per-file coverage ({avg_methods_per_file:.1f} methods/file)")
    else:
        print(f"  ⚠️  Light per-file coverage ({avg_methods_per_file:.1f} methods/file)")
    
    return len(test_files) >= 5 and total_test_methods >= 50

def estimate_coverage():
    """Provide an estimate of code coverage based on test structure."""
    print("\n📈 Coverage Estimation...")
    
    # Count Python files in scripts directory
    scripts_dir = Path(__file__).parent / "scripts"
    if not scripts_dir.exists():
        print("  ❌ Scripts directory not found")
        return False
    
    python_files = list(scripts_dir.glob("*.py"))
    python_files = [f for f in python_files if not f.name.startswith('__')]
    
    # Count test files
    test_files = list(Path(__file__).parent.glob("test_unit_*.py"))
    
    print(f"  📂 Source files in scripts/: {len(python_files)}")
    print(f"  🧪 Unit test files: {len(test_files)}")
    
    # Basic coverage estimation
    coverage_ratio = len(test_files) / len(python_files) if python_files else 0
    estimated_coverage = min(coverage_ratio * 100, 90)  # Cap at 90% for estimation
    
    print(f"  📊 Estimated coverage: {estimated_coverage:.1f}%")
    
    if estimated_coverage >= 80:
        print(f"  🎯 Coverage target likely achieved!")
    else:
        print(f"  ⚠️  More tests needed to reach 90% target")
    
    return estimated_coverage >= 80

def main():
    """Main test verification function."""
    print("🚀 CWMAI Unit Test Coverage Verification")
    print("="*60)
    
    all_checks_passed = True
    
    # Run all verification checks
    checks = [
        ("Import Capability", test_import_capability),
        ("Sample Test Execution", run_sample_tests),
        ("Test Structure Analysis", check_test_structure),
        ("Coverage Estimation", estimate_coverage)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        print(f"\n{'='*60}")
        print(f"🔍 {check_name}")
        print("="*60)
        
        try:
            result = check_func()
            results[check_name] = result
            if not result:
                all_checks_passed = False
        except Exception as e:
            print(f"❌ {check_name} failed with exception: {e}")
            results[check_name] = False
            all_checks_passed = False
    
    # Final summary
    print(f"\n{'='*60}")
    print("📋 VERIFICATION SUMMARY")
    print("="*60)
    
    for check_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {check_name}: {status}")
    
    print(f"\n🎯 Overall Result: {'✅ ALL CHECKS PASSED' if all_checks_passed else '❌ SOME CHECKS FAILED'}")
    
    if all_checks_passed:
        print("\n🎉 Unit test suite appears to be comprehensive and ready!")
        print("   Run 'python run_unit_tests.py' for full test execution with coverage analysis.")
    else:
        print("\n⚠️  Some verification checks failed. Please review the issues above.")
    
    print("="*60)
    
    return all_checks_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)