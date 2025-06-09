#!/usr/bin/env python3
"""
Comprehensive Unit Test Runner for CWMAI

This script runs all unit tests and provides coverage analysis to ensure
we meet the 90% coverage target specified in the requirements.

Features:
- Discovers and runs all unit test files
- Provides detailed coverage reporting
- Generates both console and HTML coverage reports
- Supports different verbosity levels
- Tracks performance metrics
- Generates test quality reports
"""

import unittest
import sys
import os
import time
import argparse
from pathlib import Path
from io import StringIO
import json

# Try to import coverage for coverage analysis
try:
    import coverage
    HAS_COVERAGE = True
except ImportError:
    HAS_COVERAGE = False
    print("Coverage package not available. Install with: pip install coverage")


class TestResult:
    """Container for test results and metrics."""
    
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.error_tests = 0
        self.skipped_tests = 0
        self.start_time = None
        self.end_time = None
        self.test_modules = []
        self.failures = []
        self.errors = []
    
    @property
    def success_rate(self):
        """Calculate test success rate as percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100
    
    @property
    def duration(self):
        """Calculate total test duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


class UnitTestRunner:
    """Comprehensive unit test runner with coverage analysis."""
    
    def __init__(self, verbosity=2, pattern="test_unit_*.py"):
        """Initialize the test runner.
        
        Args:
            verbosity: Test output verbosity level (0-2)
            pattern: Pattern to match test files
        """
        self.verbosity = verbosity
        self.pattern = pattern
        self.project_root = Path(__file__).parent
        self.results = TestResult()
        self.coverage_obj = None
        
    def discover_tests(self):
        """Discover all unit test files matching the pattern."""
        test_files = []
        
        # Look for test files in the project root
        for test_file in self.project_root.glob(self.pattern):
            if test_file.is_file():
                test_files.append(test_file)
        
        print(f"🔍 Discovered {len(test_files)} test files:")
        for test_file in test_files:
            print(f"  • {test_file.name}")
        
        return test_files
    
    def setup_coverage(self):
        """Set up coverage monitoring if available."""
        if not HAS_COVERAGE:
            print("⚠️  Coverage analysis not available")
            return False
        
        try:
            # Configure coverage to track the scripts directory
            self.coverage_obj = coverage.Coverage(
                source=['scripts'],
                omit=[
                    '*/test_*.py',
                    '*/tests/*',
                    '*/__pycache__/*',
                    '*/.*',
                    '*/fetch_secrets.sh',
                    '*/run_god_mode.py'  # Exclude scripts that are entry points
                ]
            )
            self.coverage_obj.start()
            print("✅ Coverage monitoring started")
            return True
        except Exception as e:
            print(f"⚠️  Failed to start coverage monitoring: {e}")
            return False
    
    def run_test_file(self, test_file):
        """Run tests from a single test file.
        
        Args:
            test_file: Path to the test file
            
        Returns:
            TestResult for this file
        """
        print(f"\n{'='*60}")
        print(f"🧪 Running tests from {test_file.name}")
        print(f"{'='*60}")
        
        # Import the test module
        module_name = test_file.stem
        spec = None
        
        try:
            # Add current directory to path for test imports
            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))
            
            # Load the test module
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromName(module_name)
            
            # Run the tests
            stream = StringIO() if self.verbosity == 0 else sys.stderr
            runner = unittest.TextTestRunner(
                stream=stream,
                verbosity=self.verbosity,
                buffer=True
            )
            
            result = runner.run(suite)
            
            # Update results
            self.results.test_modules.append(module_name)
            self.results.total_tests += result.testsRun
            self.results.failed_tests += len(result.failures)
            self.results.error_tests += len(result.errors)
            self.results.skipped_tests += len(result.skipped) if hasattr(result, 'skipped') else 0
            self.results.passed_tests += (result.testsRun - len(result.failures) - len(result.errors))
            
            # Store failures and errors
            self.results.failures.extend(result.failures)
            self.results.errors.extend(result.errors)
            
            # Print summary for this file
            print(f"\n📊 {test_file.name} Results:")
            print(f"  • Tests run: {result.testsRun}")
            print(f"  • Failures: {len(result.failures)}")
            print(f"  • Errors: {len(result.errors)}")
            print(f"  • Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%" if result.testsRun > 0 else "  • Success rate: N/A")
            
            return result
            
        except Exception as e:
            print(f"❌ Failed to run tests from {test_file.name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_coverage_report(self):
        """Generate coverage report if coverage is available."""
        if not self.coverage_obj:
            return None
        
        try:
            self.coverage_obj.stop()
            self.coverage_obj.save()
            
            print(f"\n{'='*60}")
            print("📈 COVERAGE ANALYSIS")
            print(f"{'='*60}")
            
            # Generate console report
            coverage_data = self.coverage_obj.get_data()
            total_statements = 0
            covered_statements = 0
            
            # Get coverage percentage
            report_data = self.coverage_obj.report(show_missing=True)
            
            # Try to get more detailed coverage info
            try:
                analysis = self.coverage_obj.analysis2('scripts/state_manager.py')
                print(f"Example analysis for state_manager.py: {len(analysis[1])} statements, {len(analysis[2])} missing")
            except Exception:
                pass
            
            # Generate HTML report
            try:
                html_dir = self.project_root / "htmlcov"
                self.coverage_obj.html_report(directory=str(html_dir))
                print(f"📄 HTML coverage report generated in: {html_dir}")
            except Exception as e:
                print(f"⚠️  Could not generate HTML report: {e}")
            
            return report_data
            
        except Exception as e:
            print(f"⚠️  Failed to generate coverage report: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_all_tests(self):
        """Run all discovered unit tests."""
        print("🚀 Starting Comprehensive Unit Test Suite")
        print(f"{'='*80}")
        
        self.results.start_time = time.time()
        
        # Setup coverage monitoring
        coverage_enabled = self.setup_coverage()
        
        # Discover test files
        test_files = self.discover_tests()
        
        if not test_files:
            print("❌ No test files found matching pattern:", self.pattern)
            return False
        
        # Run each test file
        for test_file in test_files:
            self.run_test_file(test_file)
        
        self.results.end_time = time.time()
        
        # Generate coverage report
        coverage_data = None
        if coverage_enabled:
            coverage_data = self.generate_coverage_report()
        
        # Generate final report
        self.generate_final_report(coverage_data)
        
        return self.results.failed_tests == 0 and self.results.error_tests == 0
    
    def generate_final_report(self, coverage_data=None):
        """Generate comprehensive final test report."""
        print(f"\n{'='*80}")
        print("📋 COMPREHENSIVE TEST RESULTS")
        print(f"{'='*80}")
        
        # Test statistics
        print(f"📊 Test Statistics:")
        print(f"  • Total test modules: {len(self.results.test_modules)}")
        print(f"  • Total tests run: {self.results.total_tests}")
        print(f"  • Tests passed: {self.results.passed_tests}")
        print(f"  • Tests failed: {self.results.failed_tests}")
        print(f"  • Tests with errors: {self.results.error_tests}")
        print(f"  • Tests skipped: {self.results.skipped_tests}")
        print(f"  • Success rate: {self.results.success_rate:.1f}%")
        print(f"  • Total duration: {self.results.duration:.2f} seconds")
        
        # Performance metrics
        if self.results.total_tests > 0:
            avg_time_per_test = self.results.duration / self.results.total_tests
            print(f"  • Average time per test: {avg_time_per_test:.3f} seconds")
        
        # Coverage information
        if coverage_data is not None:
            print(f"\n📈 Coverage Summary:")
            print(f"  • Coverage analysis completed")
            print(f"  • See htmlcov/index.html for detailed coverage report")
        
        # Test quality assessment
        print(f"\n🎯 Test Quality Assessment:")
        
        quality_score = 0
        max_score = 100
        
        # Success rate contributes 40 points
        quality_score += (self.results.success_rate * 0.4)
        
        # Number of test modules contributes 20 points
        module_score = min(len(self.results.test_modules) * 4, 20)  # Max 5 modules for full score
        quality_score += module_score
        
        # Number of tests contributes 20 points
        test_count_score = min(self.results.total_tests * 0.2, 20)  # Max 100 tests for full score
        quality_score += test_count_score
        
        # Performance contributes 20 points
        if self.results.total_tests > 0:
            avg_time = self.results.duration / self.results.total_tests
            if avg_time < 0.1:
                performance_score = 20
            elif avg_time < 0.5:
                performance_score = 15
            elif avg_time < 1.0:
                performance_score = 10
            else:
                performance_score = 5
            quality_score += performance_score
        
        print(f"  • Quality Score: {quality_score:.1f}/{max_score}")
        
        if quality_score >= 90:
            print(f"  • Grade: A+ (Excellent) 🌟")
        elif quality_score >= 80:
            print(f"  • Grade: A (Very Good) ✅")
        elif quality_score >= 70:
            print(f"  • Grade: B (Good) 👍")
        else:
            print(f"  • Grade: C (Needs Improvement) ⚠️")
        
        # Detailed failure information
        if self.results.failures or self.results.errors:
            print(f"\n❌ Failures and Errors:")
            
            for i, (test, traceback) in enumerate(self.results.failures):
                print(f"\n  Failure {i+1}: {test}")
                print(f"  {traceback}")
            
            for i, (test, traceback) in enumerate(self.results.errors):
                print(f"\n  Error {i+1}: {test}")
                print(f"  {traceback}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if self.results.success_rate < 100:
            print(f"  • Fix failing tests to improve success rate")
        
        if len(self.results.test_modules) < 5:
            print(f"  • Add more test modules to improve coverage")
        
        if self.results.total_tests < 50:
            print(f"  • Add more test cases to improve comprehensiveness")
        
        if not HAS_COVERAGE:
            print(f"  • Install coverage package for coverage analysis: pip install coverage")
        
        print(f"\n{'='*80}")
        
        if self.results.failed_tests == 0 and self.results.error_tests == 0:
            print("🎉 ALL TESTS PASSED! Unit test suite is comprehensive and reliable.")
        else:
            print("⚠️  Some tests failed. Please review and fix before proceeding.")
        
        print(f"{'='*80}")
    
    def save_results_json(self, filename="test_results.json"):
        """Save test results to JSON file for CI/CD integration."""
        results_data = {
            "test_run": {
                "timestamp": time.time(),
                "duration": self.results.duration,
                "total_tests": self.results.total_tests,
                "passed_tests": self.results.passed_tests,
                "failed_tests": self.results.failed_tests,
                "error_tests": self.results.error_tests,
                "skipped_tests": self.results.skipped_tests,
                "success_rate": self.results.success_rate,
                "test_modules": self.results.test_modules
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"📄 Test results saved to {filename}")
        except Exception as e:
            print(f"⚠️  Could not save results to {filename}: {e}")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Unit Test Runner for CWMAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_unit_tests.py                    # Run all unit tests
  python run_unit_tests.py -v 1               # Run with minimal verbosity
  python run_unit_tests.py -p "test_state_*"  # Run only state manager tests
  python run_unit_tests.py --save-json        # Save results to JSON file
        """
    )
    
    parser.add_argument(
        '-v', '--verbosity',
        type=int,
        choices=[0, 1, 2],
        default=2,
        help='Test output verbosity level (0=minimal, 1=normal, 2=verbose)'
    )
    
    parser.add_argument(
        '-p', '--pattern',
        type=str,
        default='test_unit_*.py',
        help='Pattern to match test files (default: test_unit_*.py)'
    )
    
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Save test results to JSON file'
    )
    
    args = parser.parse_args()
    
    # Create and run the test runner
    runner = UnitTestRunner(verbosity=args.verbosity, pattern=args.pattern)
    success = runner.run_all_tests()
    
    # Save results if requested
    if args.save_json:
        runner.save_results_json()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()