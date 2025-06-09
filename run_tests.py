#!/usr/bin/env python3
"""
Test Runner for CWMAI System

Comprehensive test execution script with multiple testing modes:
- Unit tests
- Integration tests
- Performance benchmarks
- Security tests
- Coverage reporting
- CI/CD integration
"""

import sys
import os
import subprocess
import argparse
import time
from datetime import datetime
from pathlib import Path


class TestRunner:
    """Comprehensive test runner for CWMAI system."""
    
    def __init__(self):
        """Initialize test runner."""
        self.project_root = Path(__file__).parent
        self.test_results = {}
        
    def run_command(self, command: str, description: str = None) -> tuple:
        """Run a command and capture output."""
        if description:
            print(f"\n🔄 {description}")
            print("-" * 50)
        
        print(f"Running: {command}")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ Success ({duration:.2f}s)")
                if result.stdout.strip():
                    print(result.stdout)
            else:
                print(f"❌ Failed ({duration:.2f}s)")
                print(f"Error: {result.stderr}")
                if result.stdout.strip():
                    print(f"Output: {result.stdout}")
            
            return result.returncode == 0, result.stdout, result.stderr, duration
            
        except subprocess.TimeoutExpired:
            print("❌ Command timed out after 10 minutes")
            return False, "", "Timeout", time.time() - start_time
        except Exception as e:
            print(f"❌ Command failed with exception: {e}")
            return False, "", str(e), time.time() - start_time

    def check_dependencies(self) -> bool:
        """Check if testing dependencies are installed."""
        print("🔍 Checking testing dependencies...")
        
        required_packages = [
            'pytest',
            'pytest-cov',
            'pytest-asyncio',
            'coverage'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            success, _, _, _ = self.run_command(f"python -c 'import {package.replace(\"-\", \"_\")}'")
            if not success:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            print("Installing missing packages...")
            
            install_cmd = f"pip install {' '.join(missing_packages)}"
            success, _, _, _ = self.run_command(install_cmd, "Installing testing dependencies")
            
            if not success:
                print("❌ Failed to install testing dependencies")
                return False
        
        print("✅ All testing dependencies are available")
        return True

    def run_unit_tests(self) -> bool:
        """Run unit tests with coverage."""
        print("\n🧪 Running Unit Tests")
        print("=" * 50)
        
        # Run pytest with unit test markers
        cmd = (
            "python -m pytest "
            "test_comprehensive_suite.py::TestAIBrainFactory "
            "test_comprehensive_suite.py::TestIntelligentAIBrain "
            "test_comprehensive_suite.py::TestTaskManager "
            "test_comprehensive_suite.py::TestStateManager "
            "test_comprehensive_suite.py::TestDynamicSwarmAgent "
            "-v --cov=scripts --cov-report=term-missing"
        )
        
        success, stdout, stderr, duration = self.run_command(cmd)
        
        self.test_results['unit_tests'] = {
            'success': success,
            'duration': duration,
            'output': stdout
        }
        
        return success

    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        print("\n🔗 Running Integration Tests")
        print("=" * 50)
        
        cmd = (
            "python -m pytest "
            "test_comprehensive_suite.py::TestIntegrationWorkflows "
            "-v --tb=short"
        )
        
        success, stdout, stderr, duration = self.run_command(cmd)
        
        self.test_results['integration_tests'] = {
            'success': success,
            'duration': duration,
            'output': stdout
        }
        
        return success

    def run_security_tests(self) -> bool:
        """Run security vulnerability tests."""
        print("\n🔒 Running Security Tests")
        print("=" * 50)
        
        cmd = (
            "python -m pytest "
            "test_comprehensive_suite.py::TestSecurityVulnerabilities "
            "-v --tb=short"
        )
        
        success, stdout, stderr, duration = self.run_command(cmd)
        
        self.test_results['security_tests'] = {
            'success': success,
            'duration': duration,
            'output': stdout
        }
        
        return success

    def run_edge_case_tests(self) -> bool:
        """Run edge case tests."""
        print("\n🎯 Running Edge Case Tests")
        print("=" * 50)
        
        cmd = (
            "python -m pytest "
            "test_comprehensive_suite.py::TestEdgeCases "
            "-v --tb=short"
        )
        
        success, stdout, stderr, duration = self.run_command(cmd)
        
        self.test_results['edge_case_tests'] = {
            'success': success,
            'duration': duration,
            'output': stdout
        }
        
        return success

    def run_performance_benchmarks(self) -> bool:
        """Run performance benchmarks."""
        print("\n⚡ Running Performance Benchmarks")
        print("=" * 50)
        
        cmd = "python test_performance_benchmarks.py"
        
        success, stdout, stderr, duration = self.run_command(cmd)
        
        self.test_results['performance_benchmarks'] = {
            'success': success,
            'duration': duration,
            'output': stdout
        }
        
        return success

    def run_coverage_analysis(self) -> bool:
        """Run comprehensive coverage analysis."""
        print("\n📊 Running Coverage Analysis")
        print("=" * 50)
        
        # Run all tests with coverage
        cmd = (
            "python -m pytest "
            "test_comprehensive_suite.py "
            "--cov=scripts "
            "--cov-report=html:htmlcov "
            "--cov-report=xml:coverage.xml "
            "--cov-report=term-missing "
            "--cov-fail-under=80 "
            "--cov-branch"
        )
        
        success, stdout, stderr, duration = self.run_command(cmd)
        
        # Generate coverage badge if coverage is good
        if success:
            print("✅ Coverage analysis completed")
            print("📁 HTML report available at: htmlcov/index.html")
            print("📄 XML report available at: coverage.xml")
        
        self.test_results['coverage_analysis'] = {
            'success': success,
            'duration': duration,
            'output': stdout
        }
        
        return success

    def run_factory_tests(self) -> bool:
        """Run existing factory pattern tests."""
        print("\n🏭 Running Factory Pattern Tests")
        print("=" * 50)
        
        cmd = "python test_factory_pattern.py"
        
        success, stdout, stderr, duration = self.run_command(cmd)
        
        self.test_results['factory_tests'] = {
            'success': success,
            'duration': duration,
            'output': stdout
        }
        
        return success

    def run_dynamic_system_tests(self) -> bool:
        """Run existing dynamic system tests."""
        print("\n🚀 Running Dynamic System Tests")
        print("=" * 50)
        
        cmd = "python test_dynamic_system.py"
        
        success, stdout, stderr, duration = self.run_command(cmd)
        
        self.test_results['dynamic_system_tests'] = {
            'success': success,
            'duration': duration,
            'output': stdout
        }
        
        return success

    def generate_test_report(self) -> str:
        """Generate comprehensive test report."""
        report = []
        report.append("CWMAI COMPREHENSIVE TEST REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        total_suites = len(self.test_results)
        successful_suites = sum(1 for result in self.test_results.values() if result['success'])
        total_duration = sum(result['duration'] for result in self.test_results.values())
        
        report.append("📋 TEST SUMMARY")
        report.append("-" * 40)
        report.append(f"Total test suites: {total_suites}")
        report.append(f"Successful suites: {successful_suites}")
        report.append(f"Failed suites: {total_suites - successful_suites}")
        report.append(f"Success rate: {(successful_suites / total_suites * 100):.1f}%")
        report.append(f"Total execution time: {total_duration:.2f} seconds")
        report.append("")
        
        # Detailed results
        for suite_name, result in self.test_results.items():
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            report.append(f"📊 {suite_name.upper().replace('_', ' ')}")
            report.append(f"Status: {status}")
            report.append(f"Duration: {result['duration']:.2f}s")
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        
        if successful_suites == total_suites:
            report.append("🎉 All tests passed! System is ready for production.")
            report.append("✅ Test coverage meets 80% requirement")
            report.append("✅ Security tests passed")
            report.append("✅ Performance benchmarks completed")
        else:
            failed_suites = [name for name, result in self.test_results.items() if not result['success']]
            report.append(f"⚠️  Failed test suites: {', '.join(failed_suites)}")
            report.append("🔧 Review failed tests and fix issues before deployment")
        
        return "\n".join(report)

    def run_all_tests(self) -> bool:
        """Run complete test suite."""
        print("🚀 CWMAI Comprehensive Test Suite")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("")
        
        # Check dependencies first
        if not self.check_dependencies():
            return False
        
        # Run all test suites
        test_suites = [
            ('Unit Tests', self.run_unit_tests),
            ('Integration Tests', self.run_integration_tests),
            ('Security Tests', self.run_security_tests),
            ('Edge Case Tests', self.run_edge_case_tests),
            ('Performance Benchmarks', self.run_performance_benchmarks),
            ('Coverage Analysis', self.run_coverage_analysis),
            ('Factory Tests', self.run_factory_tests),
            ('Dynamic System Tests', self.run_dynamic_system_tests)
        ]
        
        overall_success = True
        
        for suite_name, test_function in test_suites:
            try:
                success = test_function()
                if not success:
                    overall_success = False
            except Exception as e:
                print(f"❌ {suite_name} failed with exception: {e}")
                overall_success = False
        
        # Generate and display report
        report = self.generate_test_report()
        print("\n" + report)
        
        # Save report to file
        report_file = self.project_root / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Test report saved to: {report_file}")
        
        return overall_success

    def run_quick_tests(self) -> bool:
        """Run quick test suite for development."""
        print("⚡ Quick Test Suite")
        print("=" * 50)
        
        # Run essential tests only
        success = True
        
        if not self.check_dependencies():
            return False
        
        success &= self.run_unit_tests()
        success &= self.run_security_tests()
        
        return success

    def run_ci_tests(self) -> bool:
        """Run tests optimized for CI/CD pipeline."""
        print("🔄 CI/CD Test Suite")
        print("=" * 50)
        
        # Set environment for CI
        os.environ['CI'] = 'true'
        os.environ['PYTHONPATH'] = str(self.project_root)
        
        # Run CI-optimized test suite
        cmd = (
            "python -m pytest "
            "test_comprehensive_suite.py "
            "--tb=short "
            "--quiet "
            "--cov=scripts "
            "--cov-report=xml "
            "--cov-fail-under=80 "
            "--maxfail=5"
        )
        
        success, stdout, stderr, duration = self.run_command(cmd, "Running CI test suite")
        
        if success:
            print("✅ CI test suite passed")
        else:
            print("❌ CI test suite failed")
            print("Check the output above for details")
        
        return success


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='CWMAI Test Runner')
    parser.add_argument(
        'mode',
        choices=['all', 'quick', 'unit', 'integration', 'security', 'performance', 'coverage', 'ci'],
        help='Test mode to run'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.mode == 'all':
        success = runner.run_all_tests()
    elif args.mode == 'quick':
        success = runner.run_quick_tests()
    elif args.mode == 'unit':
        success = runner.check_dependencies() and runner.run_unit_tests()
    elif args.mode == 'integration':
        success = runner.check_dependencies() and runner.run_integration_tests()
    elif args.mode == 'security':
        success = runner.check_dependencies() and runner.run_security_tests()
    elif args.mode == 'performance':
        success = runner.check_dependencies() and runner.run_performance_benchmarks()
    elif args.mode == 'coverage':
        success = runner.check_dependencies() and runner.run_coverage_analysis()
    elif args.mode == 'ci':
        success = runner.run_ci_tests()
    else:
        print(f"Unknown mode: {args.mode}")
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())