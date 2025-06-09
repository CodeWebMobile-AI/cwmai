#!/usr/bin/env python3
"""
Test runner script for CWMAI project.

Runs comprehensive unit tests with coverage reporting and detailed output.
Supports different test categories and reporting formats.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def install_test_dependencies():
    """Install test dependencies if not already installed."""
    print("📦 Installing test dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Test dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def run_tests(test_type="all", verbose=False, coverage=True, html_report=False):
    """Run tests with specified configuration."""
    
    # Base pytest command
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add test selection
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "security":
        cmd.extend(["-m", "security"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    # "all" runs everything
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add coverage options
    if coverage:
        cmd.extend([
            "--cov=scripts",
            "--cov-report=term-missing",
            "--cov-fail-under=90"
        ])
        
        if html_report:
            cmd.append("--cov-report=html:htmlcov")
    
    # Add test path
    cmd.append("tests/")
    
    print(f"🧪 Running {test_type} tests...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def generate_coverage_report():
    """Generate detailed coverage report."""
    print("\n📊 Generating detailed coverage report...")
    
    try:
        # Generate HTML report
        subprocess.check_call([
            sys.executable, "-m", "coverage", "html", "--directory=htmlcov"
        ])
        
        # Generate console report
        result = subprocess.run([
            sys.executable, "-m", "coverage", "report", "--show-missing"
        ], capture_output=True, text=True)
        
        print("\n📈 Coverage Summary:")
        print(result.stdout)
        
        html_path = Path("htmlcov/index.html").absolute()
        if html_path.exists():
            print(f"\n🌐 HTML Coverage Report: {html_path}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate coverage report: {e}")
        return False


def check_test_quality():
    """Check test quality and completeness."""
    print("\n🔍 Checking test quality...")
    
    test_files = list(Path("tests").glob("test_*.py"))
    script_files = list(Path("scripts").glob("*.py"))
    
    print(f"📝 Test files found: {len(test_files)}")
    print(f"📄 Script files found: {len(script_files)}")
    
    # Check test coverage
    tested_modules = set()
    for test_file in test_files:
        module_name = test_file.name.replace("test_", "").replace(".py", "")
        if module_name != "conftest":
            tested_modules.add(module_name)
    
    untested_modules = []
    for script_file in script_files:
        module_name = script_file.name.replace(".py", "")
        if module_name not in ["__init__", "fetch_secrets"] and module_name not in tested_modules:
            untested_modules.append(module_name)
    
    if untested_modules:
        print(f"⚠️  Modules without tests: {untested_modules}")
    else:
        print("✅ All modules have corresponding test files")
    
    return len(untested_modules) == 0


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run CWMAI unit tests")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "security", "fast"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--no-coverage", 
        action="store_true",
        help="Skip coverage reporting"
    )
    parser.add_argument(
        "--html", 
        action="store_true",
        help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies first"
    )
    parser.add_argument(
        "--check-quality",
        action="store_true", 
        help="Check test quality and completeness"
    )
    
    args = parser.parse_args()
    
    print("🚀 CWMAI Test Runner")
    print("=" * 50)
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_test_dependencies():
            sys.exit(1)
    
    # Check test quality if requested
    if args.check_quality:
        check_test_quality()
        print()
    
    # Run tests
    success = run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=not args.no_coverage,
        html_report=args.html
    )
    
    # Generate detailed coverage report if requested
    if args.html and not args.no_coverage:
        generate_coverage_report()
    
    if success:
        print("\n🎉 All tests passed!")
        if not args.no_coverage:
            print("📊 Coverage target achieved (≥90%)")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()