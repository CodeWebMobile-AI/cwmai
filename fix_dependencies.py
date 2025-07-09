#!/usr/bin/env python3
"""
Fix missing dependencies for the evolutionary system.
"""

import subprocess
import sys

def install_missing_dependencies():
    """Install missing Python packages."""
    print("🔧 Installing missing dependencies...")
    
    missing_packages = [
        'pyyaml',  # provides the 'yaml' module
        'pydantic',
    ]
    
    for package in missing_packages:
        print(f"\n📦 Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--user', package
            ])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def verify_imports():
    """Verify that the imports now work."""
    print("\n🔍 Verifying imports...")
    
    try:
        import yaml
        print("✅ yaml module imported successfully")
    except ImportError:
        print("❌ yaml module still not available")
        return False
    
    try:
        import pydantic
        print("✅ pydantic module imported successfully")
    except ImportError:
        print("❌ pydantic module still not available")
        return False
    
    return True

def main():
    """Install dependencies and verify."""
    print("🚀 Fixing dependency issues for evolutionary system")
    print("=" * 60)
    
    # Install missing packages
    if not install_missing_dependencies():
        print("\n❌ Failed to install some dependencies")
        return False
    
    # Verify imports work
    if not verify_imports():
        print("\n❌ Dependencies installed but imports still failing")
        print("You may need to restart your Python session")
        return False
    
    print("\n✅ All dependencies fixed!")
    print("You can now run:")
    print("arch -arm64 /usr/bin/python3 run_evolutionary_improvements.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)