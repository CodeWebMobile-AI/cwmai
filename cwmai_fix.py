#!/usr/bin/env python3
"""
Fix for architecture issues - reinstalls packages for correct architecture
"""

import subprocess
import sys
import os

def fix_architecture_issues():
    """Fix architecture issues with installed packages."""
    print("Fixing architecture issues...")
    
    # List of packages that commonly have architecture issues
    problem_packages = ['psutil', 'numpy', 'pandas', 'scikit-learn']
    
    for package in problem_packages:
        print(f"\nReinstalling {package} for correct architecture...")
        try:
            # Uninstall
            subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', package], 
                         capture_output=True)
            
            # Reinstall with no binary cache
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 
                                   '--force-reinstall', package], 
                                 capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {package} reinstalled successfully")
            else:
                print(f"⚠️  {package} reinstallation had issues: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error with {package}: {e}")
    
    print("\n✅ Architecture fix complete!")
    print("\nYou can now run ./cwmai for the enhanced terminal experience")

if __name__ == "__main__":
    fix_architecture_issues()