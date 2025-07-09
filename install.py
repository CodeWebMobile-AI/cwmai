#!/usr/bin/env python3
"""
CWMAI Cross-Platform Installer
Detects system architecture and installs dependencies appropriately.
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path
import json
import urllib.request
import ssl

class CWMAIInstaller:
    def __init__(self):
        self.python_version = sys.version_info
        self.os_type = platform.system()
        self.architecture = platform.machine()
        self.is_arm = self.architecture.lower() in ['arm64', 'aarch64']
        self.is_m1_mac = self.os_type == 'Darwin' and self.is_arm
        self.venv_path = Path('venv')
        
    def print_banner(self):
        """Print installation banner."""
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║           CWMAI - Cross-Platform Python Installer                ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print()
        
    def check_python_version(self):
        """Check if Python version meets requirements."""
        print("🐍 Checking Python version...")
        print(f"   Python {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        print(f"   Platform: {self.os_type} ({self.architecture})")
        
        if self.python_version < (3, 9):
            print("❌ Python 3.9 or higher is required")
            sys.exit(1)
            
        print("✅ Python version OK")
        
    def create_virtual_environment(self):
        """Create a virtual environment."""
        print("\n🔧 Creating virtual environment...")
        
        # Remove existing venv
        if self.venv_path.exists():
            print("   Removing existing virtual environment...")
            shutil.rmtree(self.venv_path)
            
        # Create new venv
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        print("✅ Virtual environment created")
        
    def get_pip_command(self):
        """Get the pip command for the virtual environment."""
        if self.os_type == 'Windows':
            return str(self.venv_path / 'Scripts' / 'pip')
        return str(self.venv_path / 'bin' / 'pip')
        
    def get_python_command(self):
        """Get the python command for the virtual environment."""
        if self.os_type == 'Windows':
            return str(self.venv_path / 'Scripts' / 'python')
        return str(self.venv_path / 'bin' / 'python')
        
    def upgrade_pip(self):
        """Upgrade pip and install basic tools."""
        print("\n📦 Upgrading pip and installing build tools...")
        pip_cmd = self.get_pip_command()
        
        subprocess.run([
            pip_cmd, 'install', '--upgrade', 
            'pip', 'setuptools', 'wheel'
        ], check=True)
        
        print("✅ Pip and build tools updated")
        
    def fix_requirements(self):
        """Create a fixed requirements file for the current platform."""
        print("\n📋 Preparing requirements...")
        
        # Read original requirements
        with open('requirements.txt', 'r') as f:
            requirements = f.readlines()
            
        # Create fixed requirements
        fixed_requirements = []
        
        for req in requirements:
            req = req.strip()
            if not req or req.startswith('#'):
                continue
                
            # Skip conflicting packages
            if 'redis-py-cluster' in req:
                print(f"   Skipping {req} due to conflict with redis>=5.0")
                continue
            if 'fakeredis' in req and any('redis==' in r for r in requirements):
                # Only skip fakeredis if we have a specific redis version
                print(f"   Skipping {req} due to potential conflict")
                continue
                
            # Platform-specific adjustments
            if self.is_m1_mac and 'tensorflow' in req.lower():
                # TensorFlow needs special handling on M1
                print(f"   Adjusting {req} for Apple Silicon")
                req = 'tensorflow-macos==2.13.0'
                
            fixed_requirements.append(req)
            
        # Write fixed requirements
        with open('requirements-fixed.txt', 'w') as f:
            f.write('\n'.join(fixed_requirements))
            
        print("✅ Requirements prepared")
        return 'requirements-fixed.txt'
        
    def install_dependencies(self):
        """Install Python dependencies."""
        print("\n📚 Installing dependencies...")
        pip_cmd = self.get_pip_command()
        req_file = self.fix_requirements()
        
        # Set environment variables for macOS ARM64
        env = os.environ.copy()
        if self.is_m1_mac:
            env['ARCHFLAGS'] = '-arch arm64'
            
        # Try installing all at once
        try:
            subprocess.run([pip_cmd, 'install', '-r', req_file], 
                         check=True, env=env)
            print("✅ All dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("⚠️  Some dependencies failed. Installing individually...")
            self.install_problematic_packages()
            
        # Clean up
        os.remove(req_file)
        
    def install_problematic_packages(self):
        """Install packages that commonly have issues."""
        pip_cmd = self.get_pip_command()
        
        # Packages that often need special handling
        problematic = [
            'psutil',
            'cryptography',
            'numpy',
            'pandas',
            'scikit-learn',
            'matplotlib'
        ]
        
        for package in problematic:
            try:
                print(f"   Installing {package}...")
                subprocess.run([pip_cmd, 'install', package], 
                             capture_output=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"   ⚠️  Failed to install {package}: {e}")
                
    def setup_nltk_data(self):
        """Download NLTK data files."""
        print("\n📖 Setting up NLTK data...")
        python_cmd = self.get_python_command()
        
        nltk_script = """
import nltk
import ssl

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required data
for data in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.download(data, quiet=True)
        print(f'   ✅ Downloaded {data}')
    except Exception as e:
        print(f'   ⚠️  Failed to download {data}: {e}')
"""
        
        try:
            subprocess.run([python_cmd, '-c', nltk_script], check=True)
        except subprocess.CalledProcessError:
            print("   ⚠️  NLTK setup failed. The system will work with reduced NLP capabilities.")
            
    def setup_environment(self):
        """Set up environment variables."""
        print("\n🔐 Setting up environment configuration...")
        
        if not Path('.env').exists():
            if Path('.env.local').exists():
                shutil.copy('.env.local', '.env')
                print("✅ Copied .env.local to .env")
            else:
                print("⚠️  No .env file found. Please create one from .env.example")
        else:
            print("✅ .env file already exists")
            
    def create_run_scripts(self):
        """Create convenient run scripts."""
        print("\n📝 Creating run scripts...")
        
        # Create run_cwmai script
        if self.os_type == 'Windows':
            with open('run_cwmai.bat', 'w') as f:
                f.write('@echo off\n')
                f.write('call venv\\Scripts\\activate\n')
                f.write('python cwmai %*\n')
        else:
            with open('run_cwmai.sh', 'w') as f:
                f.write('#!/bin/bash\n')
                f.write('source venv/bin/activate\n')
                f.write('python cwmai "$@"\n')
            os.chmod('run_cwmai.sh', 0o755)
            
        print("✅ Run scripts created")
        
    def test_installation(self):
        """Test the installation."""
        print("\n🧪 Testing installation...")
        python_cmd = self.get_python_command()
        
        test_script = """
import sys
print(f'   Python: {sys.version.split()[0]}')

# Test critical imports
modules = ['redis', 'psutil', 'numpy', 'pandas', 'github']
for module in modules:
    try:
        __import__(module)
        print(f'   ✅ {module} OK')
    except ImportError as e:
        print(f'   ❌ {module} failed: {e}')
"""
        
        subprocess.run([python_cmd, '-c', test_script])
        
    def print_instructions(self):
        """Print usage instructions."""
        print("\n╔══════════════════════════════════════════════════════════════════╗")
        print("║                    Installation Complete! 🎉                     ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print()
        print("To use CWMAI:")
        
        if self.os_type == 'Windows':
            print("1. Activate environment: venv\\Scripts\\activate")
            print("2. Run CWMAI: python cwmai")
            print("   Or use: run_cwmai.bat")
        else:
            print("1. Activate environment: source venv/bin/activate")
            print("2. Run CWMAI: ./cwmai")
            print("   Or use: ./run_cwmai.sh")
            
        print("\nFor continuous AI: python run_continuous_ai.py")
        
    def run(self):
        """Run the installation process."""
        try:
            self.print_banner()
            self.check_python_version()
            self.create_virtual_environment()
            self.upgrade_pip()
            self.install_dependencies()
            self.setup_nltk_data()
            self.setup_environment()
            self.create_run_scripts()
            self.test_installation()
            self.print_instructions()
        except Exception as e:
            print(f"\n❌ Installation failed: {e}")
            sys.exit(1)

if __name__ == '__main__':
    installer = CWMAIInstaller()
    installer.run()