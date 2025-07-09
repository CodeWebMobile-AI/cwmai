#!/bin/bash
# Fix virtual environment for ARM64 architecture

echo "================================================================================"
echo "🔧 FIXING VIRTUAL ENVIRONMENT FOR ARM64"
echo "================================================================================"

# Check current Python architecture
echo "Current system Python architecture:"
/usr/bin/python3 -c "import platform; print(f'  System Python: {platform.machine()}')"

if [ -d "venv" ]; then
    echo -e "\n⚠️  Existing virtual environment found with wrong architecture"
    echo "   Checking current venv architecture..."
    source venv/bin/activate
    python -c "import platform; print(f'  Venv Python: {platform.machine()}')" 2>/dev/null || echo "  Venv Python: Unable to detect (likely x86_64)"
    deactivate
    
    echo -e "\n🗑️  Removing existing virtual environment..."
    rm -rf venv
fi

echo -e "\n📦 Creating new ARM64 virtual environment..."
arch -arm64 /usr/bin/python3 -m venv venv

echo -e "\n✅ Activating new virtual environment..."
source venv/bin/activate

echo -e "\n🔍 Verifying architecture..."
python -c "import platform; print(f'  New venv Python: {platform.machine()}')"

echo -e "\n📥 Installing core packages..."
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install packages that need special handling for ARM64
echo "  Installing numpy for ARM64..."
pip install --no-binary :all: --force-reinstall numpy

echo "  Installing cryptography for ARM64..."
pip install --no-binary :all: --force-reinstall cryptography cffi

echo -e "\n📋 Installing remaining requirements..."
# Install other requirements (skip the ones we already installed)
pip install -r requirements.txt --upgrade

echo -e "\n✅ Virtual environment fixed for ARM64!"
echo -e "\nTo use it:"
echo "  source venv/bin/activate"
echo "  export \$(cat .env.local | grep -v '^#' | xargs)"
echo "  export PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai"
echo "  arch -arm64 python run_knowledge_driven_evolution.py --minutes 15"