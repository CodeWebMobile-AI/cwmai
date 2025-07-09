#!/bin/bash
# Complete fix for architecture issues on Apple Silicon

echo "🔧 Fixing architecture issues for Apple Silicon..."

# Force ARM64 architecture
export ARCHFLAGS="-arch arm64"

# Check current Python interpreter
echo "📊 Current Python info:"
python -c "import platform; print(f'Platform: {platform.platform()}'); print(f'Machine: {platform.machine()}')"

# Reinstall critical packages for ARM64
echo -e "\n📦 Reinstalling packages for ARM64 architecture..."

# Uninstall problematic packages
pip uninstall -y psutil numpy scipy scikit-learn pandas cffi cryptography regex

# Install with ARM64 architecture
pip install --no-cache-dir --no-binary :all: --force-reinstall \
    psutil==7.0.0

# Install other packages normally (they should detect ARM64)
pip install --no-cache-dir \
    numpy \
    scipy \
    scikit-learn \
    pandas \
    cffi \
    cryptography \
    regex

echo -e "\n✅ Architecture fix complete!"
echo "🔄 Please restart your terminal and try running ./cwmai again"