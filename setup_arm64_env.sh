#\!/bin/bash
# Setup ARM64 environment for CWMAI

echo "Setting up ARM64 Python environment..."

# Remove old venv if exists
rm -rf venv

# Create new venv with ARM64 Python
arch -arm64 /usr/bin/python3 -m venv venv

# Activate and install packages
source venv/bin/activate

# Verify architecture
echo "Python architecture: $(python3 -c 'import platform; print(platform.machine())')"

# Upgrade pip first
pip install --upgrade pip

# Install packages with --no-binary to force compilation for ARM64
pip install --no-binary :all: psutil==5.9.5

# Install the rest of the requirements
pip install -r requirements.txt

echo "Setup complete\!"
EOF < /dev/null