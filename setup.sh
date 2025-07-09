#!/bin/bash
# CWMAI Cross-Platform Setup Script

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║            CWMAI - Cross-Platform Setup Script                   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo

# Function to detect OS and architecture
detect_system() {
    OS=$(uname -s)
    ARCH=$(uname -m)
    
    echo "🔍 Detected System: $OS ($ARCH)"
    
    # Convert architecture names to Python wheel tags
    case "$ARCH" in
        x86_64|amd64)
            PYTHON_ARCH="x86_64"
            ;;
        arm64|aarch64)
            PYTHON_ARCH="arm64"
            ;;
        *)
            PYTHON_ARCH="$ARCH"
            ;;
    esac
    
    # Detect if running on Apple Silicon with Rosetta
    if [[ "$OS" == "Darwin" ]] && [[ "$ARCH" == "x86_64" ]]; then
        if sysctl -n sysctl.proc_translated 2>/dev/null | grep -q "1"; then
            echo "⚠️  Running under Rosetta 2 translation"
            ROSETTA=true
        fi
    fi
}

# Function to check Python version
check_python() {
    echo "🐍 Checking Python installation..."
    
    # Try python3 first, then python
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo "❌ Python not found. Please install Python 3.9 or higher."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYTHON_VERSION_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_VERSION_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [[ $PYTHON_VERSION_MAJOR -lt 3 ]] || [[ $PYTHON_VERSION_MAJOR -eq 3 && $PYTHON_VERSION_MINOR -lt 9 ]]; then
        echo "❌ Python $PYTHON_VERSION detected. CWMAI requires Python 3.9 or higher."
        exit 1
    fi
    
    echo "✅ Python $PYTHON_VERSION detected"
}

# Function to create virtual environment
create_venv() {
    echo "🔧 Creating virtual environment..."
    
    # Remove old venv if exists
    if [ -d "venv" ]; then
        echo "   Removing existing virtual environment..."
        rm -rf venv
    fi
    
    # Create new virtual environment
    $PYTHON_CMD -m venv venv
    
    # Activate virtual environment
    if [[ "$OS" == "MINGW"* ]] || [[ "$OS" == "CYGWIN"* ]]; then
        # Windows Git Bash
        source venv/Scripts/activate
    else
        # Unix-like systems
        source venv/bin/activate
    fi
    
    echo "✅ Virtual environment created and activated"
}

# Function to upgrade pip and install basic tools
upgrade_pip() {
    echo "📦 Upgrading pip and installing build tools..."
    
    python -m pip install --upgrade pip setuptools wheel
    
    # Install platform-specific build tools
    if [[ "$OS" == "Darwin" ]]; then
        # macOS specific
        if ! command -v brew &> /dev/null; then
            echo "⚠️  Homebrew not found. Some dependencies might require manual installation."
        fi
    fi
    
    echo "✅ Pip and build tools updated"
}

# Function to install dependencies
install_dependencies() {
    echo "📚 Installing dependencies..."
    
    # Create a temporary requirements file with platform-specific adjustments
    cp requirements.txt requirements-temp.txt
    
    # Handle architecture-specific issues
    if [[ "$OS" == "Darwin" ]] && [[ "$ARCH" == "arm64" ]]; then
        echo "   Configuring for Apple Silicon..."
        # Force installation of universal2 or arm64 wheels when available
        export ARCHFLAGS="-arch arm64"
    fi
    
    # Install dependencies with retry logic
    MAX_RETRIES=3
    RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if pip install -r requirements-temp.txt; then
            echo "✅ Dependencies installed successfully"
            break
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            echo "⚠️  Installation attempt $RETRY_COUNT failed. Retrying..."
            
            # Try to fix common issues
            if [ $RETRY_COUNT -eq 1 ]; then
                # First retry: try no-binary for problematic packages
                pip install --no-binary :all: psutil
            elif [ $RETRY_COUNT -eq 2 ]; then
                # Second retry: install problematic packages separately
                pip install psutil cryptography numpy pandas scikit-learn
                pip install -r requirements-temp.txt || true
            fi
        fi
    done
    
    # Clean up
    rm -f requirements-temp.txt
}

# Function to download NLTK data
setup_nltk() {
    echo "📖 Setting up NLTK data..."
    
    python -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
print('✅ NLTK data downloaded')
" || echo "⚠️  NLTK data download failed. The system will still work but with reduced NLP capabilities."
}

# Function to setup environment file
setup_env() {
    echo "🔐 Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.local" ]; then
            cp .env.local .env
            echo "✅ Copied .env.local to .env"
        else
            echo "⚠️  No .env file found. Please create one from .env.example or .env.local"
        fi
    else
        echo "✅ .env file already exists"
    fi
}

# Function to create activation scripts
create_activation_scripts() {
    echo "📝 Creating activation scripts..."
    
    # Create activate.sh for Unix-like systems
    cat > activate.sh << 'EOF'
#!/bin/bash
# CWMAI Environment Activation Script

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ CWMAI environment activated"
    echo "   Python: $(which python)"
    echo "   Version: $(python --version)"
else
    echo "❌ Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Environment variables loaded"
else
    echo "⚠️  .env file not found"
fi
EOF
    chmod +x activate.sh
    
    echo "✅ Activation scripts created"
}

# Function to test the installation
test_installation() {
    echo "🧪 Testing installation..."
    
    # Test basic imports
    python -c "
import sys
print(f'   Python: {sys.version}')
try:
    import redis
    print('   ✅ Redis module OK')
except Exception as e:
    print(f'   ❌ Redis module failed: {e}')
try:
    import psutil
    print('   ✅ Psutil module OK')
except Exception as e:
    print(f'   ❌ Psutil module failed: {e}')
try:
    import numpy
    print('   ✅ Numpy module OK')
except Exception as e:
    print(f'   ❌ Numpy module failed: {e}')
"
    
    # Test cwmai script
    if python cwmai --help > /dev/null 2>&1; then
        echo "   ✅ CWMAI script OK"
    else
        echo "   ⚠️  CWMAI script has issues but may still work"
    fi
}

# Main setup flow
main() {
    detect_system
    check_python
    create_venv
    upgrade_pip
    install_dependencies
    setup_nltk
    setup_env
    create_activation_scripts
    test_installation
    
    echo
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                    Setup Complete! 🎉                            ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo
    echo "To use CWMAI:"
    echo "1. Activate the environment: source activate.sh"
    echo "2. Run CWMAI: ./cwmai"
    echo "3. Or run continuous AI: python run_continuous_ai.py"
    echo
    echo "For Windows users, use: venv\\Scripts\\activate"
}

# Run main function
main