#!/bin/bash
# Safe runner for continuous system with error handling

echo "🚀 Starting CWMAI Continuous System (Safe Mode)..."

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load environment
export $(cat "$DIR/.env.local" | grep -v '^#' | xargs)
export NLTK_DATA="$HOME/nltk_data"

# Check Python and numpy
echo "🔍 Checking system compatibility..."
arch -arm64 python -c "
import sys
import platform
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Architecture: {platform.machine()}')

try:
    import numpy as np
    print(f'NumPy: {np.__version__} ✅')
except Exception as e:
    print(f'NumPy Error: {e} ❌')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ NumPy check failed. Fixing..."
    arch -arm64 pip uninstall -y numpy
    arch -arm64 pip install --no-cache-dir "numpy<2.0"
fi

# Start with basic mode first
echo ""
echo "🚀 Starting in basic mode (no MCP) for stability..."
echo ""

# Run with proper error handling
arch -arm64 python "$DIR/run_continuous_ai.py" \
    --mode basic \
    --workers 2 \
    --no-research \
    2>&1 | tee -a continuous_ai_safe.log