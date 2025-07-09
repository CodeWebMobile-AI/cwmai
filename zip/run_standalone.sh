#!/bin/bash
# Standalone run script for zip folder

# Ensure we're in the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running CWMAI from: $(pwd)"

# Check for .env.local
if [ ! -f ".env.local" ]; then
    echo "Error: .env.local not found!"
    echo "Please copy your .env.local file to this directory"
    exit 1
fi

# Export environment variables
export $(cat .env.local | grep -v '^#' | xargs)
export PYTHONPATH="$SCRIPT_DIR"

# Run the system
echo "Starting system..."
arch -arm64 /usr/bin/python3 run_hybrid_continuous.py "$@"
