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
