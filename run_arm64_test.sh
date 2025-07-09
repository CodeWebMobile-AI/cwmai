#!/bin/bash
# Script to run tests with ARM64 architecture

echo "================================================================================"
echo "🚀 Running Knowledge-Driven Evolution with ARM64 Architecture"
echo "================================================================================"

# Export environment variables
export $(cat .env.local | grep -v '^#' | xargs)
export PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai

# Check if we have a virtual environment
if [ -d "venv" ]; then
    echo "✅ Virtual environment found"
    # Activate virtual environment with ARM64
    source venv/bin/activate
    echo "✅ Virtual environment activated"
    
    # Run with arch -arm64
    echo "🔄 Running with arch -arm64 and virtual environment..."
    arch -arm64 python run_knowledge_driven_evolution.py --minutes 15
else
    echo "⚠️  No virtual environment found, using system Python with arch -arm64"
    arch -arm64 python3 run_knowledge_driven_evolution.py --minutes 15
fi