#!/bin/bash
# Run the Interactive Hybrid Continuous System with Claude Code-like interface

# Set default number of workers
WORKERS=${1:-10}

echo "🚀 Starting Interactive Hybrid System with $WORKERS workers..."

# Export environment variables
export $(cat .env.local | grep -v '^#' | xargs)
export PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai/zip

# Run with arch for ARM64 Macs
arch -arm64 /usr/bin/python3 interactive_hybrid_continuous.py --workers $WORKERS