#!/bin/bash
# Test script to run from within the zip directory

echo "Current directory: $(pwd)"
echo "Changing to zip directory..."
cd /Users/fortressarigo/PhpstormProjects/cwmai/zip

echo "Now in: $(pwd)"
echo "Checking for local_repos in zip folder..."
ls -la local_repos 2>/dev/null || echo "No local_repos folder in zip directory"

echo ""
echo "Setting up environment and running..."
export $(cat /Users/fortressarigo/PhpstormProjects/cwmai/.env.local | grep -v '^#' | xargs)
export PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai/zip

echo "Running with 2 workers for testing..."
arch -arm64 /usr/bin/python3 run_hybrid_continuous.py --workers 2