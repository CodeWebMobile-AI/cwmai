#!/bin/bash
# Run the self-improving system with proper environment variables

echo "Starting Self-Improving CWMAI System..."
export $(cat .env.local | grep -v '^#' | xargs)

# Create logs directory if it doesn't exist
mkdir -p logs

python3 start_self_improving.py