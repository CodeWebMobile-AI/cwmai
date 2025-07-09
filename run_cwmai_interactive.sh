#!/bin/bash
# Interactive CWMAI launcher

cd /Users/fortressarigo/PhpstormProjects/cwmai

# Export environment variables
export $(cat .env.local | grep -v '^#' | xargs)
export PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai

# Run the enhanced version
python3 cwmai