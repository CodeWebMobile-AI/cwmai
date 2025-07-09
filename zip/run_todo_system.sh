#!/bin/bash
# Start the Claude Code-like Todo System

# Export environment variables
export $(cat ../.env.local | grep -v '^#' | xargs)
export PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai/zip

# Run with arch for ARM64 Macs
arch -arm64 /usr/bin/python3 start_claude_code.py cli