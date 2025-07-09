#!/bin/bash
# Run the Enhanced Conversational Assistant with Claude Code-like interface

echo "🤖 Starting Enhanced AI Assistant with Claude Code Experience..."

# Export environment variables
export $(cat .env.local | grep -v '^#' | xargs)
export PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai/zip

# Run with arch for ARM64 Macs
arch -arm64 /usr/bin/python3 scripts/enhanced_conversational_assistant.py