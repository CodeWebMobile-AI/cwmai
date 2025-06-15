#!/usr/bin/env python3
"""Test environment variable loading."""

import os
from dotenv import load_dotenv

# Load environment variables in the same order as run_continuous_ai.py
load_dotenv('.env.local')  # Load local environment first
load_dotenv()  # Then load .env as fallback

# Check if API keys are loaded
print("Environment variables loaded:")
print(f"ANTHROPIC_API_KEY: {'SET' if os.getenv('ANTHROPIC_API_KEY') else 'NOT SET'}")
print(f"GITHUB_TOKEN: {'SET' if os.getenv('GITHUB_TOKEN') else 'NOT SET'}")
print(f"CLAUDE_PAT: {'SET' if os.getenv('CLAUDE_PAT') else 'NOT SET'}")
print(f"ORCHESTRATOR_MODE: {os.getenv('ORCHESTRATOR_MODE', 'not set')}")

# Try to create the config
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from production_config import create_config

config = create_config('development')
print(f"\nConfig created in {config.mode} mode")
print(f"Anthropic API key in config: {'SET' if config.anthropic_api_key else 'NOT SET'}")
print(f"GitHub token in config: {'SET' if config.github_token else 'NOT SET'}")

# Validate
print(f"\nValidation result: {config.validate()}")