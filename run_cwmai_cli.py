#!/usr/bin/env python3
"""
Run CWMAI CLI - Natural Language Interface for CWMAI

This script provides an interactive command-line interface for natural language
interaction with the CWMAI system.

Usage:
    python run_cwmai_cli.py              # Interactive mode
    python run_cwmai_cli.py "command"    # Single command mode

Examples:
    python run_cwmai_cli.py
    python run_cwmai_cli.py "create issue for myrepo about adding dark mode"
    python run_cwmai_cli.py "search repositories for python AI"
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from scripts.cwmai_cli import main


def check_environment():
    """Check and display environment status."""
    print("Checking environment...")
    
    # Check for API keys
    providers = []
    if os.getenv('ANTHROPIC_API_KEY'):
        providers.append('Anthropic')
    if os.getenv('OPENAI_API_KEY'):
        providers.append('OpenAI')
    if os.getenv('GEMINI_API_KEY'):
        providers.append('Gemini')
    if os.getenv('DEEPSEEK_API_KEY'):
        providers.append('DeepSeek')
        
    if providers:
        print(f"✓ AI Providers available: {', '.join(providers)}")
    else:
        print("⚠ No AI provider API keys found. Please set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY")
        
    # Check for GitHub token
    if os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT'):
        print("✓ GitHub integration configured")
    else:
        print("⚠ GitHub integration not configured. Set GITHUB_TOKEN or CLAUDE_PAT for full functionality")
        
    # Check for MCP config
    mcp_config_path = project_root / 'mcp_config.json'
    if mcp_config_path.exists():
        print("✓ MCP configuration found")
    else:
        print("⚠ MCP configuration not found. Some features may be limited")
        
    print()  # Empty line for spacing


if __name__ == "__main__":
    try:
        # Show usage if --help is passed
        if '--help' in sys.argv or '-h' in sys.argv:
            print(__doc__)
            sys.exit(0)
            
        # Check environment unless --skip-check is passed
        if '--skip-check' not in sys.argv:
            check_environment()
            
        # Remove --skip-check from args if present
        if '--skip-check' in sys.argv:
            sys.argv.remove('--skip-check')
            
        # Run the CLI
        main()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)