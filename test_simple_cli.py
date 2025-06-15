#!/usr/bin/env python3
"""
Test the simple smart CLI
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from simple_smart_cli import SimpleSmartCLI, print_result


async def test_commands():
    """Test various commands."""
    print("🧪 Testing Simple Smart CLI\n")
    
    cli = SimpleSmartCLI()
    
    # Test commands
    test_cases = [
        "show status",
        "create issue for auth-api about slow login times",
        "search for python",
        "help"
    ]
    
    for command in test_cases:
        print(f"📝 Command: {command}")
        try:
            result = await cli.process_command(command)
            print_result(result)
        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 60)
        print()


if __name__ == "__main__":
    # Check environment
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("Setting up environment from .env.local...")
        os.system('export $(cat .env.local | grep -v "^#" | xargs)')
    
    asyncio.run(test_commands())