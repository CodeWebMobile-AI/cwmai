#!/usr/bin/env python3
"""
Start the Claude Code-like Todo System

This is the main entry point for our natural language todo system.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))


async def main():
    """Start the system based on command line arguments."""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            # Run the demo
            from test_claude_code_demo import demo_claude_code_experience, demo_real_workflow
            await demo_claude_code_experience()
            await demo_real_workflow()
            
        elif sys.argv[1] == "cli":
            # Run the enhanced CLI
            from enhanced_todo_cli import EnhancedTodoCLI
            cli = EnhancedTodoCLI()
            await cli.start()
            
        elif sys.argv[1] == "test":
            # Run tests
            from test_todo_simple import main as run_tests
            await run_tests()
            
        else:
            print("Unknown command. Use: demo, cli, or test")
    else:
        # Default: Run the enhanced CLI
        print("Starting Claude Code-like Todo System...")
        print("(Use 'python start_claude_code.py demo' to see a demo)\n")
        
        from enhanced_todo_cli import EnhancedTodoCLI
        cli = EnhancedTodoCLI()
        await cli.start()


if __name__ == "__main__":
    asyncio.run(main())