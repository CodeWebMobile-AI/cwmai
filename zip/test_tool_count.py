#!/usr/bin/env python3
"""Test script to count tools in the system."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.tool_calling_system import ToolCallingSystem


async def main():
    """Count and display all tools."""
    print("Initializing Tool Calling System...")
    tool_system = ToolCallingSystem()
    
    # Get all tools
    tools = tool_system.tools
    
    print(f"\nTotal tools: {len(tools)}")
    print("\nTools by category:")
    
    # Categorize tools
    builtin = []
    custom = []
    mcp = []
    
    for name, tool in tools.items():
        if ':' in name:  # MCP tools have format "server:tool_name"
            mcp.append(name)
        elif tool.created_by_ai:
            custom.append(name)
        else:
            builtin.append(name)
    
    print(f"- Built-in tools: {len(builtin)}")
    print(f"- Custom AI-created tools: {len(custom)}")
    print(f"- MCP tools: {len(mcp)}")
    
    # List all tools
    print("\nAll tools:")
    for i, (name, tool) in enumerate(tools.items(), 1):
        print(f"{i}. {name} - {tool.description[:60]}...")


if __name__ == "__main__":
    asyncio.run(main())