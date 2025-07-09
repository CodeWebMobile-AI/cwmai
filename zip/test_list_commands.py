#!/usr/bin/env python3
"""Test the list_available_commands tool."""

import asyncio
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.tool_calling_system import ToolCallingSystem


async def main():
    """Test list_available_commands tool."""
    print("Testing list_available_commands tool...")
    tool_system = ToolCallingSystem()
    
    # Test without category filter
    print("\n1. Testing list_available_commands() - no filter:")
    result = await tool_system.call_tool("list_available_commands")
    if result.get('success'):
        data = result['result']
        print(f"Total tools: {data.get('total_tools', 0)}")
        print(f"AI-created tools: {data.get('ai_created_count', 0)}")
        print("\nTools by category:")
        for category, tools in data.get('by_category', {}).items():
            if tools:  # Only show categories with tools
                print(f"\n{category.upper()} ({len(tools)} tools):")
                for tool in tools[:3]:  # Show first 3 tools per category
                    print(f"  - {tool['name']}: {tool['description'][:50]}...")
                if len(tools) > 3:
                    print(f"  ... and {len(tools) - 3} more")
    else:
        print(f"Error: {result.get('error')}")
    
    # Test with category filter
    print("\n\n2. Testing list_available_commands(category='repository'):")
    result = await tool_system.call_tool("list_available_commands", category="repository")
    if result.get('success'):
        data = result['result']
        tools = data.get('tools', [])
        if tools:
            print(f"Repository tools ({len(tools)}):")
            for tool in tools:
                print(f"  - {tool['name']}: {tool['description']}")
        else:
            print(f"No tools found in category 'repository'")
            print(f"Available categories: {', '.join(data.get('available_categories', []))}")
    else:
        print(f"Error: {result.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())