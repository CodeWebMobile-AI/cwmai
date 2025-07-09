#!/usr/bin/env python3
"""Test the get_tool_usage_stats tool to debug the count issue."""

import asyncio
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.tool_calling_system import ToolCallingSystem


async def main():
    """Test get_tool_usage_stats tool."""
    print("Testing get_tool_usage_stats tool...")
    tool_system = ToolCallingSystem()
    
    # First, let's count tools directly
    print(f"\nDirect tool count: {len(tool_system.tools)}")
    print("Tools in system:")
    for i, (name, tool) in enumerate(tool_system.tools.items(), 1):
        print(f"  {i}. {name} (AI created: {tool.created_by_ai})")
    
    # Now call get_tool_usage_stats
    print("\n\nCalling get_tool_usage_stats():")
    result = await tool_system.call_tool("get_tool_usage_stats")
    
    if result.get('success'):
        stats = result['result']
        print(f"\nTool usage stats returned:")
        print(f"  Total tools: {stats.get('total_tools', 0)}")
        print(f"  Total calls: {stats.get('total_calls', 0)}")
        print(f"  AI created tools: {stats.get('ai_created_tools', 0)}")
        
        print(f"\n  Most used tools:")
        for tool in stats.get('most_used', []):
            print(f"    - {tool['name']}: {tool['usage_count']} uses")
            
        print(f"\n  Unused tools: {len(stats.get('least_used', []))}")
        
        # Check if the counts match
        if stats.get('total_tools') != len(tool_system.tools):
            print(f"\n⚠️  MISMATCH DETECTED!")
            print(f"  get_tool_usage_stats reports: {stats.get('total_tools')} tools")
            print(f"  Direct count shows: {len(tool_system.tools)} tools")
            
            # Let's check the detailed stats
            detailed = stats.get('detailed_stats', [])
            print(f"\n  Detailed stats has {len(detailed)} entries")
            
            # Find tools that might be counted extra
            tool_names_in_stats = {t['name'] for t in detailed}
            tool_names_direct = set(tool_system.tools.keys())
            
            extra_in_stats = tool_names_in_stats - tool_names_direct
            missing_from_stats = tool_names_direct - tool_names_in_stats
            
            if extra_in_stats:
                print(f"\n  Tools in stats but not in direct count: {extra_in_stats}")
            if missing_from_stats:
                print(f"\n  Tools in direct count but not in stats: {missing_from_stats}")
    else:
        print(f"Error calling tool: {result.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())