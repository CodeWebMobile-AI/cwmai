#!/usr/bin/env python3
"""
Demonstrate automatic tool generation from natural language queries
"""

import asyncio
from scripts.tool_calling_system import ToolCallingSystem
import os


async def demonstrate_query(query: str):
    """Show how the system handles a natural language query"""
    print(f"\n{'='*60}")
    print(f"USER QUERY: {query}")
    print('='*60)
    
    tool_system = ToolCallingSystem()
    
    # The system should automatically:
    # 1. Recognize there's no existing tool
    # 2. Generate a new tool based on the query
    # 3. Execute it
    
    result = await tool_system.call_tool('handle_complex_query', query=query)
    
    if result.get('success'):
        res = result.get('result', {})
        if 'error' not in str(res):
            print("✓ Query handled successfully!")
            if 'summary' in res:
                print(f"Summary: {res['summary']}")
            elif 'result' in res:
                print(f"Result: {res['result']}")
        else:
            print(f"✗ Error: {res}")
    else:
        print(f"✗ Failed: {result.get('error', 'Unknown error')}")
    
    # Check if any new tools were created
    custom_tools_dir = "scripts/custom_tools"
    if os.path.exists(custom_tools_dir):
        tools = [f for f in os.listdir(custom_tools_dir) if f.endswith('.py')]
        new_tools = [t for t in tools if 'tool_task_' in t or query.lower().replace(' ', '_')[:20] in t]
        if new_tools:
            print(f"\n→ New tools created: {new_tools}")
    
    return result


async def main():
    print("CWMAI Automatic Tool Generation Demo")
    print("====================================")
    print("The system will automatically create tools based on your queries.\n")
    
    # Test queries
    queries = [
        "Count how many Python files are in the scripts directory",
        "List all environment variables that start with GITHUB",
        "Find the largest file in the project"
    ]
    
    for query in queries:
        await demonstrate_query(query)
        await asyncio.sleep(2)
    
    print("\n" + "="*60)
    print("Demo complete! The system has automatically generated tools")
    print("to handle your natural language queries.")


if __name__ == "__main__":
    asyncio.run(main())