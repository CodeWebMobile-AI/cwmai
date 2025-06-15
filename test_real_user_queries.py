#!/usr/bin/env python3
"""
Test the system's ability to handle real user queries by automatically
generating and using tools as needed.
"""

import asyncio
from scripts.tool_calling_system import ToolCallingSystem


async def test_user_query(query: str):
    """Test if the system can handle a user query by creating tools as needed"""
    print(f"\n{'='*60}")
    print(f"USER QUERY: '{query}'")
    print('='*60)
    
    tool_system = ToolCallingSystem()
    
    # First check if we have a tool that can handle this
    print("\n1. Checking for existing tools...")
    result = await tool_system.call_tool("find_similar_tools", query=query)
    if result.get('success'):
        matches = result.get('result', {}).get('matches', [])
        if matches:
            print(f"   Found {len(matches)} similar tools:")
            for match in matches[:3]:
                print(f"   - {match['tool_name']}: {match['similarity']}")
        else:
            print("   No existing tools found")
    
    # Try to handle the query as a complex query
    print("\n2. Attempting to handle as complex query...")
    result = await tool_system.call_tool("handle_complex_query", query=query)
    
    if result.get('success'):
        res = result.get('result', {})
        if 'error' not in res:
            print("   ✓ Successfully handled query!")
            print(f"   Strategy: {res.get('strategy', 'N/A')}")
            print(f"   Tasks executed: {res.get('tasks_executed', 0)}")
            
            # Check if any new tools were created
            import os
            custom_tools_dir = "scripts/custom_tools"
            if os.path.exists(custom_tools_dir):
                tools_before = set(os.listdir(custom_tools_dir))
                # Small delay to ensure files are written
                await asyncio.sleep(0.5)
                tools_after = set(os.listdir(custom_tools_dir))
                new_tools = tools_after - tools_before
                if new_tools:
                    print(f"   New tools created: {list(new_tools)}")
            
            # Show the result
            if 'result' in res:
                final_result = res['result']
                if isinstance(final_result, dict) and 'summary' in final_result:
                    print(f"\n   ANSWER: {final_result['summary']}")
                else:
                    print(f"\n   RESULT: {final_result}")
        else:
            print(f"   ✗ Error: {res.get('error')}")
    else:
        print(f"   ✗ Failed: {result.get('error', 'Unknown error')}")


async def main():
    """Test various realistic user queries"""
    
    # Queries that should work with existing tools
    existing_tool_queries = [
        "How many repositories do we have?",
        "Show me all the tasks",
        "What's the system status?",
    ]
    
    # Queries that would need new tools
    new_tool_queries = [
        "How many Python files are in the scripts directory?",
        "What's the total size of all log files?",
        "How many TODO comments are in the codebase?",
        "List all the environment variables that start with 'GITHUB'",
        "What's the average file size in the project?",
        "How many active workers are currently running?",
        "Find all files modified in the last 24 hours",
        "Calculate the total lines of code in Python files",
    ]
    
    print("Testing queries that should use existing tools:")
    for query in existing_tool_queries[:1]:  # Test just one for brevity
        await test_user_query(query)
    
    print("\n\nTesting queries that need new tool generation:")
    for query in new_tool_queries[:3]:  # Test a few examples
        await test_user_query(query)
        await asyncio.sleep(1)  # Small delay between queries


if __name__ == "__main__":
    asyncio.run(main())