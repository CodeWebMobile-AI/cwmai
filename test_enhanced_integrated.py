#!/usr/bin/env python3
"""
Integrated test demonstrating enhanced tool capabilities
"""

import asyncio
import json


async def main():
    """Test the enhanced tool system integration"""
    print("=== Enhanced Tool System Integration Test ===\n")
    
    # Force clean import
    import sys
    for module in list(sys.modules.keys()):
        if module.startswith('scripts.'):
            del sys.modules[module]
    
    # Import with enhanced systems
    from scripts.tool_calling_system import ToolCallingSystem
    
    print("Initializing Tool Calling System...")
    tool_system = ToolCallingSystem()
    
    # Check if enhanced systems loaded
    print("\nEnhanced Systems Status:")
    print(f"✓ Dependency Resolver: {'Loaded' if tool_system.dependency_resolver else 'Not loaded'}")
    print(f"✓ Multi-Tool Orchestrator: {'Loaded' if tool_system.multi_tool_orchestrator else 'Not loaded'}")
    print(f"✓ Tool Evolution: {'Loaded' if tool_system.tool_evolution else 'Not loaded'}")
    print(f"✓ Semantic Matcher: {'Loaded' if tool_system.semantic_matcher else 'Not loaded'}")
    
    # Test 1: Create a tool with dependency resolution
    print("\n1. Testing Tool Creation with Dependency Resolution")
    print("-" * 50)
    
    result = await tool_system.call_tool(
        "create_new_tool",
        name="analyze_json_files",
        description="Analyze JSON files and extract statistics",
        requirements="""
        1. Find all JSON files in a directory
        2. Load and parse each file
        3. Count keys, values, and nested structures
        4. Return statistics using pandas DataFrame
        """
    )
    
    if result.get('success'):
        print("✓ Tool created successfully with dependencies resolved")
    else:
        print(f"✗ Tool creation failed: {result.get('error')}")
    
    # Test 2: Find similar tools
    print("\n2. Testing Semantic Tool Discovery")
    print("-" * 50)
    
    similar = await tool_system.call_tool(
        "find_similar_tools",
        query="count files in repository"
    )
    
    if 'result' in similar and 'matches' in similar['result']:
        print(f"Found {len(similar['result']['matches'])} similar tools:")
        for match in similar['result']['matches'][:3]:
            print(f"  - {match['tool_name']}: {match['similarity']}")
    
    # Test 3: Handle complex query
    print("\n3. Testing Complex Query Handling")
    print("-" * 50)
    
    complex_result = await tool_system.call_tool(
        "handle_complex_query",
        query="Find the top 3 repositories by stars and analyze their code quality"
    )
    
    if 'result' in complex_result:
        print("✓ Complex query handled successfully")
        print(f"  - Strategy: {complex_result['result'].get('strategy', 'N/A')}")
        print(f"  - Tasks executed: {complex_result['result'].get('tasks_executed', 'N/A')}")
        print(f"  - Execution time: {complex_result['result'].get('execution_time', 'N/A'):.2f}s")
    else:
        print(f"✗ Complex query failed: {complex_result.get('error')}")
    
    # Test 4: Get tool stats for evolution
    print("\n4. Testing Tool Usage Statistics")
    print("-" * 50)
    
    stats = await tool_system.call_tool("get_tool_usage_stats")
    
    if 'result' in stats:
        print(f"Total tools: {stats['result'].get('total_tools', 0)}")
        print(f"Total calls: {stats['result'].get('total_calls', 0)}")
        print(f"Success rate: {stats['result'].get('overall_success_rate', 0):.1%}")
    
    print("\n=== Integration Test Complete ===")


if __name__ == "__main__":
    asyncio.run(main())