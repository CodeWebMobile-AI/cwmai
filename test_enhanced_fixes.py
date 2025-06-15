#!/usr/bin/env python3
"""
Test enhanced tools with fixes for all issues
"""

import asyncio
import json
from scripts.tool_calling_system import ToolCallingSystem


async def test_semantic_matching():
    """Test semantic tool matching"""
    print("1. Testing Semantic Tool Matching")
    print("-" * 40)
    
    tool_system = ToolCallingSystem()
    
    # Test finding similar tools
    result = await tool_system.call_tool(
        "find_similar_tools",
        query="count repositories"
    )
    
    if result.get('success') and result.get('result', {}).get('matches'):
        matches = result['result']['matches']
        print(f"Found {len(matches)} similar tools:")
        for match in matches[:3]:
            print(f"  - {match['tool_name']}: {match['similarity']}")
    else:
        print("No matches found")
        
    # Test semantic matching during tool creation
    print("\nTesting duplicate prevention...")
    result = await tool_system.call_tool("count_repos")  # Should match count_repositories
    if 'error' not in result:
        print("✓ Semantic matching prevented duplicate tool creation")
    else:
        print(f"✗ Tool not found: {result.get('error', 'Unknown error')}")
    print()


async def test_complex_query():
    """Test complex query handling"""
    print("2. Testing Complex Query Handling")
    print("-" * 40)
    
    tool_system = ToolCallingSystem()
    
    # Simple complex query
    query = "Count all repositories and get their total stars"
    result = await tool_system.call_tool("handle_complex_query", query=query)
    
    if result.get('success'):
        res = result.get('result', {})
        print(f"Query: {query}")
        print(f"✓ Executed successfully")
        print(f"  Strategy: {res.get('strategy', 'N/A')}")
        print(f"  Tasks: {res.get('tasks_executed', 0)}")
        if 'error' not in res:
            print("  Result: Success")
        else:
            print(f"  Error: {res.get('error')}")
    else:
        print(f"✗ Failed: {result.get('error', 'Unknown error')}")
    print()


async def test_tool_evolution():
    """Test tool evolution"""
    print("3. Testing Tool Evolution")
    print("-" * 40)
    
    tool_system = ToolCallingSystem()
    
    # First create some usage data
    print("Generating usage data...")
    for i in range(3):
        await tool_system.call_tool("get_repositories", limit=5)
    
    # Now try to evolve
    print("Attempting to evolve 'get_repositories' tool...")
    # Call evolve_tool with target_tool parameter
    result = await tool_system.call_tool("evolve_tool", target_tool="get_repositories")
    
    if result.get('success'):
        res = result.get('result', {})
        if res.get('success'):
            print(f"✓ Tool evolved successfully")
            print(f"  Improvements applied: {res.get('improvements_applied', 0)}")
            print(f"  Performance gain: {res.get('performance_gain', 'N/A')}")
        else:
            print(f"✗ Evolution failed: {res.get('errors', ['Unknown error'])}")
    else:
        print(f"✗ Failed to call evolve_tool: {result.get('error', 'Unknown error')}")
    print()


async def test_dependency_resolution():
    """Test dependency resolution in tool creation"""
    print("4. Testing Dependency Resolution")
    print("-" * 40)
    
    tool_system = ToolCallingSystem()
    
    # Create a tool that needs imports
    result = await tool_system.call_tool(
        "create_new_tool",
        name="test_pandas_tool",
        description="Test tool using pandas",
        requirements="Load CSV files using pandas and return summary statistics"
    )
    
    if result.get('success'):
        res = result.get('result', {})
        if res.get('success'):
            print("✓ Tool created with dependency resolution")
            print(f"  File: {res.get('file', 'N/A')}")
            
            # Check if the tool loads successfully
            tool_system._load_custom_tools()
            if 'test_pandas_tool' in tool_system.tools:
                print("  ✓ Tool loaded successfully")
            else:
                print("  ✗ Tool failed to load")
        else:
            print(f"✗ Creation failed: {res.get('error', 'Unknown error')}")
    else:
        print(f"✗ Failed: {result.get('error', 'Unknown error')}")
    print()


async def test_enhanced_features():
    """Test all enhanced features"""
    print("=== Testing Enhanced Tool System ===\n")
    
    # Check if enhanced systems are loaded
    tool_system = ToolCallingSystem()
    print("Enhanced Systems Status:")
    print(f"  Dependency Resolver: {'✓' if tool_system.dependency_resolver else '✗'}")
    print(f"  Multi-Tool Orchestrator: {'✓' if tool_system.multi_tool_orchestrator else '✗'}")
    print(f"  Tool Evolution: {'✓' if tool_system.tool_evolution else '✗'}")
    print(f"  Semantic Matcher: {'✓' if tool_system.semantic_matcher else '✗'}")
    print()
    
    # Run tests
    await test_semantic_matching()
    await test_complex_query()
    await test_tool_evolution()
    await test_dependency_resolution()
    
    print("=== All Tests Complete ===")


if __name__ == "__main__":
    asyncio.run(test_enhanced_features())