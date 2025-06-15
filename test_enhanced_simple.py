#!/usr/bin/env python3
"""
Simple test for enhanced tool capabilities
"""

import asyncio
import json
from scripts.tool_calling_system import ToolCallingSystem


async def test_basic_functionality():
    """Test basic enhanced tool functionality"""
    print("=== Testing Basic Enhanced Tool Features ===\n")
    
    # Initialize tool system
    tool_system = ToolCallingSystem()
    
    # Check if enhanced systems are available
    print("Enhanced Systems Status:")
    print(f"- Dependency Resolver: {'✓' if tool_system.dependency_resolver else '✗'}")
    print(f"- Multi-Tool Orchestrator: {'✓' if tool_system.multi_tool_orchestrator else '✗'}")
    print(f"- Tool Evolution: {'✓' if tool_system.tool_evolution else '✗'}")
    print(f"- Semantic Matcher: {'✓' if tool_system.semantic_matcher else '✗'}")
    print()
    
    # Test 1: Find similar tools
    print("1. Testing Find Similar Tools")
    print("-" * 40)
    try:
        result = await tool_system.call_tool(
            "find_similar_tools",
            query="count repositories"
        )
        if result.get('error'):
            print(f"Error: {result['error']}")
        else:
            print(f"Query: 'count repositories'")
            print(f"Found {len(result.get('matches', []))} similar tools:")
            for match in result.get('matches', [])[:3]:
                print(f"  - {match['tool_name']}: {match['similarity']}")
    except Exception as e:
        print(f"Error testing find_similar_tools: {e}")
    print()
    
    # Test 2: Get tool usage stats
    print("2. Testing Tool Usage Stats")
    print("-" * 40)
    try:
        stats = await tool_system.call_tool("get_tool_usage_stats")
        print(f"Total tools: {stats.get('result', {}).get('total_tools', 'N/A')}")
        print(f"Total calls: {stats.get('result', {}).get('total_calls', 'N/A')}")
        if 'result' in stats and 'top_tools' in stats['result']:
            print("\nTop used tools:")
            for tool, count in stats['result']['top_tools'][:5]:
                print(f"  - {tool}: {count} calls")
    except Exception as e:
        print(f"Error getting tool stats: {e}")
    print()
    
    # Test 3: Simple tool creation with dependency resolution
    print("3. Testing Tool Creation")
    print("-" * 40)
    try:
        result = await tool_system.call_tool(
            "create_new_tool",
            name="simple_test_tool",
            description="A simple test tool",
            requirements="Just return a greeting message with the current time"
        )
        if result.get('success'):
            print(f"✓ Successfully created tool: simple_test_tool")
            print(f"  File: {result.get('file', 'N/A')}")
        else:
            print(f"✗ Failed to create tool: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"Error creating tool: {e}")
    print()
    
    print("=== Basic Test Complete ===")


async def test_dependency_resolution():
    """Test dependency resolution specifically"""
    print("\n=== Testing Dependency Resolution ===\n")
    
    from scripts.dependency_resolver import DependencyResolver
    
    resolver = DependencyResolver()
    
    # Test code with missing imports
    test_code = '''
async def analyze_data(file_path):
    """Analyze data from a file."""
    path = Path(file_path)
    
    if not path.exists():
        return {"error": "File not found"}
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    summary = df.describe()
    
    return {"summary": summary.to_dict()}
'''
    
    print("Original code (with missing imports):")
    print(test_code)
    print("\n" + "-" * 40 + "\n")
    
    # Fix imports
    fixed_code = resolver.fix_import_paths(test_code)
    
    print("Fixed code (with imports added):")
    print(fixed_code)
    
    print("\n=== Dependency Resolution Test Complete ===")


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(test_basic_functionality())
    
    # Test dependency resolution
    asyncio.run(test_dependency_resolution())