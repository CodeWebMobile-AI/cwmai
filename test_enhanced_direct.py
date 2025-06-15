#!/usr/bin/env python3
"""
Direct test of enhanced tool components without ToolCallingSystem
"""

import asyncio
import json


async def test_dependency_resolver():
    """Test dependency resolver directly"""
    print("=== Testing Dependency Resolver ===")
    from scripts.dependency_resolver import DependencyResolver
    
    resolver = DependencyResolver()
    
    # Test code with missing imports
    test_code = '''
async def process_data(file_path):
    data = json.load(open(file_path))
    df = pd.DataFrame(data)
    
    for idx, row in df.iterrows():
        result = await analyze_item(row)
        logger.info(f"Processed {idx}: {result}")
    
    return df.to_dict()
'''
    
    print("Original code:")
    print(test_code)
    print("\nFixed code:")
    fixed = resolver.fix_import_paths(test_code)
    print(fixed)
    print("\n✓ Dependency Resolver working")


async def test_semantic_matcher():
    """Test semantic matcher capabilities"""
    print("\n=== Testing Semantic Tool Matcher ===")
    from scripts.semantic_tool_matcher import SemanticToolMatcher
    
    # Create a mock tool system
    class MockToolSystem:
        def list_tools(self):
            return {
                "analyze_repository": {"description": "Analyze repository for issues"},
                "count_repositories": {"description": "Count total repositories"},
                "search_code": {"description": "Search for code patterns"},
                "create_issue": {"description": "Create GitHub issue"},
            }
        
        def get_tool(self, name):
            tools = self.list_tools()
            return tools.get(name)
    
    matcher = SemanticToolMatcher(MockToolSystem())
    
    # Test finding similar tools
    query = "scan repositories for problems"
    matches = await matcher.find_similar_tools(query, top_k=3)
    
    print(f"Query: '{query}'")
    print(f"Found {len(matches)} matches:")
    for match in matches:
        print(f"  - {match.tool_name}: {match.similarity_score:.2%}")
    
    print("\n✓ Semantic Tool Matcher working")


async def test_tool_evolution():
    """Test tool evolution tracking"""
    print("\n=== Testing Tool Evolution ===")
    from scripts.tool_evolution import ToolEvolution
    
    evolution = ToolEvolution()
    
    # Simulate tool executions
    await evolution.track_tool_execution(
        "test_tool", {"param": "value"}, "success", 1.5, None
    )
    
    await evolution.track_tool_execution(
        "test_tool", {"param": "value2"}, None, 2.5, "ImportError: missing module"
    )
    
    # Analyze performance
    analysis = await evolution.analyze_tool_performance("test_tool")
    
    print("Tool performance analysis:")
    print(f"  Total executions: {analysis['total_executions']}")
    print(f"  Success rate: {analysis['success_rate']:.1%}")
    print(f"  Average time: {analysis['average_execution_time']:.2f}s")
    
    print("\n✓ Tool Evolution working")


async def test_multi_tool_orchestrator():
    """Test multi-tool orchestration"""
    print("\n=== Testing Multi-Tool Orchestrator ===")
    from scripts.multi_tool_orchestrator import MultiToolOrchestrator
    
    orchestrator = MultiToolOrchestrator()
    
    # Test query decomposition
    query = "Analyze all Python files and create a summary report"
    tasks = await orchestrator.decompose_query(query)
    
    print(f"Query: '{query}'")
    print(f"Decomposed into {len(tasks)} tasks:")
    for task in tasks:
        print(f"  - {task.get('id', 'unknown')}: {task.get('description', 'no description')}")
    
    print("\n✓ Multi-Tool Orchestrator working")


async def main():
    """Run all tests"""
    print("Testing Enhanced Tool Components Directly\n")
    
    try:
        await test_dependency_resolver()
    except Exception as e:
        print(f"✗ Dependency Resolver failed: {e}")
    
    try:
        await test_semantic_matcher()
    except Exception as e:
        print(f"✗ Semantic Matcher failed: {e}")
    
    try:
        await test_tool_evolution()
    except Exception as e:
        print(f"✗ Tool Evolution failed: {e}")
    
    try:
        await test_multi_tool_orchestrator()
    except Exception as e:
        print(f"✗ Multi-Tool Orchestrator failed: {e}")
    
    print("\n=== Direct Component Testing Complete ===")


if __name__ == "__main__":
    asyncio.run(main())