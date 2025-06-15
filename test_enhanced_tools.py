#!/usr/bin/env python3
"""
Test script for enhanced tool capabilities
Demonstrates dependency resolution, multi-tool orchestration, evolution, and semantic matching
"""

import asyncio
import json
from scripts.tool_calling_system import ToolCallingSystem


async def test_enhanced_tools():
    """Test all enhanced tool capabilities"""
    print("=== Testing Enhanced Tool System ===\n")
    
    # Initialize tool system
    tool_system = ToolCallingSystem()
    
    # Test 1: Semantic Tool Matching
    print("1. Testing Semantic Tool Matching")
    print("-" * 40)
    
    # Try to call a tool with a different name
    result = await tool_system.call_tool("scan_repositories_for_bugs")
    print(f"Called non-existent tool 'scan_repositories_for_bugs'")
    print(f"Result: {json.dumps(result, indent=2)}\n")
    
    # Find similar tools
    similar = await tool_system.call_tool("find_similar_tools", query="analyze code quality")
    print(f"Similar tools to 'analyze code quality':")
    print(f"{json.dumps(similar, indent=2)}\n")
    
    # Test 2: Complex Query Handling
    print("2. Testing Multi-Tool Orchestration")
    print("-" * 40)
    
    complex_query = "Count all repositories, analyze the top 3 by stars, and create a summary report"
    result = await tool_system.call_tool("handle_complex_query", query=complex_query)
    print(f"Complex query: {complex_query}")
    print(f"Result: {json.dumps(result, indent=2)}\n")
    
    # Test 3: Tool Creation with Dependency Resolution
    print("3. Testing Tool Creation with Import Resolution")
    print("-" * 40)
    
    # Create a tool that needs imports
    tool_spec = {
        "name": "analyze_code_complexity",
        "description": "Analyze code complexity metrics",
        "requirements": """
        This tool should:
        1. Use pathlib to navigate directories
        2. Use ast to parse Python files
        3. Calculate cyclomatic complexity
        4. Return metrics as JSON
        """
    }
    
    create_result = await tool_system.call_tool(
        "create_new_tool",
        **tool_spec
    )
    print(f"Created tool: {tool_spec['name']}")
    print(f"Result: {json.dumps(create_result, indent=2)}\n")
    
    # Test 4: Tool Evolution
    print("4. Testing Tool Evolution")
    print("-" * 40)
    
    # First, simulate some tool usage to generate metrics
    print("Simulating tool usage...")
    for i in range(5):
        await tool_system.call_tool("get_repositories", limit=5)
    
    # Check tool stats
    stats = await tool_system.call_tool("get_tool_usage_stats")
    print(f"Tool usage stats: {json.dumps(stats, indent=2)}\n")
    
    # Evolve a tool
    evolve_result = await tool_system.call_tool("evolve_tool", tool_name="get_repositories")
    print(f"Evolution result: {json.dumps(evolve_result, indent=2)}\n")
    
    # Test 5: Workflow Example
    print("5. Testing Complete Workflow")
    print("-" * 40)
    
    workflow_query = """
    Analyze all Python repositories for common issues like:
    - Missing documentation
    - No tests
    - Security vulnerabilities
    Then create GitHub issues for critical problems.
    """
    
    workflow_result = await tool_system.call_tool("handle_complex_query", query=workflow_query)
    print(f"Workflow query: {workflow_query.strip()}")
    print(f"Result: {json.dumps(workflow_result, indent=2)}\n")
    
    print("=== Enhanced Tool System Test Complete ===")


async def demonstrate_tool_improvement_cycle():
    """Demonstrate the continuous improvement cycle"""
    print("\n=== Tool Improvement Cycle Demo ===\n")
    
    tool_system = ToolCallingSystem()
    
    # Create a simple tool with intentional issues
    print("1. Creating a tool with potential issues...")
    await tool_system.call_tool(
        "create_new_tool",
        name="calculate_repo_score",
        description="Calculate a quality score for repositories",
        requirements="Calculate score based on stars, issues, and last update. May fail if repo data is incomplete."
    )
    
    # Use the tool multiple times to generate performance data
    print("\n2. Using the tool to generate performance data...")
    test_repos = ["repo1", "repo2", "repo3", "incomplete_repo", "another_repo"]
    
    for repo in test_repos:
        try:
            result = await tool_system.call_tool("calculate_repo_score", repo=repo)
            print(f"  - {repo}: {result.get('result', result.get('error', 'Unknown'))}")
        except Exception as e:
            print(f"  - {repo}: Error - {e}")
    
    # Check if the tool needs improvement
    print("\n3. Analyzing tool performance...")
    if tool_system.tool_evolution:
        analysis = await tool_system.tool_evolution.analyze_tool_performance("calculate_repo_score")
        print(f"Performance analysis: {json.dumps(analysis, indent=2)}")
        
        # Suggest improvements
        print("\n4. Generating improvement suggestions...")
        improvements = await tool_system.tool_evolution.suggest_improvements("calculate_repo_score")
        for imp in improvements:
            print(f"  - {imp.improvement_type}: {imp.description}")
            print(f"    Expected impact: {imp.expected_impact:.1%}")
    
    print("\n=== Improvement Cycle Demo Complete ===")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_enhanced_tools())
    
    # Demonstrate improvement cycle
    asyncio.run(demonstrate_tool_improvement_cycle())