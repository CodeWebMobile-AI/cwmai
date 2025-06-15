#!/usr/bin/env python3
"""
Test the system's ability to automatically generate tools from natural language queries
"""

import asyncio
from scripts.tool_calling_system import ToolCallingSystem
import time


async def test_query(query: str, expected_behavior: str):
    """Test a single query and report results"""
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"EXPECTED: {expected_behavior}")
    print('='*80)
    
    tool_system = ToolCallingSystem()
    start_time = time.time()
    
    # First try to handle as a complex query
    result = await tool_system.call_tool('handle_complex_query', query=query)
    
    elapsed = time.time() - start_time
    
    if result.get('success'):
        res = result.get('result', {})
        print(f"✓ Success in {elapsed:.2f}s")
        print(f"Strategy: {res.get('strategy', 'Unknown')}")
        print(f"Tasks: {res.get('tasks_executed', 0)}")
        
        # Check if new tools were created
        if 'detailed_results' in res:
            for task, task_result in res['detailed_results'].items():
                if 'created' in str(task_result).lower() or 'tool' in task:
                    print(f"  → New tool created: {task}")
        
        # Show the actual result
        if 'result' in res:
            print(f"\nRESULT: {res['result']}")
        elif 'summary' in res:
            print(f"\nSUMMARY: {res['summary']}")
            
        return True
    else:
        print(f"✗ Failed in {elapsed:.2f}s: {result.get('error', 'Unknown error')}")
        return False


async def main():
    """Test challenging queries from each category"""
    
    test_cases = [
        # Code Analysis
        ("Find all functions that use async/await", 
         "Should create a tool to scan Python files for async function definitions"),
        
        # Environment & Configuration
        ("What environment variables start with 'GITHUB'?",
         "Should create a tool to list environment variables by prefix"),
        
        # Project Statistics
        ("Calculate the code-to-comment ratio",
         "Should create a tool to analyze code vs comment lines"),
        
        # Process & Performance
        ("What's the CPU usage of our workers?",
         "Should create a tool to monitor worker process CPU usage"),
        
        # Git & Version Control
        ("How many commits were made today?",
         "Should create a tool to count recent git commits"),
        
        # Error & Log Analysis
        ("What's the most common error message?",
         "Should create a tool to analyze log files for error patterns"),
        
        # Complex Multi-Step
        ("Analyze the health of our worker system and suggest improvements",
         "Should orchestrate multiple tools to analyze system health")
    ]
    
    results = []
    
    print("\n" + "="*80)
    print("TESTING AUTOMATIC TOOL GENERATION FROM NATURAL LANGUAGE")
    print("="*80)
    
    for query, expected in test_cases:
        success = await test_query(query, expected)
        results.append((query, success))
        await asyncio.sleep(2)  # Pause between tests
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for query, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {query[:60]}...")
    
    print(f"\nTotal: {passed}/{total} passed ({passed/total*100:.0f}%)")
    
    # Check what tools were created
    import os
    custom_tools_dir = "scripts/custom_tools"
    if os.path.exists(custom_tools_dir):
        tools = [f for f in os.listdir(custom_tools_dir) if f.endswith('.py')]
        print(f"\nTools in custom_tools directory: {len(tools)}")
        for tool in sorted(tools)[-10:]:  # Show last 10 created
            print(f"  - {tool}")


if __name__ == "__main__":
    asyncio.run(main())