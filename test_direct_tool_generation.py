#!/usr/bin/env python3
"""
Test direct tool generation from user intent
"""

import asyncio
from scripts.tool_calling_system import ToolCallingSystem
from pathlib import Path


async def test_tool_generation_from_intent():
    """Test if we can generate tools directly from user intent"""
    tool_system = ToolCallingSystem()
    
    # Test queries that should generate specific tools
    test_cases = [
        {
            "query": "count python files in scripts directory",
            "expected_tool": "count_python_files_in_scripts"
        },
        {
            "query": "calculate total lines of code",
            "expected_tool": "calculate_total_lines"
        },
        {
            "query": "find all TODO comments",
            "expected_tool": "find_todo_comments"
        }
    ]
    
    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"INTENT: {test['query']}")
        print('='*60)
        
        # Try to call a tool that doesn't exist to trigger creation
        tool_name = test['expected_tool']
        
        print(f"\n1. Attempting to call non-existent tool: {tool_name}")
        result = await tool_system.call_tool(tool_name)
        
        print(f"\nResult: {result}")
        
        # Check if tool was created
        if Path(f"scripts/custom_tools/{tool_name}.py").exists():
            print(f"✓ Tool file created: {tool_name}.py")
        else:
            print(f"✗ Tool file not created")
            
        # If the tool was created, try calling it again
        if tool_name in tool_system.tools:
            print(f"\n2. Testing the generated tool...")
            result2 = await tool_system.call_tool(tool_name)
            print(f"Result: {result2}")


async def test_intent_based_creation():
    """Test the _create_tool_from_intent method directly"""
    tool_system = ToolCallingSystem()
    
    print("\n" + "="*60)
    print("TESTING INTENT-BASED TOOL CREATION")
    print("="*60)
    
    # Test cases with clear intent
    intents = [
        "count_worker_processes",
        "analyze_memory_usage",
        "list_recent_errors_in_logs"
    ]
    
    for intent in intents:
        print(f"\nCreating tool from intent: {intent}")
        
        # This simulates what happens when a tool doesn't exist
        result = await tool_system._create_tool_from_intent(intent, {})
        
        if result.get('success'):
            print(f"✓ Tool creation initiated")
            # The tool might be created asynchronously, so check after a delay
            await asyncio.sleep(1)
            
            # Reload tools and check
            tool_system._load_custom_tools()
            if intent in tool_system.tools:
                print(f"✓ Tool {intent} is now available")
            else:
                print(f"✗ Tool {intent} not found after creation")
        else:
            print(f"✗ Failed: {result}")


async def test_conversational_tool_creation():
    """Test creating tools from natural language"""
    tool_system = ToolCallingSystem()
    
    print("\n" + "="*60)
    print("TESTING CONVERSATIONAL TOOL CREATION")
    print("="*60)
    
    conversations = [
        {
            "user": "How many active workers do we have?",
            "tool_spec": "count active worker processes by checking system processes"
        },
        {
            "user": "What's the disk usage of log files?",
            "tool_spec": "calculate total disk space used by all .log files"
        },
        {
            "user": "Show me files changed today",
            "tool_spec": "list all files modified in the last 24 hours"
        }
    ]
    
    for conv in conversations:
        print(f"\nUser: {conv['user']}")
        
        # Generate a tool name from the query
        tool_name = conv['user'].lower().replace(' ', '_').replace('?', '').replace("'", '')[:30]
        
        # Create the tool
        result = await tool_system.call_tool(
            "create_new_tool",
            name=tool_name,
            description=conv['user'],
            requirements=conv['tool_spec']
        )
        
        if result.get('success'):
            print(f"✓ Created tool: {tool_name}")
            
            # Try to use it
            tool_system._load_custom_tools()
            if tool_name in tool_system.tools:
                print("✓ Tool loaded successfully")
                
                # Execute it
                exec_result = await tool_system.call_tool(tool_name)
                if 'error' not in exec_result:
                    print("✓ Tool executed successfully")
                else:
                    print(f"✗ Execution error: {exec_result.get('error')}")
        else:
            print(f"✗ Creation failed: {result.get('error')}")


async def main():
    """Run all tests"""
    # await test_tool_generation_from_intent()
    # await test_intent_based_creation()
    await test_conversational_tool_creation()


if __name__ == "__main__":
    asyncio.run(main())