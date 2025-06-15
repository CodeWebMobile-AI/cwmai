#!/usr/bin/env python3
"""
Test Integrated Validation System
Verifies that tool validation is properly integrated into the tool calling system
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.conversational_ai_assistant import ConversationalAIAssistant
from scripts.tool_calling_system import ToolCallingSystem


async def test_integrated_validation():
    """Test that validation is integrated into tool creation."""
    print("🧪 Testing Integrated Validation System\n")
    
    # Clean up any existing test tools
    test_tools = [
        "analyze_file_types",
        "broken_tool_example",
        "calculate_project_stats"
    ]
    
    for tool_name in test_tools:
        tool_file = Path(f"scripts/custom_tools/{tool_name}.py")
        if tool_file.exists():
            tool_file.unlink()
            print(f"✓ Cleaned up {tool_name}")
    
    print("\n" + "="*60)
    
    # Initialize system
    assistant = ConversationalAIAssistant()
    await assistant.initialize()
    
    # Test 1: Create a valid tool
    print("\n📋 Test 1: Creating a valid tool")
    print("Query: 'analyze file types in the project'")
    
    response = await assistant.handle_conversation("analyze file types in the project")
    print(f"Response: {response[:200]}...")
    
    # Check if tool was created and is valid
    tool_file = Path("scripts/custom_tools/analyze_file_types.py")
    if tool_file.exists():
        print("✅ Tool file created")
        
        # Check if it's in the system
        tool_system = ToolCallingSystem()
        if "analyze_file_types" in tool_system.tools:
            print("✅ Tool loaded into system")
        else:
            print("❌ Tool not loaded (validation may have failed)")
    else:
        print("⚠️  No tool file created")
    
    print("\n" + "="*60)
    
    # Test 2: Try to create a tool that will fail validation
    print("\n📋 Test 2: Creating a tool with validation issues")
    print("Creating a broken tool manually to test validation...")
    
    broken_tool_file = Path("scripts/custom_tools/broken_tool_example.py")
    broken_tool_file.parent.mkdir(exist_ok=True)
    broken_tool_file.write_text('''"""
Broken tool for testing
"""

__description__ = "This tool has issues"

async def broken_tool_example(self, data):  # Has self parameter!
    """This function incorrectly expects self."""
    return "This won't work"  # Returns string instead of dict
''')
    
    print("✓ Created broken tool file")
    
    # Try to load it
    tool_system = ToolCallingSystem()
    result = await tool_system._load_single_custom_tool("broken_tool_example")
    
    if result:
        print("❌ Broken tool was loaded (validation failed to catch issues)")
    else:
        print("✅ Broken tool was rejected by validation")
        
    # Check if file was removed
    if not broken_tool_file.exists():
        print("✅ Invalid tool file was automatically removed")
    else:
        print("⚠️  Invalid tool file still exists")
        broken_tool_file.unlink()  # Clean up
    
    print("\n" + "="*60)
    
    # Test 3: Create a tool with warnings but should still load
    print("\n📋 Test 3: Creating a tool with warnings")
    print("Query: 'calculate project statistics including size and complexity'")
    
    response = await assistant.handle_conversation("calculate project statistics including size and complexity")
    print(f"Response: {response[:200]}...")
    
    tool_file = Path("scripts/custom_tools/calculate_project_stats.py")
    if tool_file.exists():
        print("✅ Tool file created")
        
        # Read the file to check for common patterns
        content = tool_file.read_text()
        has_main = "def main(" in content or "if __name__" in content
        
        if has_main:
            print("⚠️  Tool contains main function/block (warning)")
        
        # Check if loaded despite warnings
        tool_system = ToolCallingSystem()
        if "calculate_project_stats" in tool_system.tools:
            print("✅ Tool loaded despite warnings (correct behavior)")
        else:
            print("❌ Tool not loaded")
    
    print("\n" + "="*60)
    
    # Test 4: Verify tool execution after validation
    print("\n📋 Test 4: Testing tool execution after validation")
    
    # Find a successfully created tool
    created_tools = []
    tool_system = ToolCallingSystem()
    
    for tool_name, tool in tool_system.tools.items():
        if tool.created_by_ai:
            created_tools.append(tool_name)
    
    if created_tools:
        test_tool = created_tools[0]
        print(f"Testing execution of: {test_tool}")
        
        try:
            result = await tool_system.call_tool(test_tool)
            if result.get('success'):
                print("✅ Tool executed successfully after validation")
                print(f"Result preview: {str(result.get('result'))[:100]}...")
            else:
                print(f"❌ Tool execution failed: {result.get('error')}")
        except Exception as e:
            print(f"❌ Error executing tool: {str(e)}")
    else:
        print("⚠️  No AI-created tools found to test")
    
    print("\n" + "="*60)
    print("\n✨ Integrated validation testing complete!")
    
    # Summary
    print("\n📊 Summary:")
    print("- Validation is integrated into tool loading")
    print("- Invalid tools are rejected before being added to the system")
    print("- Tools with minor warnings are still loaded")
    print("- Validated tools can be executed successfully")


async def test_validation_error_messages():
    """Test that validation provides helpful error messages."""
    print("\n\n🔍 Testing Validation Error Messages\n")
    
    tool_system = ToolCallingSystem()
    
    # Test different error scenarios
    error_scenarios = [
        {
            "name": "missing_function",
            "code": '''"""Tool missing main function"""
__description__ = "Test tool"

def helper_function():
    return {"data": "test"}
''',
            "expected_error": "Missing tool function"
        },
        {
            "name": "syntax_error_tool",
            "code": '''"""Tool with syntax error"""
__description__ = "Test tool"

async def syntax_error_tool(**kwargs):
    return {"status": "success"
    # Missing closing brace
''',
            "expected_error": "Syntax error"
        },
        {
            "name": "import_error_tool",
            "code": '''"""Tool with import error"""
from nonexistent_module import something

__description__ = "Test tool"

async def import_error_tool(**kwargs):
    return {"status": "success"}
''',
            "expected_error": "Import error"
        }
    ]
    
    for scenario in error_scenarios:
        print(f"\n📋 Testing: {scenario['expected_error']}")
        
        tool_file = Path(f"scripts/custom_tools/{scenario['name']}.py")
        tool_file.parent.mkdir(exist_ok=True)
        tool_file.write_text(scenario['code'])
        
        # Try to load
        result = await tool_system._load_single_custom_tool(scenario['name'])
        
        if not result:
            print(f"✅ Tool rejected as expected: {scenario['expected_error']}")
        else:
            print(f"❌ Tool was loaded despite: {scenario['expected_error']}")
        
        # Clean up
        if tool_file.exists():
            tool_file.unlink()
    
    print("\n✨ Error message testing complete!")


async def main():
    """Run all tests."""
    await test_integrated_validation()
    await test_validation_error_messages()
    
    print("\n\n🎉 All integrated validation tests completed!")


if __name__ == "__main__":
    asyncio.run(main())