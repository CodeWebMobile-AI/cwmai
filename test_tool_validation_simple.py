#!/usr/bin/env python3
"""
Simple Test for Tool Validation System
Tests the key aspects of autonomous tool creation and validation
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.tool_calling_system import ToolCallingSystem
from scripts.enhanced_tool_validation import EnhancedToolValidator, SafeToolLoader


async def test_tool_validation():
    """Test tool validation with a sample tool."""
    print("🧪 Testing Tool Validation System\n")
    
    # Create a sample tool for testing
    test_tool_name = "test_sample_tool"
    test_tool_file = Path(f"scripts/custom_tools/{test_tool_name}.py")
    
    # Create different test cases
    test_cases = [
        {
            "name": "Valid Tool",
            "code": '''"""
Test tool for validation
"""

__description__ = "Sample test tool"
__parameters__ = {}
__examples__ = ["test_sample_tool()"]

async def test_sample_tool(**kwargs):
    """Test tool that returns sample data."""
    return {"status": "success", "data": "test"}
''',
            "should_pass": True
        },
        {
            "name": "Tool with Self Parameter",
            "code": '''"""
Test tool with self parameter
"""

__description__ = "Sample test tool"
__parameters__ = {}

async def test_sample_tool(self, **kwargs):
    """Test tool with self parameter."""
    return {"status": "success"}
''',
            "should_pass": False
        },
        {
            "name": "Tool with Main Function",
            "code": '''"""
Test tool with main function
"""

__description__ = "Sample test tool"
__parameters__ = {}

async def test_sample_tool(**kwargs):
    """Test tool that returns sample data."""
    return {"status": "success"}

def main():
    """This should trigger a warning."""
    print("Running main")

if __name__ == "__main__":
    main()
''',
            "should_pass": True  # Should pass but with warnings
        },
        {
            "name": "Tool with Syntax Error",
            "code": '''"""
Test tool with syntax error
"""

__description__ = "Sample test tool"

async def test_sample_tool(**kwargs):
    """Test tool with syntax error."""
    return {"status": "success"
    # Missing closing brace
''',
            "should_pass": False
        },
        {
            "name": "Tool Missing Return Dict",
            "code": '''"""
Test tool that doesn't return dict
"""

__description__ = "Sample test tool"
__parameters__ = {}

async def test_sample_tool(**kwargs):
    """Test tool that returns wrong type."""
    return "This should be a dict"
''',
            "should_pass": False
        }
    ]
    
    # Initialize validator
    validator = EnhancedToolValidator()
    
    print("Running validation tests...\n")
    
    for test_case in test_cases:
        print(f"📋 Test: {test_case['name']}")
        print(f"Expected to pass: {test_case['should_pass']}")
        
        # Write test tool
        test_tool_file.parent.mkdir(exist_ok=True)
        test_tool_file.write_text(test_case['code'])
        
        # Validate
        result = await validator.validate_tool(test_tool_file, test_tool_name)
        
        print(f"Validation result: {'✅ PASSED' if result.is_valid else '❌ FAILED'}")
        
        if result.issues:
            print(f"Issues: {result.issues}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
            
        # Check if result matches expectation
        if result.is_valid == test_case['should_pass']:
            print("✅ Test behaved as expected")
        else:
            print("❌ Test did not behave as expected")
            
        print("-" * 50 + "\n")
        
        # Clean up
        if test_tool_file.exists():
            test_tool_file.unlink()
    
    print("\n✨ Validation tests complete!")


async def test_safe_loading():
    """Test the safe loading mechanism."""
    print("\n\n🔒 Testing Safe Tool Loading\n")
    
    # Initialize components
    tool_system = ToolCallingSystem()
    loader = SafeToolLoader(tool_system)
    
    # Create a valid tool
    test_tool_name = "safe_test_tool"
    test_tool_file = Path(f"scripts/custom_tools/{test_tool_name}.py")
    test_tool_file.parent.mkdir(exist_ok=True)
    
    test_tool_file.write_text('''"""
Safe test tool
"""

__description__ = "Tool for testing safe loading"
__parameters__ = {"message": {"type": "string", "required": False}}
__examples__ = ["safe_test_tool()", "safe_test_tool(message='hello')"]

async def safe_test_tool(message: str = "default", **kwargs):
    """Safe test tool implementation."""
    return {
        "status": "success",
        "message": message,
        "timestamp": "2024-01-01"
    }
''')
    
    # Test loading
    print(f"Loading tool: {test_tool_name}")
    result = await loader.load_and_validate_tool(test_tool_file, test_tool_name)
    
    if result['success']:
        print("✅ Tool loaded successfully")
        print(f"Performance: {result.get('performance_ms', 0):.2f}ms")
        
        # Test if tool works
        if test_tool_name in tool_system.tools:
            print("\nTesting loaded tool...")
            test_result = await tool_system.call_tool(test_tool_name, message="Hello from test")
            print(f"Tool execution result: {test_result}")
        else:
            print("❌ Tool not found in system after loading")
    else:
        print("❌ Tool loading failed")
        print(f"Reason: {result.get('message')}")
        if 'validation_result' in result:
            val_result = result['validation_result']
            print(f"Issues: {val_result.get('issues', [])}")
    
    # Clean up
    if test_tool_file.exists():
        test_tool_file.unlink()
    
    print("\n✨ Safe loading test complete!")


async def test_real_tool_generation():
    """Test with a real tool generation scenario."""
    print("\n\n🚀 Testing Real Tool Generation\n")
    
    tool_system = ToolCallingSystem()
    
    # Test creating a genuinely useful tool
    print("Creating a tool to count Python files...")
    
    result = await tool_system._create_new_tool(
        name="count_python_files",
        description="Count all Python files in the project",
        requirements="""
        1. Count all .py files in the project
        2. Group by directory
        3. Show total lines of code
        4. Return structured data with counts and statistics
        """
    )
    
    if result.get('success'):
        print("✅ Tool created successfully")
        
        # Validate the created tool
        tool_file = Path(result.get('file'))
        loader = SafeToolLoader(tool_system)
        
        print("\nValidating created tool...")
        validation_result = await loader.load_and_validate_tool(tool_file, "count_python_files")
        
        if validation_result['success']:
            print("✅ Tool validated and loaded")
            
            # Test execution
            print("\nExecuting tool...")
            exec_result = await tool_system.call_tool("count_python_files")
            if exec_result.get('success'):
                print("✅ Tool executed successfully")
                print(f"Result preview: {str(exec_result.get('result'))[:200]}...")
            else:
                print(f"❌ Execution failed: {exec_result.get('error')}")
        else:
            print("❌ Tool validation failed")
            print(f"Issues: {validation_result}")
    else:
        print("❌ Tool creation failed")
        print(f"Error: {result.get('error')}")
    
    print("\n✨ Real tool generation test complete!")


async def main():
    """Run all tests."""
    await test_tool_validation()
    await test_safe_loading()
    await test_real_tool_generation()
    
    print("\n\n🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())