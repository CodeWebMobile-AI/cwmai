#!/usr/bin/env python3
"""
Test Genuine Tool Creation
Test creating a tool that doesn't exist to verify the fix works
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.conversational_ai_assistant import ConversationalAIAssistant
from scripts.tool_calling_system import ToolCallingSystem


async def test_genuine_tool_creation():
    """Test creating a genuinely new tool."""
    print("🧪 Testing Genuine Tool Creation\n")
    
    # Remove any existing test tool
    test_tool_file = Path("scripts/custom_tools/analyze_code_quality.py")
    if test_tool_file.exists():
        test_tool_file.unlink()
        print("✓ Cleaned up existing test tool\n")
    
    # Initialize assistant
    assistant = ConversationalAIAssistant()
    await assistant.initialize()
    
    # Test a query that should create a new tool
    print("📝 Testing: 'analyze code quality for all projects'")
    print("This should create a new tool since it doesn't exist\n")
    
    try:
        # First check existing tools
        tool_system = ToolCallingSystem()
        if 'analyze_code_quality' in tool_system.tools:
            print("⚠️  Tool already exists, removing it...")
            del tool_system.tools['analyze_code_quality']
        
        # Now test the query
        response = await assistant.handle_conversation("analyze code quality for all projects")
        print(f"Response: {response[:200]}...")
        
        # Check if tool was created
        if test_tool_file.exists():
            print(f"\n✅ Tool file created: {test_tool_file}")
            
            # Verify the generated code
            content = test_tool_file.read_text()
            
            # Check for issues
            issues = []
            if "def main(" in content or "async def main(" in content:
                issues.append("Contains main() function")
            if "if __name__ == '__main__':" in content:
                issues.append("Contains __main__ block")
            if "(self" in content and "def analyze_code_quality" in content:
                issues.append("Function expects self parameter")
                
            if issues:
                print(f"⚠️  Issues found: {', '.join(issues)}")
            else:
                print("✅ Generated tool is clean and correct")
                
            # Show the function signature
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'async def analyze_code_quality' in line:
                    print(f"\n📝 Generated function signature:")
                    print(f"   {line}")
                    break
                    
        else:
            print("⚠️  No tool file was generated")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    """Run the test."""
    await test_genuine_tool_creation()


if __name__ == "__main__":
    asyncio.run(main())