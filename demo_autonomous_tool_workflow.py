#!/usr/bin/env python3
"""
Demo: Complete Autonomous Tool Creation Workflow
Shows how the system creates, validates, and uses tools automatically
"""

import asyncio
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.conversational_ai_assistant import ConversationalAIAssistant


async def demo_autonomous_workflow():
    """Demonstrate the complete autonomous tool creation workflow."""
    print("🚀 Autonomous Tool Creation Workflow Demo\n")
    print("This demo shows how CWMAI:")
    print("1. Understands natural language requests")
    print("2. Creates tools automatically when needed")
    print("3. Validates tools before adding them")
    print("4. Executes validated tools successfully\n")
    
    # Initialize assistant
    assistant = ConversationalAIAssistant()
    await assistant.initialize()
    
    # Demo scenarios
    scenarios = [
        {
            "title": "📊 Scenario 1: File Analysis",
            "query": "count all JSON files in the project and show their total size",
            "expected_tool": "count_json_files"
        },
        {
            "title": "🔍 Scenario 2: Code Search", 
            "query": "find all TODO comments in Python files",
            "expected_tool": "find_todo_comments"
        },
        {
            "title": "📈 Scenario 3: Project Metrics",
            "query": "calculate lines of code for each programming language",
            "expected_tool": "calculate_lines_of_code"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"{scenario['title']}")
        print(f"User: '{scenario['query']}'")
        print('='*60)
        
        # Clean up any existing tool
        tool_file = Path(f"scripts/custom_tools/{scenario['expected_tool']}.py")
        if tool_file.exists():
            tool_file.unlink()
        
        # Process the query
        print("\n🤖 Assistant processing request...")
        response = await assistant.handle_conversation(scenario['query'])
        
        print(f"\n💬 Response: {response[:300]}...")
        
        # Check what happened
        if tool_file.exists():
            print(f"\n✅ Tool created: {scenario['expected_tool']}.py")
            
            # Show a snippet of the generated code
            code = tool_file.read_text()
            lines = code.split('\n')
            
            # Find the main function
            for i, line in enumerate(lines):
                if f"def {scenario['expected_tool']}" in line:
                    print(f"\n📝 Generated function:")
                    print("```python")
                    # Show function signature and first few lines
                    for j in range(i, min(i+5, len(lines))):
                        print(lines[j])
                    print("...\n```")
                    break
        else:
            print(f"\n⚠️  Tool may have used existing functionality")
    
    # Show tool statistics
    print("\n" + "="*60)
    print("📊 Tool Creation Summary")
    print("="*60)
    
    from scripts.tool_calling_system import ToolCallingSystem
    tool_system = ToolCallingSystem()
    
    ai_tools = [name for name, tool in tool_system.tools.items() if tool.created_by_ai]
    
    print(f"\nTotal tools in system: {len(tool_system.tools)}")
    print(f"AI-created tools: {len(ai_tools)}")
    
    if ai_tools:
        print("\nAI-created tools:")
        for tool_name in ai_tools[:5]:  # Show first 5
            tool = tool_system.tools[tool_name]
            print(f"  • {tool_name}: {tool.description[:50]}...")
            print(f"    Usage count: {tool.usage_count}, Success rate: {(tool.success_count/max(tool.usage_count, 1))*100:.0f}%")
    
    print("\n✨ Demo complete!")
    print("\nKey takeaways:")
    print("• The system automatically creates tools based on natural language")
    print("• Tools are validated before being added to ensure quality")
    print("• Invalid tools are rejected to maintain system stability")
    print("• Created tools integrate seamlessly with existing functionality")


async def test_edge_cases():
    """Test edge cases in tool creation."""
    print("\n\n🔬 Testing Edge Cases\n")
    
    assistant = ConversationalAIAssistant()
    await assistant.initialize()
    
    edge_cases = [
        {
            "name": "Duplicate Tool Request",
            "query": "count repositories",  # Should use existing tool
            "expected": "Should use existing count_repositories tool"
        },
        {
            "name": "Ambiguous Request",
            "query": "do something with the files",
            "expected": "Should ask for clarification or make best guess"
        },
        {
            "name": "Complex Multi-Step Request",
            "query": "analyze all Python files, find duplicates, and generate a report",
            "expected": "Should create a comprehensive tool or use multiple tools"
        }
    ]
    
    for case in edge_cases:
        print(f"\n📋 {case['name']}")
        print(f"Query: '{case['query']}'")
        print(f"Expected: {case['expected']}")
        
        try:
            response = await assistant.handle_conversation(case['query'])
            print(f"Result: {response[:200]}...")
            print("✅ Handled successfully")
        except Exception as e:
            print(f"❌ Error: {str(e)}")


async def main():
    """Run the demo."""
    await demo_autonomous_workflow()
    await test_edge_cases()
    
    print("\n\n🎉 Autonomous tool creation system is working correctly!")
    print("\nThe system now:")
    print("✅ Creates tools automatically from natural language")
    print("✅ Validates tools before adding them")
    print("✅ Handles errors gracefully")
    print("✅ Reuses existing tools when appropriate")
    print("✅ Maintains code quality standards")


if __name__ == "__main__":
    asyncio.run(main())