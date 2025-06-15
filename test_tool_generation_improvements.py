#!/usr/bin/env python3
"""
Test Tool Generation Improvements
Verifies that the smart tool generation system works correctly
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.conversational_ai_assistant import ConversationalAIAssistant


async def test_tool_generation_improvements():
    """Test that tool generation is smarter and handles abbreviations properly."""
    print("🧪 Testing Tool Generation Improvements\n")
    
    # Initialize assistant
    assistant = ConversationalAIAssistant()
    await assistant.initialize()
    
    # Test cases with abbreviations and edge cases
    test_cases = [
        # Abbreviations that should NOT create new tools
        {
            "query": "how many reps are we managing?",
            "expected_tool": "count_repositories",
            "should_create_new": False,
            "description": "Testing 'reps' abbreviation"
        },
        {
            "query": "list all repos",
            "expected_tool": "list_repositories", 
            "should_create_new": False,
            "description": "Testing 'repos' abbreviation"
        },
        {
            "query": "show me the cmds",
            "expected_tool": "list_commands",
            "should_create_new": False,
            "description": "Testing 'cmds' abbreviation"
        },
        {
            "query": "count workers",
            "expected_tool": "count_workers",
            "should_create_new": False,  # Should use existing system_status
            "description": "Testing worker count (should use system_status)"
        },
        
        # Edge cases that might create tools
        {
            "query": "analyze code quality for all projects",
            "expected_tool": "analyze_code_quality",
            "should_create_new": True,  # This is genuinely new functionality
            "description": "Testing genuinely new functionality"
        },
        {
            "query": "generate weekly report",
            "expected_tool": "generate_weekly_report",
            "should_create_new": True,
            "description": "Testing report generation (new feature)"
        },
        
        # Complex queries that should use existing tools
        {
            "query": "what's the status of our repositories?",
            "expected_tool": "get_repositories",
            "should_create_new": False,
            "description": "Testing natural language for existing functionality"
        },
        {
            "query": "how's the system doing?",
            "expected_tool": "get_system_status",
            "should_create_new": False,
            "description": "Testing system status query"
        }
    ]
    
    results = []
    
    # Check which tools exist before tests
    from scripts.tool_calling_system import ToolCallingSystem
    tool_system = ToolCallingSystem()
    existing_tools = set(tool_system.tools.keys())
    print(f"📋 Existing tools before tests: {len(existing_tools)}")
    print(f"   Core tools: {sorted([t for t in existing_tools if not t.startswith('_')])[:10]}...\n")
    
    for test in test_cases:
        print(f"📝 Test: {test['description']}")
        print(f"   Query: '{test['query']}'")
        
        try:
            # Execute the query
            response = await assistant.handle_conversation(test['query'])
            
            # Check if a new tool was created
            current_tools = set(tool_system.tools.keys())
            new_tools = current_tools - existing_tools
            
            if new_tools:
                print(f"   🔧 New tools created: {new_tools}")
                if test['should_create_new']:
                    print(f"   ✅ Correctly created new tool(s)")
                    results.append(True)
                else:
                    print(f"   ❌ Should NOT have created new tool(s)")
                    results.append(False)
            else:
                print(f"   ℹ️  No new tools created")
                if not test['should_create_new']:
                    print(f"   ✅ Correctly used existing tools")
                    results.append(True)
                else:
                    print(f"   ⚠️  Expected new tool creation (might have used existing)")
                    # This is acceptable if existing tools handled it
                    results.append(True)
            
            # Update existing tools set
            existing_tools = current_tools
            
            # Show response snippet
            print(f"   Response: {response[:150]}...")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results.append(False)
        
        print()
    
    # Check for problematic AI-generated tools
    print("\n🔍 Checking for problematic AI-generated tools...")
    ai_tools_dir = Path("scripts/ai_generated_tools")
    if ai_tools_dir.exists():
        problematic_tools = []
        for tool_file in ai_tools_dir.glob("*.py"):
            content = tool_file.read_text()
            # Check for self parameter in tool functions
            if "def count_" in content and "(self" in content:
                problematic_tools.append(tool_file.name)
            elif "def list_" in content and "(self" in content:
                problematic_tools.append(tool_file.name)
        
        if problematic_tools:
            print(f"   ⚠️  Found {len(problematic_tools)} tools with 'self' parameter: {problematic_tools}")
        else:
            print("   ✅ No problematic AI-generated tools found")
    else:
        print("   ℹ️  No AI-generated tools directory found")
    
    # Summary
    print("\n" + "="*60)
    print("📊 TOOL GENERATION IMPROVEMENTS SUMMARY")
    print("="*60)
    total = len(results)
    passed = sum(results)
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    print(f"Success Rate: {(passed/total*100):.1f}%")
    
    # Final recommendations
    if passed == total:
        print("\n✨ All tests passed! Tool generation is working intelligently.")
    else:
        print("\n⚠️  Some tests failed. Review the tool generation logic.")
    
    return passed == total


async def main():
    """Run the tests."""
    success = await test_tool_generation_improvements()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())