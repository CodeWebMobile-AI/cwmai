#!/usr/bin/env python3
"""
Test Worker Count Issue
Diagnose and fix the count_workers self parameter issue
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.conversational_ai_assistant import ConversationalAIAssistant
from scripts.tool_calling_system import ToolCallingSystem


async def test_worker_count_issue():
    """Test and diagnose the worker count issue."""
    print("🔍 Testing Worker Count Issue\n")
    
    # First check if count_workers tool exists
    tool_system = ToolCallingSystem()
    
    print("📋 Checking for count_workers tool...")
    if 'count_workers' in tool_system.tools:
        print("   ⚠️  Found count_workers tool - this might be the problem")
        tool = tool_system.tools['count_workers']
        print(f"   Created by AI: {tool.created_by_ai}")
        print(f"   Function: {tool.func}")
        
        # Check if it expects self parameter
        import inspect
        sig = inspect.signature(tool.func)
        params = list(sig.parameters.keys())
        print(f"   Parameters: {params}")
        
        if 'self' in params:
            print("   ❌ ERROR: Tool expects 'self' parameter!")
    else:
        print("   ✅ No count_workers tool found (good)")
    
    print("\n📋 Available tools for worker/system queries:")
    for name, tool in tool_system.tools.items():
        if 'worker' in name or 'system' in name or 'status' in name:
            print(f"   • {name}: {tool.description}")
    
    # Now test the query
    print("\n🧪 Testing conversational query...")
    assistant = ConversationalAIAssistant()
    await assistant.initialize()
    
    queries = [
        "how many workers do we have?",
        "show me worker status",
        "what's the system status?"
    ]
    
    for query in queries:
        print(f"\n📝 Query: '{query}'")
        try:
            response = await assistant.handle_conversation(query)
            print(f"✅ Response: {response[:200]}...")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    return True


async def main():
    """Run the test."""
    await test_worker_count_issue()


if __name__ == "__main__":
    asyncio.run(main())