#!/usr/bin/env python3
"""
Debug count_workers Issue
Find where count_workers is being called
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.conversational_ai_assistant import ConversationalAIAssistant
from scripts.tool_calling_system import ToolCallingSystem


async def debug_count_workers():
    """Debug the count_workers issue."""
    print("🔍 Debugging count_workers Issue\n")
    
    # Initialize assistant
    assistant = ConversationalAIAssistant()
    await assistant.initialize()
    
    # Check tool calling system
    if hasattr(assistant, 'tool_system'):
        print("📋 Tool system tools:")
        if 'count_workers' in assistant.tool_system.tools:
            print("   ⚠️  Found count_workers in tool system")
        else:
            print("   ✅ No count_workers in tool system")
    
    # Try the specific query
    print("\n🧪 Testing query: 'how many workers do we have?'")
    
    # Let's trace the execution
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        # Manually check what happens with tool execution
        from scripts.tool_calling_system import ToolCallingSystem
        tool_system = ToolCallingSystem()
        
        # Check if the AI is trying to call count_workers
        print("\n📋 Simulating tool call...")
        result = await tool_system.call_tool("count_workers")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"❌ Error calling count_workers: {e}")
        
    # Now let's see what the conversational AI does
    print("\n📋 Testing through conversational AI...")
    try:
        response = await assistant.handle_conversation("how many workers do we have?")
        print(f"✅ Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run the debug."""
    await debug_count_workers()


if __name__ == "__main__":
    asyncio.run(main())