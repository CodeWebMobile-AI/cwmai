#!/usr/bin/env python3
"""Test what the enhanced assistant sees for tool count."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.enhanced_conversational_assistant import EnhancedConversationalAssistant
from scripts.conversational_ai_logger import setup_conversational_ai_logging


async def main():
    """Test the assistant's view of tools."""
    setup_conversational_ai_logging()
    
    print("Testing Enhanced Conversational Assistant tool visibility...")
    assistant = EnhancedConversationalAssistant()
    await assistant.initialize()
    
    # Check tool system directly
    print(f"\n1. Direct tool count from assistant.tool_system: {len(assistant.tool_system.tools)}")
    
    # Check what context the assistant gathers
    print("\n2. Testing context gathering for 'how many tools' query...")
    context = await assistant._gather_intelligent_context("how many tools do you have?")
    
    print(f"\nContext shows:")
    print(f"  - available_tools: {context.get('available_tools', 'N/A')}")
    print(f"  - tool_categories: {context.get('tool_categories', [])}")
    print(f"  - Number of tool_names: {len(context.get('tool_names', []))}")
    
    # Check tools by category
    if 'tools_by_category' in context:
        print(f"\n  Tools by category:")
        for category, tools in context['tools_by_category'].items():
            if isinstance(tools, dict):
                print(f"    - {category}: {len(tools.get('tools', []))} tools")
            elif isinstance(tools, list):
                print(f"    - {category}: {len(tools)} tools")
    
    # Test the actual response
    print("\n3. Testing actual response to 'how many tools do you have?'")
    response = await assistant.handle_conversation("how many tools do you have?")
    print(f"\nAssistant response: {response}")
    
    # Check tool usage stats directly
    print("\n4. Checking tool usage stats directly...")
    stats_result = await assistant.tool_system.call_tool("get_tool_usage_stats")
    if stats_result.get('success'):
        stats = stats_result['result']
        print(f"  Tool usage stats shows: {stats.get('total_tools', 0)} tools")


if __name__ == "__main__":
    asyncio.run(main())